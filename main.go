package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// --- JSON-RPC types ---

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  any    `json:"result,omitempty"`
	Error   *rpcError `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// --- MCP types ---

type mcpTool struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	InputSchema mcpSchema `json:"inputSchema"`
}

type mcpSchema struct {
	Type       string                `json:"type"`
	Properties map[string]mcpProp    `json:"properties"`
	Required   []string              `json:"required,omitempty"`
}

type mcpProp struct {
	Type        string   `json:"type"`
	Description string   `json:"description,omitempty"`
	Enum        []string `json:"enum,omitempty"`
	Default     any      `json:"default,omitempty"`
	// For array items
	Items *mcpProp `json:"items,omitempty"`
}

type mcpContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type mcpToolResult struct {
	Content []mcpContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

// --- Ollama API types ---

type ollamaGenerateReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	System string `json:"system,omitempty"`
	Stream bool   `json:"stream"`
}

type ollamaGenerateResp struct {
	Response string `json:"response"`
}

type ollamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatReq struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

type ollamaChatResp struct {
	Message ollamaChatMessage `json:"message"`
}

type ollamaEmbedReq struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type ollamaEmbedResp struct {
	Embeddings [][]float64 `json:"embeddings"`
}

type ollamaModel struct {
	Name       string `json:"name"`
	Size       int64  `json:"size"`
	ModifiedAt string `json:"modified_at"`
}

type ollamaTagsResp struct {
	Models []ollamaModel `json:"models"`
}

// --- Server ---

type server struct {
	ollamaURL string
	client    *http.Client
}

func newServer() *server {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434"
	}
	return &server{
		ollamaURL: strings.TrimRight(host, "/"),
		client: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

var tools = []mcpTool{
	{
		Name:        "list_models",
		Description: "List all available Ollama models",
		InputSchema: mcpSchema{
			Type:       "object",
			Properties: map[string]mcpProp{},
		},
	},
	{
		Name:        "generate",
		Description: "Generate a one-shot response from an Ollama model",
		InputSchema: mcpSchema{
			Type: "object",
			Properties: map[string]mcpProp{
				"model":  {Type: "string", Description: "Model name (e.g. qwen3:14b, llama3:8b)"},
				"prompt": {Type: "string", Description: "The prompt to send"},
				"system": {Type: "string", Description: "Optional system prompt to set context/behavior"},
			},
			Required: []string{"model", "prompt"},
		},
	},
	{
		Name:        "chat",
		Description: "Send a multi-turn conversation to an Ollama model",
		InputSchema: mcpSchema{
			Type: "object",
			Properties: map[string]mcpProp{
				"model": {Type: "string", Description: "Model name (e.g. qwen3:14b, llama3:8b)"},
				"messages": {
					Type:        "array",
					Description: "Conversation messages, each with 'role' (system/user/assistant) and 'content'",
					Items:       &mcpProp{Type: "object"},
				},
			},
			Required: []string{"model", "messages"},
		},
	},
	{
		Name:        "embed",
		Description: "Generate embeddings for a text using an Ollama model",
		InputSchema: mcpSchema{
			Type: "object",
			Properties: map[string]mcpProp{
				"model": {Type: "string", Description: "Embedding model name (e.g. nomic-embed-text)"},
				"text":  {Type: "string", Description: "Text to embed"},
			},
			Required: []string{"model", "text"},
		},
	},
}

func (s *server) handleInitialize(id json.RawMessage) rpcResponse {
	return rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result: map[string]any{
			"protocolVersion": "2024-11-05",
			"capabilities": map[string]any{
				"tools": map[string]any{"listChanged": false},
			},
			"serverInfo": map[string]any{
				"name":    "ollama-mcp-go",
				"version": "0.1.0",
			},
		},
	}
}

func (s *server) handleToolsList(id json.RawMessage) rpcResponse {
	return rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  map[string]any{"tools": tools},
	}
}

func (s *server) handleToolsCall(id json.RawMessage, params json.RawMessage) rpcResponse {
	var call struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	}
	if err := json.Unmarshal(params, &call); err != nil {
		return s.errorResult(id, "invalid params: "+err.Error())
	}

	var result mcpToolResult
	switch call.Name {
	case "list_models":
		result = s.toolListModels()
	case "generate":
		result = s.toolGenerate(call.Arguments)
	case "chat":
		result = s.toolChat(call.Arguments)
	case "embed":
		result = s.toolEmbed(call.Arguments)
	default:
		return s.errorResult(id, "unknown tool: "+call.Name)
	}

	return rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  result,
	}
}

// --- Tool implementations ---

func (s *server) toolListModels() mcpToolResult {
	resp, err := s.client.Get(s.ollamaURL + "/api/tags")
	if err != nil {
		return errResult("failed to reach Ollama: " + err.Error())
	}
	defer resp.Body.Close()

	var tags ollamaTagsResp
	if err := json.NewDecoder(resp.Body).Decode(&tags); err != nil {
		return errResult("failed to parse response: " + err.Error())
	}

	var lines []string
	for _, m := range tags.Models {
		sizeMB := m.Size / (1024 * 1024)
		lines = append(lines, fmt.Sprintf("- %s (%d MB)", m.Name, sizeMB))
	}
	if len(lines) == 0 {
		return textResult("No models found. Pull one with: ollama pull <model>")
	}
	return textResult(strings.Join(lines, "\n"))
}

func (s *server) toolGenerate(args map[string]any) mcpToolResult {
	model, _ := args["model"].(string)
	prompt, _ := args["prompt"].(string)
	system, _ := args["system"].(string)

	if model == "" || prompt == "" {
		return errResult("'model' and 'prompt' are required")
	}

	body, err := json.Marshal(ollamaGenerateReq{
		Model:  model,
		Prompt: prompt,
		System: system,
		Stream: false,
	})
	if err != nil {
		return errResult("marshal error: " + err.Error())
	}

	resp, err := s.client.Post(s.ollamaURL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return errResult("Ollama request failed: " + err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return errResult(fmt.Sprintf("Ollama returned %d: %s", resp.StatusCode, string(b)))
	}

	var result ollamaGenerateResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return errResult("failed to parse response: " + err.Error())
	}

	return textResult(result.Response)
}

func (s *server) toolChat(args map[string]any) mcpToolResult {
	model, _ := args["model"].(string)
	if model == "" {
		return errResult("'model' is required")
	}

	rawMsgs, ok := args["messages"]
	if !ok {
		return errResult("'messages' is required")
	}

	// Convert []any to []ollamaChatMessage
	msgSlice, ok := rawMsgs.([]any)
	if !ok {
		return errResult("'messages' must be an array")
	}

	var messages []ollamaChatMessage
	for _, m := range msgSlice {
		mMap, ok := m.(map[string]any)
		if !ok {
			return errResult("each message must be an object with 'role' and 'content'")
		}
		role, _ := mMap["role"].(string)
		content, _ := mMap["content"].(string)
		if role == "" || content == "" {
			return errResult("each message must have non-empty 'role' and 'content'")
		}
		messages = append(messages, ollamaChatMessage{Role: role, Content: content})
	}

	body, err := json.Marshal(ollamaChatReq{
		Model:    model,
		Messages: messages,
		Stream:   false,
	})
	if err != nil {
		return errResult("marshal error: " + err.Error())
	}

	resp, err := s.client.Post(s.ollamaURL+"/api/chat", "application/json", bytes.NewReader(body))
	if err != nil {
		return errResult("Ollama request failed: " + err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return errResult(fmt.Sprintf("Ollama returned %d: %s", resp.StatusCode, string(b)))
	}

	var result ollamaChatResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return errResult("failed to parse response: " + err.Error())
	}

	return textResult(result.Message.Content)
}

func (s *server) toolEmbed(args map[string]any) mcpToolResult {
	model, _ := args["model"].(string)
	text, _ := args["text"].(string)

	if model == "" || text == "" {
		return errResult("'model' and 'text' are required")
	}

	body, err := json.Marshal(ollamaEmbedReq{
		Model: model,
		Input: text,
	})
	if err != nil {
		return errResult("marshal error: " + err.Error())
	}

	resp, err := s.client.Post(s.ollamaURL+"/api/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		return errResult("Ollama request failed: " + err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return errResult(fmt.Sprintf("Ollama returned %d: %s", resp.StatusCode, string(b)))
	}

	var result ollamaEmbedResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return errResult("failed to parse response: " + err.Error())
	}

	// Return dimensions info + first few values as preview
	if len(result.Embeddings) == 0 {
		return errResult("no embeddings returned")
	}

	emb := result.Embeddings[0]
	out, _ := json.Marshal(map[string]any{
		"dimensions": len(emb),
		"embeddings": emb,
	})
	return textResult(string(out))
}

// --- Helpers ---

func textResult(text string) mcpToolResult {
	return mcpToolResult{
		Content: []mcpContent{{Type: "text", Text: text}},
	}
}

func errResult(msg string) mcpToolResult {
	return mcpToolResult{
		Content: []mcpContent{{Type: "text", Text: msg}},
		IsError: true,
	}
}

func (s *server) errorResult(id json.RawMessage, msg string) rpcResponse {
	return rpcResponse{
		JSONRPC: "2.0",
		ID:      id,
		Error:   &rpcError{Code: -32600, Message: msg},
	}
}

func writeResponse(w *bufio.Writer, resp rpcResponse) {
	data, err := json.Marshal(resp)
	if err != nil {
		log.Printf("marshal error: %v", err)
		return
	}
	fmt.Fprintf(w, "%s\n", data)
	w.Flush()
}

// --- Main loop ---

func main() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ltime)

	s := newServer()
	log.Printf("ollama-mcp-go starting (ollama: %s)", s.ollamaURL)

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024) // 4MB buffer for large messages
	writer := bufio.NewWriter(os.Stdout)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}

		var req rpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			log.Printf("invalid JSON-RPC: %v", err)
			continue
		}

		// Notifications (no ID) — just ack silently
		if req.ID == nil || string(req.ID) == "null" {
			log.Printf("notification: %s", req.Method)
			continue
		}

		var resp rpcResponse
		switch req.Method {
		case "initialize":
			resp = s.handleInitialize(req.ID)
		case "tools/list":
			resp = s.handleToolsList(req.ID)
		case "tools/call":
			resp = s.handleToolsCall(req.ID, req.Params)
		default:
			resp = rpcResponse{
				JSONRPC: "2.0",
				ID:      req.ID,
				Error:   &rpcError{Code: -32601, Message: "method not found: " + req.Method},
			}
		}

		writeResponse(writer, resp)
	}

	if err := scanner.Err(); err != nil {
		log.Printf("stdin error: %v", err)
	}
}
