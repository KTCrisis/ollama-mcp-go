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
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
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
	Type       string             `json:"type"`
	Properties map[string]mcpProp `json:"properties"`
	Required   []string           `json:"required,omitempty"`
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
	Content         []mcpContent `json:"content"`
	IsError         bool         `json:"isError,omitempty"`
	PromptEvalCount int          `json:"prompt_eval_count,omitempty"`
	EvalCount       int          `json:"eval_count,omitempty"`
}

// --- Ollama API types ---

type ollamaGenerateReq struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	System string `json:"system,omitempty"`
	Stream bool   `json:"stream"`
}

type ollamaChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	// Thinking carries the reasoning of models that emit a separate channel
	// (gpt-oss and friends). It is only ever received, never sent — omitempty
	// keeps it out of requests.
	Thinking string `json:"thinking,omitempty"`
}

type ollamaChatReq struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
}

// ollamaStreamChunk is one NDJSON line of a streamed /api/generate or
// /api/chat response. The two endpoints differ only in where they put the
// text: "response" for generate, "message.content" for chat. A single struct
// covers both — the unused field simply stays zero.
type ollamaStreamChunk struct {
	Response        string            `json:"response"`
	Message         ollamaChatMessage `json:"message"`
	Done            bool              `json:"done"`
	PromptEvalCount int               `json:"prompt_eval_count"`
	EvalCount       int               `json:"eval_count"`
}

func (c ollamaStreamChunk) token() string {
	if c.Response != "" {
		return c.Response
	}
	return c.Message.Content
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

// --- Streaming ---

// traceFile returns the path tokens are mirrored to as they arrive, or ""
// when tracing is off. Watch it with: tail -f /tmp/ollama-mcp-trace.log
func traceFile() string {
	path := os.Getenv("OLLAMA_MCP_TRACE")
	if path == "" {
		return defaultTracePath
	}
	if path == "off" {
		return ""
	}
	return path
}

const defaultTracePath = "/tmp/ollama-mcp-trace.log"

// streamSink accumulates the full reply for the MCP response while mirroring
// each token to the trace file. MCP cannot deliver a partial tool result, so
// the file is the only way to watch a reply as it is being written.
type streamSink struct {
	f          *os.File
	buf        strings.Builder
	inThinking bool
}

// newStreamSink opens the trace file and writes a header. A file that cannot
// be opened is not an error: tracing degrades to nothing and generation goes
// on, since the caller wants the reply far more than it wants the log.
func newStreamSink(kind, model, prompt string) *streamSink {
	s := &streamSink{}
	path := traceFile()
	if path == "" {
		return s
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		log.Printf("trace disabled: %v", err)
		return s
	}
	s.f = f
	fmt.Fprintf(f, "\n=== %s  %s  %s ===\n> %s\n---\n",
		time.Now().Format("15:04:05"), kind, model, prompt)
	return s
}

// write mirrors a token. os.File is unbuffered, so each call is a write(2)
// syscall and reaches a `tail -f` immediately — buffering here would defeat
// the whole point.
func (s *streamSink) write(tok string) {
	s.buf.WriteString(tok)
	if s.f == nil {
		return
	}
	if s.inThinking {
		io.WriteString(s.f, "\n--- repond ---\n")
		s.inThinking = false
	}
	io.WriteString(s.f, tok)
}

// writeThinking mirrors reasoning tokens to the trace only. They never reach
// buf, so the MCP client still gets the answer alone — the trace is the one
// place where the model's deliberation is visible.
func (s *streamSink) writeThinking(tok string) {
	if s.f == nil {
		return
	}
	if !s.inThinking {
		io.WriteString(s.f, "\n--- pense ---\n")
		s.inThinking = true
	}
	io.WriteString(s.f, tok)
}

func (s *streamSink) close(promptEval, eval int) {
	if s.f != nil {
		fmt.Fprintf(s.f, "\n[done: %d prompt / %d eval]\n", promptEval, eval)
		s.f.Close()
		s.f = nil
	}
}

func (s *streamSink) text() string { return s.buf.String() }

// postStream sends body to path and consumes the NDJSON reply line by line,
// feeding every token to sink. It returns the usage counters carried by the
// final chunk.
func (s *server) postStream(path string, body []byte, sink *streamSink) (promptEval, eval int, err error) {
	resp, err := s.client.Post(s.ollamaURL+path, "application/json", bytes.NewReader(body))
	if err != nil {
		return 0, 0, fmt.Errorf("Ollama request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return 0, 0, fmt.Errorf("Ollama returned %d: %s", resp.StatusCode, string(b))
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var chunk ollamaStreamChunk
		if err := json.Unmarshal(line, &chunk); err != nil {
			return 0, 0, fmt.Errorf("failed to parse stream chunk: %w", err)
		}
		if th := chunk.Message.Thinking; th != "" {
			sink.writeThinking(th)
		}
		// An empty token must not reach write(): chunks that carry reasoning
		// only would otherwise flip the sink out of thinking mode too early.
		if tok := chunk.token(); tok != "" {
			sink.write(tok)
		}
		if chunk.Done {
			promptEval, eval = chunk.PromptEvalCount, chunk.EvalCount
		}
	}
	if err := scanner.Err(); err != nil {
		return 0, 0, fmt.Errorf("stream read failed: %w", err)
	}
	return promptEval, eval, nil
}

// --- Tool implementations ---

func (s *server) toolListModels() mcpToolResult {
	resp, err := s.client.Get(s.ollamaURL + "/api/tags")
	if err != nil {
		return errResult("failed to reach Ollama: " + err.Error())
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(resp.Body)
		return errResult(fmt.Sprintf("Ollama returned %d: %s", resp.StatusCode, string(b)))
	}

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
		Stream: true,
	})
	if err != nil {
		return errResult("marshal error: " + err.Error())
	}

	sink := newStreamSink("generate", model, prompt)
	promptEval, eval, err := s.postStream("/api/generate", body, sink)
	if err != nil {
		sink.close(0, 0)
		return errResult(err.Error())
	}
	sink.close(promptEval, eval)

	out := textResult(sink.text())
	out.PromptEvalCount = promptEval
	out.EvalCount = eval
	return out
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
	if len(messages) == 0 {
		return errResult("'messages' must not be empty")
	}

	body, err := json.Marshal(ollamaChatReq{
		Model:    model,
		Messages: messages,
		Stream:   true,
	})
	if err != nil {
		return errResult("marshal error: " + err.Error())
	}

	sink := newStreamSink("chat", model, messages[len(messages)-1].Content)
	promptEval, eval, err := s.postStream("/api/chat", body, sink)
	if err != nil {
		sink.close(0, 0)
		return errResult(err.Error())
	}
	sink.close(promptEval, eval)

	out := textResult(sink.text())
	out.PromptEvalCount = promptEval
	out.EvalCount = eval
	return out
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
	preview := emb
	if len(preview) > 16 {
		preview = preview[:16]
	}
	out, _ := json.Marshal(map[string]any{
		"dimensions": len(emb),
		"preview":    preview,
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
		Error:   &rpcError{Code: -32602, Message: msg},
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
