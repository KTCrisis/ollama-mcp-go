package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
)

// --- MCP protocol tests ---

func TestInitialize(t *testing.T) {
	s := newServer()
	resp := s.handleInitialize(json.RawMessage(`1`))

	result, ok := resp.Result.(map[string]any)
	if !ok {
		t.Fatal("expected map result")
	}
	if result["protocolVersion"] != "2024-11-05" {
		t.Errorf("expected protocol 2024-11-05, got %v", result["protocolVersion"])
	}
	info := result["serverInfo"].(map[string]any)
	if info["name"] != "ollama-mcp-go" {
		t.Errorf("expected name ollama-mcp-go, got %v", info["name"])
	}
}

func TestToolsList(t *testing.T) {
	s := newServer()
	resp := s.handleToolsList(json.RawMessage(`2`))

	result, ok := resp.Result.(map[string]any)
	if !ok {
		t.Fatal("expected map result")
	}
	toolList := result["tools"].([]mcpTool)
	if len(toolList) != 4 {
		t.Errorf("expected 4 tools, got %d", len(toolList))
	}

	names := map[string]bool{}
	for _, tool := range toolList {
		names[tool.Name] = true
	}
	for _, expected := range []string{"list_models", "generate", "chat", "embed"} {
		if !names[expected] {
			t.Errorf("missing tool: %s", expected)
		}
	}
}

func TestToolsCallUnknown(t *testing.T) {
	s := newServer()
	params, _ := json.Marshal(map[string]any{"name": "nope", "arguments": map[string]any{}})
	resp := s.handleToolsCall(json.RawMessage(`3`), params)

	if resp.Error == nil {
		t.Fatal("expected error for unknown tool")
	}
	if !strings.Contains(resp.Error.Message, "unknown tool") {
		t.Errorf("expected 'unknown tool' error, got: %s", resp.Error.Message)
	}
}

// --- Tool implementation tests (with mock Ollama) ---

func mockOllama(t *testing.T) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		defer r.Body.Close()

		switch r.URL.Path {
		case "/api/tags":
			json.NewEncoder(w).Encode(ollamaTagsResp{
				Models: []ollamaModel{
					{Name: "testmodel:7b", Size: 4_000_000_000, ModifiedAt: "2026-01-01T00:00:00Z"},
					{Name: "embedmodel:latest", Size: 200_000_000, ModifiedAt: "2026-01-01T00:00:00Z"},
				},
			})

		case "/api/generate":
			var req ollamaGenerateReq
			json.Unmarshal(body, &req)
			if req.Model == "" {
				http.Error(w, `{"error":"model required"}`, 400)
				return
			}
			resp := ollamaGenerateResp{Response: fmt.Sprintf("mock response to: %s", req.Prompt)}
			if req.System != "" {
				resp.Response = fmt.Sprintf("[system:%s] %s", req.System, resp.Response)
			}
			json.NewEncoder(w).Encode(resp)

		case "/api/chat":
			var req ollamaChatReq
			json.Unmarshal(body, &req)
			if req.Model == "" {
				http.Error(w, `{"error":"model required"}`, 400)
				return
			}
			last := req.Messages[len(req.Messages)-1].Content
			json.NewEncoder(w).Encode(ollamaChatResp{
				Message: ollamaChatMessage{Role: "assistant", Content: fmt.Sprintf("mock chat reply to: %s", last)},
			})

		case "/api/embed":
			var req ollamaEmbedReq
			json.Unmarshal(body, &req)
			if req.Model == "" {
				http.Error(w, `{"error":"model required"}`, 400)
				return
			}
			json.NewEncoder(w).Encode(ollamaEmbedResp{
				Embeddings: [][]float64{{0.1, 0.2, 0.3, 0.4, 0.5}},
			})

		default:
			http.Error(w, "not found", 404)
		}
	}))
}

func testServer(t *testing.T) (*server, func()) {
	t.Helper()
	mock := mockOllama(t)
	s := &server{
		ollamaURL: mock.URL,
		client:    mock.Client(),
	}
	return s, mock.Close
}

func TestListModels(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolListModels()
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content[0].Text)
	}
	text := result.Content[0].Text
	if !strings.Contains(text, "testmodel:7b") {
		t.Errorf("expected testmodel:7b in output, got: %s", text)
	}
	if !strings.Contains(text, "embedmodel:latest") {
		t.Errorf("expected embedmodel:latest in output, got: %s", text)
	}
}

func TestGenerate(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolGenerate(map[string]any{
		"model":  "testmodel:7b",
		"prompt": "hello",
	})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content[0].Text)
	}
	if !strings.Contains(result.Content[0].Text, "mock response to: hello") {
		t.Errorf("unexpected response: %s", result.Content[0].Text)
	}
}

func TestGenerateWithSystem(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolGenerate(map[string]any{
		"model":  "testmodel:7b",
		"prompt": "hello",
		"system": "be concise",
	})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content[0].Text)
	}
	text := result.Content[0].Text
	if !strings.Contains(text, "[system:be concise]") {
		t.Errorf("system prompt not propagated: %s", text)
	}
}

func TestGenerateMissingParams(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolGenerate(map[string]any{"model": "x"})
	if !result.IsError {
		t.Error("expected error for missing prompt")
	}

	result = s.toolGenerate(map[string]any{"prompt": "x"})
	if !result.IsError {
		t.Error("expected error for missing model")
	}
}

func TestChat(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolChat(map[string]any{
		"model": "testmodel:7b",
		"messages": []any{
			map[string]any{"role": "user", "content": "bonjour"},
		},
	})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content[0].Text)
	}
	if !strings.Contains(result.Content[0].Text, "mock chat reply to: bonjour") {
		t.Errorf("unexpected response: %s", result.Content[0].Text)
	}
}

func TestChatMultiTurn(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolChat(map[string]any{
		"model": "testmodel:7b",
		"messages": []any{
			map[string]any{"role": "system", "content": "speak french"},
			map[string]any{"role": "user", "content": "hello"},
			map[string]any{"role": "assistant", "content": "bonjour"},
			map[string]any{"role": "user", "content": "how are you?"},
		},
	})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content[0].Text)
	}
	if !strings.Contains(result.Content[0].Text, "how are you?") {
		t.Errorf("last message not forwarded: %s", result.Content[0].Text)
	}
}

func TestChatMissingParams(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolChat(map[string]any{"model": "x"})
	if !result.IsError {
		t.Error("expected error for missing messages")
	}

	result = s.toolChat(map[string]any{
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if !result.IsError {
		t.Error("expected error for missing model")
	}
}

func TestChatInvalidMessages(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolChat(map[string]any{
		"model":    "x",
		"messages": "not an array",
	})
	if !result.IsError {
		t.Error("expected error for non-array messages")
	}

	result = s.toolChat(map[string]any{
		"model":    "x",
		"messages": []any{"not an object"},
	})
	if !result.IsError {
		t.Error("expected error for non-object message")
	}

	result = s.toolChat(map[string]any{
		"model":    "x",
		"messages": []any{map[string]any{"role": "", "content": "hi"}},
	})
	if !result.IsError {
		t.Error("expected error for empty role")
	}
}

func TestEmbed(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolEmbed(map[string]any{
		"model": "embedmodel:latest",
		"text":  "test embedding",
	})
	if result.IsError {
		t.Fatalf("unexpected error: %s", result.Content[0].Text)
	}

	var parsed map[string]any
	if err := json.Unmarshal([]byte(result.Content[0].Text), &parsed); err != nil {
		t.Fatalf("response is not valid JSON: %v", err)
	}
	if parsed["dimensions"].(float64) != 5 {
		t.Errorf("expected 5 dimensions, got %v", parsed["dimensions"])
	}
}

func TestEmbedMissingParams(t *testing.T) {
	s, cleanup := testServer(t)
	defer cleanup()

	result := s.toolEmbed(map[string]any{"model": "x"})
	if !result.IsError {
		t.Error("expected error for missing text")
	}

	result = s.toolEmbed(map[string]any{"text": "x"})
	if !result.IsError {
		t.Error("expected error for missing model")
	}
}

// --- Integration test: full stdio round-trip ---

func TestStdioRoundTrip(t *testing.T) {
	mock := mockOllama(t)
	defer mock.Close()

	// Override OLLAMA_HOST for the test
	os.Setenv("OLLAMA_HOST", mock.URL)
	defer os.Unsetenv("OLLAMA_HOST")

	s := newServer()

	messages := []string{
		`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`,
		`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_models","arguments":{}}}`,
		`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"generate","arguments":{"model":"testmodel:7b","prompt":"hi"}}}`,
	}

	input := strings.NewReader(strings.Join(messages, "\n") + "\n")
	scanner := bufio.NewScanner(input)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	writer := bufio.NewWriter(io.Discard)

	var responses []rpcResponse
	realWriter := &strings.Builder{}
	w := bufio.NewWriter(realWriter)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var req rpcRequest
		if err := json.Unmarshal(line, &req); err != nil {
			t.Fatalf("invalid JSON: %v", err)
		}

		if req.ID == nil || string(req.ID) == "null" {
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
		}

		data, _ := json.Marshal(resp)
		fmt.Fprintf(w, "%s\n", data)
		responses = append(responses, resp)
	}
	w.Flush()
	_ = writer

	if len(responses) != 4 {
		t.Fatalf("expected 4 responses (skipping notification), got %d", len(responses))
	}

	// Check initialize
	if responses[0].Error != nil {
		t.Errorf("initialize failed: %s", responses[0].Error.Message)
	}

	// Check tools/list
	if responses[1].Error != nil {
		t.Errorf("tools/list failed: %s", responses[1].Error.Message)
	}

	// Check list_models
	if responses[2].Error != nil {
		t.Errorf("list_models failed: %s", responses[2].Error.Message)
	}

	// Check generate
	if responses[3].Error != nil {
		t.Errorf("generate failed: %s", responses[3].Error.Message)
	}
}
