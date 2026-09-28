package main

import (
	"encoding/json"
	"testing"
)

func call(t *testing.T, s *server, raw string) map[string]any {
	t.Helper()
	var req rpcRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(s.dispatch(req))
	var out map[string]any
	json.Unmarshal(b, &out)
	return out
}

const modernMeta = `"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28"}`

func TestDiscover(t *testing.T) {
	out := call(t, newServer(), `{"jsonrpc":"2.0","id":1,"method":"server/discover","params":{`+modernMeta+`}}`)
	r := out["result"].(map[string]any)
	if r["resultType"] != "complete" || r["cacheScope"] != "public" {
		t.Errorf("discover result = %v", r)
	}
	versions := r["supportedVersions"].([]any)
	if versions[0] != "2026-07-28" || len(versions) != 5 {
		t.Errorf("supportedVersions = %v", versions)
	}
	if r["_meta"].(map[string]any)["io.modelcontextprotocol/serverInfo"] == nil {
		t.Error("serverInfo missing from _meta")
	}
}

func TestModernToolsList(t *testing.T) {
	out := call(t, newServer(), `{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{`+modernMeta+`}}`)
	r := out["result"].(map[string]any)
	if r["resultType"] != "complete" || r["ttlMs"] == nil || r["cacheScope"] != "public" {
		t.Errorf("tools/list lacks 2026 fields: %v", r)
	}
	for _, x := range r["tools"].([]any) {
		tool := x.(map[string]any)
		a, _ := tool["annotations"].(map[string]any)
		if tool["title"] == nil || a == nil || a["readOnlyHint"] != true || a["openWorldHint"] != false {
			t.Errorf("tool %v lacks title or annotations: %v", tool["name"], tool)
		}
	}
}

func TestUnsupportedModernVersion(t *testing.T) {
	out := call(t, newServer(), `{"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2030-01-01"}}}`)
	e := out["error"].(map[string]any)
	if e["code"].(float64) != -32022 {
		t.Fatalf("error = %v, want -32022", e)
	}
	d := e["data"].(map[string]any)
	if d["requested"] != "2030-01-01" || len(d["supported"].([]any)) != 5 {
		t.Errorf("error data = %v", d)
	}
}

func TestLegacyHandshake(t *testing.T) {
	s := newServer()
	for asked, want := range map[string]string{"2025-06-18": "2025-06-18", "2024-11-05": "2024-11-05", "1999-01-01": "2025-11-25"} {
		out := call(t, s, `{"jsonrpc":"2.0","id":4,"method":"initialize","params":{"protocolVersion":"`+asked+`"}}`)
		if got := out["result"].(map[string]any)["protocolVersion"]; got != want {
			t.Errorf("initialize asking %s answered %v, want %s", asked, got, want)
		}
	}
}

func TestPingOnlyForLegacy(t *testing.T) {
	s := newServer()
	if out := call(t, s, `{"jsonrpc":"2.0","id":5,"method":"ping"}`); out["error"] != nil {
		t.Errorf("legacy ping must answer: %v", out)
	}
	out := call(t, s, `{"jsonrpc":"2.0","id":6,"method":"ping","params":{`+modernMeta+`}}`)
	if e, _ := out["error"].(map[string]any); e == nil || e["code"].(float64) != -32601 {
		t.Errorf("ping was removed in 2026-07-28, want -32601: %v", out)
	}
}

func TestToolsCallCarriesResultType(t *testing.T) {
	s, done := testServer(t)
	defer done()
	out := call(t, s, `{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"list_models","arguments":{},`+modernMeta+`}}`)
	r := out["result"].(map[string]any)
	if r["resultType"] != "complete" || r["_meta"] == nil {
		t.Errorf("tools/call result = %v", r)
	}
}
