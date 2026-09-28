package main

import (
	"encoding/json"
)

// ollama-mcp-go is a dual-era MCP server (spec 2026-07-28, "Versioning and
// Compatibility"): modern clients declare the protocol version on every
// request in _meta and may probe with server/discover; legacy clients open
// with the initialize handshake. Both are served from the same loop.

const modernVersion = "2026-07-28"

// legacyVersions are the handshake-era revisions answered through
// initialize. The tool surface is the same across all of them.
var legacyVersions = []string{"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}

// supportedVersions is what server/discover and UnsupportedProtocolVersion
// errors report.
var supportedVersions = append([]string{modernVersion}, legacyVersions...)

const (
	metaVersion    = "io.modelcontextprotocol/protocolVersion"
	metaServerInfo = "io.modelcontextprotocol/serverInfo"

	// errUnsupportedVersion is UnsupportedProtocolVersionError (2026-07-28).
	errUnsupportedVersion = -32022

	// toolsTTLMs: the tool list is compiled in, so clients may cache it for
	// an hour; it changes only with a new binary.
	toolsTTLMs = 3600000
)

var serverInfo = map[string]any{"name": "ollama-mcp-go", "version": version}

// requestVersion returns the protocol version a modern request declares in
// params._meta, or "" for a legacy request.
func requestVersion(params json.RawMessage) string {
	var p struct {
		Meta map[string]any `json:"_meta"`
	}
	if len(params) == 0 || json.Unmarshal(params, &p) != nil {
		return ""
	}
	v, _ := p.Meta[metaVersion].(string)
	return v
}

// dispatch answers one request. Modern requests (version in _meta) must name
// a version this server implements, or get UnsupportedProtocolVersionError
// with the list to choose from. ping exists only for legacy clients: the
// 2026-07-28 revision removed it.
func (s *server) dispatch(req rpcRequest) rpcResponse {
	v := requestVersion(req.Params)
	modern := v != ""
	if modern && v != modernVersion {
		return rpcResponse{JSONRPC: "2.0", ID: req.ID, Error: &rpcError{
			Code:    errUnsupportedVersion,
			Message: "Unsupported protocol version",
			Data:    map[string]any{"supported": supportedVersions, "requested": v},
		}}
	}

	switch req.Method {
	case "server/discover":
		return s.handleDiscover(req.ID)
	case "initialize":
		return s.handleInitialize(req.ID, req.Params)
	case "ping":
		if !modern {
			return rpcResponse{JSONRPC: "2.0", ID: req.ID, Result: map[string]any{}}
		}
	case "tools/list":
		return s.handleToolsList(req.ID)
	case "tools/call":
		return s.handleToolsCall(req.ID, req.Params)
	}
	return rpcResponse{JSONRPC: "2.0", ID: req.ID,
		Error: &rpcError{Code: -32601, Message: "method not found: " + req.Method}}
}

func (s *server) handleDiscover(id json.RawMessage) rpcResponse {
	return rpcResponse{JSONRPC: "2.0", ID: id, Result: map[string]any{
		"resultType":        "complete",
		"supportedVersions": supportedVersions,
		"capabilities":      map[string]any{"tools": map[string]any{"listChanged": false}},
		"instructions": "Local Ollama models: list_models first, then generate (one shot), " +
			"chat (multi-turn) or embed. Nothing leaves the machine.",
		"_meta":      map[string]any{metaServerInfo: serverInfo},
		"ttlMs":      toolsTTLMs,
		"cacheScope": "public",
	}}
}

// negotiate picks the legacy version to answer initialize with: the one the
// client asked for when supported, else the newest legacy one.
func negotiate(params json.RawMessage) string {
	var p struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if json.Unmarshal(params, &p) == nil {
		for _, v := range legacyVersions {
			if v == p.ProtocolVersion {
				return v
			}
		}
	}
	return legacyVersions[0]
}
