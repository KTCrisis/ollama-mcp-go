# ollama-mcp-go

**MCP server for Ollama.** Lets Claude Code, Cursor, or any MCP client talk to your local Ollama models.

One binary. Zero dependencies. Runs over stdio.

## Why this one

There are several Ollama MCP servers out there. This one exists because:

- **Go, single binary** — no npm, no pip, no runtime. Download and run.
- **Agent-oriented** — exposes `generate` (with system prompts), `chat` (multi-turn), and `embed`. No admin tools (pull/push/delete) cluttering the tool list.
- **Designed for [agent-mesh](https://github.com/KTCrisis/agent-mesh)** — works as an upstream MCP server behind policy, tracing, and approval workflows. Also works standalone with any MCP client.
- **Watchable** — MCP delivers a tool result in one piece, so a long reply is invisible until it is done. This server streams from Ollama and mirrors every token to a trace file as it arrives, reasoning included. `tail -f` and you see the model write.

## Tools

| Tool | Description |
|---|---|
| `list_models` | List all available Ollama models |
| `generate` | One-shot generation with optional system prompt |
| `chat` | Multi-turn conversation (system/user/assistant messages) |
| `embed` | Generate embeddings (for semantic search, RAG, etc.) |

`generate` and `chat` stream internally and mirror their output to a trace
file — see [Watching a reply as it is written](#watching-a-reply-as-it-is-written).
The MCP client still receives the reply in one piece, along with the
`prompt_eval_count` / `eval_count` usage counters.

Conversation state lives entirely on the client: `/api/chat` is stateless, so
every call sends the whole history. Nothing accumulates server-side, and
nothing needs clearing here.

## Install

### From source

```bash
git clone https://github.com/KTCrisis/ollama-mcp-go.git
cd ollama-mcp-go
go build -o ollama-mcp-go .
```

Requires Go 1.22+ and a running [Ollama](https://ollama.com) instance.

### Pre-built binaries

Coming soon.

## Setup with Claude Code

Add to your Claude Code MCP config (`~/.claude/settings.json` or project `.mcp.json`):

```json
{
  "mcpServers": {
    "ollama": {
      "command": "/path/to/ollama-mcp-go"
    }
  }
}
```

If Ollama runs on a non-default host:

```json
{
  "mcpServers": {
    "ollama": {
      "command": "/path/to/ollama-mcp-go",
      "env": {
        "OLLAMA_HOST": "http://192.168.1.10:11434"
      }
    }
  }
}
```

## Setup with agent-mesh

```yaml
mcp_servers:
  - name: ollama
    transport: stdio
    command: /path/to/ollama-mcp-go

policies:
  - name: agents
    agent: "*"
    rules:
      - tools: ["ollama.*"]
        action: allow
```

## Usage examples

Once connected, your MCP client can call:

**Generate with system prompt:**
```json
{
  "name": "generate",
  "arguments": {
    "model": "qwen3:14b",
    "prompt": "Explain service meshes in 2 sentences.",
    "system": "You are a concise technical writer."
  }
}
```

**Multi-turn chat:**
```json
{
  "name": "chat",
  "arguments": {
    "model": "llama3:8b",
    "messages": [
      {"role": "system", "content": "You answer in French."},
      {"role": "user", "content": "What is the capital of Germany?"}
    ]
  }
}
```

**Embeddings:**
```json
{
  "name": "embed",
  "arguments": {
    "model": "nomic-embed-text",
    "text": "AI agent governance"
  }
}
```

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MCP_TRACE` | `/tmp/ollama-mcp-trace.log` | Where replies are mirrored as they stream. Set to `off` to disable. |

## Watching a reply as it is written

`generate` and `chat` stream from Ollama, but MCP has no way to deliver a
partial tool result: the client receives the reply in one piece, at the end.
So the server mirrors every token to a trace file the moment it arrives.

```bash
tail -f /tmp/ollama-mcp-trace.log
```

Each call writes a header, the prompt, the reply as it lands, and a footer
with the token counts:

```
=== 21:15:33  chat  gpt-oss:20b ===
> Dis bonjour en trois mots exactement.
---
--- pense ---
Three words greeting: "Bonjour à tous" - Bonjour(1) à(2) tous(3)...
--- repond ---
Bonjour à tous.
[done: 75 prompt / 201 eval]
```

Models that emit a separate reasoning channel (gpt-oss and friends) have it
mirrored under `--- pense ---`. Reasoning **never** reaches the MCP client —
it goes to the trace and nowhere else, so the answer stays clean while the
deliberation stays visible.

Failing to open the trace file is not fatal — tracing turns itself off and
generation carries on.

## Protocol

Implements [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) over stdio transport using JSON-RPC 2.0. Compatible with protocol version `2024-11-05`.

## License

MIT
