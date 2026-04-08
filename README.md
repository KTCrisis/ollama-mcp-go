# ollama-mcp-go

**MCP server for Ollama.** Lets Claude Code, Cursor, or any MCP client talk to your local Ollama models.

One binary. Zero dependencies. Runs over stdio.

## Why this one

There are several Ollama MCP servers out there. This one exists because:

- **Go, single binary** — no npm, no pip, no runtime. Download and run.
- **Agent-oriented** — exposes `generate` (with system prompts), `chat` (multi-turn), and `embed`. No admin tools (pull/push/delete) cluttering the tool list.
- **Designed for [agent-mesh](https://github.com/KTCrisis/agent-mesh)** — works as an upstream MCP server behind policy, tracing, and approval workflows. Also works standalone with any MCP client.

## Tools

| Tool | Description |
|---|---|
| `list_models` | List all available Ollama models |
| `generate` | One-shot generation with optional system prompt |
| `chat` | Multi-turn conversation (system/user/assistant messages) |
| `embed` | Generate embeddings (for semantic search, RAG, etc.) |

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

## Protocol

Implements [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) over stdio transport using JSON-RPC 2.0. Compatible with protocol version `2024-11-05`.

## License

MIT
