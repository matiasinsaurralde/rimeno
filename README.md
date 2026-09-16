# rimeno

> An embeddable LLM-agent harness for Go — lean, provider-agnostic, with a dependency-free core.

[![CI](https://github.com/matiasinsaurralde/rimeno/actions/workflows/ci.yml/badge.svg)](https://github.com/matiasinsaurralde/rimeno/actions/workflows/ci.yml)
[![Lint](https://github.com/matiasinsaurralde/rimeno/actions/workflows/lint.yml/badge.svg)](https://github.com/matiasinsaurralde/rimeno/actions/workflows/lint.yml)
[![Security](https://github.com/matiasinsaurralde/rimeno/actions/workflows/security.yml/badge.svg)](https://github.com/matiasinsaurralde/rimeno/actions/workflows/security.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/matiasinsaurralde/rimeno.svg)](https://pkg.go.dev/github.com/matiasinsaurralde/rimeno)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Go 1.23+](https://img.shields.io/badge/Go-1.23%2B-00ADD8?logo=go&logoColor=white)
![core: zero dependencies](https://img.shields.io/badge/core-zero_dependencies-2ea44f)

**rimeno** (Esperanto for *strap* — the strap of a harness) is a small Go library for building
LLM agents you embed **inside your own program**. It gives you the whole agent loop —
tool-calling, budgets, streaming, structured output, multi-agent fan-out, sandboxing, and
tracing — behind a tiny API, and the core imports only the Go standard library.

It scales down to a one-shot classifier and up to a multi-file coding agent without changing
shape: the same `Agent` type, fitted to the task.

## Highlights

- **Tiny surface, whole loop.** `rimeno.New(...)` → `agent.Run(ctx, input)`. The tool-calling
  loop, retries, and finalization are handled for you.
- **Dependency-free core.** The root package and every core subpackage import only `std`.
  Tracing over OpenTelemetry is an opt-in submodule, not a core dependency.
- **Typed tools.** `rimeno.NewTool("add", "…", func(ctx, in Args) (Out, error))` — arguments
  and JSON Schema are derived from your Go types.
- **Bounded by construction.** Per-run `Budget` caps tokens, tool calls, and wall-clock; a
  per-tool circuit breaker stops a model thrashing a broken tool.
- **Multi-agent.** Agents-as-tools, parallel tool fan-out, and a small `workflow` DAG — with
  child-budget reservation so concurrent fan-out can't over-commit.
- **Batteries included.** Structured output, streaming, human-in-the-loop approvals, context
  compaction, an MCP client, a JSON-RPC server, a sandboxed shell, and reasoning-model support
  (effort control, reasoning-token accounting, Responses-API session threading).
- **Provider-agnostic.** Any OpenAI-compatible endpoint (OpenAI, OpenRouter, …) via
  `rimeno/openai`, over Chat Completions or the Responses API. Bring your own by implementing
  one `Model` interface.

## Install

```bash
go get github.com/matiasinsaurralde/rimeno
```

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/openai"
)

type addArgs struct {
	A int `json:"a" jsonschema:"description=first addend"`
	B int `json:"b" jsonschema:"description=second addend"`
}

func main() {
	model := openai.New(
		openai.WithBaseURL(openai.DefaultBaseURL), // or an OpenRouter URL
		openai.WithAPIKey(os.Getenv("RIMENO_API_KEY")),
		openai.WithModel("gpt-4o-mini"),
	)

	add := rimeno.NewTool("add", "Add two integers",
		func(ctx context.Context, in addArgs) (int, error) { return in.A + in.B, nil })

	agent, err := rimeno.New(rimeno.Config{
		Model:        model,
		Instructions: "You are precise. Use tools for arithmetic.",
		Tools:        []rimeno.Tool{add},
	})
	if err != nil {
		log.Fatal(err)
	}

	res, err := agent.Run(context.Background(), "what is 21 + 21?")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)

	s := res.Trace.Summary() // per-run tokens / cost / call counts
	fmt.Printf("model_calls=%d tool_calls=%d tokens=%d cost=$%.4f\n",
		s.ModelCalls, s.ToolCalls, s.Usage.TotalTokens, s.Usage.CostUSD)
}
```

```bash
RIMENO_API_KEY=sk-... go run ./examples/quickstart
```

Runnable examples live in [`examples/`](./examples): `quickstart`, `subagents` (parallel
fan-out), and `mcp` (driving a real MCP tool server).

## Multi-turn sessions

`Run` is a one-shot; for a conversation that retains history and accumulates usage, use a
`Session`:

```go
s := agent.NewSession()
s.Send(ctx, "remember the number 7")
res, _ := s.Send(ctx, "what number did I say?")   // → "7"
fmt.Println(s.Usage().TotalTokens)                // cumulative across turns
```

## What's in the box

| package | what it gives you |
|---|---|
| `rimeno` | agent, session, run loop, tools, budget, trace, compaction |
| `rimeno/openai` | OpenAI-compatible provider (Chat Completions + Responses API, streaming) |
| `rimeno/sandbox` | command execution — local, a persistent shell, or Docker |
| `rimeno/workflow` | a small DAG for orchestrating agents deterministically |
| `rimeno/mcp` | Model Context Protocol client |
| `rimeno/rpc` | JSON-RPC agent server |
| `rimeno/schema` | JSON Schema derivation for tool arguments and structured output |
| `rimeno/rimenotest` | scripted fake model + doubles for testing agents offline |
| `rimeno/otel` | optional OpenTelemetry trace export (separate module) |

## License

[MIT](LICENSE)
