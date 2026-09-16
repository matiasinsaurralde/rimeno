// Package rpc exposes a rimeno [rimeno.Agent] over a line-delimited JSON-RPC 2.0
// connection (typically stdio), so a host written in any language can embed rimeno
// the way editors embed a language server or Codex embeds its app-server.
//
// The Go program that links rimeno stays in control: it supplies a [ConfigFactory]
// that builds a [rimeno.Config] — with a real model, tools, budgets, approvals —
// for each new session. The server owns the wire protocol and session lifecycle;
// your factory owns the agent.
//
// # Protocol
//
// Each line on the connection is one JSON-RPC 2.0 message (request, response, or
// notification). Requests carry an "id"; notifications do not. The server writes
// exactly one JSON object per line.
//
// Methods (client → server):
//
//   - initialize                      → {server_info:{name,version}, methods:[...]}
//   - ping                            → {pong:true}
//   - session/new    {instructions?}  → {session_id}
//   - session/prompt {session_id,input}
//     → {text, output?, usage, stop_reason, steps, summary}
//   - session/snapshot {session_id}   → {state}   (persist to resume later)
//   - session/resume {state}          → {session_id}
//   - session/close  {session_id}     → {closed:true}
//
// While a session/prompt runs, the server streams the agent's events as
// "session/update" notifications:
//
//	{"jsonrpc":"2.0","method":"session/update",
//	 "params":{"session_id":"sess-1","event":{"kind":"tool_call_started",...}}}
//
// Event payloads mirror rimeno's [rimeno.Event] types (see [eventPayload]); every one
// carries "kind", "agent", and "time".
//
// # Concurrency
//
// Serve handles each incoming message in its own goroutine, so a long-running
// prompt does not block other requests (a second session, a ping, or a close).
// Writes to the connection are serialized, so every emitted line is whole.
// Prompts on the *same* session are serialized (a [rimeno.Session] is single-turn
// at a time); different sessions run concurrently.
//
// # Example
//
//	srv := rpc.NewServer(func(ctx context.Context, p rpc.NewSessionParams) (rimeno.Config, error) {
//		return rimeno.Config{
//			Model:        openai.New(openai.WithAPIKey(key), openai.WithModel("gpt-4o-mini")),
//			Instructions: p.Instructions,
//			Tools:        []rimeno.Tool{myTool},
//		}, nil
//	})
//	log.Fatal(srv.ServeStdio(context.Background()))
package rpc
