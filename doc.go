// Package rimeno is an embeddable Go agent harness for OpenAI-compatible models.
//
// Unlike a CLI agent, rimeno is a library: you import it, define tools as ordinary
// Go functions, and drive one agent — or a graph of subagents — with full
// programmatic control over budgets, approvals, structured output, compaction,
// and tracing.
//
// # Quick start
//
//	model := openai.New(openai.WithAPIKey(key), openai.WithModel("gpt-4o-mini"))
//
//	type addArgs struct {
//	    A int `json:"a" jsonschema:"description=first addend"`
//	    B int `json:"b" jsonschema:"description=second addend"`
//	}
//	add := rimeno.NewTool("add", "Add two integers",
//	    func(ctx context.Context, in addArgs) (int, error) { return in.A + in.B, nil })
//
//	agent, err := rimeno.New(rimeno.Config{
//	    Model:        model,
//	    Instructions: "You are a precise calculator. Use tools for arithmetic.",
//	    Tools:        []rimeno.Tool{add},
//	    Budget:       rimeno.Budget{MaxToolCalls: 10, Timeout: 30 * time.Second},
//	})
//	res, err := agent.Run(ctx, "what is 21 + 21?")
//	fmt.Println(res.Text)
//	fmt.Printf("%+v\n", res.Trace.Summary()) // cost/tokens/calls
//
// # Concepts
//
//   - [Model] abstracts a provider; the openai subpackage implements the
//     OpenAI-compatible chat-completions API, and rimenotest provides a scripted
//     fake for deterministic tests.
//
//   - [Tool] is any callable; [NewTool] builds one from a typed function with a
//     reflection-derived JSON Schema.
//
//   - [Agent] runs the loop; [Session] keeps multi-turn history.
//
//   - [OutputOf] enforces and validates structured output.
//
//   - [Budget] bounds tokens/cost/tool-calls/wall-clock, hierarchically for
//     subagents.
//
//   - [Trace] records every step of a run (cost, tokens, call counts).
//
//   - [Compactor] manages context-window compaction.
//
//   - [SessionState] with [Session.Snapshot]/[Agent.ResumeSession] persists and
//     resumes a conversation across process restarts.
//
// # Subpackages
//
//   - openai — the OpenAI-compatible provider (Generate + streaming, pricing).
//   - rimenotest — a scripted, deterministic Model for testing agents offline.
//   - workflow — explicit multi-agent orchestration as a DAG of steps.
//   - rpc — serve rimeno over JSON-RPC 2.0 (embed it from any language).
//   - mcp — a Model Context Protocol client to consume external tool servers.
//   - sandbox — a Provider/Sandbox seam for running commands in isolation, plus
//     a run_command tool.
//
// Subagents and multi-agent workflows build on these primitives: [AgentTool]
// turns an agent into a tool another agent can call (parallel subagents run as
// goroutines under a budget carved from the parent's), and the workflow
// subpackage wires agents into an explicit graph.
package rimeno
