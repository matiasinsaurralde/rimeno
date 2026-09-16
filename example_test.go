package rimeno_test

import (
	"context"
	"fmt"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// Example_toolCalling shows the core DX: a tool is an ordinary Go function; the
// model calls it and the result is fed back. (A fake model is used so the example
// is deterministic; swap in openai.New for real use.)
func Example_toolCalling() {
	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 2, B: 3})}},
		rimenotest.Turn{Text: "The answer is 5."},
	)
	add := rimeno.NewTool("add", "Add two integers",
		func(_ context.Context, in addArgs) (int, error) { return in.A + in.B, nil })

	agent, _ := rimeno.New(rimeno.Config{Model: model, Tools: []rimeno.Tool{add}})
	res, _ := agent.Run(context.Background(), "what is 2 + 3?")

	fmt.Println(res.Text)
	fmt.Println("model calls:", res.Trace.Summary().ModelCalls)
	// Output:
	// The answer is 5.
	// model calls: 2
}

// Example_structuredOutput shows validated structured output decoded into a Go
// struct.
func Example_structuredOutput() {
	model := rimenotest.NewModel(rimenotest.Turn{Text: `{"score":9,"reasons":["clear","tested"]}`})
	agent, _ := rimeno.New(rimeno.Config{Model: model, Output: rimeno.OutputOf[review]()})

	res, _ := agent.Run(context.Background(), "review the PR")
	r := res.Output.(*review)

	fmt.Println(r.Score, r.Reasons)
	// Output: 9 [clear tested]
}

// Example_subagents shows an orchestrator delegating to a worker subagent via
// AgentTool. The worker runs in its own session/model; its trace nests into the
// parent's and its usage rolls up.
func Example_subagents() {
	worker, _ := rimeno.New(rimeno.Config{
		Model: rimenotest.NewModel(rimenotest.Turn{Text: "sunny, 24C"}),
		Name:  "weather",
	})
	boss := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "weather", map[string]string{"task": "weather in Paris"})}},
		rimenotest.Turn{Text: "It's sunny and 24C in Paris."},
	)
	agent, _ := rimeno.New(rimeno.Config{
		Model: boss,
		Tools: []rimeno.Tool{rimeno.AgentTool(worker, "weather", "Get the weather for a place")},
	})

	res, _ := agent.Run(context.Background(), "what's the weather in Paris?")
	fmt.Println(res.Text)
	fmt.Println("subagents:", res.Trace.Summary().Subagents)
	// Output:
	// It's sunny and 24C in Paris.
	// subagents: 1
}

// Example_persistence shows snapshotting a session and resuming it later (e.g. in
// a new process) with full context intact.
func Example_persistence() {
	model := rimenotest.NewModel(
		rimenotest.Turn{Text: "Noted: a trip to Japan."},
		rimenotest.Turn{Text: "You were planning a trip to Japan."},
	)

	agent, _ := rimeno.New(rimeno.Config{Model: model})
	sess := agent.NewSession()
	_, _ = sess.Send(context.Background(), "I'm planning a trip to Japan")
	state := sess.Snapshot() // persist this JSON anywhere

	// ...later, new process: rebuild the agent and resume from the snapshot.
	agent2, _ := rimeno.New(rimeno.Config{Model: model})
	resumed := agent2.ResumeSession(state)
	res, _ := resumed.Send(context.Background(), "what was I planning?")

	fmt.Println(res.Text)
	// Output: You were planning a trip to Japan.
}
