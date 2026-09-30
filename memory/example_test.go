package memory_test

import (
	"context"
	"fmt"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/memory"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// Example shows the core flow: a scoped memory, an agent using the memory tools,
// and recall of what was stored. A fake model is used so the example is
// deterministic; swap in openai.New for real use.
func Example() {
	ctx := context.Background()
	mem := memory.New(memory.NewInMemory(), "task-1")

	// The model calls `remember`, then answers.
	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{
			rimenotest.ToolCall("c1", "remember", map[string]any{
				"kind": "preference", "text": "user prefers dark mode", "salience": 0.9,
			}),
		}},
		rimenotest.Turn{Text: "Got it — I'll remember that."},
	)

	agent, _ := rimeno.New(rimeno.Config{Model: model, Tools: mem.Tools()})
	res, _ := agent.Run(ctx, "I prefer dark mode")
	fmt.Println(res.Text)

	// The fact is now durable and recallable.
	recs, _ := mem.Recall(ctx, memory.Query{Kinds: []string{"preference"}})
	fmt.Println("remembered:", recs[0].Text)

	// Output:
	// Got it — I'll remember that.
	// remembered: user prefers dark mode
}

// ExampleMemory_Preamble shows building a budgeted context block from the most
// salient records — what you inject at the top of a session to ground the model.
func ExampleMemory_Preamble() {
	ctx := context.Background()
	mem := memory.New(memory.NewInMemory(), "task-1")
	_, _ = mem.Remember(ctx, memory.Record{Kind: "fact", Text: "deploy target is us-east-1", Salience: 0.9})
	_, _ = mem.Remember(ctx, memory.Record{Kind: "fact", Text: "CI runs on Go 1.25", Salience: 0.8})

	preamble, _ := mem.Preamble(ctx, memory.Query{}, 256)
	fmt.Println(preamble)

	// Output:
	// Recalled memory (durable facts from earlier in this task):
	// - [fact] deploy target is us-east-1
	// - [fact] CI runs on Go 1.25
}
