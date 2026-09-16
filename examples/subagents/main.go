// Command subagents is an offline rimeno example (fake models, no API key) showing
// an orchestrator delegating to two worker subagents in parallel.
//
// Run:
//
//	go run ./examples/subagents
package main

import (
	"context"
	"fmt"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

func main() {
	weather := worker("weather", "It is sunny, 24C in Paris.")
	news := worker("news", "Markets are up 1% today.")

	// The orchestrator (a scripted fake model) fans out to both workers at once.
	boss := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{
			rimenotest.ToolCall("c1", "weather", map[string]string{"task": "weather in Paris"}),
			rimenotest.ToolCall("c2", "news", map[string]string{"task": "today's headline"}),
		}},
		rimenotest.Turn{Text: "Here is your morning briefing."},
	)

	agent, err := rimeno.New(rimeno.Config{
		Model:         boss,
		ParallelTools: true, // run the two subagents concurrently (goroutines)
		Tools: []rimeno.Tool{
			rimeno.AgentTool(weather, "weather", "Get the weather for a place"),
			rimeno.AgentTool(news, "news", "Get today's news headline"),
		},
	})
	if err != nil {
		panic(err)
	}

	res, err := agent.Run(context.Background(), "give me a morning briefing")
	if err != nil {
		panic(err)
	}
	fmt.Println(res.Text)

	s := res.Trace.Summary()
	fmt.Printf("subagents=%d model_calls=%d tool_calls=%d\n", s.Subagents, s.ModelCalls, s.ToolCalls)
}

func worker(name, reply string) *rimeno.Agent {
	a, _ := rimeno.New(rimeno.Config{Model: rimenotest.NewModel(rimenotest.Turn{Text: reply}), Name: name})
	return a
}
