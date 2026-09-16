package rimeno_test

import (
	"context"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

func TestAgentTool_Delegation(t *testing.T) {
	// Worker (model B) answers directly.
	workerModel := rimenotest.NewModel(rimenotest.Turn{Text: "found: the capital is Paris"}).WithID("model-B")
	worker, err := rimeno.New(rimeno.Config{Model: workerModel, Name: "researcher", Instructions: "research"})
	if err != nil {
		t.Fatal(err)
	}

	// Orchestrator (model A) delegates then answers.
	bossModel := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "research", map[string]string{"task": "capital of France?"})}},
		rimenotest.Turn{Text: "The capital of France is Paris."},
	).WithID("model-A")

	var subStarted, subFinished bool
	boss, err := rimeno.New(rimeno.Config{
		Model: bossModel,
		Name:  "boss",
		Tools: []rimeno.Tool{rimeno.AgentTool(worker, "research", "Delegate a research task")},
		OnEvent: func(e rimeno.Event) {
			switch e.(type) {
			case rimeno.SubAgentStartedEvent:
				subStarted = true
			case rimeno.SubAgentFinishedEvent:
				subFinished = true
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	res, err := boss.Run(context.Background(), "what is the capital of France?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "The capital of France is Paris." {
		t.Fatalf("text = %q", res.Text)
	}
	if !subStarted || !subFinished {
		t.Errorf("subagent lifecycle events missing: started=%v finished=%v", subStarted, subFinished)
	}

	// Worker output must have been fed back to the orchestrator.
	last := bossModel.Requests[1].Messages
	found := false
	for _, m := range last {
		if m.Role == rimeno.RoleTool && strings.Contains(m.Text, "Paris") {
			found = true
		}
	}
	if !found {
		t.Errorf("subagent result not fed back: %+v", last)
	}

	sum := res.Trace.Summary()
	if sum.Subagents != 1 {
		t.Errorf("subagents = %d, want 1", sum.Subagents)
	}
	if sum.ModelCalls != 3 { // 2 orchestrator + 1 worker
		t.Errorf("model calls = %d, want 3", sum.ModelCalls)
	}
	if sum.Usage.TotalTokens == 0 {
		t.Errorf("usage not rolled up: %+v", sum.Usage)
	}
}

func TestAgentTool_ParallelFanOut(t *testing.T) {
	mkWorker := func(name, answer string) *rimeno.Agent {
		w, _ := rimeno.New(rimeno.Config{Model: rimenotest.NewModel(rimenotest.Turn{Text: answer}), Name: name})
		return w
	}
	a := mkWorker("a", "result-A")
	b := mkWorker("b", "result-B")

	boss := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{
			rimenotest.ToolCall("c1", "ask_a", map[string]string{"task": "do A"}),
			rimenotest.ToolCall("c2", "ask_b", map[string]string{"task": "do B"}),
		}},
		rimenotest.Turn{Text: "combined"},
	)
	agent, err := rimeno.New(rimeno.Config{
		Model:         boss,
		ParallelTools: true,
		Tools: []rimeno.Tool{
			rimeno.AgentTool(a, "ask_a", "ask A"),
			rimeno.AgentTool(b, "ask_b", "ask B"),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "fan out")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "combined" {
		t.Fatalf("text = %q", res.Text)
	}
	// both subagent results present in the second request, order preserved
	last := boss.Requests[1].Messages
	var toolResults []string
	for _, m := range last {
		if m.Role == rimeno.RoleTool {
			toolResults = append(toolResults, m.Text)
		}
	}
	if len(toolResults) != 2 || toolResults[0] != "result-A" || toolResults[1] != "result-B" {
		t.Fatalf("tool results = %v, want [result-A result-B]", toolResults)
	}
	if sum := res.Trace.Summary(); sum.Subagents != 2 {
		t.Errorf("subagents = %d, want 2", sum.Subagents)
	}
}
