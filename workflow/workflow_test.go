package workflow_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
	"github.com/matiasinsaurralde/rimeno/workflow"
)

func mkAgent(t *testing.T, name, reply string) (*rimeno.Agent, *rimenotest.Model) {
	t.Helper()
	m := rimenotest.NewModel(rimenotest.Turn{Text: reply}).WithID(name)
	a, err := rimeno.New(rimeno.Config{Model: m, Name: name})
	if err != nil {
		t.Fatal(err)
	}
	return a, m
}

func TestWorkflow_Sequential(t *testing.T) {
	planner, _ := mkAgent(t, "planner", "PLAN")
	researcher, resModel := mkAgent(t, "researcher", "RESEARCH")
	writer, _ := mkAgent(t, "writer", "FINAL")

	wf := workflow.New("seq").
		Step(workflow.AgentStep("plan", planner, func(s *workflow.State) string { return s.Input })).
		Step(workflow.AgentStep("research", researcher, func(s *workflow.State) string {
			return "research based on: " + s.Get("plan")
		}, "plan")).
		Step(workflow.AgentStep("write", writer, func(s *workflow.State) string {
			return "write from: " + s.Get("research")
		}, "research"))

	res, err := wf.Run(context.Background(), "topic X")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "FINAL" {
		t.Fatalf("Output = %q", res.Output)
	}
	if res.Outputs["plan"] != "PLAN" || res.Outputs["research"] != "RESEARCH" {
		t.Fatalf("outputs = %+v", res.Outputs)
	}
	// researcher must have received the plan output in its prompt.
	if got := resModel.Requests[0].Messages; !strings.Contains(lastUserText(got), "PLAN") {
		t.Errorf("researcher prompt missing plan output: %q", lastUserText(got))
	}
	if res.Usage.TotalTokens == 0 {
		t.Errorf("usage not aggregated")
	}
	if len(res.StepTraces) != 3 {
		t.Errorf("expected 3 step traces, got %d", len(res.StepTraces))
	}
}

func TestWorkflow_Parallel(t *testing.T) {
	plan, _ := mkAgent(t, "plan", "P")
	a, _ := mkAgent(t, "a", "AA")
	b, _ := mkAgent(t, "b", "BB")

	wf := workflow.New("par").
		Step(workflow.AgentStep("plan", plan, func(s *workflow.State) string { return s.Input })).
		Step(workflow.AgentStep("a", a, func(s *workflow.State) string { return s.Get("plan") }, "plan")).
		Step(workflow.AgentStep("b", b, func(s *workflow.State) string { return s.Get("plan") }, "plan")).
		Step(workflow.FuncStep("combine", func(_ context.Context, s *workflow.State) (string, error) {
			return s.Get("a") + "+" + s.Get("b"), nil
		}, "a", "b"))

	res, err := wf.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if res.Output != "AA+BB" {
		t.Fatalf("Output = %q", res.Output)
	}
}

func TestWorkflow_ErrorPropagates(t *testing.T) {
	ok, _ := mkAgent(t, "ok", "ok")
	wf := workflow.New("err").
		Step(workflow.AgentStep("ok", ok, func(s *workflow.State) string { return s.Input })).
		Step(workflow.FuncStep("boom", func(_ context.Context, _ *workflow.State) (string, error) {
			return "", errors.New("boom")
		}, "ok"))
	_, err := wf.Run(context.Background(), "x")
	if err == nil || !strings.Contains(err.Error(), "boom") || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected boom error, got %v", err)
	}
}

func TestWorkflow_MissingDependency(t *testing.T) {
	ok, _ := mkAgent(t, "ok", "ok")
	wf := workflow.New("bad").
		Step(workflow.AgentStep("ok", ok, func(s *workflow.State) string { return s.Input }, "ghost"))
	_, err := wf.Run(context.Background(), "x")
	if err == nil || !strings.Contains(err.Error(), "unknown step") {
		t.Fatalf("expected missing dependency error, got %v", err)
	}
}

func TestWorkflow_Cycle(t *testing.T) {
	ok, _ := mkAgent(t, "ok", "ok")
	ok2, _ := mkAgent(t, "ok2", "ok2")
	wf := workflow.New("cyc").
		Step(workflow.AgentStep("x", ok, func(s *workflow.State) string { return s.Input }, "y")).
		Step(workflow.AgentStep("y", ok2, func(s *workflow.State) string { return s.Input }, "x"))
	_, err := wf.Run(context.Background(), "x")
	if err == nil || !strings.Contains(err.Error(), "cycle") {
		t.Fatalf("expected cycle error, got %v", err)
	}
}

func lastUserText(msgs []rimeno.Message) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == rimeno.RoleUser {
			return msgs[i].Text
		}
	}
	return ""
}

// TestWorkflow_StepPanicRecovered verifies a panicking step fails the workflow
// with an error instead of crashing the process (steps run in goroutines).
func TestWorkflow_StepPanicRecovered(t *testing.T) {
	wf := workflow.New("panicky").
		Step(workflow.FuncStep("boom", func(_ context.Context, _ *workflow.State) (string, error) {
			panic("step exploded")
		}))
	_, err := wf.Run(context.Background(), "go")
	if err == nil {
		t.Fatal("expected an error from a panicking step")
	}
	if !strings.Contains(err.Error(), "step exploded") {
		t.Errorf("error = %v, want it to mention the panic", err)
	}
}
