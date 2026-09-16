package rimeno_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

type addArgs struct {
	A int `json:"a"`
	B int `json:"b"`
}

func newAddTool(calls *int) rimeno.Tool {
	return rimeno.NewTool("add", "add two integers", func(_ context.Context, in addArgs) (int, error) {
		*calls++
		return in.A + in.B, nil
	})
}

func TestRun_PlainAnswer(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: "hello there"})
	agent, err := rimeno.New(rimeno.Config{Model: m, Instructions: "be nice"})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "hi")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "hello there" {
		t.Fatalf("Text = %q", res.Text)
	}
	if res.StopReason != rimeno.StopReasonStop {
		t.Fatalf("StopReason = %q", res.StopReason)
	}
	// system instruction + user must be in the first request.
	if len(m.Requests) != 1 {
		t.Fatalf("model calls = %d, want 1", len(m.Requests))
	}
	first := m.Requests[0].Messages
	if first[0].Role != rimeno.RoleSystem || first[0].Text != "be nice" {
		t.Errorf("missing system instruction: %+v", first[0])
	}
}

func TestRun_ToolCallThenAnswer(t *testing.T) {
	calls := 0
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 21, B: 21})}},
		rimenotest.Turn{Text: "the answer is 42"},
	)
	agent, err := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{newAddTool(&calls)}})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "21+21?")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("tool calls = %d, want 1", calls)
	}
	if res.Text != "the answer is 42" {
		t.Fatalf("Text = %q", res.Text)
	}
	if res.Steps != 2 {
		t.Fatalf("Steps = %d, want 2", res.Steps)
	}
	// second request must include the tool result message
	last := m.Requests[1].Messages
	foundToolResult := false
	for _, msg := range last {
		if msg.Role == rimeno.RoleTool && strings.Contains(msg.Text, "42") {
			foundToolResult = true
		}
	}
	if !foundToolResult {
		t.Errorf("tool result 42 not fed back to model: %+v", last)
	}
	// trace summary
	sum := res.Trace.Summary()
	if sum.ModelCalls != 2 || sum.ToolCalls != 1 {
		t.Errorf("summary = %+v, want 2 model / 1 tool", sum)
	}
	if sum.Usage.TotalTokens == 0 {
		t.Errorf("summary usage not aggregated: %+v", sum.Usage)
	}
}

func TestRun_UnknownToolIsRecoverable(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "missing", map[string]any{})}},
		rimenotest.Turn{Text: "recovered"},
	)
	agent, _ := rimeno.New(rimeno.Config{Model: m})
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("unknown tool should be recoverable, got %v", err)
	}
	if res.Text != "recovered" {
		t.Fatalf("Text = %q", res.Text)
	}
}

func TestRun_FatalToolAborts(t *testing.T) {
	boom := rimeno.NewTool("boom", "always fatal", func(_ context.Context, _ struct{}) (string, error) {
		return "", rimeno.Fatal(errors.New("kaboom"))
	})
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "boom", struct{}{})}},
		rimenotest.Turn{Text: "should not reach"},
	)
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{boom}})
	_, err := agent.Run(context.Background(), "go")
	if err == nil || !strings.Contains(err.Error(), "kaboom") {
		t.Fatalf("expected fatal error, got %v", err)
	}
}

func TestRun_MaxSteps(t *testing.T) {
	// model always asks for a tool → loop must stop at MaxSteps.
	m := rimenotest.NewModel()
	m.GenerateFn = func(_ context.Context, _ *rimeno.Request, _ int) (*rimeno.Response, error) {
		return &rimeno.Response{
			Message:    rimeno.Message{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c", "noop", struct{}{})}},
			Usage:      rimeno.Usage{TotalTokens: 1},
			StopReason: rimeno.StopReasonToolCalls,
		}, nil
	}
	noop := rimeno.NewTool("noop", "no op", func(_ context.Context, _ struct{}) (string, error) { return "ok", nil })
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{noop}, MaxSteps: 3})
	res, err := agent.Run(context.Background(), "loop")
	if err != nil {
		t.Fatal(err)
	}
	if res.StopReason != rimeno.StopReasonMaxSteps {
		t.Fatalf("StopReason = %q, want max_steps", res.StopReason)
	}
	if res.Steps != 3 {
		t.Fatalf("Steps = %d, want 3", res.Steps)
	}
}

func TestRun_BudgetToolCalls(t *testing.T) {
	m := rimenotest.NewModel()
	m.GenerateFn = func(_ context.Context, _ *rimeno.Request, _ int) (*rimeno.Response, error) {
		return &rimeno.Response{
			Message:    rimeno.Message{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c", "noop", struct{}{})}},
			Usage:      rimeno.Usage{TotalTokens: 1},
			StopReason: rimeno.StopReasonToolCalls,
		}, nil
	}
	noop := rimeno.NewTool("noop", "no op", func(_ context.Context, _ struct{}) (string, error) { return "ok", nil })
	agent, _ := rimeno.New(rimeno.Config{
		Model:    m,
		Tools:    []rimeno.Tool{noop},
		MaxSteps: 100,
		Budget:   rimeno.Budget{MaxToolCalls: 2},
	})
	_, err := agent.Run(context.Background(), "loop")
	if !errors.Is(err, rimeno.ErrBudgetExceeded) {
		t.Fatalf("expected budget error, got %v", err)
	}
	var be *rimeno.BudgetError
	if !errors.As(err, &be) || be.Limit != "tool_calls" {
		t.Fatalf("expected tool_calls budget error, got %v", err)
	}
}

func TestRun_Approval(t *testing.T) {
	calls := 0
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "add", addArgs{A: 1, B: 2})}},
		rimenotest.Turn{Text: "done"},
	)
	agent, _ := rimeno.New(rimeno.Config{
		Model: m,
		Tools: []rimeno.Tool{newAddTool(&calls)},
		Approve: func(_ context.Context, req rimeno.ApprovalRequest) (rimeno.ApprovalDecision, string) {
			return rimeno.ApprovalDeny, "not allowed"
		},
	})
	res, err := agent.Run(context.Background(), "add")
	if err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("tool should not run when denied, calls = %d", calls)
	}
	// denial reason should have been fed back to the model.
	last := m.Requests[1].Messages
	found := false
	for _, msg := range last {
		if msg.Role == rimeno.RoleTool && strings.Contains(msg.Text, "not allowed") {
			found = true
		}
	}
	if !found {
		t.Errorf("denial reason not fed back: %+v", last)
	}
	_ = res
}

func TestSession_MultiTurn(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{Text: "first"},
		rimenotest.Turn{Text: "second"},
	)
	agent, _ := rimeno.New(rimeno.Config{Model: m, Instructions: "sys"})
	sess := agent.NewSession()
	if _, err := sess.Send(context.Background(), "one"); err != nil {
		t.Fatal(err)
	}
	if _, err := sess.Send(context.Background(), "two"); err != nil {
		t.Fatal(err)
	}
	// second request should contain history from the first turn.
	second := m.Requests[1].Messages
	joined := ""
	for _, msg := range second {
		joined += string(msg.Role) + ":" + msg.Text + "|"
	}
	for _, want := range []string{"sys", "one", "first", "two"} {
		if !strings.Contains(joined, want) {
			t.Errorf("history missing %q in %q", want, joined)
		}
	}
	if sess.Usage().TotalTokens == 0 {
		t.Error("session usage not accumulated")
	}
	if len(sess.Traces()) != 2 {
		t.Errorf("expected 2 traces, got %d", len(sess.Traces()))
	}
}

func TestRun_Timeout(t *testing.T) {
	m := rimenotest.NewModel(rimenotest.Turn{Text: "hi"})
	agent, _ := rimeno.New(rimeno.Config{Model: m, Budget: rimeno.Budget{Deadline: time.Now().Add(-time.Second)}})
	_, err := agent.Run(context.Background(), "hi")
	if !errors.Is(err, rimeno.ErrBudgetExceeded) {
		t.Fatalf("expected deadline budget error, got %v", err)
	}
}
