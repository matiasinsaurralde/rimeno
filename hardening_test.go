package rimeno_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// TestRun_ContextCancellation verifies that canceling the context during a tool
// aborts the run with a canceled stop reason.
func TestRun_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	slow := rimeno.NewTool("slow", "blocks until canceled", func(c context.Context, _ struct{}) (string, error) {
		<-c.Done()
		return "", c.Err()
	})
	m := rimenotest.NewModel()
	m.GenerateFn = func(_ context.Context, _ *rimeno.Request, _ int) (*rimeno.Response, error) {
		return &rimeno.Response{
			Message:    rimeno.Message{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c", "slow", struct{}{})}},
			Usage:      rimeno.Usage{TotalTokens: 1},
			StopReason: rimeno.StopReasonToolCalls,
		}, nil
	}
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{slow}})

	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	res, err := agent.Run(ctx, "go")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if res.StopReason != rimeno.StopReasonCanceled {
		t.Errorf("StopReason = %q, want canceled", res.StopReason)
	}
}

// TestAgent_ConcurrentRuns verifies a single Agent is safe for concurrent Run
// calls (each run has its own session).
func TestAgent_ConcurrentRuns(t *testing.T) {
	m := rimenotest.NewModel()
	m.GenerateFn = func(_ context.Context, _ *rimeno.Request, _ int) (*rimeno.Response, error) {
		return &rimeno.Response{
			Message:    rimeno.Message{Role: rimeno.RoleAssistant, Text: "ok"},
			Usage:      rimeno.Usage{TotalTokens: 2},
			StopReason: rimeno.StopReasonStop,
		}, nil
	}
	agent, _ := rimeno.New(rimeno.Config{Model: m, Instructions: "sys"})

	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res, err := agent.Run(context.Background(), "hi")
			if err != nil {
				errs <- err
				return
			}
			if res.Text != "ok" {
				errs <- errors.New("unexpected text: " + res.Text)
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatalf("concurrent run failed: %v", err)
	}
}

// TestTool_NilResult verifies a tool returning (nil, nil) yields an empty tool
// result rather than panicking.
func TestTool_NilResult(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c", "noop", struct{}{})}},
		rimenotest.Turn{Text: "done"},
	)
	noop := rimeno.NewTool("noop", "returns nil", func(_ context.Context, _ struct{}) (any, error) { return nil, nil })
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{noop}})
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "done" {
		t.Fatalf("Text = %q", res.Text)
	}
}

// TestTool_EmptyArgs verifies a tool with no required args works when the model
// sends empty/no arguments.
func TestTool_EmptyArgs(t *testing.T) {
	m := rimenotest.NewModel()
	call := 0
	m.GenerateFn = func(_ context.Context, _ *rimeno.Request, n int) (*rimeno.Response, error) {
		call++
		if n == 0 {
			// send a tool call with empty raw args
			return &rimeno.Response{
				Message: rimeno.Message{Role: rimeno.RoleAssistant, ToolCalls: []rimeno.ToolCall{
					{ID: "c", Name: "ping", Arguments: nil},
				}},
				StopReason: rimeno.StopReasonToolCalls,
				Usage:      rimeno.Usage{TotalTokens: 1},
			}, nil
		}
		return &rimeno.Response{Message: rimeno.Message{Role: rimeno.RoleAssistant, Text: "pong"}, StopReason: rimeno.StopReasonStop}, nil
	}
	ping := rimeno.NewTool("ping", "no args", func(_ context.Context, _ struct{}) (string, error) { return "pinged", nil })
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{ping}})
	res, err := agent.Run(context.Background(), "ping")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "pong" {
		t.Fatalf("Text = %q", res.Text)
	}
}

// TestTool_PanicRecovered verifies a tool that panics does not crash the run: the
// panic is converted to a recoverable tool error and the loop continues.
func TestTool_PanicRecovered(t *testing.T) {
	m := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c", "boom", struct{}{})}},
		rimenotest.Turn{Text: "recovered"},
	)
	boom := rimeno.NewTool("boom", "panics", func(_ context.Context, _ struct{}) (string, error) {
		panic("kaboom")
	})
	agent, _ := rimeno.New(rimeno.Config{Model: m, Tools: []rimeno.Tool{boom}})
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("run should survive a tool panic, got: %v", err)
	}
	if res.Text != "recovered" {
		t.Fatalf("Text = %q, want recovered", res.Text)
	}
	// The panic should have been fed back to the model as a tool result.
	var sawPanic bool
	for _, msg := range res.Messages {
		if msg.Role == rimeno.RoleTool && strings.Contains(msg.Text, "kaboom") {
			sawPanic = true
		}
	}
	if !sawPanic {
		t.Errorf("expected the panic message in a tool result: %+v", res.Messages)
	}
}
