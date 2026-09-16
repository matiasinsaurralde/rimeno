package rimeno_test

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// TestBudget_TimeoutHardInterrupt verifies Budget.Timeout is a hard wall-clock
// bound: a model call that blocks until its context is canceled is interrupted at
// the deadline (without the deadline wired into ctx, this test would hang).
func TestBudget_TimeoutHardInterrupt(t *testing.T) {
	m := rimenotest.NewModel()
	m.GenerateFn = func(ctx context.Context, _ *rimeno.Request, _ int) (*rimeno.Response, error) {
		<-ctx.Done() // simulate a slow/hung provider
		return nil, ctx.Err()
	}
	agent, _ := rimeno.New(rimeno.Config{Model: m, Budget: rimeno.Budget{Timeout: 50 * time.Millisecond}})

	start := time.Now()
	res, err := agent.Run(context.Background(), "go")
	if elapsed := time.Since(start); elapsed > 5*time.Second {
		t.Fatalf("run was not interrupted promptly: %s", elapsed)
	}

	var be *rimeno.BudgetError
	if !errors.As(err, &be) {
		t.Fatalf("err = %v, want *rimeno.BudgetError", err)
	}
	if !errors.Is(err, rimeno.ErrBudgetExceeded) {
		t.Errorf("err should match ErrBudgetExceeded")
	}
	if be.Limit != "deadline" {
		t.Errorf("limit = %q, want deadline", be.Limit)
	}
	if res.StopReason != rimeno.StopReasonBudget {
		t.Errorf("stop = %q, want budget", res.StopReason)
	}
}

// TestBudget_CallerCancelStillCancels verifies that a caller's own cancellation
// (no budget deadline) is still reported as a cancellation, not a budget stop.
func TestBudget_CallerCancelStillCancels(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	m := rimenotest.NewModel()
	m.GenerateFn = func(c context.Context, _ *rimeno.Request, _ int) (*rimeno.Response, error) {
		<-c.Done()
		return nil, c.Err()
	}
	agent, _ := rimeno.New(rimeno.Config{Model: m}) // no budget

	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()
	res, err := agent.Run(ctx, "go")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if res.StopReason != rimeno.StopReasonCanceled {
		t.Errorf("stop = %q, want canceled", res.StopReason)
	}
}
