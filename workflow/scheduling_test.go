package workflow_test

// Workflow DAG scheduling gaps not covered by workflow_test.go:
// (1) independent steps really run concurrently (timing overlap), and
// (2) the first error cancels in-flight sibling steps (fail-fast is not just
// "returns an error" — the running siblings must be stopped). Offline, zero tokens.

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno/workflow"
)

// TestWorkflow_IndependentStepsRunConcurrently asserts two dependency-free steps
// overlap in wall-clock time, not merely that their outputs are correct.
func TestWorkflow_IndependentStepsRunConcurrently(t *testing.T) {
	const d = 60 * time.Millisecond
	var mu sync.Mutex
	var aStart, aEnd, bStart, bEnd time.Time
	mark := func(s *time.Time) { mu.Lock(); *s = time.Now(); mu.Unlock() }

	wf := workflow.New("concurrent").
		Step(workflow.FuncStep("a", func(ctx context.Context, _ *workflow.State) (string, error) {
			mark(&aStart)
			time.Sleep(d)
			mark(&aEnd)
			return "A", nil
		})).
		Step(workflow.FuncStep("b", func(ctx context.Context, _ *workflow.State) (string, error) {
			mark(&bStart)
			time.Sleep(d)
			mark(&bEnd)
			return "B", nil
		}))

	start := time.Now()
	if _, err := wf.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	if elapsed >= 2*d {
		t.Fatalf("elapsed %v ~ serial (2×%v) — independent steps did not run concurrently", elapsed, d)
	}
	// Their execution intervals must overlap.
	mu.Lock()
	defer mu.Unlock()
	if !aStart.Before(bEnd) || !bStart.Before(aEnd) {
		t.Fatalf("step intervals did not overlap: a=[%v,%v] b=[%v,%v]", aStart, aEnd, bStart, bEnd)
	}
}

// TestWorkflow_FailFastCancelsSiblings asserts that when one ready step fails, a
// concurrently-running sibling is canceled (not left to finish).
func TestWorkflow_FailFastCancelsSiblings(t *testing.T) {
	var canceled int32
	wf := workflow.New("failfast").
		Step(workflow.FuncStep("fast", func(ctx context.Context, _ *workflow.State) (string, error) {
			time.Sleep(20 * time.Millisecond)
			return "", errBoom
		})).
		Step(workflow.FuncStep("slow", func(ctx context.Context, _ *workflow.State) (string, error) {
			select {
			case <-time.After(2 * time.Second):
				return "slow-done", nil
			case <-ctx.Done():
				atomic.AddInt32(&canceled, 1)
				return "", ctx.Err()
			}
		}))

	start := time.Now()
	_, err := wf.Run(context.Background(), "go")
	elapsed := time.Since(start)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected the boom error, got %v", err)
	}
	if elapsed > time.Second {
		t.Fatalf("run took %v — the slow sibling was not canceled on fail-fast", elapsed)
	}
	if atomic.LoadInt32(&canceled) == 0 {
		t.Errorf("slow sibling did not observe cancellation")
	}
}

type boomError struct{}

func (boomError) Error() string { return "boom" }

var errBoom = boomError{}
