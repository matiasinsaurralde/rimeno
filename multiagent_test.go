package rimeno_test

// Parallel + multi-agent correctness/determinism tests. All offline: they run
// on rimenotest's scripted fake model and cost zero LLM tokens. Run under
// `go test -race` to exercise the concurrent trace/budget paths. These cover
// gaps left by subagent_test.go / budget_test.go: semaphore bound, real
// concurrency, forced out-of-order result ordering, Fatal-cancels-siblings,
// panic isolation, deadline propagation to subagents, hierarchical vs
// per-level budget characterizations, cancellation of in-flight parallel
// tools, and a goroutine-leak guard.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// --- helpers ---------------------------------------------------------------

// rawProbe builds a no-schema tool that runs fn. Used to inject concurrency
// probes, delays, panics and Fatal aborts into the parallel tool path.
func rawProbe(name string, fn func(ctx context.Context) (any, error)) rimeno.Tool {
	return rimeno.RawTool(name, "probe:"+name, json.RawMessage(`{"type":"object"}`),
		func(ctx context.Context, _ json.RawMessage) (any, error) { return fn(ctx) })
}

// fanOutModel scripts an orchestrator that requests every named tool in one turn
// (the fan-out), then answers "done".
func fanOutModel(names ...string) *rimenotest.Model {
	calls := make([]rimeno.ToolCall, len(names))
	for i, n := range names {
		calls[i] = rimenotest.ToolCall(fmt.Sprintf("c%d", i), n, nil)
	}
	return rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: calls},
		rimenotest.Turn{Text: "done"},
	)
}

// toolResults extracts tool-result message texts from a captured request, in
// order, so result ordering can be asserted.
func toolResults(req *rimeno.Request) []string {
	var out []string
	for _, m := range req.Messages {
		if m.Role == rimeno.RoleTool {
			out = append(out, m.Text)
		}
	}
	return out
}

// ctxSleep waits d or returns ctx's error if it is canceled first.
func ctxSleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// concGauge tracks concurrent in-flight count and its peak.
type concGauge struct {
	mu        sync.Mutex
	cur, peak int
}

func (g *concGauge) enter() {
	g.mu.Lock()
	g.cur++
	if g.cur > g.peak {
		g.peak = g.cur
	}
	g.mu.Unlock()
}
func (g *concGauge) leave() { g.mu.Lock(); g.cur--; g.mu.Unlock() }
func (g *concGauge) max() int {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.peak
}

// --- semaphore bounds concurrency ------------------------------------------

func TestParallel_SemaphoreBoundsConcurrency(t *testing.T) {
	const P, N = 3, 9
	var g concGauge
	tools := make([]rimeno.Tool, N)
	names := make([]string, N)
	for i := 0; i < N; i++ {
		names[i] = fmt.Sprintf("t%d", i)
		tools[i] = rawProbe(names[i], func(ctx context.Context) (any, error) {
			g.enter()
			defer g.leave()
			_ = ctxSleep(ctx, 25*time.Millisecond)
			return "ok", nil
		})
	}
	agent, err := rimeno.New(rimeno.Config{
		Model:            fanOutModel(names...),
		ParallelTools:    true,
		MaxParallelTools: P,
		Tools:            tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	if peak := g.max(); peak > P {
		t.Fatalf("peak concurrency %d exceeded MaxParallelTools %d", peak, P)
	} else if peak < 2 {
		t.Fatalf("peak concurrency %d — tools did not run in parallel", peak)
	} else {
		t.Logf("peak concurrency = %d (cap %d, %d tools)", peak, P, N)
	}
}

// --- parallel tools actually run concurrently ------------------------------

func TestParallel_RunsConcurrently(t *testing.T) {
	const N = 6
	const d = 40 * time.Millisecond
	mk := func(parallel bool) time.Duration {
		names := make([]string, N)
		tools := make([]rimeno.Tool, N)
		for i := 0; i < N; i++ {
			names[i] = fmt.Sprintf("t%d", i)
			tools[i] = rawProbe(names[i], func(ctx context.Context) (any, error) {
				return "ok", ctxSleep(ctx, d)
			})
		}
		agent, err := rimeno.New(rimeno.Config{
			Model: fanOutModel(names...), ParallelTools: parallel,
			MaxParallelTools: N, Tools: tools,
		})
		if err != nil {
			t.Fatal(err)
		}
		start := time.Now()
		if _, err := agent.Run(context.Background(), "go"); err != nil {
			t.Fatal(err)
		}
		return time.Since(start)
	}
	par := mk(true)
	seq := mk(false)
	t.Logf("parallel=%v sequential=%v (N=%d, d=%v)", par, seq, N, d)
	if seq < N/2*d {
		t.Fatalf("sequential run %v too fast — expected ~%v", seq, N*d)
	}
	if par > seq/2 {
		t.Fatalf("parallel run %v not meaningfully faster than sequential %v", par, seq)
	}
}

// --- result order preserved despite out-of-order completion ----------------

func TestParallel_ResultOrderPreservedOutOfOrder(t *testing.T) {
	// Call order a,b,c; completion order b(10ms),c(30ms),a(60ms).
	mk := func(name, ret string, d time.Duration) rimeno.Tool {
		return rawProbe(name, func(ctx context.Context) (any, error) {
			if err := ctxSleep(ctx, d); err != nil {
				return nil, err
			}
			return ret, nil
		})
	}
	model := fanOutModel("a", "b", "c")
	agent, err := rimeno.New(rimeno.Config{
		Model: model, ParallelTools: true, MaxParallelTools: 3,
		Tools: []rimeno.Tool{
			mk("a", "A", 60*time.Millisecond),
			mk("b", "B", 10*time.Millisecond),
			mk("c", "C", 30*time.Millisecond),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.Run(context.Background(), "go"); err != nil {
		t.Fatal(err)
	}
	got := toolResults(model.Requests[1])
	want := []string{"A", "B", "C"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Fatalf("tool results = %v, want %v (call order, not completion order)", got, want)
	}
}

// --- a Fatal in one parallel tool cancels the siblings ---------------------

func TestParallel_FatalCancelsSiblings(t *testing.T) {
	var canceled int32
	slow := func(name string) rimeno.Tool {
		return rawProbe(name, func(ctx context.Context) (any, error) {
			if err := ctxSleep(ctx, 2*time.Second); err != nil {
				atomic.AddInt32(&canceled, 1)
				return nil, err
			}
			return "ok", nil
		})
	}
	boom := rawProbe("boom", func(ctx context.Context) (any, error) {
		return nil, rimeno.Fatal(errors.New("boom"))
	})
	agent, err := rimeno.New(rimeno.Config{
		Model: fanOutModel("boom", "slow1", "slow2"), ParallelTools: true, MaxParallelTools: 3,
		Tools: []rimeno.Tool{boom, slow("slow1"), slow("slow2")},
	})
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	_, err = agent.Run(context.Background(), "go")
	elapsed := time.Since(start)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected a fatal 'boom' error, got %v", err)
	}
	if elapsed > time.Second {
		t.Fatalf("run took %v — siblings were not canceled by the Fatal", elapsed)
	}
	if n := atomic.LoadInt32(&canceled); n == 0 {
		t.Errorf("no sibling observed cancellation (canceled=%d)", n)
	}
}

// --- a panic in one parallel tool is isolated; the run continues -----------

func TestParallel_PanicIsolated(t *testing.T) {
	model := fanOutModel("ok1", "panic", "ok2")
	ok := func(name string) rimeno.Tool {
		return rawProbe(name, func(ctx context.Context) (any, error) { return "ok:" + name, nil })
	}
	boom := rawProbe("panic", func(ctx context.Context) (any, error) { panic("tool exploded") })
	agent, err := rimeno.New(rimeno.Config{
		Model: model, ParallelTools: true, MaxParallelTools: 3,
		Tools: []rimeno.Tool{ok("ok1"), boom, ok("ok2")},
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("panic should be contained, not abort the run: %v", err)
	}
	if res.Text != "done" {
		t.Fatalf("text = %q, want \"done\"", res.Text)
	}
	got := toolResults(model.Requests[1])
	if len(got) != 3 {
		t.Fatalf("want 3 tool results, got %v", got)
	}
	if !strings.Contains(got[1], "panic") {
		t.Errorf("panic result = %q, want it to report the panic", got[1])
	}
	if got[0] != "ok:ok1" || got[2] != "ok:ok2" {
		t.Errorf("sibling results not preserved: %v", got)
	}
}

// --- concurrent subagent trace grafting is race-free -----------------------

func TestParallel_TraceGraftRaceFree(t *testing.T) {
	const K = 6
	names := make([]string, K)
	tools := make([]rimeno.Tool, K)
	for i := 0; i < K; i++ {
		names[i] = fmt.Sprintf("w%d", i)
		w, err := rimeno.New(rimeno.Config{
			Model: rimenotest.NewModel(rimenotest.Turn{Text: "ok"}).WithID(names[i]),
			Name:  names[i],
		})
		if err != nil {
			t.Fatal(err)
		}
		tools[i] = rimeno.AgentTool(w, names[i], "worker "+names[i])
	}
	agent, err := rimeno.New(rimeno.Config{
		Model: fanOutModel(names...), ParallelTools: true, MaxParallelTools: K, Tools: tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	sum := res.Trace.Summary()
	if sum.Subagents != K {
		t.Errorf("subagents = %d, want %d", sum.Subagents, K)
	}
	if want := K + 2; sum.ModelCalls != want { // K workers + 2 orchestrator turns
		t.Errorf("model calls = %d, want %d", sum.ModelCalls, want)
	}
	if sum.ToolCalls != K {
		t.Errorf("tool calls = %d, want %d", sum.ToolCalls, K)
	}
}

// --- a subagent inherits (and is stopped by) the parent deadline -----------

func TestSubagent_InheritsParentDeadline(t *testing.T) {
	// Worker would take 2s; parent budget allows 60ms. The parent's hard deadline
	// context must propagate into the worker's model call and cut it off.
	worker, err := rimeno.New(rimeno.Config{
		Model: rimenotest.NewModel(rimenotest.Turn{Text: "late", Delay: 2 * time.Second}).WithID("slow-worker"),
		Name:  "slow-worker",
	})
	if err != nil {
		t.Fatal(err)
	}
	parent := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c0", "work", map[string]string{"task": "go"})}},
		rimenotest.Turn{Text: "unreachable"},
	)
	agent, err := rimeno.New(rimeno.Config{
		Model:  parent,
		Budget: rimeno.Budget{Timeout: 60 * time.Millisecond},
		Tools:  []rimeno.Tool{rimeno.AgentTool(worker, "work", "delegate")},
	})
	if err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	_, err = agent.Run(context.Background(), "go")
	elapsed := time.Since(start)
	if elapsed > time.Second {
		t.Fatalf("run took %v — parent deadline did not reach the subagent", elapsed)
	}
	if err == nil {
		t.Errorf("expected a deadline/budget error, got nil")
	}
}

// --- concurrent subagent budgets are reserved, so a fan-out cannot
// over-commit the parent's token cap. childBudget reserves each child's grant
// before the sibling carves, so K genuinely-concurrent children share the cap
// rather than each receiving the full balance. (A Turn.Delay keeps all K in-flight
// at carve time — with an instant model they'd finish near-sequentially and never
// contend.)

func TestSubagentBudget_ConcurrentCarvingIsBounded(t *testing.T) {
	// K unbounded workers under a cap that funds ~one of them. The reservation must
	// bound the aggregate to ~the cap (funded worker wins; the rest fail fast),
	// rather than letting all K each spend against the full balance.
	const K, perChild, parentCap = 3, 800, 1000
	names := make([]string, K)
	tools := make([]rimeno.Tool, K)
	for i := 0; i < K; i++ {
		names[i] = fmt.Sprintf("w%d", i)
		w, err := rimeno.New(rimeno.Config{
			Model: rimenotest.NewModel(rimenotest.Turn{
				Text:  "done",
				Delay: 60 * time.Millisecond, // keep all K in-flight while they carve
				Usage: &rimeno.Usage{InputTokens: perChild, TotalTokens: perChild},
			}).WithID(names[i]),
			Name: names[i],
		})
		if err != nil {
			t.Fatal(err)
		}
		tools[i] = rimeno.AgentTool(w, names[i], "worker "+names[i])
	}
	agent, err := rimeno.New(rimeno.Config{
		Model:            fanOutModel(names...),
		ParallelTools:    true,
		MaxParallelTools: K,
		Budget:           rimeno.Budget{MaxTotalTokens: parentCap},
		Tools:            tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, _ := agent.Run(context.Background(), "go")
	sum := res.Trace.Summary()
	// The healthy budget is one worker (~800). Allow one worker's between-steps slack
	// but nothing like the pre-fix 3×800=2400.
	if sum.Usage.TotalTokens > parentCap+perChild {
		t.Fatalf("aggregate usage %d exceeded parent cap %d by more than one worker — reservation not bounding fan-out",
			sum.Usage.TotalTokens, parentCap)
	}
	t.Logf("aggregate usage = %d tokens under a %d-token cap across %d concurrent workers (bounded)",
		sum.Usage.TotalTokens, parentCap, K)
}

func TestSubagentBudget_ExplicitChildBudgetsAllRun(t *testing.T) {
	// The recommended fan-out pattern: workers carry explicit budgets that sum under
	// the parent cap, so the reservation funds all of them and the aggregate stays
	// bounded.
	const K, perChild, parentCap = 3, 800, 10000
	names := make([]string, K)
	tools := make([]rimeno.Tool, K)
	for i := 0; i < K; i++ {
		names[i] = fmt.Sprintf("w%d", i)
		w, err := rimeno.New(rimeno.Config{
			Model: rimenotest.NewModel(rimenotest.Turn{
				Text:  "done",
				Delay: 60 * time.Millisecond,
				Usage: &rimeno.Usage{InputTokens: perChild, TotalTokens: perChild},
			}).WithID(names[i]),
			Name:   names[i],
			Budget: rimeno.Budget{MaxTotalTokens: 2000}, // explicit per-worker cap
		})
		if err != nil {
			t.Fatal(err)
		}
		tools[i] = rimeno.AgentTool(w, names[i], "worker "+names[i])
	}
	agent, err := rimeno.New(rimeno.Config{
		Model:            fanOutModel(names...),
		ParallelTools:    true,
		MaxParallelTools: K,
		Budget:           rimeno.Budget{MaxTotalTokens: parentCap},
		Tools:            tools,
	})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatalf("run errored: %v", err)
	}
	sum := res.Trace.Summary()
	if sum.Subagents != K {
		t.Errorf("subagents = %d, want %d (all workers should be funded)", sum.Subagents, K)
	}
	if want := K * perChild; sum.Usage.TotalTokens < want-100 {
		t.Errorf("aggregate usage %d < %d — not all workers ran", sum.Usage.TotalTokens, want)
	}
	if sum.Usage.TotalTokens > parentCap {
		t.Errorf("aggregate usage %d exceeded parent cap %d", sum.Usage.TotalTokens, parentCap)
	}
}

// --- MaxToolCalls is intentionally per-level, not carved to children.
// Token/cost budgets are hierarchical (and reserved — see above), but a
// tool-call cap is a per-agent-loop guardrail, so a parent capped at 1 tool
// call still lets its (single) subagent make several. The parent counts the
// AgentTool invocation itself as its one call.

func TestSubagentBudget_ToolCallsNotCarvedToChildren(t *testing.T) {
	noop := rawProbe("noop", func(ctx context.Context) (any, error) { return "ok", nil })
	workerModel := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("w1", "noop", nil)}},
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("w2", "noop", nil)}},
		rimenotest.Turn{Text: "worker done"},
	).WithID("worker")
	worker, err := rimeno.New(rimeno.Config{Model: workerModel, Name: "worker", Tools: []rimeno.Tool{noop}})
	if err != nil {
		t.Fatal(err)
	}
	parent := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c0", "work", map[string]string{"task": "go"})}},
		rimenotest.Turn{Text: "final"},
	)
	agent, err := rimeno.New(rimeno.Config{
		Model:  parent,
		Budget: rimeno.Budget{MaxToolCalls: 1}, // parent may make ONE tool call
		Tools:  []rimeno.Tool{rimeno.AgentTool(worker, "work", "delegate")},
	})
	if err != nil {
		t.Fatal(err)
	}
	res, _ := agent.Run(context.Background(), "go") // parent trips its own 1-call cap after the subagent returns
	sum := res.Trace.Summary()
	// 1 parent AgentTool span + 2 child noop spans = 3, i.e. the child was not
	// constrained by the parent's MaxToolCalls of 1.
	if sum.ToolCalls != 3 {
		t.Fatalf("total tool calls = %d, want 3 (1 parent + 2 child); child was unexpectedly capped", sum.ToolCalls)
	}
	if sum.Subagents != 1 {
		t.Errorf("subagents = %d, want 1", sum.Subagents)
	}
	t.Logf("parent MaxToolCalls=1 but the subagent made 2 tool calls — tool-call budget is per-level, not carved")
}

// --- a caller cancel stops in-flight parallel tools ------------------------

func TestParallel_CallerCancelStopsInflight(t *testing.T) {
	var canceled int32
	slow := func(name string) rimeno.Tool {
		return rawProbe(name, func(ctx context.Context) (any, error) {
			if err := ctxSleep(ctx, 2*time.Second); err != nil {
				atomic.AddInt32(&canceled, 1)
				return nil, err
			}
			return "ok", nil
		})
	}
	agent, err := rimeno.New(rimeno.Config{
		Model: fanOutModel("s1", "s2", "s3"), ParallelTools: true, MaxParallelTools: 3,
		Tools: []rimeno.Tool{slow("s1"), slow("s2"), slow("s3")},
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	start := time.Now()
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	_, err = agent.Run(ctx, "go")
	elapsed := time.Since(start)
	if err == nil {
		t.Fatalf("expected a cancellation error, got nil")
	}
	if elapsed > time.Second {
		t.Fatalf("run took %v — cancel did not stop in-flight tools", elapsed)
	}
	if n := atomic.LoadInt32(&canceled); n == 0 {
		t.Errorf("no tool observed cancellation (canceled=%d)", n)
	}
}

// --- fan-out does not leak goroutines --------------------------------------

func TestNoGoroutineLeakAfterFanOut(t *testing.T) {
	run := func() {
		names := []string{"w0", "w1", "w2", "w3"}
		tools := make([]rimeno.Tool, len(names))
		for i, n := range names {
			w, _ := rimeno.New(rimeno.Config{Model: rimenotest.NewModel(rimenotest.Turn{Text: "ok"}), Name: n})
			tools[i] = rimeno.AgentTool(w, n, "worker")
		}
		agent, _ := rimeno.New(rimeno.Config{
			Model: fanOutModel(names...), ParallelTools: true, MaxParallelTools: 4, Tools: tools,
		})
		if _, err := agent.Run(context.Background(), "go"); err != nil {
			t.Fatal(err)
		}
	}
	run() // warm up any lazily-created goroutines
	// settle waits (bounded) for the goroutine count to fall to target, then
	// returns it. Used only with a reachable target (baseline+slack).
	settle := func(target int) int {
		deadline := time.Now().Add(2 * time.Second)
		for {
			runtime.GC()
			n := runtime.NumGoroutine()
			if n <= target || time.Now().After(deadline) {
				return n
			}
			time.Sleep(20 * time.Millisecond)
		}
	}
	runtime.GC()
	time.Sleep(50 * time.Millisecond)
	runtime.GC()
	base := runtime.NumGoroutine()
	for i := 0; i < 20; i++ {
		run()
	}
	const slack = 10
	if final := settle(base + slack); final > base+slack {
		t.Fatalf("goroutines after 20 fan-out runs = %d, baseline %d (slack %d) — possible leak",
			final, base, slack)
	}
}

// --- tool-failure circuit breaker (Config.MaxToolErrors) -------------------

// TestToolCircuitBreaker asserts that once a tool has failed MaxToolErrors times
// in a run, further calls to it are refused without invoking it — so a model that
// keeps retrying a broken tool (e.g. a dead subagent) cannot thrash indefinitely.
func TestToolCircuitBreaker(t *testing.T) {
	var invoked int32
	flaky := rimeno.RawTool("flaky", "always fails", json.RawMessage(`{"type":"object"}`),
		func(ctx context.Context, _ json.RawMessage) (any, error) {
			atomic.AddInt32(&invoked, 1)
			return nil, errors.New("boom")
		})
	mkCall := func(i int) rimeno.ToolCall { return rimenotest.ToolCall(fmt.Sprintf("c%d", i), "flaky", nil) }
	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{mkCall(0)}},
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{mkCall(1)}},
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{mkCall(2)}},
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{mkCall(3)}},
		rimenotest.Turn{Text: "gave up"},
	)
	agent, err := rimeno.New(rimeno.Config{Model: model, Tools: []rimeno.Tool{flaky}, MaxToolErrors: 2})
	if err != nil {
		t.Fatal(err)
	}
	res, err := agent.Run(context.Background(), "go")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "gave up" {
		t.Fatalf("text = %q, want \"gave up\"", res.Text)
	}
	if n := atomic.LoadInt32(&invoked); n != 2 {
		t.Fatalf("tool invoked %d times, want 2 (circuit breaker should stop calls 3+)", n)
	}
	// The refused calls must have fed the model the 'disabled' short-circuit message.
	last := model.Requests[len(model.Requests)-1].Messages
	disabled := false
	for _, m := range last {
		if m.Role == rimeno.RoleTool && strings.Contains(m.Text, "disabled") {
			disabled = true
		}
	}
	if !disabled {
		t.Errorf("model never received the 'disabled' short-circuit message")
	}
}
