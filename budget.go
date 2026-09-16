package rimeno

import (
	"sync"
	"time"
)

// Budget bounds a run's resource consumption. The zero value imposes no limits
// (aside from Config.MaxSteps). Token/cost/tool-call limits are enforced before
// each step and after each model call; the wall-clock deadline is additionally
// wired into the run's context, so a long model or tool call is hard-interrupted
// at the deadline rather than only caught once it returns. For subagents, budgets
// are allocated hierarchically so a child can never exceed its parent's remaining
// allowance.
type Budget struct {
	// MaxTotalTokens caps cumulative input+output tokens for the run.
	MaxTotalTokens int
	// MaxCostUSD caps cumulative estimated cost for the run.
	MaxCostUSD float64
	// MaxToolCalls caps the number of tool invocations.
	MaxToolCalls int
	// Timeout bounds wall-clock time from the start of the run (enforced as a hard
	// context deadline). If both Timeout and Deadline are set, the earlier applies.
	Timeout time.Duration
	// Deadline is an absolute wall-clock cutoff (also a hard context deadline).
	Deadline time.Time
}

// budgetTracker enforces a Budget and tracks per-run state (usage, tool-call and
// tool-error counts, and reservations held by in-flight subagents). It is safe
// for concurrent use so parallel tool calls and subagents can update it from
// multiple goroutines.
type budgetTracker struct {
	mu             sync.Mutex
	b              Budget
	used           Usage
	toolCalls      int
	deadline       time.Time
	reservedTokens int            // tokens granted to in-flight subagents, not yet settled
	reservedCost   float64        // cost granted to in-flight subagents, not yet settled
	toolErrors     map[string]int // per-tool error counts this run (for the circuit breaker)
}

func newBudgetTracker(b Budget) *budgetTracker {
	t := &budgetTracker{b: b, toolErrors: map[string]int{}}
	if !b.Deadline.IsZero() {
		t.deadline = b.Deadline
	}
	if b.Timeout > 0 {
		d := time.Now().Add(b.Timeout)
		if t.deadline.IsZero() || d.Before(t.deadline) {
			t.deadline = d
		}
	}
	return t
}

// deadlineTime returns the effective wall-clock deadline (from Timeout/Deadline),
// or the zero time if the budget sets none.
func (t *budgetTracker) deadlineTime() time.Time { return t.deadline }

func (t *budgetTracker) addUsage(u Usage) {
	t.mu.Lock()
	t.used.Add(u)
	t.mu.Unlock()
}

func (t *budgetTracker) incToolCall() {
	t.mu.Lock()
	t.toolCalls++
	t.mu.Unlock()
}

// check returns a *BudgetError if any limit has been reached.
func (t *budgetTracker) check() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.deadline.IsZero() && time.Now().After(t.deadline) {
		return &BudgetError{Limit: "deadline", Detail: "wall-clock deadline exceeded"}
	}
	if t.b.MaxTotalTokens > 0 && t.used.TotalTokens >= t.b.MaxTotalTokens {
		return &BudgetError{Limit: "total_tokens", Detail: "token budget exhausted"}
	}
	if t.b.MaxCostUSD > 0 && t.used.CostUSD >= t.b.MaxCostUSD {
		return &BudgetError{Limit: "cost", Detail: "cost budget exhausted"}
	}
	if t.b.MaxToolCalls > 0 && t.toolCalls >= t.b.MaxToolCalls {
		return &BudgetError{Limit: "tool_calls", Detail: "tool-call budget exhausted"}
	}
	return nil
}

// childBudget derives a subagent budget from the parent's remaining allowance and
// *reserves* the granted tokens/cost, so that concurrent siblings carved in the
// same fan-out see the reduced remainder rather than each being handed the full
// balance. This bounds the aggregate a fan-out of subagents can commit to roughly
// the parent's cap (plus the between-steps slack inherent to rimeno's budgeting),
// instead of cap×N. The returned grant must be passed to settleChild once the
// child finishes, which records the child's actual usage and releases the
// reservation. A zero parent limit stays unlimited. If the parent's token or cost
// budget is already fully committed, the child is handed an already-passed
// deadline so it fails before spending anything.
func (t *budgetTracker) childBudget(requested Budget) (child, grant Budget) {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := requested
	exhausted := false

	if t.b.MaxTotalTokens > 0 {
		rem := t.b.MaxTotalTokens - t.used.TotalTokens - t.reservedTokens
		if rem <= 0 {
			exhausted = true
		} else {
			if out.MaxTotalTokens == 0 || out.MaxTotalTokens > rem {
				out.MaxTotalTokens = rem
			}
			grant.MaxTotalTokens = out.MaxTotalTokens
			t.reservedTokens += grant.MaxTotalTokens
		}
	}
	if t.b.MaxCostUSD > 0 {
		rem := t.b.MaxCostUSD - t.used.CostUSD - t.reservedCost
		if rem <= 0 {
			exhausted = true
		} else {
			if out.MaxCostUSD == 0 || out.MaxCostUSD > rem {
				out.MaxCostUSD = rem
			}
			grant.MaxCostUSD = out.MaxCostUSD
			t.reservedCost += grant.MaxCostUSD
		}
	}
	if !t.deadline.IsZero() {
		if out.Deadline.IsZero() || t.deadline.Before(out.Deadline) {
			out.Deadline = t.deadline
		}
	}
	if exhausted {
		out.Deadline = time.Now().Add(-time.Second) // already passed → child fails before spending
	}
	return out, grant
}

// settleChild records a finished subagent's actual usage against the parent and
// releases the reservation held for it (see childBudget).
func (t *budgetTracker) settleChild(grant Budget, actual Usage) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.used.Add(actual)
	if t.reservedTokens -= grant.MaxTotalTokens; t.reservedTokens < 0 {
		t.reservedTokens = 0
	}
	if t.reservedCost -= grant.MaxCostUSD; t.reservedCost < 0 {
		t.reservedCost = 0
	}
}

// noteToolError records that a tool errored this run (for the circuit breaker).
func (t *budgetTracker) noteToolError(name string) {
	t.mu.Lock()
	t.toolErrors[name]++
	t.mu.Unlock()
}

// toolErrorCount reports how many times a tool has errored this run.
func (t *budgetTracker) toolErrorCount(name string) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.toolErrors[name]
}
