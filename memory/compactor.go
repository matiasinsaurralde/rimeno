package memory

import (
	"context"

	"github.com/matiasinsaurralde/rimeno"
)

// Default token budget for the recalled-memory block injected on compaction.
const defaultPreambleBudget = 1024

// compactor wraps an inner rimeno.Compactor so that durable memory survives
// compaction. On Compact it runs the inner compactor as usual, then inserts a
// recalled-memory system message (the [Memory.Preamble]) immediately after the
// leading system messages — so the facts an agent stored via the memory tools
// are re-injected even though the turns that produced them were summarized away.
type compactor struct {
	mem    *Memory
	inner  rimeno.Compactor
	query  Query
	budget int
}

// CompactOpt configures a memory [Compactor].
type CompactOpt func(*compactor)

// WithRecallQuery sets the query used to select which records are re-injected on
// compaction. By default all records in the namespace are eligible, ranked by
// salience and recency and trimmed to the token budget.
func WithRecallQuery(q Query) CompactOpt {
	return func(c *compactor) { c.query = q }
}

// WithPreambleBudget sets the approximate token budget for the re-injected memory
// block (default 1024).
func WithPreambleBudget(tokens int) CompactOpt {
	return func(c *compactor) { c.budget = tokens }
}

// Compactor returns a rimeno.Compactor that delegates the compaction decision and
// summarization to inner, then re-injects recalled memory so durable facts are
// not lost. The agent is expected to write facts explicitly via the memory tools
// (see [Memory.Tools]); this wrapper guarantees those facts come back after a
// summary drops the turns that created them.
//
// If inner is nil, ShouldCompact always reports false and Compact is a no-op —
// the wrapper only re-injects memory that an inner compactor's summary would
// otherwise have shadowed.
func Compactor(m *Memory, inner rimeno.Compactor, opts ...CompactOpt) rimeno.Compactor {
	c := &compactor{mem: m, inner: inner, budget: defaultPreambleBudget}
	for _, o := range opts {
		o(c)
	}
	return c
}

// ShouldCompact delegates to the inner compactor (false when there is none).
func (c *compactor) ShouldCompact(msgs []rimeno.Message, estimatedTokens int) bool {
	if c.inner == nil {
		return false
	}
	return c.inner.ShouldCompact(msgs, estimatedTokens)
}

// Compact runs the inner compactor, then splices a recalled-memory system message
// in after the leading system messages.
func (c *compactor) Compact(ctx context.Context, msgs []rimeno.Message) ([]rimeno.Message, error) {
	// Count the agent's own leading system instructions in the input. Inner
	// compactors preserve these verbatim at the front, so this index is where the
	// recalled memory belongs — after the instructions but before any summary the
	// inner appends (which is itself a system message we must not sit behind).
	head := leadingSystem(msgs)

	out := msgs
	if c.inner != nil {
		var err error
		out, err = c.inner.Compact(ctx, msgs)
		if err != nil {
			return out, err
		}
	}

	preamble, err := c.mem.Preamble(ctx, c.query, c.budget)
	if err != nil {
		return out, err
	}
	if preamble == "" {
		return out, nil
	}

	if head > len(out) {
		head = leadingSystem(out)
	}
	recalled := rimeno.SystemMessage(preamble)
	result := make([]rimeno.Message, 0, len(out)+1)
	result = append(result, out[:head]...)
	result = append(result, recalled)
	result = append(result, out[head:]...)
	return result, nil
}

// leadingSystem counts the contiguous run of system messages at the front of
// msgs (the agent's instructions).
func leadingSystem(msgs []rimeno.Message) int {
	n := 0
	for n < len(msgs) && msgs[n].Role == rimeno.RoleSystem {
		n++
	}
	return n
}
