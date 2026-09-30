package memory

import (
	"context"
	"fmt"
	"strings"
)

// preambleHeader labels the injected memory block so both the model and the
// Compactor can recognize it.
const preambleHeader = "Recalled memory (durable facts from earlier in this task):"

// Preamble renders the most relevant records matching q into a compact,
// human-readable context block, bounded by an approximate token budget.
//
// Records are taken in the store's ranked order (salience × recency, nudged by
// substring relevance to q.Text) and appended until the next line would exceed
// tokenBudget. The budget is approximate: it uses rimeno's ~4-chars-per-token
// heuristic. An empty result yields an empty string, so callers can inject it
// unconditionally.
//
// This is what a caller composes into a session's opening turn, or what
// [Compactor] re-injects after compaction, so durable facts stay in view.
func (m *Memory) Preamble(ctx context.Context, q Query, tokenBudget int) (string, error) {
	recs, err := m.Recall(ctx, q)
	if err != nil {
		return "", err
	}
	if len(recs) == 0 {
		return "", nil
	}

	var b strings.Builder
	b.WriteString(preambleHeader)
	used := approxTokens(b.String())

	written := 0
	for _, r := range recs {
		line := "\n" + formatRecord(r)
		cost := approxTokens(line)
		if tokenBudget > 0 && used+cost > tokenBudget && written > 0 {
			break
		}
		b.WriteString(line)
		used += cost
		written++
	}
	if written == 0 {
		return "", nil
	}
	return b.String(), nil
}

// formatRecord renders one record as a single bullet line for the preamble.
func formatRecord(r Record) string {
	var sb strings.Builder
	sb.WriteString("- [")
	sb.WriteString(r.Kind)
	sb.WriteString("] ")
	sb.WriteString(strings.TrimSpace(r.Text))
	if len(r.Tags) > 0 {
		fmt.Fprintf(&sb, " (%s)", strings.Join(r.Tags, ", "))
	}
	return sb.String()
}

// approxTokens mirrors rimeno's built-in ~4-chars-per-token heuristic so the
// preamble budget lines up with the harness's own token estimate without taking
// a dependency on an exact tokenizer.
func approxTokens(s string) int {
	return len(s)/4 + 1
}
