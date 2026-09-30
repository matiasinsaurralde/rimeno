package memory

import (
	"context"
	"sort"
	"strings"
)

// Store is the persistence seam for memory. Implementations must be safe for
// concurrent use: an in-process store is shared by an orchestrator and its child
// agents, which may read and write it from multiple goroutines.
type Store interface {
	// Put upserts a record by (Namespace, ID). If a record with the same key
	// exists it is replaced; implementations preserve the original CreatedAt.
	Put(ctx context.Context, r Record) error
	// Get returns the record for (ns, id). The bool is false (with a nil error)
	// when no such record exists.
	Get(ctx context.Context, ns, id string) (Record, bool, error)
	// Query returns records matching q, ranked most-relevant first.
	Query(ctx context.Context, q Query) ([]Record, error)
	// Delete removes (ns, id). Deleting a missing record is not an error.
	Delete(ctx context.Context, ns, id string) error
}

// matches reports whether r satisfies the constraints of q. Namespace is assumed
// already filtered by the caller.
func matches(r Record, q Query) bool {
	if len(q.Kinds) > 0 && !containsStr(q.Kinds, r.Kind) {
		return false
	}
	for _, want := range q.Tags {
		if !containsStr(r.Tags, want) {
			return false
		}
	}
	if q.Text != "" && !strings.Contains(strings.ToLower(r.Text), strings.ToLower(q.Text)) {
		return false
	}
	return true
}

func containsStr(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// rank orders records by descending salience, then most-recent UpdatedAt, then
// ID for a stable tiebreak. The optional query text nudges records whose Text
// contains it ahead of those that merely pass the (already-applied) filter, so
// substring relevance participates in ranking, not just filtering.
func rank(recs []Record, q Query) {
	textLower := strings.ToLower(q.Text)
	score := func(r Record) float64 {
		s := r.effectiveSalience()
		if textLower != "" && strings.Contains(strings.ToLower(r.Text), textLower) {
			s += 1 // relevance boost above the 0..1 salience band
		}
		return s
	}
	sort.SliceStable(recs, func(i, j int) bool {
		si, sj := score(recs[i]), score(recs[j])
		if si != sj {
			return si > sj
		}
		if !recs[i].UpdatedAt.Equal(recs[j].UpdatedAt) {
			return recs[i].UpdatedAt.After(recs[j].UpdatedAt)
		}
		return recs[i].ID < recs[j].ID
	})
}

// applyLimit truncates recs to q.Limit in place-safe fashion.
func applyLimit(recs []Record, q Query) []Record {
	if q.Limit > 0 && len(recs) > q.Limit {
		return recs[:q.Limit]
	}
	return recs
}
