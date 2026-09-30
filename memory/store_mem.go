package memory

import (
	"context"
	"sync"
	"time"
)

// memStore is an in-memory, mutex-guarded [Store]. It is the default backend and
// is safe for concurrent use by multiple agents sharing one namespace.
type memStore struct {
	mu   sync.RWMutex
	recs map[string]map[string]Record // namespace -> id -> record
	now  func() time.Time             // injectable clock for tests
}

// NewInMemory returns a thread-safe in-memory [Store]. It is the default backend
// and the natural choice for a shared, within-run blackboard: the orchestrator
// and its child agents can read and write it concurrently.
func NewInMemory() Store {
	return &memStore{recs: make(map[string]map[string]Record), now: time.Now}
}

func (s *memStore) Put(_ context.Context, r Record) error {
	r = normalize(r, s.now())
	s.mu.Lock()
	defer s.mu.Unlock()
	ns := s.recs[r.Namespace]
	if ns == nil {
		ns = make(map[string]Record)
		s.recs[r.Namespace] = ns
	}
	if prev, ok := ns[r.ID]; ok {
		r.CreatedAt = prev.CreatedAt // preserve first-write time on upsert
	}
	ns[r.ID] = r
	return nil
}

func (s *memStore) Get(_ context.Context, ns, id string) (Record, bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	r, ok := s.recs[ns][id]
	return r, ok, nil
}

func (s *memStore) Query(_ context.Context, q Query) ([]Record, error) {
	s.mu.RLock()
	var out []Record
	for _, r := range s.recs[q.Namespace] {
		if matches(r, q) {
			out = append(out, r)
		}
	}
	s.mu.RUnlock()
	rank(out, q)
	return applyLimit(out, q), nil
}

func (s *memStore) Delete(_ context.Context, ns, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.recs[ns], id)
	return nil
}

// normalize fills in the derived fields shared by all backends: content-hash ID
// when unset, default salience, and timestamps. Callers pass the current time so
// a single Put uses one consistent instant.
func normalize(r Record, now time.Time) Record {
	if r.ID == "" {
		r.ID = r.contentID()
	}
	if r.Salience == 0 {
		r.Salience = 0.5
	}
	if r.CreatedAt.IsZero() {
		r.CreatedAt = now
	}
	r.UpdatedAt = now
	return r
}
