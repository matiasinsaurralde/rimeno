package memory

import "context"

// Memory is a scoped handle over a [Store], bound to a single namespace. It is
// the primary object callers hold: create one per scope (task, session,
// "repo@commit") and share it with the agents that should read and write that
// scope's memory.
//
// A Memory is safe for concurrent use as long as its Store is (both built-in
// stores are).
type Memory struct {
	store     Store
	namespace string
	source    string
}

// Option configures a [Memory].
type Option func(*Memory)

// WithSource sets the provenance string stamped on records written through this
// handle (and on records created by its tools) when the caller does not supply
// one. Useful to tag which agent wrote a fact.
func WithSource(source string) Option {
	return func(m *Memory) { m.source = source }
}

// New returns a [Memory] over store scoped to namespace.
func New(store Store, namespace string, opts ...Option) *Memory {
	m := &Memory{store: store, namespace: namespace}
	for _, o := range opts {
		o(m)
	}
	return m
}

// Namespace returns the scope this handle is bound to.
func (m *Memory) Namespace() string { return m.namespace }

// Remember upserts a record into this handle's namespace. The record's Namespace
// is set from the handle; an empty Source defaults to the handle's source. It
// returns the stored record's ID (a content hash when the caller left ID empty).
func (m *Memory) Remember(ctx context.Context, r Record) (string, error) {
	r.Namespace = m.namespace
	if r.Source == "" {
		r.Source = m.source
	}
	if r.ID == "" {
		r.ID = r.contentID()
	}
	if err := m.store.Put(ctx, r); err != nil {
		return "", err
	}
	return r.ID, nil
}

// Recall returns records in this namespace matching q, ranked most-relevant
// first. The query's Namespace is set from the handle.
func (m *Memory) Recall(ctx context.Context, q Query) ([]Record, error) {
	q.Namespace = m.namespace
	return m.store.Query(ctx, q)
}

// Get returns a single record by ID from this namespace.
func (m *Memory) Get(ctx context.Context, id string) (Record, bool, error) {
	return m.store.Get(ctx, m.namespace, id)
}

// Forget deletes a record by ID from this namespace.
func (m *Memory) Forget(ctx context.Context, id string) error {
	return m.store.Delete(ctx, m.namespace, id)
}
