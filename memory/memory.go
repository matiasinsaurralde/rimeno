// Package memory is a small, generic, opt-in memory layer for rimeno agents.
//
// An agent's conversation history is transient: it grows until it is compacted
// away, and nothing survives the summary as a durable, addressable fact. memory
// adds a structured store of [Record]s that outlives any single turn, is
// shareable across concurrent agents, and — via [Compactor] — is re-injected
// into context after compaction so ground truth is never lost.
//
// The design is deliberately layered:
//
//   - [Store] is the persistence seam: an upsert-by-(namespace,id) key-value
//     store with tag/kind/substring [Query]. Two backends ship in this package,
//     both std-only: [NewInMemory] (thread-safe, the default) and [NewFileStore]
//     (JSON-per-namespace, durable across restarts).
//
//   - [Memory] is a scoped handle over a Store bound to one namespace (a task id,
//     a session id, a "repo@commit" — whatever the caller uses to partition
//     memory). It offers [Memory.Remember]/[Memory.Recall], a budgeted
//     [Memory.Preamble] for injecting salient facts as a context block, and
//     [Memory.Tools] — rimeno tools the model can call to manage its own memory.
//
//   - [Compactor] wraps any rimeno.Compactor so that recalled memory is
//     re-injected on every compaction.
//
// Quick start:
//
//	store := memory.NewInMemory()
//	mem := memory.New(store, "task-42")
//	agent, _ := rimeno.New(rimeno.Config{
//	    Model:        model,
//	    Instructions: "You are a helpful assistant. Use remember/recall to keep notes.",
//	    Tools:        mem.Tools(),
//	    Compactor:    memory.Compactor(mem, inner),
//	})
//
// The package depends only on the standard library and rimeno core; heavier
// backends (SQLite, vector/semantic recall) are intended as further opt-in
// subpackages and are not part of this core.
package memory

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"time"
)

// Common record kinds. Kind is a free-form string — these are the conventions
// the built-in tools use, but callers may define their own domain kinds.
const (
	// KindFact is a general durable fact worth recalling later.
	KindFact = "fact"
	// KindTodo is a task item; its status lives in the Data payload.
	KindTodo = "todo"
	// KindNote is a quick, low-ceremony note.
	KindNote = "note"
)

// Record is a single unit of memory. It is upserted by (Namespace, ID): writing
// a record whose (Namespace, ID) already exists replaces it (preserving the
// original CreatedAt).
type Record struct {
	// ID is a stable identifier within the namespace. If empty at write time, a
	// content hash of the record is used, so identical content collapses to one
	// record (useful when several agents record the same fact).
	ID string `json:"id"`
	// Namespace scopes the record: a task id, session id, "repo@commit", etc.
	// It is set from the owning [Memory] handle on writes.
	Namespace string `json:"namespace"`
	// Kind categorizes the record ("fact", "todo", "note", or a caller kind).
	Kind string `json:"kind"`
	// Tags are free-form labels used for filtered recall.
	Tags []string `json:"tags,omitempty"`
	// Text is the human-readable summary — this is what [Memory.Preamble] injects
	// into context and what recall returns.
	Text string `json:"text"`
	// Data is an optional structured payload, typed by the caller.
	Data json.RawMessage `json:"data,omitempty"`
	// Salience is a 0..1 importance weight used to rank recall (default 0.5 when
	// zero). Higher is more important.
	Salience float64 `json:"salience,omitempty"`
	// Source is provenance: the agent or tool that wrote the record.
	Source string `json:"source,omitempty"`
	// CreatedAt is set on first write; UpdatedAt on every write.
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// effectiveSalience returns the record's salience, defaulting to 0.5 when unset
// so that unranked records sort sensibly between explicit low and high values.
func (r Record) effectiveSalience() float64 {
	if r.Salience <= 0 {
		return 0.5
	}
	if r.Salience > 1 {
		return 1
	}
	return r.Salience
}

// contentID derives a stable ID from the fields that define a record's identity,
// so two agents writing the same fact into the same namespace collapse to one
// record instead of duplicating.
func (r Record) contentID() string {
	h := sha256.New()
	h.Write([]byte(r.Kind))
	h.Write([]byte{0})
	h.Write([]byte(r.Text))
	for _, t := range r.Tags {
		h.Write([]byte{0})
		h.Write([]byte(t))
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// Query selects records from a [Store]. Empty fields do not constrain: a
// zero-value Query matches every record in the namespace.
type Query struct {
	// Namespace scopes the query; set from the [Memory] handle.
	Namespace string
	// Kinds, if non-empty, restricts to records whose Kind is in this set.
	Kinds []string
	// Tags, if non-empty, restricts to records carrying all of these tags.
	Tags []string
	// Text, if non-empty, restricts to records whose Text contains this
	// substring (case-insensitive). A semantic backend may treat it as a
	// similarity query instead.
	Text string
	// Limit caps the number of results (0 = no limit). Results are returned
	// ranked by salience then recency.
	Limit int
}
