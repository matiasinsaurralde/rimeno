package memory

import (
	"context"
	"encoding/json"
	"testing"
)

// storeFactory builds a fresh store for the shared conformance suite.
type storeFactory struct {
	name string
	make func(t *testing.T) Store
}

func stores(t *testing.T) []storeFactory {
	return []storeFactory{
		{"inmemory", func(t *testing.T) Store { return NewInMemory() }},
		{"file", func(t *testing.T) Store {
			s, err := NewFileStore(t.TempDir())
			if err != nil {
				t.Fatalf("NewFileStore: %v", err)
			}
			return s
		}},
	}
}

func TestStore_PutGet(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			if err := s.Put(ctx, Record{ID: "a", Namespace: "ns", Kind: "fact", Text: "hello"}); err != nil {
				t.Fatalf("Put: %v", err)
			}
			got, ok, err := s.Get(ctx, "ns", "a")
			if err != nil || !ok {
				t.Fatalf("Get: ok=%v err=%v", ok, err)
			}
			if got.Text != "hello" {
				t.Fatalf("Text = %q, want hello", got.Text)
			}
			if got.CreatedAt.IsZero() || got.UpdatedAt.IsZero() {
				t.Fatalf("timestamps not set: %+v", got)
			}
			if got.Salience != 0.5 {
				t.Fatalf("default salience = %v, want 0.5", got.Salience)
			}
		})
	}
}

func TestStore_GetMissing(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			_, ok, err := s.Get(ctx, "ns", "nope")
			if err != nil {
				t.Fatalf("Get err: %v", err)
			}
			if ok {
				t.Fatal("expected ok=false for missing record")
			}
		})
	}
}

func TestStore_UpsertPreservesCreatedAt(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			if err := s.Put(ctx, Record{ID: "a", Namespace: "ns", Text: "v1"}); err != nil {
				t.Fatalf("Put v1: %v", err)
			}
			first, _, _ := s.Get(ctx, "ns", "a")
			if err := s.Put(ctx, Record{ID: "a", Namespace: "ns", Text: "v2"}); err != nil {
				t.Fatalf("Put v2: %v", err)
			}
			second, _, _ := s.Get(ctx, "ns", "a")
			if second.Text != "v2" {
				t.Fatalf("Text = %q, want v2 (upsert should replace)", second.Text)
			}
			if !second.CreatedAt.Equal(first.CreatedAt) {
				t.Fatalf("CreatedAt changed on upsert: %v -> %v", first.CreatedAt, second.CreatedAt)
			}
			if !second.UpdatedAt.After(first.UpdatedAt) && !second.UpdatedAt.Equal(first.UpdatedAt) {
				t.Fatalf("UpdatedAt went backwards")
			}
		})
	}
}

func TestStore_ContentHashDedupe(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			// Two writes with no ID but identical content collapse to one record.
			r := Record{Namespace: "ns", Kind: "fact", Text: "same", Tags: []string{"x"}}
			r1 := normalizeForTest(r)
			r2 := normalizeForTest(r)
			if err := s.Put(ctx, r1); err != nil {
				t.Fatal(err)
			}
			if err := s.Put(ctx, r2); err != nil {
				t.Fatal(err)
			}
			all, err := s.Query(ctx, Query{Namespace: "ns"})
			if err != nil {
				t.Fatal(err)
			}
			if len(all) != 1 {
				t.Fatalf("got %d records, want 1 (dedupe by content hash)", len(all))
			}
		})
	}
}

// normalizeForTest fills the content-hash ID the way Memory.Remember does, since
// the raw Store does that inside Put but returns nothing.
func normalizeForTest(r Record) Record {
	if r.ID == "" {
		r.ID = r.contentID()
	}
	return r
}

func TestStore_QueryFilters(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			seed := []Record{
				{ID: "1", Namespace: "ns", Kind: "fact", Text: "loader parses input", Tags: []string{"loader"}},
				{ID: "2", Namespace: "ns", Kind: "todo", Text: "check the parser", Tags: []string{"loader", "parser"}},
				{ID: "3", Namespace: "ns", Kind: "fact", Text: "unrelated", Tags: []string{"misc"}},
				{ID: "4", Namespace: "other", Kind: "fact", Text: "loader in other ns", Tags: []string{"loader"}},
			}
			for _, r := range seed {
				if err := s.Put(ctx, r); err != nil {
					t.Fatal(err)
				}
			}

			// Namespace isolation.
			got, _ := s.Query(ctx, Query{Namespace: "ns"})
			if len(got) != 3 {
				t.Fatalf("namespace ns: got %d, want 3", len(got))
			}

			// Kind filter.
			got, _ = s.Query(ctx, Query{Namespace: "ns", Kinds: []string{"fact"}})
			if len(got) != 2 {
				t.Fatalf("kind fact: got %d, want 2", len(got))
			}

			// Tag filter requires ALL tags.
			got, _ = s.Query(ctx, Query{Namespace: "ns", Tags: []string{"loader", "parser"}})
			if len(got) != 1 || got[0].ID != "2" {
				t.Fatalf("tag AND: got %+v, want only id 2", got)
			}

			// Text substring, case-insensitive.
			got, _ = s.Query(ctx, Query{Namespace: "ns", Text: "LOADER"})
			if len(got) != 1 || got[0].ID != "1" {
				t.Fatalf("text match: got %+v, want id 1", got)
			}

			// Limit.
			got, _ = s.Query(ctx, Query{Namespace: "ns", Limit: 2})
			if len(got) != 2 {
				t.Fatalf("limit: got %d, want 2", len(got))
			}
		})
	}
}

func TestStore_QueryRanking(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			for _, r := range []Record{
				{ID: "low", Namespace: "ns", Text: "low", Salience: 0.2},
				{ID: "high", Namespace: "ns", Text: "high", Salience: 0.9},
				{ID: "mid", Namespace: "ns", Text: "mid", Salience: 0.5},
			} {
				if err := s.Put(ctx, r); err != nil {
					t.Fatal(err)
				}
			}
			got, _ := s.Query(ctx, Query{Namespace: "ns"})
			want := []string{"high", "mid", "low"}
			for i, id := range want {
				if got[i].ID != id {
					t.Fatalf("rank[%d] = %q, want %q (order: %v)", i, got[i].ID, id, ids(got))
				}
			}
		})
	}
}

func TestStore_Delete(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			_ = s.Put(ctx, Record{ID: "a", Namespace: "ns", Text: "x"})
			if err := s.Delete(ctx, "ns", "a"); err != nil {
				t.Fatalf("Delete: %v", err)
			}
			_, ok, _ := s.Get(ctx, "ns", "a")
			if ok {
				t.Fatal("record still present after delete")
			}
			// Deleting a missing record is not an error.
			if err := s.Delete(ctx, "ns", "gone"); err != nil {
				t.Fatalf("Delete missing: %v", err)
			}
		})
	}
}

func TestStore_DataPayloadRoundTrips(t *testing.T) {
	ctx := context.Background()
	for _, sf := range stores(t) {
		t.Run(sf.name, func(t *testing.T) {
			s := sf.make(t)
			payload := json.RawMessage(`{"severity":"high","line":42}`)
			_ = s.Put(ctx, Record{ID: "a", Namespace: "ns", Text: "finding", Data: payload})
			got, _, _ := s.Get(ctx, "ns", "a")
			var m map[string]any
			if err := json.Unmarshal(got.Data, &m); err != nil {
				t.Fatalf("payload unmarshal: %v", err)
			}
			if m["severity"] != "high" {
				t.Fatalf("payload = %v, want severity high", m)
			}
		})
	}
}

func ids(recs []Record) []string {
	out := make([]string, len(recs))
	for i, r := range recs {
		out[i] = r.ID
	}
	return out
}
