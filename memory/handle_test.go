package memory

import (
	"context"
	"strings"
	"testing"
)

func TestMemory_RememberScopesNamespaceAndSource(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "task-1", WithSource("recon"))
	id, err := mem.Remember(ctx, Record{Kind: "fact", Text: "x"})
	if err != nil {
		t.Fatalf("Remember: %v", err)
	}
	got, ok, err := mem.Get(ctx, id)
	if err != nil || !ok {
		t.Fatalf("Get: ok=%v err=%v", ok, err)
	}
	if got.Namespace != "task-1" {
		t.Fatalf("Namespace = %q, want task-1", got.Namespace)
	}
	if got.Source != "recon" {
		t.Fatalf("Source = %q, want recon (from handle)", got.Source)
	}
}

func TestMemory_RecallIsNamespaceScoped(t *testing.T) {
	ctx := context.Background()
	store := NewInMemory()
	a := New(store, "task-a")
	b := New(store, "task-b")
	if _, err := a.Remember(ctx, Record{Text: "in a"}); err != nil {
		t.Fatal(err)
	}
	if _, err := b.Remember(ctx, Record{Text: "in b"}); err != nil {
		t.Fatal(err)
	}
	got, err := a.Recall(ctx, Query{})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Text != "in a" {
		t.Fatalf("recall from task-a = %+v, want only 'in a'", got)
	}
}

func TestMemory_Forget(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "ns")
	id, _ := mem.Remember(ctx, Record{Text: "temp"})
	if err := mem.Forget(ctx, id); err != nil {
		t.Fatalf("Forget: %v", err)
	}
	_, ok, _ := mem.Get(ctx, id)
	if ok {
		t.Fatal("record present after Forget")
	}
}

func TestPreamble_Budgeting(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "ns")
	for i, s := range []float64{0.9, 0.8, 0.7, 0.6, 0.5} {
		text := strings.Repeat("x", 40) + itoa(i) // ~10 tokens each, unique so no dedupe
		if _, err := mem.Remember(ctx, Record{Kind: "fact", Text: text, Salience: s}); err != nil {
			t.Fatal(err)
		}
	}
	// Generous budget: all five.
	full, err := mem.Preamble(ctx, Query{}, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Count(full, "\n- "); got != 5 {
		t.Fatalf("full preamble has %d records, want 5", got)
	}

	// Tight budget: fewer records, and it must include at least one.
	tight, err := mem.Preamble(ctx, Query{}, 20)
	if err != nil {
		t.Fatal(err)
	}
	n := strings.Count(tight, "\n- ")
	if n == 0 {
		t.Fatal("tight preamble dropped everything; must keep at least one")
	}
	if n >= 5 {
		t.Fatalf("tight budget kept %d records, expected fewer than 5", n)
	}
}

func TestPreamble_EmptyWhenNoRecords(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "ns")
	got, err := mem.Preamble(ctx, Query{}, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if got != "" {
		t.Fatalf("preamble = %q, want empty", got)
	}
}

func TestPreamble_HighestSalienceFirst(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "ns")
	_, _ = mem.Remember(ctx, Record{Kind: "fact", Text: "less important", Salience: 0.2})
	_, _ = mem.Remember(ctx, Record{Kind: "fact", Text: "most important", Salience: 0.95})
	got, err := mem.Preamble(ctx, Query{}, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(got, preambleHeader) {
		t.Fatalf("preamble missing header: %q", got)
	}
	if strings.Index(got, "most important") > strings.Index(got, "less important") {
		t.Fatalf("preamble not ranked by salience:\n%s", got)
	}
}
