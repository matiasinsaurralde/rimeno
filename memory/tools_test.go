package memory

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// TestTools_RoundTripViaAgent drives the memory tools through a real rimeno agent
// with a scripted model: the model calls remember, then recall, and we assert the
// stored fact comes back.
func TestTools_RoundTripViaAgent(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "task-1")

	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{
			rimenotest.ToolCall("c1", "remember", rememberArgs{
				Kind: "decision", Text: "use JSON file store", Tags: []string{"storage"}, Salience: 0.9,
			}),
		}},
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{
			rimenotest.ToolCall("c2", "recall", recallArgs{Tags: []string{"storage"}}),
		}},
		rimenotest.Turn{Text: "done"},
	)

	agent, err := rimeno.New(rimeno.Config{Model: model, Tools: mem.Tools()})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	res, err := agent.Run(ctx, "remember and recall a decision")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Text != "done" {
		t.Fatalf("final text = %q, want done", res.Text)
	}

	// The fact is actually in the store.
	recs, err := mem.Recall(ctx, Query{Tags: []string{"storage"}})
	if err != nil {
		t.Fatal(err)
	}
	if len(recs) != 1 || recs[0].Text != "use JSON file store" {
		t.Fatalf("store contents = %+v, want the stored decision", recs)
	}

	// The recall tool's result (the 3rd request's tool message) contains the fact.
	lastReq := model.Requests[len(model.Requests)-1]
	if !transcriptContains(lastReq.Messages, "use JSON file store") {
		t.Fatalf("recall tool output not fed back to model:\n%s", renderMessages(lastReq.Messages))
	}
}

func TestTools_ForgetAndTodo(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "ns")

	todo := mem.updateTodoTool()
	out, err := todo.Invoke(ctx, mustJSON(updateTodoArgs{Text: "write tests", Status: "in_progress"}))
	if err != nil {
		t.Fatalf("update_todo: %v", err)
	}
	id := out.(idResult).ID

	got, ok, _ := mem.Get(ctx, id)
	if !ok || got.Kind != KindTodo {
		t.Fatalf("todo not stored: ok=%v kind=%q", ok, got.Kind)
	}
	var d todoData
	_ = json.Unmarshal(got.Data, &d)
	if d.Status != "in_progress" {
		t.Fatalf("todo status = %q, want in_progress", d.Status)
	}

	// Updating the same todo id transitions status without creating a duplicate.
	if _, err := todo.Invoke(ctx, mustJSON(updateTodoArgs{ID: id, Text: "write tests", Status: "done"})); err != nil {
		t.Fatalf("update_todo done: %v", err)
	}
	all, _ := mem.Recall(ctx, Query{Kinds: []string{KindTodo}})
	if len(all) != 1 {
		t.Fatalf("got %d todos, want 1 (update, not insert)", len(all))
	}

	// Forget removes it.
	forget := mem.forgetTool()
	if _, err := forget.Invoke(ctx, mustJSON(forgetArgs{ID: id})); err != nil {
		t.Fatalf("forget: %v", err)
	}
	if _, ok, _ := mem.Get(ctx, id); ok {
		t.Fatal("todo still present after forget")
	}
}

func TestTools_RecallCapsResults(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "ns")
	for i := 0; i < maxRecallLimit+20; i++ {
		if _, err := mem.Remember(ctx, Record{Kind: "fact", Text: strings.Repeat("y", 5) + itoa(i)}); err != nil {
			t.Fatal(err)
		}
	}
	// Ask for more than the hard cap; result must be clamped.
	out, err := mem.recallTool().Invoke(ctx, mustJSON(recallArgs{Limit: 1000}))
	if err != nil {
		t.Fatal(err)
	}
	got := out.(recallResult)
	if len(got.Records) > maxRecallLimit {
		t.Fatalf("recall returned %d records, cap is %d", len(got.Records), maxRecallLimit)
	}
}

func TestTools_RecallTruncatesLongText(t *testing.T) {
	ctx := context.Background()
	mem := New(NewInMemory(), "ns")
	long := strings.Repeat("z", maxRecallTextLen+100)
	if _, err := mem.Remember(ctx, Record{Kind: "fact", Text: long}); err != nil {
		t.Fatal(err)
	}
	out, err := mem.recallTool().Invoke(ctx, mustJSON(recallArgs{}))
	if err != nil {
		t.Fatal(err)
	}
	got := out.(recallResult)
	if len(got.Records) != 1 {
		t.Fatalf("got %d records, want 1", len(got.Records))
	}
	if runeLen(got.Records[0].Text) > maxRecallTextLen+1 { // +1 for the ellipsis
		t.Fatalf("text length %d exceeds cap %d", runeLen(got.Records[0].Text), maxRecallTextLen)
	}
}

func TestTools_SchemasAreValid(t *testing.T) {
	mem := New(NewInMemory(), "ns")
	for _, tool := range mem.Tools() {
		if tool.Name() == "" {
			t.Fatal("tool with empty name")
		}
		var schema map[string]any
		if err := json.Unmarshal(tool.ParametersSchema(), &schema); err != nil {
			t.Fatalf("tool %q has invalid schema: %v", tool.Name(), err)
		}
		if schema["type"] != "object" {
			t.Fatalf("tool %q schema type = %v, want object", tool.Name(), schema["type"])
		}
	}
	names := map[string]bool{}
	for _, tool := range mem.Tools() {
		names[tool.Name()] = true
	}
	for _, want := range []string{"remember", "recall", "note", "update_todo", "forget"} {
		if !names[want] {
			t.Fatalf("missing tool %q", want)
		}
	}
}

// --- helpers ---

func mustJSON(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func transcriptContains(msgs []rimeno.Message, sub string) bool {
	return strings.Contains(renderMessages(msgs), sub)
}

func renderMessages(msgs []rimeno.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(string(m.Role))
		b.WriteString(": ")
		b.WriteString(m.Text)
		b.WriteByte('\n')
	}
	return b.String()
}

func runeLen(s string) int { return len([]rune(s)) }

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf []byte
	neg := i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		buf = append([]byte{byte('0' + i%10)}, buf...)
		i /= 10
	}
	if neg {
		buf = append([]byte{'-'}, buf...)
	}
	return string(buf)
}
