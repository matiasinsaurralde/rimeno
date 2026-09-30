package memory

import (
	"context"
	"encoding/json"

	"github.com/matiasinsaurralde/rimeno"
)

// Tool output caps: a chatty agent must not be able to blow its own context by
// recalling everything. Recall returns at most defaultRecallLimit records, each
// truncated to maxRecallTextLen runes.
const (
	defaultRecallLimit = 10
	maxRecallLimit     = 50
	maxRecallTextLen   = 500
)

// Tools returns the rimeno tools an agent can call to manage its own memory,
// scoped to this handle's namespace:
//
//	remember{kind,text,tags?,salience?,data?} -> {id}
//	recall{kinds?,tags?,text?,limit?}         -> {records:[{id,kind,text,tags,updated_at}]}
//	note{text,tags?}                          -> {id}   (a quick "fact" record)
//	update_todo{id?,text,status}              -> {id}   (a "todo" record)
//	forget{id}                                -> {ok}
//
// Tool results are capped in count and length so a verbose agent cannot flood its
// own context.
func (m *Memory) Tools() []rimeno.Tool {
	return []rimeno.Tool{
		m.rememberTool(),
		m.recallTool(),
		m.noteTool(),
		m.updateTodoTool(),
		m.forgetTool(),
	}
}

type rememberArgs struct {
	Kind     string          `json:"kind" jsonschema:"description=Category of the fact, e.g. fact/decision/constraint"`
	Text     string          `json:"text" jsonschema:"description=The recallable summary to store"`
	Tags     []string        `json:"tags,omitempty" jsonschema:"description=Optional labels for filtered recall"`
	Salience float64         `json:"salience,omitempty" jsonschema:"description=Importance 0..1 for ranked recall (default 0.5)"`
	Data     json.RawMessage `json:"data,omitempty" jsonschema:"description=Optional structured JSON payload"`
}

type idResult struct {
	ID string `json:"id"`
}

func (m *Memory) rememberTool() rimeno.Tool {
	return rimeno.NewTool("remember",
		"Store a durable fact in memory so it survives context compaction and can be recalled later.",
		func(ctx context.Context, in rememberArgs) (idResult, error) {
			kind := in.Kind
			if kind == "" {
				kind = KindFact
			}
			id, err := m.Remember(ctx, Record{
				Kind: kind, Text: in.Text, Tags: in.Tags,
				Salience: in.Salience, Data: in.Data,
			})
			return idResult{ID: id}, err
		})
}

type recallArgs struct {
	Kinds []string `json:"kinds,omitempty" jsonschema:"description=Restrict to these kinds"`
	Tags  []string `json:"tags,omitempty" jsonschema:"description=Restrict to records with all these tags"`
	Text  string   `json:"text,omitempty" jsonschema:"description=Substring to match against record text"`
	Limit int      `json:"limit,omitempty" jsonschema:"description=Max records to return (default 10)"`
}

type recalledRecord struct {
	ID        string   `json:"id"`
	Kind      string   `json:"kind"`
	Text      string   `json:"text"`
	Tags      []string `json:"tags,omitempty"`
	UpdatedAt string   `json:"updated_at"`
}

type recallResult struct {
	Records []recalledRecord `json:"records"`
}

func (m *Memory) recallTool() rimeno.Tool {
	return rimeno.NewTool("recall",
		"Retrieve durable facts from memory, ranked by importance and recency.",
		func(ctx context.Context, in recallArgs) (recallResult, error) {
			limit := in.Limit
			if limit <= 0 {
				limit = defaultRecallLimit
			}
			if limit > maxRecallLimit {
				limit = maxRecallLimit
			}
			recs, err := m.Recall(ctx, Query{
				Kinds: in.Kinds, Tags: in.Tags, Text: in.Text, Limit: limit,
			})
			if err != nil {
				return recallResult{}, err
			}
			out := recallResult{Records: make([]recalledRecord, 0, len(recs))}
			for _, r := range recs {
				out.Records = append(out.Records, recalledRecord{
					ID:        r.ID,
					Kind:      r.Kind,
					Text:      truncate(r.Text, maxRecallTextLen),
					Tags:      r.Tags,
					UpdatedAt: r.UpdatedAt.UTC().Format("2006-01-02T15:04:05Z"),
				})
			}
			return out, nil
		})
}

type noteArgs struct {
	Text string   `json:"text" jsonschema:"description=A quick note or fact to remember"`
	Tags []string `json:"tags,omitempty" jsonschema:"description=Optional labels"`
}

func (m *Memory) noteTool() rimeno.Tool {
	return rimeno.NewTool("note",
		"Jot a quick fact into memory (a low-ceremony remember).",
		func(ctx context.Context, in noteArgs) (idResult, error) {
			id, err := m.Remember(ctx, Record{Kind: KindNote, Text: in.Text, Tags: in.Tags})
			return idResult{ID: id}, err
		})
}

type updateTodoArgs struct {
	ID     string `json:"id,omitempty" jsonschema:"description=Existing todo id to update; omit to create a new one"`
	Text   string `json:"text" jsonschema:"description=The task description"`
	Status string `json:"status,omitempty" jsonschema:"description=open|in_progress|done (default open)"`
}

// todoData is the structured payload stored on a todo record.
type todoData struct {
	Status string `json:"status"`
}

func (m *Memory) updateTodoTool() rimeno.Tool {
	return rimeno.NewTool("update_todo",
		"Create or update a task item in memory's todo list.",
		func(ctx context.Context, in updateTodoArgs) (idResult, error) {
			status := in.Status
			if status == "" {
				status = "open"
			}
			data, _ := json.Marshal(todoData{Status: status})
			id, err := m.Remember(ctx, Record{
				ID: in.ID, Kind: KindTodo, Text: in.Text, Data: data,
			})
			return idResult{ID: id}, err
		})
}

type forgetArgs struct {
	ID string `json:"id" jsonschema:"description=The id of the record to delete"`
}

type okResult struct {
	OK bool `json:"ok"`
}

func (m *Memory) forgetTool() rimeno.Tool {
	return rimeno.NewTool("forget",
		"Delete a record from memory by id.",
		func(ctx context.Context, in forgetArgs) (okResult, error) {
			if err := m.Forget(ctx, in.ID); err != nil {
				return okResult{}, err
			}
			return okResult{OK: true}, nil
		})
}

// truncate shortens s to at most n runes, appending an ellipsis when it cuts.
func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
