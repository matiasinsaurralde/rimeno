package rimeno

import (
	"encoding/json"
	"io"
	"strings"
	"sync"
	"time"
)

// SpanKind categorizes a [Span] in a [Trace].
type SpanKind string

const (
	SpanRun        SpanKind = "run"        // a whole agent run (root, or a subagent)
	SpanStep       SpanKind = "step"       // one loop iteration
	SpanModel      SpanKind = "model"      // a single model call
	SpanTool       SpanKind = "tool"       // a single tool invocation
	SpanCompaction SpanKind = "compaction" // a context compaction
)

// Span is one node in a [Trace] tree.
type Span struct {
	ID         string         `json:"id"`
	Kind       SpanKind       `json:"kind"`
	Name       string         `json:"name"`
	Start      time.Time      `json:"start"`
	End        time.Time      `json:"end,omitempty"`
	Usage      Usage          `json:"usage,omitempty"`
	Model      string         `json:"model,omitempty"`
	Tool       string         `json:"tool,omitempty"`
	StopReason StopReason     `json:"stop_reason,omitempty"`
	Error      string         `json:"error,omitempty"`
	Attrs      map[string]any `json:"attrs,omitempty"`
	Children   []*Span        `json:"children,omitempty"`
}

// Duration reports the span's wall-clock time (0 if not yet ended).
func (s *Span) Duration() time.Duration {
	if s.End.IsZero() {
		return 0
	}
	return s.End.Sub(s.Start)
}

func (s *Span) end() {
	if s.End.IsZero() {
		s.End = time.Now()
	}
}

// Trace is the structured, exportable record of a run: a tree of [Span]s built
// as the run executes. See [Trace.Summary], [Trace.JSON], and [Trace.JSONL].
type Trace struct {
	Root *Span `json:"root"`
	mu   sync.Mutex
}

func newTrace(name string) *Trace {
	return &Trace{Root: &Span{ID: newID(), Kind: SpanRun, Name: name, Start: time.Now()}}
}

// child creates and appends a child span under parent. Safe for concurrent use
// (parallel tools/subagents).
func (t *Trace) child(parent *Span, kind SpanKind, name string) *Span {
	s := &Span{ID: newID(), Kind: kind, Name: name, Start: time.Now()}
	t.mu.Lock()
	parent.Children = append(parent.Children, s)
	t.mu.Unlock()
	return s
}

// graft attaches an already-built subtree (e.g. a subagent's trace root) under
// parent. Used to nest subagent traces into the caller's trace.
func (t *Trace) graft(parent, sub *Span) {
	if sub == nil {
		return
	}
	t.mu.Lock()
	parent.Children = append(parent.Children, sub)
	t.mu.Unlock()
}

// Summary aggregates the trace into headline numbers: cost, tokens,
// model/tool call counts, wall-clock, and per-tool stats.
type Summary struct {
	Wallclock     time.Duration       `json:"wallclock"`
	Steps         int                 `json:"steps"`
	ModelCalls    int                 `json:"model_calls"`
	ToolCalls     int                 `json:"tool_calls"`
	Compactions   int                 `json:"compactions"`
	Subagents     int                 `json:"subagents"`
	Usage         Usage               `json:"usage"`
	ToolBreakdown map[string]ToolStat `json:"tool_breakdown,omitempty"`
}

// ToolStat is the per-tool rollup in a [Summary].
type ToolStat struct {
	Calls         int           `json:"calls"`
	Errors        int           `json:"errors"`
	TotalDuration time.Duration `json:"total_duration"`
}

// Summary walks the whole trace tree (including nested subagents) and aggregates
// it. Usage is summed only from model spans to avoid double counting.
func (t *Trace) Summary() Summary {
	t.mu.Lock()
	defer t.mu.Unlock()
	s := Summary{ToolBreakdown: map[string]ToolStat{}}
	if t.Root != nil {
		s.Wallclock = t.Root.Duration()
	}
	var walk func(sp *Span, depth int)
	walk = func(sp *Span, depth int) {
		switch sp.Kind {
		case SpanStep:
			s.Steps++
		case SpanModel:
			s.ModelCalls++
			s.Usage.Add(sp.Usage)
		case SpanTool:
			s.ToolCalls++
			st := s.ToolBreakdown[sp.Tool]
			st.Calls++
			st.TotalDuration += sp.Duration()
			if sp.Error != "" {
				st.Errors++
			}
			s.ToolBreakdown[sp.Tool] = st
		case SpanCompaction:
			s.Compactions++
		case SpanRun:
			if depth > 0 { // a nested run is a subagent
				s.Subagents++
			}
		}
		for _, c := range sp.Children {
			walk(c, depth+1)
		}
	}
	if t.Root != nil {
		walk(t.Root, 0)
	}
	return s
}

// JSON returns the full trace tree as indented JSON.
func (t *Trace) JSON() ([]byte, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return json.MarshalIndent(t.Root, "", "  ")
}

// JSONL returns the trace flattened to newline-delimited JSON, one span per line
// (each line carries parent_id and depth), convenient for streaming and diffing.
func (t *Trace) JSONL() ([]byte, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	var b strings.Builder
	var walk func(sp, parent *Span, depth int) error
	walk = func(sp, parent *Span, depth int) error {
		rec := flatSpan{
			ID: sp.ID, Kind: sp.Kind, Name: sp.Name, Depth: depth,
			Start: sp.Start, End: sp.End, DurationMS: sp.Duration().Milliseconds(),
			Usage: sp.Usage, Model: sp.Model, Tool: sp.Tool,
			StopReason: sp.StopReason, Error: sp.Error,
		}
		if parent != nil {
			rec.ParentID = parent.ID
		}
		line, err := json.Marshal(rec)
		if err != nil {
			return err
		}
		b.Write(line)
		b.WriteByte('\n')
		for _, c := range sp.Children {
			if err := walk(c, sp, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if t.Root != nil {
		if err := walk(t.Root, nil, 0); err != nil {
			return nil, err
		}
	}
	return []byte(b.String()), nil
}

// LoadTrace reads a trace previously written by [Trace.JSON] (a span tree).
func LoadTrace(r io.Reader) (*Trace, error) {
	var root Span
	if err := json.NewDecoder(r).Decode(&root); err != nil {
		return nil, err
	}
	return &Trace{Root: &root}, nil
}

// WriteTo writes the trace as newline-delimited JSON (one span per line),
// implementing io.WriterTo.
func (t *Trace) WriteTo(w io.Writer) (int64, error) {
	b, err := t.JSONL()
	if err != nil {
		return 0, err
	}
	n, err := w.Write(b)
	return int64(n), err
}

type flatSpan struct {
	ID         string     `json:"id"`
	ParentID   string     `json:"parent_id,omitempty"`
	Depth      int        `json:"depth"`
	Kind       SpanKind   `json:"kind"`
	Name       string     `json:"name"`
	Start      time.Time  `json:"start"`
	End        time.Time  `json:"end,omitempty"`
	DurationMS int64      `json:"duration_ms"`
	Usage      Usage      `json:"usage,omitempty"`
	Model      string     `json:"model,omitempty"`
	Tool       string     `json:"tool,omitempty"`
	StopReason StopReason `json:"stop_reason,omitempty"`
	Error      string     `json:"error,omitempty"`
}
