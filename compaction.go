package rimeno

import (
	"context"
	"fmt"
	"strings"
)

// Compactor decides when a conversation has grown too large and rewrites it into
// a smaller form. A nil Compactor disables compaction.
type Compactor interface {
	// ShouldCompact reports whether the history should be compacted now.
	ShouldCompact(msgs []Message, estimatedTokens int) bool
	// Compact returns a replacement history (typically: system messages, a
	// summary, and the most recent turns verbatim).
	Compact(ctx context.Context, msgs []Message) ([]Message, error)
}

// EstimateTokens is a fast, provider-agnostic heuristic (~4 chars/token) used for
// compaction thresholds when no exact tokenizer is configured.
func EstimateTokens(msgs []Message) int {
	chars := 0
	for _, m := range msgs {
		chars += len(m.Text)
		for _, tc := range m.ToolCalls {
			chars += len(tc.Name) + len(tc.Arguments)
		}
	}
	return chars/4 + len(msgs)*3
}

const defaultSummarizerPrompt = `Condense the conversation below into a compact handoff note so the work can ` +
	`continue with less context.

Do this in a single pass: write only the note, take no actions, and call no ` +
	`tools. Include what someone resuming the task would need — the objective, the ` +
	`key decisions and why they were made, the files, paths, and identifiers ` +
	`involved, what has and hasn't worked so far, and the concrete next step. Keep ` +
	`it specific and true to what actually happened; leave out filler.`

// SummarizingCompactor compacts by summarizing older messages with a model,
// keeping the system prefix and the most recent turns verbatim. It mirrors the
// "floor cutoff + minimum retained + summary" design common to production
// harnesses.
type SummarizingCompactor struct {
	// Model performs the summarization (may differ from the agent's model).
	Model Model
	// MaxContextTokens is the model's context window used to compute the trigger.
	MaxContextTokens int
	// TriggerFraction is the fraction of MaxContextTokens at which compaction
	// fires (default 0.8).
	TriggerFraction float64
	// KeepRecent is the number of most-recent messages kept verbatim (default 6).
	KeepRecent int
	// Prompt overrides the summarizer system prompt.
	Prompt string
}

func (c *SummarizingCompactor) trigger() float64 {
	if c.TriggerFraction > 0 {
		return c.TriggerFraction
	}
	return 0.8
}

func (c *SummarizingCompactor) keepRecent() int {
	if c.KeepRecent > 0 {
		return c.KeepRecent
	}
	return 6
}

// ShouldCompact fires when the estimate crosses TriggerFraction of the window.
func (c *SummarizingCompactor) ShouldCompact(_ []Message, estimatedTokens int) bool {
	if c.MaxContextTokens <= 0 {
		return false
	}
	return estimatedTokens >= int(float64(c.MaxContextTokens)*c.trigger())
}

// Compact summarizes everything older than the retained tail into a single
// system message.
func (c *SummarizingCompactor) Compact(ctx context.Context, msgs []Message) ([]Message, error) {
	if c.Model == nil {
		return msgs, fmt.Errorf("rimeno: SummarizingCompactor has no Model")
	}
	// Preserve the leading system messages verbatim.
	i := 0
	var head []Message
	for i < len(msgs) && msgs[i].Role == RoleSystem {
		head = append(head, msgs[i])
		i++
	}
	body := msgs[i:]
	keep := c.keepRecent()
	if len(body) <= keep {
		return msgs, nil // nothing worth summarizing yet
	}
	toSummarize := body[:len(body)-keep]
	recent := body[len(body)-keep:]

	prompt := c.Prompt
	if prompt == "" {
		prompt = defaultSummarizerPrompt
	}
	req := &Request{
		Model: c.Model.ID(),
		Messages: []Message{
			{Role: RoleSystem, Text: prompt},
			{Role: RoleUser, Text: renderTranscript(toSummarize)},
		},
	}
	resp, err := c.Model.Generate(ctx, req)
	if err != nil {
		return msgs, err
	}
	summary := Message{Role: RoleSystem, Text: resp.Message.Text}

	out := make([]Message, 0, len(head)+1+len(recent))
	out = append(out, head...)
	out = append(out, summary)
	out = append(out, recent...)
	return out, nil
}

// renderTranscript flattens messages into a plain-text transcript so it can be
// summarized by any provider without tool-role semantics getting in the way.
func renderTranscript(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(string(m.Role))
		b.WriteString(": ")
		if m.Text != "" {
			b.WriteString(m.Text)
		}
		for _, tc := range m.ToolCalls {
			fmt.Fprintf(&b, "\n  [tool_call %s(%s)]", tc.Name, string(tc.Arguments))
		}
		b.WriteByte('\n')
	}
	return b.String()
}
