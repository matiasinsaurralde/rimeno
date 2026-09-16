package rimeno

// Tokenizer estimates the number of tokens in a piece of text. Implement it to
// plug in an exact tokenizer (e.g. a tiktoken binding) for precise compaction
// thresholds and budgeting. When no Tokenizer is configured, rimeno uses a fast
// ~4-characters-per-token heuristic ([EstimateTokens]).
type Tokenizer interface {
	CountTokens(text string) int
}

// estimateTokens counts the tokens of a conversation using the agent's Tokenizer
// if set, otherwise the built-in heuristic.
func (a *Agent) estimateTokens(msgs []Message) int {
	if a.tokenizer == nil {
		return EstimateTokens(msgs)
	}
	total := 0
	for _, m := range msgs {
		total += a.tokenizer.CountTokens(m.Text)
		for _, tc := range m.ToolCalls {
			total += a.tokenizer.CountTokens(tc.Name) + a.tokenizer.CountTokens(string(tc.Arguments))
		}
		total += 3 // per-message overhead
	}
	return total
}
