package jev

import (
	"encoding/json"
	"math"
	"strconv"
)

// Response is the decoded result of a Decide call.
type Response struct {
	Model    string            `json:"model"`
	Answers  map[string]Answer `json:"answers"`
	Usage    Usage             `json:"usage"`
	ID       string            `json:"id"`
	Provider string            `json:"provider"`
	// Raw is the untouched response body, for debugging.
	Raw json.RawMessage `json:"-"`
}

// Usage accounts tokens and cost for one decision call.
type Usage struct {
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	Cost         float64 `json:"cost"`
}

// Answer is one question's result. It carries every field jev may return across
// the three question types; use the typed accessors on Response (Noul, Choice,
// Score) rather than reading these directly.
type Answer struct {
	Type QuestionType `json:"type"`
	// noul
	NoulValue float64 `json:"noul"`
	// choice
	ChoiceValue string `json:"choice"`
	// score
	ScoreValue float64 `json:"score"`
	// choice + score
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
	// score only: index -> label
	Legend map[string]string `json:"legend"`
}

// Noul returns the [0,1] truth value for a noul question. ok is false if the
// named answer is missing or is not a noul question.
func (r *Response) Noul(name string) (value float64, ok bool) {
	a, present := r.Answers[name]
	if !present || a.Type != TypeNoul {
		return 0, false
	}
	return a.NoulValue, true
}

// ChoiceAnswer is the result of a choice question.
type ChoiceAnswer struct {
	Value         string
	Confidence    float64
	Probabilities map[string]float64
}

// Choice returns the selected option for a choice question. ok is false if the
// named answer is missing or is not a choice question.
func (r *Response) Choice(name string) (c ChoiceAnswer, ok bool) {
	a, present := r.Answers[name]
	if !present || a.Type != TypeChoice {
		return ChoiceAnswer{}, false
	}
	return ChoiceAnswer{
		Value:         a.ChoiceValue,
		Confidence:    a.Confidence,
		Probabilities: a.Probabilities,
	}, true
}

// ScoreAnswer is the result of a score question.
type ScoreAnswer struct {
	// Value is the continuous position across the scale indices (e.g. 1.03).
	Value         float64
	Confidence    float64
	Probabilities map[string]float64
	// Legend maps scale index to label (0 -> "Calm", ...).
	Legend map[int]string
}

// Label returns the legend label nearest to Value (rounded to the closest
// index). Empty string if there is no matching legend entry.
func (s ScoreAnswer) Label() string {
	return s.Legend[int(math.Round(s.Value))]
}

// Score returns the result for a score question. ok is false if the named answer
// is missing or is not a score question.
func (r *Response) Score(name string) (s ScoreAnswer, ok bool) {
	a, present := r.Answers[name]
	if !present || a.Type != TypeScore {
		return ScoreAnswer{}, false
	}
	legend := make(map[int]string, len(a.Legend))
	for k, v := range a.Legend {
		if i, err := strconv.Atoi(k); err == nil {
			legend[i] = v
		}
	}
	return ScoreAnswer{
		Value:         a.ScoreValue,
		Confidence:    a.Confidence,
		Probabilities: a.Probabilities,
		Legend:        legend,
	}, true
}
