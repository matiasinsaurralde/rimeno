package jev

import "encoding/json"

// QuestionType names a jev classifier kind.
type QuestionType string

const (
	// TypeNoul is a truth value in [0,1] — how true a statement is.
	TypeNoul QuestionType = "noul"
	// TypeChoice picks one of N labeled options.
	TypeChoice QuestionType = "choice"
	// TypeScore places the state on an ordered scale of labels.
	TypeScore QuestionType = "score"
)

// Request is a single decision call: classify State against a set of Questions.
type Request struct {
	// Model overrides the client's configured model for this call. Optional.
	Model string
	// State is the text to classify.
	State string
	// Questions maps a caller-chosen name to a typed question. Answers in the
	// Response are keyed by these same names.
	Questions map[string]Question
}

// Question is one typed classifier. Build one with Noul, Choice, or Score rather
// than populating it directly — those helpers encode Criteria in the shape jev
// expects for each type (an object for noul/choice, an array for score).
type Question struct {
	Type         QuestionType `json:"type"`
	Instructions string       `json:"instructions"`
	// Criteria is polymorphic on Type: an object (label->description) for noul and
	// choice, an array (ordered labels) for score. json.RawMessage keeps it exact.
	Criteria json.RawMessage `json:"criteria"`
}

// Noul builds a noul (truth-value) question. trueDesc/falseDesc describe what the
// true and false ends mean.
func Noul(instructions, trueDesc, falseDesc string) Question {
	crit, _ := json.Marshal(map[string]string{"true": trueDesc, "false": falseDesc})
	return Question{Type: TypeNoul, Instructions: instructions, Criteria: crit}
}

// Choice builds a choice question. criteria maps each option label to its
// description; the model returns one of these labels.
func Choice(instructions string, criteria map[string]string) Question {
	crit, _ := json.Marshal(criteria)
	return Question{Type: TypeChoice, Instructions: instructions, Criteria: crit}
}

// Score builds a score question over an ordered scale. labels are given
// low-to-high (e.g. "Calm", "Frustrated", "Very angry"); the model returns a
// continuous position across their indices.
func Score(instructions string, labels ...string) Question {
	crit, _ := json.Marshal(labels)
	return Question{Type: TypeScore, Instructions: instructions, Criteria: crit}
}

// wireRequest is the exact JSON body sent to /decisions.
type wireRequest struct {
	Model     string              `json:"model"`
	State     string              `json:"state"`
	Questions map[string]Question `json:"questions"`
}
