package rimeno

import (
	"errors"
	"fmt"
)

// Sentinel errors. Use errors.Is to test for them.
var (
	// ErrNoModel is returned by New when Config.Model is nil.
	ErrNoModel = errors.New("rimeno: no model configured")
	// ErrBudgetExceeded matches any *BudgetError via errors.Is.
	ErrBudgetExceeded = errors.New("rimeno: budget exceeded")
)

// BudgetError reports which budget limit was hit. It matches ErrBudgetExceeded
// under errors.Is.
type BudgetError struct {
	Limit  string // "total_tokens", "cost", "tool_calls", "deadline"
	Detail string
}

func (e *BudgetError) Error() string {
	return fmt.Sprintf("rimeno: budget exceeded (%s): %s", e.Limit, e.Detail)
}

func (e *BudgetError) Is(target error) bool { return target == ErrBudgetExceeded }

// ToolError wraps an error returned by a tool invocation. By default such errors
// are reported back to the model as the tool result so it can recover; wrap with
// [Fatal] to abort the run instead.
type ToolError struct {
	Tool string
	Err  error
}

func (e *ToolError) Error() string { return fmt.Sprintf("rimeno: tool %q: %v", e.Tool, e.Err) }
func (e *ToolError) Unwrap() error { return e.Err }

// fatalToolError signals that a tool failure must abort the whole run.
type fatalToolError struct{ err error }

func (e *fatalToolError) Error() string { return "rimeno: fatal tool error: " + e.err.Error() }
func (e *fatalToolError) Unwrap() error { return e.err }

// Fatal wraps err so that returning it from a Tool aborts the entire run instead
// of being fed back to the model as a recoverable tool result.
func Fatal(err error) error {
	if err == nil {
		return nil
	}
	return &fatalToolError{err: err}
}

func isFatal(err error) (error, bool) {
	var f *fatalToolError
	if errors.As(err, &f) {
		return f.err, true
	}
	return nil, false
}

// OutputValidationError is returned when the model's final message does not
// satisfy the configured structured-output schema.
type OutputValidationError struct {
	Err error
	Raw string
}

func (e *OutputValidationError) Error() string {
	return "rimeno: output failed schema validation: " + e.Err.Error()
}
func (e *OutputValidationError) Unwrap() error { return e.Err }
