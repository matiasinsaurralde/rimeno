package rimeno

import "time"

// Event is the sealed sum type emitted through Config.OnEvent during a run.
// Consume it with a type switch; use [EventKind] for a short label.
//
// The set of concrete events mirrors the run loop: run/step lifecycle, model
// responses, tool-call start/finish, usage updates, compaction, and subagent
// lifecycle. Events are delivered synchronously on the run goroutine, so
// handlers should be cheap (offload heavy work to another goroutine).
type Event interface{ isEvent() }

// EventBase is embedded by every event and carries common metadata.
type EventBase struct {
	Agent string
	Time  time.Time
}

func (EventBase) isEvent() {}

type (
	// RunStartedEvent fires once when a run begins.
	RunStartedEvent struct{ EventBase }
	// StepStartedEvent fires at the top of each loop step.
	StepStartedEvent struct {
		EventBase
		Step int
	}
	// ModelResponseEvent fires after each model completion.
	ModelResponseEvent struct {
		EventBase
		Usage      Usage
		StopReason StopReason
		Text       string
	}
	// UsageUpdatedEvent reports cumulative run usage after a model call.
	UsageUpdatedEvent struct {
		EventBase
		Total Usage
	}
	// ToolCallStartedEvent fires before a tool is invoked.
	ToolCallStartedEvent struct {
		EventBase
		Call ToolCall
	}
	// ToolCallFinishedEvent fires after a tool invocation (Err is nil on success;
	// Result is the string fed back to the model).
	ToolCallFinishedEvent struct {
		EventBase
		Call   ToolCall
		Result string
		Err    error
	}
	// CompactionStartedEvent fires before context compaction.
	CompactionStartedEvent struct {
		EventBase
		Messages        int
		EstimatedTokens int
	}
	// CompactionFinishedEvent fires after context compaction.
	CompactionFinishedEvent struct {
		EventBase
		Messages int
	}
	// SubAgentStartedEvent fires when a subagent run begins.
	SubAgentStartedEvent struct {
		EventBase
		Name string
	}
	// SubAgentFinishedEvent fires when a subagent run completes.
	SubAgentFinishedEvent struct {
		EventBase
		Name  string
		Usage Usage
	}
	// TextDeltaEvent carries incremental assistant text when streaming.
	TextDeltaEvent struct {
		EventBase
		Delta string
	}
	// ReasoningDeltaEvent carries incremental reasoning (chain-of-thought) text when
	// streaming a reasoning model, so a host can render a "thinking…" pane distinct
	// from the answer.
	ReasoningDeltaEvent struct {
		EventBase
		Delta string
	}
	// RunFinishedEvent fires once when a run completes successfully.
	RunFinishedEvent struct {
		EventBase
		Result *Result
	}
	// ErrorEvent fires when the run terminates with an error.
	ErrorEvent struct {
		EventBase
		Err error
	}
)

// EventKind returns a short, stable label for an event, convenient for logging.
func EventKind(e Event) string {
	switch e.(type) {
	case RunStartedEvent:
		return "run_started"
	case StepStartedEvent:
		return "step_started"
	case ModelResponseEvent:
		return "model_response"
	case UsageUpdatedEvent:
		return "usage_updated"
	case ToolCallStartedEvent:
		return "tool_call_started"
	case ToolCallFinishedEvent:
		return "tool_call_finished"
	case CompactionStartedEvent:
		return "compaction_started"
	case CompactionFinishedEvent:
		return "compaction_finished"
	case SubAgentStartedEvent:
		return "subagent_started"
	case SubAgentFinishedEvent:
		return "subagent_finished"
	case TextDeltaEvent:
		return "text_delta"
	case ReasoningDeltaEvent:
		return "reasoning_delta"
	case RunFinishedEvent:
		return "run_finished"
	case ErrorEvent:
		return "error"
	default:
		return "unknown"
	}
}
