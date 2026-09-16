package rimeno

import "log/slog"

// SlogSink returns an OnEvent handler that logs the run's event stream to a
// structured slog.Logger. Pass it as Config.OnEvent (or fan out to it from your
// own handler). A nil logger uses slog.Default.
//
//	agent, _ := rimeno.New(rimeno.Config{Model: m, OnEvent: rimeno.SlogSink(nil)})
func SlogSink(logger *slog.Logger) func(Event) {
	if logger == nil {
		logger = slog.Default()
	}
	return func(e Event) {
		switch ev := e.(type) {
		case RunStartedEvent:
			logger.Debug("rimeno.run_started", "agent", ev.Agent)
		case StepStartedEvent:
			logger.Debug("rimeno.step", "agent", ev.Agent, "step", ev.Step)
		case ModelResponseEvent:
			logger.Info("rimeno.model_response", "agent", ev.Agent,
				"tokens", ev.Usage.TotalTokens, "stop", string(ev.StopReason))
		case UsageUpdatedEvent:
			logger.Debug("rimeno.usage", "agent", ev.Agent,
				"total_tokens", ev.Total.TotalTokens, "cost_usd", ev.Total.CostUSD)
		case ToolCallStartedEvent:
			logger.Info("rimeno.tool_call", "agent", ev.Agent, "tool", ev.Call.Name)
		case ToolCallFinishedEvent:
			if ev.Err != nil {
				logger.Warn("rimeno.tool_error", "agent", ev.Agent, "tool", ev.Call.Name, "err", ev.Err)
			} else {
				logger.Debug("rimeno.tool_done", "agent", ev.Agent, "tool", ev.Call.Name)
			}
		case SubAgentStartedEvent:
			logger.Info("rimeno.subagent_started", "agent", ev.Agent, "name", ev.Name)
		case SubAgentFinishedEvent:
			logger.Info("rimeno.subagent_finished", "agent", ev.Agent, "name", ev.Name,
				"tokens", ev.Usage.TotalTokens)
		case CompactionStartedEvent:
			logger.Info("rimeno.compaction", "agent", ev.Agent,
				"messages", ev.Messages, "est_tokens", ev.EstimatedTokens)
		case RunFinishedEvent:
			logger.Info("rimeno.run_finished", "agent", ev.Agent)
		case ErrorEvent:
			logger.Error("rimeno.error", "agent", ev.Agent, "err", ev.Err)
		}
	}
}
