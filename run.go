package rimeno

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
)

// Result is the outcome of a run.
type Result struct {
	// Text is the assistant's final natural-language answer.
	Text string
	// Output is the decoded structured output (*T) when Config.Output was set.
	Output any
	// Messages are the messages produced during this run (assistant + tool).
	Messages []Message
	// Usage is cumulative usage for the session up to and including this run.
	Usage Usage
	// Steps is the number of loop iterations executed.
	Steps int
	// StopReason explains why the run ended.
	StopReason StopReason
	// Trace is the structured record of the run (see Trace.Summary).
	Trace *Trace
}

// Send appends the user input, runs the agent loop to completion, and returns the
// result. History and usage are retained on the session for subsequent turns.
func (s *Session) Send(ctx context.Context, input string) (*Result, error) {
	return s.runWith(ctx, UserMessage(input), s.agent.budget)
}

// runWith executes a run with an explicit budget. Subagents use this to run under
// a budget derived from the parent's remaining allowance.
func (s *Session) runWith(ctx context.Context, userMsg Message, budget Budget) (*Result, error) {
	a := s.agent
	tr := newTrace(a.name + ":run")
	res := &Result{Trace: tr}

	s.mu.Lock()
	s.history = append(s.history, userMsg)
	s.traces = append(s.traces, tr)
	s.mu.Unlock()

	bt := newBudgetTracker(budget)
	// Enforce the budget's wall-clock deadline as a hard context deadline, so a
	// long model or tool call is actually interrupted — not merely caught by the
	// between-steps check() after it finally returns.
	if dl := bt.deadlineTime(); !dl.IsZero() {
		var cancel context.CancelFunc
		ctx, cancel = context.WithDeadline(ctx, dl)
		defer cancel()
	}
	repairsLeft := a.outputRepairs
	a.emit(RunStartedEvent{a.base()})

	for step := 0; ; step++ {
		if err := ctx.Err(); err != nil {
			reason, retErr := classifyContextErr(ctx, bt)
			res.StopReason = reason
			tr.Root.StopReason = reason
			tr.Root.end()
			a.emit(ErrorEvent{a.base(), retErr})
			return res, retErr
		}
		if step >= a.maxSteps {
			res.StopReason = StopReasonMaxSteps
			break
		}
		if err := bt.check(); err != nil {
			res.StopReason = StopReasonBudget
			tr.Root.StopReason = StopReasonBudget
			tr.Root.Error = err.Error()
			tr.Root.end()
			a.emit(ErrorEvent{a.base(), err})
			return res, err
		}

		s.maybeCompact(ctx, tr)

		stepSpan := tr.child(tr.Root, SpanStep, "step-"+strconv.Itoa(step))
		a.emit(StepStartedEvent{a.base(), step})

		s.mu.Lock()
		req := s.buildRequestLocked()
		s.mu.Unlock()

		mspan := tr.child(stepSpan, SpanModel, a.model.ID())
		mspan.Model = a.model.ID()
		resp, err := a.generate(ctx, req)
		if err != nil && req.PreviousResponseID != "" && ctx.Err() == nil && threadingUnsupported(a.model, err) {
			// The provider advertised Responses threading but rejected the
			// previous_response_id (e.g. OpenRouter for a model without server-side
			// state). Latch threading off for this session, rebuild the turn with full
			// history, and retry once so the run continues transparently.
			s.mu.Lock()
			s.threadingDisabled = true
			s.lastResponseID = ""
			s.threadSentIdx = 0
			req = a.buildRequest(append([]Message(nil), s.history...))
			s.mu.Unlock()
			mspan.Attrs = map[string]any{"threading_fallback": true}
			resp, err = a.generate(ctx, req)
		}
		if err != nil {
			mspan.Error = err.Error()
			mspan.end()
			stepSpan.end()
			tr.Root.end()
			// A canceled context or an exceeded budget deadline is not a provider
			// error — classify it so callers can match on it.
			if ctx.Err() != nil {
				reason, retErr := classifyContextErr(ctx, bt)
				res.StopReason = reason
				a.emit(ErrorEvent{a.base(), retErr})
				return res, retErr
			}
			res.StopReason = StopReasonError
			a.emit(ErrorEvent{a.base(), err})
			return res, fmt.Errorf("rimeno: model generate: %w", err)
		}
		mspan.Usage = resp.Usage
		mspan.StopReason = resp.StopReason
		if resp.Usage.ReasoningTokens > 0 {
			mspan.Attrs = map[string]any{"reasoning_tokens": resp.Usage.ReasoningTokens}
		}
		mspan.end()

		bt.addUsage(resp.Usage)
		s.mu.Lock()
		s.usage.Add(resp.Usage)
		s.history = append(s.history, resp.Message)
		// With threading active, record the response to continue and how much history
		// the server now holds, so the next turn sends only the new items. An empty
		// resp.ID (model didn't return one) leaves lastResponseID empty, which safely
		// falls back to full history next turn.
		if s.threadingActive() {
			s.lastResponseID = resp.ID
			s.threadSentIdx = len(s.history)
		}
		s.usageSnapshot(res)
		s.mu.Unlock()
		res.Steps = step + 1
		res.Messages = append(res.Messages, resp.Message)
		a.emit(ModelResponseEvent{a.base(), resp.Usage, resp.StopReason, resp.Message.Text})
		a.emit(UsageUpdatedEvent{a.base(), res.Usage})

		if len(resp.Message.ToolCalls) == 0 {
			if a.output != nil {
				out, err := a.decodeOutput(resp.Message.Text)
				if err != nil {
					// Ask the model to repair invalid output, bounded by
					// OutputRepairAttempts, instead of failing immediately.
					if repairsLeft > 0 {
						repairsLeft--
						repair := UserMessage(buildRepairPrompt(err))
						s.mu.Lock()
						s.history = append(s.history, repair)
						s.mu.Unlock()
						res.Messages = append(res.Messages, repair)
						a.emit(ErrorEvent{a.base(), err})
						stepSpan.end()
						continue
					}
					res.Text = resp.Message.Text
					res.StopReason = StopReasonError
					stepSpan.end()
					tr.Root.end()
					a.emit(ErrorEvent{a.base(), err})
					return res, err
				}
				res.Output = out
			}
			res.Text = resp.Message.Text
			res.StopReason = StopReasonStop
			stepSpan.end()
			tr.Root.Usage = res.Usage
			tr.Root.StopReason = StopReasonStop
			tr.Root.end()
			a.emit(RunFinishedEvent{a.base(), res})
			return res, nil
		}

		toolMsgs, fatal := a.runToolCalls(ctx, tr, stepSpan, resp.Message.ToolCalls, bt)
		s.mu.Lock()
		s.history = append(s.history, toolMsgs...)
		s.mu.Unlock()
		res.Messages = append(res.Messages, toolMsgs...)
		stepSpan.end()
		if fatal != nil {
			res.StopReason = StopReasonError
			tr.Root.StopReason = StopReasonError
			tr.Root.Error = fatal.Error()
			tr.Root.end()
			a.emit(ErrorEvent{a.base(), fatal})
			return res, fatal
		}
	}

	tr.Root.Usage = res.Usage
	tr.Root.StopReason = res.StopReason
	tr.Root.end()
	a.emit(RunFinishedEvent{a.base(), res})
	return res, nil
}

// usageSnapshot copies session usage into the result (caller holds s.mu).
func (s *Session) usageSnapshot(res *Result) { res.Usage = s.usage }

// threadingActive reports whether this session should continue turns server-side
// via the Responses API (Config.ResponsesThreading set, the model can thread, and the
// provider hasn't rejected threading earlier in this session).
func (s *Session) threadingActive() bool {
	return !s.threadingDisabled && s.agent.responsesThreading && supportsResponseThreading(s.agent.model)
}

// buildRequestLocked builds the model request for the next step. With threading
// active and a prior response to continue, it sends only the messages the server
// does not yet have (history[threadSentIdx:]) plus PreviousResponseID; otherwise it
// sends the full history. Caller holds s.mu.
func (s *Session) buildRequestLocked() *Request {
	a := s.agent
	if s.threadingActive() && s.lastResponseID != "" && s.threadSentIdx <= len(s.history) {
		delta := append([]Message(nil), s.history[s.threadSentIdx:]...)
		req := a.buildRequest(delta)
		req.PreviousResponseID = s.lastResponseID
		return req
	}
	return a.buildRequest(append([]Message(nil), s.history...))
}

func (s *Session) maybeCompact(ctx context.Context, tr *Trace) {
	a := s.agent
	if a.compactor == nil {
		return
	}
	s.mu.Lock()
	hist := append([]Message(nil), s.history...)
	s.mu.Unlock()

	est := a.estimateTokens(hist)
	if !a.compactor.ShouldCompact(hist, est) {
		return
	}
	cspan := tr.child(tr.Root, SpanCompaction, "compact")
	cspan.Attrs = map[string]any{"before_messages": len(hist), "estimated_tokens": est}
	a.emit(CompactionStartedEvent{a.base(), len(hist), est})
	nh, err := a.compactor.Compact(ctx, hist)
	if err != nil {
		cspan.Error = err.Error()
		cspan.end()
		return
	}
	s.mu.Lock()
	s.history = nh
	// Compaction rewrote history, so any server-side thread no longer matches it.
	// Drop the thread; the next turn re-establishes it from the compacted history.
	s.lastResponseID = ""
	s.threadSentIdx = 0
	s.mu.Unlock()
	cspan.Attrs["after_messages"] = len(nh)
	cspan.end()
	a.emit(CompactionFinishedEvent{a.base(), len(nh)})
}

// generate performs one completion, streaming (and emitting TextDeltaEvent) when
// Config.Stream is set and the model supports it, otherwise a single Generate.
func (a *Agent) generate(ctx context.Context, req *Request) (*Response, error) {
	if a.stream {
		if sm, ok := a.model.(StreamingModel); ok {
			return a.generateStream(ctx, sm, req)
		}
	}
	return a.model.Generate(ctx, req)
}

func (a *Agent) generateStream(ctx context.Context, sm StreamingModel, req *Request) (*Response, error) {
	st, err := sm.Stream(ctx, req)
	if err != nil {
		return nil, err
	}
	defer func() { _ = st.Close() }()
	var final *Response
	for {
		chunk, err := st.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		if chunk.TextDelta != "" {
			a.emit(TextDeltaEvent{a.base(), chunk.TextDelta})
		}
		if chunk.ReasoningDelta != "" {
			a.emit(ReasoningDeltaEvent{a.base(), chunk.ReasoningDelta})
		}
		if chunk.Final != nil {
			final = chunk.Final
		}
	}
	if final == nil {
		return nil, fmt.Errorf("rimeno: stream ended without a final response")
	}
	return final, nil
}

func (a *Agent) buildRequest(history []Message) *Request {
	req := &Request{
		Model:              a.model.ID(),
		Messages:           history,
		Tools:              a.toolDefs,
		Temperature:        a.temperature,
		MaxTokens:          a.maxTokens,
		ToolChoice:         a.toolChoice,
		ReasoningEffort:    a.reasoningEffort,
		ReasoningMaxTokens: a.reasoningMaxTokens,
	}
	if a.output != nil {
		req.ResponseFormat = &ResponseFormat{
			Name:   a.output.Name,
			Schema: a.output.Schema,
			Strict: a.output.Strict,
		}
	}
	return req
}

// classifyContextErr maps a context error to a stop reason. A deadline the budget
// imposed is reported as a budget stop (with the *BudgetError); a plain
// cancellation — or a caller deadline shorter than the budget's — is a
// cancellation carrying the raw context error.
func classifyContextErr(ctx context.Context, bt *budgetTracker) (StopReason, error) {
	err := ctx.Err()
	if err == nil {
		return StopReasonStop, nil
	}
	if errors.Is(err, context.DeadlineExceeded) {
		if be := bt.check(); be != nil {
			return StopReasonBudget, be
		}
	}
	return StopReasonCanceled, err
}

func buildRepairPrompt(err error) string {
	return "Your previous response did not satisfy the required JSON schema: " +
		err.Error() + ". Reply again with ONLY corrected JSON that conforms to the schema — no prose, no code fences."
}

func (a *Agent) decodeOutput(text string) (any, error) {
	raw := []byte(strings.TrimSpace(text))
	// validate/decode are unexported hooks that only in-package constructors
	// (OutputOf, OutputFromSchema) populate. Tolerate a hand-built OutputSpec that
	// left them nil rather than panicking: nil validate means "no local check",
	// nil decode means "surface the raw bytes".
	if a.output.validate != nil {
		if err := a.output.validate(raw); err != nil {
			return nil, &OutputValidationError{Err: err, Raw: text}
		}
	}
	if a.output.decode == nil {
		return json.RawMessage(append([]byte(nil), raw...)), nil
	}
	out, err := a.output.decode(raw)
	if err != nil {
		return nil, &OutputValidationError{Err: err, Raw: text}
	}
	return out, nil
}

// runContext carries per-run state that subagent tools need to integrate with
// the parent: the trace to graft into, the budget to charge, the span to nest
// under, and the event emitter.
type runContext struct {
	trace       *Trace
	budget      *budgetTracker
	currentSpan *Span
	emit        func(Event)
}

// subagentInvoker is implemented by tools that run a nested rimeno agent. The loop
// detects it to graft the child trace, roll up usage into the parent budget, and
// emit subagent lifecycle events. It returns whether the subagent failed so the
// loop can count it toward the tool circuit breaker.
type subagentInvoker interface {
	Tool
	invokeSub(ctx context.Context, args json.RawMessage, rc *runContext) (string, bool)
}

// runToolCalls executes the requested tool calls (sequentially, or concurrently
// when ParallelTools is set) and returns the tool result messages. A non-nil
// error is returned only when a tool aborts the run via Fatal.
func (a *Agent) runToolCalls(ctx context.Context, tr *Trace, parent *Span, calls []ToolCall, bt *budgetTracker) ([]Message, error) {
	if !a.parallelTools || len(calls) <= 1 {
		out := make([]Message, 0, len(calls))
		for _, call := range calls {
			bt.incToolCall()
			tspan := tr.child(parent, SpanTool, call.Name)
			tspan.Tool = call.Name
			content, fatal := a.invokeOne(ctx, call, tspan, tr, bt)
			tspan.end()
			out = append(out, toolResultMessage(call.ID, call.Name, content))
			if fatal != nil {
				return out, fatal
			}
		}
		return out, nil
	}

	// Parallel: each call runs in its own goroutine, bounded by a semaphore.
	// Result order is preserved. A Fatal result cancels the rest.
	type outcome struct {
		msg   Message
		fatal error
	}
	results := make([]outcome, len(calls))
	sem := make(chan struct{}, a.maxParallel())
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	for i, call := range calls {
		wg.Add(1)
		sem <- struct{}{}
		go func(i int, call ToolCall) {
			defer wg.Done()
			defer func() { <-sem }()
			bt.incToolCall()
			tspan := tr.child(parent, SpanTool, call.Name)
			tspan.Tool = call.Name
			content, fatal := a.invokeOne(cctx, call, tspan, tr, bt)
			tspan.end()
			results[i] = outcome{toolResultMessage(call.ID, call.Name, content), fatal}
			if fatal != nil {
				cancel()
			}
		}(i, call)
	}
	wg.Wait()

	out := make([]Message, 0, len(calls))
	var fatal error
	for _, r := range results {
		out = append(out, r.msg)
		if r.fatal != nil && fatal == nil {
			fatal = r.fatal
		}
	}
	return out, fatal
}

// safeInvoke runs a tool, converting a panic into a recoverable *ToolError rather
// than letting it crash the run (and, e.g., the RPC server process). A buggy tool
// or a misbehaving MCP tool thus fails its one call and is reported to the model,
// like any other tool error; wrap with [Fatal] inside the tool to abort instead.
func safeInvoke(ctx context.Context, tool Tool, args json.RawMessage) (result any, err error) {
	defer func() {
		if r := recover(); r != nil {
			result = nil
			err = &ToolError{Tool: tool.Name(), Err: fmt.Errorf("panic: %v", r)}
		}
	}()
	return tool.Invoke(ctx, args)
}

func (a *Agent) invokeOne(ctx context.Context, call ToolCall, span *Span, tr *Trace, bt *budgetTracker) (string, error) {
	tool := a.tools[call.Name]
	if tool == nil {
		msg := "error: unknown tool " + strconv.Quote(call.Name)
		span.Error = msg
		a.emit(ToolCallFinishedEvent{a.base(), call, msg, nil})
		return msg, nil
	}
	// Circuit breaker: once a tool has failed maxToolErrors times this run, refuse
	// further calls to it without invoking it, so the model cannot thrash retrying a
	// broken tool (e.g. a dead subagent) until it burns the whole budget.
	if a.maxToolErrors > 0 && bt.toolErrorCount(call.Name) >= a.maxToolErrors {
		msg := fmt.Sprintf("error: tool %q disabled after %d failures in this run; do not call it again", call.Name, a.maxToolErrors)
		span.Error = msg
		a.emit(ToolCallFinishedEvent{a.base(), call, msg, nil})
		return msg, nil
	}
	if a.approve != nil {
		dec, reason := a.approve(ctx, ApprovalRequest{AgentName: a.name, Call: call, Tool: tool})
		if dec == ApprovalDeny {
			msg := "tool call denied by approver"
			if reason != "" {
				msg += ": " + reason
			}
			span.Error = msg
			a.emit(ToolCallFinishedEvent{a.base(), call, msg, nil})
			return msg, nil
		}
	}
	a.emit(ToolCallStartedEvent{a.base(), call})

	// Subagent tools integrate with the parent trace/budget.
	if sub, ok := tool.(subagentInvoker); ok {
		content, failed := sub.invokeSub(ctx, call.Arguments, &runContext{
			trace: tr, budget: bt, currentSpan: span, emit: a.emit,
		})
		if failed {
			bt.noteToolError(call.Name)
			span.Error = content
		}
		a.emit(ToolCallFinishedEvent{a.base(), call, content, nil})
		return content, nil
	}

	result, err := safeInvoke(ctx, tool, call.Arguments)
	if err != nil {
		if inner, ok := isFatal(err); ok {
			span.Error = inner.Error()
			a.emit(ToolCallFinishedEvent{a.base(), call, "fatal: " + inner.Error(), err})
			return "fatal error: " + inner.Error(), inner
		}
		bt.noteToolError(call.Name)
		msg := "error: " + err.Error()
		span.Error = err.Error()
		a.emit(ToolCallFinishedEvent{a.base(), call, msg, err})
		return msg, nil
	}
	content := marshalToolResult(result)
	a.emit(ToolCallFinishedEvent{a.base(), call, content, nil})
	return content, nil
}
