package rpc

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"sync"
	"sync/atomic"

	"github.com/matiasinsaurralde/rimeno"
)

// JSON-RPC 2.0 error codes. The -320xx range is reserved for server-defined
// errors (see https://www.jsonrpc.org/specification#error_object).
const (
	codeParseError     = -32700
	codeInvalidRequest = -32600
	codeMethodNotFound = -32601
	codeInvalidParams  = -32602
	codeInternalError  = -32603
	codeSessionUnknown = -32000
	codeRunError       = -32001
)

// NewSessionParams is the decoded "session/new" request params, handed to the
// [ConfigFactory]. Raw carries the full params object so a host can decode its
// own custom fields.
type NewSessionParams struct {
	// Instructions, if set by the client, is a convenience system-prompt override
	// the factory may honor (it is not applied automatically).
	Instructions string `json:"instructions,omitempty"`
	// Raw is the complete params object as received, for host-specific fields.
	Raw json.RawMessage `json:"-"`
}

// ConfigFactory builds the [rimeno.Config] for a new session. It is called once per
// "session/new". The returned config's Model is required; the server chains its
// own event notifier onto Config.OnEvent (any handler you set still runs). The
// context is the connection context, canceled when the connection ends.
type ConfigFactory func(ctx context.Context, params NewSessionParams) (rimeno.Config, error)

// Option configures a [Server].
type Option func(*Server)

// WithServerInfo sets the name and version reported by the "initialize" method.
func WithServerInfo(name, version string) Option {
	return func(s *Server) {
		if name != "" {
			s.name = name
		}
		if version != "" {
			s.version = version
		}
	}
}

// Server adapts a rimeno agent to JSON-RPC over an io.Reader/io.Writer pair. It is
// safe to Serve multiple connections from one Server; each connection has its own
// independent set of sessions.
type Server struct {
	factory ConfigFactory
	name    string
	version string
}

// NewServer returns a Server that builds each session's agent from factory.
func NewServer(factory ConfigFactory, opts ...Option) *Server {
	s := &Server{factory: factory, name: "rimeno", version: "0.1.0"}
	for _, o := range opts {
		o(s)
	}
	return s
}

// ServeStdio serves on os.Stdin/os.Stdout. It is the usual entry point for a host
// that launches the rimeno process and speaks JSON-RPC over the child's stdio.
func (s *Server) ServeStdio(ctx context.Context) error {
	return s.Serve(ctx, os.Stdin, os.Stdout)
}

// Serve reads line-delimited JSON-RPC requests from r and writes responses and
// notifications to w until r reaches EOF or ctx is canceled between messages. It
// returns nil on a clean EOF.
func (s *Server) Serve(ctx context.Context, r io.Reader, w io.Writer) error {
	if s.factory == nil {
		return errors.New("rimeno/rpc: nil ConfigFactory")
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	c := &conn{srv: s, enc: json.NewEncoder(w), sessions: map[string]*rpcSession{}}
	br := bufio.NewReader(r)
	var wg sync.WaitGroup
	for {
		line, err := br.ReadBytes('\n')
		if trimmed := bytes.TrimSpace(line); len(trimmed) > 0 {
			msg := append([]byte(nil), trimmed...)
			wg.Add(1)
			go func() {
				defer wg.Done()
				c.handleLine(ctx, msg)
			}()
		}
		if err != nil {
			wg.Wait()
			if errors.Is(err, io.EOF) {
				return nil
			}
			return err
		}
		if ctx.Err() != nil {
			wg.Wait()
			return ctx.Err()
		}
	}
}

// conn is the per-connection state.
type conn struct {
	srv *Server
	enc *json.Encoder
	wmu sync.Mutex // serializes writes so each emitted line is whole

	mu       sync.Mutex // guards sessions
	sessions map[string]*rpcSession
	seq      atomic.Uint64
}

// rpcSession pairs a rimeno session with a mutex serializing prompts on it.
type rpcSession struct {
	id   string
	sess *rimeno.Session
	mu   sync.Mutex
}

// --- wire envelopes ---

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

func (e *rpcError) Error() string { return e.Message }

// --- read/dispatch ---

func (c *conn) handleLine(ctx context.Context, line []byte) {
	var req rpcRequest
	if err := json.Unmarshal(line, &req); err != nil {
		c.writeError(json.RawMessage("null"), &rpcError{Code: codeParseError, Message: "parse error", Data: err.Error()})
		return
	}
	notification := len(req.ID) == 0
	if req.JSONRPC != "" && req.JSONRPC != "2.0" {
		if !notification {
			c.writeError(req.ID, &rpcError{Code: codeInvalidRequest, Message: "unsupported jsonrpc version"})
		}
		return
	}
	if req.Method == "" {
		if !notification {
			c.writeError(req.ID, &rpcError{Code: codeInvalidRequest, Message: "missing method"})
		}
		return
	}

	result, rerr := c.safeDispatch(ctx, &req)
	if notification {
		return // JSON-RPC notifications get no response
	}
	if rerr != nil {
		c.writeError(req.ID, rerr)
		return
	}
	c.writeResult(req.ID, result)
}

// safeDispatch runs dispatch, converting any panic in a handler into a JSON-RPC
// internal error rather than crashing the server process (defense-in-depth beyond
// rimeno's own tool-panic recovery).
func (c *conn) safeDispatch(ctx context.Context, req *rpcRequest) (result any, rerr *rpcError) {
	defer func() {
		if r := recover(); r != nil {
			result = nil
			rerr = &rpcError{Code: codeInternalError, Message: fmt.Sprintf("internal error: %v", r)}
		}
	}()
	return c.dispatch(ctx, req)
}

func (c *conn) dispatch(ctx context.Context, req *rpcRequest) (any, *rpcError) {
	switch req.Method {
	case "initialize":
		return c.handleInitialize(), nil
	case "ping":
		return map[string]any{"pong": true}, nil
	case "session/new":
		return c.handleNewSession(ctx, req)
	case "session/prompt":
		return c.handlePrompt(ctx, req)
	case "session/snapshot":
		return c.handleSnapshot(req)
	case "session/resume":
		return c.handleResume(ctx, req)
	case "session/close":
		return c.handleClose(req)
	default:
		return nil, &rpcError{Code: codeMethodNotFound, Message: "method not found: " + req.Method}
	}
}

// --- method handlers ---

type serverInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type initializeResult struct {
	ServerInfo serverInfo `json:"server_info"`
	Methods    []string   `json:"methods"`
}

func (c *conn) handleInitialize() initializeResult {
	return initializeResult{
		ServerInfo: serverInfo{Name: c.srv.name, Version: c.srv.version},
		Methods: []string{"initialize", "ping", "session/new", "session/prompt",
			"session/snapshot", "session/resume", "session/close"},
	}
}

type newSessionResult struct {
	SessionID string `json:"session_id"`
}

func (c *conn) handleNewSession(ctx context.Context, req *rpcRequest) (any, *rpcError) {
	var p NewSessionParams
	if len(req.Params) > 0 {
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return nil, invalidParams(err)
		}
	}
	p.Raw = req.Params

	agent, id, rerr := c.buildAgent(ctx, p)
	if rerr != nil {
		return nil, rerr
	}
	c.store(id, &rpcSession{id: id, sess: agent.NewSession()})
	return newSessionResult{SessionID: id}, nil
}

// buildAgent runs the factory, allocates a session id, wires the event notifier,
// and constructs the agent. It is shared by session/new and session/resume.
func (c *conn) buildAgent(ctx context.Context, p NewSessionParams) (*rimeno.Agent, string, *rpcError) {
	cfg, err := c.srv.factory(ctx, p)
	if err != nil {
		return nil, "", &rpcError{Code: codeInternalError, Message: "config factory: " + err.Error()}
	}
	id := "sess-" + strconv.FormatUint(c.seq.Add(1), 10)
	host := cfg.OnEvent
	cfg.OnEvent = func(e rimeno.Event) {
		if host != nil {
			host(e)
		}
		c.emitEvent(id, e)
	}
	agent, err := rimeno.New(cfg)
	if err != nil {
		return nil, "", &rpcError{Code: codeInvalidParams, Message: "invalid config: " + err.Error()}
	}
	return agent, id, nil
}

func (c *conn) store(id string, rs *rpcSession) {
	c.mu.Lock()
	c.sessions[id] = rs
	c.mu.Unlock()
}

// handleSnapshot returns a serializable snapshot of a session for persistence.
func (c *conn) handleSnapshot(req *rpcRequest) (any, *rpcError) {
	var p struct {
		SessionID string `json:"session_id"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		return nil, invalidParams(err)
	}
	rs := c.lookup(p.SessionID)
	if rs == nil {
		return nil, &rpcError{Code: codeSessionUnknown, Message: "unknown session: " + p.SessionID}
	}
	rs.mu.Lock()
	state := rs.sess.Snapshot()
	rs.mu.Unlock()
	return map[string]any{"state": state}, nil
}

// handleResume creates a new session seeded from a prior snapshot.
func (c *conn) handleResume(ctx context.Context, req *rpcRequest) (any, *rpcError) {
	var p struct {
		Instructions string              `json:"instructions"`
		State        rimeno.SessionState `json:"state"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		return nil, invalidParams(err)
	}
	agent, id, rerr := c.buildAgent(ctx, NewSessionParams{Instructions: p.Instructions, Raw: req.Params})
	if rerr != nil {
		return nil, rerr
	}
	c.store(id, &rpcSession{id: id, sess: agent.ResumeSession(p.State)})
	return newSessionResult{SessionID: id}, nil
}

// PromptParams is the decoded "session/prompt" request params.
type PromptParams struct {
	SessionID string `json:"session_id"`
	Input     string `json:"input"`
}

type promptResult struct {
	Text       string         `json:"text"`
	Output     any            `json:"output,omitempty"`
	Usage      rimeno.Usage   `json:"usage"`
	StopReason string         `json:"stop_reason"`
	Steps      int            `json:"steps"`
	Summary    rimeno.Summary `json:"summary"`
}

func (c *conn) handlePrompt(ctx context.Context, req *rpcRequest) (any, *rpcError) {
	var p PromptParams
	if err := json.Unmarshal(req.Params, &p); err != nil {
		return nil, invalidParams(err)
	}
	if p.SessionID == "" {
		return nil, &rpcError{Code: codeInvalidParams, Message: "session_id required"}
	}
	rs := c.lookup(p.SessionID)
	if rs == nil {
		return nil, &rpcError{Code: codeSessionUnknown, Message: "unknown session: " + p.SessionID}
	}

	rs.mu.Lock()
	defer rs.mu.Unlock()
	res, err := rs.sess.Send(ctx, p.Input)
	if err != nil {
		data := map[string]any{}
		if res != nil {
			data["stop_reason"] = string(res.StopReason)
		}
		return nil, &rpcError{Code: codeRunError, Message: err.Error(), Data: data}
	}

	out := promptResult{
		Text:       res.Text,
		Output:     res.Output,
		Usage:      res.Usage,
		StopReason: string(res.StopReason),
		Steps:      res.Steps,
	}
	if res.Trace != nil {
		out.Summary = res.Trace.Summary()
	}
	return out, nil
}

func (c *conn) handleClose(req *rpcRequest) (any, *rpcError) {
	var p struct {
		SessionID string `json:"session_id"`
	}
	if err := json.Unmarshal(req.Params, &p); err != nil {
		return nil, invalidParams(err)
	}
	c.mu.Lock()
	_, ok := c.sessions[p.SessionID]
	delete(c.sessions, p.SessionID)
	c.mu.Unlock()
	if !ok {
		return nil, &rpcError{Code: codeSessionUnknown, Message: "unknown session: " + p.SessionID}
	}
	return map[string]any{"closed": true}, nil
}

func (c *conn) lookup(id string) *rpcSession {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sessions[id]
}

// --- notifications & writing ---

type sessionUpdate struct {
	SessionID string         `json:"session_id"`
	Event     map[string]any `json:"event"`
}

func (c *conn) emitEvent(sessionID string, e rimeno.Event) {
	c.write(rpcNotification{
		JSONRPC: "2.0",
		Method:  "session/update",
		Params:  sessionUpdate{SessionID: sessionID, Event: eventPayload(e)},
	})
}

func (c *conn) writeResult(id json.RawMessage, result any) {
	c.write(rpcResponse{JSONRPC: "2.0", ID: id, Result: result})
}

func (c *conn) writeError(id json.RawMessage, e *rpcError) {
	c.write(rpcResponse{JSONRPC: "2.0", ID: id, Error: e})
}

func (c *conn) write(v any) {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	_ = c.enc.Encode(v) // Encode appends '\n', keeping the stream line-delimited
}

func invalidParams(err error) *rpcError {
	return &rpcError{Code: codeInvalidParams, Message: "invalid params", Data: err.Error()}
}

// eventPayload renders a rimeno event as a stable wire map. Every payload carries
// "kind", "agent", and "time"; other fields depend on the event type.
func eventPayload(e rimeno.Event) map[string]any {
	m := map[string]any{"kind": rimeno.EventKind(e)}
	var base rimeno.EventBase
	switch ev := e.(type) {
	case rimeno.RunStartedEvent:
		base = ev.EventBase
	case rimeno.StepStartedEvent:
		base = ev.EventBase
		m["step"] = ev.Step
	case rimeno.ModelResponseEvent:
		base = ev.EventBase
		m["usage"] = ev.Usage
		m["stop_reason"] = string(ev.StopReason)
		m["text"] = ev.Text
	case rimeno.UsageUpdatedEvent:
		base = ev.EventBase
		m["usage"] = ev.Total
	case rimeno.ToolCallStartedEvent:
		base = ev.EventBase
		m["tool"] = ev.Call.Name
		m["call_id"] = ev.Call.ID
		if len(ev.Call.Arguments) > 0 {
			m["arguments"] = ev.Call.Arguments
		}
	case rimeno.ToolCallFinishedEvent:
		base = ev.EventBase
		m["tool"] = ev.Call.Name
		m["call_id"] = ev.Call.ID
		m["result"] = ev.Result
		if ev.Err != nil {
			m["error"] = ev.Err.Error()
		}
	case rimeno.CompactionStartedEvent:
		base = ev.EventBase
		m["messages"] = ev.Messages
		m["estimated_tokens"] = ev.EstimatedTokens
	case rimeno.CompactionFinishedEvent:
		base = ev.EventBase
		m["messages"] = ev.Messages
	case rimeno.SubAgentStartedEvent:
		base = ev.EventBase
		m["name"] = ev.Name
	case rimeno.SubAgentFinishedEvent:
		base = ev.EventBase
		m["name"] = ev.Name
		m["usage"] = ev.Usage
	case rimeno.TextDeltaEvent:
		base = ev.EventBase
		m["delta"] = ev.Delta
	case rimeno.RunFinishedEvent:
		base = ev.EventBase
		if ev.Result != nil {
			m["stop_reason"] = string(ev.Result.StopReason)
			m["steps"] = ev.Result.Steps
			m["usage"] = ev.Result.Usage
		}
	case rimeno.ErrorEvent:
		base = ev.EventBase
		if ev.Err != nil {
			m["error"] = ev.Err.Error()
		}
	}
	m["agent"] = base.Agent
	m["time"] = base.Time
	return m
}
