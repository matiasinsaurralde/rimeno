package rpc

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"strconv"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// --- test client over an in-memory pipe pair ---

type clientResponse struct {
	ID     json.RawMessage `json:"id"`
	Result json.RawMessage `json:"result"`
	Error  *rpcError       `json:"error"`
}

type testClient struct {
	t    *testing.T
	in   *io.PipeWriter
	out  *bufio.Reader
	id   int
	done chan error
}

// serve wires a Server to a pipe pair and returns a driving client.
func serve(t *testing.T, factory ConfigFactory) *testClient {
	t.Helper()
	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	srv := NewServer(factory)
	done := make(chan error, 1)
	go func() {
		err := srv.Serve(context.Background(), inR, outW)
		outW.CloseWithError(io.EOF)
		done <- err
	}()
	tc := &testClient{t: t, in: inW, out: bufio.NewReader(outR), done: done}
	t.Cleanup(func() {
		_ = inW.Close()
		<-done
	})
	return tc
}

// call sends a request and returns the matching response plus any notifications
// received before it. Params may be nil.
func (tc *testClient) call(method string, params any) (clientResponse, []sessionUpdate) {
	tc.t.Helper()
	tc.id++
	id := tc.id
	req := map[string]any{"jsonrpc": "2.0", "id": id, "method": method}
	if params != nil {
		req["params"] = params
	}
	b, err := json.Marshal(req)
	if err != nil {
		tc.t.Fatalf("marshal request: %v", err)
	}
	if _, err := tc.in.Write(append(b, '\n')); err != nil {
		tc.t.Fatalf("write request: %v", err)
	}

	var notes []sessionUpdate
	for {
		line, err := tc.out.ReadBytes('\n')
		if err != nil {
			tc.t.Fatalf("read response: %v", err)
		}
		var probe struct {
			ID     json.RawMessage `json:"id"`
			Method string          `json:"method"`
		}
		if err := json.Unmarshal(line, &probe); err != nil {
			tc.t.Fatalf("decode line %q: %v", line, err)
		}
		if probe.Method == "session/update" {
			var n struct {
				Params sessionUpdate `json:"params"`
			}
			if err := json.Unmarshal(line, &n); err != nil {
				tc.t.Fatalf("decode notification: %v", err)
			}
			notes = append(notes, n.Params)
			continue
		}
		var resp clientResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			tc.t.Fatalf("decode response: %v", err)
		}
		if got := string(resp.ID); got != strconv.Itoa(id) {
			tc.t.Fatalf("response id = %s, want %d", got, id)
		}
		return resp, notes
	}
}

// writeRaw sends a raw line (for malformed-input tests) and reads one response.
func (tc *testClient) writeRaw(line string) clientResponse {
	tc.t.Helper()
	if _, err := tc.in.Write([]byte(line + "\n")); err != nil {
		tc.t.Fatalf("write raw: %v", err)
	}
	b, err := tc.out.ReadBytes('\n')
	if err != nil {
		tc.t.Fatalf("read raw response: %v", err)
	}
	var resp clientResponse
	if err := json.Unmarshal(b, &resp); err != nil {
		tc.t.Fatalf("decode raw response: %v", err)
	}
	return resp
}

// --- factories ---

type echoArgs struct {
	Text string `json:"text"`
}

// toolFactory builds an agent with one echo tool and a scripted model that calls
// the tool once, then answers.
func toolFactory(_ context.Context, _ NewSessionParams) (rimeno.Config, error) {
	echo := rimeno.NewTool("echo", "echoes its input",
		func(_ context.Context, in echoArgs) (string, error) { return in.Text, nil })
	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "echo", echoArgs{Text: "hello"})}},
		rimenotest.Turn{Text: "done: hello"},
	)
	return rimeno.Config{Model: model, Tools: []rimeno.Tool{echo}, Name: "tester"}, nil
}

// --- tests ---

func TestServer_InitializeAndPing(t *testing.T) {
	tc := serve(t, toolFactory)

	resp, _ := tc.call("initialize", nil)
	if resp.Error != nil {
		t.Fatalf("initialize error: %+v", resp.Error)
	}
	var init initializeResult
	if err := json.Unmarshal(resp.Result, &init); err != nil {
		t.Fatalf("decode initialize: %v", err)
	}
	if init.ServerInfo.Name != "rimeno" || init.ServerInfo.Version == "" {
		t.Errorf("server_info = %+v", init.ServerInfo)
	}
	if len(init.Methods) == 0 {
		t.Errorf("expected non-empty methods list")
	}

	resp, _ = tc.call("ping", nil)
	if resp.Error != nil {
		t.Fatalf("ping error: %+v", resp.Error)
	}
	var pong map[string]any
	if err := json.Unmarshal(resp.Result, &pong); err != nil {
		t.Fatalf("decode ping: %v", err)
	}
	if pong["pong"] != true {
		t.Errorf("ping = %v", pong)
	}
}

func TestServer_NewPromptClose(t *testing.T) {
	tc := serve(t, toolFactory)

	resp, _ := tc.call("session/new", map[string]any{})
	if resp.Error != nil {
		t.Fatalf("session/new error: %+v", resp.Error)
	}
	var ns newSessionResult
	if err := json.Unmarshal(resp.Result, &ns); err != nil {
		t.Fatalf("decode session/new: %v", err)
	}
	if ns.SessionID == "" {
		t.Fatal("empty session_id")
	}

	resp, notes := tc.call("session/prompt", PromptParams{SessionID: ns.SessionID, Input: "go"})
	if resp.Error != nil {
		t.Fatalf("session/prompt error: %+v", resp.Error)
	}
	var pr promptResult
	if err := json.Unmarshal(resp.Result, &pr); err != nil {
		t.Fatalf("decode prompt result: %v", err)
	}
	if pr.Text != "done: hello" {
		t.Errorf("text = %q, want %q", pr.Text, "done: hello")
	}
	if pr.StopReason != "stop" {
		t.Errorf("stop_reason = %q", pr.StopReason)
	}
	if pr.Summary.ToolCalls != 1 {
		t.Errorf("summary.tool_calls = %d, want 1", pr.Summary.ToolCalls)
	}
	if pr.Summary.ModelCalls != 2 {
		t.Errorf("summary.model_calls = %d, want 2", pr.Summary.ModelCalls)
	}

	// Notifications should include the tool lifecycle and a run_finished, all
	// tagged with the session id.
	kinds := map[string]int{}
	for _, n := range notes {
		if n.SessionID != ns.SessionID {
			t.Errorf("notification session_id = %q, want %q", n.SessionID, ns.SessionID)
		}
		kind, _ := n.Event["kind"].(string)
		kinds[kind]++
	}
	for _, want := range []string{"run_started", "tool_call_started", "tool_call_finished", "run_finished"} {
		if kinds[want] == 0 {
			t.Errorf("missing %q notification (got kinds %v)", want, kinds)
		}
	}
	// The tool-start notification should carry the tool name and arguments.
	var sawArgs bool
	for _, n := range notes {
		if n.Event["kind"] == "tool_call_started" {
			if n.Event["tool"] != "echo" {
				t.Errorf("tool = %v, want echo", n.Event["tool"])
			}
			if _, ok := n.Event["arguments"]; ok {
				sawArgs = true
			}
		}
	}
	if !sawArgs {
		t.Errorf("tool_call_started missing arguments")
	}

	// Close, then a prompt on the closed session must fail.
	resp, _ = tc.call("session/close", map[string]any{"session_id": ns.SessionID})
	if resp.Error != nil {
		t.Fatalf("session/close error: %+v", resp.Error)
	}
	resp, _ = tc.call("session/prompt", PromptParams{SessionID: ns.SessionID, Input: "again"})
	if resp.Error == nil || resp.Error.Code != codeSessionUnknown {
		t.Errorf("expected session-unknown error, got %+v", resp.Error)
	}
}

func TestServer_StructuredOutput(t *testing.T) {
	type greeting struct {
		Message string `json:"message"`
	}
	factory := func(_ context.Context, _ NewSessionParams) (rimeno.Config, error) {
		model := rimenotest.NewModel(rimenotest.Turn{Text: `{"message":"hi"}`})
		return rimeno.Config{Model: model, Output: rimeno.OutputOf[greeting]()}, nil
	}
	tc := serve(t, factory)

	resp, _ := tc.call("session/new", nil)
	var ns newSessionResult
	_ = json.Unmarshal(resp.Result, &ns)

	resp, _ = tc.call("session/prompt", PromptParams{SessionID: ns.SessionID, Input: "greet"})
	if resp.Error != nil {
		t.Fatalf("prompt error: %+v", resp.Error)
	}
	var pr struct {
		Output greeting `json:"output"`
	}
	if err := json.Unmarshal(resp.Result, &pr); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if pr.Output.Message != "hi" {
		t.Errorf("output.message = %q, want hi", pr.Output.Message)
	}
}

func TestServer_Errors(t *testing.T) {
	tc := serve(t, toolFactory)

	// Unknown method.
	resp, _ := tc.call("does/not/exist", nil)
	if resp.Error == nil || resp.Error.Code != codeMethodNotFound {
		t.Errorf("expected method-not-found, got %+v", resp.Error)
	}

	// Prompt on an unknown session.
	resp, _ = tc.call("session/prompt", PromptParams{SessionID: "nope", Input: "x"})
	if resp.Error == nil || resp.Error.Code != codeSessionUnknown {
		t.Errorf("expected session-unknown, got %+v", resp.Error)
	}

	// Malformed JSON → parse error with null id.
	resp = tc.writeRaw("{not json")
	if resp.Error == nil || resp.Error.Code != codeParseError {
		t.Errorf("expected parse error, got %+v", resp.Error)
	}
	if string(resp.ID) != "null" {
		t.Errorf("parse-error id = %s, want null", resp.ID)
	}
}

func TestServer_FactoryError(t *testing.T) {
	factory := func(_ context.Context, _ NewSessionParams) (rimeno.Config, error) {
		return rimeno.Config{}, io.ErrUnexpectedEOF
	}
	tc := serve(t, factory)
	resp, _ := tc.call("session/new", nil)
	if resp.Error == nil || resp.Error.Code != codeInternalError {
		t.Errorf("expected internal error from factory, got %+v", resp.Error)
	}
}

func TestServer_MultipleSessionsIsolated(t *testing.T) {
	tc := serve(t, toolFactory)

	resp, _ := tc.call("session/new", nil)
	var a newSessionResult
	_ = json.Unmarshal(resp.Result, &a)
	resp, _ = tc.call("session/new", nil)
	var b newSessionResult
	_ = json.Unmarshal(resp.Result, &b)

	if a.SessionID == b.SessionID {
		t.Fatalf("session ids not unique: %q", a.SessionID)
	}

	for _, id := range []string{a.SessionID, b.SessionID} {
		resp, _ := tc.call("session/prompt", PromptParams{SessionID: id, Input: "go"})
		if resp.Error != nil {
			t.Fatalf("prompt on %s: %+v", id, resp.Error)
		}
		var pr promptResult
		_ = json.Unmarshal(resp.Result, &pr)
		if pr.Text != "done: hello" {
			t.Errorf("session %s text = %q", id, pr.Text)
		}
	}
}

func TestServer_SnapshotResume(t *testing.T) {
	// A model that answers differently on each call, so we can tell turns apart.
	factory := func(_ context.Context, _ NewSessionParams) (rimeno.Config, error) {
		m := rimenotest.NewModel(
			rimenotest.Turn{Text: "answer one"},
			rimenotest.Turn{Text: "answer two"},
		)
		return rimeno.Config{Model: m, Instructions: "sys"}, nil
	}
	tc := serve(t, factory)

	// New session, one prompt, then snapshot.
	resp, _ := tc.call("session/new", nil)
	var ns newSessionResult
	_ = json.Unmarshal(resp.Result, &ns)
	tc.call("session/prompt", PromptParams{SessionID: ns.SessionID, Input: "q1"})

	resp, _ = tc.call("session/snapshot", map[string]any{"session_id": ns.SessionID})
	if resp.Error != nil {
		t.Fatalf("snapshot error: %+v", resp.Error)
	}
	var snap struct {
		State rimeno.SessionState `json:"state"`
	}
	if err := json.Unmarshal(resp.Result, &snap); err != nil {
		t.Fatalf("decode snapshot: %v", err)
	}
	if len(snap.State.Messages) == 0 {
		t.Fatal("snapshot has no messages")
	}

	// Resume into a fresh session from the snapshot; its history should carry over
	// (a fresh factory model starts at turn 0, so the next answer is "answer one").
	resp, _ = tc.call("session/resume", map[string]any{"state": snap.State})
	if resp.Error != nil {
		t.Fatalf("resume error: %+v", resp.Error)
	}
	var rs newSessionResult
	_ = json.Unmarshal(resp.Result, &rs)
	if rs.SessionID == ns.SessionID {
		t.Errorf("resumed session should get a new id")
	}

	// The resumed session must contain the prior turn's history.
	resp, _ = tc.call("session/snapshot", map[string]any{"session_id": rs.SessionID})
	var snap2 struct {
		State rimeno.SessionState `json:"state"`
	}
	_ = json.Unmarshal(resp.Result, &snap2)
	if len(snap2.State.Messages) != len(snap.State.Messages) {
		t.Errorf("resumed history len = %d, want %d", len(snap2.State.Messages), len(snap.State.Messages))
	}
}

func TestServer_HandlerPanicRecovered(t *testing.T) {
	// A factory that panics on session/new must not crash the server; the client
	// gets an internal error and can keep using the connection.
	var calls int
	factory := func(_ context.Context, _ NewSessionParams) (rimeno.Config, error) {
		calls++
		if calls == 1 {
			panic("factory boom")
		}
		return rimeno.Config{Model: rimenotest.NewModel(rimenotest.Turn{Text: "ok"})}, nil
	}
	tc := serve(t, factory)

	resp, _ := tc.call("session/new", nil)
	if resp.Error == nil || resp.Error.Code != codeInternalError {
		t.Fatalf("expected internal error from panic, got %+v", resp.Error)
	}
	// The connection still works afterward.
	resp, _ = tc.call("ping", nil)
	if resp.Error != nil {
		t.Fatalf("connection broken after panic: %+v", resp.Error)
	}
}

func TestServer_NilFactory(t *testing.T) {
	srv := NewServer(nil)
	err := srv.Serve(context.Background(), io.LimitReader(nil, 0), io.Discard)
	if err == nil {
		t.Fatal("expected error for nil factory")
	}
}
