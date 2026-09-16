package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"testing"

	"github.com/matiasinsaurralde/rimeno"
	"github.com/matiasinsaurralde/rimeno/rimenotest"
)

// TestMain lets this test binary double as a minimal MCP server subprocess, so
// the stdio transport can be exercised for real without an external dependency.
func TestMain(m *testing.M) {
	if os.Getenv("RIMENO_MCP_STUB") == "1" {
		stubServe(os.Stdin, os.Stdout)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// --- a minimal in-process MCP server used by the tests ---

func stubServe(r io.Reader, w io.Writer) {
	br := bufio.NewReader(r)
	enc := json.NewEncoder(w)
	for {
		line, err := br.ReadBytes('\n')
		if t := bytes.TrimSpace(line); len(t) > 0 {
			var req struct {
				ID     json.RawMessage `json:"id"`
				Method string          `json:"method"`
				Params json.RawMessage `json:"params"`
			}
			if json.Unmarshal(t, &req) == nil {
				result := stubHandle(req.Method, req.Params)
				if len(req.ID) > 0 { // requests get a response; notifications don't
					out := map[string]any{"jsonrpc": "2.0", "id": req.ID}
					if e, ok := result.(*rpcError); ok {
						out["error"] = e
					} else {
						out["result"] = result
					}
					_ = enc.Encode(out)
				}
			}
		}
		if err != nil {
			return
		}
	}
}

func stubHandle(method string, params json.RawMessage) any {
	switch method {
	case "initialize":
		return map[string]any{
			"protocolVersion": DefaultProtocolVersion,
			"serverInfo":      map[string]any{"name": "stub", "version": "9.9"},
			"capabilities":    map[string]any{"tools": map[string]any{}},
			"instructions":    "stub server",
		}
	case "notifications/initialized":
		return nil
	case "ping":
		return map[string]any{}
	case "tools/list":
		var p struct {
			Cursor string `json:"cursor"`
		}
		_ = json.Unmarshal(params, &p)
		if p.Cursor == "" {
			// First page: one tool + a cursor, to exercise pagination.
			return map[string]any{
				"tools": []map[string]any{{
					"name":        "echo",
					"description": "echoes text back",
					"inputSchema": map[string]any{
						"type":       "object",
						"properties": map[string]any{"text": map[string]any{"type": "string"}},
						"required":   []string{"text"},
					},
				}},
				"nextCursor": "page2",
			}
		}
		return map[string]any{
			"tools": []map[string]any{{
				"name":        "fail",
				"description": "always errors",
				"inputSchema": map[string]any{"type": "object"},
			}},
		}
	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		_ = json.Unmarshal(params, &p)
		switch p.Name {
		case "echo":
			var a struct {
				Text string `json:"text"`
			}
			_ = json.Unmarshal(p.Arguments, &a)
			return map[string]any{"content": []map[string]any{{"type": "text", "text": a.Text}}}
		case "fail":
			return map[string]any{
				"content": []map[string]any{{"type": "text", "text": "boom"}},
				"isError": true,
			}
		default:
			return &rpcError{Code: -32602, Message: "unknown tool: " + p.Name}
		}
	default:
		return &rpcError{Code: -32601, Message: "method not found: " + method}
	}
}

// dialStub wires a Client to an in-process stub over pipes and returns it
// connected, with cleanup registered.
func dialStub(t *testing.T, opts ...Option) *Client {
	t.Helper()
	srvInR, srvInW := io.Pipe()   // client writes, server reads
	srvOutR, srvOutW := io.Pipe() // server writes, client reads
	go stubServe(srvInR, srvOutW)

	c := NewClient(srvOutR, srvInW, opts...)
	t.Cleanup(func() {
		_ = c.Close()
		_ = srvInW.Close()
		_ = srvOutW.Close()
		_ = srvInR.Close()
		_ = srvOutR.Close()
	})
	if _, err := c.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	return c
}

// --- tests ---

func TestClient_Connect(t *testing.T) {
	c := dialStub(t)
	si := c.ServerInfo()
	if si == nil || si.ServerInfo.Name != "stub" {
		t.Fatalf("server info = %+v", si)
	}
	if si.Instructions != "stub server" {
		t.Errorf("instructions = %q", si.Instructions)
	}
	if err := c.Ping(context.Background()); err != nil {
		t.Errorf("ping: %v", err)
	}
}

func TestClient_ListToolsPaginated(t *testing.T) {
	c := dialStub(t)
	infos, err := c.ListTools(context.Background())
	if err != nil {
		t.Fatalf("list tools: %v", err)
	}
	if len(infos) != 2 {
		t.Fatalf("got %d tools, want 2 (pagination): %+v", len(infos), infos)
	}
	if infos[0].Name != "echo" || infos[1].Name != "fail" {
		t.Errorf("tool names = %q, %q", infos[0].Name, infos[1].Name)
	}
}

func TestClient_CallTool(t *testing.T) {
	c := dialStub(t)
	out, err := c.CallTool(context.Background(), "echo", json.RawMessage(`{"text":"hi there"}`))
	if err != nil {
		t.Fatalf("call echo: %v", err)
	}
	if out != "hi there" {
		t.Errorf("echo = %q, want %q", out, "hi there")
	}

	// A tool reporting isError surfaces as a Go error carrying the content.
	_, err = c.CallTool(context.Background(), "fail", nil)
	if err == nil {
		t.Fatal("expected error from failing tool")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("boom")) {
		t.Errorf("error = %v, want it to contain \"boom\"", err)
	}

	// An unknown tool returns the server's JSON-RPC error.
	_, err = c.CallTool(context.Background(), "nope", nil)
	if err == nil {
		t.Fatal("expected error for unknown tool")
	}
}

func TestClient_ToolNamePrefix(t *testing.T) {
	c := dialStub(t, WithToolNamePrefix("stub_"))
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatalf("tools: %v", err)
	}
	if tools[0].Name() != "stub_echo" {
		t.Errorf("prefixed name = %q, want stub_echo", tools[0].Name())
	}
	// The wire name is still "echo": calling the wrapped tool works.
	out, err := tools[0].Invoke(context.Background(), json.RawMessage(`{"text":"x"}`))
	if err != nil {
		t.Fatalf("invoke: %v", err)
	}
	if out != "x" {
		t.Errorf("invoke = %v, want x", out)
	}
}

// TestClient_ToolsInAgent proves MCP tools drop straight into a rimeno agent: the
// model requests a tool, rimeno invokes it over MCP, and the result flows back.
func TestClient_ToolsInAgent(t *testing.T) {
	c := dialStub(t)
	tools, err := c.Tools(context.Background())
	if err != nil {
		t.Fatalf("tools: %v", err)
	}

	model := rimenotest.NewModel(
		rimenotest.Turn{ToolCalls: []rimeno.ToolCall{rimenotest.ToolCall("c1", "echo", map[string]string{"text": "from mcp"})}},
		rimenotest.Turn{Text: "done"},
	)
	agent, err := rimeno.New(rimeno.Config{Model: model, Tools: tools})
	if err != nil {
		t.Fatalf("new agent: %v", err)
	}
	res, err := agent.Run(context.Background(), "echo something")
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if res.Text != "done" {
		t.Errorf("text = %q, want done", res.Text)
	}
	var sawResult bool
	for _, m := range res.Messages {
		if m.Role == rimeno.RoleTool && m.Text == "from mcp" {
			sawResult = true
		}
	}
	if !sawResult {
		t.Errorf("expected a tool-result message %q in %+v", "from mcp", res.Messages)
	}
}

// TestStdioClient_Integration exercises the real subprocess stdio transport by
// re-executing this test binary in MCP-server mode (see TestMain).
func TestStdioClient_Integration(t *testing.T) {
	cmd := exec.Command(os.Args[0])
	cmd.Env = append(os.Environ(), "RIMENO_MCP_STUB=1")
	c, err := NewStdioCmdClient(cmd)
	if err != nil {
		t.Fatalf("start stdio client: %v", err)
	}
	defer func() { _ = c.Close() }()

	si, err := c.Connect(context.Background())
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	if si.ServerInfo.Name != "stub" {
		t.Errorf("server name = %q", si.ServerInfo.Name)
	}
	out, err := c.CallTool(context.Background(), "echo", json.RawMessage(`{"text":"over stdio"}`))
	if err != nil {
		t.Fatalf("call: %v", err)
	}
	if out != "over stdio" {
		t.Errorf("echo = %q", out)
	}
}

// TestClient_CallAfterClose verifies a call after Close returns promptly.
func TestClient_CallAfterClose(t *testing.T) {
	srvInR, srvInW := io.Pipe()
	srvOutR, srvOutW := io.Pipe()
	go stubServe(srvInR, srvOutW)
	c := NewClient(srvOutR, srvInW)
	if _, err := c.Connect(context.Background()); err != nil {
		t.Fatalf("connect: %v", err)
	}
	_ = c.Close()
	_ = srvInW.Close()
	_ = srvOutW.Close()
	if _, err := c.CallTool(context.Background(), "echo", nil); err == nil {
		t.Error("expected error calling a closed client")
	}
}

// TestNotificationPanicRecovered verifies a panicking notification handler does
// not crash the reader goroutine/process.
func TestNotificationPanicRecovered(t *testing.T) {
	c := &rpcClient{
		pending:  map[uint64]chan rpcMessage{},
		done:     make(chan struct{}),
		onNotify: func(string, json.RawMessage) { panic("handler boom") },
	}
	// Must return normally despite the handler panicking.
	c.dispatch(rpcMessage{Method: "notifications/message"})
}
