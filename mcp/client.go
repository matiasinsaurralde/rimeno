package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/matiasinsaurralde/rimeno"
)

// DefaultProtocolVersion is the MCP protocol version rimeno advertises during the
// initialize handshake. Servers negotiate down if they only support an older one.
const DefaultProtocolVersion = "2025-06-18"

// Implementation identifies a party in the MCP handshake (client or server).
type Implementation struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ServerInfo is what a server reports from the initialize handshake.
type ServerInfo struct {
	ProtocolVersion string          `json:"protocolVersion"`
	ServerInfo      Implementation  `json:"serverInfo"`
	Capabilities    json.RawMessage `json:"capabilities,omitempty"`
	Instructions    string          `json:"instructions,omitempty"`
}

// ToolInfo describes one tool advertised by an MCP server.
type ToolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// Option configures a [Client].
type Option func(*Client)

// WithClientInfo sets the name/version rimeno reports to the server.
func WithClientInfo(name, version string) Option {
	return func(c *Client) { c.info = Implementation{Name: name, Version: version} }
}

// WithProtocolVersion overrides the advertised protocol version.
func WithProtocolVersion(v string) Option {
	return func(c *Client) {
		if v != "" {
			c.protocol = v
		}
	}
}

// WithToolNamePrefix prefixes every wrapped tool's name (e.g. "fs_"), so tools
// from several servers can coexist without name collisions. The remote name used
// on the wire is unchanged.
func WithToolNamePrefix(prefix string) Option {
	return func(c *Client) { c.prefix = prefix }
}

// WithNotificationHandler registers a handler for server→client notifications
// (e.g. "notifications/message" logging, "notifications/tools/list_changed").
func WithNotificationHandler(fn func(method string, params json.RawMessage)) Option {
	return func(c *Client) { c.rc.onNotify = fn }
}

// Client is a Model Context Protocol client bound to one server. It performs the
// handshake, lists tools, and exposes them as [rimeno.Tool]s a rimeno agent can call.
type Client struct {
	rc       *rpcClient
	info     Implementation
	protocol string
	prefix   string
	onClose  func() error

	mu        sync.Mutex
	server    *ServerInfo
	connected bool
}

// NewClient builds a client over an arbitrary transport: it reads server messages
// from r and writes client messages to w. The caller owns r and w; Close does not
// close them (use [NewStdioClient] to also manage a subprocess). Options that set
// a notification handler must be passed here (before the reader starts).
func NewClient(r io.Reader, w io.Writer, opts ...Option) *Client {
	rc := &rpcClient{
		enc:     json.NewEncoder(w),
		pending: make(map[uint64]chan rpcMessage),
		done:    make(chan struct{}),
	}
	c := &Client{rc: rc, info: Implementation{Name: "rimeno", Version: "0.1.0"}, protocol: DefaultProtocolVersion}
	for _, o := range opts {
		o(c)
	}
	rc.start(r)
	return c
}

// NewStdioClient starts command as a subprocess and speaks MCP over its stdio,
// the standard MCP transport. Close terminates the process.
func NewStdioClient(command string, args []string, opts ...Option) (*Client, error) {
	// The caller chooses which MCP server binary to launch; running that
	// command is the entire purpose of the stdio transport.
	return NewStdioCmdClient(exec.Command(command, args...), opts...) // #nosec G204
}

// NewStdioCmdClient is like [NewStdioClient] but takes a pre-configured
// *exec.Cmd, so the caller can set Env, Dir, etc. It wires the command's stdio,
// starts it, and returns a client whose Close closes stdin and waits for exit.
func NewStdioCmdClient(cmd *exec.Cmd, opts ...Option) (*Client, error) {
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if cmd.Stderr == nil {
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	c := NewClient(stdout, stdin, opts...)
	c.onClose = func() error {
		_ = stdin.Close() // EOF signals the server to exit
		done := make(chan error, 1)
		go func() { done <- cmd.Wait() }()
		select {
		case err := <-done:
			return err
		case <-time.After(3 * time.Second):
			_ = cmd.Process.Kill()
			<-done
			return fmt.Errorf("mcp: server did not exit; killed")
		}
	}
	return c, nil
}

type initializeParams struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    struct{}       `json:"capabilities"`
	ClientInfo      Implementation `json:"clientInfo"`
}

// Connect performs the initialize handshake and sends the initialized
// notification. It must be called once before [Client.ListTools]/[Client.Tools].
func (c *Client) Connect(ctx context.Context) (*ServerInfo, error) {
	raw, err := c.rc.call(ctx, "initialize", initializeParams{
		ProtocolVersion: c.protocol,
		ClientInfo:      c.info,
	})
	if err != nil {
		return nil, fmt.Errorf("mcp: initialize: %w", err)
	}
	var si ServerInfo
	if err := json.Unmarshal(raw, &si); err != nil {
		return nil, fmt.Errorf("mcp: decode initialize result: %w", err)
	}
	if err := c.rc.notify("notifications/initialized", struct{}{}); err != nil {
		return nil, fmt.Errorf("mcp: initialized notification: %w", err)
	}
	c.mu.Lock()
	c.server, c.connected = &si, true
	c.mu.Unlock()
	return &si, nil
}

type listToolsResult struct {
	Tools      []ToolInfo `json:"tools"`
	NextCursor string     `json:"nextCursor,omitempty"`
}

// ListTools returns every tool the server advertises, following pagination.
func (c *Client) ListTools(ctx context.Context) ([]ToolInfo, error) {
	var all []ToolInfo
	cursor := ""
	for {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}
		raw, err := c.rc.call(ctx, "tools/list", params)
		if err != nil {
			return nil, fmt.Errorf("mcp: tools/list: %w", err)
		}
		var res listToolsResult
		if err := json.Unmarshal(raw, &res); err != nil {
			return nil, fmt.Errorf("mcp: decode tools/list: %w", err)
		}
		all = append(all, res.Tools...)
		if res.NextCursor == "" {
			break
		}
		cursor = res.NextCursor
	}
	return all, nil
}

// Tools returns every server tool wrapped as a [rimeno.Tool], ready to drop into a
// rimeno.Config. A tool call becomes an MCP tools/call request under the hood.
func (c *Client) Tools(ctx context.Context) ([]rimeno.Tool, error) {
	infos, err := c.ListTools(ctx)
	if err != nil {
		return nil, err
	}
	tools := make([]rimeno.Tool, 0, len(infos))
	for _, ti := range infos {
		tools = append(tools, c.wrap(ti))
	}
	return tools, nil
}

func (c *Client) wrap(ti ToolInfo) rimeno.Tool {
	schema := ti.InputSchema
	if len(schema) == 0 {
		schema = json.RawMessage(`{"type":"object"}`)
	}
	remote := ti.Name
	return rimeno.RawTool(c.prefix+ti.Name, ti.Description, schema,
		func(ctx context.Context, args json.RawMessage) (any, error) {
			return c.CallTool(ctx, remote, args)
		})
}

type callToolParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type contentBlock struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

type callToolResult struct {
	Content []contentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

// CallTool invokes a remote tool by its server-side name and returns its content
// rendered to text. A tool that reports isError is returned as a Go error, which
// the rimeno run loop feeds back to the model as a recoverable tool result.
func (c *Client) CallTool(ctx context.Context, name string, args json.RawMessage) (string, error) {
	if len(args) == 0 || string(args) == "null" {
		args = json.RawMessage("{}")
	}
	raw, err := c.rc.call(ctx, "tools/call", callToolParams{Name: name, Arguments: args})
	if err != nil {
		return "", fmt.Errorf("mcp: tools/call %q: %w", name, err)
	}
	var res callToolResult
	if err := json.Unmarshal(raw, &res); err != nil {
		return "", fmt.Errorf("mcp: decode tools/call result: %w", err)
	}
	text := renderContent(res.Content)
	if res.IsError {
		return "", fmt.Errorf("tool %q reported an error: %s", name, text)
	}
	return text, nil
}

// Ping issues an MCP ping (liveness check).
func (c *Client) Ping(ctx context.Context) error {
	_, err := c.rc.call(ctx, "ping", struct{}{})
	return err
}

// ServerInfo returns the server's handshake info, or nil before Connect.
func (c *Client) ServerInfo() *ServerInfo {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.server
}

// Close shuts down the client, waking any in-flight calls, and (for a stdio
// client) terminates the subprocess.
func (c *Client) Close() error {
	c.rc.close()
	if c.onClose != nil {
		return c.onClose()
	}
	return nil
}

// renderContent flattens MCP content blocks into a single string for the model.
// Text passes through; non-text blocks are summarized with a typed placeholder.
func renderContent(blocks []contentBlock) string {
	var sb strings.Builder
	for i, b := range blocks {
		if i > 0 {
			sb.WriteByte('\n')
		}
		switch b.Type {
		case "text":
			sb.WriteString(b.Text)
		case "image", "audio":
			fmt.Fprintf(&sb, "[%s %s]", b.Type, b.MimeType)
		default:
			fmt.Fprintf(&sb, "[%s]", b.Type)
		}
	}
	return sb.String()
}
