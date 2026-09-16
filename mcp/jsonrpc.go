package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// rpcError is a JSON-RPC 2.0 error object.
type rpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e *rpcError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if len(e.Data) > 0 {
		return e.Message + " (" + string(e.Data) + ")"
	}
	return e.Message
}

// rpcMessage is a decoded incoming line: it may be a response (has Result/Error),
// a server→client notification (Method, no ID), or a server→client request
// (Method and ID).
type rpcMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      uint64 `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

// rpcClient is a minimal JSON-RPC 2.0 client over a line-delimited byte stream. A
// single background goroutine reads responses and routes them to the caller
// waiting on the matching id.
type rpcClient struct {
	enc *json.Encoder
	wmu sync.Mutex // serializes writes

	nextID  atomic.Uint64
	mu      sync.Mutex
	pending map[uint64]chan rpcMessage

	onNotify func(method string, params json.RawMessage)

	done      chan struct{}
	closeOnce sync.Once
	readErr   atomic.Value // error
}

func (c *rpcClient) start(r io.Reader) { go c.readLoop(r) }

func (c *rpcClient) readLoop(r io.Reader) {
	br := bufio.NewReader(r)
	for {
		line, err := br.ReadBytes('\n')
		if t := bytes.TrimSpace(line); len(t) > 0 {
			var msg rpcMessage
			if e := json.Unmarshal(t, &msg); e == nil {
				c.dispatch(msg)
			}
		}
		if err != nil {
			c.fail(err)
			return
		}
	}
}

func (c *rpcClient) dispatch(msg rpcMessage) {
	// A server→client request (both method and id) — we advertise no capabilities
	// that require handling one, so decline politely instead of hanging the server.
	if msg.Method != "" && len(msg.ID) > 0 {
		// Best-effort decline; if the pipe is already gone there is nothing
		// useful to do with the error here.
		_ = c.write(map[string]any{
			"jsonrpc": "2.0",
			"id":      msg.ID,
			"error":   map[string]any{"code": -32601, "message": "method not found"},
		})
		return
	}
	// A notification (method, no id). The handler runs in this reader goroutine,
	// so recover a panic in it — the caller cannot wrap this call site itself.
	if msg.Method != "" {
		if c.onNotify != nil {
			func() {
				defer func() { _ = recover() }()
				c.onNotify(msg.Method, msg.Params)
			}()
		}
		return
	}
	// Otherwise a response: route by id.
	id, err := strconv.ParseUint(strings.Trim(string(msg.ID), `"`), 10, 64)
	if err != nil {
		return
	}
	c.mu.Lock()
	ch := c.pending[id]
	delete(c.pending, id)
	c.mu.Unlock()
	if ch != nil {
		ch <- msg
	}
}

// call sends a request and waits for the response, ctx, or connection close.
func (c *rpcClient) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := c.nextID.Add(1)
	ch := make(chan rpcMessage, 1)
	c.mu.Lock()
	c.pending[id] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	if err := c.write(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params}); err != nil {
		return nil, err
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.done:
		if e, ok := c.readErr.Load().(error); ok && e != nil && !errors.Is(e, io.EOF) {
			return nil, e
		}
		return nil, errors.New("mcp: connection closed")
	case msg := <-ch:
		if msg.Error != nil {
			return nil, msg.Error
		}
		return msg.Result, nil
	}
}

func (c *rpcClient) notify(method string, params any) error {
	return c.write(rpcNotification{JSONRPC: "2.0", Method: method, Params: params})
}

func (c *rpcClient) write(v any) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	return c.enc.Encode(v) // Encode appends '\n' → line-delimited
}

// fail marks the connection closed with an error, waking any pending callers.
func (c *rpcClient) fail(err error) {
	c.closeOnce.Do(func() {
		if err != nil {
			c.readErr.Store(err)
		}
		close(c.done)
	})
}

// close marks the connection closed without an error.
func (c *rpcClient) close() { c.closeOnce.Do(func() { close(c.done) }) }
