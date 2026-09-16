package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/matiasinsaurralde/rimeno"
)

// Stream implements rimeno.StreamingModel using server-sent events. It yields text
// deltas as they arrive and assembles the terminal rimeno.Response (including any
// tool calls and usage) into the final chunk's Final field.
func (c *Client) Stream(ctx context.Context, req *rimeno.Request) (rimeno.Stream, error) {
	if c.useResponses {
		return c.streamResponses(ctx, req)
	}
	body, err := json.Marshal(c.buildChatRequest(req, true))
	if err != nil {
		return nil, fmt.Errorf("openai: encode request: %w", err)
	}
	rc, sc, err := c.openSSE(ctx, body)
	if err != nil {
		return nil, err
	}
	return &stream{
		rc:        rc,
		sc:        sc,
		client:    c,
		toolCalls: map[int]*rimeno.ToolCall{},
	}, nil
}

// openSSE POSTs body and returns a line scanner over the server-sent-event stream.
// Shared by the Chat Completions and Responses streaming paths.
func (c *Client) openSSE(ctx context.Context, body []byte) (io.ReadCloser, *bufio.Scanner, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint(), strings.NewReader(string(body)))
	if err != nil {
		return nil, nil, err
	}
	c.setHeaders(httpReq)
	httpReq.Header.Set("Accept", "text/event-stream")
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, nil, err
	}
	if resp.StatusCode/100 != 2 {
		raw, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		return nil, nil, parseAPIError(resp.StatusCode, raw)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	return resp.Body, sc, nil
}

type stream struct {
	rc        io.ReadCloser
	sc        *bufio.Scanner
	client    *Client
	text      strings.Builder
	reasoning strings.Builder // reasoning-channel deltas, used as a fallback (see finalChunk)
	toolCalls map[int]*rimeno.ToolCall
	order     []int
	finish    string
	usage     *wireUsage
	done      bool
}

// Recv returns the next chunk. Text arrives incrementally in TextDelta; the
// terminal chunk carries Final (the assembled rimeno.Response). io.EOF signals the
// end of the stream.
func (s *stream) Recv() (rimeno.StreamChunk, error) {
	for s.sc.Scan() {
		line := strings.TrimSpace(s.sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return s.finalChunk()
		}
		var sr streamResponse
		if err := json.Unmarshal([]byte(data), &sr); err != nil {
			continue // ignore keep-alives / unparseable frames
		}
		var chunk rimeno.StreamChunk
		emit := false
		if sr.Usage != nil {
			s.usage = sr.Usage
			u := s.client.toUsage(sr.Usage)
			chunk.Usage = &u
			emit = true
		}
		if len(sr.Choices) > 0 {
			ch := sr.Choices[0]
			if ch.FinishReason != "" {
				s.finish = ch.FinishReason
			}
			if ch.Delta.Content != "" {
				s.text.WriteString(ch.Delta.Content)
				chunk.TextDelta = ch.Delta.Content
				emit = true
			}
			// Reasoning-channel deltas are surfaced separately (ReasoningDelta →
			// ReasoningDeltaEvent) so a host can render a "thinking…" pane, and
			// accumulated for the final Message.Reasoning and the content fallback.
			if rd := firstNonEmpty(ch.Delta.Reasoning, ch.Delta.ReasoningContent); rd != "" {
				s.reasoning.WriteString(rd)
				chunk.ReasoningDelta = rd
				emit = true
			}
			for _, tc := range ch.Delta.ToolCalls {
				acc := s.toolCalls[tc.Index]
				if acc == nil {
					acc = &rimeno.ToolCall{}
					s.toolCalls[tc.Index] = acc
					s.order = append(s.order, tc.Index)
				}
				if tc.ID != "" {
					acc.ID = tc.ID
				}
				if tc.Function.Name != "" {
					acc.Name = tc.Function.Name
				}
				if tc.Function.Arguments != "" {
					acc.Arguments = append(acc.Arguments, tc.Function.Arguments...)
				}
			}
		}
		if emit {
			return chunk, nil
		}
	}
	if err := s.sc.Err(); err != nil {
		return rimeno.StreamChunk{}, err
	}
	return s.finalChunk()
}

func (s *stream) finalChunk() (rimeno.StreamChunk, error) {
	if s.done {
		return rimeno.StreamChunk{}, io.EOF
	}
	s.done = true
	// Fall back to the reasoning channel when no content streamed and there are no
	// tool calls — mirrors the non-streaming assistantText behavior for reasoning
	// models that return the answer as reasoning with content:null.
	text := s.text.String()
	if text == "" && len(s.order) == 0 {
		text = s.reasoning.String()
	}
	msg := rimeno.Message{Role: rimeno.RoleAssistant, Text: text, Reasoning: s.reasoning.String()}
	for _, idx := range s.order {
		msg.ToolCalls = append(msg.ToolCalls, *s.toolCalls[idx])
	}
	usage := s.client.toUsage(s.usage)
	resp := &rimeno.Response{
		Message:    msg,
		Usage:      usage,
		StopReason: mapStop(s.finish),
		Model:      s.client.model,
	}
	return rimeno.StreamChunk{Final: resp, Usage: &usage}, nil
}

func (s *stream) Close() error { return s.rc.Close() }
