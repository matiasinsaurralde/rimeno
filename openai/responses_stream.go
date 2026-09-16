package openai

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/matiasinsaurralde/rimeno"
)

// streamResponses streams the Responses API. Text and reasoning arrive as
// incremental deltas (TextDelta / ReasoningDelta → Text/ReasoningDeltaEvent); the
// terminal `response.completed` event carries the full output (text, reasoning,
// tool calls) and usage, which is assembled into the final chunk's Final field.
func (c *Client) streamResponses(ctx context.Context, req *rimeno.Request) (rimeno.Stream, error) {
	rr := c.buildRespRequest(req)
	rr.Stream = true
	body, err := json.Marshal(rr)
	if err != nil {
		return nil, fmt.Errorf("openai: encode responses request: %w", err)
	}
	rc, sc, err := c.openSSE(ctx, body)
	if err != nil {
		return nil, err
	}
	return &respStream{rc: rc, sc: sc, client: c}, nil
}

type respStream struct {
	rc     io.ReadCloser
	sc     *bufio.Scanner
	client *Client
	done   bool
}

// respStreamEvent is the subset of a Responses SSE data frame we consume.
type respStreamEvent struct {
	Type     string        `json:"type"`
	Delta    string        `json:"delta"`
	Response *respResponse `json:"response"`
}

func (s *respStream) Recv() (rimeno.StreamChunk, error) {
	for s.sc.Scan() {
		line := strings.TrimSpace(s.sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue // skip `event:` lines and blanks; the data frame carries its own type
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			return s.finalChunk(nil)
		}
		var ev respStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		switch {
		case strings.HasSuffix(ev.Type, "output_text.delta"):
			if ev.Delta != "" {
				return rimeno.StreamChunk{TextDelta: ev.Delta}, nil
			}
		case strings.Contains(ev.Type, "reasoning") && strings.HasSuffix(ev.Type, ".delta"):
			if ev.Delta != "" {
				return rimeno.StreamChunk{ReasoningDelta: ev.Delta}, nil
			}
		case ev.Type == "response.completed" || ev.Type == "response.incomplete":
			return s.finalChunk(ev.Response)
		case ev.Type == "error" || ev.Type == "response.failed":
			return rimeno.StreamChunk{}, fmt.Errorf("openai: responses stream error: %s", data)
		}
	}
	if err := s.sc.Err(); err != nil {
		return rimeno.StreamChunk{}, err
	}
	return s.finalChunk(nil)
}

func (s *respStream) finalChunk(rr *respResponse) (rimeno.StreamChunk, error) {
	if s.done {
		return rimeno.StreamChunk{}, io.EOF
	}
	s.done = true
	if rr == nil {
		rr = &respResponse{}
	}
	msg := parseRespMessage(*rr)
	usage := s.client.respUsage(rr.Usage)
	resp := &rimeno.Response{
		Message:    msg,
		Usage:      usage,
		StopReason: respStop(msg),
		Model:      firstNonEmpty(rr.Model, s.client.model),
		ID:         rr.ID,
	}
	return rimeno.StreamChunk{Final: resp, Usage: &usage}, nil
}

func (s *respStream) Close() error { return s.rc.Close() }
