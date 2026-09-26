// Package openai implements a rimeno.Model backed by the OpenAI-compatible
// Chat Completions API (/v1/chat/completions). It works against OpenAI itself and
// any compatible endpoint (OpenRouter, Ollama, llama.cpp,
// vLLM, ...): point it at the base URL and supply the model id.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand/v2"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/matiasinsaurralde/rimeno"
)

// DefaultBaseURL is the OpenAI API base (including the /v1 suffix).
const DefaultBaseURL = "https://api.openai.com/v1"

// Price is per-million-token pricing used to compute Usage.CostUSD.
type Price struct {
	InputPerMillion  float64
	OutputPerMillion float64
}

// Backoff configures the retry schedule used by (*Client).do for retriable
// responses (429, 5xx, network errors, truncated 2xx).
//
// With Jitter set, waits follow AWS-style "decorrelated jitter": each wait is
// drawn uniformly from [Base, prev*3] and clamped to Cap, where prev is the
// previous wait (seeded from Base). This spreads concurrent retries so callers
// fanning out many requests to one endpoint do not retry in lockstep and
// re-saturate it. With Jitter unset, waits follow a plain exponential schedule
// (Base, Base*2, Base*4, ...) clamped to Cap.
type Backoff struct {
	Base       time.Duration // floor for the first retry wait
	Cap        time.Duration // ceiling for any single wait
	MaxRetries int           // retries after the initial attempt (total attempts = MaxRetries+1)
	Jitter     bool          // use decorrelated jitter instead of plain exponential
}

// Default backoff parameters. Chosen for endpoints that saturate under high
// concurrency: several jittered attempts spread over tens of seconds recover
// where a few tight attempts do not.
const (
	defaultBackoffBase = 500 * time.Millisecond
	defaultBackoffCap  = 30 * time.Second
	defaultMaxRetries  = 4
)

// Client is an OpenAI-compatible rimeno.Model (and rimeno.StreamingModel).
type Client struct {
	baseURL      string
	apiKey       string
	model        string
	org          string
	headers      map[string]string
	httpClient   *http.Client
	backoff      Backoff
	sem          chan struct{} // optional in-flight request cap; nil = unlimited
	prices       map[string]Price
	useResponses bool // speak /responses instead of /chat/completions
}

// Option configures a Client.
type Option func(*Client)

// WithBaseURL sets the API base URL (including /v1). Defaults to DefaultBaseURL.
func WithBaseURL(u string) Option { return func(c *Client) { c.baseURL = strings.TrimRight(u, "/") } }

// WithAPIKey sets the bearer token.
func WithAPIKey(k string) Option { return func(c *Client) { c.apiKey = k } }

// WithModel sets the model id used for requests and reported by ID.
func WithModel(m string) Option { return func(c *Client) { c.model = m } }

// WithHTTPClient supplies a custom *http.Client (timeouts, proxies, etc.).
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.httpClient = h } }

// WithOrganization sets the OpenAI-Organization header.
func WithOrganization(org string) Option { return func(c *Client) { c.org = org } }

// WithHeader adds an extra HTTP header to every request.
func WithHeader(k, v string) Option {
	return func(c *Client) {
		if c.headers == nil {
			c.headers = map[string]string{}
		}
		c.headers[k] = v
	}
}

// WithMaxRetries sets how many times to retry on 429/5xx and network errors
// (default 4). It is a shorthand for setting only Backoff.MaxRetries; use
// WithBackoff to tune the wait schedule as well.
func WithMaxRetries(n int) Option { return func(c *Client) { c.backoff.MaxRetries = n } }

// WithBackoff sets the retry schedule. Any zero-valued field is left at its
// default (Base 500ms, Cap 30s, MaxRetries 4, Jitter on), so callers can
// override only the fields they care about.
func WithBackoff(b Backoff) Option {
	return func(c *Client) {
		if b.Base > 0 {
			c.backoff.Base = b.Base
		}
		if b.Cap > 0 {
			c.backoff.Cap = b.Cap
		}
		if b.MaxRetries > 0 {
			c.backoff.MaxRetries = b.MaxRetries
		}
		c.backoff.Jitter = b.Jitter
	}
}

// WithMaxConcurrency caps the number of requests this Client keeps in flight to
// the endpoint at once, regardless of how many goroutines call Generate. A
// request holds one slot for the whole of its retry sequence. k <= 0 disables
// the cap (the default).
func WithMaxConcurrency(k int) Option {
	return func(c *Client) {
		if k > 0 {
			c.sem = make(chan struct{}, k)
		} else {
			c.sem = nil
		}
	}
}

// WithPricing registers per-million-token pricing for a model so Usage.CostUSD is
// populated so traces can report cost.
func WithPricing(model string, p Price) Option {
	return func(c *Client) {
		if c.prices == nil {
			c.prices = map[string]Price{}
		}
		c.prices[model] = p
	}
}

// New constructs a Client. At minimum set WithModel; WithAPIKey and WithBaseURL
// are required for hosted providers.
func New(opts ...Option) *Client {
	c := &Client{
		baseURL:    DefaultBaseURL,
		httpClient: &http.Client{Timeout: 120 * time.Second},
		backoff: Backoff{
			Base:       defaultBackoffBase,
			Cap:        defaultBackoffCap,
			MaxRetries: defaultMaxRetries,
			Jitter:     true,
		},
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// ID implements rimeno.Model.
func (c *Client) ID() string { return c.model }

func (c *Client) endpoint() string {
	if c.useResponses {
		return c.baseURL + "/responses"
	}
	return c.baseURL + "/chat/completions"
}

// Generate implements rimeno.Model.
func (c *Client) Generate(ctx context.Context, req *rimeno.Request) (*rimeno.Response, error) {
	if c.useResponses {
		return c.generateResponses(ctx, req)
	}
	body, err := json.Marshal(c.buildChatRequest(req, false))
	if err != nil {
		return nil, fmt.Errorf("openai: encode request: %w", err)
	}
	raw, err := c.do(ctx, body)
	if err != nil {
		return nil, err
	}
	var cr chatResponse
	if err := json.Unmarshal(raw, &cr); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}
	if len(cr.Choices) == 0 {
		return nil, fmt.Errorf("openai: response contained no choices")
	}
	choice := cr.Choices[0]
	msg := rimeno.Message{
		Role:      rimeno.RoleAssistant,
		Text:      assistantText(choice.Message),
		Reasoning: firstNonEmpty(choice.Message.Reasoning, choice.Message.ReasoningContent),
	}
	for _, tc := range choice.Message.ToolCalls {
		msg.ToolCalls = append(msg.ToolCalls, rimeno.ToolCall{
			ID:        tc.ID,
			Name:      tc.Function.Name,
			Arguments: json.RawMessage(tc.Function.Arguments),
		})
	}
	return &rimeno.Response{
		Message:    msg,
		Usage:      c.toUsage(cr.Usage),
		StopReason: mapStop(choice.FinishReason),
		Model:      firstNonEmpty(cr.Model, c.model),
		Raw:        raw,
	}, nil
}

func (c *Client) buildChatRequest(req *rimeno.Request, stream bool) *chatRequest {
	model := req.Model
	if model == "" {
		model = c.model
	}
	out := &chatRequest{
		Model:       model,
		Messages:    toWireMessages(req.Messages),
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		Stream:      stream,
	}
	if stream {
		out.StreamOptions = &streamOptions{IncludeUsage: true}
	}
	for _, t := range req.Tools {
		out.Tools = append(out.Tools, wireTool{
			Type: "function",
			Function: wireFunctionDef{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.Parameters,
			},
		})
	}
	if req.ToolChoice != "" && len(req.Tools) > 0 {
		out.ToolChoice = string(req.ToolChoice)
	}
	if req.ResponseFormat != nil {
		out.ResponseFormat = &wireResponseFormat{
			Type: "json_schema",
			JSONSchema: wireJSONSchema{
				Name:   sanitizeName(req.ResponseFormat.Name),
				Schema: req.ResponseFormat.Schema,
				Strict: req.ResponseFormat.Strict,
			},
		}
	}
	if req.ReasoningEffort != "" || req.ReasoningMaxTokens > 0 {
		out.Reasoning = &wireReasoning{
			Effort:    req.ReasoningEffort,
			MaxTokens: req.ReasoningMaxTokens,
		}
	}
	return out
}

func (c *Client) do(ctx context.Context, body []byte) ([]byte, error) {
	// Hold one concurrency slot for the whole retry sequence, so a burst of
	// callers never puts more than k requests in flight to the endpoint.
	if c.sem != nil {
		select {
		case c.sem <- struct{}{}:
			defer func() { <-c.sem }()
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	var lastErr error
	prev := c.backoff.Base // running wait, seeded for decorrelated jitter
	// override, when > 0, is the wait to use before the next attempt (e.g. a
	// 429's Retry-After hint) in place of the computed backoff.
	var override time.Duration
	for attempt := 0; attempt <= c.backoff.MaxRetries; attempt++ {
		if attempt > 0 {
			wait := c.nextWait(&prev, override)
			override = 0
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(wait):
			}
		}
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint(), bytes.NewReader(body))
		if err != nil {
			return nil, err
		}
		c.setHeaders(httpReq)
		resp, err := c.httpClient.Do(httpReq)
		if err != nil {
			lastErr = err
			continue // network errors are retriable
		}
		raw, readErr := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode/100 == 2 {
			// A 2xx whose body could not be fully read (connection cut mid-stream) or
			// that is empty/not valid JSON is a transient failure — some providers
			// return truncated or empty 200s under load. Retry rather than surfacing
			// "unexpected end of JSON input" to the caller.
			if readErr != nil {
				lastErr = fmt.Errorf("openai: read response body: %w", readErr)
				continue
			}
			if !json.Valid(raw) {
				lastErr = fmt.Errorf("openai: invalid/empty JSON in 2xx response (%d bytes)", len(raw))
				continue
			}
			return raw, nil
		}
		apiErr := parseAPIError(resp.StatusCode, raw)
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = apiErr
			if resp.StatusCode == http.StatusTooManyRequests {
				// Honor the provider's own wait hint over our schedule when it
				// gives one; clamped to Cap so a hostile/huge value can't stall us.
				if d := parseRetryAfter(resp.Header.Get("Retry-After")); d > 0 {
					if d > c.backoff.Cap {
						d = c.backoff.Cap
					}
					override = d
				}
			}
			continue
		}
		return nil, apiErr
	}
	return nil, lastErr
}

// nextWait returns the duration to sleep before the next retry, advancing prev.
// A positive override (e.g. a Retry-After hint) wins. Otherwise the wait is
// decorrelated jitter when enabled — random in [Base, prev*3], capped — or a
// plain doubling of prev, capped.
func (c *Client) nextWait(prev *time.Duration, override time.Duration) time.Duration {
	if override > 0 {
		return override
	}
	base, capD := c.backoff.Base, c.backoff.Cap
	var wait time.Duration
	if c.backoff.Jitter {
		// random in [base, prev*3]
		span := int64(*prev)*3 - int64(base) + 1
		if span < 1 {
			span = 1
		}
		// #nosec G404 -- jitter spreads retry timing, not a security context; crypto/rand is unwarranted here
		wait = base + time.Duration(rand.Int64N(span))
	} else if *prev <= base {
		// first retry: floor at base, then double each subsequent retry.
		wait = base
	} else {
		wait = *prev * 2
	}
	if capD > 0 && wait > capD {
		wait = capD
	}
	*prev = wait
	return wait
}

// parseRetryAfter parses an HTTP Retry-After header value, which is either a
// non-negative number of seconds or an HTTP-date. It returns the delay, or 0 if
// the value is absent, malformed, or in the past.
func parseRetryAfter(v string) time.Duration {
	v = strings.TrimSpace(v)
	if v == "" {
		return 0
	}
	if secs, err := strconv.Atoi(v); err == nil {
		if secs < 0 {
			return 0
		}
		return time.Duration(secs) * time.Second
	}
	if t, err := http.ParseTime(v); err == nil {
		if d := time.Until(t); d > 0 {
			return d
		}
	}
	return 0
}

func (c *Client) setHeaders(r *http.Request) {
	r.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		r.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	if c.org != "" {
		r.Header.Set("OpenAI-Organization", c.org)
	}
	for k, v := range c.headers {
		r.Header.Set(k, v)
	}
}

func (c *Client) toUsage(u *wireUsage) rimeno.Usage {
	if u == nil {
		return rimeno.Usage{}
	}
	out := rimeno.Usage{
		InputTokens:  u.PromptTokens,
		OutputTokens: u.CompletionTokens,
		TotalTokens:  u.TotalTokens,
		CostUSD:      c.cost(*u),
	}
	if u.PromptTokensDetails != nil {
		out.CachedInputTokens = u.PromptTokensDetails.CachedTokens
	}
	if u.CompletionTokensDetails != nil {
		out.ReasoningTokens = u.CompletionTokensDetails.ReasoningTokens
	}
	return out
}

func (c *Client) cost(u wireUsage) float64 {
	p, ok := c.prices[c.model]
	if !ok {
		return 0
	}
	return float64(u.PromptTokens)/1e6*p.InputPerMillion + float64(u.CompletionTokens)/1e6*p.OutputPerMillion
}

// APIError is a non-2xx response from the provider.
type APIError struct {
	StatusCode int
	Type       string
	Message    string
	Raw        string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("openai: http %d: %s", e.StatusCode, e.Message)
}

func parseAPIError(status int, body []byte) *APIError {
	var env struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	_ = json.Unmarshal(body, &env)
	msg := env.Error.Message
	if msg == "" {
		msg = strings.TrimSpace(string(body))
	}
	if msg == "" {
		msg = http.StatusText(status)
	}
	return &APIError{StatusCode: status, Type: env.Error.Type, Message: msg, Raw: string(body)}
}

func mapStop(finish string) rimeno.StopReason {
	switch finish {
	case "stop":
		return rimeno.StopReasonStop
	case "tool_calls", "function_call":
		return rimeno.StopReasonToolCalls
	case "length":
		return rimeno.StopReasonLength
	case "content_filter":
		return rimeno.StopReasonContentFilter
	default:
		return rimeno.StopReasonStop
	}
}

func toWireMessages(msgs []rimeno.Message) []wireMessage {
	out := make([]wireMessage, 0, len(msgs))
	for _, m := range msgs {
		wm := wireMessage{Role: string(m.Role), Content: m.Text}
		switch m.Role {
		case rimeno.RoleTool:
			wm.ToolCallID = m.ToolCallID
			wm.Name = m.Name
		case rimeno.RoleAssistant:
			for _, tc := range m.ToolCalls {
				wm.ToolCalls = append(wm.ToolCalls, wireToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: wireFunctionCall{
						Name:      tc.Name,
						Arguments: string(tc.Arguments),
					},
				})
			}
		}
		out = append(out, wm)
	}
	return out
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}

// assistantText extracts the assistant's answer text. It prefers content, but
// falls back to the reasoning channel when content is empty and there are no tool
// calls — some reasoning models (under json_schema response_format)
// return the answer in reasoning/reasoning_content with content:null, and without
// this fallback rimeno would see an empty message and lose the answer. When content
// is present, or the turn is a tool-call turn (where empty content is normal and
// reasoning is chain-of-thought, not the answer), reasoning is ignored.
func assistantText(m wireMessage) string {
	if m.Content != "" || len(m.ToolCalls) > 0 {
		return m.Content
	}
	return firstNonEmpty(m.Reasoning, m.ReasoningContent)
}

// sanitizeName coerces a schema name into the [a-zA-Z0-9_-] set OpenAI accepts.
func sanitizeName(name string) string {
	if name == "" {
		return "output"
	}
	var b strings.Builder
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '_', r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}
