// Package jev is a client for TypeSafe's jev Decisions API, an
// OpenRouter-hosted classifier (e.g. "typesafe/jev-1.13"). Unlike a chat model,
// jev takes a piece of text (the "state") plus a set of typed questions and
// returns structured answers with probabilities and confidence — it does not
// speak the OpenAI Chat Completions wire format, so it is a standalone client
// rather than a rimeno.Model.
//
// Point it at the Decisions base URL and supply an OpenRouter API key:
//
//	client := jev.New(
//	    jev.WithBaseURL("https://openrouter.ai/api/alpha"),
//	    jev.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
//	    jev.WithModel("typesafe/jev-1.13"),
//	)
//	res, err := client.Decide(ctx, jev.Request{
//	    State: "Help! My payouts have been failing for 3 days.",
//	    Questions: map[string]jev.Question{
//	        "is_urgent":   jev.Noul("Does this convey urgency?", "Time-sensitive", "Not urgent"),
//	        "department":  jev.Choice("Which team?", map[string]string{"billing": "...", "technical": "..."}),
//	        "frustration": jev.Score("How frustrated?", "Calm", "Frustrated", "Very angry"),
//	    },
//	})
//	res.Choice("department")   // -> Choice{Value: "billing", Confidence: 0.81, ...}
package jev

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks to the jev Decisions API.
type Client struct {
	baseURL    string
	apiKey     string
	model      string
	headers    map[string]string
	httpClient *http.Client
	maxRetries int
	backoff    time.Duration
}

// Option configures a Client.
type Option func(*Client)

// WithBaseURL sets the Decisions API base URL, e.g.
// "https://openrouter.ai/api/alpha". The client appends "/decisions". Required —
// there is no default host.
func WithBaseURL(u string) Option { return func(c *Client) { c.baseURL = strings.TrimRight(u, "/") } }

// WithAPIKey sets the bearer token (an OpenRouter "sk-or-v..." key).
func WithAPIKey(k string) Option { return func(c *Client) { c.apiKey = k } }

// WithModel sets the model id used for requests, e.g. "typesafe/jev-1.13".
func WithModel(m string) Option { return func(c *Client) { c.model = m } }

// WithHTTPClient supplies a custom *http.Client (timeouts, proxies, etc.).
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.httpClient = h } }

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
// (default 2).
func WithMaxRetries(n int) Option { return func(c *Client) { c.maxRetries = n } }

// New constructs a Client. Set WithBaseURL, WithAPIKey, and WithModel for a
// hosted provider.
func New(opts ...Option) *Client {
	c := &Client{
		httpClient: &http.Client{Timeout: 120 * time.Second},
		maxRetries: 2,
		backoff:    300 * time.Millisecond,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Model reports the configured model id.
func (c *Client) Model() string { return c.model }

func (c *Client) endpoint() string { return c.baseURL + "/decisions" }

// Decide submits a decision request and returns the structured answers. The
// request's model defaults to the client's WithModel when unset.
func (c *Client) Decide(ctx context.Context, req Request) (*Response, error) {
	if len(req.Questions) == 0 {
		return nil, fmt.Errorf("jev: request has no questions")
	}
	wire := wireRequest{
		Model:     firstNonEmpty(req.Model, c.model),
		State:     req.State,
		Questions: req.Questions,
	}
	body, err := json.Marshal(wire)
	if err != nil {
		return nil, fmt.Errorf("jev: encode request: %w", err)
	}
	raw, err := c.do(ctx, body)
	if err != nil {
		return nil, err
	}
	var res Response
	if err := json.Unmarshal(raw, &res); err != nil {
		return nil, fmt.Errorf("jev: decode response: %w", err)
	}
	res.Raw = raw
	return &res, nil
}

func (c *Client) do(ctx context.Context, body []byte) ([]byte, error) {
	if c.baseURL == "" {
		return nil, fmt.Errorf("jev: no base URL configured (use WithBaseURL)")
	}
	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(c.backoff * time.Duration(1<<(attempt-1))):
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
			// A 2xx whose body could not be fully read or is not valid JSON is a
			// transient failure under load — retry rather than surface a decode error.
			if readErr != nil {
				lastErr = fmt.Errorf("jev: read response body: %w", readErr)
				continue
			}
			if !json.Valid(raw) {
				lastErr = fmt.Errorf("jev: invalid/empty JSON in 2xx response (%d bytes)", len(raw))
				continue
			}
			return raw, nil
		}
		apiErr := parseAPIError(resp.StatusCode, raw)
		if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500 {
			lastErr = apiErr
			continue
		}
		return nil, apiErr
	}
	return nil, lastErr
}

func (c *Client) setHeaders(r *http.Request) {
	r.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		r.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	for k, v := range c.headers {
		r.Header.Set(k, v)
	}
}

// APIError is a non-2xx response from the Decisions API.
type APIError struct {
	StatusCode int
	Message    string
	Raw        string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("jev: http %d: %s", e.StatusCode, e.Message)
}

func parseAPIError(status int, body []byte) *APIError {
	var env struct {
		Error struct {
			Message string `json:"message"`
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
	return &APIError{StatusCode: status, Message: msg, Raw: string(body)}
}

func firstNonEmpty(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
