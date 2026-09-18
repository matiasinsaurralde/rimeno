package jev_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/matiasinsaurralde/rimeno/jev"
)

// combinedResponse is the exact body observed from the live API for a request
// with one question of each type (see jev.md).
const combinedResponse = `{"model":"typesafe/jev-1.13-20260917","answers":{"is_urgent":{"type":"noul","noul":0.95},"department":{"type":"choice","choice":"billing","probabilities":{"technical":0.13,"billing":0.87,"sales":0},"confidence":0.8},"frustration":{"type":"score","score":1.05,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0,"1":0.95,"2":0.05},"confidence":0.93}},"usage":{"input_tokens":427,"output_tokens":73,"cost":0.000017934},"id":"gen-dec-1789771486-AwfThkBgk3n2dUreLq7z","provider":"TypeSafe"}`

func TestDecideDecodesAllTypes(t *testing.T) {
	var gotBody wireCapture
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/decisions" {
			t.Errorf("path = %q, want /decisions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth = %q, want Bearer test-key", got)
		}
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			t.Fatalf("request body not JSON: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, combinedResponse)
	}))
	defer srv.Close()

	c := jev.New(
		jev.WithBaseURL(srv.URL),
		jev.WithAPIKey("test-key"),
		jev.WithModel("typesafe/jev-1.13"),
	)
	res, err := c.Decide(context.Background(), jev.Request{
		State: "Help! My payouts have been failing for 3 days.",
		Questions: map[string]jev.Question{
			"is_urgent":   jev.Noul("Does this convey urgency?", "Time-sensitive", "Not urgent"),
			"department":  jev.Choice("Which team?", map[string]string{"billing": "b", "technical": "t", "sales": "s"}),
			"frustration": jev.Score("How frustrated?", "Calm", "Frustrated", "Very angry"),
		},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}

	// Request wiring: model defaulted from the client, criteria polymorphism.
	if gotBody.Model != "typesafe/jev-1.13" {
		t.Errorf("sent model = %q", gotBody.Model)
	}
	if string(gotBody.Questions["frustration"].Criteria) != `["Calm","Frustrated","Very angry"]` {
		t.Errorf("score criteria not an array: %s", gotBody.Questions["frustration"].Criteria)
	}
	if string(gotBody.Questions["is_urgent"].Criteria) != `{"false":"Not urgent","true":"Time-sensitive"}` {
		t.Errorf("noul criteria not an object: %s", gotBody.Questions["is_urgent"].Criteria)
	}

	// noul
	if v, ok := res.Noul("is_urgent"); !ok || v != 0.95 {
		t.Errorf("Noul = %v, %v; want 0.95, true", v, ok)
	}
	// choice
	ch, ok := res.Choice("department")
	if !ok || ch.Value != "billing" || ch.Confidence != 0.8 || ch.Probabilities["billing"] != 0.87 {
		t.Errorf("Choice = %+v, %v", ch, ok)
	}
	// score
	sc, ok := res.Score("frustration")
	if !ok || sc.Value != 1.05 || sc.Confidence != 0.93 {
		t.Errorf("Score = %+v, %v", sc, ok)
	}
	if sc.Label() != "Frustrated" {
		t.Errorf("Score.Label() = %q, want Frustrated", sc.Label())
	}
	if sc.Legend[2] != "Very angry" {
		t.Errorf("Score.Legend[2] = %q", sc.Legend[2])
	}

	// Call-level fields.
	if res.Model != "typesafe/jev-1.13-20260917" || res.Provider != "TypeSafe" {
		t.Errorf("model/provider = %q/%q", res.Model, res.Provider)
	}
	if res.Usage.Cost != 0.000017934 {
		t.Errorf("cost = %v", res.Usage.Cost)
	}
}

func TestAccessorTypeMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, combinedResponse)
	}))
	defer srv.Close()
	c := jev.New(jev.WithBaseURL(srv.URL), jev.WithModel("m"))
	res, err := c.Decide(context.Background(), jev.Request{
		State:     "x",
		Questions: map[string]jev.Question{"department": jev.Choice("q", map[string]string{"a": "b"})},
	})
	if err != nil {
		t.Fatalf("Decide: %v", err)
	}
	// Asking for the wrong type, or a missing name, must report ok=false.
	if _, ok := res.Noul("department"); ok {
		t.Error("Noul on a choice answer should be ok=false")
	}
	if _, ok := res.Choice("nope"); ok {
		t.Error("Choice on a missing name should be ok=false")
	}
}

func TestDecideRejectsEmptyQuestions(t *testing.T) {
	c := jev.New(jev.WithBaseURL("http://example.invalid"), jev.WithModel("m"))
	if _, err := c.Decide(context.Background(), jev.Request{State: "x"}); err == nil {
		t.Fatal("expected error for empty questions")
	}
}

func TestAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, `{"error":{"message":"bad key"}}`)
	}))
	defer srv.Close()
	c := jev.New(jev.WithBaseURL(srv.URL), jev.WithModel("m"), jev.WithMaxRetries(0))
	_, err := c.Decide(context.Background(), jev.Request{
		State:     "x",
		Questions: map[string]jev.Question{"q": jev.Noul("q", "t", "f")},
	})
	apiErr, ok := err.(*jev.APIError)
	if !ok {
		t.Fatalf("error type = %T, want *jev.APIError", err)
	}
	if apiErr.StatusCode != 401 || apiErr.Message != "bad key" {
		t.Errorf("APIError = %+v", apiErr)
	}
}

// wireCapture decodes the outbound request body for assertions.
type wireCapture struct {
	Model     string `json:"model"`
	State     string `json:"state"`
	Questions map[string]struct {
		Type     string          `json:"type"`
		Criteria json.RawMessage `json:"criteria"`
	} `json:"questions"`
}

// TestLiveDecide hits the real Decisions API. Skipped unless RIMENO_LIVE=1 and
// OPENAI_API_KEY are set:
//
//	RIMENO_LIVE=1 OPENAI_API_KEY=sk-or-... go test ./jev -run Live
//
// The base URL defaults to OpenRouter's alpha Decisions host. Override it with
// JEV_BASE_URL — note this is a different base than the chat OPENAI_BASE_URL
// (.../api/alpha vs .../api/v1), so jev deliberately does not read that var.
func TestLiveDecide(t *testing.T) {
	if os.Getenv("RIMENO_LIVE") != "1" || os.Getenv("OPENAI_API_KEY") == "" {
		t.Skip("set RIMENO_LIVE=1 and OPENAI_API_KEY to run the live jev test")
	}
	base := os.Getenv("JEV_BASE_URL")
	if base == "" {
		base = "https://openrouter.ai/api/alpha"
	}
	c := jev.New(
		jev.WithBaseURL(base),
		jev.WithAPIKey(os.Getenv("OPENAI_API_KEY")),
		jev.WithModel("typesafe/jev-1.13"),
	)
	res, err := c.Decide(context.Background(), jev.Request{
		State: "Help! My payouts have been failing for 3 days.",
		Questions: map[string]jev.Question{
			"department": jev.Choice("Which team should handle this?", map[string]string{
				"billing":   "Payments, invoicing, refunds",
				"technical": "Bugs, outages, integrations",
				"sales":     "Pricing, upgrades, new accounts",
			}),
		},
	})
	if err != nil {
		t.Fatalf("live Decide: %v", err)
	}
	ch, ok := res.Choice("department")
	if !ok || ch.Value == "" {
		t.Fatalf("no choice returned: %+v", res.Answers)
	}
	t.Logf("department=%s confidence=%.2f cost=%.8f", ch.Value, ch.Confidence, res.Usage.Cost)
}
