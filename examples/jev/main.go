// Command jev is a live rimeno example: it triages a support ticket with the jev
// Decisions API (an OpenRouter-hosted classifier), reading one answer of each
// question type — noul, choice, and score.
//
// It needs a real OpenRouter key. Run:
//
//	OPENAI_API_KEY=sk-or-... go run ./examples/jev
//
// The base URL defaults to OpenRouter's alpha Decisions host; override with
// JEV_BASE_URL. Note this is a different base than the chat OPENAI_BASE_URL
// (.../api/alpha vs .../api/v1), so pass it explicitly rather than reusing that.
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/matiasinsaurralde/rimeno/jev"
)

func main() {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("set OPENAI_API_KEY to an OpenRouter (sk-or-v...) key")
	}
	baseURL := os.Getenv("JEV_BASE_URL")
	if baseURL == "" {
		baseURL = "https://openrouter.ai/api/alpha"
	}

	client := jev.New(
		jev.WithBaseURL(baseURL),
		jev.WithAPIKey(apiKey),
		jev.WithModel("typesafe/jev-1.13"),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	ticket := "Help! My payouts have been failing for 3 days."

	res, err := client.Decide(ctx, jev.Request{
		State: ticket,
		Questions: map[string]jev.Question{
			"is_urgent": jev.Noul("Does this message convey urgency?",
				"Explicitly time-sensitive", "No urgency expressed"),
			"department": jev.Choice("Which team should handle this?", map[string]string{
				"billing":   "Payments, invoicing, refunds",
				"technical": "Bugs, outages, integrations",
				"sales":     "Pricing, upgrades, new accounts",
			}),
			"frustration": jev.Score("How frustrated is the customer?",
				"Calm", "Frustrated", "Very angry"),
		},
	})
	if err != nil {
		log.Fatalf("jev decide: %v", err)
	}

	fmt.Printf("ticket: %q\n\n", ticket)

	// Each accessor returns (value, ok). ok is false only if the name is missing
	// or the question was a different type — so branch on it, don't ignore it.
	if urgent, ok := res.Noul("is_urgent"); ok {
		fmt.Printf("urgent:      %.0f%% likely\n", urgent*100)
	}

	if dep, ok := res.Choice("department"); ok {
		fmt.Printf("department:  %s (confidence %.2f)\n", dep.Value, dep.Confidence)
		if dep.Confidence < 0.6 {
			fmt.Println("  -> low confidence, route to a human")
		}
	}

	if fr, ok := res.Score("frustration"); ok {
		// .Value is continuous (e.g. 1.03); .Label() snaps to the nearest legend rung.
		fmt.Printf("frustration: %.2f = %q (confidence %.2f)\n",
			fr.Value, fr.Label(), fr.Confidence)
	}

	fmt.Printf("\nmodel=%s cost=$%.6f\n", res.Model, res.Usage.Cost)
}
