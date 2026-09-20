/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    10|Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Command jev-vs-classifier compares a frozen-label classifier, an LLM JSON
// prompt, and a Jev-shaped System One request on the same tickets.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

func main() {
	backend := flag.String("backend", "local", "System One backend: local or typesafe")
	flag.Parse()

	eval, model, err := evaluatorFromEnv(*backend)
	if err != nil {
		fmt.Fprintf(os.Stderr, "jev-vs-classifier: %v\n", err)
		os.Exit(1)
	}

	if err := runComparison(os.Stdout, eval, model); err != nil {
		fmt.Fprintf(os.Stderr, "jev-vs-classifier: %v\n", err)
		os.Exit(1)
	}
}

func runComparison(out io.Writer, eval Evaluator, model string) error {
	opsTaxonomy := map[string]string{
		"payments": "Payout failures, payouts, and payment processing",
		"account":  "Login, password reset, and account access",
		"other":    "Something else — product bugs, HTTP errors, APIs",
	}
	// Same state, different rubric — the frozen head cannot do this.
	incidentTaxonomy := map[string]string{
		"payout_ops":      "Failed disbursements, payout retries, wallet rails",
		"bank_escalation": "The customer's bank or payment network is involved",
		"product":         "Product or API behavior, not money movement",
	}

	fmt.Fprintln(out, "Jev vs regular classifiers")
	fmt.Fprintln(out, "Guide: docs/jev.md")
	fmt.Fprintf(out, "System One backend: %s\n\n", model)

	for _, ticket := range demoTickets {
		fmt.Fprintf(out, "======== %s ========\n%s\n\n", ticket.ID, ticket.Message)

		classic := ClassifyFrozen(ticket.Message)
		fmt.Fprintln(out, "1) Frozen-label classifier")
		fmt.Fprintf(out, "   labels are baked in: %s\n", strings.Join(frozenLabels, ", "))
		fmt.Fprintf(out, "   prediction: %s\n", classic.Label)
		fmt.Fprintf(out, "   %s\n\n", formatProbs(classic.Probabilities))

		fmt.Fprintln(out, "2) LLM JSON mode (criteria in a prompt, output is still text)")
		fmt.Fprintln(out, indent(LLMClassifyPrompt(ticket.Message, opsTaxonomy), "   "))
		fmt.Fprintln(out, "   parse risks:")
		for _, r := range LLMParseRisks {
			fmt.Fprintf(out, "     - %s\n", r)
		}
		fmt.Fprintln(out)

		questions := map[string]Question{
			"queue": {
				Type:         "choice",
				Instructions: "Which team should handle this ticket?",
				Criteria:     opsTaxonomy,
			},
			"urgent": {
				Type:         "noul",
				Instructions: "Does this message require urgent human attention?",
				Criteria: map[string]string{
					"true":  "Explicitly time-sensitive or repeated failure",
					"false": "Ordinary request, no urgency",
				},
			},
			"frustration": {
				Type:         "score",
				Instructions: "How frustrated is the customer?",
				Levels:       []string{"Calm and neutral", "Concerned but civil", "Very angry"},
			},
		}

		res, err := eval.Evaluate(ticket.Message, questions)
		if err != nil {
			return err
		}

		fmt.Fprintln(out, "3) System One (criteria live on the question)")
		printAnswer(out, "   queue", res.Answers["queue"])
		printAnswer(out, "   urgent", res.Answers["urgent"])
		printAnswer(out, "   frustration", res.Answers["frustration"])
		fmt.Fprintln(out)

		if ticket.ID != "T1" {
			continue
		}

		// The point of the example: retarget the same state without retraining.
		retarget := map[string]Question{
			"queue": {
				Type:         "choice",
				Instructions: "Which incident lane should own this?",
				Criteria:     incidentTaxonomy,
			},
		}
		retargeted, err := eval.Evaluate(ticket.Message, retarget)
		if err != nil {
			return err
		}
		fmt.Fprintln(out, "   same state, new criteria (no retrain):")
		printAnswer(out, "   queue", retargeted.Answers["queue"])
		fmt.Fprintln(out, "   frozen classifier still has no label for payout_ops / bank_escalation.")
		fmt.Fprintln(out)
	}

	fmt.Fprintln(out, "TypeSafe docs: https://docs.typesafe.ai/introduction.md")
	return nil
}

func printAnswer(out io.Writer, name string, a Answer) {
	switch a.Type {
	case "choice":
		fmt.Fprintf(out, "%s: %s  (confidence=%.2f)\n", name, a.Choice, a.Confidence)
		fmt.Fprintf(out, "%s  %s\n", strings.Repeat(" ", len(name)), formatProbs(a.Probabilities))
	case "noul":
		fmt.Fprintf(out, "%s: noul=%.2f\n", name, a.Noul)
	case "score":
		fmt.Fprintf(out, "%s: score=%.2f  (confidence=%.2f)\n", name, a.Score, a.Confidence)
	default:
		fmt.Fprintf(out, "%s: %+v\n", name, a)
	}
}

func formatProbs(probs map[string]float64) string {
	keys := make([]string, 0, len(probs))
	for k := range probs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%.2f", k, probs[k]))
	}
	return strings.Join(parts, "  ")
}

func indent(s, prefix string) string {
	lines := strings.Split(strings.TrimSuffix(s, "\n"), "\n")
	for i, line := range lines {
		lines[i] = prefix + line
	}
	return strings.Join(lines, "\n")
}
