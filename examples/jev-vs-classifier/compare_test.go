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

package main

import (
	"io"
	"math"
	"testing"
)

func TestLocalMockFollowsRequestCriteria(t *testing.T) {
	eval := LocalEvaluator{}
	ops := map[string]Question{
		"queue": {
			Type:         "choice",
			Instructions: "Which team should handle this ticket?",
			Criteria: map[string]string{
				"payments": "Payout failures, payouts, and payment processing",
				"account":  "Login, password reset, and account access",
				"other":    "Something else — product bugs, HTTP errors, APIs",
			},
		},
	}
	want := map[string]string{"T1": "payments", "T2": "account", "T3": "other"}
	for _, ticket := range demoTickets {
		res, err := eval.Evaluate(ticket.Message, ops)
		if err != nil {
			t.Fatal(err)
		}
		if got := res.Answers["queue"].Choice; got != want[ticket.ID] {
			t.Fatalf("%s: queue=%s want %s (%v)", ticket.ID, got, want[ticket.ID],
				res.Answers["queue"].Probabilities)
		}
	}
}

func TestClassicStaysOnFrozenLabels(t *testing.T) {
	for _, ticket := range demoTickets {
		got := ClassifyFrozen(ticket.Message)
		if !contains(frozenLabels, got.Label) {
			t.Fatalf("%s: unexpected label %q", ticket.ID, got.Label)
		}
		if math.Abs(sumProbs(got.Probabilities)-1) > 1e-9 {
			t.Fatalf("%s: probabilities sum to %f", ticket.ID, sumProbs(got.Probabilities))
		}
	}
}

func TestSystemOneAcceptsRuntimeCriteria(t *testing.T) {
	eval := LocalEvaluator{}
	state := demoTickets[0].Message

	ops := map[string]Question{
		"queue": {
			Type:         "choice",
			Instructions: "Which team should handle this ticket?",
			Criteria: map[string]string{
				"payments": "Payout failures, payouts, and payment processing",
				"account":  "Login, password reset, and account access",
				"other":    "Something else — product bugs, HTTP errors, APIs",
			},
		},
	}
	incident := map[string]Question{
		"queue": {
			Type:         "choice",
			Instructions: "Which incident lane should own this?",
			Criteria: map[string]string{
				"payout_ops":      "Failed disbursements, payout retries, wallet rails",
				"bank_escalation": "The customer's bank or payment network is involved",
				"product":         "Product or API behavior, not money movement",
			},
		},
	}

	a, err := eval.Evaluate(state, ops)
	if err != nil {
		t.Fatal(err)
	}
	b, err := eval.Evaluate(state, incident)
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := a.Answers["queue"].Probabilities["payments"]; !ok {
		t.Fatal("ops taxonomy missing payments")
	}
	if _, ok := b.Answers["queue"].Probabilities["payout_ops"]; !ok {
		t.Fatal("incident taxonomy missing payout_ops")
	}
	if _, ok := b.Answers["queue"].Probabilities[b.Answers["queue"].Choice]; !ok {
		t.Fatalf("winner %q is not in the request criteria", b.Answers["queue"].Choice)
	}
	classic := ClassifyFrozen(state)
	if contains([]string{"payout_ops", "bank_escalation", "product"}, classic.Label) {
		t.Fatal("frozen classifier must not invent the new taxonomy")
	}
}

func TestQuestionsAreIndependent(t *testing.T) {
	eval := LocalEvaluator{}
	state := demoTickets[0].Message
	choiceOnly := map[string]Question{
		"queue": {
			Type:         "choice",
			Instructions: "Which team should handle this ticket?",
			Criteria: map[string]string{
				"payments": "Payout failures, payouts, and payment processing",
				"account":  "Login, password reset, and account access",
			},
		},
	}
	withNoul := map[string]Question{
		"queue":  choiceOnly["queue"],
		"urgent": {Type: "noul", Instructions: "Is this urgent?"},
	}

	a, err := eval.Evaluate(state, choiceOnly)
	if err != nil {
		t.Fatal(err)
	}
	b, err := eval.Evaluate(state, withNoul)
	if err != nil {
		t.Fatal(err)
	}
	if a.Answers["queue"].Choice != b.Answers["queue"].Choice {
		t.Fatalf("adding a noul changed the choice: %s vs %s",
			a.Answers["queue"].Choice, b.Answers["queue"].Choice)
	}
}

func TestChoiceConfidenceFormula(t *testing.T) {
	// Three options, pMax=0.8 → (0.8 - 1/3) / (1 - 1/3) = 0.7
	got := choiceConfidence(map[string]float64{"a": 0.8, "b": 0.15, "c": 0.05})
	if math.Abs(got-0.7) > 1e-9 {
		t.Fatalf("got %f want 0.7", got)
	}
}

func TestRunComparisonLocal(t *testing.T) {
	if err := runComparison(io.Discard, LocalEvaluator{}, "local-lexical-mock"); err != nil {
		t.Fatal(err)
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

func sumProbs(probs map[string]float64) float64 {
	var s float64
	for _, p := range probs {
		s += p
	}
	return s
}
