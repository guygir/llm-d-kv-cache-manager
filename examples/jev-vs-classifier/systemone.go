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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

const typesafeEndpoint = "https://api.typesafe.ai/v1/systemone"

// Question is one System One primitive. Criteria travel with the request.
type Question struct {
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria,omitempty"`
	Levels       []string          `json:"-"`
}

// MarshalJSON emits TypeSafe's wire shape: Choice/Noul use a criteria object,
// Score uses a criteria array.
func (q Question) MarshalJSON() ([]byte, error) {
	switch q.Type {
	case "score":
		return json.Marshal(struct {
			Type         string   `json:"type"`
			Instructions string   `json:"instructions"`
			Criteria     []string `json:"criteria"`
		}{q.Type, q.Instructions, q.Levels})
	default:
		return json.Marshal(struct {
			Type         string            `json:"type"`
			Instructions string            `json:"instructions"`
			Criteria     map[string]string `json:"criteria,omitempty"`
		}{q.Type, q.Instructions, q.Criteria})
	}
}

// Answer is the typed result for one question.
type Answer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice,omitempty"`
	Noul          float64            `json:"noul,omitempty"`
	Score         float64            `json:"score,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    float64            `json:"confidence,omitempty"`
}

// Result is one System One response.
type Result struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
}

// Evaluator scores a state against a question map.
type Evaluator interface {
	Evaluate(state string, questions map[string]Question) (Result, error)
}

// LocalEvaluator is a teaching mock: lexical overlap of (instructions +
// criteria) against the state, then softmax. It is not Jev. It exists so the
// example runs without an API key and still shows that criteria are inputs.
type LocalEvaluator struct{}

func (LocalEvaluator) Evaluate(state string, questions map[string]Question) (Result, error) {
	answers := make(map[string]Answer, len(questions))
	for id, q := range questions {
		ans, err := scoreLocal(state, q)
		if err != nil {
			return Result{}, fmt.Errorf("question %q: %w", id, err)
		}
		answers[id] = ans
	}
	return Result{Model: "local-lexical-mock", Answers: answers}, nil
}

func scoreLocal(state string, q Question) (Answer, error) {
	switch q.Type {
	case "choice":
		if len(q.Criteria) == 0 {
			return Answer{}, fmt.Errorf("choice requires criteria")
		}
		scores := make(map[string]float64, len(q.Criteria))
		for opt, desc := range q.Criteria {
			scores[opt] = overlapScore(state, q.Instructions+" "+opt+" "+desc)
		}
		probs := softmax(scores, 1.5)
		return Answer{
			Type:          "choice",
			Choice:        argmax(probs),
			Probabilities: probs,
			Confidence:    choiceConfidence(probs),
		}, nil
	case "noul":
		yesDesc, noDesc := q.Criteria["true"], q.Criteria["false"]
		yes := overlapScore(state, q.Instructions+" "+yesDesc)
		no := overlapScore(state, noDesc)
		if no == 0 && yes == 0 {
			yes, no = 0.5, 0.5
		}
		noul := yes / (yes + no)
		return Answer{Type: "noul", Noul: noul}, nil
	case "score":
		if len(q.Levels) < 2 {
			return Answer{}, fmt.Errorf("score requires at least two levels")
		}
		scores := make(map[string]float64, len(q.Levels))
		for i, level := range q.Levels {
			key := fmt.Sprintf("%d", i)
			scores[key] = overlapScore(state, q.Instructions+" "+level)
		}
		probs := softmax(scores, 1.5)
		var expected float64
		for i := range q.Levels {
			expected += float64(i) * probs[fmt.Sprintf("%d", i)]
		}
		return Answer{
			Type:          "score",
			Score:         expected,
			Probabilities: probs,
			Confidence:    choiceConfidence(probs),
		}, nil
	default:
		return Answer{}, fmt.Errorf("unknown type %q", q.Type)
	}
}

// TypeSafeEvaluator calls TypeSafe's hosted System One API.
type TypeSafeEvaluator struct {
	APIKey string
	Model  string
	Client *http.Client
}

func newTypeSafeEvaluator(apiKey string) *TypeSafeEvaluator {
	return &TypeSafeEvaluator{
		APIKey: apiKey,
		Model:  "jev-latest",
		Client: &http.Client{Timeout: 30 * time.Second},
	}
}

func (e *TypeSafeEvaluator) Evaluate(state string, questions map[string]Question) (Result, error) {
	body, err := json.Marshal(struct {
		State     string              `json:"state"`
		Model     string              `json:"model"`
		Questions map[string]Question `json:"questions"`
	}{state, e.Model, questions})
	if err != nil {
		return Result{}, err
	}

	req, err := http.NewRequest(http.MethodPost, typesafeEndpoint, bytes.NewReader(body))
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Authorization", "Bearer "+e.APIKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.Client.Do(req)
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return Result{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return Result{}, fmt.Errorf("typesafe %s: %s", resp.Status, payload)
	}

	var out Result
	if err := json.Unmarshal(payload, &out); err != nil {
		return Result{}, err
	}
	return out, nil
}

func evaluatorFromEnv(backend string) (Evaluator, string, error) {
	switch backend {
	case "", "local":
		return LocalEvaluator{}, "local-lexical-mock", nil
	case "typesafe":
		key := os.Getenv("TYPESAFE_API_KEY")
		if key == "" {
			return nil, "", fmt.Errorf("TYPESAFE_API_KEY is required for --backend=typesafe")
		}
		return newTypeSafeEvaluator(key), "jev-latest", nil
	default:
		return nil, "", fmt.Errorf("unknown backend %q (want local or typesafe)", backend)
	}
}
