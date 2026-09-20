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
	"math"
	"strings"
	"unicode"
)

func tokenize(text string) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		if f != "" {
			out = append(out, f)
		}
	}
	return out
}

func stems(tok string) []string {
	out := []string{tok}
	switch {
	case strings.HasSuffix(tok, "ing") && len(tok) > 5:
		out = append(out, strings.TrimSuffix(tok, "ing"))
	case strings.HasSuffix(tok, "ed") && len(tok) > 4:
		out = append(out, strings.TrimSuffix(tok, "ed"))
	case strings.HasSuffix(tok, "ies") && len(tok) > 5:
		out = append(out, strings.TrimSuffix(tok, "ies")+"y")
	case strings.HasSuffix(tok, "es") && len(tok) > 4:
		out = append(out, strings.TrimSuffix(tok, "es"))
	case strings.HasSuffix(tok, "s") && len(tok) > 3:
		out = append(out, strings.TrimSuffix(tok, "s"))
	}
	if strings.HasSuffix(tok, "ure") && len(tok) > 5 {
		out = append(out, strings.TrimSuffix(tok, "ure"))
	}
	return out
}

func tokenSet(text string) map[string]struct{} {
	set := make(map[string]struct{})
	for _, tok := range tokenize(text) {
		for _, s := range stems(tok) {
			set[s] = struct{}{}
		}
	}
	return set
}

func overlapScore(state, rubric string) float64 {
	stateSet := tokenSet(state)
	var hits float64
	rubricTokens := tokenize(rubric)
	if len(rubricTokens) == 0 {
		return 0.1
	}
	for _, tok := range rubricTokens {
		if _, ok := stateSet[tok]; ok {
			hits++
		}
	}
	return hits + 0.1
}

func softmax(scores map[string]float64, temperature float64) map[string]float64 {
	if temperature <= 0 {
		temperature = 1
	}
	var max float64
	first := true
	for _, s := range scores {
		if first || s > max {
			max = s
			first = false
		}
	}
	out := make(map[string]float64, len(scores))
	var sum float64
	for k, s := range scores {
		v := math.Exp((s - max) / temperature)
		out[k] = v
		sum += v
	}
	for k := range out {
		out[k] /= sum
	}
	return out
}

func argmax(probs map[string]float64) string {
	var best string
	var bestP float64
	first := true
	for k, p := range probs {
		if first || p > bestP || (p == bestP && k < best) {
			best, bestP, first = k, p, false
		}
	}
	return best
}

// choiceConfidence is TypeSafe's published Choice summary:
// (pMax - 1/K) / (1 - 1/K) for K > 1.
func choiceConfidence(probs map[string]float64) float64 {
	k := float64(len(probs))
	if k <= 1 {
		return 1
	}
	var pMax float64
	for _, p := range probs {
		if p > pMax {
			pMax = p
		}
	}
	return math.Max(0, math.Min(1, (pMax-1/k)/(1-1/k)))
}
