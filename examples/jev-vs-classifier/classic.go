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

// Frozen labels a sklearn-style head would have learned. Changing this set
// means a new training run, not a new request.
var frozenLabels = []string{"billing", "technical", "account"}

// keywordWeights is a stand-in for a trained linear classifier: each token
// votes for one frozen label. There is no slot for "payout_ops" or
// "bank_escalation" until someone edits this table and ships a new model.
var keywordWeights = map[string]map[string]float64{
	"billing": {
		"payout": 2, "payouts": 2, "charge": 2, "charged": 2,
		"refund": 2, "invoice": 2, "payment": 2, "bank": 1.2,
	},
	"technical": {
		"500": 2, "error": 1.5, "api": 1.5, "dashboard": 1.2,
		"crash": 2, "bug": 2, "outage": 2, "http": 1.5,
	},
	"account": {
		"password": 2, "login": 2, "reset": 1.5, "email": 1,
		"account": 1.5, "username": 1.5,
	},
}

// ClassicPrediction is one frozen-head decision.
type ClassicPrediction struct {
	Label         string
	Probabilities map[string]float64
}

// ClassifyFrozen scores text against the baked-in label set.
func ClassifyFrozen(text string) ClassicPrediction {
	scores := make(map[string]float64, len(frozenLabels))
	for _, label := range frozenLabels {
		scores[label] = 0.15 // small prior so softmax is defined
		for _, tok := range tokenize(text) {
			scores[label] += keywordWeights[label][tok]
		}
	}
	probs := softmax(scores, 1)
	return ClassicPrediction{Label: argmax(probs), Probabilities: probs}
}
