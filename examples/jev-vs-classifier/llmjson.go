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

import "fmt"

// LLMClassifyPrompt is the text you would send to a chat model to approximate
// System One. The criteria are in the prompt — same idea as Jev — but the
// model still generates tokens. Your code owns the parse, the schema failures,
// and the fact that a generated "90% confident" is not p=0.90.
func LLMClassifyPrompt(state string, criteria map[string]string) string {
	return fmt.Sprintf(
		`Classify the ticket into exactly one of the following options.
Reply with JSON only: {"queue": <key>, "confidence": <0-1>}.

Options:
%s

Ticket:
%s`,
		formatCriteria(criteria), state)
}

func formatCriteria(criteria map[string]string) string {
	out := ""
	for k, v := range criteria {
		out += fmt.Sprintf("- %s: %s\n", k, v)
	}
	return out
}

// LLMParseRisks are failure modes a typed System One answer cannot have.
var LLMParseRisks = []string{
	"model writes a label that is not in the list",
	"model wraps the JSON in a markdown fence or a sentence",
	"generated confidence is prose, not a calibrated probability",
	"a second question needs a second generation (or a longer, coupled prompt)",
}
