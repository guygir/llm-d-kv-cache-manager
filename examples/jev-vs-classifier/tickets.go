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

// Ticket is a support message used as System One "state".
type Ticket struct {
	ID      string
	Message string
}

var demoTickets = []Ticket{
	{
		ID: "T1",
		Message: "Help! My payouts have been failing for 3 days. " +
			"The bank says everything is fine. Can someone please fix this?",
	},
	{
		ID:      "T2",
		Message: "I forgot my password and the reset email never arrived.",
	},
	{
		ID:      "T3",
		Message: "The dashboard returns HTTP 500 whenever I open the API keys page.",
	},
}
