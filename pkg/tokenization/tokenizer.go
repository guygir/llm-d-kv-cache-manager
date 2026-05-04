/*
Copyright 2025 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tokenization

import (
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	types "github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
)

// MultiModalFeatures holds multimodal metadata produced by the tokenizer.
// Decoupled from proto types so callers don't depend on generated code.
type MultiModalFeatures struct {
	// MMHashes maps modality (e.g. "image") to per-item content hashes.
	MMHashes map[string][]string
	// MMPlaceholders maps modality to per-item placeholder token ranges.
	MMPlaceholders map[string][]kvblock.PlaceholderRange
}

// MultiModalMetadataRequest asks the tokenizer sidecar for lightweight
// multimodal item hashes and placeholder counts.
type MultiModalMetadataRequest struct {
	ModelName               string
	Items                   []MultiModalMetadataItemRequest
	ProcessorKwargsJSON     string
	AllowPreprocessFallback bool
	HashMode                string
}

// MultiModalMetadataItemRequest describes one multimodal request item.
type MultiModalMetadataItemRequest struct {
	Modality string
	URL      string
	Data     string
	UUID     string
	MIMEType string
}

// MultiModalMetadataResponse contains one metadata result per requested item.
type MultiModalMetadataResponse struct {
	Items        []MultiModalMetadataItem
	Success      bool
	ErrorMessage string
}

// MultiModalMetadataItem is lightweight metadata for one multimodal item.
type MultiModalMetadataItem struct {
	Modality              string
	Hash                  string
	PlaceholderCount      int
	Width                 int
	Height                int
	ExactHash             bool
	ExactPlaceholderCount bool
	Method                string
	Error                 string
}

// Tokenizer interface defines the methods for tokenization.
type Tokenizer interface {
	RenderChat(*types.RenderChatRequest) ([]uint32, *MultiModalFeatures, error)
	Render(string) ([]uint32, []types.Offset, error)
	Type() string
}
