# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility exports for lightweight multimodal metadata."""

from __future__ import annotations

from typing import Any

from tokenizer_service.mm_metadata.placeholder import smart_resize
from tokenizer_service.mm_metadata.service import MultiModalMetadataService
from tokenizer_service.mm_metadata.types import MetadataItem, UnsupportedCounterError


def _qwen_placeholder_count(
    width: int,
    height: int,
    attrs: dict[str, Any],
    kwargs: dict[str, Any],
) -> int:
    patch_size = int(kwargs.get("patch_size") or attrs.get("patch_size") or 14)
    merge_size = int(kwargs.get("merge_size") or attrs.get("merge_size") or 2)
    factor = patch_size * merge_size
    min_pixels = int(kwargs.get("min_pixels") or attrs.get("min_pixels") or factor * factor)
    max_pixels = int(kwargs.get("max_pixels") or attrs.get("max_pixels") or 1280 * 28 * 28)
    resized_h, resized_w = smart_resize(height, width, factor, min_pixels, max_pixels)
    return max(1, (resized_h // factor) * (resized_w // factor))
