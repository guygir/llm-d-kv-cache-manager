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

"""Shared types for lightweight multimodal metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetadataItem:
    modality: str
    mm_hash: str = ""
    placeholder_count: int = 0
    width: int = 0
    height: int = 0
    exact_hash: bool = False
    exact_placeholder_count: bool = False
    method: str = ""
    error: str = ""


@dataclass(frozen=True)
class ImageDimensions:
    width: int
    height: int


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    config: Any
    processor: Any
    image_processor: Any
    model_type: str
    processor_class: str
    image_processor_class: str
    attrs: dict[str, Any]


@dataclass(frozen=True)
class PlaceholderCountResult:
    count: int
    exact: bool
    method: str


class UnsupportedCounterError(Exception):
    pass
