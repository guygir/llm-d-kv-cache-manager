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

"""Hash strategies for multimodal metadata."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from .image_item import load_rgb_image


def hash_item(
    item: dict[str, str],
    raw: bytes,
    processor_kwargs: dict[str, Any],
    hash_mode: str,
) -> tuple[str, bool, str]:
    if item.get("uuid"):
        return item["uuid"], True, "uuid"
    if hash_mode == "stable-identifier":
        identifier = item.get("url") or item.get("data") or ""
        return hashlib.sha256(identifier.encode()).hexdigest(), False, "stable-identifier"
    if hash_mode != "vllm":
        return "", False, f"unsupported-hash-mode-{hash_mode}"

    try:
        from vllm.multimodal.hasher import MultiModalHasher

        image = load_rgb_image(raw)
        hasher = MultiModalHasher()
        if hasattr(hasher, "hash"):
            return str(hasher.hash(image, **processor_kwargs)), True, "vllm-hash"
        if hasattr(MultiModalHasher, "hash"):
            return str(MultiModalHasher.hash(image, **processor_kwargs)), True, "vllm-hash"
    except Exception as exc:
        logging.warning("vLLM multimodal hash failed: %s", exc, exc_info=True)
    return "", False, "vllm-hash-unavailable"
