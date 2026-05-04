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

"""Public service façade for lightweight multimodal metadata."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from .hashing import hash_item
from .image_item import image_bytes, probe_dimensions
from .model_bundle import get_model_bundle
from .placeholder import count_placeholders
from .preprocess import count_placeholders_from_preprocess
from .types import MetadataItem, PlaceholderCountResult, UnsupportedCounterError


class MultiModalMetadataService:
    """Computes multimodal metadata without full HF preprocessing by default."""

    def prewarm_model(self, model_name: str) -> None:
        """Load cached processor/config metadata before the first request."""
        start = time.perf_counter()
        bundle = get_model_bundle(model_name)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logging.info(
            "MM metadata model prewarmed: model=%s processor=%s image_processor=%s elapsed_ms=%.2f",
            model_name,
            bundle.processor_class,
            bundle.image_processor_class,
            elapsed_ms,
        )

    def get_metadata(
        self,
        model_name: str,
        items: list[dict[str, str]],
        processor_kwargs_json: str = "",
        allow_preprocess_fallback: bool = False,
        hash_mode: str = "vllm",
    ) -> list[MetadataItem]:
        processor_kwargs = self._parse_json(processor_kwargs_json)
        results: list[MetadataItem] = []
        for item in items:
            try:
                results.append(
                    self._get_item_metadata(
                        model_name,
                        item,
                        processor_kwargs,
                        allow_preprocess_fallback,
                        hash_mode or "vllm",
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive result boundary
                logging.warning("MM metadata item failed: %s", exc, exc_info=True)
                results.append(
                    MetadataItem(
                        modality=item.get("modality") or "image",
                        method="error",
                        error=str(exc),
                    )
                )
        return results

    def _get_item_metadata(
        self,
        model_name: str,
        item: dict[str, str],
        processor_kwargs: dict[str, Any],
        allow_preprocess_fallback: bool,
        hash_mode: str,
    ) -> MetadataItem:
        modality = item.get("modality") or "image"
        if modality != "image":
            return MetadataItem(
                modality=modality,
                method="unsupported-modality",
                error=f"unsupported modality {modality!r}",
            )

        raw = image_bytes(item)
        dims = probe_dimensions(raw)
        mm_hash, exact_hash, hash_method = hash_item(
            item,
            raw,
            processor_kwargs,
            hash_mode,
        )

        force_preprocess = (
            allow_preprocess_fallback
            and os.getenv("MM_METADATA_FORCE_PREPROCESS_FALLBACK", "").lower()
            in ("1", "true", "yes")
        )
        count_start = time.perf_counter()
        try:
            if force_preprocess:
                raise UnsupportedCounterError("forced metadata-only preprocessing")
            count_result = _as_count_result(
                self._count_placeholders(
                    model_name,
                    dims.width,
                    dims.height,
                    processor_kwargs,
                )
            )
            elapsed_ms = (time.perf_counter() - count_start) * 1000
            logging.debug(
                "MM metadata count resolved: method=%s exact=%s count=%s elapsed_ms=%.2f",
                count_result.method,
                count_result.exact,
                count_result.count,
                elapsed_ms,
            )
            return MetadataItem(
                modality=modality,
                mm_hash=mm_hash,
                placeholder_count=count_result.count,
                width=dims.width,
                height=dims.height,
                exact_hash=exact_hash,
                exact_placeholder_count=count_result.exact,
                method=f"{hash_method}+{count_result.method}",
            )
        except UnsupportedCounterError as exc:
            if allow_preprocess_fallback:
                try:
                    fallback_start = time.perf_counter()
                    count_result = count_placeholders_from_preprocess(
                        model_name,
                        raw,
                        processor_kwargs,
                    )
                    elapsed_ms = (time.perf_counter() - fallback_start) * 1000
                    logging.debug(
                        "MM metadata preprocess fallback resolved: method=%s exact=%s count=%s elapsed_ms=%.2f",
                        count_result.method,
                        count_result.exact,
                        count_result.count,
                        elapsed_ms,
                    )
                    return MetadataItem(
                        modality=modality,
                        mm_hash=mm_hash,
                        placeholder_count=count_result.count,
                        width=dims.width,
                        height=dims.height,
                        exact_hash=exact_hash,
                        exact_placeholder_count=count_result.exact,
                        method=f"{hash_method}+{count_result.method}",
                    )
                except UnsupportedCounterError as fallback_exc:
                    exc = fallback_exc
            method = "preprocess-fallback-failed" if allow_preprocess_fallback else "unsupported"
            return MetadataItem(
                modality=modality,
                mm_hash=mm_hash,
                width=dims.width,
                height=dims.height,
                exact_hash=exact_hash,
                method=f"{hash_method}+{method}",
                error=str(exc),
            )

    @staticmethod
    def _parse_json(value: str) -> dict[str, Any]:
        if not value:
            return {}
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("processor kwargs must be a JSON object")
        return parsed

    def _count_placeholders(
        self,
        model_name: str,
        width: int,
        height: int,
        processor_kwargs: dict[str, Any],
    ) -> PlaceholderCountResult:
        return count_placeholders(model_name, width, height, processor_kwargs)


def _as_count_result(value: PlaceholderCountResult | int) -> PlaceholderCountResult:
    if isinstance(value, PlaceholderCountResult):
        return value
    if isinstance(value, int) and value > 0:
        return PlaceholderCountResult(
            count=value,
            exact=True,
            method="test-helper-count",
        )
    raise UnsupportedCounterError(f"invalid placeholder count result: {value!r}")
