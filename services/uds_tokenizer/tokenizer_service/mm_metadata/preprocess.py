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

"""Metadata-only preprocessing fallback for multimodal placeholder counts."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping
from typing import Any

from .image_item import load_rgb_image
from .model_bundle import first_number, get_model_bundle
from .types import ModelBundle, PlaceholderCountResult


EXACT_COUNT_FIELDS = (
    "num_image_tokens",
    "num_img_tokens",
    "image_tokens",
    "num_tokens",
    "image_token_count",
)


def count_placeholders_from_preprocess(
    model_name: str,
    raw: bytes,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult:
    """Run the cheapest image-only processor path and infer placeholder count.

    This intentionally avoids chat-template rendering and tokenization. It uses
    processor outputs only when those outputs expose count-like metadata.
    """

    bundle = get_model_bundle(model_name)
    image = load_rgb_image(raw)
    width, height = image.size

    helper_result = _processor_multimodal_token_count(
        bundle,
        width,
        height,
        processor_kwargs,
    )
    if helper_result is not None:
        return helper_result

    for owner_name, owner in (
        ("image_processor", bundle.image_processor),
        ("processor", bundle.processor),
    ):
        if owner is None:
            continue
        output = _call_image_only(owner, image, processor_kwargs)
        if output is None:
            continue
        result = _extract_count(bundle, output)
        if result is not None:
            return PlaceholderCountResult(
                count=result.count,
                exact=result.exact,
                method=f"preprocess-fallback.{owner_name}.{result.method}",
            )

    return PlaceholderCountResult(
        count=_raw_area_proxy_count(width, height, processor_kwargs),
        exact=False,
        method="preprocess-fallback.raw-pixel-area-proxy",
    )


def _processor_multimodal_token_count(
    bundle: ModelBundle,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult | None:
    method = getattr(bundle.processor, "_get_num_multimodal_tokens", None)
    if not callable(method):
        return None

    for kwargs in (
        {"image_sizes": [(height, width)]},
        {"image_sizes": [[height, width]]},
        {"images_kwargs": {"image_sizes": [(height, width)]}},
    ):
        call_kwargs = kwargs | _accepted_kwargs(method, processor_kwargs)
        try:
            output = method(**call_kwargs)
        except Exception:
            continue
        count = _count_from_multimodal_token_output(output)
        if count:
            return PlaceholderCountResult(
                count=count,
                exact=True,
                method="preprocess-fallback.processor._get_num_multimodal_tokens",
            )
    return None


def _call_image_only(
    owner: Any,
    image: Any,
    processor_kwargs: dict[str, Any],
) -> Mapping[str, Any] | None:
    attempts = (
        {"images": [image], "return_tensors": "pt"},
        {"images": image, "return_tensors": "pt"},
        {"image": image, "return_tensors": "pt"},
        {"images": [image]},
        {"images": image},
    )
    for kwargs in attempts:
        call_kwargs = kwargs | _accepted_kwargs(owner, processor_kwargs)
        try:
            output = owner(**call_kwargs)
        except Exception:
            continue
        mapping = _as_mapping(output)
        if mapping:
            return mapping
    return None


def _accepted_kwargs(owner: Callable[..., Any], values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(owner)
    except (TypeError, ValueError):
        return {}
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return values
    return {k: v for k, v in values.items() if k in signature.parameters}


def _as_mapping(output: Any) -> Mapping[str, Any] | None:
    if isinstance(output, Mapping):
        return output
    if hasattr(output, "data") and isinstance(output.data, Mapping):
        return output.data
    try:
        as_dict = dict(output)
    except Exception:
        return None
    return as_dict if as_dict else None


class _ExtractedCount:
    def __init__(self, count: int, exact: bool, method: str):
        self.count = count
        self.exact = exact
        self.method = method


def _extract_count(
    bundle: ModelBundle,
    output: Mapping[str, Any],
) -> _ExtractedCount | None:
    grid_count = _count_from_image_grid(bundle, output.get("image_grid_thw"))
    if grid_count:
        return _ExtractedCount(
            grid_count,
            _is_qwen_grid_family(bundle),
            "image_grid_thw",
        )

    for field in EXACT_COUNT_FIELDS:
        count = _single_positive_int(output.get(field))
        if count:
            return _ExtractedCount(count, True, field)

    patch_count = _single_positive_int(output.get("num_patches"))
    if not patch_count:
        patch_count = _single_positive_int(output.get("num_image_patches"))
    image_seq_length = int(first_number(bundle.attrs, names=("image_seq_length",)) or 0)
    if patch_count and image_seq_length:
        return _ExtractedCount(
            patch_count * image_seq_length,
            False,
            "num_patches_x_image_seq_length",
        )
    if patch_count:
        return _ExtractedCount(patch_count, False, "num_patches")

    spatial_count = _count_from_spatial_shapes(output.get("spatial_shapes"))
    if spatial_count:
        return _ExtractedCount(spatial_count, False, "spatial_shapes")

    mask_count = _count_from_pixel_attention_mask(output.get("pixel_attention_mask"))
    if mask_count:
        return _ExtractedCount(mask_count, False, "pixel_attention_mask")

    placeholder_count = _count_from_placeholders(output.get("image_placeholders"))
    if placeholder_count:
        return _ExtractedCount(placeholder_count, True, "image_placeholders")

    return None


def _count_from_image_grid(bundle: ModelBundle, value: Any) -> int:
    rows = _rows(value)
    if not rows:
        return 0
    merge_size = int(
        first_number(
            bundle.attrs,
            getattr(bundle.config, "vision_config", None),
            names=("spatial_merge_size", "merge_size"),
        )
        or 1
    )
    denominator = max(1, merge_size**2)
    total = 0
    for row in rows:
        if len(row) < 3:
            continue
        total += max(1, int(row[0]) * int(row[1]) * int(row[2]) // denominator)
    return total


def _is_qwen_grid_family(bundle: ModelBundle) -> bool:
    lower = f"{bundle.model_type} {bundle.model_name.lower()} {bundle.processor_class.lower()} {bundle.image_processor_class.lower()}"
    return "qwen" in lower or "glm4v" in lower or "glm-4" in lower


def _count_from_spatial_shapes(value: Any) -> int:
    rows = _rows(value)
    if not rows:
        return 0
    total = 0
    for row in rows:
        if len(row) < 2:
            continue
        total += max(1, int(row[-2]) * int(row[-1]))
    return total


def _count_from_placeholders(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value else 0
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _count_from_placeholders(value[0])
        return len(value)
    return 0


def _count_from_pixel_attention_mask(value: Any) -> int:
    normalized = _to_python(value)
    if isinstance(normalized, (list, tuple)):
        total = 0
        for item in _flatten(normalized):
            if isinstance(item, bool):
                total += 1 if item else 0
            elif isinstance(item, (int, float)):
                total += 1 if item > 0 else 0
        return total
    return 0


def _count_from_multimodal_token_output(value: Any) -> int:
    mapping = _as_mapping(value)
    if mapping:
        for key in ("num_image_tokens", "image_tokens", "num_tokens", "num_img_tokens"):
            count = _sum_positive_ints(mapping.get(key))
            if count:
                return count
        nested = mapping.get("image")
        if nested is not None:
            count = _count_from_multimodal_token_output(nested)
            if count:
                return count

    for attr in ("num_image_tokens", "image_tokens", "num_tokens", "num_img_tokens"):
        if hasattr(value, attr):
            count = _sum_positive_ints(getattr(value, attr))
            if count:
                return count
    return 0


def _raw_area_proxy_count(
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> int:
    proxy_patch_area = int(processor_kwargs.get("proxy_patch_area") or 28 * 28)
    return max(1, int(math.ceil((width * height) / max(1, proxy_patch_area))))


def _single_positive_int(value: Any) -> int:
    values = _flatten(value)
    if len(values) != 1:
        return 0
    item = values[0]
    if isinstance(item, bool):
        return 0
    if isinstance(item, (int, float)) and item > 0:
        return int(item)
    return 0


def _sum_positive_ints(value: Any) -> int:
    total = 0
    for item in _flatten(value):
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)) and item > 0:
            total += int(item)
    return total


def _rows(value: Any) -> list[list[int]]:
    normalized = _to_python(value)
    if not isinstance(normalized, (list, tuple)):
        return []
    if normalized and all(isinstance(item, (int, float)) for item in normalized):
        return [[int(item) for item in normalized]]
    rows: list[list[int]] = []
    for row in normalized:
        if isinstance(row, (list, tuple)) and all(
            isinstance(item, (int, float)) for item in row
        ):
            rows.append([int(item) for item in row])
    return rows


def _flatten(value: Any) -> list[Any]:
    value = _to_python(value)
    if isinstance(value, (list, tuple)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    if value is None:
        return []
    return [value]


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
