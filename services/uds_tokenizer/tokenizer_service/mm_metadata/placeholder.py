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

"""Placeholder count strategies aligned with vLLM processor/config logic."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

from .model_bundle import first_int, first_number, get_model_bundle, get_nested
from .types import ModelBundle, PlaceholderCountResult, UnsupportedCounterError


FIXED_COUNT_FIELDS = (
    "num_image_tokens",
    "image_seq_length",
    "image_token_len",
    "num_image_patches",
    "image_seq_len",
    "num_query_tokens",
)


def count_placeholders(
    model_name: str,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult:
    bundle = get_model_bundle(model_name)
    for strategy in (
        _processor_helper_count,
        _fixed_config_count,
        _vision_encoder_count,
        _llava_next_count,
        _qwen_smart_resize_count,
        _internvl_approx_count,
    ):
        result = strategy(bundle, width, height, processor_kwargs)
        if result is not None:
            return result
    raise UnsupportedCounterError(
        "no lightweight placeholder counter for "
        f"model_type={bundle.model_type!r}, processor={bundle.processor_class!r}"
    )


def _processor_helper_count(
    bundle: ModelBundle,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult | None:
    for owner_name, owner in (
        ("processor", bundle.processor),
        ("image_processor", bundle.image_processor),
    ):
        if owner is None:
            continue
        for method_name in (
            "get_num_image_tokens",
            "calc_num_image_tokens_from_image_size",
            "calc_num_image_tokens",
            "_compute_num_image_tokens",
        ):
            method = getattr(owner, method_name, None)
            count = _try_dimension_method(method, width, height, processor_kwargs)
            if count:
                return PlaceholderCountResult(
                    count=count,
                    exact=True,
                    method=f"{owner_name}.{method_name}",
                )

    count = _try_transformers_multimodal_helper(
        getattr(bundle.processor, "_get_num_multimodal_tokens", None),
        width,
        height,
        processor_kwargs,
    )
    if count:
        return PlaceholderCountResult(
            count=count,
            exact=True,
            method="processor._get_num_multimodal_tokens",
        )

    return None


def _fixed_config_count(
    bundle: ModelBundle,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult | None:
    del width, height
    count = first_int(
        processor_kwargs,
        bundle.attrs,
        getattr(bundle.config, "vision_config", None),
        bundle.config,
        names=FIXED_COUNT_FIELDS,
    )
    if count:
        return PlaceholderCountResult(
            count=count,
            exact=True,
            method="fixed-config",
        )
    return None


def _vision_encoder_count(
    bundle: ModelBundle,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult | None:
    del processor_kwargs
    if not getattr(bundle.config, "vision_config", None):
        return None
    try:
        from vllm.model_executor.models.vision import (
            get_num_selected_vision_tokens,
            get_vision_encoder_info,
        )

        encoder_info = get_vision_encoder_info(bundle.config)
        count = encoder_info.get_num_image_tokens(
            image_width=width,
            image_height=height,
        )
        strategy = getattr(bundle.config, "vision_feature_select_strategy", None)
        if strategy:
            count = get_num_selected_vision_tokens(count, strategy)
        return PlaceholderCountResult(
            count=int(count),
            exact=True,
            method="vllm-vision-encoder-info",
        )
    except Exception:
        return None


def _llava_next_count(
    bundle: ModelBundle,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult | None:
    del processor_kwargs
    grid_pinpoints = getattr(bundle.config, "image_grid_pinpoints", None)
    if not grid_pinpoints:
        return None
    try:
        from transformers.models.llava_next.modeling_llava_next import (
            get_anyres_image_grid_shape,
        )
        from vllm.model_executor.models.vision import (
            get_num_selected_vision_tokens,
            get_vision_encoder_info,
        )

        encoder_info = get_vision_encoder_info(bundle.config)
        base = get_num_selected_vision_tokens(
            encoder_info.get_num_image_tokens(
                image_width=width,
                image_height=height,
            ),
            getattr(bundle.config, "vision_feature_select_strategy", "default"),
        )
        patch_h, patch_w = get_anyres_image_grid_shape(
            image_size=(height, width),
            grid_pinpoints=grid_pinpoints,
            patch_size=encoder_info.get_image_size(),
        )
        unpadded, newline = _llava_unpadded_features(
            original_height=height,
            original_width=width,
            npatches=encoder_info.get_patch_grid_length(),
            num_patch_height=patch_h,
            num_patch_width=patch_w,
        )
        return PlaceholderCountResult(
            count=int(base + unpadded + newline),
            exact=True,
            method="vllm-llava-next-anyres",
        )
    except Exception:
        return None


def _qwen_smart_resize_count(
    bundle: ModelBundle,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult | None:
    if not _is_qwen_smart_resize_family(bundle):
        return None

    vision_config = getattr(bundle.config, "vision_config", None)
    patch_size = int(
        first_number(
            processor_kwargs,
            vision_config,
            bundle.attrs,
            names=("patch_size",),
        )
        or 14
    )
    merge_size = int(
        first_number(
            processor_kwargs,
            vision_config,
            bundle.attrs,
            names=("spatial_merge_size", "merge_size"),
        )
        or 2
    )
    factor = patch_size * merge_size
    min_pixels, max_pixels = _resolve_min_max_pixels(bundle, processor_kwargs, factor)
    resized_h, resized_w = smart_resize(height, width, factor, min_pixels, max_pixels)
    temporal_patch_size = int(
        first_number(
            processor_kwargs,
            vision_config,
            bundle.attrs,
            names=("temporal_patch_size",),
        )
        or 1
    )
    grid_t = max((1 + 1 % temporal_patch_size) // temporal_patch_size, 1)
    grid_h = resized_h // patch_size
    grid_w = resized_w // patch_size
    count = grid_t * grid_h * grid_w // (merge_size**2)
    exact = _is_verified_qwen_descendant(bundle)
    return PlaceholderCountResult(
        count=max(1, int(count)),
        exact=exact,
        method="vllm-qwen-smart-resize"
        if exact
        else "smart-resize-family-unverified",
    )


def _internvl_approx_count(
    bundle: ModelBundle,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> PlaceholderCountResult | None:
    del processor_kwargs
    family = "internvl" in bundle.model_type or "internvl" in bundle.model_name.lower()
    family = family or any(
        name in bundle.model_name.lower()
        for name in ("h2ovl", "nvlm", "nemotron", "eagle2")
    )
    if not family:
        return None

    image_size = int(
        first_number(bundle.attrs, names=("image_size",))
        or get_nested(bundle.attrs, "size", "height")
        or 448
    )
    patch_size = int(first_number(bundle.attrs, names=("patch_size",)) or 14)
    image_seq_length = int(first_number(bundle.attrs, names=("image_seq_length",)) or 1)
    tokens_per_tile = image_seq_length or (image_size // patch_size) ** 2
    max_num = int(
        first_number(
            bundle.attrs,
            names=("max_dynamic_patch", "max_num", "max_dynamic_patch_num"),
        )
        or 12
    )
    aspect = width / height
    best_tiles = 1
    best_diff = float("inf")
    for rows in range(1, max_num + 1):
        for cols in range(1, max_num + 1):
            tiles = rows * cols
            if tiles > max_num:
                continue
            diff = abs((cols / rows) - aspect)
            if diff < best_diff:
                best_diff = diff
                best_tiles = tiles
    return PlaceholderCountResult(
        count=max(1, best_tiles * tokens_per_tile),
        exact=False,
        method="internvl-target-ratio-approx",
    )


def _try_dimension_method(
    method: Callable[..., Any] | None,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> int:
    if not callable(method):
        return 0

    attempts = (
        {"image_width": width, "image_height": height},
        {"img_width": width, "img_height": height},
        {"width": width, "height": height},
        {"image_size": (height, width)},
    )
    for kwargs in attempts:
        kwargs = kwargs | _accepted_kwargs(method, processor_kwargs)
        try:
            count = _normalize_count(method(**kwargs))
            if count:
                return count
        except Exception:
            pass
    return 0


def _try_transformers_multimodal_helper(
    method: Callable[..., Any] | None,
    width: int,
    height: int,
    processor_kwargs: dict[str, Any],
) -> int:
    if not callable(method):
        return 0
    attempts = (
        {"image_sizes": [(height, width)]},
        {"image_sizes": [[height, width]]},
        {"images_kwargs": {"image_sizes": [(height, width)]}},
    )
    for kwargs in attempts:
        kwargs = kwargs | _accepted_kwargs(method, processor_kwargs)
        try:
            count = _normalize_count(method(**kwargs))
            if count:
                return count
        except Exception:
            pass
    return 0


def _accepted_kwargs(method: Callable[..., Any], values: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return {}
    if any(param.kind == param.VAR_KEYWORD for param in signature.parameters.values()):
        return values
    return {k: v for k, v in values.items() if k in signature.parameters}


def _normalize_count(value: Any) -> int:
    if isinstance(value, int) and value > 0:
        return value
    if hasattr(value, "item"):
        try:
            return _normalize_count(value.item())
        except Exception:
            return 0
    if isinstance(value, dict):
        for key in (
            "num_image_tokens",
            "image_tokens",
            "num_tokens",
            "tokens",
        ):
            count = _normalize_count(value.get(key))
            if count:
                return count
        if "image" in value:
            return _normalize_count(value["image"])
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _normalize_count(value[0])
    return 0


def _is_qwen_smart_resize_family(bundle: ModelBundle) -> bool:
    lower = f"{bundle.model_type} {bundle.model_name.lower()} {bundle.processor_class.lower()}"
    return any(
        name in lower
        for name in (
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
            "hunyuan",
            "keye",
            "dots",
            "opencua",
            "openpangu",
            "interns1_pro",
            "paddleocr",
            "ernie",
        )
    )


def _is_verified_qwen_descendant(bundle: ModelBundle) -> bool:
    lower = f"{bundle.model_type} {bundle.model_name.lower()} {bundle.processor_class.lower()}"
    return any(
        name in lower
        for name in (
            "qwen2_vl",
            "qwen2_5_vl",
            "qwen3_vl",
            "qwen3_5",
            "dots",
            "opencua",
            "openpangu",
            "tarsier2",
            "colqwen",
            "interns1_pro",
        )
    )


def _resolve_min_max_pixels(
    bundle: ModelBundle,
    processor_kwargs: dict[str, Any],
    factor: int,
) -> tuple[int, int]:
    min_pixels = first_number(
        processor_kwargs,
        bundle.attrs,
        names=("min_pixels", "shortest_edge"),
    )
    max_pixels = first_number(
        processor_kwargs,
        bundle.attrs,
        names=("max_pixels", "longest_edge"),
    )
    size = processor_kwargs.get("size") or bundle.attrs.get("size") or {}
    if isinstance(size, dict):
        min_pixels = min_pixels or size.get("shortest_edge")
        max_pixels = max_pixels or size.get("longest_edge")
    return (
        int(min_pixels or factor * factor),
        int(max_pixels or 1280 * 28 * 28),
    )


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)
    return h_bar, w_bar


def _llava_unpadded_features(
    *,
    original_height: int,
    original_width: int,
    npatches: int,
    num_patch_height: int,
    num_patch_width: int,
) -> tuple[int, int]:
    current_height = npatches * num_patch_height
    current_width = npatches * num_patch_width
    aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height
    if aspect_ratio > current_aspect_ratio:
        new_height = int(round(original_height * (current_width / original_width), 7))
        padding = (current_height - new_height) // 2
        current_height = current_height - (2 * padding)
    else:
        new_width = int(round(original_width * (current_height / original_height), 7))
        padding = (current_width - new_width) // 2
        current_width = current_width - (2 * padding)
    return current_height * current_width, current_height
