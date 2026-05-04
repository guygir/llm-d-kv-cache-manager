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

"""Cached model and processor metadata for placeholder counting."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from transformers import AutoConfig, AutoProcessor

from .types import ModelBundle


@lru_cache(maxsize=16)
def get_model_bundle(model_name: str) -> ModelBundle:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    image_processor = getattr(processor, "image_processor", None)
    attrs = _processor_attrs(processor, image_processor)
    return ModelBundle(
        model_name=model_name,
        config=config,
        processor=processor,
        image_processor=image_processor,
        model_type=(getattr(config, "model_type", "") or "").lower(),
        processor_class=processor.__class__.__name__,
        image_processor_class=image_processor.__class__.__name__
        if image_processor is not None
        else "",
        attrs=attrs,
    )


def _processor_attrs(processor: Any, image_processor: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for obj in (image_processor, processor):
        if obj is None:
            continue
        attrs.update(getattr(obj, "__dict__", {}))
        # Some top-level ProcessorMixin.to_dict() implementations serialize
        # internals and cost hundreds of milliseconds. Image processors own most
        # geometry fields and are cheap to serialize for the audited families.
        if obj is image_processor and hasattr(obj, "to_dict"):
            try:
                attrs.update(obj.to_dict())
            except Exception:
                pass
    return attrs


def first_int(*sources: Any, names: tuple[str, ...]) -> int:
    for source in sources:
        if source is None:
            continue
        for name in names:
            value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return 0


def first_number(*sources: Any, names: tuple[str, ...]) -> int | float:
    for source in sources:
        if source is None:
            continue
        for name in names:
            value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
            if isinstance(value, (int, float)) and value > 0:
                return value
    return 0


def get_nested(source: Any, *names: str) -> Any:
    value = source
    for name in names:
        if value is None:
            return None
        value = value.get(name) if isinstance(value, dict) else getattr(value, name, None)
    return value
