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

"""Image source helpers for metadata-only multimodal inspection."""

from __future__ import annotations

import base64
import io
import urllib.request

from PIL import Image

from .types import ImageDimensions


def image_bytes(item: dict[str, str]) -> bytes:
    data = item.get("data") or ""
    url = item.get("url") or ""
    if data:
        return _decode_data(data)
    if url.startswith("data:"):
        return _decode_data_url(url)
    if url.startswith("http://") or url.startswith("https://"):
        with urllib.request.urlopen(url, timeout=3) as response:
            return response.read()
    if url.startswith("file://"):
        with open(url.removeprefix("file://"), "rb") as f:
            return f.read()
    if url:
        with open(url, "rb") as f:
            return f.read()
    raise ValueError("image item has neither data nor url")


def probe_dimensions(raw: bytes) -> ImageDimensions:
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
    return ImageDimensions(width=width, height=height)


def load_rgb_image(raw: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(raw))
    image.load()
    return image.convert("RGB")


def _decode_data_url(value: str) -> bytes:
    _, encoded = value.split(",", 1)
    return base64.b64decode(encoded)


def _decode_data(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception:
        return value.encode()
