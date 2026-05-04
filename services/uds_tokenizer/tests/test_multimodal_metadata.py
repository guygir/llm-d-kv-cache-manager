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

import base64
import io

from PIL import Image

from tokenizer_service.multimodal_metadata import (
    MultiModalMetadataService,
    UnsupportedCounterError,
    _qwen_placeholder_count,
)
from tokenizer_service.mm_metadata import hashing, model_bundle, placeholder, preprocess, service
from tokenizer_service.mm_metadata.types import ModelBundle, PlaceholderCountResult


def _png_data_url(width: int, height: int) -> str:
    image = Image.new("RGB", (width, height), color=(255, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_qwen_placeholder_count_scales_with_image_size():
    small = _qwen_placeholder_count(224, 224, {}, {})
    large = _qwen_placeholder_count(1024, 1024, {}, {})

    assert small > 0
    assert large > small


def test_uuid_metadata_uses_exact_hash_and_dimensions(monkeypatch):
    service = MultiModalMetadataService()

    monkeypatch.setattr(
        service,
        "_count_placeholders",
        lambda model_name, width, height, kwargs: width // 16 + height // 16,
    )

    items = service.get_metadata(
        "test-model",
        [
            {
                "modality": "image",
                "url": _png_data_url(64, 32),
                "uuid": "client-uuid",
            }
        ],
        hash_mode="uuid",
    )

    assert len(items) == 1
    assert items[0].mm_hash == "client-uuid"
    assert items[0].exact_hash is True
    assert items[0].exact_placeholder_count is True
    assert items[0].placeholder_count == 6


def test_unsupported_counter_is_explicit(monkeypatch):
    service = MultiModalMetadataService()

    def unsupported(model_name, width, height, kwargs):
        raise UnsupportedCounterError("unsupported")

    monkeypatch.setattr(service, "_count_placeholders", unsupported)
    items = service.get_metadata(
        "unknown-model",
        [{"modality": "image", "url": _png_data_url(16, 16)}],
        hash_mode="stable-identifier",
    )

    assert len(items) == 1
    assert items[0].exact_placeholder_count is False
    assert "unsupported" in items[0].error
    assert items[0].method.endswith("+unsupported")


def test_preprocess_fallback_runs_when_enabled(monkeypatch):
    metadata_service = MultiModalMetadataService()

    def unsupported(model_name, width, height, kwargs):
        raise UnsupportedCounterError("unsupported")

    monkeypatch.setattr(metadata_service, "_count_placeholders", unsupported)
    monkeypatch.setattr(
        service,
        "count_placeholders_from_preprocess",
        lambda model_name, raw, kwargs: PlaceholderCountResult(
            count=77,
            exact=True,
            method="preprocess-fallback.image_processor.num_image_tokens",
        ),
    )

    items = metadata_service.get_metadata(
        "fallback-model",
        [{"modality": "image", "url": _png_data_url(16, 16)}],
        allow_preprocess_fallback=True,
        hash_mode="stable-identifier",
    )

    assert items[0].placeholder_count == 77
    assert items[0].exact_placeholder_count is True
    assert items[0].method.endswith("+preprocess-fallback.image_processor.num_image_tokens")


def test_preprocess_fallback_extracts_image_grid_thw(monkeypatch):
    class ImageProcessor:
        def __call__(self, **kwargs):
            assert "images" in kwargs
            return {"image_grid_thw": [[1, 32, 32]]}

    bundle = ModelBundle(
        model_name="grid-model",
        config=object(),
        processor=object(),
        image_processor=ImageProcessor(),
        model_type="qwen2_vl",
        processor_class="Processor",
        image_processor_class="ImageProcessor",
        attrs={"spatial_merge_size": 2},
    )
    monkeypatch.setattr(preprocess, "get_model_bundle", lambda model_name: bundle)

    result = preprocess.count_placeholders_from_preprocess(
        "grid-model",
        base64.b64decode(_png_data_url(16, 16).split(",", 1)[1]),
        {},
    )

    assert result.count == 256
    assert result.exact is True
    assert result.method == "preprocess-fallback.image_processor.image_grid_thw"


def test_preprocess_fallback_prefers_multimodal_token_helper(monkeypatch):
    class Processor:
        def _get_num_multimodal_tokens(self, *, image_sizes):
            assert image_sizes == [(16, 16)]
            return {"num_image_tokens": [123]}

    class ImageProcessor:
        def __call__(self, **kwargs):
            raise AssertionError("processor helper should run before preprocessing")

    bundle = ModelBundle(
        model_name="helper-model",
        config=object(),
        processor=Processor(),
        image_processor=ImageProcessor(),
        model_type="test",
        processor_class="Processor",
        image_processor_class="ImageProcessor",
        attrs={},
    )
    monkeypatch.setattr(preprocess, "get_model_bundle", lambda model_name: bundle)

    result = preprocess.count_placeholders_from_preprocess(
        "helper-model",
        base64.b64decode(_png_data_url(16, 16).split(",", 1)[1]),
        {},
    )

    assert result.count == 123
    assert result.exact is True
    assert result.method == "preprocess-fallback.processor._get_num_multimodal_tokens"


def test_preprocess_fallback_treats_num_image_patches_as_approximate(monkeypatch):
    class ImageProcessor:
        def __call__(self, **kwargs):
            return {"num_image_patches": [4]}

    bundle = ModelBundle(
        model_name="patch-model",
        config=object(),
        processor=object(),
        image_processor=ImageProcessor(),
        model_type="test",
        processor_class="Processor",
        image_processor_class="ImageProcessor",
        attrs={"image_seq_length": 16},
    )
    monkeypatch.setattr(preprocess, "get_model_bundle", lambda model_name: bundle)

    result = preprocess.count_placeholders_from_preprocess(
        "patch-model",
        base64.b64decode(_png_data_url(16, 16).split(",", 1)[1]),
        {},
    )

    assert result.count == 64
    assert result.exact is False
    assert result.method == (
        "preprocess-fallback.image_processor.num_patches_x_image_seq_length"
    )


def test_preprocess_fallback_uses_spatial_shapes_as_approximate(monkeypatch):
    class ImageProcessor:
        def __call__(self, **kwargs):
            return {"spatial_shapes": [[12, 10]]}

    bundle = ModelBundle(
        model_name="spatial-model",
        config=object(),
        processor=object(),
        image_processor=ImageProcessor(),
        model_type="test",
        processor_class="Processor",
        image_processor_class="ImageProcessor",
        attrs={},
    )
    monkeypatch.setattr(preprocess, "get_model_bundle", lambda model_name: bundle)

    result = preprocess.count_placeholders_from_preprocess(
        "spatial-model",
        base64.b64decode(_png_data_url(16, 16).split(",", 1)[1]),
        {},
    )

    assert result.count == 120
    assert result.exact is False
    assert result.method == "preprocess-fallback.image_processor.spatial_shapes"


def test_preprocess_fallback_raw_area_proxy_when_no_count_metadata(monkeypatch):
    class ImageProcessor:
        def __call__(self, **kwargs):
            return {"pixel_values": [1]}

    bundle = ModelBundle(
        model_name="raw-proxy-model",
        config=object(),
        processor=object(),
        image_processor=ImageProcessor(),
        model_type="test",
        processor_class="Processor",
        image_processor_class="ImageProcessor",
        attrs={},
    )
    monkeypatch.setattr(preprocess, "get_model_bundle", lambda model_name: bundle)

    result = preprocess.count_placeholders_from_preprocess(
        "raw-proxy-model",
        base64.b64decode(_png_data_url(64, 32).split(",", 1)[1]),
        {},
    )

    assert result.count == 3
    assert result.exact is False
    assert result.method.startswith("preprocess-fallback.raw-pixel-area-proxy")


def test_stable_identifier_uses_dimension_only_path(monkeypatch):
    service = MultiModalMetadataService()

    def fail_rgb_load(raw):
        raise AssertionError("stable-identifier should not decode RGB pixels")

    monkeypatch.setattr(hashing, "load_rgb_image", fail_rgb_load)
    monkeypatch.setattr(
        service,
        "_count_placeholders",
        lambda model_name, width, height, kwargs: width + height,
    )

    items = service.get_metadata(
        "test-model",
        [{"modality": "image", "url": _png_data_url(64, 32)}],
        hash_mode="stable-identifier",
    )

    assert items[0].width == 64
    assert items[0].height == 32
    assert items[0].placeholder_count == 96


def test_processor_helper_strategy_is_exact(monkeypatch):
    class Processor:
        def get_num_image_tokens(self, *, image_width, image_height):
            return image_width + image_height

    bundle = ModelBundle(
        model_name="processor-helper",
        config=object(),
        processor=Processor(),
        image_processor=None,
        model_type="test",
        processor_class="Processor",
        image_processor_class="",
        attrs={},
    )
    monkeypatch.setattr(placeholder, "get_model_bundle", lambda model_name: bundle)

    result = placeholder.count_placeholders("processor-helper", 20, 10, {})

    assert result.count == 30
    assert result.exact is True
    assert result.method == "processor.get_num_image_tokens"


def test_image_processor_helper_strategy_is_exact(monkeypatch):
    class ImageProcessor:
        def calc_num_image_tokens_from_image_size(self, *, width, height):
            return width * height // 64

    bundle = ModelBundle(
        model_name="image-processor-helper",
        config=object(),
        processor=object(),
        image_processor=ImageProcessor(),
        model_type="test",
        processor_class="Processor",
        image_processor_class="ImageProcessor",
        attrs={},
    )
    monkeypatch.setattr(placeholder, "get_model_bundle", lambda model_name: bundle)

    result = placeholder.count_placeholders("image-processor-helper", 32, 16, {})

    assert result.count == 8
    assert result.exact is True
    assert result.method == "image_processor.calc_num_image_tokens_from_image_size"


def test_processor_attrs_avoid_expensive_top_level_to_dict():
    class ImageProcessor:
        patch_size = 14

        def to_dict(self):
            return {"merge_size": 2}

    class Processor:
        image_processor = ImageProcessor()

        def to_dict(self):
            raise AssertionError("top-level processor.to_dict must stay off hot path")

    attrs = model_bundle._processor_attrs(Processor(), Processor().image_processor)

    assert attrs["patch_size"] == 14
    assert attrs["merge_size"] == 2


def test_internvl_approximation_is_not_marked_exact(monkeypatch):
    bundle = ModelBundle(
        model_name="internvl-test",
        config=object(),
        processor=object(),
        image_processor=object(),
        model_type="internvl_chat",
        processor_class="Processor",
        image_processor_class="ImageProcessor",
        attrs={
            "image_size": 448,
            "patch_size": 14,
            "image_seq_length": 256,
            "max_num": 4,
        },
    )
    monkeypatch.setattr(placeholder, "get_model_bundle", lambda model_name: bundle)

    result = placeholder.count_placeholders("internvl-test", 640, 480, {})

    assert result.count > 0
    assert result.exact is False
    assert result.method == "internvl-target-ratio-approx"
