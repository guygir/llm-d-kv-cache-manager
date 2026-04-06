# Copyright 2025 The llm-d Authors.
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

"""Unit tests for IsoQuantConfig."""

import pytest
import tempfile
import os

from llmd_fs_backend.isoquant_config import IsoQuantConfig


class TestIsoQuantConfig:
    """Test IsoQuantConfig initialization and validation."""

    def test_disabled_by_default(self):
        """Test that IsoQuant is disabled by default."""
        config = IsoQuantConfig()
        assert config.enabled is False
        assert config.bits == 3
        assert config.mode == "fast"

    def test_enabled_with_defaults(self):
        """Test enabling IsoQuant with default settings."""
        config = IsoQuantConfig(enabled=True)
        assert config.enabled is True
        assert config.bits == 3  # Default 3-bit
        assert config.mode == "fast"  # Default fast mode
        assert config.calibration_path is None

    def test_full_mode(self):
        """Test enabling full mode (q_L v q̄_R)."""
        config = IsoQuantConfig(enabled=True, mode="full")
        assert config.mode == "full"

    def test_invalid_mode_raises(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be one of"):
            IsoQuantConfig(enabled=True, mode="invalid")

    def test_bits_validation_valid(self):
        """Test valid bits values (1, 2, 3, 4, 8)."""
        for bits in [1, 2, 3, 4, 8]:
            config = IsoQuantConfig(enabled=True, bits=bits)
            assert config.bits == bits

    def test_bits_validation_invalid(self):
        """Test that invalid bits values raise ValueError."""
        for invalid_bits in [0, 5, 6, 7, 16]:
            with pytest.raises(ValueError, match="bits must be one of"):
                IsoQuantConfig(enabled=True, bits=invalid_bits)

    def test_enabled_without_calibration_path(self):
        """Test enabling without calibration_path is allowed (uses default quantizers)."""
        config = IsoQuantConfig(enabled=True, calibration_path=None)
        assert config.calibration_path is None

    def test_enabled_with_invalid_calibration_path(self):
        """Test that invalid calibration_path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            IsoQuantConfig(enabled=True, calibration_path="/nonexistent/path.pt")

    def test_enabled_with_wrong_extension(self):
        """Test that wrong calibration file extension raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name
        try:
            with pytest.raises(ValueError, match="must be .pt format"):
                IsoQuantConfig(enabled=True, calibration_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_enabled_with_valid_calibration(self):
        """Test enabling with valid calibration file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            temp_path = f.name
        try:
            config = IsoQuantConfig(enabled=True, calibration_path=temp_path)
            assert config.calibration_path == temp_path
        finally:
            os.unlink(temp_path)

    def test_layer_pattern_override_valid(self):
        """Test valid layer pattern override."""
        config = IsoQuantConfig(
            enabled=True,
            layer_pattern_override="model.layers.{layer_idx}.self_attn"
        )
        assert config.layer_pattern_override == "model.layers.{layer_idx}.self_attn"

    def test_layer_pattern_override_missing_placeholder(self):
        """Test that layer pattern without {layer_idx} raises ValueError."""
        with pytest.raises(ValueError, match="must contain"):
            IsoQuantConfig(
                enabled=True,
                layer_pattern_override="model.layers.0.self_attn"
            )

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "enabled": True,
            "bits": 4,
            "mode": "full",
            "layer_pattern_override": "model.layers.{layer_idx}.attention",
        }
        config = IsoQuantConfig.from_dict(config_dict)
        assert config.enabled is True
        assert config.bits == 4
        assert config.mode == "full"
        assert config.layer_pattern_override == "model.layers.{layer_idx}.attention"

    def test_from_dict_defaults(self):
        """Test from_dict with minimal config uses defaults."""
        config = IsoQuantConfig.from_dict({"enabled": True})
        assert config.enabled is True
        assert config.bits == 3  # Default
        assert config.mode == "fast"  # Default
        assert config.calibration_path is None

    def test_compression_ratio_property(self):
        """Test compression_ratio property calculation."""
        config = IsoQuantConfig(enabled=True, bits=3)
        ratio = config.compression_ratio
        assert ratio > 4.0  # 16 / (3 + 0.5) ≈ 4.57
        assert ratio < 6.0


class TestIsoQuantConfigEdgeCases:
    """Test edge cases and corner scenarios."""

    def test_disabled_skips_validation(self):
        """Test that disabled config skips all validation."""
        config = IsoQuantConfig(
            enabled=False,
            bits=999,  # Would be invalid if enabled
            mode="invalid",  # Would be invalid if enabled
            calibration_path="/nonexistent.pt",  # Would fail if enabled
        )
        assert config.enabled is False

    def test_from_dict_empty(self):
        """Test from_dict with empty dictionary."""
        config = IsoQuantConfig.from_dict({})
        assert config.enabled is False

    def test_from_dict_with_extra_keys(self):
        """Test from_dict ignores extra keys."""
        config = IsoQuantConfig.from_dict({
            "enabled": True,
            "unknown_key": "value",
            "another_unknown": 123,
        })
        assert config.enabled is True
        assert not hasattr(config, "unknown_key")
