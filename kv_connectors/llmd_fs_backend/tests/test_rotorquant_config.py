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

"""Unit tests for RotorQuant configuration."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from llmd_fs_backend.rotorquant_config import (
    RotorQuantConfig,
    extract_layer_pattern_from_vllm_config,
)


class TestRotorQuantConfig:
    """Test RotorQuantConfig dataclass."""

    def test_disabled_config(self):
        """Test that disabled config doesn't require calibration path."""
        config = RotorQuantConfig(enabled=False)
        assert config.enabled is False
        assert config.calibration_path is None

    def test_enabled_without_calibration_path(self):
        """Test that enabled config works without calibration path (uses fallback)."""
        # Calibration path is now optional - codec will use fallback quantizers
        config = RotorQuantConfig(enabled=True, calibration_path=None)
        assert config.enabled is True
        assert config.calibration_path is None
        assert config.bits == 3  # default value

    def test_enabled_with_nonexistent_calibration_file(self):
        """Test that calibration file must exist."""
        with pytest.raises(FileNotFoundError, match="Calibration file not found"):
            RotorQuantConfig(
                enabled=True,
                calibration_path="/nonexistent/path/calibration.pt"
            )

    def test_enabled_with_wrong_file_extension(self):
        """Test that calibration file must be .pt format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            try:
                with pytest.raises(ValueError, match="must be .pt format"):
                    RotorQuantConfig(enabled=True, calibration_path=f.name)
            finally:
                Path(f.name).unlink()

    def test_valid_config(self):
        """Test valid configuration."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            # Create a dummy calibration file
            torch.save({"dummy": "data"}, f.name)
            try:
                config = RotorQuantConfig(
                    enabled=True,
                    calibration_path=f.name
                )
                assert config.enabled is True
                assert config.calibration_path == f.name
            finally:
                Path(f.name).unlink()

    def test_layer_pattern_override_validation(self):
        """Test that layer_pattern_override must contain {layer_idx}."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"dummy": "data"}, f.name)
            try:
                # Invalid pattern (missing {layer_idx})
                with pytest.raises(ValueError, match="must contain"):
                    RotorQuantConfig(
                        enabled=True,
                        calibration_path=f.name,
                        layer_pattern_override="model.layers.0.self_attn"
                    )
                
                # Valid pattern
                config = RotorQuantConfig(
                    enabled=True,
                    calibration_path=f.name,
                    layer_pattern_override="model.layers.{layer_idx}.self_attn"
                )
                assert config.layer_pattern_override == "model.layers.{layer_idx}.self_attn"
            finally:
                Path(f.name).unlink()

    def test_bits_validation(self):
        """Test that bits parameter must be valid."""
        # Invalid bits
        with pytest.raises(ValueError, match="bits must be one of"):
            RotorQuantConfig(enabled=True, bits=5)
        
        with pytest.raises(ValueError, match="bits must be one of"):
            RotorQuantConfig(enabled=True, bits=0)
        
        # Valid bits values
        for valid_bits in [1, 2, 3, 4, 8]:
            config = RotorQuantConfig(enabled=True, bits=valid_bits)
            assert config.bits == valid_bits

    def test_compression_params_validation(self):
        """Test compression parameters validation."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"dummy": "data"}, f.name)
            try:
                # Invalid mse_iterations
                with pytest.raises(ValueError, match="mse_iterations must be positive"):
                    RotorQuantConfig(
                        enabled=True,
                        calibration_path=f.name,
                        compression_params={"mse_iterations": 0}
                    )
                
                # Invalid qjl_bits
                with pytest.raises(ValueError, match="qjl_bits must be"):
                    RotorQuantConfig(
                        enabled=True,
                        calibration_path=f.name,
                        compression_params={"qjl_bits": 3}
                    )
                
                # Valid params
                config = RotorQuantConfig(
                    enabled=True,
                    calibration_path=f.name,
                    compression_params={"mse_iterations": 10, "qjl_bits": 1}
                )
                assert config.compression_params["mse_iterations"] == 10
                assert config.compression_params["qjl_bits"] == 1
            finally:
                Path(f.name).unlink()

    def test_from_dict(self):
        """Test creating config from dictionary."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({"dummy": "data"}, f.name)
            try:
                config_dict = {
                    "enabled": True,
                    "bits": 4,
                    "calibration_path": f.name,
                    "layer_pattern_override": "model.layers.{layer_idx}.self_attn",
                    "compression_params": {"mse_iterations": 10}
                }
                config = RotorQuantConfig.from_dict(config_dict)
                assert config.enabled is True
                assert config.bits == 4
                assert config.calibration_path == f.name
                assert config.layer_pattern_override == "model.layers.{layer_idx}.self_attn"
                assert config.compression_params["mse_iterations"] == 10
            finally:
                Path(f.name).unlink()
    
    def test_from_dict_defaults(self):
        """Test that from_dict uses defaults for missing fields."""
        config = RotorQuantConfig.from_dict({"enabled": True})
        assert config.enabled is True
        assert config.bits == 3  # default
        assert config.calibration_path is None


class TestExtractLayerPatternFromVllmConfig:
    """Test extract_layer_pattern_from_vllm_config function."""

    def _create_mock_config_with_layers(self, layer_names):
        """Helper to create mock config with attention layers."""
        mock_config = Mock()
        
        # Mock the static_forward_context with attention layers
        mock_attention = Mock()
        mock_attention.__class__.__name__ = "Attention"
        
        forward_context = {name: mock_attention for name in layer_names}
        mock_config.compilation_config.static_forward_context = forward_context
        
        return mock_config

    def test_llama_style_pattern(self):
        """Test pattern extraction for Llama-style models."""
        layer_names = [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
            "model.layers.2.self_attn",
        ]
        mock_config = self._create_mock_config_with_layers(layer_names)
        
        pattern = extract_layer_pattern_from_vllm_config(mock_config)
        assert pattern == "model.layers.{layer_idx}.self_attn"

    def test_gpt2_style_pattern(self):
        """Test pattern extraction for GPT-2 style models."""
        layer_names = [
            "transformer.h.0.attn",
            "transformer.h.1.attn",
            "transformer.h.2.attn",
        ]
        mock_config = self._create_mock_config_with_layers(layer_names)
        
        pattern = extract_layer_pattern_from_vllm_config(mock_config)
        assert pattern == "transformer.h.{layer_idx}.attn"

    def test_opt_style_pattern(self):
        """Test pattern extraction for OPT-style models."""
        layer_names = [
            "model.decoder.layers.0.self_attn",
            "model.decoder.layers.1.self_attn",
        ]
        mock_config = self._create_mock_config_with_layers(layer_names)
        
        pattern = extract_layer_pattern_from_vllm_config(mock_config)
        assert pattern == "model.decoder.layers.{layer_idx}.self_attn"

    def test_custom_architecture_pattern(self):
        """Test pattern extraction for custom architecture."""
        layer_names = [
            "custom.encoder.blocks.5.attention",
            "custom.encoder.blocks.6.attention",
        ]
        mock_config = self._create_mock_config_with_layers(layer_names)
        
        pattern = extract_layer_pattern_from_vllm_config(mock_config)
        assert pattern == "custom.encoder.blocks.{layer_idx}.attention"

    def test_no_attention_layers(self):
        """Test that missing attention layers raises ValueError."""
        mock_config = Mock()
        mock_config.compilation_config.static_forward_context = {}
        
        with pytest.raises(ValueError, match="No attention layers found"):
            extract_layer_pattern_from_vllm_config(mock_config)

    def test_pattern_contains_placeholder(self):
        """Test that extracted pattern contains {layer_idx} placeholder."""
        layer_names = ["model.layers.0.self_attn"]
        mock_config = self._create_mock_config_with_layers(layer_names)
        
        pattern = extract_layer_pattern_from_vllm_config(mock_config)
        assert "{layer_idx}" in pattern

    def test_multiple_digit_layer_indices(self):
        """Test pattern extraction with multi-digit layer indices."""
        layer_names = [
            "model.layers.99.self_attn",
            "model.layers.100.self_attn",
        ]
        mock_config = self._create_mock_config_with_layers(layer_names)
        
        pattern = extract_layer_pattern_from_vllm_config(mock_config)
        assert pattern == "model.layers.{layer_idx}.self_attn"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
