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

"""Configuration for RotorQuant KV cache compression."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def extract_layer_pattern_from_vllm_config(vllm_config) -> str:
    """Dynamically extract layer naming pattern from vLLM's instantiated model.
    
    This function uses vLLM's built-in `get_layers_from_vllm_config()` to query
    the actual model structure, eliminating the need for hardcoded pattern mappings.
    
    Args:
        vllm_config: vLLM configuration object with instantiated model
        
    Returns:
        Layer naming pattern with {layer_idx} placeholder
        
    Raises:
        ValueError: If no attention layers found in model
        ImportError: If vLLM imports fail
        
    Examples:
        >>> config = VllmConfig(...)  # Any model
        >>> extract_layer_pattern_from_vllm_config(config)
        'model.layers.{layer_idx}.self_attn'  # Extracted from actual model
    """
    try:
        # Import vLLM's dynamic layer extraction utilities
        from vllm.config import get_layers_from_vllm_config
        from vllm.attention import Attention
    except ImportError as e:
        raise ImportError(
            f"Failed to import vLLM utilities: {e}. "
            "Ensure vLLM is properly installed."
        ) from e
    
    # Get all attention layers from the instantiated model
    attn_layers = get_layers_from_vllm_config(vllm_config, Attention)
    
    if not attn_layers:
        raise ValueError(
            "No attention layers found in model. "
            "Ensure the model is properly initialized in vLLM config."
        )
    
    # Get first layer name (e.g., "model.layers.0.self_attn")
    first_layer_name = next(iter(attn_layers.keys()))
    
    # Extract pattern by replacing layer index with placeholder
    # "model.layers.0.self_attn" -> "model.layers.{layer_idx}.self_attn"
    # "transformer.h.5.attn" -> "transformer.h.{layer_idx}.attn"
    pattern = re.sub(r'\.(\d+)\.', r'.{layer_idx}.', first_layer_name)
    
    logger.info(
        f"Dynamically extracted layer pattern: {pattern} "
        f"(from {len(attn_layers)} attention layers)"
    )
    return pattern


@dataclass
class RotorQuantConfig:
    """Configuration for RotorQuant compression.
    
    Attributes:
        enabled: Whether to enable RotorQuant compression
        bits: Quantization bit width (1, 2, 3, 4, or 8)
        calibration_path: Path to calibration data file (.pt)
        layer_pattern_override: Optional custom layer pattern (auto-detected if None)
        compression_params: Optional compression parameters
    """
    
    enabled: bool = False
    bits: int = 3  # Default to 3-bit quantization
    calibration_path: Optional[str] = None
    layer_pattern_override: Optional[str] = None
    compression_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.enabled:
            # Validate bits parameter
            valid_bits = [1, 2, 3, 4, 8]
            if self.bits not in valid_bits:
                raise ValueError(
                    f"bits must be one of {valid_bits}, got: {self.bits}"
                )
            
            # Calibration path is optional for testing (if None, codec uses fallback)
            if self.calibration_path is None:
                logger.info("No calibration_path provided, codec will use fallback quantizers")
            else:
                # Validate calibration path exists
                calib_path = Path(self.calibration_path)
                if not calib_path.exists():
                    raise FileNotFoundError(
                        f"Calibration file not found: {self.calibration_path}"
                    )
                
                # Validate calibration file extension
                if calib_path.suffix != ".pt":
                    raise ValueError(
                        f"Calibration file must be .pt format, got: {calib_path.suffix}"
                    )
            
            # Validate layer pattern override if provided
            if self.layer_pattern_override:
                if "{layer_idx}" not in self.layer_pattern_override:
                    raise ValueError(
                        f"layer_pattern_override must contain '{{layer_idx}}': {self.layer_pattern_override}"
                    )
                logger.info(f"Using custom layer pattern: {self.layer_pattern_override}")
            
            # Validate compression params
            self._validate_compression_params()
            
            logger.info(f"RotorQuant compression enabled with calibration: {self.calibration_path}")
        else:
            logger.info("RotorQuant compression disabled")
    
    def _validate_compression_params(self):
        """Validate compression parameters."""
        if not self.compression_params:
            return
        
        # Validate mse_iterations
        if "mse_iterations" in self.compression_params:
            mse_iter = self.compression_params["mse_iterations"]
            if not isinstance(mse_iter, int) or mse_iter < 1:
                raise ValueError(
                    f"mse_iterations must be positive integer, got: {mse_iter}"
                )
        
        # Validate qjl_bits
        if "qjl_bits" in self.compression_params:
            qjl_bits = self.compression_params["qjl_bits"]
            if not isinstance(qjl_bits, int) or qjl_bits not in [1, 2, 4, 8]:
                raise ValueError(
                    f"qjl_bits must be 1, 2, 4, or 8, got: {qjl_bits}"
                )
        
        logger.info(f"Using compression params: {self.compression_params}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RotorQuantConfig":
        """Create RotorQuantConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary from YAML/JSON
            
        Returns:
            RotorQuantConfig instance
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If calibration file not found
        """
        return cls(
            enabled=config_dict.get("enabled", False),
            bits=config_dict.get("bits", 3),
            calibration_path=config_dict.get("calibration_path"),
            layer_pattern_override=config_dict.get("layer_pattern_override"),
            compression_params=config_dict.get("compression_params", {}),
        )

# Made with Bob
