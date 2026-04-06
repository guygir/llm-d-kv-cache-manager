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

"""Configuration for IsoQuant KV cache compression.

IsoQuant uses quaternion-based 4D block rotations for vector quantization.
It provides better quality than RotorQuant with fewer FMAs (512 vs 2,400).

Modes:
- 'full': T(v) = q_L v q̄_R  — full SO(4), 6 DOF per block, best quality
- 'fast': T(v) = q_L v      — isoclinic SO(3) subgroup, 3 DOF, faster
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

logger = logging.getLogger(__name__)


@dataclass
class IsoQuantConfig:
    """Configuration for IsoQuant compression.
    
    Attributes:
        enabled: Whether to enable IsoQuant compression
        bits: Quantization bit width (1, 2, 3, 4, or 8)
        mode: 'full' (best quality) or 'fast' (faster)
        calibration_path: Optional path to calibration data file (.pt)
        layer_pattern_override: Optional custom layer pattern (auto-detected if None)
    """
    
    enabled: bool = False
    bits: int = 3  # Default to 3-bit quantization (best quality/size tradeoff)
    mode: Literal['full', 'fast'] = 'fast'  # 'fast' is recommended default
    calibration_path: Optional[str] = None
    layer_pattern_override: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.enabled:
            # Validate bits parameter
            valid_bits = [1, 2, 3, 4, 8]
            if self.bits not in valid_bits:
                raise ValueError(
                    f"bits must be one of {valid_bits}, got: {self.bits}"
                )
            
            # Validate mode parameter
            valid_modes = ['full', 'fast']
            if self.mode not in valid_modes:
                raise ValueError(
                    f"mode must be one of {valid_modes}, got: {self.mode}"
                )
            
            # Calibration path is optional (quantizers work without calibration)
            if self.calibration_path is None:
                logger.info("No calibration_path provided, using default quantizers")
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
            
            logger.info(
                f"IsoQuant compression enabled: bits={self.bits}, mode={self.mode}"
            )
        else:
            logger.info("IsoQuant compression disabled")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "IsoQuantConfig":
        """Create IsoQuantConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary from YAML/JSON
            
        Returns:
            IsoQuantConfig instance
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If calibration file not found
        """
        return cls(
            enabled=config_dict.get("enabled", False),
            bits=config_dict.get("bits", 3),
            mode=config_dict.get("mode", "fast"),
            calibration_path=config_dict.get("calibration_path"),
            layer_pattern_override=config_dict.get("layer_pattern_override"),
        )
    
    @property
    def compression_ratio(self) -> float:
        """Estimated compression ratio based on bit width.
        
        FP16 = 16 bits per element
        IsoQuant at b bits = b bits per element + overhead for norms
        """
        # Approximate ratio (actual depends on overhead)
        return 16.0 / (self.bits + 0.5)  # +0.5 for norm storage overhead
