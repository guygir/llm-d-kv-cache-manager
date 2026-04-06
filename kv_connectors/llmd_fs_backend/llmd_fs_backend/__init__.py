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

import logging
import os

from vllm.logger import init_logger

# Monkey-patch OffloadingConnector to disable cross-layer KV cache
# This enables IsoQuant/RotorQuant compression tensor swap to work correctly.
# Cross-layer format combines all layers into one tensor, which is incompatible
# with the current compression mechanism that requires per-layer tensors.
def _patch_offloading_connector():
    """Disable cross-layer blocks for compression compatibility."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnector
        )
        
        # Check if IsoQuant/RotorQuant is enabled via environment
        disable_cross_layer = os.environ.get("LLMD_DISABLE_CROSS_LAYER", "").lower() in ("1", "true", "yes")
        
        # Always disable for now since our compression requires it
        if True:  # or disable_cross_layer
            original_property = OffloadingConnector.prefer_cross_layer_blocks
            
            @property
            def patched_prefer_cross_layer_blocks(self) -> bool:
                return False
            
            OffloadingConnector.prefer_cross_layer_blocks = patched_prefer_cross_layer_blocks
            init_logger(__name__).info(
                "Patched OffloadingConnector.prefer_cross_layer_blocks to False "
                "for IsoQuant/RotorQuant compression compatibility"
            )
    except ImportError:
        pass  # vLLM connector not available
    except Exception as e:
        init_logger(__name__).warning(f"Failed to patch OffloadingConnector: {e}")

_patch_offloading_connector()

_LEVEL_MAP = {
    "TRACE": logging.DEBUG,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
}

# STORAGE_LOG_LEVEL controls log level (default: INFO).
_log_level_str = os.environ.get("STORAGE_LOG_LEVEL", "INFO").upper()
_logger = init_logger(__name__)
_logger.setLevel(_LEVEL_MAP.get(_log_level_str, logging.INFO))

# Ensure logger has a handler. vllm's init_logger creates loggers without handlers
if not _logger.handlers:
    _handler = logging.StreamHandler()
    # Set logger format
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    _logger.addHandler(_handler)
