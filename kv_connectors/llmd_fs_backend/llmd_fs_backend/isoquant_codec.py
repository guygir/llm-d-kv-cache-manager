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

"""IsoQuant codec for KV cache compression.

IsoQuant uses quaternion-based 4D block rotations, providing:
- Better quality than RotorQuant (PPL 6.91 vs 7.07 at 3-bit)
- 4.7x fewer FMAs (512 vs 2,400 for d=128)
- Clean 4D alignment (no tail handling for power-of-2 dims)

This codec integrates with llmd_fs_backend for storage offloading.
It uses batch-level operations for efficiency, matching the upstream API.
"""

import torch
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from .isoquant_config import IsoQuantConfig

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """Tracks compression performance metrics."""
    total_encode_time_ms: float = 0.0
    total_decode_time_ms: float = 0.0
    total_bytes_original: int = 0
    total_bytes_compressed: int = 0
    num_encode_calls: int = 0
    num_decode_calls: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.total_bytes_compressed == 0:
            return 0.0
        return self.total_bytes_original / self.total_bytes_compressed

    @property
    def avg_encode_time_ms(self) -> float:
        if self.num_encode_calls == 0:
            return 0.0
        return self.total_encode_time_ms / self.num_encode_calls

    @property
    def avg_decode_time_ms(self) -> float:
        if self.num_decode_calls == 0:
            return 0.0
        return self.total_decode_time_ms / self.num_decode_calls


class IsoQuantCodec:
    """
    IsoQuant compression codec for KV cache tensors.
    
    Uses quaternion-based 4D block rotations for vector quantization.
    - Keys: IsoQuantProd (MSE + QJL for unbiased inner products)
    - Values: IsoQuantMSE (MSE only)
    
    Design principles (matching upstream):
    - Batch operations for GPU efficiency
    - Direct index storage (no complex bit-packing)
    - Separate norm storage for quality preservation
    """

    def __init__(
        self,
        config: IsoQuantConfig,
        layer_names: List[str],
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize IsoQuant codec.
        
        Args:
            config: IsoQuant configuration
            layer_names: List of layer names (e.g., ["model.layers.0.self_attn", ...])
            num_blocks: Number of blocks in batch buffer (NOT total KV cache blocks)
            block_size: Tokens per block
            num_heads: Number of attention heads
            head_size: Dimension per head (d)
            device: Target device (cuda)
            dtype: Original tensor dtype (fp16/bf16)
        """
        self.config = config
        self.layer_names = layer_names
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.device = device
        self.dtype = dtype
        
        # Calculate compression parameters
        self.bits = config.bits
        self.mode = config.mode
        self.n_groups = (head_size + 3) // 4  # 4D blocks for quaternions
        self.n_levels = 2 ** self.bits
        
        # Metrics tracking
        self._metrics = CompressionMetrics()
        
        # Initialize quantizers
        self._quantizers = self._initialize_quantizers()
        
        # Pre-allocate encoded buffers for batch operations
        self._encoded_buffers = self._allocate_encoded_buffers()
        
        logger.info(
            f"IsoQuantCodec initialized: {len(layer_names)} layers, "
            f"{num_blocks} blocks/batch, {block_size} tokens/block, "
            f"{num_heads} heads, {head_size} head_dim, "
            f"bits={self.bits}, mode={self.mode}, device={device}"
        )

    def _initialize_quantizers(self) -> Dict[str, Tuple[Any, Any]]:
        """
        Initialize per-layer IsoQuant quantizers.
        
        Returns:
            Dict mapping layer_name -> (key_quantizer, value_quantizer)
        """
        try:
            from turboquant import IsoQuantProd, IsoQuantMSE
            
            quantizers = {}
            for i, layer_name in enumerate(self.layer_names):
                # Keys use IsoQuantProd (MSE + QJL for unbiased inner products)
                key_quantizer = IsoQuantProd(
                    d=self.head_size,
                    bits=self.bits,
                    mode=self.mode,
                    seed=42 + i,
                    device=str(self.device)
                )
                
                # Values use IsoQuantMSE (MSE only, no QJL needed)
                value_quantizer = IsoQuantMSE(
                    d=self.head_size,
                    bits=self.bits,
                    mode=self.mode,
                    seed=1000 + i,
                    device=str(self.device)
                )
                
                quantizers[layer_name] = (key_quantizer, value_quantizer)
                
            logger.info(f"Initialized IsoQuant quantizers for {len(quantizers)} layers")
            return quantizers
            
        except ImportError as e:
            logger.error(f"Failed to import IsoQuant from turboquant: {e}")
            logger.warning("Falling back to placeholder quantizers (no compression)")
            return {name: (None, None) for name in self.layer_names}

    def _allocate_encoded_buffers(self) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
        """
        Pre-allocate GPU buffers for encoded data.
        
        IsoQuant stores:
        - indices: uint8 tensor of quantization indices
        - norms: float16 tensor of vector norms
        
        Returns:
            Dict: layer_name -> {"k": {"indices": tensor, "norms": tensor}, "v": {...}}
        """
        buffers = {}
        
        # Calculate buffer shapes
        # indices: [num_blocks, block_size, num_heads, n_groups * 4]
        # norms: [num_blocks, block_size, num_heads]
        indices_shape = (self.num_blocks, self.block_size, self.num_heads, self.n_groups * 4)
        norms_shape = (self.num_blocks, self.block_size, self.num_heads)
        
        for layer_name in self.layer_names:
            buffers[layer_name] = {
                "k": {
                    "indices": torch.zeros(indices_shape, dtype=torch.uint8, device=self.device),
                    "norms": torch.zeros(norms_shape, dtype=torch.float16, device=self.device),
                },
                "v": {
                    "indices": torch.zeros(indices_shape, dtype=torch.uint8, device=self.device),
                    "norms": torch.zeros(norms_shape, dtype=torch.float16, device=self.device),
                },
            }
        
        # Calculate total memory
        indices_bytes = self.num_blocks * self.block_size * self.num_heads * self.n_groups * 4
        norms_bytes = self.num_blocks * self.block_size * self.num_heads * 2  # float16
        total_mb = len(self.layer_names) * 2 * (indices_bytes + norms_bytes) / (1024 * 1024)
        
        logger.info(f"Allocated {total_mb:.1f} MB of encoded buffers")
        return buffers

    def encode_batch(
        self,
        layer_name: str,
        kv_type: str,
        source_tensor: torch.Tensor,
        source_block_ids: List[int],
        dest_buffer: torch.Tensor,
        dest_start_idx: int = 0,
    ) -> None:
        """
        Encode blocks from source KV cache into batch buffer.
        
        Uses batch-level operations for GPU efficiency (not per-block loops).
        
        Args:
            layer_name: Layer identifier
            kv_type: "k" for keys, "v" for values
            source_tensor: Full KV cache, shape (total_blocks, block_size, num_heads, head_size)
            source_block_ids: Block IDs to encode
            dest_buffer: Destination buffer (encoded indices tensor)
            dest_start_idx: Starting index in dest_buffer
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        key_quantizer, value_quantizer = self._quantizers[layer_name]
        quantizer = key_quantizer if kv_type == "k" else value_quantizer
        
        if quantizer is not None:
            # Check if source_tensor is already sliced (cross-layer mode) or needs indexing
            # If source_block_ids starts at 0 and is contiguous, tensor might be pre-sliced
            if len(source_block_ids) > 0 and source_block_ids[0] == 0 and \
               source_block_ids == list(range(len(source_block_ids))) and \
               source_tensor.shape[0] == len(source_block_ids):
                # Tensor is already sliced to the correct blocks
                blocks = source_tensor
            else:
                # Extract blocks as batch: [num_blocks, block_size, num_heads, head_size]
                block_ids_tensor = torch.tensor(source_block_ids, device=self.device)
                blocks = source_tensor[block_ids_tensor]  # Batch indexing
            
            # Reshape for quantization: [N, head_size] where N = blocks * tokens * heads
            original_shape = blocks.shape
            vectors = blocks.reshape(-1, self.head_size).float()
            
            # Quantize entire batch at once
            # IsoQuantProd.quantize() returns dict with 'mse_indices' (which has 'indices', '_norms')
            # IsoQuantMSE() returns (x_hat, {'indices': ..., '_norms': ...})
            if kv_type == "k":
                # Keys use IsoQuantProd (with QJL for unbiased inner products)
                compressed = quantizer.quantize(vectors)
                indices = compressed['mse_indices']['indices']
                norms = compressed['mse_indices']['_norms']
            else:
                # Values use IsoQuantMSE (MSE only)
                _, result = quantizer(vectors)
                indices = result['indices']
                norms = result['_norms']
            
            # Reshape indices back to block structure
            # Upstream IsoQuant indices shape: [N, n_groups * 4] where n_groups = ceil(head_size / 4)
            num_blocks = len(source_block_ids)
            n_components = indices.shape[-1] if indices.dim() > 1 else self.n_groups * 4
            indices_reshaped = indices.reshape(num_blocks, self.block_size, self.num_heads, n_components)
            norms_reshaped = norms.reshape(num_blocks, self.block_size, self.num_heads)
            
            # Store in encoded buffer
            end_idx = dest_start_idx + num_blocks
            encoded_buf = self._encoded_buffers[layer_name][kv_type]
            
            # Indices are stored as uint8 (valid for bits <= 8)
            indices_uint8 = indices_reshaped.to(torch.uint8)
            encoded_buf["indices"][dest_start_idx:end_idx, :, :, :indices_uint8.shape[-1]] = indices_uint8
            encoded_buf["norms"][dest_start_idx:end_idx] = norms_reshaped.to(torch.float16)
            
            # Copy indices to dest_buffer for C++ engine transfer
            # The dest_buffer is what gets written to disk
            dest_buffer[dest_start_idx:end_idx].copy_(
                indices_uint8.to(dest_buffer.dtype)
            )
        else:
            # Fallback: zero fill
            logger.warning(f"Using fallback encoding for {layer_name}.{kv_type}")
            num_blocks = len(source_block_ids)
            dest_buffer[dest_start_idx:dest_start_idx + num_blocks].fill_(0)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        
        # Update metrics
        self._metrics.num_encode_calls += 1
        self._metrics.total_encode_time_ms += elapsed_ms
        num_blocks = len(source_block_ids)
        self._metrics.total_bytes_original += (
            num_blocks * self.block_size * self.num_heads * self.head_size * 2  # FP16
        )
        self._metrics.total_bytes_compressed += (
            num_blocks * self.block_size * self.num_heads * (self.n_groups * 4 + 2)  # indices + norm
        )

    def decode_batch(
        self,
        layer_name: str,
        kv_type: str,
        source_buffer: torch.Tensor,
        source_start_idx: int,
        dest_tensor: torch.Tensor,
        dest_block_ids: List[int],
    ) -> None:
        """
        Decode blocks from batch buffer into KV cache.
        
        Uses batch-level operations for GPU efficiency.
        
        Args:
            layer_name: Layer identifier
            kv_type: "k" for keys, "v" for values
            source_buffer: Source buffer (encoded indices from disk)
            source_start_idx: Starting index in source_buffer
            dest_tensor: Destination KV cache, shape (total_blocks, block_size, num_heads, head_size)
            dest_block_ids: Block IDs to write to
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        
        key_quantizer, value_quantizer = self._quantizers[layer_name]
        quantizer = key_quantizer if kv_type == "k" else value_quantizer
        
        if quantizer is not None:
            num_blocks = len(dest_block_ids)
            end_idx = source_start_idx + num_blocks
            
            # Get stored indices and norms
            encoded_buf = self._encoded_buffers[layer_name][kv_type]
            indices = encoded_buf["indices"][source_start_idx:end_idx].long()
            norms = encoded_buf["norms"][source_start_idx:end_idx].float()
            
            # Reshape for dequantization: flatten to [N, n_components]
            # N = num_blocks * block_size * num_heads
            indices_flat = indices.reshape(-1, indices.shape[-1])
            norms_flat = norms.reshape(-1)
            
            # Build the indices_dict in the format expected by upstream IsoQuant
            # IsoQuantMSE.dequantize expects {'indices': tensor, '_norms': tensor}
            indices_dict = {'indices': indices_flat, '_norms': norms_flat}
            
            # Dequantize entire batch
            if kv_type == "k":
                # For IsoQuantProd, dequantize goes through the MSE component
                decoded_vectors = quantizer.mse.dequantize(indices_dict)
            else:
                # For IsoQuantMSE, call dequantize directly
                decoded_vectors = quantizer.dequantize(indices_dict)
            
            # Reshape back to blocks: [num_blocks, block_size, num_heads, head_size]
            decoded_blocks = decoded_vectors.reshape(
                num_blocks, self.block_size, self.num_heads, self.head_size
            ).to(self.dtype)
            
            # Write to destination KV cache
            block_ids_tensor = torch.tensor(dest_block_ids, device=self.device)
            dest_tensor[block_ids_tensor] = decoded_blocks
        else:
            # Fallback: zero fill
            logger.warning(f"Using fallback decoding for {layer_name}.{kv_type}")
            for block_id in dest_block_ids:
                dest_tensor[block_id].fill_(0)
        
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        
        # Update metrics
        self._metrics.num_decode_calls += 1
        self._metrics.total_decode_time_ms += elapsed_ms

    def get_encoded_buffer(self, layer_name: str, kv_type: str) -> torch.Tensor:
        """
        Get the encoded indices buffer for a specific layer and KV type.
        
        This is the tensor that will be transferred by the C++ engine.
        
        Args:
            layer_name: Layer identifier
            kv_type: "k" for keys, "v" for values
        
        Returns:
            Encoded indices buffer tensor
        """
        return self._encoded_buffers[layer_name][kv_type]["indices"]

    def get_encoded_buffers_flat(self) -> List[torch.Tensor]:
        """
        Get all encoded buffers as a flat list for C++ engine registration.
        
        Returns:
            List of encoded buffer tensors (indices only, norms stored separately)
        """
        buffers = []
        for layer_name in self.layer_names:
            buffers.append(self._encoded_buffers[layer_name]["k"]["indices"])
            buffers.append(self._encoded_buffers[layer_name]["v"]["indices"])
        return buffers

    def get_metrics(self) -> CompressionMetrics:
        """Get current compression metrics."""
        return CompressionMetrics(
            total_encode_time_ms=self._metrics.total_encode_time_ms,
            total_decode_time_ms=self._metrics.total_decode_time_ms,
            total_bytes_original=self._metrics.total_bytes_original,
            total_bytes_compressed=self._metrics.total_bytes_compressed,
            num_encode_calls=self._metrics.num_encode_calls,
            num_decode_calls=self._metrics.num_decode_calls,
        )

    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics as a dictionary."""
        metrics = self.get_metrics()
        return {
            "compression_ratio": metrics.compression_ratio,
            "avg_encode_time_ms": metrics.avg_encode_time_ms,
            "avg_decode_time_ms": metrics.avg_decode_time_ms,
            "total_encode_calls": metrics.num_encode_calls,
            "total_decode_calls": metrics.num_decode_calls,
        }
