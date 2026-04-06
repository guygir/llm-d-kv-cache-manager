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

import math
import os
import time

import storage_offload
import torch
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    TransferResult,
    TransferSpec,
    TransferType,
)

from llmd_fs_backend import _logger as logger
from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec

# ----------------------------------------------------------------------
# Base Storage Offloading Handler
# ----------------------------------------------------------------------
DEFAULT_MAX_STAGING_MEMORY_GB = 150
DEFAULT_THREADS_PER_GPU = 64
DEFAULT_READ_PREFERRING_WORKERS_RATIO = 0.75


class BaseStorageOffloadingHandler(OffloadingHandler):
    """
    BaseStorageOffloadingHandler handles transfers for both directions,
    either GPU->Storage (PUT) or Storage->GPU (GET).
    """

    def __init__(
        self,
        gpu_blocks_per_file: int,
        file_mapper: FileMapper,
        engine: storage_offload.StorageOffloadEngine,
        transfer_type: TransferType,
        per_block_bytes: int,
        codec=None,
        tensor_to_layer_map=None,
        original_tensors=None,
        encoded_tensors=None,
    ):
        """
        Initialize a SingleStorageDirectionOffloadingHandler.

        Args:
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            file_mapper: The FileMapper mapping blocks to files.
            engine: the storage engine.
            transfer_type: The type of transfer (src, dst) for metrics.
            per_block_bytes: Size of a single GPU block in bytes.
            codec: Optional RotorQuantCodec for compression.
            tensor_to_layer_map: Optional mapping from tensor index to layer name.
            original_tensors: Optional list of original FP16 tensors.
            encoded_tensors: Optional list of encoded uint8 tensors.
        """
        self.file_mapper = file_mapper
        self.gpu_blocks_per_file = gpu_blocks_per_file
        self.engine = engine
        self.transfer_type = transfer_type
        self.per_block_bytes = per_block_bytes
        self.codec = codec
        self.tensor_to_layer_map = tensor_to_layer_map
        self.original_tensors = original_tensors
        self.encoded_tensors = encoded_tensors

        # Maps job_id -> (submit_time, transfer_size_bytes).
        # Shared across handlers via StorageOffloadingHandlers.
        self._pending_jobs: dict[int, tuple[float, int]] = {}
        
        # Maps job_id -> per_file_block_ids for decoding after LOAD completes
        self._pending_decode_jobs: dict[int, list[list[int]]] = {}

    def _record_job(self, job_id: int, num_blocks: int):
        """Record job submission metadata for metrics."""
        transfer_size = num_blocks * self.per_block_bytes
        self._pending_jobs[job_id] = (
            time.monotonic(),
            transfer_size,
        )

    def get_finished(self) -> list[TransferResult]:
        """
        Poll finished async transfers.

        Returns:
            List of completed transfer results.
        """
        now = time.monotonic()
        results = []
        for job_id, success in self.engine.get_finished():
            job_info = self._pending_jobs.pop(job_id, None)
            if job_info is not None:
                submit_time, transfer_size = job_info
                transfer_time = now - submit_time
                results.append(
                    TransferResult(
                        job_id=job_id,
                        success=success,
                        transfer_size=transfer_size,
                        transfer_time=transfer_time,
                        transfer_type=self.transfer_type,
                    )
                )
                logger.debug(
                    "Transfer finished: job_id=%d status=%s "
                    "size=%.2f [MB] time=%.3f [s] throughput=%.2f [GB/s] type=%s",
                    job_id,
                    "OK" if success else "FAIL",
                    transfer_size / (1 << 20),
                    transfer_time,
                    (transfer_size / transfer_time if transfer_time > 0 else 0)
                    / (1 << 30),
                    f"{self.transfer_type[0]}->{self.transfer_type[1]}",
                )
            else:
                logger.warning(
                    "Transfer finished with unknown job_id=%d, metrics unavailable",
                    job_id,
                )
                results.append(TransferResult(job_id=job_id, success=success))
        return results

    def wait(self, job_ids: set[int]):
        """
        Block until the specified transfer jobs complete.

        Args:
            job_ids: Set of job IDs to wait for.
        """
        for job_id in job_ids:
            self.engine.wait_job(job_id)

    def _build_file_block_mapping(
        self,
        block_hashes,
        block_ids,
    ):
        """
        Build per-file block ID lists for grouped transfers.

        Returns:
            tuple[list[str], list[list[int]]]
                - file paths
                - per-file block ID lists
        """
        files = []
        per_file_block_ids = []

        # The first file in get may contain fewer blocks than gpu_blocks_per_file
        first_size = (
            len(block_ids) % self.gpu_blocks_per_file or self.gpu_blocks_per_file
        )

        start = 0
        size = first_size

        for block_hash in block_hashes:
            end = min(start + size, len(block_ids))
            block_ids_chunk = block_ids[start:end]

            # Build file path for this group of blocks
            files.append(self.file_mapper.get_file_name(block_hash))
            per_file_block_ids.append(block_ids_chunk)

            start += size
            size = self.gpu_blocks_per_file

        return files, per_file_block_ids


class GPUToStorageHandler(BaseStorageOffloadingHandler):
    """Handler for GPU -> Storage (PUT) transfers."""

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Launch an asynchronous transfer GPU -> Storage.

        Args:
            job_id: Unique identifier for the transfer job.
            spec: Transfer specification describing source and destination
                block IDs and file hashes.

        Returns:
            True if the transfer was successfully submitted.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, GPULoadStoreSpec)
        assert isinstance(dst_spec, SharedStorageLoadStoreSpec)

        dst_files, per_file_block_ids = self._build_file_block_mapping(
            block_hashes=dst_spec.block_hashes,
            block_ids=src_spec.block_ids,
        )

        # IsoQuant/RotorQuant encoding
        if self.codec and self.tensor_to_layer_map:
            logger.info(f"Encoding {sum(len(ids) for ids in per_file_block_ids)} blocks")
            
            # Encode blocks into batch buffers
            for file_idx, block_ids in enumerate(per_file_block_ids):
                for tensor_idx in range(len(self.original_tensors)):
                    layer_name = self.tensor_to_layer_map.get(tensor_idx)
                    if layer_name:
                        # Determine if this is K or V tensor (even=K, odd=V)
                        kv_type = "k" if tensor_idx % 2 == 0 else "v"
                        
                        # Encode batch using codec
                        self.codec.encode_batch(
                            layer_name=layer_name,
                            kv_type=kv_type,
                            source_tensor=self.original_tensors[tensor_idx],
                            source_block_ids=block_ids,
                            dest_buffer=self.encoded_tensors[tensor_idx],
                            dest_start_idx=0,
                        )
            
            # Swap engine to use encoded tensors
            self.engine.set_tensors(self.encoded_tensors)
            logger.debug("Engine switched to encoded tensors")

        # Submit async PUT transfer (now uses encoded data if RotorQuant enabled)
        success = self.engine.async_store_gpu_blocks(
            job_id, dst_files, per_file_block_ids
        )
        
        # Restore original tensors after async submission
        if self.codec and self.tensor_to_layer_map:
            self.engine.set_tensors(self.original_tensors)
            logger.debug("Engine restored to original tensors")
        if success:
            total_blocks = sum(len(ids) for ids in per_file_block_ids)
            self._record_job(job_id, total_blocks)
        return success


class StorageToGPUHandler(BaseStorageOffloadingHandler):
    """Handler for asynchronous transfers from storage to GPU."""

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        """
        Launch an asynchronous transfer Storage -> GPU.

        Args:
            job_id: Unique identifier for the transfer job.
            spec: Transfer specification describing source and destination
                block IDs and file hashes.

        Returns:
            True if the transfer was successfully submitted.
        """
        src_spec, dst_spec = spec
        assert isinstance(src_spec, SharedStorageLoadStoreSpec)
        assert isinstance(dst_spec, GPULoadStoreSpec)

        src_files, per_file_block_ids = self._build_file_block_mapping(
            block_hashes=src_spec.block_hashes,
            block_ids=dst_spec.block_ids,
        )

        # RotorQuant: Set engine to load into encoded tensors (Finding 10)
        if self.codec and self.tensor_to_layer_map:
            self.engine.set_tensors(self.encoded_tensors)
            logger.debug(f"Engine switched to encoded tensors for LOAD job {job_id}")
            # Store block IDs for decoding after transfer completes
            self._pending_decode_jobs[job_id] = per_file_block_ids

        # Submit async GET transfer (loads into encoded tensors if RotorQuant enabled)
        success = self.engine.async_load_gpu_blocks(
            job_id, src_files, per_file_block_ids
        )
        
        # Restore original tensors after async submission
        if self.codec and self.tensor_to_layer_map:
            self.engine.set_tensors(self.original_tensors)
            logger.debug("Engine restored to original tensors")
        
        if success:
            total_blocks = sum(len(ids) for ids in per_file_block_ids)
            self._record_job(job_id, total_blocks)
        return success

    def get_finished(self) -> list[TransferResult]:
        """
        Poll finished async transfers and decode if RotorQuant enabled.
        
        Returns:
            List of completed transfer results.
        """
        # Get finished transfers from base implementation
        results = super().get_finished()
        
        # Decode any completed LOAD jobs
        if self.codec and self.tensor_to_layer_map:
            for result in results:
                if result.success and result.job_id in self._pending_decode_jobs:
                    per_file_block_ids = self._pending_decode_jobs.pop(result.job_id)
                    
                    logger.info(f"RotorQuant decoding {sum(len(ids) for ids in per_file_block_ids)} blocks")
                    
                    # Decode from encoded tensors to original tensors
                    for file_idx, block_ids in enumerate(per_file_block_ids):
                        for tensor_idx in range(len(self.encoded_tensors)):
                            layer_name = self.tensor_to_layer_map.get(tensor_idx)
                            if layer_name:
                                # Determine if this is K or V tensor (even=K, odd=V)
                                kv_type = "k" if tensor_idx % 2 == 0 else "v"
                                
                                # Decode batch using codec
                                self.codec.decode_batch(
                                    layer_name=layer_name,
                                    kv_type=kv_type,
                                    source_buffer=self.encoded_tensors[tensor_idx],
                                    source_start_idx=0,
                                    dest_tensor=self.original_tensors[tensor_idx],
                                    dest_block_ids=block_ids,
                                )
                    
                    logger.debug(f"Decoded job {result.job_id} successfully")
        
        return results


class StorageOffloadingHandlers:
    """Base handler with common helpers for Storage offloading."""

    def __init__(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
        file_mapper: FileMapper,
        gpu_block_size: int,
        gpu_blocks_per_file: int,
        threads_per_gpu: int,
        max_staging_memory_gb: int = DEFAULT_MAX_STAGING_MEMORY_GB,
        read_preferring_ratio: float = DEFAULT_READ_PREFERRING_WORKERS_RATIO,
        rotorquant_config=None,
        isoquant_config=None,
        vllm_config=None,
    ):
        """Initialize StorageOffloadingHandlers.
        
        Args:
            kv_caches: Dictionary of KV cache tensors
            attn_backends: Dictionary of attention backend types
            file_mapper: FileMapper for block-to-file mapping
            gpu_block_size: Size of GPU blocks
            gpu_blocks_per_file: Number of GPU blocks per file
            threads_per_gpu: Number of I/O threads per GPU
            max_staging_memory_gb: Maximum staging memory in GB
            read_preferring_ratio: Ratio of read-preferring workers
            rotorquant_config: Optional RotorQuant configuration
            isoquant_config: Optional IsoQuant configuration (takes precedence)
            vllm_config: Optional vLLM configuration (for codec initialization)
        """
        # Store configs and tensor mapping for codec
        self.rotorquant_config = rotorquant_config
        self.isoquant_config = isoquant_config
        self.vllm_config = vllm_config
        self.codec = None  # Will be initialized after tensor extraction
        self.original_tensors = None  # Store reference to original FP16 tensors
        self.encoded_tensors = None  # Store reference to encoded uint8 tensors
        
        threads_per_gpu = min(threads_per_gpu, int(os.cpu_count()))
        tensors, kernel_block_size, self.tensor_to_layer_map = (
            StorageOffloadingHandlers._get_tensors(kv_caches, attn_backends)
        )
        
        # Store original tensor references for RotorQuant tensor swapping
        self.original_tensors = tensors
        assert tensors
        assert gpu_block_size % kernel_block_size == 0

        kernel_blocks_per_gpu_block = gpu_block_size // kernel_block_size

        # Compute staging memory buffer size
        buffer_size_mb = self._compute_buffer_size_mb(
            tensors, gpu_blocks_per_file, kernel_blocks_per_gpu_block
        )

        # Adjust threads_per_gpu if exceeding max_staging_memory_gb
        if buffer_size_mb * threads_per_gpu > max_staging_memory_gb * 1024:
            threads_per_gpu = min(
                threads_per_gpu, int(max_staging_memory_gb * 1024 / buffer_size_mb)
            )
            logger.warning(
                f"Adjusted threads_per_gpu to {threads_per_gpu} due to "
                f"max_staging_memory_gb {max_staging_memory_gb} "
                f"limit (buffer_size_mb={buffer_size_mb})."
            )

        # Calculate number of read-preferring workers
        read_preferring_workers = max(1, int(threads_per_gpu * read_preferring_ratio))

        # Initialize codec if quantization is enabled (IsoQuant takes precedence)
        use_isoquant = self.isoquant_config and self.isoquant_config.enabled
        use_rotorquant = self.rotorquant_config and self.rotorquant_config.enabled
        
        if (use_isoquant or use_rotorquant) and self.vllm_config:
            # Extract unique layer names from tensor mapping
            unique_layer_names = sorted(set(self.tensor_to_layer_map.values()))
            
            # Extract KV cache parameters from first tensor
            first_tensor = tensors[0]
            total_blocks = first_tensor.shape[0]  # Total blocks in KV cache
            
            # Get model config parameters from vllm_config
            model_config = self.vllm_config.model_config
            num_heads = model_config.get_num_kv_heads(self.vllm_config.parallel_config)
            head_size = model_config.get_head_size()
            
            # Get device from first tensor
            device = first_tensor.device
            dtype = first_tensor.dtype
            
            # Use batch-level buffer allocation (Finding 9: Option A-Modified)
            # Allocate encoded buffers for ONE BATCH (gpu_blocks_per_file) instead of
            # all blocks to save GPU memory. This reduces overhead from 1.6× to 1.015×
            # with negligible performance cost (<1%).
            
            if use_isoquant:
                from llmd_fs_backend.isoquant_codec import IsoQuantCodec
                
                logger.info(
                    f"Initializing IsoQuantCodec: {len(unique_layer_names)} layers, "
                    f"{total_blocks} total blocks, {gpu_blocks_per_file} blocks/batch, "
                    f"{gpu_block_size} tokens/block, {num_heads} heads, {head_size} head_dim, "
                    f"bits={self.isoquant_config.bits}, mode={self.isoquant_config.mode}"
                )
                
                self.codec = IsoQuantCodec(
                    config=self.isoquant_config,
                    layer_names=unique_layer_names,
                    num_blocks=gpu_blocks_per_file,  # Batch size, not total blocks
                    block_size=gpu_block_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    device=device,
                    dtype=dtype,
                )
                logger.info(
                    f"IsoQuantCodec initialized with batch-level buffers "
                    f"({gpu_blocks_per_file} blocks/batch)"
                )
            else:
                from llmd_fs_backend.rotorquant_codec import RotorQuantCodec
                
                logger.info(
                    f"Initializing RotorQuantCodec: {len(unique_layer_names)} layers, "
                    f"{total_blocks} total blocks, {gpu_blocks_per_file} blocks/batch, "
                    f"{gpu_block_size} tokens/block, {num_heads} heads, {head_size} head_dim"
                )
                
                self.codec = RotorQuantCodec(
                    config=self.rotorquant_config,
                    layer_names=unique_layer_names,
                    num_blocks=gpu_blocks_per_file,  # Batch size, not total blocks
                    block_size=gpu_block_size,
                    num_heads=num_heads,
                    head_size=head_size,
                    device=device,
                    dtype=dtype,
                )
                logger.info(
                    f"RotorQuantCodec initialized with batch-level buffers "
                    f"({gpu_blocks_per_file} blocks/batch)"
                )
            
            # Create encoded tensor list for C++ engine
            # These are batch-level buffers that will be populated during encoding
            self.encoded_tensors = []
            for layer_name in unique_layer_names:
                # Get encoded buffers from codec (one for K, one for V per layer)
                k_buffer = self.codec.get_encoded_buffer(layer_name, "k")
                v_buffer = self.codec.get_encoded_buffer(layer_name, "v")
                self.encoded_tensors.append(k_buffer)
                self.encoded_tensors.append(v_buffer)
            
            quant_type = "IsoQuant" if use_isoquant else "RotorQuant"
            logger.info(
                f"Created {len(self.encoded_tensors)} encoded tensor buffers "
                f"for {quant_type} compression"
            )
        
        # Initialize storage offload resources for async transfers
        self.engine = storage_offload.StorageOffloadEngine(
            io_threads=threads_per_gpu,
            gpu_blocks_per_file=gpu_blocks_per_file,
            tensors=tensors,
            read_preferring_workers=read_preferring_workers,
        )

        # Compute per-GPU-block size in bytes for metrics across all layers.
        kernel_block_bytes = sum(t.stride(0) * t.element_size() for t in tensors)
        per_block_bytes = kernel_block_bytes * kernel_blocks_per_gpu_block
        logger.info(
            f"StorageOffloadingHandlers: "
            f"threads_per_gpu={threads_per_gpu},"
            f"offloading block_size={gpu_blocks_per_file * gpu_block_size}, "
            f"staging_buffer_size_mb={buffer_size_mb}, "
            f"max_staging_memory_gb={max_staging_memory_gb}, "
            f"read_preferring_workers={read_preferring_workers}, "
        )

        # Shared across both handlers since the engine has a single completion queue.
        pending_jobs: dict[int, tuple[float, int, TransferType]] = {}

        self.gpu_to_storage_handler = GPUToStorageHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
            transfer_type=("GPU", "SHARED_STORAGE"),
            per_block_bytes=per_block_bytes,
            codec=self.codec,
            tensor_to_layer_map=self.tensor_to_layer_map,
            original_tensors=self.original_tensors,
            encoded_tensors=self.encoded_tensors,
        )
        self.gpu_to_storage_handler._pending_jobs = pending_jobs

        self.storage_to_gpu_handler = StorageToGPUHandler(
            engine=self.engine,
            file_mapper=file_mapper,
            gpu_blocks_per_file=gpu_blocks_per_file,
            transfer_type=("SHARED_STORAGE", "GPU"),
            per_block_bytes=per_block_bytes,
            codec=self.codec,
            tensor_to_layer_map=self.tensor_to_layer_map,
            original_tensors=self.original_tensors,
            encoded_tensors=self.encoded_tensors,
        )
        self.storage_to_gpu_handler._pending_jobs = pending_jobs

    def _compute_buffer_size_mb(
        self,
        tensors: list[torch.Tensor],
        gpu_blocks_per_file: int,
        kernel_blocks_per_gpu_block: int,
    ):
        """
        Estimate staging memory size in MB, applying min/max limits.

        Args:
            tensors: List of KV-cache tensors used to infer per-block memory usage.
            gpu_blocks_per_file: Number of GPU blocks grouped into a single file.
            kernel_blocks_per_gpu_block: Number of kernel blocks grouped into
                                         a single GPU block.

        Returns:
            Estimated staging buffer size in megabytes.
        """
        kernel_block_size_in_bytes = 0
        for tensor in tensors:
            kernel_block_size_in_bytes += tensor.stride(0) * tensor.element_size()
        kernel_blocks_per_file = kernel_blocks_per_gpu_block * gpu_blocks_per_file
        file_size_in_bytes = kernel_block_size_in_bytes * kernel_blocks_per_file
        file_size_mb = math.ceil(file_size_in_bytes / (1 << 20))
        return file_size_mb

    @staticmethod
    def _get_tensors(
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> tuple[list[torch.Tensor], int, dict[int, str]]:
        """
        Splits the given KV caches to tensors such that
            each tensor shape is (num_blocks, ...).

        Returns:
            (list_of_kv_cache_tensors, kernel_block_size, tensor_to_layer_map)
            
        The tensor_to_layer_map maps tensor indices to layer names, enabling
        the codec to select the correct quantizer for each tensor.
        """
        tensors: list[torch.Tensor] = []
        tensor_to_layer_map: dict[int, str] = {}
        kernel_block_size: int | None = None
        tensor_idx = 0

        for layer_name, gpu_tensor in kv_caches.items():
            gpu_shape = gpu_tensor.shape
            attn_backend = attn_backends[layer_name]

            # Generate a reference KV-cache shape using known parameters.
            # We compare gpu_shape with this synthetic shape to infer the layout.
            test_shape = attn_backend.get_kv_cache_shape(
                num_blocks=1234, block_size=16, num_kv_heads=8, head_size=256
            )

            split_k_and_v = False
            has_layers_dim = False
            if len(gpu_shape) != len(test_shape):
                # Case 1: Cross-layer tensor - an extra layer dimension exists.
                # In this case, num_blocks is the leading dimension.
                assert len(gpu_shape) == len(test_shape) + 1
                has_layers_dim = True
                # prepend a dummy num_layers=80 to test_shape
                test_shape = (80,) + test_shape
            elif test_shape[0] == 1234:
                # Case 2: Standard layout - each element represents a single layer with
                # tensor shaped as (num_blocks, ...).
                # The first dimension matches num_blocks.
                pass
            else:
                # Case 3: (2, num_blocks, ...) - standard layout but with KV first:
                # (2, num_blocks, heads, block_size, head_size).
                assert test_shape[0] == 2
                assert test_shape[1] == 1234
                assert gpu_shape[0] == 2
                split_k_and_v = True

            if split_k_and_v:
                # split tensor to k-tensor and v-tensor
                # Map both K and V tensors to the same layer name
                tensor_to_layer_map[tensor_idx] = layer_name
                tensor_to_layer_map[tensor_idx + 1] = layer_name
                for sub_tensor in gpu_tensor:
                    tensors.append(sub_tensor)
                tensor_idx += 2
            else:
                # Single tensor (combined KV or cross-layer)
                tensor_to_layer_map[tensor_idx] = layer_name
                tensors.append(gpu_tensor)
                tensor_idx += 1

            try:
                kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
                    include_num_layers_dimension=has_layers_dim
                )
                assert len(kv_cache_stride_order) == len(gpu_shape)
            except (AttributeError, NotImplementedError):
                kv_cache_stride_order = tuple(range(len(gpu_shape)))

            # permute test_shape according to stride_order
            test_shape = tuple(test_shape[i] for i in kv_cache_stride_order)

            # find block_size (16) dimension index
            block_size_idx = test_shape.index(16)
            if kernel_block_size is not None:
                assert kernel_block_size == gpu_shape[block_size_idx]
            else:
                kernel_block_size = gpu_shape[block_size_idx]

        assert len({t.stride(0) for t in tensors}) == 1, (
            "All KV-cache tensors must have the same block element stride."
        )
        assert kernel_block_size
        return tensors, kernel_block_size, tensor_to_layer_map
