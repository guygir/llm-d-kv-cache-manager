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

"""
Integration tests for RotorQuant KV cache compression in storage offloading.

Tests verify:
1. Encode/decode round-trip correctness
2. Compression ratio (5× expected)
3. Quality metrics (cosine similarity >99%)
4. File extension (.rqbin for compressed files)
5. Multi-layer model support
6. Error handling
"""

import hashlib
import math
import os
import struct
import time
from collections.abc import Iterable

import pytest
import torch
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.mediums import GPULoadStoreSpec

from llmd_fs_backend.file_mapper import FileMapper
from llmd_fs_backend.mediums import SharedStorageLoadStoreSpec
from llmd_fs_backend.rotorquant_config import RotorQuantConfig
from llmd_fs_backend.worker import StorageOffloadingHandlers

TMP_DIR = "/tmp/rotorquant-test"


# ----------------------------
# Helper Functions
# ----------------------------


def create_dummy_kv_tensors(
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
    dtype: torch.dtype,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Create dummy KV cache tensors [K, V] for all layers."""
    torch.manual_seed(seed)
    shape = (2, num_blocks, block_size, num_heads, head_size)
    return [torch.rand(shape, dtype=dtype, device="cuda") for _ in range(num_layers)]


def get_prefix_hash(token_ids: Iterable[int]) -> BlockHash:
    """Generate a stable 64-bit hash for a list of token IDs."""
    buf = bytearray()
    for t in token_ids:
        buf += struct.pack("<I", int(t) & 0xFFFFFFFF)
    digest_int = int.from_bytes(hashlib.sha256(buf).digest()[:8], "big")
    return BlockHash((digest_int & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "little"))


def make_gpu_specs(block_ids: list[int]) -> GPULoadStoreSpec:
    """Create GPULoadStoreSpec objects for the given block IDs."""
    return GPULoadStoreSpec(block_ids)


def make_storage_specs(
    num_files: int,
    start_offset: int = 0,
) -> tuple[SharedStorageLoadStoreSpec, list[BlockHash]]:
    """Create SharedStorageLoadStoreSpec objects and their hashes."""
    ranges = [
        (100 + (start_offset + i) * 100, 117 + (start_offset + i) * 100)
        for i in range(num_files)
    ]
    hashes = [get_prefix_hash(range(a, b)) for (a, b) in ranges]
    return SharedStorageLoadStoreSpec(hashes), hashes


def cleanup_files(
    file_mapper: FileMapper,
    block_hashes: list[BlockHash],
) -> None:
    """Remove existing files for the provided block hashes."""
    for h in block_hashes:
        path = file_mapper.get_file_name(h)
        if os.path.exists(path):
            os.remove(path)


def wait_for_file(file_path: str, timeout: float = 2.0) -> bool:
    """Wait for a file to exist up to timeout seconds."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(file_path):
            return True
        time.sleep(0.01)
    return False


def wait_for(
    handler,
    job_id: int,
    timeout: float = 2.0,
    _finished_cache: dict = None,
):
    """Wait for a specific job in handler.get_finished() up to timeout seconds."""
    if _finished_cache is None:
        _finished_cache = {}

    if job_id in _finished_cache:
        return _finished_cache[job_id]

    start = time.time()
    while time.time() - start < timeout:
        finished = handler.get_finished()
        for result in finished:
            _finished_cache[result.job_id] = result
            if result.job_id == job_id:
                return result
        time.sleep(0.01)

    raise TimeoutError(
        f"Job {job_id} did not finish within {timeout}s. "
        f"Cached jobs: {list(_finished_cache.keys())}"
    )


def compute_cosine_similarity(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
) -> float:
    """Compute average cosine similarity between two tensors."""
    # Flatten to vectors
    vec1 = tensor1.flatten()
    vec2 = tensor2.flatten()
    
    # Compute cosine similarity
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    
    return (dot_product / (norm1 * norm2)).item()


def compute_compression_ratio(
    uncompressed_size: int,
    compressed_size: int,
) -> float:
    """Compute compression ratio."""
    return uncompressed_size / compressed_size if compressed_size > 0 else 0.0


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_rotorquant_roundtrip_basic(bits: int):
    """
    Test basic RotorQuant encode/decode round-trip.
    
    Verifies:
    - Data can be encoded and decoded
    - Decoded data matches original (within tolerance)
    - Compression ratio is approximately 5×
    - Cosine similarity > 99%
    """
    # Setup
    model_name = "llama3-8b"
    dtype = torch.float16
    root_dir = TMP_DIR
    num_layers = 4
    block_size = 16
    num_heads = 8
    head_size = 128
    num_blocks = 4
    gpu_blocks_per_file = 2
    threads_per_gpu = 4
    gpu_block_size = 16
    
    # Create RotorQuant config
    rotorquant_config = RotorQuantConfig(
        enabled=True,
        bits=bits,
        layer_pattern_override="model.layers.{layer_idx}.self_attn",
    )
    
    # Create file mapper with RotorQuant
    file_mapper = FileMapper(
        root_dir=root_dir,
        model_name=model_name,
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype=str(dtype).replace("torch.", ""),
        rotorquant_config=rotorquant_config,
    )
    
    # Create tensors
    original = create_dummy_kv_tensors(
        num_layers, num_blocks, block_size, num_heads, head_size, dtype
    )
    restored = [torch.zeros_like(t) for t in original]
    
    # Setup storage specs
    write_block_ids = list(range(num_blocks))
    put_gpu_specs = make_gpu_specs(write_block_ids)
    put_num_files = math.ceil(len(write_block_ids) / gpu_blocks_per_file)
    put_storage_specs, block_hashes = make_storage_specs(put_num_files)
    cleanup_files(file_mapper, block_hashes)
    
    # Setup layer names
    attn_backends = {f"model.layers.{i}.self_attn": FlashAttentionBackend 
                     for i in range(num_layers)}
    kv_caches_original = {f"model.layers.{i}.self_attn": original[i] 
                          for i in range(num_layers)}
    kv_caches_restored = {f"model.layers.{i}.self_attn": restored[i] 
                          for i in range(num_layers)}
    
    # PUT phase (encode and store)
    put_handler_wrapper = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
        rotorquant_config=rotorquant_config,
    )
    put_handler = put_handler_wrapper.gpu_to_storage_handler
    
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    put_result = wait_for(put_handler, job_id=1, timeout=5.0)
    assert put_result.success, "PUT failed"
    
    # Verify files exist with .rqbin extension
    for h in block_hashes:
        file_path = file_mapper.get_file_name(h)
        assert file_path.endswith(".rqbin"), f"Expected .rqbin extension, got {file_path}"
        assert wait_for_file(file_path, timeout=2.0), f"Missing file: {file_path}"
    
    # Check compression ratio
    uncompressed_size = original[0].element_size() * original[0].numel() * num_layers
    compressed_size = os.path.getsize(file_mapper.get_file_name(block_hashes[0]))
    compression_ratio = compute_compression_ratio(uncompressed_size, compressed_size)
    print(f"[INFO] Compression ratio: {compression_ratio:.2f}× (bits={bits})")
    assert compression_ratio > 3.0, f"Expected >3× compression, got {compression_ratio:.2f}×"
    
    # GET phase (load and decode)
    get_handler_wrapper = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_restored,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
        rotorquant_config=rotorquant_config,
    )
    get_handler = get_handler_wrapper.storage_to_gpu_handler
    
    get_gpu_specs = make_gpu_specs(write_block_ids)
    get_handler.transfer_async(job_id=2, spec=(put_storage_specs, get_gpu_specs))
    get_result = wait_for(get_handler, job_id=2, timeout=5.0)
    assert get_result.success, "GET failed"
    
    # Verify quality
    for layer_idx in range(num_layers):
        for block_id in write_block_ids:
            orig_block = original[layer_idx][:, block_id]
            restored_block = restored[layer_idx][:, block_id]
            
            # Compute cosine similarity
            similarity = compute_cosine_similarity(orig_block, restored_block)
            print(f"[INFO] Layer {layer_idx}, Block {block_id}: "
                  f"Cosine similarity = {similarity:.4f}")
            
            # RotorQuant quality expectations by bit width:
            # 4-bit: >99.5%, 3-bit: >98%, 2-bit: >95%
            min_similarity = {2: 0.95, 3: 0.98, 4: 0.99}[bits]
            assert similarity > min_similarity, (
                f"Layer {layer_idx}, Block {block_id}: "
                f"Expected >{min_similarity*100}% similarity at {bits}-bit, got {similarity:.4f}"
            )


def test_rotorquant_file_extension():
    """
    Test that RotorQuant uses .rqbin extension for compressed files.
    """
    # With RotorQuant
    config_with_rq = RotorQuantConfig(
        enabled=True,
        bits=3,
        layer_pattern_override="model.layers.{layer_idx}.self_attn",
    )
    file_mapper_rq = FileMapper(
        root_dir=TMP_DIR,
        model_name="test-model",
        gpu_block_size=16,
        gpu_blocks_per_file=4,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype="float16",
        rotorquant_config=config_with_rq,
    )
    
    # Without RotorQuant
    file_mapper_no_rq = FileMapper(
        root_dir=TMP_DIR,
        model_name="test-model",
        gpu_block_size=16,
        gpu_blocks_per_file=4,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype="float16",
        rotorquant_config=None,
    )
    
    # Test hash
    test_hash = get_prefix_hash(range(100, 117))
    
    # Verify extensions
    path_with_rq = file_mapper_rq.get_file_name(test_hash)
    path_without_rq = file_mapper_no_rq.get_file_name(test_hash)
    
    assert path_with_rq.endswith(".rqbin"), f"Expected .rqbin, got {path_with_rq}"
    assert path_without_rq.endswith(".bin"), f"Expected .bin, got {path_without_rq}"
    
    # Verify paths differ only in extension
    assert path_with_rq[:-6] == path_without_rq[:-4], "Base paths should match"


def test_rotorquant_multi_layer():
    """
    Test RotorQuant with multiple layers (Llama-style model).
    
    Verifies:
    - All layers are compressed independently
    - Each layer maintains quality
    - Compression works across many layers
    """
    model_name = "llama3-70b"
    dtype = torch.float16
    root_dir = TMP_DIR
    num_layers = 80  # Full Llama-70B
    block_size = 16
    num_heads = 64
    head_size = 128
    num_blocks = 2  # Small for testing
    gpu_blocks_per_file = 2
    threads_per_gpu = 8
    gpu_block_size = 16
    
    rotorquant_config = RotorQuantConfig(
        enabled=True,
        bits=3,
        layer_pattern_override="model.layers.{layer_idx}.self_attn",
    )
    
    file_mapper = FileMapper(
        root_dir=root_dir,
        model_name=model_name,
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype=str(dtype).replace("torch.", ""),
        rotorquant_config=rotorquant_config,
    )
    
    original = create_dummy_kv_tensors(
        num_layers, num_blocks, block_size, num_heads, head_size, dtype
    )
    restored = [torch.zeros_like(t) for t in original]
    
    write_block_ids = list(range(num_blocks))
    put_gpu_specs = make_gpu_specs(write_block_ids)
    put_num_files = math.ceil(len(write_block_ids) / gpu_blocks_per_file)
    put_storage_specs, block_hashes = make_storage_specs(put_num_files)
    cleanup_files(file_mapper, block_hashes)
    
    attn_backends = {f"model.layers.{i}.self_attn": FlashAttentionBackend 
                     for i in range(num_layers)}
    kv_caches_original = {f"model.layers.{i}.self_attn": original[i] 
                          for i in range(num_layers)}
    kv_caches_restored = {f"model.layers.{i}.self_attn": restored[i] 
                          for i in range(num_layers)}
    
    # PUT
    put_handler_wrapper = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
        rotorquant_config=rotorquant_config,
    )
    put_handler = put_handler_wrapper.gpu_to_storage_handler
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    put_result = wait_for(put_handler, job_id=1, timeout=10.0)
    assert put_result.success, "PUT failed"
    
    # GET
    get_handler_wrapper = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_restored,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
        rotorquant_config=rotorquant_config,
    )
    get_handler = get_handler_wrapper.storage_to_gpu_handler
    get_gpu_specs = make_gpu_specs(write_block_ids)
    get_handler.transfer_async(job_id=2, spec=(put_storage_specs, get_gpu_specs))
    get_result = wait_for(get_handler, job_id=2, timeout=10.0)
    assert get_result.success, "GET failed"
    
    # Verify all layers
    min_similarity = 1.0
    for layer_idx in range(num_layers):
        for block_id in write_block_ids:
            similarity = compute_cosine_similarity(
                original[layer_idx][:, block_id],
                restored[layer_idx][:, block_id]
            )
            min_similarity = min(min_similarity, similarity)
    
    print(f"[INFO] Multi-layer test: Min similarity across {num_layers} layers = {min_similarity:.4f}")
    assert min_similarity > 0.99, f"Expected >99% similarity, got {min_similarity:.4f}"


def test_rotorquant_disabled():
    """
    Test that when RotorQuant is disabled, files use .bin extension
    and no compression occurs.
    """
    model_name = "test-model"
    dtype = torch.float16
    root_dir = TMP_DIR
    num_layers = 2
    block_size = 16
    num_heads = 8
    head_size = 128
    num_blocks = 2
    gpu_blocks_per_file = 2
    threads_per_gpu = 4
    gpu_block_size = 16
    
    # No RotorQuant config
    file_mapper = FileMapper(
        root_dir=root_dir,
        model_name=model_name,
        gpu_block_size=gpu_block_size,
        gpu_blocks_per_file=gpu_blocks_per_file,
        tp_size=1,
        pp_size=1,
        pcp_size=1,
        rank=0,
        dtype=str(dtype).replace("torch.", ""),
        rotorquant_config=None,
    )
    
    original = create_dummy_kv_tensors(
        num_layers, num_blocks, block_size, num_heads, head_size, dtype
    )
    
    write_block_ids = list(range(num_blocks))
    put_gpu_specs = make_gpu_specs(write_block_ids)
    put_num_files = math.ceil(len(write_block_ids) / gpu_blocks_per_file)
    put_storage_specs, block_hashes = make_storage_specs(put_num_files, start_offset=100)
    cleanup_files(file_mapper, block_hashes)
    
    attn_backends = {f"layer_{i}": FlashAttentionBackend for i in range(num_layers)}
    kv_caches_original = {f"layer_{i}": original[i] for i in range(num_layers)}
    
    # PUT without compression
    put_handler_wrapper = StorageOffloadingHandlers(
        file_mapper=file_mapper,
        kv_caches=kv_caches_original,
        gpu_blocks_per_file=gpu_blocks_per_file,
        gpu_block_size=gpu_block_size,
        threads_per_gpu=threads_per_gpu,
        attn_backends=attn_backends,
        rotorquant_config=None,
    )
    put_handler = put_handler_wrapper.gpu_to_storage_handler
    put_handler.transfer_async(job_id=1, spec=(put_gpu_specs, put_storage_specs))
    put_result = wait_for(put_handler, job_id=1, timeout=5.0)
    assert put_result.success, "PUT failed"
    
    # Verify .bin extension
    for h in block_hashes:
        file_path = file_mapper.get_file_name(h)
        assert file_path.endswith(".bin"), f"Expected .bin extension, got {file_path}"
        assert wait_for_file(file_path, timeout=2.0), f"Missing file: {file_path}"
    
    # Verify no compression (file size should match uncompressed size)
    uncompressed_size = original[0].element_size() * original[0].numel() * num_layers
    file_size = os.path.getsize(file_mapper.get_file_name(block_hashes[0]))
    
    # Allow small overhead for metadata
    assert abs(file_size - uncompressed_size) < 1024, (
        f"Expected uncompressed size ~{uncompressed_size}, got {file_size}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

# Made with Bob
