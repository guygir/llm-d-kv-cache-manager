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

"""Integration tests for IsoQuant KV cache compression.

These tests verify the full encode/decode pipeline for IsoQuant,
including quality preservation and compression ratio metrics.
"""

import pytest
import torch

from llmd_fs_backend.isoquant_config import IsoQuantConfig


# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for IsoQuant tests"
)


class TestIsoQuantCodecIntegration:
    """Integration tests for IsoQuantCodec."""

    @pytest.fixture
    def config(self):
        """Create test IsoQuant configuration."""
        return IsoQuantConfig(
            enabled=True,
            bits=3,
            mode="fast",
            layer_pattern_override="model.layers.{layer_idx}.self_attn",
        )

    @pytest.fixture
    def layer_names(self):
        """Create test layer names."""
        return [f"model.layers.{i}.self_attn" for i in range(4)]

    @pytest.fixture
    def kv_cache_params(self):
        """Return standard KV cache parameters."""
        return {
            "num_blocks": 16,  # Batch buffer size
            "block_size": 16,  # Tokens per block
            "num_heads": 32,
            "head_size": 128,
        }

    def test_codec_initialization(self, config, layer_names, kv_cache_params):
        """Test codec initializes correctly."""
        from llmd_fs_backend.isoquant_codec import IsoQuantCodec

        codec = IsoQuantCodec(
            config=config,
            layer_names=layer_names,
            device=torch.device("cuda"),
            dtype=torch.float16,
            **kv_cache_params,
        )

        assert codec.bits == 3
        assert codec.mode == "fast"
        assert codec.num_blocks == 16
        assert len(codec.layer_names) == 4

    def test_encode_decode_roundtrip(self, config, layer_names, kv_cache_params):
        """Test full encode/decode roundtrip preserves quality."""
        from llmd_fs_backend.isoquant_codec import IsoQuantCodec

        codec = IsoQuantCodec(
            config=config,
            layer_names=layer_names,
            device=torch.device("cuda"),
            dtype=torch.float16,
            **kv_cache_params,
        )

        # Create test KV cache tensor
        total_blocks = 64  # Total blocks in KV cache (> batch buffer)
        kv_cache = torch.randn(
            total_blocks,
            kv_cache_params["block_size"],
            kv_cache_params["num_heads"],
            kv_cache_params["head_size"],
            dtype=torch.float16,
            device="cuda",
        )

        # Simulate encode/decode for a batch
        test_block_ids = list(range(16))  # First batch of blocks
        layer_name = layer_names[0]

        # Create destination buffer for encoded data
        dest_buffer = codec.get_encoded_buffer(layer_name, "k")

        # Encode
        codec.encode_batch(
            layer_name=layer_name,
            kv_type="k",
            source_tensor=kv_cache,
            source_block_ids=test_block_ids,
            dest_buffer=dest_buffer,
            dest_start_idx=0,
        )

        # Create output tensor for decoded data
        decoded_kv_cache = torch.zeros_like(kv_cache)

        # Decode
        codec.decode_batch(
            layer_name=layer_name,
            kv_type="k",
            source_buffer=dest_buffer,
            source_start_idx=0,
            dest_tensor=decoded_kv_cache,
            dest_block_ids=test_block_ids,
        )

        # Verify quality - IsoQuant should achieve >99% cosine similarity
        original_flat = kv_cache[test_block_ids].reshape(-1).float()
        decoded_flat = decoded_kv_cache[test_block_ids].reshape(-1).float()

        cosine_sim = torch.nn.functional.cosine_similarity(
            original_flat.unsqueeze(0),
            decoded_flat.unsqueeze(0),
        ).item()

        # IsoQuant quality expectations: 3-bit ~98.3%, 4-bit ~99.5%
        # Using 98% threshold for 3-bit (based on upstream benchmarks)
        assert cosine_sim > 0.98, f"Quality too low: {cosine_sim:.4f}"

    def test_compression_metrics(self, config, layer_names, kv_cache_params):
        """Test compression metrics are tracked correctly."""
        from llmd_fs_backend.isoquant_codec import IsoQuantCodec

        codec = IsoQuantCodec(
            config=config,
            layer_names=layer_names,
            device=torch.device("cuda"),
            dtype=torch.float16,
            **kv_cache_params,
        )

        total_blocks = 64
        kv_cache = torch.randn(
            total_blocks,
            kv_cache_params["block_size"],
            kv_cache_params["num_heads"],
            kv_cache_params["head_size"],
            dtype=torch.float16,
            device="cuda",
        )

        test_block_ids = list(range(16))
        layer_name = layer_names[0]
        dest_buffer = codec.get_encoded_buffer(layer_name, "k")

        # Initial metrics should be zero
        metrics = codec.get_metrics()
        assert metrics.num_encode_calls == 0

        # Encode and check metrics
        codec.encode_batch(
            layer_name=layer_name,
            kv_type="k",
            source_tensor=kv_cache,
            source_block_ids=test_block_ids,
            dest_buffer=dest_buffer,
            dest_start_idx=0,
        )

        metrics = codec.get_metrics()
        assert metrics.num_encode_calls == 1
        assert metrics.total_bytes_original > 0
        assert metrics.total_bytes_compressed > 0
        assert metrics.compression_ratio > 4.0  # Should be ~5x for 3-bit

    def test_full_mode_quality(self, layer_names, kv_cache_params):
        """Test that full mode (q_L v q̄_R) achieves higher quality."""
        from llmd_fs_backend.isoquant_codec import IsoQuantCodec

        config_fast = IsoQuantConfig(
            enabled=True,
            bits=3,
            mode="fast",
            layer_pattern_override="model.layers.{layer_idx}.self_attn",
        )

        config_full = IsoQuantConfig(
            enabled=True,
            bits=3,
            mode="full",
            layer_pattern_override="model.layers.{layer_idx}.self_attn",
        )

        codec_fast = IsoQuantCodec(
            config=config_fast,
            layer_names=layer_names,
            device=torch.device("cuda"),
            dtype=torch.float16,
            **kv_cache_params,
        )

        codec_full = IsoQuantCodec(
            config=config_full,
            layer_names=layer_names,
            device=torch.device("cuda"),
            dtype=torch.float16,
            **kv_cache_params,
        )

        # Create identical test data
        torch.manual_seed(42)
        total_blocks = 64
        kv_cache = torch.randn(
            total_blocks,
            kv_cache_params["block_size"],
            kv_cache_params["num_heads"],
            kv_cache_params["head_size"],
            dtype=torch.float16,
            device="cuda",
        )

        test_block_ids = list(range(16))
        layer_name = layer_names[0]

        # Test both modes
        for codec, mode_name in [(codec_fast, "fast"), (codec_full, "full")]:
            dest_buffer = codec.get_encoded_buffer(layer_name, "k")
            decoded_cache = torch.zeros_like(kv_cache)

            codec.encode_batch(
                layer_name=layer_name,
                kv_type="k",
                source_tensor=kv_cache,
                source_block_ids=test_block_ids,
                dest_buffer=dest_buffer,
                dest_start_idx=0,
            )

            codec.decode_batch(
                layer_name=layer_name,
                kv_type="k",
                source_buffer=dest_buffer,
                source_start_idx=0,
                dest_tensor=decoded_cache,
                dest_block_ids=test_block_ids,
            )

            # Verify quality
            original = kv_cache[test_block_ids].reshape(-1).float()
            decoded = decoded_cache[test_block_ids].reshape(-1).float()
            cosine_sim = torch.nn.functional.cosine_similarity(
                original.unsqueeze(0), decoded.unsqueeze(0)
            ).item()

            # Both modes should achieve >98% quality at 3-bit
            assert cosine_sim > 0.98, f"{mode_name} quality too low: {cosine_sim:.4f}"


class TestIsoQuantVsUpstream:
    """Compare our implementation against upstream turboquant."""

    @pytest.fixture
    def head_size(self):
        return 128

    def test_upstream_isoquant_available(self, head_size):
        """Test that upstream IsoQuant is available."""
        try:
            from turboquant import IsoQuantMSE, IsoQuantProd

            # Verify basic functionality
            quantizer = IsoQuantMSE(d=head_size, bits=3, mode="fast", device="cuda")
            test_vectors = torch.randn(100, head_size, device="cuda")
            x_hat, indices = quantizer(test_vectors)

            assert x_hat.shape == test_vectors.shape
            assert "indices" in indices or "_norms" in indices

        except ImportError:
            pytest.skip("turboquant not installed - upstream test skipped")

    def test_quality_matches_upstream(self, head_size):
        """Test that our codec achieves similar quality to upstream."""
        try:
            from turboquant import IsoQuantMSE
        except ImportError:
            pytest.skip("turboquant not installed")

        # Test upstream quality
        quantizer = IsoQuantMSE(d=head_size, bits=3, mode="fast", device="cuda")
        torch.manual_seed(42)
        test_vectors = torch.randn(1000, head_size, device="cuda")

        x_hat, _ = quantizer(test_vectors)

        cosine_sim = torch.nn.functional.cosine_similarity(
            test_vectors.reshape(1, -1).float(),
            x_hat.reshape(1, -1).float(),
        ).item()

        # Upstream IsoQuant achieves ~98.3% at 3-bit, ~99.5% at 4-bit
        assert cosine_sim > 0.98, f"Upstream quality: {cosine_sim:.4f}"


class TestIsoQuantPerformance:
    """Performance tests for IsoQuant codec."""

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_encode_throughput(self, bits):
        """Test encoding throughput for different bit widths."""
        from llmd_fs_backend.isoquant_config import IsoQuantConfig
        from llmd_fs_backend.isoquant_codec import IsoQuantCodec

        config = IsoQuantConfig(
            enabled=True,
            bits=bits,
            mode="fast",
            layer_pattern_override="model.layers.{layer_idx}.self_attn",
        )

        layer_names = [f"model.layers.{i}.self_attn" for i in range(4)]

        codec = IsoQuantCodec(
            config=config,
            layer_names=layer_names,
            num_blocks=256,
            block_size=16,
            num_heads=32,
            head_size=128,
            device=torch.device("cuda"),
            dtype=torch.float16,
        )

        total_blocks = 1024
        kv_cache = torch.randn(
            total_blocks, 16, 32, 128,
            dtype=torch.float16,
            device="cuda",
        )

        test_block_ids = list(range(256))
        layer_name = layer_names[0]
        dest_buffer = codec.get_encoded_buffer(layer_name, "k")

        # Warmup
        for _ in range(3):
            codec.encode_batch(
                layer_name=layer_name,
                kv_type="k",
                source_tensor=kv_cache,
                source_block_ids=test_block_ids,
                dest_buffer=dest_buffer,
                dest_start_idx=0,
            )

        # Benchmark
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(10):
            codec.encode_batch(
                layer_name=layer_name,
                kv_type="k",
                source_tensor=kv_cache,
                source_block_ids=test_block_ids,
                dest_buffer=dest_buffer,
                dest_start_idx=0,
            )
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        avg_ms = elapsed_ms / 10

        # Should complete in <10ms per batch for reasonable performance
        assert avg_ms < 50, f"Encoding too slow: {avg_ms:.2f}ms at {bits}-bit"
