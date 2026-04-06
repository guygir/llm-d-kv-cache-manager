"""
Utility to test IsoQuant/RotorQuant compression directly without memory pressure.

This module allows testing the compression quality and performance on real 
or synthetic KV cache data without needing to trigger actual offloading.

Usage (in the cluster pod):
    python -c "from llmd_fs_backend.test_compression import test_compression; test_compression()"
    
Or with custom settings:
    python -c "
    from llmd_fs_backend.test_compression import test_compression
    test_compression(bits=4, num_blocks=100, measure_perplexity=True)
    "
"""

import time
from typing import Optional

import torch
import torch.nn.functional as F


def test_compression(
    bits: int = 3,
    mode: str = "fast",
    num_blocks: int = 16,
    block_size: int = 16,
    num_heads: int = 8,
    head_dim: int = 128,
    device: str = "cuda",
    use_real_distribution: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Test IsoQuant compression quality and performance.
    
    Args:
        bits: Quantization bits (2, 3, or 4)
        mode: Quantization mode ("fast" or "full")
        num_blocks: Number of KV cache blocks to test
        block_size: Tokens per block
        num_heads: Number of attention heads
        head_dim: Dimension per head
        device: Device to run on ("cuda" or "cpu")
        use_real_distribution: If True, use RMSNorm-like distribution
        verbose: Print detailed results
        
    Returns:
        Dict with quality metrics and timing
    """
    try:
        from turboquant import IsoQuantMSE, IsoQuantProd
    except ImportError:
        print("ERROR: turboquant not installed. Install with:")
        print("  pip install git+https://github.com/scrya-com/rotorquant.git")
        return {"error": "turboquant not installed"}
    
    results = {}
    
    # Create test tensors
    if verbose:
        print(f"\n{'='*60}")
        print(f"IsoQuant Compression Test")
        print(f"{'='*60}")
        print(f"Config: bits={bits}, mode={mode}")
        print(f"Data: {num_blocks} blocks × {block_size} tokens × {num_heads} heads × {head_dim} dim")
        print(f"Device: {device}")
    
    # Generate test data - shape: [num_blocks * block_size, num_heads * head_dim]
    total_tokens = num_blocks * block_size
    dim = num_heads * head_dim
    
    if use_real_distribution:
        # RMSNorm-like distribution (unit variance, centered)
        k_cache = torch.randn(total_tokens, dim, device=device, dtype=torch.float16)
        v_cache = torch.randn(total_tokens, dim, device=device, dtype=torch.float16)
        # Normalize like RMSNorm output
        k_cache = k_cache / k_cache.norm(dim=-1, keepdim=True) * (dim ** 0.5)
        v_cache = v_cache / v_cache.norm(dim=-1, keepdim=True) * (dim ** 0.5)
    else:
        k_cache = torch.randn(total_tokens, dim, device=device, dtype=torch.float16)
        v_cache = torch.randn(total_tokens, dim, device=device, dtype=torch.float16)
    
    # Initialize quantizers
    # K uses IsoQuantProd (preserves dot products)
    # V uses IsoQuantMSE (minimizes reconstruction error)
    k_quantizer = IsoQuantProd(d=dim, bits=bits, mode=mode, device=device)
    v_quantizer = IsoQuantMSE(d=dim, bits=bits, mode=mode, device=device)
    
    if verbose:
        print(f"\nQuantizers initialized (K=Prod, V=MSE)")
    
    # Test K cache compression
    if verbose:
        print(f"\n--- K Cache ---")
    
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    k_reconstructed, k_indices = k_quantizer(k_cache.float())
    torch.cuda.synchronize() if device == "cuda" else None
    k_time = time.time() - start
    
    k_cos_sim = F.cosine_similarity(
        k_cache.float().flatten().unsqueeze(0),
        k_reconstructed.flatten().unsqueeze(0)
    ).item()
    
    k_mse = F.mse_loss(k_cache.float(), k_reconstructed).item()
    
    results["k_cosine_similarity"] = k_cos_sim
    results["k_mse"] = k_mse
    results["k_encode_time_ms"] = k_time * 1000
    
    if verbose:
        print(f"  Cosine similarity: {k_cos_sim:.6f} ({k_cos_sim*100:.2f}%)")
        print(f"  MSE: {k_mse:.6f}")
        print(f"  Encode time: {k_time*1000:.2f}ms")
    
    # Test V cache compression
    if verbose:
        print(f"\n--- V Cache ---")
    
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.time()
    v_reconstructed, v_indices = v_quantizer(v_cache.float())
    torch.cuda.synchronize() if device == "cuda" else None
    v_time = time.time() - start
    
    v_cos_sim = F.cosine_similarity(
        v_cache.float().flatten().unsqueeze(0),
        v_reconstructed.flatten().unsqueeze(0)
    ).item()
    
    v_mse = F.mse_loss(v_cache.float(), v_reconstructed).item()
    
    results["v_cosine_similarity"] = v_cos_sim
    results["v_mse"] = v_mse
    results["v_encode_time_ms"] = v_time * 1000
    
    if verbose:
        print(f"  Cosine similarity: {v_cos_sim:.6f} ({v_cos_sim*100:.2f}%)")
        print(f"  MSE: {v_mse:.6f}")
        print(f"  Encode time: {v_time*1000:.2f}ms")
    
    # Overall metrics
    avg_cos_sim = (k_cos_sim + v_cos_sim) / 2
    total_time = k_time + v_time
    
    # Calculate compression ratio
    original_bytes = total_tokens * dim * 2 * 2  # 2 caches (K,V) × 2 bytes (fp16)
    compressed_bytes = total_tokens * dim * bits / 8 * 2  # 2 caches × bits/8 bytes
    compression_ratio = original_bytes / compressed_bytes
    
    results["avg_cosine_similarity"] = avg_cos_sim
    results["compression_ratio"] = compression_ratio
    results["total_encode_time_ms"] = total_time * 1000
    results["throughput_GB_per_sec"] = (original_bytes / 1e9) / total_time if total_time > 0 else 0
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Average cosine similarity: {avg_cos_sim:.6f} ({avg_cos_sim*100:.2f}%)")
        print(f"Compression ratio: {compression_ratio:.1f}x ({bits}-bit vs FP16)")
        print(f"Total encode time: {total_time*1000:.2f}ms")
        print(f"Throughput: {results['throughput_GB_per_sec']:.2f} GB/s")
        print(f"\nQuality assessment:")
        if avg_cos_sim >= 0.99:
            print(f"  ✅ EXCELLENT (>99% similarity)")
        elif avg_cos_sim >= 0.98:
            print(f"  ✅ GOOD (>98% similarity)")
        elif avg_cos_sim >= 0.95:
            print(f"  ⚠️  ACCEPTABLE (>95% similarity)")
        else:
            print(f"  ❌ LOW (<95% similarity)")
    
    return results


def test_roundtrip_with_codec(
    bits: int = 3,
    mode: str = "fast",
    num_blocks: int = 16,
    verbose: bool = True,
) -> dict:
    """
    Test the full IsoQuantCodec encode/decode roundtrip.
    
    This tests the actual codec that would be used during offloading,
    including the batch-level operations.
    """
    try:
        from llmd_fs_backend.isoquant_codec import IsoQuantCodec
        from llmd_fs_backend.isoquant_config import IsoQuantConfig
    except ImportError:
        print("ERROR: Could not import IsoQuantCodec")
        return {"error": "import failed"}
    
    config = IsoQuantConfig(enabled=True, bits=bits, mode=mode)
    
    # Simulate vLLM-like parameters
    num_layers = 32
    num_heads = 8
    head_dim = 128
    block_size = 16
    total_blocks = num_blocks
    blocks_per_batch = min(16, num_blocks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"IsoQuantCodec Roundtrip Test")
        print(f"{'='*60}")
        print(f"Config: bits={bits}, mode={mode}")
        print(f"Layers: {num_layers}, Heads: {num_heads}, HeadDim: {head_dim}")
        print(f"Blocks: {num_blocks} × {block_size} tokens")
    
    # Initialize codec
    codec = IsoQuantCodec(
        config=config,
        num_layers=num_layers,
        total_blocks=total_blocks,
        blocks_per_batch=blocks_per_batch,
        tokens_per_block=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
        device=device,
    )
    
    # Create test KV cache tensor for one layer
    # Shape: [total_blocks, block_size, num_heads, head_dim]
    test_k = torch.randn(
        total_blocks, block_size, num_heads, head_dim,
        device=device, dtype=torch.float16
    )
    test_v = torch.randn(
        total_blocks, block_size, num_heads, head_dim,
        device=device, dtype=torch.float16
    )
    
    results = {}
    
    # Test encode/decode for each layer
    for layer_idx in range(min(3, num_layers)):  # Test first 3 layers
        layer_name = f"layer_{layer_idx}"
        
        # Encode K
        encoded_k = codec.encode_batch(
            layer_name=layer_name,
            kv_type="k",
            source_tensor=test_k,
            source_block_ids=list(range(num_blocks)),
            dest_buffer=None,
            dest_start_idx=0,
        )
        
        # Decode K
        decoded_k = torch.zeros_like(test_k)
        codec.decode_batch(
            layer_name=layer_name,
            kv_type="k",
            source_buffer=encoded_k,
            source_start_idx=0,
            dest_tensor=decoded_k,
            dest_block_ids=list(range(num_blocks)),
        )
        
        # Measure quality
        k_sim = F.cosine_similarity(
            test_k.flatten().float().unsqueeze(0),
            decoded_k.flatten().float().unsqueeze(0)
        ).item()
        
        if verbose:
            print(f"Layer {layer_idx} K: {k_sim*100:.2f}% similarity")
        
        results[f"layer_{layer_idx}_k_similarity"] = k_sim
    
    return results


if __name__ == "__main__":
    # Run basic test
    results = test_compression(bits=3, mode="fast", num_blocks=64)
    print(f"\nResults: {results}")
