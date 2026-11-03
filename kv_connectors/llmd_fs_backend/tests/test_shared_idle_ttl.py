# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""One inode across idle lifetimes, atomic max deadline, native/store transport."""

import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from test_storage_ttl import ttl


@pytest.fixture
def shared_file(tmp_path):
    p = tmp_path / ttl.SHARED_IDLE_NAME / "model_r0/abc/00_g0/abc000.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"same shared KV bytes")
    os.setxattr(p, ttl.DEADLINE_XATTR, str(time.time_ns() - 1).encode())
    return p


@pytest.mark.no_cuda_required
@pytest.mark.parametrize("value", [None, b"", b"invalid", b"-1", b"0", b"9" * 32])
def test_unknown_deadline_is_not_a_hit_or_proof_of_expiry(shared_file, value):
    import storage_offload

    if value is None:
        os.removexattr(shared_file, ttl.DEADLINE_XATTR)
    else:
        os.setxattr(shared_file, ttl.DEADLINE_XATTR, value)
    assert ttl.shared_idle_deadline(shared_file) is None
    assert storage_offload.try_open_idle_hit(str(shared_file), 30) == -1
    assert ttl.try_delete(shared_file, pressure=False, idle_seconds=0) is None
    assert shared_file.exists()


@pytest.mark.no_cuda_required
def test_short_hit_cannot_shorten_long_lease_or_replace_bytes(shared_file):
    import storage_offload

    before = shared_file.stat()
    lower = time.time_ns() + 600 * ttl.NANOSECONDS
    fd = storage_offload.try_open_idle_hit(str(shared_file), 600)
    assert fd >= 0
    os.close(fd)
    long = ttl.shared_idle_deadline(shared_file)
    assert long >= lower
    fd = storage_offload.try_open_idle_hit(str(shared_file), 1)
    assert fd >= 0
    assert ttl.shared_idle_deadline(fd) == long
    assert os.pread(fd, 100, 0) == b"same shared KV bytes"
    after = shared_file.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert ttl.try_delete(shared_file, pressure=False, idle_seconds=0) is None
    os.setxattr(shared_file, ttl.DEADLINE_XATTR, b"1")
    assert ttl.try_delete(shared_file, pressure=False, idle_seconds=0)
    assert os.pread(fd, 100, 0) == b"same shared KV bytes"
    os.close(fd)


@pytest.mark.no_cuda_required
def test_parallel_short_and_long_hits_preserve_max_deadline(shared_file):
    import storage_offload

    start = time.time_ns()

    def renew(seconds):
        end = time.monotonic() + 5
        while True:
            fd = storage_offload.try_open_idle_hit(str(shared_file), seconds)
            if fd == -2:
                assert time.monotonic() < end
                time.sleep(0.001)
                continue
            assert fd >= 0
            os.close(fd)
            return ttl.shared_idle_deadline(shared_file)

    with ThreadPoolExecutor(max_workers=8) as pool:
        values = list(pool.map(renew, [600, 1, 30, 1] * 8))
    assert all(x is not None for x in values)
    assert ttl.shared_idle_deadline(shared_file) >= start + 600 * ttl.NANOSECONDS
    assert len(list(shared_file.parents[3].rglob("*.bin"))) == 1


@pytest.mark.no_cuda_required
def test_busy_shared_hit_is_pending_and_expired_candidate_is_rechecked(shared_file):
    code = (
        "import fcntl,sys; f=open(sys.argv[1],'a+'); "
        "fcntl.lockf(f,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.readline()"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(ttl.lock_path(shared_file))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    pins = ttl.InodePins(30, idle=True, shared=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        assert pins.lookup("r", b"key", str(shared_file)) is None
        with pytest.raises(BlockingIOError):
            ttl.try_delete(shared_file, pressure=False, idle_seconds=0, defer_busy=True)
        child.communicate("release\n", timeout=5)
        deadline = time.monotonic() + 2
        while pins.lookup("r", b"key", str(shared_file)) is None:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        assert ttl.try_delete(shared_file, pressure=False, idle_seconds=0) is None
    finally:
        pins.close()
        if child.poll() is None:
            child.kill()
            child.wait()


@pytest.mark.no_cuda_required
def test_request_durations_share_mapper_and_reach_internal_store_spec(tmp_path):
    from test_spec import make_spec
    from vllm.v1.kv_offload.base import ReqContext

    spec = make_spec(
        tmp_path, {"storage_ttl_seconds": 30, "storage_ttl_allowlist": [1, 30]}
    )
    manager = spec.get_manager()
    key = b"prefix" + (0).to_bytes(4, "big")
    paths = []
    for seconds in [1, 30]:
        ctx = ReqContext(str(seconds), {"llmd_storage_ttl_seconds": seconds})
        manager.on_new_request(ctx)
        out = manager.prepare_store([key], ctx)
        restored = pickle.loads(pickle.dumps(out.store_spec))
        assert restored.ttl_seconds == seconds
        paths.append(next(iter(restored.source_paths.values())))
        manager.complete_store([key], ctx, success=False)
        manager.on_request_finished(ctx)
    assert paths[0] == paths[1]
    assert (
        manager._managers[1].file_mapper.base_path
        == manager._managers[30].file_mapper.base_path
    )
    manager.shutdown()


def test_native_shared_store_keeps_one_generation_across_durations(tmp_path):
    from types import SimpleNamespace

    import torch
    from test_request_storage_ttl_native import wait
    from vllm.v1.kv_offload.base import GPULoadStoreSpec, ReqContext, make_offload_key

    from llmd_fs_backend.request_manager import RequestStorageTTLManager
    from llmd_fs_backend.worker import (
        GPUToStorageHandler,
        StorageToGPUHandler,
        storage_offload,
    )

    root = tmp_path / ttl.SHARED_IDLE_NAME
    mapper = SimpleNamespace(
        model_name="shared",
        get_file_name=lambda k: str(root / "model_r0/abc/00_g0" / (k.hex() + ".bin")),
    )
    manager = RequestStorageTTLManager(
        mapper, {1: mapper, 600: mapper}, {"storage_ttl_policy": "idle"}, 600
    )
    tensor = (
        torch.arange(512, dtype=torch.int32, device="cuda").view(8, 64).to(torch.int8)
    )
    expected = tensor[0].clone()
    engine = storage_offload.StorageOffloadEngine(
        2, 1, [tensor], [[0]], [64], 1, "disabled", 0, True
    )
    put = GPUToStorageHandler(1, mapper, engine, ("GPU", "SHARED_STORAGE"), [64])
    get = StorageToGPUHandler(1, mapper, engine, ("SHARED_STORAGE", "GPU"), [64])
    key = make_offload_key(bytes.fromhex("ab" * 16), 0)
    path = Path(mapper.get_file_name(key))
    before = time.time_ns()
    for job, seconds in [(1, 600), (2, 1)]:
        ctx = ReqContext(str(job), {"llmd_storage_ttl_seconds": seconds})
        manager.on_new_request(ctx)
        out = manager.prepare_store([key], ctx)
        assert out.store_spec.ttl_seconds == seconds
        assert put.transfer_async(
            job, (GPULoadStoreSpec([0], [1], [0]), out.store_spec)
        )
        wait(put, job)
        manager.complete_store([key], ctx)
        manager.on_request_finished(ctx)
        if job == 1:
            first = path.stat()
            deadline = ttl.shared_idle_deadline(path)
            assert deadline >= before + 600 * ttl.NANOSECONDS
            tensor[0].zero_()
        else:
            assert (path.stat().st_ino, path.stat().st_mtime_ns) == (
                first.st_ino,
                first.st_mtime_ns,
            )
            assert ttl.shared_idle_deadline(path) == deadline
    assert len(list(tmp_path.rglob("*.bin"))) == 1
    ctx = ReqContext("read", {"llmd_storage_ttl_seconds": 1})
    manager.on_new_request(ctx)
    assert manager.lookup(key, ctx)
    spec = manager.prepare_load([key], ctx)
    path.unlink()
    manager.on_request_finished(ctx)
    assert get.transfer_async(3, (spec, GPULoadStoreSpec([3], [1], [0])))
    wait(get, 3)
    manager.complete_load([key], ctx)
    assert torch.equal(tensor[3], expected)
    assert not manager._tokens and not manager._jobs
    manager.shutdown()
