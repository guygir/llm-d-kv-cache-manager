# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Idle policy accepts/renews existing bytes under native publication locks."""

import os
import subprocess
import sys
import time

import pytest
import torch
from test_storage_ttl import ttl


@pytest.fixture
def idle_file(tmp_path):
    p = tmp_path / (ttl.IDLE_PREFIX + "2") / "model_r0/abc/00_g0/abc000.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"original bytes")
    old = time.time_ns() - 10_000_000_000
    os.utime(p, ns=(old, old))
    return p


@pytest.mark.no_cuda_required
def test_idle_deadline_uses_access_and_fixed_uses_publication(idle_file):
    stat = idle_file.stat()
    assert ttl.expired(stat, ttl.policy_for_path(idle_file))
    now = time.time_ns()
    os.utime(idle_file, ns=(now, stat.st_mtime_ns))
    assert not ttl.expired(idle_file.stat(), ttl.policy_for_path(idle_file))
    assert ttl.expired(idle_file.stat(), (idle_file.parent / (ttl.PREFIX + "2"), 2))
    with pytest.raises(ValueError):
        ttl.policy_prefix("unknown")


@pytest.mark.no_cuda_required
@pytest.mark.skipif(sys.platform != "linux", reason="Native Linux descriptor protocol")
def test_native_idle_hit_renews_without_replacing_and_pins_unlinked_bytes(idle_file):
    import storage_offload

    before = idle_file.stat()
    pins = ttl.InodePins(2, idle=True, idle_opener=storage_offload.try_open_idle_hit)
    try:
        assert pins.lookup("r", b"key", str(idle_file)) is True
        after = idle_file.stat()
        assert after.st_ino == before.st_ino and after.st_mtime_ns == before.st_mtime_ns
        assert after.st_atime_ns > before.st_atime_ns
        assert ttl.try_delete(idle_file, pressure=False, idle_seconds=0) is None
        fd = pins.requests["r"]["current"][b"key"][0]
        idle_file.unlink()
        assert os.pread(fd, 100, 0) == b"original bytes"
    finally:
        pins.close()
    assert storage_offload.try_open_idle_hit(str(idle_file)) == -1


@pytest.mark.no_cuda_required
@pytest.mark.skipif(sys.platform != "linux", reason="Cross-process POSIX record locks")
def test_busy_idle_lookup_is_pending_then_renews(idle_file):
    import storage_offload

    code = (
        "import fcntl,sys; f=open(sys.argv[1],'a+'); "
        "fcntl.lockf(f,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.readline()"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(ttl.lock_path(idle_file))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    pins = ttl.InodePins(2, idle=True, idle_opener=storage_offload.try_open_idle_hit)
    try:
        assert child.stdout.readline().strip() == "ready"
        assert pins.lookup("r", b"key", str(idle_file)) is None
        assert not pins.requests["r"]["fds"]
        child.communicate("release\n", timeout=5)
        deadline = time.monotonic() + 1
        while pins.lookup("r", b"key", str(idle_file)) is None:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        assert pins.requests["r"]["current"]
    finally:
        pins.close()
        if child.poll() is None:
            child.kill()
            child.wait()


def test_idle_store_reuses_expired_present_generation(tmp_path):
    import storage_offload
    from test_access_time import wait_finished

    tensors = [
        torch.arange(256, dtype=torch.int32, device="cuda").view(4, 64).to(torch.int8)
    ]
    expected = tensors[0][0].clone()
    engine = storage_offload.StorageOffloadEngine(
        2, 1, tensors, [[0]], [64], 1, "disabled", 0, True
    )
    path = tmp_path / (ttl.IDLE_PREFIX + "2") / "model/abc/00_g0/abc000.bin"
    assert engine.async_store_gpu_blocks(1, [0], [str(path)], [[0]], [0])
    wait_finished(engine, 1)
    old = time.time_ns() - 10_000_000_000
    os.utime(path, ns=(old, old))
    before = path.stat()
    tensors[0][0].zero_()
    assert engine.async_store_gpu_blocks(2, [0], [str(path)], [[0]], [0])
    wait_finished(engine, 2)
    after = path.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert after.st_atime_ns > old
    assert engine.async_load_gpu_blocks(3, [0], [str(path)], [[3]], [0])
    wait_finished(engine, 3)
    assert torch.equal(tensors[0][3], expected)


@pytest.mark.no_cuda_required
@pytest.mark.skipif(sys.platform != "linux", reason="Linux descriptor protocol")
def test_busy_scheduler_polls_do_not_repeat_filesystem_probes(monkeypatch):
    calls = []
    clock = [1.0]
    monkeypatch.setattr(ttl.time, "monotonic", lambda: clock[0])
    pins = ttl.InodePins(2, idle=True, idle_opener=lambda p: calls.append(p) or -2)
    for _ in range(1000):
        assert pins.lookup("r", b"key", "path") is None
    assert calls == ["path"]
    clock[0] += 0.02
    assert pins.lookup("r", b"key", "path") is None
    assert calls == ["path", "path"]
    pins.finish("r")
    assert not pins.requests
