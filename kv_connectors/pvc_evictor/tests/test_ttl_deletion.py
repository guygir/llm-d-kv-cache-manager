"""Exercise the packaged TTL protocol through the production deletion path."""

import importlib.util
import logging
import os
import sys
import time
import types
from pathlib import Path

import pytest
from test_deleter import _deleter


@pytest.fixture
def ttl_protocol(monkeypatch):
    path = Path(__file__).parents[2] / "llmd_fs_backend/llmd_fs_backend/storage_ttl.py"
    spec = importlib.util.spec_from_file_location("llmd_fs_backend.storage_ttl", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The evictor image copies this filesystem-only module without vLLM's init.
    package = types.ModuleType("llmd_fs_backend")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "llmd_fs_backend", package)
    monkeypatch.setitem(sys.modules, "llmd_fs_backend.storage_ttl", module)
    return module


def candidate(root, *, ttl=True, age=0, idle=0):
    namespace = root / ".llmd-ttl-v1-30" if ttl else root
    path = namespace / "model_r0/abc/00_g0/abc000.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"KV bytes")
    now = time.time()
    os.utime(path, (now - idle, now - age))
    return path


@pytest.mark.parametrize("dry_run", [False, True])
def test_expiry_without_pressure_respects_dry_run(tmp_path, ttl_protocol, dry_run):
    expired = candidate(tmp_path, age=60)
    result = _deleter.delete_batch([str(expired)], dry_run, logging.getLogger("test"), pressure_enabled=False)
    assert result == ((1, 0, []) if dry_run else (1, 8, [str(expired)]))
    assert expired.exists() == dry_run


def test_pressure_off_keeps_unexpired_ttl_and_legacy_files(tmp_path, ttl_protocol):
    fresh = candidate(tmp_path, idle=7200)
    legacy = candidate(tmp_path, ttl=False, idle=7200)
    assert _deleter.delete_batch(
        [str(fresh), str(legacy)], False, logging.getLogger("test"), pressure_enabled=False
    ) == (0, 0, [])
    assert fresh.exists() and legacy.exists()


def test_pressure_can_remove_idle_unexpired_ttl(tmp_path, ttl_protocol):
    idle = candidate(tmp_path, idle=7200)
    assert _deleter.delete_batch([str(idle)], False, logging.getLogger("test")) == (1, 8, [str(idle)])
    assert not idle.exists()


def test_mixed_batch_preserves_hot_legacy_file(tmp_path, ttl_protocol):
    expired = candidate(tmp_path, age=60)
    hot = candidate(tmp_path, ttl=False)
    assert _deleter.delete_batch([str(expired), str(hot)], False, logging.getLogger("test")) == (1, 8, [str(expired)])
    assert hot.exists() and not expired.exists()


def test_vanished_ttl_candidate_is_not_counted(tmp_path, ttl_protocol):
    vanished = candidate(tmp_path, age=60)
    vanished.unlink()
    assert _deleter.delete_batch([str(vanished)], False, logging.getLogger("test")) == (0, 0, [])


def test_retry_queue_is_bounded_deduplicated_and_delayed(monkeypatch):
    monkeypatch.setattr(_deleter.time, "monotonic", lambda: 10)
    pending = _deleter.PendingDeletes(limit=2, delay=0.25)
    assert pending.add("a") and pending.add("a") and pending.add("b")
    assert not pending.add("overflow")
    assert pending.take_due(2) == []
    monkeypatch.setattr(_deleter.time, "monotonic", lambda: 10.25)
    assert pending.take_due(1) == ["a"]
    assert pending.take_due(1) == ["b"]
    assert not pending.paths and not pending.heap


def test_retry_rechecks_renewed_idle_generation(tmp_path, ttl_protocol):
    path = tmp_path / (ttl_protocol.IDLE_PREFIX + "30") / "model/abc/00_g0/abc000.bin"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bytes")
    old = time.time_ns() - 60_000_000_000
    os.utime(path, ns=(old, old))
    pending = _deleter.PendingDeletes(delay=0)
    pending.add(str(path))
    os.utime(path, ns=(time.time_ns(), old))
    assert _deleter.delete_batch(
        pending.take_due(1), False, logging.getLogger("test"), pressure_enabled=False, pending_deletes=pending
    ) == (0, 0, [])
    assert path.read_bytes() == b"bytes"


@pytest.mark.parametrize("release", [False, True])
def test_production_deleter_retries_busy_bucket_and_shuts_down(tmp_path, ttl_protocol, release):
    import queue
    import subprocess
    import threading

    busy = candidate(tmp_path, age=60)
    other = busy.parent / "def000.bin"
    other.write_bytes(b"other")
    old = time.time() - 60
    os.utime(other, (old, old))
    code = (
        "import fcntl,sys; f=open(sys.argv[1],'a+'); "
        "fcntl.lockf(f,fcntl.LOCK_EX); print('ready',flush=True); sys.stdin.readline()"
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(ttl_protocol.lock_path(busy))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    shutdown, pressure = threading.Event(), threading.Event()
    files, results = queue.Queue(), queue.Queue()
    files.put(str(busy))
    files.put(str(other))
    thread = threading.Thread(
        target=_deleter.deleter_process,
        args=(
            1,
            tmp_path,
            {
                "deletion_batch_size": 1,
                "dry_run": False,
                "enable_storage_ttl": True,
                "file_access_time_threshold_minutes": 60,
            },
            pressure,
            files,
            results,
            shutdown,
        ),
    )
    try:
        assert holder.stdout.readline().strip() == "ready"
        thread.start()
        deadline = time.monotonic() + 5
        while other.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not other.exists() and busy.exists()
        if release:
            holder.communicate("release\n", timeout=5)
            while busy.exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not busy.exists()  # No crawler or second queue insertion.
    finally:
        shutdown.set()
        if thread.ident is not None:
            thread.join(timeout=2)
        if holder.poll() is None:
            holder.kill()
            holder.wait()
    assert not thread.is_alive()
    assert busy.exists() != release


@pytest.mark.skipif(sys.platform != "linux", reason="Shared idle TTL uses Linux inode xattrs")
def test_shared_deadline_rechecked_and_unknown_metadata_kept(tmp_path, ttl_protocol):
    p = tmp_path / ttl_protocol.SHARED_IDLE_NAME / "model_r0/abc/00_g0/abc000.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"KV")
    pending = _deleter.PendingDeletes(delay=0)
    os.setxattr(p, ttl_protocol.DEADLINE_XATTR, b"1")
    pending.add(str(p))
    os.setxattr(p, ttl_protocol.DEADLINE_XATTR, str(time.time_ns() + 60_000_000_000).encode())
    assert _deleter.delete_batch(pending.take_due(1), False, logging.getLogger("test"), pressure_enabled=False) == (0, 0, [])
    os.removexattr(p, ttl_protocol.DEADLINE_XATTR)
    assert _deleter.delete_batch([str(p)], False, logging.getLogger("test"), pressure_enabled=False) == (0, 0, [])
    assert p.read_bytes() == b"KV"


@pytest.mark.skipif(sys.platform != "linux", reason="Shared idle TTL uses Linux inode xattrs")
@pytest.mark.parametrize("dry_run", [False, True])
def test_shared_deadline_expiry_respects_dry_run(tmp_path, ttl_protocol, dry_run):
    p = tmp_path / ttl_protocol.SHARED_IDLE_NAME / "model_r0/abc/00_g0/abc000.bin"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"KV")
    os.setxattr(p, ttl_protocol.DEADLINE_XATTR, b"1")
    result = _deleter.delete_batch([str(p)], dry_run, logging.getLogger("test"), pressure_enabled=False)
    assert result[0] == 1 and p.exists() == dry_run


@pytest.mark.skipif(sys.platform != "linux", reason="Shared idle TTL uses Linux inode xattrs")
def test_shared_actual_crawler_and_deleter_expire_without_pressure(tmp_path, ttl_protocol):
    import queue
    import threading
    from processes.crawler import crawler_process

    root = tmp_path / ttl_protocol.SHARED_IDLE_NAME / "model_r0/abc/00_g0"
    root.mkdir(parents=True)
    due = []
    deadline = time.time_ns() + 1_000_000_000
    for i in range(8):
        p = root / f"abc{i:03x}.bin"
        p.write_bytes(b"KV")
        os.setxattr(p, ttl_protocol.DEADLINE_XATTR, str(deadline).encode())
        due.append(p)
    fresh = root / "abcfff.bin"
    fresh.write_bytes(b"fresh")
    os.setxattr(fresh, ttl_protocol.DEADLINE_XATTR, str(deadline + 60_000_000_000).encode())
    before = fresh.stat()
    pressure, shutdown = threading.Event(), threading.Event()
    files, results = queue.Queue(), queue.Queue()
    conf = dict(file_queue_min_size=64, file_queue_maxsize=128, file_access_time_threshold_minutes=60, enable_storage_ttl=True, dry_run=False, deletion_batch_size=1)
    workers = [threading.Thread(target=crawler_process, args=(0, (0, 15), tmp_path, conf, pressure, files, results, shutdown)), threading.Thread(target=_deleter.deleter_process, args=(2, tmp_path, conf, pressure, files, results, shutdown))]
    try:
        for thread in workers:
            thread.start()
        end = time.monotonic() + 8
        while any(p.exists() for p in due) and time.monotonic() < end:
            time.sleep(0.02)
        assert not any(p.exists() for p in due)
        assert not pressure.is_set()
        assert fresh.exists() and (fresh.stat().st_ino, fresh.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    finally:
        shutdown.set()
        for thread in workers:
            thread.join(timeout=2)
    assert all(not thread.is_alive() for thread in workers)
