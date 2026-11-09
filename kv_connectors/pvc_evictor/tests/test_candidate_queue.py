"""Age-band behavior and real cross-process/filesystem safety."""

import contextlib
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from queue import Empty, Full

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from candidate_queue import AgeBandStore, Candidate, CandidateQueue, validate_bands
from config import Config


def ticket(path, age=0, **kw):
    return Candidate(path, time.time_ns() - int(age * 1e9), 1, 1, 1, 1, **kw)


def make(path, age):
    path.write_bytes(b"cache")
    stamp = time.time_ns() - int(age * 1e9)
    os.utime(path, ns=(stamp, stamp))
    return Candidate.from_stat(path, path.stat())


def test_oldest_band_first_not_exact_within_band(tmp_path):
    q = AgeBandStore(8, (60, 300, 3600))
    for name, age in [("recent", 90), ("old", 9000), ("middle", 900)]:
        q.put(ticket(str(tmp_path / name), age))
    assert [q.get(False).path.rsplit("/", 1)[-1] for _ in range(3)] == ["old", "middle", "recent"]


def test_lazy_aging_fixed_bucket_count(monkeypatch):
    now = [1_000_000_000_000]
    monkeypatch.setattr("candidate_queue.time.time_ns", lambda: now[0])
    q = AgeBandStore(4, (60, 300), aging_budget=8)
    q.put(Candidate("/old", now[0] - 30_000_000_000, 1, 1, 1, 1))
    now[0] += 400_000_000_000
    q.age_step(now[0])
    assert q.stats()["band_sizes"] == [0, 0, 1]
    assert q.stats()["aged"] == 1
    assert q.get(False).path == "/old"
    assert len(q.stats()["band_sizes"]) == 3


def test_bounded_deduplication_and_delayed_discovery():
    q = AgeBandStore(1, (60, 300))
    newer = ticket("/same", 100)
    q.put(newer)
    q.put(replace(newer, atime_ns=newer.atime_ns - 100_000_000_000))
    assert q.qsize() == 1 and q.get(False).atime_ns == newer.atime_ns
    q.put(newer)
    with pytest.raises(Full):
        q.put(ticket("/different", 200), False)
    assert q.stats()["occupancy"] == 1


def test_seeded_churn_cannot_accumulate_stale_tickets():
    rng = random.Random(73)
    q = AgeBandStore(8, (60, 300, 3600))
    for _ in range(2000):
        if rng.randrange(3):
            with contextlib.suppress(Full):
                q.put(ticket("/" + str(rng.randrange(16)), rng.randrange(10000)), False)
        else:
            with contextlib.suppress(Empty):
                q.get(False)
        stats = q.stats()
        assert stats["occupancy"] <= 8
        assert sum(stats["band_sizes"]) == stats["occupancy"]


def test_blocked_producer_wakes_on_space_and_close():
    q = AgeBandStore(1, (60,))
    q.put(ticket("/a"))
    outcome = []

    def put():
        q.put(ticket("/b"), timeout=2)
        outcome.append(True)

    thread = threading.Thread(target=put)
    thread.start()
    assert q.get(False).path == "/a"
    thread.join(2)
    assert outcome == [True]
    assert q.get(False).path == "/b"
    errors = []

    def get():
        try:
            q.get()
        except EOFError:
            errors.append(True)

    thread = threading.Thread(target=get)
    thread.start()
    q.close()
    thread.join(2)
    assert errors == [True] and not thread.is_alive()


def test_hot_hit_and_replaced_generation_are_not_candidates(tmp_path):
    store = AgeBandStore(8, (60, 300))
    q = CandidateQueue(store, (60, 300), 30)
    path = tmp_path / "warm.bin"
    old = make(path, 1000)
    q.put(old)
    os.utime(path, ns=(time.time_ns(), old.mtime_ns))
    with pytest.raises(Empty):
        q.get_nowait()
    assert store.stats()["hot_dropped"] == 1
    old = make(path, 1000)
    q.put(old)
    held = os.open(path, os.O_RDONLY)
    try:
        replacement = tmp_path / "new.bin"
        replacement.write_bytes(b"replacement generation")
        os.replace(replacement, path)
        with pytest.raises(Empty):
            q.get_nowait()
        assert path.read_bytes() == b"replacement generation"
        assert store.stats()["generation_dropped"] == 1
    finally:
        os.close(held)


def test_still_eligible_hit_moves_to_younger_band(tmp_path):
    store = AgeBandStore(8, (60, 300, 3600))
    q = CandidateQueue(store, (60, 300, 3600), 30)
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    q.put(make(a, 9000))
    q.put(make(b, 8000))
    st = a.stat()
    os.utime(a, ns=(time.time_ns() - 120_000_000_000, st.st_mtime_ns))
    assert q.get_nowait() == str(b)
    assert q.get_nowait() == str(a)
    assert store.stats()["reclassified"] == 1


@pytest.mark.parametrize("bands", [(), (0,), (60, 60), (300, 60), (float("inf"),)])
def test_invalid_boundaries(bands):
    with pytest.raises(ValueError):
        validate_bands(bands)


def test_opt_in_and_legacy_environment(monkeypatch):
    monkeypatch.setenv("CANDIDATE_AGE_BANDS_SECONDS", "bad")
    assert Config.from_env().candidate_queue_policy == "fifo"
    monkeypatch.setenv("CANDIDATE_QUEUE_POLICY", "age_bands")
    with pytest.raises(ValueError):
        Config.from_env()
    monkeypatch.setenv("CANDIDATE_AGE_BANDS_SECONDS", "60,300,3600,86400")
    assert Config.from_env().candidate_age_bands_seconds == (60, 300, 3600, 86400)


def test_factory_crawler_broker_and_actual_deleter(tmp_path):
    import signal
    from dataclasses import replace

    from evictor import PVCEvictor
    from processes.crawler import crawler_process
    from processes.deleter import deleter_process

    root = tmp_path / "cache"
    bucket = root / "model_0123456789ab_r0" / "abc" / "de_g0"
    bucket.mkdir(parents=True)
    old = [bucket / f"{i:016x}.bin" for i in range(8)]
    for path in old:
        make(path, 1000)
    hot = bucket / "hot.bin"
    hot.write_bytes(b"protected")
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    config = replace(
        Config.from_env(),
        pvc_mount_path=str(tmp_path),
        cache_directory="cache",
        candidate_queue_policy="age_bands",
        candidate_age_bands_seconds=(60, 300),
        file_queue_maxsize=32,
        file_queue_min_size=0,
        num_crawler_processes=1,
        deletion_batch_size=4,
        file_access_time_threshold_minutes=0.5,
        log_level="WARNING",
    )
    instance = PVCEvictor(config)
    ctx = mp.get_context("spawn")
    q, pressure, stop, results = (
        instance.deletion_queue,
        instance.deletion_event,
        instance.shutdown_event,
        instance.result_queue,
    )
    producers = ctx.Process(
        target=crawler_process, args=(1, (0, 15), root, instance.config_dict, pressure, q, results, stop)
    )
    consumer = ctx.Process(target=deleter_process, args=(10, root, instance.config_dict, pressure, q, results, stop))
    try:
        assert isinstance(q, CandidateQueue)
        producers.start()
        consumer.start()
        time.sleep(0.3)
        for path in old:
            stamp = time.time_ns() - 1000_000_000_000
            os.utime(path, ns=(stamp, stamp))
        pressure.set()
        end = time.monotonic() + 10
        while any(p.exists() for p in old) and time.monotonic() < end:
            with contextlib.suppress(Empty):
                results.get(timeout=0.05)
        assert not any(p.exists() for p in old)
        assert hot.read_bytes() == b"protected"
        stop.set()
        pressure.clear()
        end = time.monotonic() + 5
        while any(p.is_alive() for p in (producers, consumer)) and time.monotonic() < end:
            with contextlib.suppress(Empty):
                results.get(timeout=0.05)
            for process in (producers, consumer):
                process.join(0.02)
        assert producers.exitcode == consumer.exitcode == 0
    finally:
        stop.set()
        pressure.clear()
        for process in (producers, consumer):
            if process.is_alive():
                process.terminate()
                process.join(3)
        q.close()
        instance._candidate_manager.shutdown()
        for queue in (results, instance.folder_queue):
            if queue is not None:
                queue.close()
                queue.join_thread()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf")])
def test_invalid_timeout_never_spins(timeout):
    q = AgeBandStore(1, (60,))
    with pytest.raises(ValueError):
        q.get(timeout=timeout)
    with pytest.raises(ValueError):
        q.put(ticket("/x"), timeout=timeout)


def test_close_wakes_a_full_queue_producer():
    q = AgeBandStore(1, (60,))
    q.put(ticket("/x"))
    errors = []

    def put():
        try:
            q.put(ticket("/y"))
        except ValueError:
            errors.append(True)

    thread = threading.Thread(target=put)
    thread.start()
    q.close()
    thread.join(2)
    assert errors == [True] and not thread.is_alive()


def test_broker_death_fails_before_spawning_cleanup(tmp_path, monkeypatch):
    import signal

    from evictor import PVCEvictor

    cfg = replace(Config.from_env(), pvc_mount_path=str(tmp_path), candidate_queue_policy="age_bands")
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    instance = PVCEvictor(cfg)
    instance._candidate_manager.shutdown()
    monkeypatch.setattr(
        "evictor.multiprocessing.Process", lambda *a, **k: pytest.fail("cleanup spawned with dead broker")
    )
    try:
        with pytest.raises(RuntimeError, match="index process died"):
            instance.run()
    finally:
        for q in (instance.result_queue, instance.folder_queue):
            if q is not None:
                q.close()
                q.join_thread()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def _broker_crash_child(config, result):
    from evictor import PVCEvictor

    instance = PVCEvictor(config)

    def crash():
        instance._candidate_manager._process.terminate()
        instance._candidate_manager._process.join(2)

    timer = threading.Timer(1, crash)
    timer.start()
    try:
        instance.run()
        result.send(("unexpected success", True))
    except RuntimeError as exc:
        result.send((str(exc), any(p.is_alive() for p in mp.active_children())))
    finally:
        timer.cancel()
        timer.join(2)
        result.close()


def test_active_broker_failure_stops_cleanup_children(tmp_path):
    cfg = replace(
        Config.from_env(),
        pvc_mount_path=str(tmp_path),
        cache_directory="cache",
        candidate_queue_policy="age_bands",
        enable_dir_cleanup=False,
        num_crawler_processes=1,
        log_level="CRITICAL",
    )
    (tmp_path / "cache").mkdir()
    ctx = mp.get_context("spawn")
    read, write = ctx.Pipe(False)
    process = ctx.Process(target=_broker_crash_child, args=(cfg, write))
    process.start()
    write.close()
    try:
        assert read.poll(20), "Cleanup supervisor did not terminate after broker failure"
        message, alive = read.recv()
        process.join(3)
        assert "index process died" in message and not alive and process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(3)
        read.close()
