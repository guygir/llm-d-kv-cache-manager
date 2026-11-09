"""Policy and resource-envelope invariants; no GPU/cluster dependencies."""

import io
import multiprocessing
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from activity import fetch_activity, parse_gauges
from cleanup_policy import ActivitySource, CleanupController, CleanupSettings, validate_cpu_budget
from config import Config


def settings(**kw):
    return CleanupSettings((ActivitySource("http://127.0.0.1:8000/metrics"),), 1.0, **kw)


def test_hysteresis_quiet_and_immediate_busy_pause():
    c = CleanupController(settings(), 85, 70)
    assert not c.decide(80, True, 0).enabled
    assert c.decide(86, True, 2).mode == "quiet-drain"
    assert c.decide(82, True, 3).enabled  # pressure latch survives below start threshold
    assert not c.decide(82, False, 3.1).enabled  # never hold ON across incoming work
    assert not c.decide(82, True, 4).enabled
    assert c.decide(82, True, 6).enabled
    assert not c.decide(70, True, 7).enabled
    assert not c.decide(82, True, 8).enabled  # target cleared the pressure latch


def test_urgent_unknown_activity_and_missing_disk():
    c = CleanupController(settings(), 85, 70)
    assert not c.decide(86, None, 0).enabled
    assert c.decide(96, None, 1).mode == "urgent-continuous"
    assert c.decide(90, False, 2).enabled
    assert not c.decide(85, False, 3).enabled  # urgency release uses lower watermark
    assert not c.decide(None, True, 4).enabled
    assert c.decide(90, True, 6).enabled


def test_urgent_windows_recovery_and_backwards_clock():
    c = CleanupController(settings(busy_mode="windowed", quiet_seconds=2), 85, 70)
    assert c.decide(96, False, 10).enabled
    assert not c.decide(96, False, 11).enabled
    assert c.decide(96, False, 12).enabled
    assert c.decide(96, True, 14).enabled
    assert c.decide(96, True, 16).mode == "quiet-drain"  # quiet drain has no periodic pauses
    assert c.decide(96, False, 0).enabled  # monotonic reset restarts urgent window
    assert not c.decide(float("nan"), True, 1).enabled


@pytest.mark.parametrize(
    "kw",
    [
        {"cpu_budget_cores": 0},
        {"quiet_seconds": -1},
        {"window_on_seconds": float("nan")},
        {"poll_timeout_seconds": 0},
        {"emergency_threshold": 84},
        {"busy_mode": "guess"},
        {"cpu_ids": (1, 1)},
    ],
)
def test_invalid_settings_fail_before_service_start(kw):
    with pytest.raises(ValueError):
        replace(settings(), **kw).validate(85, 70)


def test_disabled_default_preserves_legacy_config(monkeypatch):
    monkeypatch.delenv("CLEANUP_LOAD_AWARE", raising=False)
    monkeypatch.setenv("CLEANUP_ACTIVITY_SOURCES", "not-json")
    assert Config.from_env().cleanup_policy is None
    assert Config.from_env().to_dict()["cleanup_policy"] is None
    monkeypatch.setenv("CLEANUP_LOAD_AWARE", "true")
    with pytest.raises(ValueError):
        Config.from_env()


def test_runtime_cpu_limit_and_permitted_affinity(tmp_path, monkeypatch):
    (tmp_path / "cpu.max").write_text("25000 100000")
    calls = []
    monkeypatch.setattr("os.sched_getaffinity", lambda _: {2, 3}, raising=False)
    monkeypatch.setattr("os.sched_setaffinity", lambda _, ids: calls.append(tuple(ids)), raising=False)
    assert validate_cpu_budget(replace(settings(), cpu_ids=(2,)), tmp_path) == 0.25
    assert calls == [(2,)]
    with pytest.raises(ValueError):
        validate_cpu_budget(replace(settings(), cpu_ids=(0,)), tmp_path)
    (tmp_path / "cpu.max").write_text("200000 100000")
    with pytest.raises(ValueError):
        validate_cpu_budget(settings(), tmp_path)
    (tmp_path / "cpu.max").write_text("max 100000")
    with pytest.raises(ValueError):
        validate_cpu_budget(settings(), tmp_path)


def test_gauges_sum_engines_and_never_assume_missing_is_zero():
    text = '# HELP running test\nrunning{engine="0"} 0\nrunning{engine="1"} 2\nwaiting 0\n'
    assert parse_gauges(text, ("running", "waiting")) == {"running": 2, "waiting": 0}
    for broken in ["running 0", "running NaN\nwaiting 0", "running -1\nwaiting 0"]:
        with pytest.raises(ValueError):
            parse_gauges(broken, ("running", "waiting"))


def test_all_sources_required_and_no_stale_idle_fallback():
    sources = tuple(ActivitySource("http://localhost/" + n, "running", "waiting") for n in ("a", "b"))
    responses = {"a": b"running 0\nwaiting 0", "b": b"running 1\nwaiting 0"}

    def opener(url, **_):
        return io.BytesIO(responses[url.rsplit("/", 1)[1]])

    assert fetch_activity(sources, 1, opener=opener).idle is False
    responses["b"] = b"running 0\nwaiting 0"
    assert fetch_activity(sources, 1, opener=opener).idle is True
    responses["b"] = b"invalid"
    assert fetch_activity(sources, 1, opener=opener).idle is None
    assert fetch_activity([], 1, opener=opener).idle is None


def test_activity_timeout_and_response_limit():
    source = ActivitySource("http://localhost/metrics", "running", "waiting")
    assert fetch_activity([source], 0, opener=lambda *a, **k: pytest.fail("must not connect")).idle is None
    assert fetch_activity([source], 1, opener=lambda *a, **k: io.BytesIO(b"x" * ((1 << 20) + 1))).idle is None


def test_paused_crawler_does_not_advance_walk():
    from processes.crawler import admitted_files

    pressure, stop = multiprocessing.Event(), multiprocessing.Event()

    def never_walk():
        pytest.fail("Paused crawler must not touch the filesystem iterator")
        yield "unused"

    original_wait = stop.wait

    def wait(_):
        stop.set()
        return original_wait(0)

    stop.wait = wait
    assert list(admitted_files(iter(never_walk()), lambda: 0, pressure, stop, 0, 10, pause_when_off=True)) == []


def test_deletion_admission_and_shutdown_do_not_flush_pending_batch(tmp_path):
    import logging

    from processes.deleter import delete_file_batch

    event, shutdown = multiprocessing.Event(), multiprocessing.Event()
    file = tmp_path / "keep.bin"
    file.write_bytes(b"keep")
    queue = multiprocessing.Queue()
    args = ([str(file)], False, logging.getLogger("test"), 1, 0, 0, 0, queue)
    assert delete_file_batch(*args, admission_event=event, shutdown_event=shutdown) == (0, 0, 0)
    event.set()
    shutdown.set()
    assert delete_file_batch(*args, admission_event=event, shutdown_event=shutdown) == (0, 0, 0)
    assert file.read_bytes() == b"keep"
    queue.close()
    queue.join_thread()


def _activator_with_fixed_test_pressure(config, pressure, stop, results):
    import processes.activator as module
    from utils.system import DiskUsage

    module.get_disk_usage_from_statvfs = lambda _: DiskUsage(
        total_bytes=1000,
        used_bytes=900,
        available_bytes=100,
        usage_percent=90,
    )
    module.activator_process(9, "/unused", 85, 70, 0.05, pressure, results, stop, config)


def test_real_http_activity_controls_local_crawler_and_deleter(tmp_path):
    """Real localhost HTTP + spawned processes; fixed disk pressure, no fake latency claim."""
    import contextlib
    import http.server
    import os
    import threading
    import time
    from queue import Empty

    from processes.crawler import crawler_process
    from processes.deleter import deleter_process

    state = {"running": 1, "samples": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            state["samples"] += 1
            payload = f"vllm:num_requests_running {state['running']}\nvllm:num_requests_waiting 0\n".encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "cache"
    bucket = root / "model_0123456789ab_r0" / "abc" / "de_g0"
    bucket.mkdir(parents=True)

    def old_files(start):
        paths = [bucket / f"{i:016x}.bin" for i in range(start, start + 8)]
        for p in paths:
            p.write_bytes(b"old")
            os.utime(p, (time.time() - 1000, time.time() - 1000))
        return paths

    assert fetch_activity([ActivitySource(f"http://127.0.0.1:{server.server_port}/metrics")], 1).idle is False
    old = old_files(0)
    hot = bucket / "fresh.bin"
    hot.write_bytes(b"fresh")
    ctx = multiprocessing.get_context("spawn")
    pressure, stop, results, candidates = ctx.Event(), ctx.Event(), ctx.Queue(), ctx.Queue(32)
    cfg = dict(
        log_level="WARNING",
        file_queue_min_size=8,
        file_queue_maxsize=32,
        file_access_time_threshold_minutes=1,
        deletion_batch_size=4,
        dry_run=False,
        cleanup_policy=CleanupSettings(
            (ActivitySource(f"http://127.0.0.1:{server.server_port}/metrics"),),
            1.0,
            quiet_seconds=0.05,
        ).to_dict(),
    )
    processes = [
        ctx.Process(target=_activator_with_fixed_test_pressure, args=(cfg, pressure, stop, results)),
        ctx.Process(target=crawler_process, args=(1, (0, 15), root, cfg, pressure, candidates, results, stop)),
        ctx.Process(target=deleter_process, args=(10, root, cfg, pressure, candidates, results, stop)),
    ]

    def wait_for(predicate, timeout=8):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            assert all(p.is_alive() for p in processes)
            if predicate():
                return
            with contextlib.suppress(Empty):
                results.get(timeout=0.02)
            time.sleep(0.02)
        pytest.fail(
            f"Local policy integration deadline; admitted={pressure.is_set()}; remaining={list(bucket.iterdir())}"
        )

    try:
        for process in processes:
            process.start()
        wait_for(lambda: state["samples"] >= 2)
        assert not pressure.is_set() and all(p.exists() for p in old)
        for p in old:
            os.utime(p, (time.time() - 1000, time.time() - 1000))
        state["running"] = 0
        wait_for(lambda: all(not p.exists() for p in old))
        state["running"] = 1
        wait_for(lambda: not pressure.is_set())
        again = old_files(32)
        time.sleep(0.25)
        assert all(p.exists() for p in again)
        for p in again:
            os.utime(p, (time.time() - 1000, time.time() - 1000))
        state["running"] = 0
        wait_for(lambda: all(not p.exists() for p in again))
        assert hot.read_bytes() == b"fresh"
    finally:
        stop.set()
        pressure.clear()
        end = time.monotonic() + 8
        while any(p.is_alive() for p in processes) and time.monotonic() < end:
            for q in (results, candidates):
                with contextlib.suppress(Empty):
                    q.get_nowait()
            for process in processes:
                process.join(0.02)
        server.shutdown()
        server.server_close()
        thread.join(2)
        codes = [p.exitcode for p in processes]
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(2)
        for q in (results, candidates):
            q.close()
            q.join_thread()
    assert codes == [0, 0, 0]


def test_folder_cleaner_does_not_consume_work_while_paused(tmp_path):
    from unittest.mock import Mock

    from processes.folder_cleaner import folder_cleaner_process

    pressure, stop = multiprocessing.Event(), multiprocessing.Event()
    queue = Mock()
    queue.get.side_effect = AssertionError("paused folder cleaner consumed work")
    stop.wait = lambda _: stop.set()
    folder_cleaner_process(
        1,
        queue,
        Mock(),
        stop,
        {"pvc_mount_path": str(tmp_path), "cache_directory": "cache", "cleanup_policy": True},
        pressure,
    )
    queue.get.assert_not_called()


def test_stuck_reader_does_not_block_policy_or_reuse_late_idle():
    import threading
    import time

    from activity import Activity, ActivitySampler

    release = threading.Event()
    calls = []

    def fetch(sources, timeout):
        calls.append(True)
        if len(calls) == 1:
            release.wait(2)
            return Activity(True, "late idle")
        return Activity(False, "currently busy")

    reader = ActivitySampler((), 0.05, fetch)
    try:
        start = time.monotonic()
        assert reader.sample().idle is None
        assert time.monotonic() - start < 0.5
        for _ in range(10):
            assert reader.sample().idle is None
        assert len(calls) == 1
        release.set()
        end = time.monotonic() + 1
        while reader._responses.empty() and time.monotonic() < end:
            time.sleep(0.005)
        assert reader.sample().idle is False
    finally:
        release.set()
        reader.close()
        reader._thread.join(1)
    assert reader.sample().idle is None


def test_below_trigger_avoids_activity_endpoint_polling(monkeypatch):
    from unittest.mock import Mock

    import processes.activator as module
    from utils.system import DiskUsage

    reader = Mock()
    monkeypatch.setattr("activity.ActivitySampler", lambda *args: reader)
    monkeypatch.setattr(module, "get_disk_usage_from_statvfs", lambda _: DiskUsage(100, 50, 50, 50))
    stop, pressure = multiprocessing.Event(), multiprocessing.Event()
    stop.wait = lambda _: stop.set()
    module.activator_process(
        1,
        "unused",
        85,
        70,
        0.05,
        pressure,
        Mock(),
        stop,
        {"cleanup_policy": settings().to_dict(), "log_level": "WARNING"},
    )
    reader.sample.assert_not_called()
    reader.close.assert_called_once()
    assert not pressure.is_set()
