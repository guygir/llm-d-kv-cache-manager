"""Opt-in cleanup scheduling. Resource limits are enforced by the container runtime."""

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit


def _number(value, name, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return value


@dataclass(frozen=True)
class ActivitySource:
    url: str
    running: str = "vllm:num_requests_running"
    waiting: str = "vllm:num_requests_waiting"
    busy_metrics: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.url, str) or not isinstance(self.busy_metrics, (tuple, list)):
            raise ValueError("Activity URL must be text and busy_metrics must be a list")
        object.__setattr__(self, "busy_metrics", tuple(self.busy_metrics))
        parsed = urlsplit(self.url)
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.fragment
        ):
            raise ValueError("Activity URLs must be HTTP(S) without credentials or fragments")
        import re

        if any(not isinstance(n, str) or not re.fullmatch(r"[a-zA-Z_:][a-zA-Z0-9_:]*", n) for n in self.metrics):
            raise ValueError("Activity metric names must be valid Prometheus names")
        if len(set(self.metrics)) != len(self.metrics):
            raise ValueError("Activity metric names must be distinct")

    @property
    def metrics(self):
        return (self.running, self.waiting, *self.busy_metrics)


@dataclass(frozen=True)
class CleanupSettings:
    sources: tuple[ActivitySource, ...]
    cpu_budget_cores: float
    emergency_threshold: float = 95.0
    quiet_seconds: float = 2.0
    busy_mode: str = "continuous"
    window_on_seconds: float = 1.0
    window_off_seconds: float = 1.0
    poll_timeout_seconds: float = 1.0
    cpu_ids: tuple[int, ...] = ()

    def validate(self, cleanup_threshold, target_threshold):
        numeric = (
            self.cpu_budget_cores,
            self.emergency_threshold,
            self.quiet_seconds,
            self.window_on_seconds,
            self.window_off_seconds,
            self.poll_timeout_seconds,
        )
        if any(type(v) not in (int, float) for v in numeric):
            raise ValueError("Cleanup limits and intervals must be numeric")
        if not self.sources or len(self.sources) > 16:
            raise ValueError("Load-aware cleanup needs 1–16 activity sources")
        for source in self.sources:
            source.__post_init__()
        _number(self.cpu_budget_cores, "CPU budget")
        _number(self.quiet_seconds, "Quiet interval", zero=True)
        for name in ("window_on_seconds", "window_off_seconds", "poll_timeout_seconds"):
            _number(getattr(self, name), name)
        if self.busy_mode not in ("continuous", "windowed"):
            raise ValueError("CLEANUP_BUSY_MODE must be continuous or windowed")
        if not (0 <= target_threshold < cleanup_threshold < self.emergency_threshold <= 100):
            raise ValueError("Require 0 <= target < cleanup < emergency <= 100")
        if len(set(self.cpu_ids)) != len(self.cpu_ids) or any(type(i) is not int or i < 0 for i in self.cpu_ids):
            raise ValueError("CPU IDs must be distinct nonnegative integers")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["sources"] = tuple(ActivitySource(**s) for s in value["sources"])
        value["cpu_ids"] = tuple(value.get("cpu_ids", ()))
        return cls(**value)

    @classmethod
    def from_env(cls, cleanup_threshold, target_threshold):
        enabled = os.getenv("CLEANUP_LOAD_AWARE", "false").lower()
        if enabled not in ("true", "false"):
            raise ValueError("CLEANUP_LOAD_AWARE must be true or false")
        if enabled == "false":
            return None
        raw = json.loads(os.environ.get("CLEANUP_ACTIVITY_SOURCES", "[]"))
        if not isinstance(raw, list):
            raise ValueError("CLEANUP_ACTIVITY_SOURCES must be a JSON list")
        settings = cls(
            sources=tuple(ActivitySource(**s) for s in raw),
            cpu_budget_cores=_number(os.environ.get("CLEANUP_CPU_BUDGET_CORES", "nan"), "CPU budget"),
            emergency_threshold=float(os.getenv("CLEANUP_EMERGENCY_THRESHOLD", "95")),
            quiet_seconds=float(os.getenv("CLEANUP_QUIET_SECONDS", "2")),
            busy_mode=os.getenv("CLEANUP_BUSY_MODE", "continuous"),
            window_on_seconds=float(os.getenv("CLEANUP_WINDOW_ON_SECONDS", "1")),
            window_off_seconds=float(os.getenv("CLEANUP_WINDOW_OFF_SECONDS", "1")),
            poll_timeout_seconds=float(os.getenv("CLEANUP_ACTIVITY_TIMEOUT_SECONDS", "1")),
            cpu_ids=tuple(int(n.strip()) for n in os.getenv("CLEANUP_CPU_IDS", "").split(",") if n.strip()),
        )
        return settings.validate(cleanup_threshold, target_threshold)


def validate_cpu_budget(settings, root=Path("/sys/fs/cgroup")):
    """Verify the runtime's CPU limit; never mutate host cgroups or CPU ownership."""
    try:
        if (root / "cpu.max").exists():
            quota, period = (root / "cpu.max").read_text().split()
            actual = None if quota == "max" else int(quota) / int(period)
        else:
            quota = int((root / "cpu/cpu.cfs_quota_us").read_text())
            period = int((root / "cpu/cpu.cfs_period_us").read_text())
            actual = None if quota < 0 else quota / period
    except (OSError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("Cannot verify cleanup CPU limit; configure a container CPU limit") from exc
    if actual is None or not math.isfinite(actual) or not 0 < actual <= settings.cpu_budget_cores:
        raise ValueError("Container CPU limit must be finite, positive and <= CLEANUP_CPU_BUDGET_CORES")
    if settings.cpu_ids:
        if not hasattr(os, "sched_getaffinity") or not set(settings.cpu_ids) <= os.sched_getaffinity(0):
            raise ValueError("CLEANUP_CPU_IDS must be a subset of the permitted CPU mask")
        os.sched_setaffinity(0, settings.cpu_ids)
    return actual


@dataclass(frozen=True)
class CleanupDecision:
    mode: str
    enabled: bool
    reason: str


class CleanupController:
    """Disk hysteresis plus quiet-period admission; unknown activity means busy."""

    def __init__(self, settings, cleanup_threshold, target_threshold):
        self.settings = settings.validate(cleanup_threshold, target_threshold)
        self.cleanup_threshold = cleanup_threshold
        self.target_threshold = target_threshold
        self.requested = False
        self.urgent = False
        self.urgent_since = None
        self.quiet_since = None
        self.last_time = None

    def decide(self, usage_percent, idle, now):
        if self.last_time is not None and now < self.last_time:
            self.quiet_since = None
            self.urgent_since = now if self.urgent else None
        self.last_time = now
        if idle is True:
            if self.quiet_since is None:
                self.quiet_since = now
        else:
            self.quiet_since = None
        if usage_percent is None or not math.isfinite(usage_percent) or not 0 <= usage_percent <= 100:
            return CleanupDecision("paused", False, "disk usage unavailable")
        if usage_percent <= self.target_threshold:
            self.requested = self.urgent = False
            self.urgent_since = None
        elif usage_percent >= self.cleanup_threshold:
            self.requested = True
        if not self.requested:
            return CleanupDecision("off", False, "below cleanup trigger")
        if usage_percent >= self.settings.emergency_threshold:
            if not self.urgent:
                self.urgent_since = now
            self.urgent = True
        elif usage_percent <= self.cleanup_threshold:
            self.urgent = False
            self.urgent_since = None
        quiet = self.quiet_since is not None and now - self.quiet_since >= self.settings.quiet_seconds
        if quiet:
            return CleanupDecision("quiet-drain", True, "all sources quiet")
        if self.urgent:
            enabled = True
            if self.settings.busy_mode == "windowed":
                position = (now - self.urgent_since) % (
                    self.settings.window_on_seconds + self.settings.window_off_seconds
                )
                enabled = position < self.settings.window_on_seconds
            return CleanupDecision("urgent-" + self.settings.busy_mode, enabled, "urgent disk pressure")
        return CleanupDecision("deferred", False, "serving busy, unknown, or quiet grace not reached")
