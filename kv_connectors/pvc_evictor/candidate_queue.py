"""Bounded candidate-only age bands; filesystem recency remains authoritative.

The manager stores metadata only. Dequeue revalidation runs in the consumer,
without holding the shared index lock across filesystem calls. This is an
approximate order over discovered candidates, not a global filesystem LRU.
"""

import math
import os
import stat
import threading
import time
from bisect import bisect_right
from collections import OrderedDict, deque
from dataclasses import dataclass
from multiprocessing.managers import BaseManager
from queue import Empty, Full


def validate_bands(values):
    values = tuple(float(v) for v in values)
    if not 1 <= len(values) <= 32 or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Age bands require 1–32 finite positive boundaries in seconds")
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError("Age band boundaries must be strictly increasing")
    return values


@dataclass(frozen=True)
class Candidate:
    path: str
    atime_ns: int
    device: int
    inode: int
    mtime_ns: int
    size: int

    @classmethod
    def from_stat(cls, path, info):
        return cls(
            os.path.abspath(os.fspath(path)), info.st_atime_ns, info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size
        )

    @property
    def generation(self):
        return self.device, self.inode, self.mtime_ns, self.size


class AgeBandStore:
    """One entry per path; bounded storage with incremental age reclassification."""

    def __init__(self, maxsize, bands, aging_budget=32):
        if maxsize <= 0 or aging_budget <= 0:
            raise ValueError("Candidate capacity and aging budget must be positive")
        self.maxsize = maxsize
        self.bands = validate_bands(bands)
        self.aging_budget = aging_budget
        self._buckets = [OrderedDict() for _ in range(len(self.bands) + 1)]
        self._index = {}
        self._condition = threading.Condition()
        self._cursor = 0
        self._closed = False
        self._counts = dict(
            inserted=0,
            deduplicated=0,
            aged=0,
            popped=0,
            hot_dropped=0,
            generation_dropped=0,
            missing_dropped=0,
            reclassified=0,
            requeue_full=0,
        )

    def band(self, atime_ns, now_ns):
        return bisect_right(self.bands, max(0, now_ns - atime_ns) / 1e9)

    def _remove(self, path):
        candidate, band = self._index.pop(path)
        del self._buckets[band][path]
        return candidate

    def put(self, candidate, block=True, timeout=None):
        if not isinstance(candidate, Candidate):
            raise TypeError("AgeBandStore accepts Candidate metadata only")
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while len(self._index) >= self.maxsize and candidate.path not in self._index:
                if self._closed:
                    raise ValueError("Candidate queue is closed")
                if not block:
                    raise Full
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise Full
                self._condition.wait(remaining)
            if self._closed:
                raise ValueError("Candidate queue is closed")
            old = self._index.get(candidate.path)
            if old:
                if old[0].generation == candidate.generation and old[0].atime_ns > candidate.atime_ns:
                    return False  # delayed discovery cannot undo a newer known access
                self._remove(candidate.path)
                self._counts["deduplicated"] += 1
            else:
                self._counts["inserted"] += 1
            band = self.band(candidate.atime_ns, time.time_ns())
            self._buckets[band][candidate.path] = candidate
            self._index[candidate.path] = (candidate, band)
            self._condition.notify_all()
            return True

    def age_step(self, now_ns, budget=None):
        """Bounded rotating scan; no full index walk or population rebalancing."""
        with self._condition:
            for _ in range(self.aging_budget if budget is None else budget):
                band = self._cursor % len(self._buckets)
                self._cursor += 1
                bucket = self._buckets[band]
                if not bucket:
                    continue
                path, candidate = next(iter(bucket.items()))
                target = self.band(candidate.atime_ns, now_ns)
                if target != band:
                    self._remove(path)
                    self._buckets[target][path] = candidate
                    self._index[path] = (candidate, target)
                    self._counts["aged"] += 1
                else:
                    bucket.move_to_end(path)

    def get(self, block=True, timeout=None):
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._index:
                if self._closed:
                    raise EOFError("Candidate queue closed")
                if not block:
                    raise Empty
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise Empty
                self._condition.wait(remaining)
            self.age_step(time.time_ns())
            return self._pop_one()

    def _pop_one(self):
        for bucket in reversed(self._buckets):
            if bucket:
                path = next(iter(bucket))
                candidate = self._remove(path)
                self._counts["popped"] += 1
                self._condition.notify_all()
                return candidate
        raise Empty

    def get_many(self, limit=32, block=True, timeout=None):
        if not 1 <= limit <= 32:
            raise ValueError("Prefetch limit must be in [1, 32]")
        first = self.get(block, timeout)
        with self._condition:
            result = [first]
            while self._index and len(result) < limit:
                result.append(self._pop_one())
            return result

    def qsize(self):
        with self._condition:
            return len(self._index)

    def empty(self):
        return self.qsize() == 0

    def full(self):
        return self.qsize() >= self.maxsize

    def discard(self, path):
        with self._condition:
            if path in self._index:
                self._remove(path)
                self._condition.notify_all()
                return True
            return False

    def record(self, name):
        with self._condition:
            self._counts[name] += 1

    def stats(self):
        with self._condition:
            return dict(
                self._counts,
                occupancy=len(self._index),
                capacity=self.maxsize,
                band_sizes=[len(b) for b in self._buckets],
            )

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class CandidateManager(BaseManager):
    pass


CandidateManager.register(
    "AgeBandStore",
    AgeBandStore,
    exposed=("put", "get_many", "qsize", "empty", "full", "record", "stats", "close"),
)


class CandidateQueue:
    """Multiprocessing queue facade; returns only current, still-cold pathnames."""

    def __init__(self, backend, bands, protection_seconds):
        self.backend = backend
        self.bands = validate_bands(bands)
        if not math.isfinite(protection_seconds) or protection_seconds < 0:
            raise ValueError("Hot protection must be finite and nonnegative")
        self.protection_seconds = protection_seconds
        self._ready = deque()

    def put(self, item, block=True, timeout=None):
        if not isinstance(item, Candidate):
            info = os.lstat(item)
            if not stat.S_ISREG(info.st_mode):
                return False
            item = Candidate.from_stat(item, info)
        return self.backend.put(item, block, timeout)

    def put_nowait(self, item):
        return self.put(item, False)

    def get(self, block=True, timeout=None):
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        attempts = 0
        while True:
            if attempts and (not block and attempts >= 32 or deadline is not None and time.monotonic() >= deadline):
                raise Empty
            remaining = None if deadline is None else max(0, deadline - time.monotonic())
            if not self._ready:
                self._ready.extend(self.backend.get_many(32, block, remaining))
            candidate = self._ready.popleft()
            attempts += 1
            try:
                info = os.lstat(candidate.path)
            except OSError:
                self.backend.record("missing_dropped")
                continue
            current = Candidate.from_stat(candidate.path, info)
            if not stat.S_ISREG(info.st_mode) or current.generation != candidate.generation:
                self.backend.record("generation_dropped")
                continue
            now = time.time_ns()
            last_access = max(current.atime_ns, candidate.atime_ns)
            if (now - last_access) / 1e9 < self.protection_seconds:
                self.backend.record("hot_dropped")
                continue
            if bisect_right(self.bands, max(0, now - last_access) / 1e9) != bisect_right(
                self.bands, max(0, now - candidate.atime_ns) / 1e9
            ):
                updated = Candidate(current.path, last_access, *current.generation)
                try:
                    self.backend.put(updated, False)
                    self.backend.record("reclassified")
                except Full:
                    self.backend.record("requeue_full")  # safe to rediscover; never evict another entry for this
                continue
            return candidate.path

    def get_nowait(self):
        return self.get(False)

    def qsize(self):
        return self.backend.qsize() + len(self._ready)

    def empty(self):
        return not self._ready and self.backend.empty()

    def full(self):
        return self.backend.full()

    def stats(self):
        return self.backend.stats()

    def close(self):
        self.backend.close()
