"""Read configured request gauges directly; failed/incomplete sources never imply idle."""

import math
import re
import time
import urllib.request
from dataclasses import dataclass

_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([^\s]+)(?:\s+\d+)?(?:\s+#.*)?$")
MAX_RESPONSE_BYTES = 1 << 20


@dataclass(frozen=True)
class Activity:
    idle: bool | None
    reason: str


def parse_gauges(text, names):
    totals = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(line.strip())
        if match and match[1] in names:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid activity gauge")
            totals[match[1]] = totals.get(match[1], 0.0) + value
    if set(totals) != set(names):
        raise ValueError("Required activity gauge missing")
    return totals


def fetch_activity(sources, timeout_seconds, *, opener=urllib.request.urlopen, clock=time.monotonic):
    deadline = clock() + timeout_seconds
    busy = False
    for source in sources:
        try:
            remaining = deadline - clock()
            if remaining <= 0:
                raise TimeoutError("Activity sampling deadline")
            with opener(source.url, timeout=min(0.2, remaining)) as response:
                chunks = []
                length = 0
                while True:
                    if clock() >= deadline:
                        raise TimeoutError("Activity sampling deadline")
                    block = response.read1(min(16384, MAX_RESPONSE_BYTES + 1 - length))
                    if not block:
                        break
                    chunks.append(block)
                    length += len(block)
                    if length > MAX_RESPONSE_BYTES:
                        raise ValueError("Activity response too large")
                gauges = parse_gauges(b"".join(chunks).decode("utf-8"), source.metrics)
                busy |= any(v > 0 for v in gauges.values())
        except (OSError, ValueError, TimeoutError) as exc:
            return Activity(None, type(exc).__name__)
    return Activity(not busy if sources else None, "sampled" if sources else "no sources")


class ActivitySampler:
    """One daemon reader bounds DNS/socket stalls without accumulating work.

    A response arriving after its caller's deadline is discarded. A stuck
    reader makes activity unknown; it cannot stall the policy loop or cause an
    unbounded succession of abandoned threads. The reader performs no deletion.
    """

    def __init__(self, sources, timeout_seconds, fetcher=fetch_activity):
        import queue
        import threading

        self.sources = sources
        self.timeout = timeout_seconds
        self.fetcher = fetcher
        self._requests = queue.Queue(1)
        self._responses = queue.Queue(1)
        self._pending = None
        self._next = 0
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="cleanup-activity", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            token = self._requests.get()
            if token is None or self._closed:
                return
            try:
                response = self.fetcher(self.sources, self.timeout)
            except Exception as exc:
                response = Activity(None, type(exc).__name__)
            if self._closed:
                return
            self._responses.put((token, response))

    def sample(self):
        from queue import Empty

        if self._closed:
            return Activity(None, "sampler closed")
        if self._pending is not None:
            try:
                self._responses.get_nowait()  # discard an expired response, even if it said idle
            except Empty:
                return Activity(None, "previous sample still pending")
            self._pending = None
        self._next += 1
        self._pending = self._next
        self._requests.put_nowait(self._pending)
        try:
            token, response = self._responses.get(timeout=self.timeout)
        except Empty:
            return Activity(None, "sampling deadline")
        if token != self._pending:
            raise RuntimeError("Activity response token mismatch")
        self._pending = None
        return response

    def close(self):
        import contextlib
        from queue import Full

        self._closed = True
        # A busy reader is a daemon; it performs no policy/deletion work.
        with contextlib.suppress(Full):
            self._requests.put_nowait(None)
