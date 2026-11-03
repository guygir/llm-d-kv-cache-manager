# Copyright 2026 The llm-d Authors.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem-only fixed and idle lifetime protocol; no vLLM or Torch imports.

Versioned namespaces retain fixed-publication and renewable-idle semantics.
Readers pin the inode with an open descriptor; a same-container native worker
reads through /proc/<scheduler-pid>/fd/<fd>. Unlink/replace cannot break an
accepted load. Writers and deleters serialize pathname changes with stable
bucket POSIX record locks. Locks are never removed by directory cleanup.
"""

import errno
import fcntl
import os
import sys
import tempfile
import time
from pathlib import Path

PREFIX = ".llmd-ttl-v1-"
IDLE_PREFIX = ".llmd-ttl-v2-idle-"
SHARED_IDLE_NAME = ".llmd-ttl-v3-idle"
DEADLINE_XATTR = "user.llmd.ttl_deadline_ns"
NANOSECONDS = 1_000_000_000


def validate_seconds(value):
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= 2_147_483_647:
        raise ValueError("storage_ttl_seconds must be an integer in [1, 2147483647]")
    if sys.platform != "linux" or not Path("/proc/self/fd").is_dir():
        raise ValueError("storage TTL requires Linux procfs descriptor access")
    return value


def policy_prefix(value="idle"):
    if value not in ("fixed", "idle"):
        raise ValueError("storage_ttl_policy must be fixed or idle")
    return IDLE_PREFIX if value == "idle" else PREFIX


def policy_namespace(seconds, policy="idle"):
    prefix = policy_prefix(policy)
    return SHARED_IDLE_NAME if prefix == IDLE_PREFIX else prefix + str(seconds)


def validate_shared_idle_root(root):
    """Fail startup if this filesystem cannot persist the inode deadline."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryFile(dir=root) as f:
            os.setxattr(f.fileno(), DEADLINE_XATTR, b"1")
            if os.getxattr(f.fileno(), DEADLINE_XATTR) != b"1":
                raise OSError("Deadline xattr readback mismatch")
    except OSError as error:
        raise ValueError("Idle storage TTL requires writable inode xattrs") from error


def shared_idle_deadline(path_or_fd):
    try:
        if isinstance(path_or_fd, int):
            value = os.getxattr(path_or_fd, DEADLINE_XATTR)
        else:
            value = os.getxattr(path_or_fd, DEADLINE_XATTR, follow_symlinks=False)
    except OSError as error:
        if error.errno in (errno.ENODATA, errno.ENOENT, errno.ERANGE):
            return None
        raise
    if not value or not value.isdigit() or len(value) > 19:
        return None
    deadline = int(value)
    return deadline if 0 < deadline <= 2**63 - 1 else None


def expired(stat, policy, now_ns=None, *, path=None):
    now_ns = time.time_ns() if now_ns is None else now_ns
    if policy[0].name == SHARED_IDLE_NAME:
        deadline = shared_idle_deadline(path) if path is not None else None
        return deadline is not None and now_ns >= deadline
    origin = (
        stat.st_atime_ns if policy[0].name.startswith(IDLE_PREFIX) else stat.st_mtime_ns
    )
    return now_ns >= origin + policy[1] * NANOSECONDS


def policy_for_path(path):
    path = Path(path)
    for parent in path.parents:
        if parent.name == SHARED_IDLE_NAME:
            return parent, 0  # Duration is not part of shared-file identity.
        prefix = next(
            (p for p in (PREFIX, IDLE_PREFIX) if parent.name.startswith(p)), None
        )
        if prefix is not None:
            suffix = parent.name[len(prefix) :]
            if not suffix.isdecimal() or not 1 <= int(suffix) <= 2_147_483_647:
                raise ValueError("Invalid reserved TTL namespace")
            return parent, int(suffix)
    return None


def lock_path(path):
    policy = policy_for_path(path)
    if policy is None:
        raise ValueError("Not a TTL namespace")
    folder = policy[0] / ".locks"
    folder.mkdir(exist_ok=True)
    bucket = Path(path).stem[:3]
    if len(bucket) != 3 or any(c not in "0123456789abcdef" for c in bucket):
        raise ValueError("Invalid TTL block filename")
    return folder / (bucket + ".lock")


def try_delete(path, *, pressure, idle_seconds, dry_run=False, defer_busy=False):
    """Return deleted stat, or None if not eligible/busy/missing.

    Deadline is rechecked under the same exclusive lock as publication. An
    admitted reader's FD keeps old data valid after this pathname is removed.
    """
    policy = policy_for_path(path)
    if policy is None:
        return None
    descriptor = os.open(lock_path(path), os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    file_fd = None
    try:
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if defer_busy:
                raise
            return None
        try:
            if policy[0].name == SHARED_IDLE_NAME:
                file_fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
                stat = os.fstat(file_fd)
            else:
                stat = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return None
        now = time.time_ns()
        if not expired(stat, policy, now, path=file_fd) and not (
            pressure and now - stat.st_atime_ns >= idle_seconds * NANOSECONDS
        ):
            return None
        if not dry_run:
            os.unlink(path)
        return stat
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(descriptor)


class InodePins:
    """Request lifetime pins, including speculative lookups and late completions."""

    def __init__(self, seconds, *, idle=False, shared=False, idle_opener=None):
        self.seconds = validate_seconds(seconds)
        self.idle = idle
        self.shared = shared
        self.idle_opener = idle_opener
        self.requests = {}
        self.counters = dict(opened=0, expired=0, misses=0, busy=0)

    def lookup(self, request_id, key, path):
        state = self.requests.setdefault(
            request_id, dict(current={}, fds=[], loads=0, finished=False, busy_until={})
        )
        old = state["current"].get(key)
        now = time.time_ns()
        if old is not None and (self.idle or now < old[1] + self.seconds * NANOSECONDS):
            return True
        try:
            if self.idle:
                retry_at = state["busy_until"].get(key)
                if retry_at is not None and time.monotonic() < retry_at:
                    return None
                if self.idle_opener is None:
                    from storage_offload import try_open_idle_hit

                    self.idle_opener = try_open_idle_hit
                fd = (
                    self.idle_opener(str(path), self.seconds)
                    if self.shared
                    else self.idle_opener(str(path))
                )
                if fd == -2:
                    self.counters["busy"] += 1
                    state["busy_until"][key] = time.monotonic() + 0.01
                    return None  # Pending; avoid filesystem probes on every poll.
                state["busy_until"].pop(key, None)
                if fd == -1:
                    self.counters["misses"] += 1
                    return False
            else:
                fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except FileNotFoundError:
            self.counters["misses"] += 1
            return False
        try:
            stat = os.fstat(fd)
            if (
                not self.idle
                and time.time_ns() >= stat.st_mtime_ns + self.seconds * NANOSECONDS
            ):
                self.counters["expired"] += 1
                os.close(fd)
                return False
            state["current"][key] = (fd, stat.st_mtime_ns)
            state["fds"].append(fd)
            self.counters["opened"] += 1
            return True
        except BaseException:
            os.close(fd)
            raise

    def prepare_load(self, request_id, key_paths):
        state = self.requests[request_id]
        if state["finished"]:
            raise RuntimeError("Load submitted after request finished")
        result = {}
        for key, path in key_paths:
            fd = state["current"][key][0]
            result[path] = f"/proc/{os.getpid()}/fd/{fd}"
        state["loads"] += 1
        return result

    def complete_load(self, request_id):
        state = self.requests[request_id]
        if state["loads"] <= 0:
            raise RuntimeError("Unexpected TTL load completion")
        state["loads"] -= 1
        if state["finished"] and state["loads"] == 0:
            self._close(request_id)

    def finish(self, request_id):
        state = self.requests.get(request_id)
        if state is not None:
            state["finished"] = True
            if state["loads"] == 0:
                self._close(request_id)

    def _close(self, request_id):
        state = self.requests.pop(request_id)
        for fd in state["fds"]:
            os.close(fd)

    def close(self):
        for request_id in list(self.requests):
            self._close(request_id)
