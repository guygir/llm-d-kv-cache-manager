# Storage TTL retention (experimental implementation)

TTL is disabled unless `storage_ttl_seconds` is configured. Once configured,
`storage_ttl_policy` selects one of two policies:

- `"idle"` (default): TTL is cleanup eligibility after the last accepted storage
  access. Matching bytes still present after the deadline remain reusable; an
  accepted hit renews retention without rewriting bytes or changing inode/mtime.
- `"fixed"`: TTL is a maximum publication lifetime. Reads and duplicate stores
  do not extend it. New lookups after the deadline miss and may recompute.

Both policies are independent of pressure eviction. Already accepted reads may
finish after unlink. Removing a pathname and returning allocated space are
asynchronous; open readers can delay frees. No configured TTL means legacy
non-TTL behavior, regardless of this new policy default.

The initial implementation supports Linux procfs, the native POSIX CPU-staging
backend and one worker rank. GDS, object storage and multi-rank TTL configurations
are rejected. Native correctness, HTTP transport and concurrent serving/cleanup have been
validated on GPFS. These finite experiments do not establish a universal latency
budget or production-traffic benefit. This draft does not enable TTL by default.

Configure the existing OffloadingConnector with:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "SharedStorageOffloadingSpec",
    "spec_module_path": "llmd_fs_backend.spec",
    "shared_storage_path": "/kv-cache",
    "track_file_access": true,
    "storage_ttl_seconds": 3600,
    "storage_ttl_policy": "idle"
  }
}
```

Idle mode uses `.llmd-ttl-v2-idle-3600`; explicit fixed mode uses
`.llmd-ttl-v1-3600`. Policy and duration select separate namespaces. Existing v1
files retain fixed semantics and are not reinterpreted as idle files. Switching
policy starts a separate cache population. Idle deadlines use atime + TTL;
fixed deadlines use the publication mtime + TTL. Access tracking preserves mtime.

A request may confirm the namespace policy through vLLM's existing transport:

```json
{"kv_transfer_params": {"llmd_storage_ttl_seconds": 3600}}
```

This first version does not merge arbitrary per-request lifetimes for a shared
prefix. A conflicting or invalid hint makes that request decline storage reuse
and persistence, with a warning; it does not throw inside the inference
scheduler. GPU cache reuse and inference continue according to vLLM's normal
behavior. Frontend HTTP validation is outside this backend feature. With no
configured TTL and no TTL hint, existing behavior remains the default.

Run the updated PVC evictor against the parent storage root with
`ENABLE_STORAGE_TTL=true`. It discovers expired namespace files and reclaims them
even when pressure eviction is off. Unexpired files are not removed below the
pressure threshold. Pressure removal of TTL files uses the same mutation
coordination. All writers/deleters touching these policy namespaces must use the
updated protocol; do not run an old uncoordinated deleter against them.

Readers pin an inode before accepting a hit. Its same-container native worker
uses `/proc/<scheduler-pid>/fd/<fd>` while the scheduler retains that descriptor.
Request completion releases speculative pins; accepted reads retain their pins
until completion, including late callbacks. This keeps data readable across
unlink or replacement without shared read locks across a distributed transfer.
Cache reset is refused while accepted TTL loads are in flight because this vLLM
API discards old callbacks before acknowledging the worker flush; drain loads
before reset.

Native publishers and deleters serialize pathname mutations with stable bucket
POSIX record locks. Native writers additionally serialize threads locally, since
record locks are process-owned. Lock identities are not removed by directory
cleanup. The GPFS test mount passed cross-node POSIX exclusion/release; its
`flock` probe did not. Other deployments need equivalent locking semantics and
shared writable lock-file permissions. Both local and GPFS native/end-to-end gates were exercised. Revalidate locking
and descriptor permissions for a different deployment.

Storage routing events remain best-effort hints, with the backend lookup
providing authoritative expiry. This does not invalidate GPU/CPU-resident copies
or promise globally ordered events across storage generations.

## Optional per-request lifetime allowlist

One server-selected lifetime is the default. To let a caller choose
among explicit server policies, configure the existing spec with both a default
and an allowlist:

```json
{
  "spec_name": "SharedStorageOffloadingSpec",
  "spec_module_path": "llmd_fs_backend.spec",
  "shared_storage_path": "/cache",
  "storage_ttl_seconds": 60,
  "storage_ttl_allowlist": [30, 60, 90]
}
```

A caller supplies `kv_transfer_params.llmd_storage_ttl_seconds`. Missing hints use
the configured default. A supplied value outside the allowlist (including null,
booleans or strings) declines storage reuse and persistence for that request;
serving can still compute normally. Lifetimes remain positive integer seconds.
The default must belong to the allowlist. This option supports one worker rank,
native POSIX CPU staging, and disabled storage events. Policy-blind events cannot
advertise a lifetime-specific namespace safely, so enabling them is rejected.

`spec.py` copies trusted `FileMapper` instances into separate `.llmd-ttl-v1-N`
namespaces. `RequestStorageTTLManager` binds one route generation to the
scheduler-owned `ReqContext`; it delegates to the selected fixed-TTL manager.
Internal `SharedStorageLoadStoreSpec.source_paths` aliases carry store targets
or accepted read FDs through the existing worker transfer path. Clients cannot
supply these paths. Neither a late callback nor a changed request-parameter dict
can select another request's namespace. The route survives request finish until
all accepted load and store completions drain. A stale context cannot complete
work for a later request that reuses its ID.

Identical prefixes under two lifetimes have separate physical copies. A shorter
owner cannot shorten a longer copy; reading or an unexpired duplicate store does
not renew its publication deadline. This is isolation with duplication, not a
shared-file maximum-deadline ownership algorithm.

The PVC evictor still discovers files, reads the lifetime from each namespace,
rechecks generation metadata under the stable lock, and unlinks eligible files.
Set `ENABLE_STORAGE_TTL=true` and include the **common root** `/cache` in its
`CACHE_DIRECTORY` so it can see every allowed namespace. This port adds no private
quota, LRU policy or inline expiry deleter. Physical capacity feedback and cleanup
remain the evictor's responsibility. Earlier research utility comparisons used a
separate conservative private authority and must not be treated as performance
measurements of this integration. GPU/CPU-resident cache lifetimes remain outside
this storage policy. No automatic llm-d policy injector or TTL predictor is added.

## Scope of this draft

This branch adds TTL only on top of the recency/backpressure foundation in
PR #741. It contains no deletion work-rate or file-attempt ceilings, adaptive
capacity controller, or benchmark-only C++ batching helper. TTL remains opt-in.

## Renewable idle retention and delayed retries

The default `storage_ttl_policy: "idle"` uses `.llmd-ttl-v2-idle-N` namespaces
when `storage_ttl_seconds` is configured. Select `"fixed"` explicitly for the
non-renewing v1 publication deadline. Both also support the optional allowlist.
An idle hit may reuse matching bytes still on disk after the former deadline.
Accepting the hit opens/pins the inode and refreshes atime under the same native
stripe mutex and POSIX record lock used by stores. KV bytes, inode and publication
mtime stay unchanged. A busy bucket returns pending to the scheduler; per-key native probes back off
for 10 ms between attempts rather than repeating filesystem operations on every
scheduler poll. It does not wait on the lock. Ordinary filesystem metadata operations can still incur I/O.

The evictor discovers expiry using atime + N in v2-idle namespaces and mtime + N
in v1 namespaces. Before unlinking it locks and rechecks the current file.
A renewed or replaced file therefore invalidates a stale deletion candidate.
Already accepted loads keep their descriptor until completion.

Each TTL-enabled deleter retains up to 4,096 distinct busy paths in a monotonic
250 ms delayed retry queue and continues unrelated work. Due paths are rechecked
without another crawl. Overflow falls back to discovery; the queue does not
promise an unbounded backlog or hard cleanup deadline. Shutdown does not wait
for busy locks. TTL cleanup still runs below the pressure threshold. This queue
adds no CPU quota, global capacity controller or deletion-rate option.
