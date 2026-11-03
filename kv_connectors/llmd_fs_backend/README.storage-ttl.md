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

Idle mode uses one `.llmd-ttl-v3-idle` namespace per cache root, independent of
requested duration. The same model/rank/key therefore maps to one file for all
idle lifetimes. Each inode stores `user.llmd.ttl_deadline_ns`, an absolute
nanosecond deadline. Under the stable bucket lock, an accepted hit or existing
store extends it to `max(existing_deadline, now + requested_TTL)` and refreshes
atime. Shorter requests cannot shorten an existing deadline; bytes, inode and
publication mtime are preserved on reuse.

Explicit fixed mode still uses `.llmd-ttl-v1-3600`, with immutable publication
mtime + TTL. Older `.llmd-ttl-v2-idle-N` files retain their old semantics; this
revision neither migrates nor reinterprets them. Switching to v3 starts a new
shared cache population. Version separation protects incompatible formats;
there is no longer a namespace per idle duration.

V3 requires writable inode extended attributes; startup probes support and
fails if unavailable. Publication and deadline initialization finish under the
same bucket lock before a supported reader can accept the generation. A crash
between rename and metadata initialization can leave an invalid generation:
missing/malformed deadline metadata is not accepted as a hit or taken as proof
of TTL expiry. A successful new publication can replace it. Pressure eviction
continues to follow its independent idle-age rule. Cache metadata is not a
power-loss durability guarantee or a hard space reservation.

A request may confirm the namespace policy through vLLM's existing transport:

```json
{"kv_transfer_params": {"llmd_storage_ttl_seconds": 3600}}
```

Without an allowlist, a conflicting or invalid hint makes that request decline storage reuse
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

`RequestStorageTTLManager` binds one immutable lifetime route to the scheduler's
`ReqContext`. Idle routes use identical mapper paths for all supported durations;
explicit fixed routes retain per-duration v1 paths. Internal
`SharedStorageLoadStoreSpec.source_paths` aliases carry accepted read FDs or
trusted store targets, and `ttl_seconds` carries the selected idle store lifetime.
Clients never supply paths. Routes survive until accepted jobs drain, preventing
late callbacks or reused request IDs from changing another job's lifetime.

Identical prefixes with different idle lifetimes share one physical copy. The
native maximum-deadline update prevents shortening previous retention. Fixed
publication policies retain isolated validity and do not renew their deadlines.

The PVC evictor discovers files and reads the persisted v3 deadline or legacy
namespace policy, rechecks under the stable lock, and unlinks eligible files.
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

## Shared idle retention and delayed retries

The default `storage_ttl_policy: "idle"` uses `.llmd-ttl-v3-idle` when a duration
is configured. The optional allowlist selects request lifetimes, not cache
identity: its managers use the same mapper/path. The immutable internal store
spec carries the selected lifetime to the native job; no process-global policy
is changed between requests. A successful store initializes or extends the
inode deadline. Accepted lookup uses the same native stripe/POSIX guard to
open, pin and extend the deadline. Rebuild the native extension with the Python
changes before using this format.

An idle hit may reuse matching bytes still on disk after the former deadline.
KV bytes, inode and publication mtime stay unchanged. A busy bucket returns
pending with a 10 ms per-key metadata-probe backoff; ordinary filesystem
operations can still incur I/O.

The crawler reads the persisted deadline; the deleter locks and rechecks the
current inode and deadline before unlinking. Renewed or replaced files therefore
invalidate stale expiry candidates. Already accepted loads retain their file
descriptors until completion. Below the pressure threshold, a future shared
deadline keeps a file eligible for reuse; pressure eviction can still reclaim
sufficiently idle files earlier.

Each TTL-enabled deleter retains up to 4,096 distinct busy paths with 250 ms
monotonic retries and continues unrelated work. Due paths are rechecked without
another crawl. Overflow falls back to discovery; there is no unbounded backlog,
hard cleanup deadline, CPU quota or global capacity controller. Shutdown does
not wait for busy locks.
