# Age-banded deletion candidates (opt-in)

The default deletion queue remains FIFO. Set `CANDIDATE_QUEUE_POLICY=age_bands` to prefer older **discovered deletion candidates**, without maintaining a global filesystem LRU index. This branch builds on PR741 and is independent of the load-aware and TTL branches.

```bash
CANDIDATE_QUEUE_POLICY=age_bands
CANDIDATE_AGE_BANDS_SECONDS=60,300,3600,86400
FILE_QUEUE_MAXSIZE=10000
```

Boundaries must be finite, positive, strictly increasing; 1–32 boundaries are accepted. The example defines five bands: <1 minute, 1–5 minutes, 5 minutes–1 hour, 1 hour–1 day, and >=1 day. These are examples, not an optimized policy. The existing hot-protection threshold applies first, so bands entirely younger than that threshold are normally empty.

## Candidate flow

1. The crawler applies hot-file eligibility as before and includes its already-read stat metadata in the queued ticket. It does not send every cache access or inventory every file into this index.
2. A local manager process maintains bounded ordered maps and one indexed entry per pathname. Repeated discoveries update that entry; an older observation of the same generation cannot undo a newer known access time.
3. On dequeue, a small rotating metadata scan lazily reclassifies aging entries. The oldest currently classified nonempty band is preferred; order within a band is approximate.
4. The consumer rechecks the current inode without following symlinks. A missing/replaced generation is dropped; a now-hot file leaves candidacy. An access that remains eligible but changes bands is requeued in its current band.
5. The existing deleter checks current access age again before unlinking. PR741's non-atomic stat-to-unlink window remains; this queue does not claim the stable-lock/FD protocol of the separate TTL PR.

No physical cache file moves. There is no equal-population rebalance, per-file timer or unbounded history of stale tickets. Coldest-band entries lose fine ordering by design. Lazy aging means older candidates can temporarily remain in younger bands; the scheme is approximate even among discovered candidates, and says nothing about files not yet discovered.

## Bounds, cost and restart

`FILE_QUEUE_MAXSIZE` bounds the shared index. The single consumer prefetches at most 32 tickets per IPC round trip and holds the ordinary deletion batch separately. Producers' queue-size checks remain approximate: they do not see another process's prefetched tickets. Memory is bounded by index capacity plus fixed prefetch/batch/in-flight producer buffers, not by total PVC file count. Queue statistics expose occupancy, band sizes, deduplication, aging and revalidation/drop counters.

The broker performs no filesystem I/O while holding its index lock. It performs a fixed metadata-aging budget per prefetch, not a full index scan per operation. Filesystem rechecks occur in the consumer. A generation or eligible-recency requeue can lose admission to a concurrently filled index; it is safely discarded for later discovery rather than evicting another entry or blocking indefinitely.

The manager adds an IPC process inside the cleanup container's CPU budget. Its IPC/metadata cost must be measured; better victim ordering is not proof of lower CPU or inference latency. The index is disposable, rebuilt through discovery after supervised restart. Broker death stops the evictor so a supervisor can restart it rather than leaving cleanup silently stalled. No distributed or crash-persistent LRU index is introduced.

Hot reads are learned through filesystem metadata at dequeue (and the deleter's final check); there is no connector notification on every hit. Enable PR741's explicit access tracking where mount-managed atime is insufficient.

## Validation status

Local tests cover boundaries, bounded churn/deduplication, aging, blocking/close behavior, hot hits, replaced generations, recency rebucketing and the actual factory/crawler/broker/deleter flow with spawned processes and temporary files. They do not establish GPFS throughput, oldest-first utility for production traffic, or a memory budget for an unbounded all-files index. Partial-reclamation and inference benchmarks are required before policy/default recommendations.
