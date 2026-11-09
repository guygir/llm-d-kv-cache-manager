# Load-aware cleanup (opt-in)

Cleanup can compete with serving for CPU and shared-storage metadata work. This controller delays optional cleanup while requests are running or queued, drains after a quiet interval, and permits reclamation under urgent disk pressure. A CPU quota bounds the whole cleanup container; it is not a latency guarantee.

With `CLEANUP_LOAD_AWARE=false` (the default), the existing disk-pressure hysteresis and FIFO candidate pipeline are unchanged. This branch builds on PR741 and has no dependency on the TTL or age-band branches.

## Configuration

Run the evictor in its own container. Configure the container CPU limit **and** the matching verified budget. For example, this container-spec excerpt uses one CPU as an example, not an optimal/default production size:

```yaml
name: pvc-evictor
# Build the image from this branch before deployment.
resources:
  requests: {cpu: 250m, memory: 256Mi}
  limits: {cpu: "1", memory: 1Gi}
env:
  - {name: CLEANUP_LOAD_AWARE, value: "true"}
  - {name: CLEANUP_CPU_BUDGET_CORES, value: "1"}
  - name: CLEANUP_ACTIVITY_SOURCES
    value: '[{"url":"http://127.0.0.1:8000/metrics"}]'
  - {name: CLEANUP_EMERGENCY_THRESHOLD, value: "95"}
  - {name: CLEANUP_QUIET_SECONDS, value: "2"}
  - {name: CLEANUP_BUSY_MODE, value: continuous}
```

`CLEANUP_CPU_BUDGET_CORES=X` verifies that the runtime has installed a finite CPU quota no larger than X. It does not create the quota. Set `resources.limits.cpu` to X (fractional values are supported) in the pod/container spec. Startup fails if the quota cannot be verified or is too large. Linux cgroup v2 and the standard cgroup v1 CPU mount are supported. All cleanup children inherit this container budget. Do not share that container with the model server.

`CLEANUP_CPU_IDS=2,3` optionally restricts this process and its children to a subset of the existing permitted mask. CPU IDs are deployment-specific. Affinity does not reserve exclusive cores, and quota does not select cores. Neither bounds GPFS server CPU or filesystem metadata throughput.

`CLEANUP_ACTIVITY_SOURCES` is a nonempty JSON list of raw Prometheus HTTP(S) endpoints. Endpoints are sampled only while cleanup is requested, not continuously below its trigger. Every configured source must provide every required gauge. The default names are `vllm:num_requests_running` and `vllm:num_requests_waiting`; per-engine/model series are summed. Optional `busy_metrics` lists additional gauges which must also be zero. Custom `running` and `waiting` names can read llm-d router gauges, for example:

```json
{"url":"http://router:9090/metrics", "running":"llm_d_epp_inflight_requests", "waiting":"llm_d_epp_flow_control_queue_size"}
```

Router producers/metrics must actually be enabled and exported. Use sources covering every relevant server, including disaggregated stages, rather than assuming one worker represents an entire shared PVC. A load-balanced Service URL can return only one replica’s gauges and is not proof of full coverage; use individual endpoints or a verified aggregate. Larger pools need an aggregate rather than silently omitting engines to fit the source limit. Do not point this parser at the Prometheus JSON query API. Credentials in URLs are rejected; use appropriately protected internal endpoints with standard TLS trust.

Missing gauges, invalid/negative/nonfinite values, failed requests and responses above 1 MiB mean **unknown/busy**, never zero load. No previous successful idle sample is reused after failure. `CLEANUP_ACTIVITY_TIMEOUT_SECONDS` is a sampling budget (default 1 second); individual network operations time out after at most 0.2 seconds, and the reader checks the total budget between reads. One persistent daemon reader and bounded request/response slots keep DNS/socket stalls outside the policy loop. A sample exceeding its deadline is unknown; late results are discarded, and no new fetch starts while the old one is stuck. The reader has bounded response size and performs no deletion. Request gauges do not prove all background KV transfers have stopped; add exported transfer gauges if that distinction is needed.

## State transitions

- The existing `CLEANUP_THRESHOLD` starts a cleanup request; `TARGET_THRESHOLD` clears it.
- While cleanup is requested, all sources must remain zero for `CLEANUP_QUIET_SECONDS` before **quiet-drain** admits work.
- A positive or unknown activity sample immediately closes optional admission on the next poll; there is no minimum ON time that delays this pause.
- At `CLEANUP_EMERGENCY_THRESHOLD`, **urgent** cleanup is allowed even with traffic or unknown activity. Urgency stays latched until usage falls to `CLEANUP_THRESHOLD`; the cleanup request itself remains until target.
- `CLEANUP_BUSY_MODE=continuous` continuously admits urgent work within the CPU quota. `windowed` instead uses `CLEANUP_WINDOW_ON_SECONDS` / `CLEANUP_WINDOW_OFF_SECONDS` (defaults 1 / 1). These modes are alternatives for validation, not established latency winners.
- Missing disk usage pauses admission; it does not assume an empty disk or clear the pressure history.

Paused admission stops further crawler iteration, deletion-batch admission and folder-cleaner work. Existing syscalls and a batch already admitted can finish. This is polling-based control, not hard preemption or a worst-case response-time guarantee. The usual hot-file recheck still applies. Pending queued work is not force-deleted during shutdown. Disk/quiet hysteresis avoids rapid restarts; the CPU ceiling is unchanged between modes.

## Validation status

Local tests cover parser/configuration errors, quota/affinity validation, unknown activity, quiet/urgent hysteresis, windows, cancellation and loopback HTTP controlling real spawned crawler/deleter processes. They do not validate GPFS latency, deployment CPU isolation, or physical capacity recovery. The separate benchmark plan must pass before recommending settings.
