# HNSW build diagnostic report

- Status: `passed`
- Rows / dimensions: `50000` / `64`
- `maintenance_work_mem`: `8MB`
- `m` / `ef_construction`: `16` / `64`
- Requested parallel workers: `0`
- Container image: `opentenbase-pg18-pgvector:timing-v1-arm64`
- Launched parallel workers: `0`
- Build time: `72.999s`
- Index size: `28475392` bytes
- Approximate peak container-memory delta: `36.82 MiB`
- Verification query used HNSW: `true`

## Sampled phase timeline

| Phase | First seen | Last seen | Sampled span | Tuple progress | Samples |
|---|---:|---:|---:|---:|---:|
| building index: loading tuples in memory | 0.005s | 2.036s | 2.031s | 0.088%–14.090% | 4 |
| building index: loading tuples on disk | 2.704s | 72.725s | 70.022s | 17.636%–99.806% | 105 |

## Internal build timing

- Timing status: `complete`.
- Internal HNSW elapsed wall time: `72.820972s`.
- One leader timeline; not summed worker CPU time, full CREATE INDEX time, or commit latency.

| Internal phase | Elapsed |
|---|---:|
| setup | 0.000103s |
| memory_build | 2.426489s |
| spill_drain | 0.000178s |
| flush | 0.012644s |
| disk_insert | 70.307945s |
| finalize | 0.000000s |
| wal | 0.073585s |
| cleanup | 0.000028s |

### Evidence → judgment → recommendation → revalidation

- Evidence: `disk_insert` has the largest measured interval (96.5% of internal elapsed time); spill=`True`.
- Judgment: this locates elapsed time, not a causal CPU/I/O attribution. The memory interval includes scan and worker startup; spill drain includes outstanding memory inserts and coordination; disk insertion includes scan/waits until scans finish. Flush measures page materialization, not storage-device fsync latency.
- Recommendation: if memory headroom permits, raise maintenance_work_mem; keep the same data, image, workers, m and ef_construction.
- Revalidation: compare repeated low/high-memory runs; verify spill/disk becomes N/A, then compare internal and command elapsed time and peak memory. Do not claim a speedup from this single run.

## Diagnosis

- The graph spilled after `8644` tuples.
- Spill context: `9437056` bytes used, `8388608` bytes nominal limit, dimensions `64`, `m=16`, `ef_construction=64`.
- Recommendation: if the host has enough headroom, increase `maintenance_work_mem` and rerun the same workload; accept the change only if spill disappears or build time improves without unsafe memory use.
- Observed phase order monotonic: `true`.

## Interpretation limits

- The sampled timeline contains sampled spans, not exact instrumentation; a short phase may be missed. Internal timing, when enabled, is reported separately.
- Tuple percentage is `tuples_done / requested rows`; the input must contain only indexable, non-null vectors.
- Container memory includes the database process and supporting state, not only HNSW.
- Mac ARM64 results support local functional and relative claims, not production sizing.
