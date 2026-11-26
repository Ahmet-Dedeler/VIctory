# Dynamic descriptor slot allocator benchmark

This tests the CPU work of taking and releasing contiguous descriptor slots. It runs without VIctory, Wine, Direct3D, or game data. The driver extracts the original allocation and release methods from `10e3c562` at build time. The harness gives all three allocators the same slot count and operation sequence, and checks each returned slot index or refusal. It also verifies 262,144 mixed operations across 128 small heap sizes.

```sh
python3 scripts/dynamic-slot-bench/run.py
```

Both Rust Darwin targets, Apple Silicon and Rosetta are required for the default run. Use `--arch aarch64` for the native result alone. Samples alternate allocator order. The timings include allocation, release, and the same uncontended mutex. They exclude initial state construction, correctness checking, GPU calls, and diagnostics logging and counters.

## Workload scale

- **Steady.** Archived `up-r1.log:21611,21613` showed 6,520 draws, 16,682 allocations, zero refusals, and 6.2 ms of aggregate allocation time per frame. Its older build had a different wrap bound. The steady fixture uses 16,682 requests with synthetic widths and two frames of overlapping lifetimes.
- **High load.** `nb-av-d.log:14630,14643` showed a 14,661-draw transition with 33,337 requests, 68 refusals, and 52.8 ms of aggregate allocation time per frame. The recorded line includes time waiting on the mutex. This is outside the 3,000–9,000-draw gameplay filter. The high-load fixture uses 33,337 requests and four frames of overlapping lifetimes at the Mac capacity ceiling.
- **Fragmentation.** `nb-av-d.log:14599` has a refused width-13 request. The sparse and dense controls force 68 width-13 requests to search a 262,144-slot heap that has ample total free space in holes smaller than 13. These deliberately reset the baseline's failed-width shortcut between requests. The log does not identify how many refusals actually scanned the heap.

## Results

M5 Max, Rust 1.100.0 optimized, 61 paired samples per workload. Times are p50 / p95 in milliseconds per synthetic frame. Both prototypes matched the baseline on every requested allocation.

| CPU | Workload | Current byte scan | Free intervals | Packed occupancy |
| --- | --- | ---: | ---: | ---: |
| Rosetta | Steady | 0.792 / 0.924 | 0.855 / 0.943 | **0.448 / 0.532** |
| Rosetta | High load | 1.544 / 1.711 | 1.695 / 1.890 | **0.911 / 1.113** |
| Rosetta | Fragmented, 256 holes | 25.931 / 29.020 | **0.040 / 0.055** | 0.823 / 1.020 |
| Rosetta | Fragmented, 65,536 holes | 25.158 / 26.278 | 6.172 / 6.554 | **1.192 / 1.341** |
| Arm64 | Steady | 0.549 / 0.758 | 0.543 / 0.827 | **0.291 / 0.361** |
| Arm64 | High load | 1.191 / 1.468 | 1.167 / 1.560 | **0.639 / 0.892** |
| Arm64 | Fragmented, 256 holes | 18.301 / 19.960 | **0.034 / 0.057** | 0.586 / 0.671 |
| Arm64 | Fragmented, 65,536 holes | 16.651 / 18.328 | 4.704 / 5.347 | **0.798 / 0.966** |

The packed implementation groups occupancy into 64-bit words and searches up to 64 slots at a time. It keeps counters inside the existing lock. At the Mac capacity ceiling the occupancy array takes 128 KiB per heap, down from 1 MiB. The free-interval implementation excels on sparse failures but adds overhead to ordinary Rosetta workloads and takes longer than packed occupancy in the dense control.

**Decision:** keep the packed allocator as a candidate for high-pressure frames. The ordinary Rosetta gain is under 1 ms per tested frame. Track actual scans, scanned slots, and time waiting for the allocator mutex in the next gameplay capture to decide whether to port it. The latest base-family archive does not establish steady-gameplay time for this isolated search or show whether the 52.8 ms came from repeated long scans. The native versus Rosetta difference on the packed path is too small to justify an out-of-process worker here. This branch changes only the benchmark.
