# Dynamic descriptor slot allocator benchmark

This tests the CPU work of taking and releasing contiguous descriptor slots. It runs without VIctory, Wine, Direct3D, or game data. The driver extracts the original allocation and release methods from `10e3c562` at build time and compiles the current runtime allocator module. The harness gives the original allocator, free-interval prototype, and both runtime modes the same slot count and operation sequence. It checks each returned slot index or refusal and verifies 262,144 mixed operations across 128 small heap sizes.

```sh
python3 scripts/dynamic-slot-bench/run.py
```

Both Rust Darwin targets, Apple Silicon and Rosetta are required for the default run. Use `--arch aarch64` for the native result alone. Samples alternate allocator order. The timings include allocation, release, and the same uncontended mutex. They exclude initial state construction, correctness checking, GPU calls, and diagnostics logging and counters.

## Workload scale

- **Steady.** Archived `up-r1.log:21611,21613` showed 6,520 draws, 16,682 allocations, zero refusals, and 6.2 ms of aggregate allocation time per frame. Its older build had a different wrap bound. The steady fixture uses 16,682 requests with synthetic widths and two frames of overlapping lifetimes.
- **High load.** `nb-av-d.log:14630,14643` showed a 14,661-draw transition with 33,337 requests, 68 refusals, and 52.8 ms of aggregate allocation time per frame. The recorded line includes time waiting on the mutex. This is outside the 3,000–9,000-draw gameplay filter. The high-load fixture uses 33,337 requests and four frames of overlapping lifetimes at the Mac capacity ceiling.
- **Fragmentation.** `nb-av-d.log:14599` has a refused width-13 request. The sparse and dense controls force 68 width-13 requests to search a 262,144-slot heap that has ample total free space in holes smaller than 13. These deliberately reset the baseline's failed-width shortcut between requests. The log does not identify how many refusals actually scanned the heap.

## Results

M5 Max, Rust 1.98.1 optimized, 61 paired samples per workload. Times are p50 / p95 in milliseconds per synthetic frame. All three alternatives matched the original allocator on every requested allocation.

| CPU | Workload | Original byte scan | Runtime byte scan | Free intervals | Runtime packed |
| --- | --- | ---: | ---: | ---: | ---: |
| Rosetta | Steady | 0.607 / 0.724 | 0.517 / 0.623 | 0.676 / 0.798 | **0.340 / 0.389** |
| Rosetta | High load | 1.221 / 1.482 | 1.050 / 1.159 | 1.347 / 1.528 | **0.685 / 0.885** |
| Rosetta | Fragmented, 256 holes | 18.694 / 21.311 | 9.120 / 10.016 | **0.032 / 0.040** | 0.562 / 0.881 |
| Rosetta | Fragmented, 65,536 holes | 19.081 / 21.792 | 9.466 / 10.319 | 4.849 / 5.174 | **0.821 / 1.318** |
| Arm64 | Steady | 0.432 / 0.562 | 0.394 / 0.544 | 0.422 / 0.575 | **0.226 / 0.307** |
| Arm64 | High load | 0.852 / 0.869 | 0.784 / 0.805 | 0.821 / 0.849 | **0.449 / 0.460** |
| Arm64 | Fragmented, 256 holes | 11.595 / 13.604 | 5.641 / 7.118 | **0.025 / 0.030** | 0.350 / 0.398 |
| Arm64 | Fragmented, 65,536 holes | 11.110 / 13.297 | 5.481 / 6.935 | 3.063 / 3.405 | **0.471 / 0.586** |

The packed implementation groups occupancy into 64-bit words and searches up to 64 slots at a time. Both runtime modes keep their free-slot count and failed-width cache inside the existing lock. At the Mac capacity ceiling the packed occupancy array takes 128 KiB per heap, down from 1 MiB. The free-interval implementation excels on sparse failures but adds overhead to ordinary Rosetta workloads and takes longer than packed occupancy in the dense control.

## Runtime check

`VICTORY_PACKED_DYNAMIC_SLOTS=1` opts into packed occupancy. The heap keeps the mode it began with. In a diagnostics build, `VICTORY_DYNAMIC_SLOT_STATS=1` reports mean per-frame allocation mutex wait, hold time, slots or words examined, and searches that failed after actually looking. This supplements the existing aggregate allocation clock without timing individual slots.

Keep packed occupancy opt-in. The latest base-family archive does not establish steady-gameplay time for the isolated search or show whether the 52.8 ms came from repeated long scans. To promote it, compare the same save and route across both modes with the new counters enabled. In windows with repeated long searches, require at least 25% and 1 ms/frame less allocator lock hold time at p95 across comparable counter windows. Keep draw and allocation counts comparable, verify output and refusal counts, and require steady frame p95 within 5%. The expected ordinary gain from the standalone test is under 1 ms/frame. The measured native versus Rosetta gap on the packed path is too small to justify moving this allocator out of process.
