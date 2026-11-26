# Reuse of identical raw buffer descriptors

The base at `10e3c562` already caches native view descriptors within each list recording. Commit `4e77a879` adds `ListViews`, a per-thread CPU-only heap keyed by the title's three descriptor words, with each recording getting a fresh cache. `State::fresh_tables` also reuses entire tables. Constant buffers materialized into new draw scratch remain outside the descriptor cache. An older estimate of 47 ms to make views predates this base.

On cache misses, `buffer_srv` and `buffer_uav` still make both the main view and its raw twin. When the normalized main view is already raw and has the same range, the patch copies its freshly made CPU-only descriptor into the twin. Typed, shifted, and shader-visible-source cases still create the twin normally. Both descriptors use the same resolved buffer, and the existing code keeps that buffer alive. The patch stores no new descriptors across calls or recordings.

In archived new-base logs, `nb-av-d.log:14505,14560,14606` reported 34,804, 55,000, and 63,180 buffer twin candidates per frame over different windows approaching and entering the scene. The counter includes attempts. Alignment alone does not tell how many of those views were identical raw twins. The diagnostics line now includes the number actually copied.

## Probe

The no-game fixture checks 64 SRV/UAV pairs against known buffer contents and checks every word in both writable buffers after dispatch. It covers independently created raw views, copied raw pairs, typed fallback, and direct creation into shader-visible slots. Each staged case rewrites the same CPU heap handles with a new resource or offset before the GPU reads the earlier visible descriptors.

On the M5 Max, CrossOver 26.3 and D3DMetal 4.0b3, all 64 pairs matched. The timing loop creates both main descriptors and then either creates or copies their raw twins. It uses 27,500 SRV/UAV pairs for 55,000 twin opportunities per sample, alternating order across 31 samples per arm. Timings are p50 / p95 in milliseconds.

| Eligible twins | Create both | Copy identical twins |
| ---: | ---: | ---: |
| 100% | 42.960 / 48.131 | 22.364 / 23.113 |
| 50% | 47.894 / 57.975 | 36.773 / 43.758 |
| 0% | 44.694 / 47.040 | 44.416 / 47.106 |

The synthetic median saving was about 0.37–0.40 μs per copied twin. It measures D3D12 descriptor creation and copying in the probe. Live eligibility, recording-thread overlap, and frame-time impact still need measurement. The Windows GNU checks passed with and without diagnostics.

## Run

Build in a scratch directory. Use the `hdxc` wrapper with the same shader arguments if the regular DXC CLI is unavailable. The probe draws no window and reads no game assets.

```sh
src=/absolute/path/to/scripts/raw-twin-probe
x86_64-w64-mingw32-g++ -std=c++20 -O2 -Wall -Wextra -Werror -static \
    "$src/probe.cpp" -o probe.exe -ld3d12 -ldxguid
dxc -T cs_6_0 -E verify -Fo verify.dxil "$src/verify.hlsl"
wine="$HOME/Applications/CrossOver.app/Contents/SharedSupport/CrossOver/bin/wine"
WINEDEBUG=-all CX_GRAPHICS_BACKEND=d3dmetal "$wine" --bottle xcl-probe "Z:$PWD/probe.exe"
```
