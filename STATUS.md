# Native DXIL half rounding: status

Branch `air-fp16`, on top of the source snapshot of 35db0038. Prototype only: it is
**off by default**, it has **not been run on a GPU**, and no game was run.

## What changed

There is a new opt-in switch, `VICTORY_F16_NATIVE=1`. It only does anything together with
`VICTORY_F16_NEAREST=1`, which is when exact nearest-even rounding has already been asked for.

| Settings | 16-bit float results (`v_add_f16`, `v_mul_f16`, `v_cvt_f16_f32`, `v_fma_f16`, ...) |
|---|---|
| neither (default) | host `legacyF32ToF16` (toward zero on the tested backends), unchanged |
| `F16_NATIVE` only | the same, byte for byte |
| `F16_NEAREST` only | software nearest-even (`Lower::f16_nearest`, about 12 instructions + 3 host conversions), unchanged |
| both | `fptrunc float to half` → `bitcast half to i16` → `zext i16 to i32` (`Lower::f16_nearest_native`) |

The change is limited to the `float_result` closure in `Lower::valu16_op` (`lower.rs`).
`Lower::f16_nearest` itself is untouched. It also feeds `Lower::half_sum` (the
wave sum of halves), which is exact on the default path and stays exact by software
rounding under every setting. No half arithmetic, FMA, resource binding, mesh or Wave64
code was touched.

A shader that ends up using native 16-bit code declares it the same way DXC does for the
clean-room `reference-native.ll`:

- the `half` type (LLVM `TYPE_CODE_HALF` = 10) and `i16`, plus the `fptrunc` cast (`CAST_FPTRUNC` = 7);
- the data layout `…-i8:8-i16:16-…-f16:16-…` (DXC `kNewLayoutString`) instead of the
  minimum-precision `…-i8:32-i16:32-…-f16:32-…`;
- the shader flags `LowPrecisionPresent` (bit 5) and `UseNativeLowPrecision` (bit 23). DXC's
  value for the reference is 8388656 = `RAW_BUFFERS | LOW_PRECISION | NATIVE_LOW_PRECISION`;
- the `SFI0` feature bit `NativeLowPrecision` (0x40000);
- shader model 6.6 and DXIL 1.6, as before. That is above the 6.2 that native 16-bit types need.

The flags and the layout are set from one test (`Module::uses_16_bit`, run after the
clean-up passes, over reached blocks). They are declared exactly when written code makes
or reads a `half`/`i16`, because the validator compares flags against the code and ties the
layout to the flag. Modules that never create a 16-bit type skip that scan entirely.

`VICTORY_F16_NATIVE` is part of the shader cache key (`crates/victory-gpu/src/shader_cache.rs`),
next to `VICTORY_F16_NEAREST`.

Files: `crates/victory-shader/src/emit/{bitcode.rs, shader.rs, lower.rs, mod.rs,
native_half_tests.rs}` and `crates/victory-gpu/src/shader_cache.rs`.

## Verified here (M4 Air, macOS, no GPU run)

- `cargo test -p victory-shader`: **210 passed, 0 failed, 2 ignored** (baseline 202 passed,
  2 ignored; the 8 new tests are in `emit/native_half_tests.rs`). These tests run natively and
  need neither `dxil.dll`, a D3D12 device, nor Apple's IR converter. They read the emitted
  container and its bitcode back with a small bitstream reader and check:
  - one chained `fptrunc`→half / `bitcast`→i16 / `zext`→i32 per rounding, with no
    `dx.op.legacyF32ToF16` left in the shader, and at least 10 fewer instructions than the
    software path (for `v_add_f16` and `v_cvt_f16_f32`);
  - layout, flags (= software flags | bits 5, 23), `SFI0` (= software | 0x40000), SM ≥ 6.2,
    DXIL ≥ 1.2;
  - the constants against `reference-native.ll`: layout string, `i64 8388656`, `fptrunc float … to half`,
    `cs 6.2`. The file is read only when present at the checkout root. It is **not committed**
    because shader dumps are kept out of the repo (`docs/asset-policy.md`), so the test
    falls back to the embedded constants;
  - `F16_NATIVE` without `F16_NEAREST` gives identical bytes; `v_add_f32` gives identical bytes
    under all four settings; the exact wave sum of halves gives identical bytes and keeps
    software rounding; a module whose 16-bit code is unreachable declares nothing.
- **Byte identity with 35db0038**: with a throwaway hashing test run on the original sources
  (`git stash`) and on this branch, the default and `F16_NEAREST`-only containers for
  `v_add_f16`, `v_cvt_f16_f32`, `v_add_f32` and the wave sum of halves were identical.
- The test's CPU oracle (`f16_oracle`, IEEE RNE with denormals) matched LLVM's own
  `f32 as f16` on 1,382,921,948 non-NaN inputs, including every f32 with exponent 100–145 of
  either sign (half denormals, normals and overflow), with 0 mismatches. This was a scratch
  program outside the repo.
- `cargo clippy -p victory-shader --tests` is clean for the new code on both the macOS and
  `x86_64-pc-windows-msvc` targets. `cargo check --tests --target x86_64-pc-windows-msvc`
  type-checks the Windows-only device test. It could not be linked or run here.

Not done here: DXIL validation (`dxil.dll`), `dxc -dumpbin`, Apple's converter, any GPU
dispatch. `victory-gpu` does not build on macOS in this snapshot (pre-existing errors in
`list.rs`/`resource.rs`, Windows-only modules configured out), and its Windows cross-check
stops at blake3's C build here. The cache-key edit is one string in a list.

## Portable validation commands

Run from the checkout root. Any OS works for the first three.

```sh
# The shader suite (structure, flags, identical off output). No GPU, no dxil.dll needed.
nice -n 15 cargo test -p victory-shader -j 3

# Only the native-half tests, with the reference cross-check printed.
nice -n 15 cargo test -p victory-shader -j 3 native_half -- --nocapture

# Type-check the Windows-only device test from a Mac or Linux box.
nice -n 15 cargo check -p victory-shader --tests --target x86_64-pc-windows-msvc -j 3

# On a machine with a D3D12 device (Windows, or the Windows test binary in the D3DMetal
# setup the Pro used): validate with dxil.dll, then sweep special values both ways.
#   VICTORY_DXIL=<path to dxil.dll>   (optional: otherwise the newest Windows SDK's is used)
cargo test -p victory-shader --target x86_64-pc-windows-msvc \
    on_the_card_native_rounding_matches_the_software_rounding -- --nocapture
```

The device test skips (passes with a note) when there is no device or the device refuses
native 16-bit programs. Check its output for `not run`.

## Hardware qualification still required (Pro, before anyone turns it on)

1. **Validator and driver acceptance.** Run `on_the_card_native_rounding_matches_the_software_rounding`
   with `dxil.dll` available. Both containers must pass validation (this confirms the
   flag/layout/`SFI0` coupling that is inferred from DXC behaviour here). Both pipelines must
   be created. Under D3DMetal, confirm that Apple's converter takes the translator's container,
   and not just DXC's. On Windows hosts, native 16-bit programs need
   `D3D12_FEATURE_DATA_D3D12_OPTIONS4::Native16BitShaderOpsSupported`; without it the
   pipeline is refused.
2. **Special values.** The same test converts 18 sweeps × 64 lanes with `v_cvt_f16_f32`
   through both paths and compares each against the IEEE oracle. It **fails** if a normal
   case (including overflow and ±inf) differs from IEEE on either path, or if a NaN comes
   out as a non-NaN or with the other sign. It **reports** the rest, which a person must judge:
   - **NaN payloads.** The software path keeps whatever bits the host's `legacyF32ToF16` makes of a NaN. `fptrunc`
     may truncate the payload, canonicalise to 0x7E00/0xFE00, or quiet sNaNs differently.
     Neither path is known to match the console here. Decide whether payload differences
     matter (they only would if a title tests NaN bits).
   - **Half denormal results** (|x| < 2^-14) and ties at 2^-25. SM 6.2 keeps 16-bit denormals
     with native types. The software path's `legacyF32ToF16`/`legacyF16ToF32` pair may flush
     them. If the paths differ here, native is probably the IEEE (console-like) one, but
     confirm against the oracle column.
   - **f32 denormal inputs.** Either path may flush to a zero of either sign first.
   - **Overflow.** IEEE (and the software path) send |x| ≥ 65520 to ±inf. A host conversion that
     saturates to ±65504 fails the test. Sweeps cover 65504–65520–65536 and far above, for both
     signs.
   - **Negative values.** The Pro's 4,276,224-sample result covered positive normals only. The
     sweeps include negative ties and halves.
3. **Arithmetic results, not just conversions.** The sweep uses `v_cvt_f16_f32`. `v_add_f16`,
   `v_mul_f16`, `v_fma_f16` etc. go through the same `fptrunc`. Apple's compiler could
   still narrow `fptrunc(f32 op)` differently, e.g. by contracting an f32 multiply-add into a
   half FMA. Repeat the Pro's earlier 4.27M-sample comparison through the *translated* path
   (`VICTORY_F16_NEAREST=1 VICTORY_F16_NATIVE=1` vs `VICTORY_F16_NEAREST=1`), including
   negative and FMA cases.
4. **Speed claim.** The 0.82 ms vs 1.23 ms p50 was hand-written HLSL in a synthetic program.
   Measure translated shaders with and without `VICTORY_F16_NATIVE` before claiming any gain.
   This change only removes the software rounding sequence per 16-bit result.

Until 1–3 pass on hardware, keep `VICTORY_F16_NATIVE` unset. The software path stays the
known-exact choice.

## Dependencies and limits

- Only `Lower::valu16_op` 16-bit float results use the native rounding. `half_sum` and
  `add_halves` keep `f16_nearest`. Moving them over would change the default path, which
  is out of scope.
- The writer now supports `half`, `i16` via `int(16)`, `fptrunc`, and both data layouts.
  There are no `half` constants, half arithmetic, 16-bit loads/stores or min-precision
  types, and none are needed for this change.
- The optimiser treats `fptrunc` as opaque (no folding). It does fold `bitcast`/`zext` only
  where they already fold for 32-bit types; casts of other widths pass through untouched.
