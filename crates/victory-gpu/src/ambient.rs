//! A stand-in indirect diffuse light for scenes where ray-traced probes have
//! not run yet, shaded in creases and under objects from the scene's depth
//! (ported from Belobug/VIctory-Belobug-AMD `cd9f78c`, compiled via DXC on
//! Wine/D3DMetal).
//!
//! Grand Theft Auto VI's indirect diffuse light comes from probes it fills
//! by tracing rays, which do not run here yet: its program that resolves
//! them (`CS_ScreenProbes_TemporalResolve_*`) leaves next to nothing (or
//! uninitialized garbage indoors), and whatever the sun does not reach
//! directly is pitch black. Right after that program, its output is replaced
//! by a depth-occluded ambient level (`VICTORY_AMBIENT=0.4`,
//! `VICTORY_AMBIENT_SHADE=1.0,0.6,0.1,0.45`, `VICTORY_AMBIENT=off` to disable).
//! `VICTORY_AMBIENT=auto` starts from the same level and adapts it to the
//! scene's direct light (see the shader); the fixed 0.4 stays the default:
//! the adaptation's input mapping is not measured yet.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, Ordering};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::{HostList, compile, host};
use crate::object::This;

/// The programs it follows, by the start of their name.
pub(crate) const PROGRAM: &[u8] = b"CS_ScreenProbes_TemporalResolve";
/// Where they write the indirect diffuse light (R9G9B9E5 as 32-bit
/// integers): root parameter and place in its table.
pub(crate) const LIGHT: (u8, u32) = (1, 176);
/// Where the resolve reads the scene's depth (reversed, D32 with stencil):
/// root parameter and place in its table.
pub(crate) const DEPTH: (u8, u32) = (0, 464);

/// 0.4 balances night scenes (opening cutscene) and daytime shadows without
/// clipping bright highlights under night auto-exposure; override via
/// `VICTORY_AMBIENT=<level>` (e.g. `0.6` for brighter daytime) or `off`.
const DEFAULT: f32 = 0.4;

/// The stand-in diffuse level; `None` if off. With `VICTORY_AMBIENT=auto`
/// this is the level used until the pass's own estimate is ready.
pub(crate) fn level() -> Option<f32> {
    static LEVEL: OnceLock<f32> = OnceLock::new();
    let level = *LEVEL.get_or_init(|| {
        let set = std::env::var("VICTORY_AMBIENT").unwrap_or_default();
        if set.trim().eq_ignore_ascii_case("off") {
            return -1.0;
        }
        set.trim()
            .parse()
            .ok()
            .filter(|l: &f32| l.is_finite() && *l >= 0.0)
            .unwrap_or(DEFAULT)
    });
    (level >= 0.0).then_some(level)
}

/// Whether the level follows the scene (`VICTORY_AMBIENT=auto`). Any other
/// value keeps the fixed level above; `off` is not auto.
fn automatic() -> bool {
    static AUTO: OnceLock<bool> = OnceLock::new();
    *AUTO.get_or_init(|| {
        std::env::var("VICTORY_AMBIENT").is_ok_and(|v| v.trim().eq_ignore_ascii_case("auto"))
    })
}

/// Colour of the stand-in ambient light (`VICTORY_AMBIENT_TINT=r,g,b`).
fn tint() -> [f32; 3] {
    if let Some((_, t)) = *LIVE.lock().unwrap_or_else(|e| e.into_inner()) {
        return t;
    }
    static TINT: OnceLock<[f32; 3]> = OnceLock::new();
    *TINT.get_or_init(|| {
        let mut tint = [1.0, 0.98, 0.94];
        let set = std::env::var("VICTORY_AMBIENT_TINT").unwrap_or_default();
        for (to, v) in tint.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        tint
    })
}

/// The level and colour last read from `VICTORY_AMBIENT_FILE`, if any.
static LIVE: std::sync::Mutex<Option<(f32, [f32; 3])>> = std::sync::Mutex::new(None);

/// How much less of the light reaches creases and corners, worked out from
/// the depth image alone: strength (0 for none), the radius looked at in
/// metres, the near plane's distance, and the tangent of half the vertical
/// field of view. `VICTORY_AMBIENT_SHADE=s,r,n,t` sets them.
static SHADE: std::sync::Mutex<Option<[f32; 4]>> = std::sync::Mutex::new(None);

fn shade() -> [f32; 4] {
    if let Some(live) = *SHADE.lock().unwrap_or_else(|e| e.into_inner()) {
        return live;
    }
    static SET: OnceLock<[f32; 4]> = OnceLock::new();
    *SET.get_or_init(|| {
        let mut shade = [1.0, 0.6, 0.1, 0.45];
        let set = std::env::var("VICTORY_AMBIENT_SHADE").unwrap_or_default();
        for (to, v) in shade.iter_mut().zip(set.split(',')) {
            if let Ok(v) = v.trim().parse() {
                *to = v;
            }
        }
        shade
    })
}

/// The level from `VICTORY_AMBIENT_FILE`, if it gave one; starts the thread
/// that reads the file.
fn live_level() -> Option<f32> {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let Ok(path) = std::env::var("VICTORY_AMBIENT_FILE") else {
            return;
        };
        std::thread::spawn(move || {
            loop {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    let read: Vec<f32> = text
                        .split_whitespace()
                        .filter_map(|v| v.parse().ok())
                        .collect();
                    if let [level, r, g, b, ref rest @ ..] = read[..] {
                        if let [s, rad, n, t] = rest[..] {
                            *SHADE.lock().unwrap_or_else(|e| e.into_inner()) = Some([s, rad, n, t]);
                        }
                        let new = Some((level.max(0.0), [r, g, b]));
                        let mut live = LIVE.lock().unwrap_or_else(|e| e.into_inner());
                        if *live != new {
                            tracing::info!(level, tint = ?[r, g, b], "ambient light updated");
                            *live = new;
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(250));
            }
        });
    });
    LIVE.lock()
        .unwrap_or_else(|e| e.into_inner())
        .map(|(l, _)| l)
}

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=12, b0), DescriptorTable(UAV(u0), UAV(u1), SRV(t0), SRV(t1))"
cbuffer Floor : register(b0) {
    float level; float3 tint;
    float strength; float radius; float near; float tan_y;
    // adaptive is 1 with VICTORY_AMBIENT=auto and a direct-light texture,
    // else 0. frame counts this pass's dispatches; its low bit chooses the
    // sample and state slots below. Two constants of padding keep the cbuffer
    // layout explicit.
    float adaptive; float frame; float2 padding;
};
RWTexture2D<uint> light : register(u0);
RWBuffer<float4> adapt : register(u1);
Texture2D<float> depth : register(t0);
Texture2D<float4> direct : register(t1);

// The point seen at a texel, in metres from the eye (x right, y up, z away).
float3 seen(int2 p, float2 size) {
    float d = depth.Load(int3(clamp(p, int2(0, 0), int2(size) - 1), 0));
    float z = near / max(d, 1e-7);
    float2 ndc = (float2(p) + 0.5) / size * 2.0 - 1.0;
    return float3(ndc.x * tan_y * size.x / size.y * z, -ndc.y * tan_y * z, z);
}
// How much of the light from around reaches the texel: 1 in the open.
float open_to_light(int2 p, float2 size) {
    float3 c = seen(p, size);
    if (c.z > 400.0) return 1.0;
    float3 l = seen(p + int2(-1, 0), size), r = seen(p + int2(1, 0), size);
    float3 u = seen(p + int2(0, -1), size), d = seen(p + int2(0, 1), size);
    float3 dx = abs(l.z - c.z) < abs(r.z - c.z) ? c - l : r - c;
    float3 dy = abs(u.z - c.z) < abs(d.z - c.z) ? c - u : d - c;
    float3 n = normalize(cross(dy, dx));
    if (dot(n, c) > 0.0) n = -n;
    float px = radius * size.y / (2.0 * tan_y * c.z);
    if (px < 1.5) return 1.0;
    px = min(px, 128.0);
    float noise = frac(52.9829189 * frac(dot(float2(p), float2(0.06711056, 0.00583715))));
    float sum = 0.0;
    const int N = 16;
    for (int i = 0; i < N; i++) {
        float a = i * 2.399963 + noise * 6.2831853;
        float2 o = float2(cos(a), sin(a)) * (sqrt((i + 0.5) / N) * px);
        float3 v = seen(p + int2(round(o)), size) - c;
        float vv = dot(v, v);
        float over = dot(v, n) - 0.004 * c.z - 0.005;
        sum += max(over, 0.0) / sqrt(vv + 1e-6) * saturate(1.0 - vv / (radius * radius));
    }
    return saturate(1.0 - strength * 2.0 * sum / N);
}

// Scene adaptation. The persistent buffer holds two float4 states, then two
// banks of 8x8 log-luminance samples:
//   [0,1]                  state slot: x level, yzw tint
//   [2,65], [66,129]       sample banks for even and odd passes: x log2
//                          luminance, y 1 for a valid sample
// Pass N reads the state written by pass N-1 and the samples written by pass
// N-1, while it fills the other bank and writes the other state. Nothing in
// one dispatch therefore reads a word another thread of that dispatch may
// write. The first two passes fall back to the CPU-given level and tint.
static const uint GRID = 8;
static const uint SAMPLES = GRID * GRID;

void sample_direct(uint parity, int2 at, uint2 size) {
    // Exactly one texel near the centre of every cell of a fixed screen grid.
    // The cell of that texel is itself, so every cell is written once.
    uint gx = min(uint(at.x) * GRID / size.x, GRID - 1);
    uint gy = min(uint(at.y) * GRID / size.y, GRID - 1);
    uint sx = (2 * gx + 1) * size.x / (2 * GRID);
    uint sy = (2 * gy + 1) * size.y / (2 * GRID);
    if (uint(at.x) != sx || uint(at.y) != sy) return;
    uint dw, dh;
    direct.GetDimensions(dw, dh);
    int2 p = int2(min(sx * dw / size.x, dw - 1), min(sy * dh / size.y, dh - 1));
    float3 c = direct.Load(int3(p, 0)).rgb;
    float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
    // NaNs are invalid; negative light is clamped by the epsilon. Log average,
    // not linear average: one lamp or the sun must not stand for the scene.
    float valid = lum == lum ? 1.0 : 0.0;
    adapt[2 + parity * SAMPLES + gy * GRID + gx] = float4(log2(max(lum, 1e-5)), valid, 0.0, 0.0);
}

// The level anchors are Belobug's eye-tuned ones: about 0.07-0.15 at night,
// 0.3 at dusk or dawn and 1.0-1.4 at noon. The scene brightness is the mean
// log2 luminance of the direct-light texture. NIGHT_EV and DAY_EV spread that
// mean over eight stops: <= -5 (about 1/32 linear) is night, >= +3 (8 linear)
// is daylight, with the measured dusk level halfway between them. The units
// are the title's HDR light units, so these two constants are the part most
// likely to need a measured adjustment.
static const float NIGHT_EV = -5.0;
static const float DAY_EV = 3.0;

float4 target_state(float ev) {
    float t = saturate((ev - NIGHT_EV) / (DAY_EV - NIGHT_EV));
    // Piecewise levels hit the anchors above at t = 0, 0.5 and 1 rather than
    // running halfway to noon's light in dusk scenes.
    float target = t < 0.5 ? lerp(0.10, 0.30, t * 2.0) : lerp(0.30, 1.20, t * 2.0 - 1.0);
    // Blue night sky, warm low sun, slightly blue noon sky. These are mild:
    // the title multiplies the colour by albedo again below.
    float3 night = float3(0.68, 0.80, 1.20);
    float3 dusk = float3(1.05, 0.88, 0.70);
    float3 noon = float3(0.88, 0.98, 1.12);
    float3 colour = t < 0.5 ? lerp(night, dusk, t * 2.0) : lerp(dusk, noon, t * 2.0 - 1.0);
    return float4(target, colour);
}

void reduce_adaptive(uint parity) {
    // The first pass has no complete sample bank to reduce. Its samples are
    // read by the second, whose state the third pass is the first to use.
    if (frame < 1.0) return;
    uint read = 1 - parity;
    float sum = 0.0;
    float count = 0.0;
    [loop] for (uint i = 0; i < SAMPLES; i++) {
        float4 s = adapt[2 + read * SAMPLES + i];
        sum += s.x * s.y;
        count += s.y;
    }
    if (count < 16.0) return;
    float4 target = target_state(sum / count);
    float4 old = adapt[1 - parity];
    // A four-pass exponential memory: long enough to keep moving lamps and
    // noisy frames from pumping the whole picture, short enough to follow a
    // cut or a fast time-of-day change in a few passes.
    adapt[parity] = old.x > 0.0 ? lerp(target, old, 0.75) : target;
}

uint pack(float3 c) {
    c = clamp(c, 0.0, 65408.0);
    float most = max(c.r, max(c.g, c.b));
    int e = clamp(int(floor(log2(max(most, 1e-9)))) + 16, 0, 31);
    float scale = exp2(float(24 - e));
    if (most * scale >= 511.5) { e = min(e + 1, 31); scale = exp2(float(24 - e)); }
    uint3 m = min(uint3(c * scale + 0.5), 511u);
    return m.r | (m.g << 9) | (m.b << 18) | (uint(e) << 27);
}
[RootSignature(RS)]
[numthreads(8, 8, 1)]
void main(uint3 at : SV_DispatchThreadID) {
    uint w, h;
    light.GetDimensions(w, h);
    if (at.x >= w || at.y >= h) return;
    uint parity = uint(frame) & 1;
    float used_level = level;
    float3 used_tint = tint;
    if (adaptive > 0.0) {
        float4 state = adapt[1 - parity];
        if (state.x > 0.0) {
            used_level = state.x;
            used_tint = state.yzw;
        }
        sample_direct(parity, int2(at.xy), uint2(w, h));
        if (at.x == 0 && at.y == 0) reduce_adaptive(parity);
    }
    // What the resolve left is not kept: without rays its inputs are never
    // written and it holds whatever their memory did (bright magenta on
    // translucent surfaces, blocks of colours indoors: run v6, 2026-11-25).
    float open = strength > 0.0 ? open_to_light(int2(at.xy), float2(w, h)) : 1.0;
    light[at.xy] = pack(used_tint * used_level * open);
}
"#;

/// Views, used in turn: more than the passes of every list in flight.
const SLOTS: u32 = 1024;
/// Two states and two 8x8 sample banks, as float4s (the shader's layout).
const ADAPT_FLOAT4S: u32 = 2 + 2 * 8 * 8;
/// Light, adaptive buffer, depth and direct-light descriptors per dispatch.
const VIEWS: u32 = 4;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    clear_views: ID3D12DescriptorHeap,
    adaptive: ID3D12Resource,
    step: u32,
    next: AtomicU32,
}
// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Pass {}
// SAFETY: as above.
unsafe impl Sync for Pass {}

fn make(device: &ID3D12Device) -> windows::core::Result<Pass> {
    let code = compile(
        SOURCE,
        windows::core::s!("main"),
        windows::core::s!("cs_5_1"),
    )?;
    // SAFETY: the root signature is inside the shader; plain creations.
    unsafe {
        let root: ID3D12RootSignature = device.CreateRootSignature(0, &code)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: code.as_ptr().cast(),
                BytecodeLength: code.len(),
            },
            ..Default::default()
        })?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        let views = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: SLOTS * VIEWS,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: u64::from(ADAPT_FLOAT4S) * 16,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut adaptive: Option<ID3D12Resource> = None;
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            None,
            &mut adaptive,
        )?;
        let adaptive = adaptive.ok_or_else(windows::core::Error::empty)?;
        let clear_views: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: kind,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })?;
        let clear_view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    NumElements: ADAPT_FLOAT4S,
                    ..Default::default()
                },
            },
        };
        device.CreateUnorderedAccessView(
            &adaptive,
            None::<&ID3D12Resource>,
            Some(&clear_view),
            clear_views.GetCPUDescriptorHandleForHeapStart(),
        );
        Ok(Pass {
            root,
            pso,
            views,
            clear_views,
            adaptive,
            step: device.GetDescriptorHandleIncrementSize(kind),
            next: AtomicU32::new(0),
        })
    }
}

fn uav_barrier(list: &ID3D12GraphicsCommandList) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        ..Default::default()
    };
    // SAFETY: an open list; the barrier names no resource.
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// A format a float colour SRV of `format` can have. The accumulation of the
/// scene's direct light is HDR, so single and two-channel G-buffer formats
/// are not candidates even though the resolve reads them too.
fn direct_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    match format {
        DXGI_FORMAT_R32G32B32A32_TYPELESS => Some(DXGI_FORMAT_R32G32B32A32_FLOAT),
        DXGI_FORMAT_R32G32B32A32_FLOAT => Some(format),
        DXGI_FORMAT_R32G32B32_TYPELESS => Some(DXGI_FORMAT_R32G32B32_FLOAT),
        DXGI_FORMAT_R32G32B32_FLOAT => Some(format),
        DXGI_FORMAT_R16G16B16A16_TYPELESS => Some(DXGI_FORMAT_R16G16B16A16_FLOAT),
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(format),
        DXGI_FORMAT_R11G11B10_FLOAT | DXGI_FORMAT_R9G9B9E5_SHAREDEXP => Some(format),
        _ => None,
    }
}

/// The most plausible direct-light texture among the resolve's scene inputs.
/// A full-resolution HDR texture that the card can also write is preferred;
/// the smaller screen-probe inputs are in the other descriptor table and are
/// not in `candidates`.
fn direct_light<'a>(
    candidates: &[This],
    width: u64,
    height: u32,
) -> Option<(&'a ID3D12Resource, DXGI_FORMAT)> {
    let mut best = None;
    let mut best_score = 0;
    for &candidate in candidates {
        let Some(texture) = crate::resource::host_resource(candidate) else {
            continue;
        };
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        let Some(format) = direct_format(d.Format) else {
            continue;
        };
        let mut score = 1;
        score += 8 * u32::from((d.Width, d.Height) == (width, height));
        score += 3 * u32::from(d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0);
        score += u32::from(d.MipLevels == 1);
        score += u32::from(d.DepthOrArraySize == 1);
        if score > best_score {
            best_score = score;
            best = Some((texture, format));
        }
    }
    best
}

/// Replaces what `light` holds with the depth-shaded stand-in ambient level,
/// in `list`, after the program that wrote it as an image. Returns true if it
/// recorded a dispatch (the list's descriptor heaps / compute PSO are then ours).
pub(crate) fn raise(
    list: &HostList,
    light: This,
    depth: Option<This>,
    direct_candidates: &[This],
) -> bool {
    let (Some(level), Some(host)) = (level(), host()) else {
        return false;
    };
    let Some(light) = crate::resource::host_resource(light) else {
        return false;
    };
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    let Some(pass) = PASS
        .get_or_init(|| {
            make(&host.device)
                .inspect_err(|e| tracing::warn!(error = %e, "no ambient light"))
                .ok()
        })
        .as_ref()
    else {
        return false;
    };
    // SAFETY: a live resource.
    let d = unsafe { light.GetDesc() };
    static ONCE: std::sync::Once = std::sync::Once::new();
    if d.Format != DXGI_FORMAT_R32_UINT {
        ONCE.call_once(|| tracing::warn!(format = d.Format.0, "no ambient light: another format"));
        return false;
    }
    // The depth the resolve just read, in the state it read it in; of the
    // light's size, or the shading is left out.
    let depth = depth.and_then(crate::resource::host_resource).filter(|t| {
        // SAFETY: a live resource.
        let t = unsafe { t.GetDesc() };
        t.Format == DXGI_FORMAT_R32G8X24_TYPELESS && (t.Width, t.Height) == (d.Width, d.Height)
    });
    let live = live_level();
    let direct = if automatic() && live.is_none() {
        direct_light(direct_candidates, d.Width, d.Height)
    } else {
        None
    };
    ONCE.call_once(|| {
        tracing::info!(
            level,
            width = d.Width,
            height = d.Height,
            depth_ao = depth.is_some(),
            "ambient light"
        );
    });
    static ADAPT_ONCE: std::sync::Once = std::sync::Once::new();
    if automatic() {
        ADAPT_ONCE.call_once(|| match direct {
            Some((_, format)) => tracing::info!(
                format = format.0,
                "ambient light adapts to the direct light"
            ),
            None => tracing::warn!(
                "ambient light stays fixed: no HDR direct-light texture among the resolve's inputs"
            ),
        });
    }
    let serial = pass.next.fetch_add(1, Ordering::Relaxed);
    let slot = serial % SLOTS * VIEWS;
    list.keep(light);
    if let Some(depth) = depth {
        list.keep(depth);
    }
    if let Some((direct, _)) = direct {
        list.keep(direct);
    }
    let l = &list.list;
    uav_barrier(l);
    // SAFETY: an open list; live textures the list keeps, in the states the
    // resolve left them in; views in a slot pair no list in flight reads.
    unsafe {
        let cpu = pass.views.GetCPUDescriptorHandleForHeapStart();
        let gpu = pass.views.GetGPUDescriptorHandleForHeapStart();
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: cpu.ptr + (slot * pass.step) as usize,
        };
        let view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
            ..Default::default()
        };
        host.device
            .CreateUnorderedAccessView(light, None::<&ID3D12Resource>, Some(&view), at);
        let at_adaptive = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: at.ptr + pass.step as usize,
        };
        let adaptive_view = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV {
                    FirstElement: 0,
                    NumElements: ADAPT_FLOAT4S,
                    StructureByteStride: 0,
                    CounterOffsetInBytes: 0,
                    Flags: D3D12_BUFFER_UAV_FLAG_NONE,
                },
            },
        };
        host.device.CreateUnorderedAccessView(
            &pass.adaptive,
            None::<&ID3D12Resource>,
            Some(&adaptive_view),
            at_adaptive,
        );
        let seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let at_depth = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: at.ptr + 2 * pass.step as usize,
        };
        host.device
            .CreateShaderResourceView(depth, Some(&seen), at_depth);
        let direct_format = direct
            .map(|(_, format)| format)
            .unwrap_or(DXGI_FORMAT_R16G16B16A16_FLOAT);
        let direct_seen = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: direct_format,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        let at_direct = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: at.ptr + 3 * pass.step as usize,
        };
        host.device.CreateShaderResourceView(
            direct.map(|(texture, _)| texture),
            Some(&direct_seen),
            at_direct,
        );
        if serial == 0 {
            // Committed memory is not zeroed. Clear the small persistent
            // buffer once so every state starts invalid (level zero), rather
            // than whatever the allocator handed us.
            let adaptive_gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from((slot + 1) * pass.step),
            };
            let values = [0.0; 4];
            l.ClearUnorderedAccessViewFloat(
                adaptive_gpu,
                pass.clear_views.GetCPUDescriptorHandleForHeapStart(),
                &pass.adaptive,
                &values,
                &[],
            );
            uav_barrier(l);
        }
        l.SetDescriptorHeaps(&[Some(pass.views.clone())]);
        l.SetComputeRootSignature(&pass.root);
        l.SetPipelineState(&pass.pso);
        let [r, g, b] = tint();
        let level = live.unwrap_or(level);
        let [strength, radius, near, tan_y] = shade();
        let strength = if depth.is_some() { strength } else { 0.0 };
        let adapting = if direct.is_some() { 1.0 } else { 0.0 };
        let constants = [
            level,
            r,
            g,
            b,
            strength,
            radius,
            near,
            tan_y,
            adapting,
            serial as f32,
            0.0,
            0.0,
        ];
        l.SetComputeRoot32BitConstants(0, 12, constants.as_ptr().cast(), 0);
        l.SetComputeRootDescriptorTable(
            1,
            D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: gpu.ptr + u64::from(slot * pass.step),
            },
        );
        l.Dispatch((d.Width as u32).div_ceil(8), d.Height.div_ceil(8), 1);
    }
    uav_barrier(l);
    true
}
