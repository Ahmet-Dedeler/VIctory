//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So every record is clamped on the GPU,
//! just before its execution, by a compute shader of ours (HLSL compiled by
//! the system's compiler, or DXC when it refuses; the root signature is
//! serialized separately on the host).
//!
//! Each list also has a budget (2^27 vertex instances or threads, groups
//! counted as 64 threads, mesh groups as 128): records past it run empty,
//! so a list of thousands of garbage records cannot run for seconds
//! either. When ExecuteIndirect has a count buffer, the clamp reads that
//! GPU count and only clamps and charges the records it names: bytes past
//! the count are not work, and may be stale. Every execution still charges
//! its own records, even when the title runs the same range again for
//! another view; that keeps the budget a bound on actual work rather than
//! on addresses. Skipping a repeated range would undercount work whose
//! values the title changed between runs.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more. A max beyond 4,194,240 records (65,535 groups of
//! 64) is dropped rather than split into several dispatches.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{Interface, s};

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
cbuffer Record : register(b0) { uint kind; uint stride; uint records; uint count_at; };
// The argument records and the list's budget, as dwords. The legacy
// d3dcompiler in Wine (vkd3d-shader) has no Load methods on UAVs, so the
// clamp indexes a structured buffer instead: FXC, DXC and vkd3d all accept
// element indexing and InterlockedAdd, and the host checks that the stride
// is a multiple of 4.
RWStructuredBuffer<uint> args : register(u0);
RWStructuredBuffer<uint> budget : register(u1);
// ExecuteIndirect's count, when the title gave one: count_at is its dword
// offset, or NOT_COUNTED. Reading it on the GPU keeps records past the
// count (stale, possibly garbage) out of the budget.
RWStructuredBuffer<uint> count : register(u2);
static const uint BUDGET = 1u << 27;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint NOT_COUNTED = 0xFFFFFFFF;
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint last = records;
    if (count_at != NOT_COUNTED) {
        // Only the records ExecuteIndirect will run are clamped and
        // charged; a count buffer's value is a UINT, so this also bounds
        // it by the host's max.
        last = min(records, count[count_at]);
    }
    if (id.x >= last) return;
    // `stride` is a multiple of 4 (checked on the host), so a record's byte
    // offset is a dword offset.
    uint at = id.x * (stride >> 2);
    uint cost;
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint vertices = min(args[at], 1u << 24);
        uint instances = min(args[at + 1], (1u << 26) / max(vertices, 1u));
        cost = vertices * instances;
        args[at] = vertices;
        args[at + 1] = instances;
    } else if (kind == 2) {
        uint3 g = min(uint3(args[at], args[at + 1], args[at + 2]), uint3(65535, 65535, 65535));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64;
        args[at] = g.x;
        args[at + 1] = g.y;
        args[at + 2] = g.z;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = uint3(args[at], args[at + 1], args[at + 2]);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128;
        args[at] = g.x;
        args[at + 1] = g.y;
        args[at + 2] = g.z;
    }
    uint before;
    InterlockedAdd(budget[0], cost, before);
    if (before >= BUDGET || before + cost > BUDGET) {
        // Draws keep their layout with no vertices; groups become zero.
        args[at] = 0;
    }
}
"#;

/// The text of a compiler or serializer error blob.
fn blob_text(blob: &ID3DBlob) -> String {
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    String::from_utf8_lossy(bytes).trim().to_owned()
}

/// A DXC path as the Windows host sees it. `VICTORY_DXC_DIR` may be the
/// Linux path a parent installed DXC under; Wine maps `/` to `Z:`.
fn dxc_path(dir: &str, file: &str) -> String {
    let mut base = dir.to_owned();
    if base.starts_with('/') {
        base.insert_str(0, "Z:");
    }
    let base = base.replace('/', "\\");
    format!("{}\\{file}", base.trim_end_matches('\\'))
}

/// `dxcompiler.dll`, from `VICTORY_DXC` (a DLL path), `VICTORY_DXC_DIR`
/// (a directory), or the default search path. `dxil.dll` beside a
/// configured compiler is loaded first, so DXC finds its signing library
/// without the DLLs being deployed next to the executable.
fn load_dxc() -> windows::core::Result<windows::Win32::Foundation::HMODULE> {
    use windows::Win32::System::LibraryLoader::LoadLibraryW;
    use windows::core::PCWSTR;

    fn load(path: &str) -> windows::core::Result<windows::Win32::Foundation::HMODULE> {
        let wide: Vec<u16> = path.encode_utf16().chain([0]).collect();
        // SAFETY: a NUL-terminated path; the loader reads it during the call.
        unsafe { LoadLibraryW(PCWSTR(wide.as_ptr())) }
    }

    if let Ok(dir) = std::env::var("VICTORY_DXC_DIR") {
        let _ = load(&dxc_path(&dir, "dxil.dll"));
        let path = dxc_path(&dir, "dxcompiler.dll");
        match load(&path) {
            Ok(library) => {
                tracing::debug!(path = %path, "argument clamp: DXC loaded");
                return Ok(library);
            }
            Err(e) => tracing::debug!(path = %path, error = %e, "argument clamp: no DXC there"),
        }
    }
    if let Ok(path) = std::env::var("VICTORY_DXC") {
        let mut path = path;
        if path.starts_with('/') {
            path.insert_str(0, "Z:");
        }
        let path = path.replace('/', "\\");
        match load(&path) {
            Ok(library) => {
                tracing::debug!(path = %path, "argument clamp: DXC loaded");
                return Ok(library);
            }
            Err(e) => tracing::debug!(path = %path, error = %e, "argument clamp: no DXC there"),
        }
    }
    // The default DLL search path, as before.
    let library = load("dxcompiler.dll")?;
    tracing::debug!(path = "dxcompiler.dll", "argument clamp: DXC loaded");
    Ok(library)
}

/// Compiles a compute shader with DXC (`dxcompiler.dll`).
fn dxc_compile(source: &str) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Dxc::*;
    use windows::Win32::System::LibraryLoader::GetProcAddress;
    use windows::core::w;
    type Create = unsafe extern "system" fn(
        *const windows::core::GUID,
        *const windows::core::GUID,
        *mut *mut std::ffi::c_void,
    ) -> windows::core::HRESULT;
    // SAFETY: loading the compiler and calling its documented entry point with
    // out pointers of ours; the buffer and arguments outlive the calls.
    unsafe {
        let library = load_dxc()?;
        let create: Create = std::mem::transmute(
            GetProcAddress(library, s!("DxcCreateInstance"))
                .ok_or_else(windows::core::Error::from_thread)?,
        );
        let mut raw = std::ptr::null_mut();
        create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &mut raw).ok()?;
        let compiler = IDxcCompiler3::from_raw(raw);
        let buffer = DxcBuffer {
            Ptr: source.as_ptr().cast(),
            Size: source.len(),
            Encoding: DXC_CP_UTF8.0,
        };
        let arguments = [w!("-T"), w!("cs_6_0"), w!("-E"), w!("main"), w!("-O3")];
        let result: IDxcResult = compiler.Compile(&buffer, Some(&arguments), None)?;
        let status = result.GetStatus()?;
        if status.is_err() {
            let text = result
                .GetErrorBuffer()
                .map(|errors| {
                    // SAFETY: the blob's own bytes.
                    let bytes = std::slice::from_raw_parts(
                        errors.GetBufferPointer().cast::<u8>(),
                        errors.GetBufferSize(),
                    );
                    String::from_utf8_lossy(bytes).trim().to_owned()
                })
                .unwrap_or_default();
            tracing::warn!(
                error = %windows::core::Error::from_hresult(status),
                text = %text,
                "argument clamp: DXC refused it"
            );
            return Err(windows::core::Error::from_hresult(status));
        }
        let blob = result.GetResult()?;
        Ok(
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
                .to_vec(),
        )
    }
}

/// Compiles with the system's compiler (FXC, or Wine's vkd3d-shader
/// implementation), logging the compiler's own text when it refuses.
fn compile_legacy() -> windows::core::Result<Vec<u8>> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            SOURCE.as_ptr().cast(),
            SOURCE.len(),
            s!("clamp"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    match compiled {
        Ok(()) => {
            let code = code.ok_or_else(windows::core::Error::empty)?;
            // SAFETY: the blob's own bytes.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    code.GetBufferPointer().cast::<u8>(),
                    code.GetBufferSize(),
                )
            };
            Ok(bytes.to_vec())
        }
        Err(e) => {
            match &errors {
                Some(errors) => {
                    tracing::warn!(
                        error = %e,
                        text = %blob_text(errors),
                        "argument clamp: the system compiler refused it"
                    );
                }
                None => {
                    tracing::warn!(error = %e, "argument clamp: the system compiler refused it");
                }
            }
            Err(e)
        }
    }
}

/// The compiled clamp and which compiler made it.
fn compile() -> windows::core::Result<(Vec<u8>, &'static str)> {
    match compile_legacy() {
        Ok(bytes) => {
            tracing::info!("argument clamp: compiled by the system compiler (cs_5_1)");
            Ok((bytes, "system"))
        }
        Err(_) => {
            let bytes = dxc_compile(SOURCE)
                .inspect_err(|e| tracing::warn!(error = %e, "argument clamp: DXC failed"))?;
            tracing::info!("argument clamp: compiled by DXC (cs_6_0)");
            Ok((bytes, "dxc"))
        }
    }
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let host = host().ok_or_else(windows::core::Error::empty)?;
    let (bytes, compiler) = compile()?;
    tracing::info!(compiler, bytes = bytes.len(), "argument clamp compiled");
    // An embedded signature is valid D3D12, but not all system compilers
    // emit it. Serialize our four root parameters independently instead.
    let mut params = vec![D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: 4,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    }];
    for register in 0..3 {
        params.push(D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR {
                    ShaderRegister: register,
                    RegisterSpace: 0,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        });
    }
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: params.len() as u32,
        pParameters: params.as_ptr(),
        ..Default::default()
    };
    let mut blob = None;
    let mut errors = None;
    // SAFETY: the description borrows the live parameter vector.
    unsafe {
        D3D12SerializeRootSignature(
            &desc,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut blob,
            Some(&mut errors),
        )?;
    }
    let blob = blob.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob owns this byte slice through CreateRootSignature.
    let signature = unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
    };
    let root = match unsafe { host.device.CreateRootSignature(0, signature) } {
        Ok(root) => root,
        Err(e) => {
            tracing::warn!(error = %e, compiler, "argument clamp: root signature refused");
            return Err(e);
        }
    };
    let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
        // SAFETY: borrowed root pointer; the descriptor never releases it.
        pRootSignature: unsafe { std::mem::transmute_copy(&root) },
        CS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        },
        ..Default::default()
    };
    // SAFETY: a live device, root signature and shader bytes.
    let pso = match unsafe { host.device.CreateComputePipelineState(&desc) } {
        Ok(pso) => pso,
        Err(e) => {
            tracing::warn!(error = %e, compiler, "argument clamp: host pipeline refused");
            host.log_messages();
            return Err(e);
        }
    };
    Ok(Clamp { root, pso })
}

/// The GPU count of an indirect execution: the host copy ExecuteIndirect
/// reads (its GPU count) and the byte offset of the count in it.
#[derive(Clone, Copy)]
pub(crate) struct Count<'a> {
    pub(crate) buffer: &'a ID3D12Resource,
    pub(crate) offset: u64,
}

/// Records, in `list`, the clamp of `records` records of `kind` (PC
/// argument numbering) `stride` bytes apart at host address `gpu` in
/// `buffer`, and leaves the buffer ready to be read as arguments. Replaces
/// the list's compute root signature and pipeline (callers bind theirs
/// after). False if the clamp is not available.
#[cfg(test)]
pub(crate) fn bound(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    args: (u32, u32, u32),
) -> bool {
    bound_counted(list, buffer, gpu, args, None)
}

/// [`bound`], with the count buffer ExecuteIndirect will read: the clamp
/// reads that GPU count and only clamps and charges the records it names.
/// The count buffer is moved to INDIRECT_ARGUMENT along with the arguments;
/// pass the same buffer to [`done_counted`]. On false nothing was
/// transitioned, so `done_counted` must not be called.
pub(crate) fn bound_counted(
    list: &HostList,
    buffer: &ID3D12Resource,
    gpu: u64,
    (kind, stride, records): (u32, u32, u32),
    count: Option<Count<'_>>,
) -> bool {
    // One Dispatch can carry at most 65,535 groups of 64 records. A larger
    // max is a title (or memory) fault: fail closed rather than record an
    // illegal dispatch.
    const GROUPS: u32 = 65_535;
    if !gpu.is_multiple_of(4) || !stride.is_multiple_of(4) {
        return false;
    }
    if records > 64 * GROUPS {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::warn!(
                records,
                limit = 64 * GROUPS,
                "argument clamp: too many records for one dispatch; the execution is dropped"
            )
        });
        return false;
    }
    let count = match count {
        Some(count) => {
            if !count.offset.is_multiple_of(4) {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        offset = count.offset,
                        "argument clamp: count buffer offset not 4-byte aligned; the execution is dropped"
                    )
                });
                return false;
            }
            // The clamp reads one dword; keep it inside the host copy.
            // SAFETY: a live resource.
            let width = unsafe { count.buffer.GetDesc() }.Width;
            if count.offset > width || width - count.offset < 4 {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        offset = count.offset,
                        width,
                        "argument clamp: count buffer offset outside its buffer; the execution is dropped"
                    )
                });
                return false;
            }
            if count.offset / 4 >= u64::from(u32::MAX) {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        offset = count.offset,
                        "argument clamp: count buffer offset is not representable; the execution is dropped"
                    )
                });
                return false;
            }
            Some(count)
        }
        None => None,
    };
    let (Some(c), Some(budget)) = (clamp(), list.budget()) else {
        return false;
    };
    let count_gpu = match count {
        // SAFETY: a live buffer; the caller passes its host copy.
        Some(count) => unsafe { count.buffer.GetGPUVirtualAddress() },
        None => gpu,
    };
    let count_at = match count {
        Some(count) => (count.offset / 4) as u32,
        None => 0xFFFF_FFFF,
    };
    let same = count.is_some_and(|count| count.buffer.as_raw() == buffer.as_raw());
    let l = &list.list;
    let constants = [kind, stride, records, count_at];
    // SAFETY: an open list; our root signature, pipeline and live buffers
    // the caller's list holds. The clamp reads and writes both argument and
    // count buffers as UAVs; production buffer copies are CPU-side title
    // writes, not GPU COPY_DEST work, and shadow buffers have no generic
    // state tracking, so a GPU copy into either buffer in this list would
    // need its own transition before the clamp.
    unsafe {
        // Earlier shader writes (the title's argument passes) first.
        l.ResourceBarrier(&[uav_barrier()]);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.SetComputeRootUnorderedAccessView(1, gpu);
            l.SetComputeRootUnorderedAccessView(2, budget);
            l.SetComputeRootUnorderedAccessView(3, count_gpu);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        l.ResourceBarrier(&[
            uav_barrier(),
            transition(
                buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            ),
        ]);
        if let Some(count) = count
            && !same
        {
            l.ResourceBarrier(&[transition(
                count.buffer,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }
    }
    true
}

/// After the execution: the buffer back to unordered access, where the
/// title's passes keep using it.
#[cfg(test)]
pub(crate) fn done(list: &HostList, buffer: &ID3D12Resource) {
    done_counted(list, buffer, None);
}

/// [`done`], giving the count buffer back as well; pass the same one
/// [`bound_counted`] got (and only after it returned true).
pub(crate) fn done_counted(
    list: &HostList,
    buffer: &ID3D12Resource,
    count: Option<&ID3D12Resource>,
) {
    // SAFETY: an open list and live buffers.
    unsafe {
        list.list.ResourceBarrier(&[transition(
            buffer,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        if let Some(count) = count
            && count.as_raw() != buffer.as_raw()
        {
            list.list.ResourceBarrier(&[transition(
                count,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::host::HostList;

    /// PC argument numbering for the clamp's `kind`.
    const DRAW: u32 = 0;
    const DISPATCH: u32 = 2;
    const MESH: u32 = 10;

    /// Writes `words` to the mapped host-visible buffer at byte `at`.
    fn put(data: *mut u8, at: usize, words: &[u32]) {
        // SAFETY: the caller maps at least `at + words.len() * 4` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                words.as_ptr().cast::<u8>(),
                data.add(at),
                words.len() * 4,
            );
        }
    }

    /// Reads `n` dwords from the mapped host-visible buffer at byte `at`.
    fn get(data: *mut u8, at: usize, n: usize) -> Vec<u32> {
        let mut out = vec![0u32; n];
        // SAFETY: the caller maps at least `at + n * 4` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(data.add(at), out.as_mut_ptr().cast::<u8>(), n * 4);
        }
        out
    }

    fn submit(host: &crate::host::Host, list: &HostList) {
        list.close();
        host.execute(&[list]);
    }

    fn built() {
        if super::clamp().is_none() {
            let error = super::make().err();
            panic!("the argument clamp did not build: {error:?}");
        }
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        built();
    }

    /// Legal mesh dispatches keep their counts; a malicious one is capped.
    #[test]
    fn clamp_preserves_legal_mesh_args_and_bounds_malicious_ones() {
        let Some(host) = super::host() else {
            return;
        };
        built();
        let (stride, records) = (12u32, 4u32);
        let (buffer, data) = host
            .shared_buffer(u64::from(stride * records))
            .expect("argument buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        put(data, 0, &[3, 1, 1]);
        put(data, stride as usize, &[u32::MAX; 3]);
        put(data, 2 * stride as usize, &[2, 2, 2]);
        put(data, 3 * stride as usize, &[1, 0, 1]);
        let list = host.new_list().expect("host list");
        assert!(super::bound(&list, &buffer, gpu, (MESH, stride, records)));
        super::done(&list, &buffer);
        submit(host, &list);
        assert_eq!(get(data, 0, 3), vec![3, 1, 1]);
        assert_eq!(get(data, stride as usize, 3), vec![65535, 1, 1]);
        assert_eq!(get(data, 2 * stride as usize, 3), vec![2, 1, 1]);
        assert_eq!(get(data, 3 * stride as usize, 3), vec![1, 0, 1]);
    }

    /// Draw counts are capped to 2^24 vertices and 2^26 vertex instances.
    #[test]
    fn clamp_bounds_draw_counts_and_instances() {
        let Some(host) = super::host() else {
            return;
        };
        built();
        let (stride, records) = (16u32, 2u32);
        let (buffer, data) = host
            .shared_buffer(u64::from(stride * records))
            .expect("argument buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        put(data, 0, &[1000, 2, 0, 0]);
        put(data, stride as usize, &[u32::MAX, u32::MAX, 0, 0]);
        let list = host.new_list().expect("host list");
        assert!(super::bound(&list, &buffer, gpu, (DRAW, stride, records)));
        super::done(&list, &buffer);
        submit(host, &list);
        assert_eq!(get(data, 0, 4), vec![1000, 2, 0, 0]);
        assert_eq!(get(data, stride as usize, 4), vec![1u32 << 24, 4, 0, 0]);
    }

    /// The clamp reads ExecuteIndirect's GPU count: a cheap counted record
    /// is clamped, the expensive records past the count are left alone and
    /// do not starve a later execution's budget.
    #[test]
    fn clamp_charges_only_the_records_the_count_buffer_names() {
        let Some(host) = super::host() else {
            return;
        };
        built();
        let (stride, records) = (12u32, 4u32);
        let (args, args_data) = host
            .shared_buffer(u64::from(stride * records))
            .expect("argument buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let args_gpu = unsafe { args.GetGPUVirtualAddress() };
        put(args_data, 0, &[1, 1, 1]);
        for i in 1..records as usize {
            put(args_data, i * stride as usize, &[65535, 16, 1]);
        }
        let (count, count_data) = host.shared_buffer(4).expect("count buffer");
        put(count_data, 0, &[1]);
        let (second, second_data) = host.shared_buffer(12).expect("second argument buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let second_gpu = unsafe { second.GetGPUVirtualAddress() };
        put(second_data, 0, &[65535, 16, 1]);
        let list = host.new_list().expect("host list");
        assert!(super::bound_counted(
            &list,
            &args,
            args_gpu,
            (DISPATCH, stride, records),
            Some(super::Count {
                buffer: &count,
                offset: 0,
            }),
        ));
        super::done_counted(&list, &args, Some(&count));
        assert!(super::bound(
            &list,
            &second,
            second_gpu,
            (DISPATCH, stride, 1)
        ));
        super::done(&list, &second);
        submit(host, &list);
        // Only record 0 was counted; the rest were never work and stay as
        // they were (ExecuteIndirect will not read them).
        assert_eq!(get(args_data, 0, 3), vec![1, 1, 1]);
        for i in 1..records as usize {
            assert_eq!(get(args_data, i * stride as usize, 3), vec![65535, 16, 1]);
        }
        // Had the three uncounted records been charged (201M), this one
        // would have been zeroed by the budget.
        assert_eq!(get(second_data, 0, 3), vec![65535, 16, 1]);
    }

    /// The budget is per list and accumulates in execution order; the third
    /// maximal dispatch overruns it and is zeroed. The same range clamped
    /// twice is charged twice, deliberately (see the module docs).
    #[test]
    fn clamp_budget_zeroes_records_past_its_limit() {
        let Some(host) = super::host() else {
            return;
        };
        built();
        let (stride, records) = (12u32, 1u32);
        let (buffer, data) = host
            .shared_buffer(u64::from(stride * 3))
            .expect("argument buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        for i in 0..3usize {
            put(data, i * stride as usize, &[65535, 16, 1]);
        }
        let list = host.new_list().expect("host list");
        // Each dispatch costs 65535 * 16 * 64 = 67,107,840; two fit the
        // 2^27 budget, the third does not.
        for i in 0..3u64 {
            assert!(super::bound(
                &list,
                &buffer,
                gpu + i * u64::from(stride),
                (DISPATCH, stride, records),
            ));
            super::done(&list, &buffer);
        }
        submit(host, &list);
        assert_eq!(get(data, 0, 3), vec![65535, 16, 1]);
        assert_eq!(get(data, stride as usize, 3), vec![65535, 16, 1]);
        assert_eq!(get(data, 2 * stride as usize, 3), vec![0, 16, 1]);
    }

    /// Zero records record no Dispatch and leave the arguments untouched.
    #[test]
    fn clamp_with_zero_records_leaves_the_arguments() {
        let Some(host) = super::host() else {
            return;
        };
        built();
        let (buffer, data) = host.shared_buffer(12).expect("argument buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        put(data, 0, &[7, 8, 9]);
        let list = host.new_list().expect("host list");
        assert!(super::bound(&list, &buffer, gpu, (MESH, 12, 0)));
        super::done(&list, &buffer);
        submit(host, &list);
        assert_eq!(get(data, 0, 3), vec![7, 8, 9]);
    }

    /// A max that needs more than 65,535 groups is rejected before any GPU
    /// work, rather than recording an illegal Dispatch.
    #[test]
    fn clamp_rejects_more_records_than_one_dispatch_can_carry() {
        let Some(host) = super::host() else {
            return;
        };
        let (buffer, _data) = host.shared_buffer(12).expect("argument buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let list = host.new_list().expect("host list");
        assert!(!super::bound(
            &list,
            &buffer,
            gpu,
            (MESH, 12, 64 * 65_535 + 1)
        ));
        list.close();
    }

    /// The count buffer is read one dword at its offset: reject an
    /// unaligned offset and one that does not fit the host copy.
    #[test]
    fn clamp_rejects_a_bad_count_offset() {
        let Some(host) = super::host() else {
            return;
        };
        let (args, _data) = host.shared_buffer(12).expect("argument buffer");
        let (count, _count_data) = host.shared_buffer(4).expect("count buffer");
        // SAFETY: a live buffer; the host copy is GPU-visible.
        let gpu = unsafe { args.GetGPUVirtualAddress() };
        let list = host.new_list().expect("host list");
        assert!(!super::bound_counted(
            &list,
            &args,
            gpu,
            (MESH, 12, 1),
            Some(super::Count {
                buffer: &count,
                offset: 2,
            }),
        ));
        assert!(!super::bound_counted(
            &list,
            &args,
            gpu,
            (MESH, 12, 1),
            Some(super::Count {
                buffer: &count,
                offset: 4,
            }),
        ));
        list.close();
    }
}
