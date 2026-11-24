//! A reproduction outside the title (diagnostics, `VICTORY_PROBE_INDIRECT=1`):
//! one triangle covering an 8x8 target, drawn directly and through
//! `ExecuteIndirect`, with and without a geometry shader in the pipeline,
//! the record in each kind of buffer the runtime keeps one in, with and
//! without the argument clamp before it. Each case logs how many of the 64
//! pixels it covered. The title's tiled lighting is a non-indexed indirect
//! draw through a geometry-shader pipeline with the clamp before it; this
//! tells which of those a host does not run.

use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::w;

use crate::host::{host, transition};

const SIDE: u32 = 8;
const VS: &str = "float4 main(uint id : SV_VertexID) : SV_Position {\n\
    return float4((id & 1) ? 3.0 : -1.0, (id & 2) ? 3.0 : -1.0, 0.5, 1.0); }\n";
const GS: &str = "struct V { float4 p : SV_Position; };\n\
    [maxvertexcount(3)] void main(triangle V i[3], inout TriangleStream<V> o) {\n\
    o.Append(i[0]); o.Append(i[1]); o.Append(i[2]); }\n";
const PS: &str = "float4 main() : SV_Target { return float4(1, 1, 1, 1); }\n";

/// How a case issues its draw.
#[derive(Clone, Copy, Debug)]
enum Case {
    Direct,
    /// Non-indexed, the record in a CPU-mapped buffer (as host copies of
    /// title memory are).
    Mapped,
    /// The same after the argument clamp ran over the record.
    MappedClamped,
    /// Non-indexed, the record in a plain GPU buffer filled by a copy.
    Plain,
    /// Indexed over the numbers 0, 1, 2, the record in a CPU-mapped buffer.
    Indexed,
}

/// Runs every case once and logs what each covered.
pub(crate) fn indirect() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("VICTORY_PROBE_INDIRECT").is_none() {
            return;
        }
        if run().is_none() {
            tracing::warn!("indirect probe: no test objects");
        }
    });
}

fn compiled(source: &str, profile: windows::core::PCWSTR) -> Option<Vec<u8>> {
    let mut bytes = crate::indirect::dxc_compile_as(source, profile)
        .inspect_err(|e| tracing::warn!(error = %e, "indirect probe: shader not compiled"))
        .ok()?;
    if bytes.len() > 20 && bytes[4..20].iter().all(|&b| b == 0) {
        victory_shader::emit::sign(&mut bytes);
    }
    Some(bytes)
}

fn run() -> Option<()> {
    let host = host()?;
    let device = &host.device;
    let (vs, gs, ps) = (compiled(VS, w!("vs_6_0"))?, compiled(GS, w!("gs_6_0"))?, compiled(PS, w!("ps_6_0"))?);
    // SAFETY: plain D3D12 object creation with descriptions that outlive
    // the calls; every list is run to its end before its objects go.
    unsafe {
        let desc = D3D12_ROOT_SIGNATURE_DESC::default();
        let mut blob = None;
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None).ok()?;
        let blob = blob?;
        let root: ID3D12RootSignature = device
            .CreateRootSignature(
                0,
                std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()),
            )
            .ok()?;
        let pipeline = |geometry: bool| -> Option<ID3D12PipelineState> {
            let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
                pShaderBytecode: b.as_ptr().cast(),
                BytecodeLength: b.len(),
            };
            let mut d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                SampleMask: u32::MAX,
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            if geometry {
                d.GS = code(&gs);
            }
            d.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            d.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8;
            d.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            d.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            device
                .CreateGraphicsPipelineState(&d)
                .inspect_err(|e| tracing::warn!(error = %e, geometry, "indirect probe: pipeline refused"))
                .ok()
        };
        let buffer = |kind, bytes: u64, state, flags| -> Option<ID3D12Resource> {
            let d = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: bytes,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: flags,
                ..Default::default()
            };
            let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
            let mut made: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &d, state, None, &mut made)
                .ok()?;
            made
        };
        // The target, its view and where it is read back.
        let target_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(SIDE),
            Height: SIDE,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            ..Default::default()
        };
        let mut target: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &target_desc,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                None,
                &mut target,
            )
            .ok()?;
        let target = target?;
        let views: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .ok()?;
        let view = views.GetCPUDescriptorHandleForHeapStart();
        device.CreateRenderTargetView(&target, None, view);
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total = 0;
        device.GetCopyableFootprints(&target_desc, 0, 1, 0, Some(&mut footprint), None, None, Some(&mut total));
        // The records: a draw of 3 vertices, 1 instance.
        let (mapped, mapped_at) = host.shared_buffer(256)?;
        let (indices, indices_at) = host.shared_buffer(256)?;
        indices_at.cast::<[u32; 3]>().write_unaligned([0, 1, 2]);
        let plain = buffer(D3D12_HEAP_TYPE_DEFAULT, 256, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE)?;
        let upload = buffer(D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE)?;
        let mut at = std::ptr::null_mut();
        upload.Map(0, None, Some(&mut at)).ok()?;
        at.cast::<[u32; 5]>().write_unaligned([3, 1, 0, 0, 0]);
        upload.Unmap(0, None);
        let filled = host.new_list().ok()?;
        filled.list.CopyBufferRegion(&plain, 0, &upload, 0, 256);
        filled.list.ResourceBarrier(&[transition(
            &plain,
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        )]);
        filled.close();
        host.execute(&[&filled]);
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        let (draw, indexed) = (crate::draw::host_signature(DRAW, 16)?, crate::draw::host_signature(INDEXED, 20)?);
        for geometry in [false, true] {
            let Some(pso) = pipeline(geometry) else { continue };
            for case in [Case::Direct, Case::Mapped, Case::MappedClamped, Case::Plain, Case::Indexed] {
                mapped_at.cast::<[u32; 5]>().write_unaligned([3, 1, 0, 0, 0]);
                let back = buffer(D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE)?;
                let list = host.new_list().ok()?;
                let l = &list.list;
                let (common, argument) = (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                // The clamp binds a compute pipeline of its own: first.
                let clamped = matches!(case, Case::MappedClamped)
                    && crate::indirect::bound_counted(&list, &mapped, mapped.GetGPUVirtualAddress(), (DRAW, 16, 1), None);
                l.ResourceBarrier(&[transition(
                    &target,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                )]);
                l.SetGraphicsRootSignature(&root);
                l.SetPipelineState(&pso);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: SIDE as f32,
                    Height: SIDE as f32,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                l.RSSetScissorRects(&[RECT { left: 0, top: 0, right: SIDE as i32, bottom: SIDE as i32 }]);
                l.OMSetRenderTargets(1, Some(&view), false, None);
                l.ClearRenderTargetView(view, &[0.0; 4], None);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                match case {
                    Case::Direct => l.DrawInstanced(3, 1, 0, 0),
                    Case::Mapped => {
                        l.ResourceBarrier(&[transition(&mapped, common, argument)]);
                        l.ExecuteIndirect(&draw, 1, &mapped, 0, None::<&ID3D12Resource>, 0);
                        l.ResourceBarrier(&[transition(&mapped, argument, common)]);
                    }
                    Case::MappedClamped => {
                        if clamped {
                            l.ExecuteIndirect(&draw, 1, &mapped, 0, None::<&ID3D12Resource>, 0);
                            crate::indirect::done_counted(&list, &mapped, None);
                        }
                    }
                    Case::Plain => l.ExecuteIndirect(&draw, 1, &plain, 0, None::<&ID3D12Resource>, 0),
                    Case::Indexed => {
                        l.IASetIndexBuffer(Some(&D3D12_INDEX_BUFFER_VIEW {
                            BufferLocation: indices.GetGPUVirtualAddress(),
                            SizeInBytes: 12,
                            Format: DXGI_FORMAT_R32_UINT,
                        }));
                        l.ResourceBarrier(&[transition(&mapped, common, argument)]);
                        l.ExecuteIndirect(&indexed, 1, &mapped, 0, None::<&ID3D12Resource>, 0);
                        l.ResourceBarrier(&[transition(&mapped, argument, common)]);
                    }
                }
                l.ResourceBarrier(&[transition(
                    &target,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                l.CopyTextureRegion(
                    &D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&back),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: footprint },
                    },
                    0,
                    0,
                    0,
                    &D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&target),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                    },
                    None,
                );
                list.close();
                host.execute(&[&list]);
                let mut data = std::ptr::null_mut();
                back.Map(0, None, Some(&mut data)).ok()?;
                let pitch = footprint.Footprint.RowPitch as usize;
                let covered = (0..SIDE as usize)
                    .flat_map(|y| (0..SIDE as usize).map(move |x| (x, y)))
                    .filter(|&(x, y)| *data.cast::<u8>().add(y * pitch + x * 4) == 255)
                    .count();
                back.Unmap(0, None);
                // The record as the CPU sees the mapped buffer afterwards.
                let record = mapped_at.cast::<[u32; 5]>().read_unaligned();
                tracing::info!(geometry, ?case, covered, of = SIDE * SIDE, clamped, ?record, "indirect probe");
            }
        }
    }
    Some(())
}
