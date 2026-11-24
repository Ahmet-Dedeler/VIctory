//! Host pipelines for title pipelines (decision 0007, step 3): the
//! title's graphics state with the translated shaders, under a root
//! signature of our own that the translated shaders' bindings define
//! (`victory_shader::emit`): per stage one block of root constants for the
//! user data, root views for the title's root views, static samplers for
//! the immediate samplers, and directly indexed heaps.

use std::ffi::{CStr, CString};
use std::sync::Arc;

use victory_shader::emit::{ROOT_CBV_SPACE, SAMPLER_SPACE, STRIDE_SPACE, USER_DATA_SPACE};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::{Shader, Translated};
use crate::host::host;

/// A stream's graphics state, owned (the title's stream lives only for the
/// creation call). Subobjects the stream leaves out keep D3D12's defaults.
pub(crate) struct GraphicsState {
    blend: D3D12_BLEND_DESC,
    sample_mask: u32,
    rasterizer: D3D12_RASTERIZER_DESC,
    depth: D3D12_DEPTH_STENCIL_DESC,
    /// Input elements, their names pointing into `names`.
    inputs: Vec<D3D12_INPUT_ELEMENT_DESC>,
    names: Vec<CString>,
    strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE,
    pub(crate) topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE,
    pub(crate) targets: D3D12_RT_FORMAT_ARRAY,
    depth_format: DXGI_FORMAT,
    samples: DXGI_SAMPLE_DESC,
}

impl Default for GraphicsState {
    fn default() -> Self {
        let target = D3D12_RENDER_TARGET_BLEND_DESC {
            SrcBlend: D3D12_BLEND_ONE,
            DestBlend: D3D12_BLEND_ZERO,
            BlendOp: D3D12_BLEND_OP_ADD,
            SrcBlendAlpha: D3D12_BLEND_ONE,
            DestBlendAlpha: D3D12_BLEND_ZERO,
            BlendOpAlpha: D3D12_BLEND_OP_ADD,
            LogicOp: D3D12_LOGIC_OP_NOOP,
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            ..Default::default()
        };
        let keep = D3D12_DEPTH_STENCILOP_DESC {
            StencilFailOp: D3D12_STENCIL_OP_KEEP,
            StencilDepthFailOp: D3D12_STENCIL_OP_KEEP,
            StencilPassOp: D3D12_STENCIL_OP_KEEP,
            StencilFunc: D3D12_COMPARISON_FUNC_ALWAYS,
        };
        GraphicsState {
            blend: D3D12_BLEND_DESC {
                RenderTarget: [target; 8],
                ..Default::default()
            },
            sample_mask: u32::MAX,
            rasterizer: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_BACK,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
            depth: D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS,
                StencilReadMask: 0xff,
                StencilWriteMask: 0xff,
                FrontFace: keep,
                BackFace: keep,
                ..Default::default()
            },
            inputs: Vec::new(),
            names: Vec::new(),
            strip_cut: D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
            topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
            targets: D3D12_RT_FORMAT_ARRAY::default(),
            depth_format: DXGI_FORMAT_UNKNOWN,
            samples: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        }
    }
}

impl GraphicsState {
    /// Takes in one subobject of a stream: its type, then its data at +4
    /// (+8 for those holding pointers).
    ///
    /// # Safety
    /// `at` points at a whole subobject of type `kind`.
    pub(crate) unsafe fn read(&mut self, kind: u32, at: *const u8) {
        // SAFETY: the caller passes a whole subobject.
        unsafe {
            let data = at.add(4);
            match kind {
                8 => self.blend = data.cast::<D3D12_BLEND_DESC>().read_unaligned(),
                9 => self.sample_mask = data.cast::<u32>().read_unaligned(),
                10 => self.rasterizer = data.cast::<D3D12_RASTERIZER_DESC>().read_unaligned(),
                11 => self.depth = data.cast::<D3D12_DEPTH_STENCIL_DESC>().read_unaligned(),
                21 => {
                    // shortcut: the depth bounds test is dropped (the host
                    // pipeline uses the plain description); bring it back
                    // with a stream if a title enables it.
                    let d = data.cast::<D3D12_DEPTH_STENCIL_DESC1>().read_unaligned();
                    self.depth = D3D12_DEPTH_STENCIL_DESC {
                        DepthEnable: d.DepthEnable,
                        DepthWriteMask: d.DepthWriteMask,
                        DepthFunc: d.DepthFunc,
                        StencilEnable: d.StencilEnable,
                        StencilReadMask: d.StencilReadMask,
                        StencilWriteMask: d.StencilWriteMask,
                        FrontFace: d.FrontFace,
                        BackFace: d.BackFace,
                    };
                }
                12 => {
                    let layout = at.add(8).cast::<D3D12_INPUT_LAYOUT_DESC>().read_unaligned();
                    if layout.pInputElementDescs.is_null() {
                        return;
                    }
                    let elements = std::slice::from_raw_parts(
                        layout.pInputElementDescs,
                        layout.NumElements as usize,
                    );
                    self.names = elements
                        .iter()
                        .map(|e| CStr::from_ptr(e.SemanticName.0.cast()).to_owned())
                        .collect();
                    self.inputs = elements
                        .iter()
                        .zip(&self.names)
                        .map(|(e, name)| D3D12_INPUT_ELEMENT_DESC {
                            SemanticName: PCSTR(name.as_ptr().cast()),
                            Format: DXGI_FORMAT(
                                crate::resource::pc_format(e.Format.0 as u32) as i32
                            ),
                            ..*e
                        })
                        .collect();
                }
                13 => {
                    self.strip_cut =
                        D3D12_INDEX_BUFFER_STRIP_CUT_VALUE(data.cast::<i32>().read_unaligned())
                }
                14 => {
                    self.topology =
                        D3D12_PRIMITIVE_TOPOLOGY_TYPE(data.cast::<i32>().read_unaligned())
                }
                15 => {
                    self.targets = data.cast::<D3D12_RT_FORMAT_ARRAY>().read_unaligned();
                    for f in &mut self.targets.RTFormats {
                        f.0 = crate::resource::pc_format(f.0 as u32) as i32;
                    }
                }
                16 => {
                    let f = data.cast::<u32>().read_unaligned();
                    self.depth_format = DXGI_FORMAT(crate::resource::pc_format(f) as i32);
                }
                17 => self.samples = data.cast::<DXGI_SAMPLE_DESC>().read_unaligned(),
                _ => {}
            }
        }
    }

    /// Everything the state holds, as text: part of a pipeline's name in
    /// the library and of the key a title pipeline is started early by.
    pub(crate) fn describe(&self) -> String {
        let inputs: Vec<String> = self
            .inputs
            .iter()
            .zip(&self.names)
            .map(|(e, name)| {
                format!(
                    "{name:?} {} {} {} {} {} {}",
                    e.SemanticIndex,
                    e.Format.0,
                    e.InputSlot,
                    e.AlignedByteOffset,
                    e.InputSlotClass.0,
                    e.InstanceDataStepRate
                )
            })
            .collect();
        format!(
            "{:?} {} {:?} {:?} {inputs:?} {} {} {:?} {} {:?}",
            self.blend,
            self.sample_mask,
            self.rasterizer,
            self.depth,
            self.strip_cut.0,
            self.topology.0,
            self.targets,
            self.depth_format.0,
            self.samples
        )
    }

    /// Depth, stencil and the first target's blend in short (frame account).
    pub(crate) fn short(&self) -> String {
        let d = &self.depth;
        let b = &self.blend.RenderTarget[0];
        format!(
            "depth {}{} f{} | stencil {} r{:#x} w{:#x} front f{} p{} z{} x{} back f{} p{} | blend {} {}+{} op{} mask {:#x} | cull {}",
            if d.DepthEnable.as_bool() { "test" } else { "off" },
            if d.DepthWriteMask.0 != 0 { "+write" } else { "" },
            d.DepthFunc.0,
            if d.StencilEnable.as_bool() { "on" } else { "off" },
            d.StencilReadMask,
            d.StencilWriteMask,
            d.FrontFace.StencilFunc.0,
            d.FrontFace.StencilPassOp.0,
            d.FrontFace.StencilDepthFailOp.0,
            d.FrontFace.StencilFailOp.0,
            d.BackFace.StencilFunc.0,
            d.BackFace.StencilPassOp.0,
            if b.BlendEnable.as_bool() { "on" } else { "off" },
            b.SrcBlend.0,
            b.DestBlend.0,
            b.BlendOp.0,
            b.RenderTargetWriteMask,
            self.rasterizer.CullMode.0,
        )
    }

    /// The vertex buffer slots the input layout reads.
    fn slots(&self) -> u32 {
        self.inputs
            .iter()
            .fold(0, |m, e| m | 1 << e.InputSlot.min(31))
    }

    /// The input elements with their byte offsets worked out.
    fn elements(&self) -> Vec<Input> {
        let mut next = [0u32; 32];
        self.inputs
            .iter()
            .map(|e| {
                let slot = e.InputSlot.min(31) as usize;
                let format = e.Format.0 as u32;
                let offset = if e.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT {
                    next[slot]
                } else {
                    e.AlignedByteOffset
                };
                let bytes = crate::resource::element(format).map_or(0, |(b, _)| b as u32);
                next[slot] = offset + bytes;
                Input {
                    slot: slot as u32,
                    offset,
                    format,
                    per_instance: e.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                }
            })
            .collect()
    }
}

/// One input element: vertex buffer slot, byte offset, format.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Input {
    pub(crate) slot: u32,
    pub(crate) offset: u32,
    pub(crate) format: u32,
    pub(crate) per_instance: bool,
}

/// The console's own primitive topology types.
const RECTANGLES: i32 = 19;
const QUADS: i32 = 32;

/// One stage of a host pipeline and where its bindings go.
pub(crate) struct Stage {
    pub(crate) shader: Arc<Shader>,
    /// The root parameter of its user data, and its size in dwords.
    pub(crate) user_data: Option<(u32, u32)>,
    /// Root parameters of its root views: (title root parameter, ours).
    pub(crate) root_cbvs: Vec<(u8, u32)>,
    pub(crate) root_srvs: Vec<(u8, u32)>,
    /// Root parameter of the runtime stride table (`t0`, [`STRIDE_SPACE`]).
    pub(crate) stride_table: Option<u32>,
    pub(crate) root_uavs: Vec<(u8, u32)>,
}

impl Stage {
    pub(crate) fn translated(&self) -> &Translated {
        self.shader
            .translated()
            .expect("host pipelines are made from translated shaders")
    }
}

pub(crate) struct HostPipeline {
    /// A compute pipeline (else graphics).
    pub(crate) compute: bool,
    /// A mesh pipeline (graphics without vertex input).
    pub(crate) mesh: bool,
    pub(crate) pso: ID3D12PipelineState,
    /// For a rectangle-list pipeline that makes its corners from the vertex
    /// id: the same pipeline with the geometry shader that adds each
    /// rectangle's fourth corner (see `crate::rects`).
    pub(crate) rects: Option<ID3D12PipelineState>,
    pub(crate) root: ID3D12RootSignature,
    /// Vertex, then pixel stage, or the compute stage.
    pub(crate) stages: Vec<Stage>,
    /// The vertex buffer slots it reads (bit per slot).
    pub(crate) slots: u32,
    pub(crate) inputs: Vec<Input>,
    /// Its shaders, for reports of slow submissions.
    pub(crate) name: String,
    /// Its depth, stencil and blend state in short, for the frame account.
    pub(crate) state: String,
}

/// Logs a pipeline that gets no host pipeline, once per reason.
fn refuse(reason: &'static str, shaders: &[Arc<Shader>]) -> Option<HostPipeline> {
    static SEEN: std::sync::Mutex<Vec<&str>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&reason) {
        seen.push(reason);
        tracing::warn!(?shaders, "no host pipeline: {reason} (logged once)");
    }
    None
}

impl HostPipeline {
    pub(crate) fn has_root_uavs(&self) -> bool {
        self.stages.iter().any(|s| !s.root_uavs.is_empty())
    }

    /// A host pipeline for vertex and pixel shaders with the graphics
    /// state `g`, or for one compute shader.
    pub(crate) fn new(shaders: &[Arc<Shader>], g: Option<&GraphicsState>) -> Option<HostPipeline> {
        let find = |stage| shaders.iter().find(|s| s.stage == stage);
        let compute = match shaders {
            [cs] if cs.stage == "compute" => Some(cs),
            _ => None,
        };
        if compute.is_none()
            && shaders
                .iter()
                .any(|s| !matches!(s.stage, "vertex" | "pixel" | "mesh"))
        {
            return refuse(
                "only vertex, mesh, pixel and compute stages are translated",
                shaders,
            );
        }
        let mesh = find("mesh");
        let stages = if let Some(cs) = compute {
            vec![(cs, D3D12_SHADER_VISIBILITY_ALL)]
        } else {
            let first = match (mesh, find("vertex")) {
                (Some(ms), _) => (ms, D3D12_SHADER_VISIBILITY_MESH),
                (None, Some(vs)) => (vs, D3D12_SHADER_VISIBILITY_VERTEX),
                (None, None) => return refuse("no vertex or mesh shader", shaders),
            };
            let mut stages = vec![first];
            stages.extend(find("pixel").map(|ps| (ps, D3D12_SHADER_VISIBILITY_PIXEL)));
            stages
        };
        if mesh.is_some() && !host()?.mesh_shaders() {
            return refuse(
                "the host GPU has no mesh shaders (or no shader model 6.6)",
                shaders,
            );
        }
        for (s, _) in &stages {
            s.translated()?;
        }
        let host = host()?;

        // The root signature the translated shaders' bindings define.
        let mut params: Vec<D3D12_ROOT_PARAMETER1> = Vec::new();
        // Ranges of the one-view tables root SRVs and UAVs become, alive
        // until the root signature is made.
        let mut ranges: Vec<Box<D3D12_DESCRIPTOR_RANGE1>> = Vec::new();
        let mut samplers = Vec::new();
        let mut bound = Vec::new();
        let mut root_views: Vec<((i32, u8), u32)> = Vec::new();
        for &(shader, visibility) in &stages {
            let t = &shader.translated()?.host;
            let mut stage = Stage {
                shader: shader.clone(),
                user_data: None,
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                stride_table: None,
                root_uavs: Vec::new(),
            };
            let dwords = t.user_data_dwords();
            if dwords > 0 {
                stage.user_data = Some((params.len() as u32, dwords));
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: USER_DATA_SPACE,
                            Num32BitValues: dwords,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (slots, kind, out) in [
                (
                    &t.root_cbvs,
                    D3D12_ROOT_PARAMETER_TYPE_CBV,
                    &mut stage.root_cbvs,
                ),
                (
                    &t.root_srvs,
                    D3D12_ROOT_PARAMETER_TYPE_SRV,
                    &mut stage.root_srvs,
                ),
                (
                    &t.root_uavs,
                    D3D12_ROOT_PARAMETER_TYPE_UAV,
                    &mut stage.root_uavs,
                ),
            ] {
                for &slot in slots {
                    let param = push_root_view(
                        &mut params,
                        &mut ranges,
                        &mut root_views,
                        kind,
                        slot,
                        visibility,
                    );
                    if !out.contains(&(slot, param)) {
                        out.push((slot, param));
                    }
                }
            }
            if t.stride_table {
                stage.stride_table = Some(params.len() as u32);
                params.push(D3D12_ROOT_PARAMETER1 {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_SRV,
                    Anonymous: D3D12_ROOT_PARAMETER1_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                            ShaderRegister: 0,
                            RegisterSpace: STRIDE_SPACE,
                            Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                        },
                    },
                    ShaderVisibility: visibility,
                });
            }
            for (n, s) in t.static_samplers.iter().enumerate() {
                samplers.push(D3D12_STATIC_SAMPLER_DESC {
                    Filter: D3D12_FILTER(s.filter as i32),
                    AddressU: D3D12_TEXTURE_ADDRESS_MODE(s.address[0] as i32),
                    AddressV: D3D12_TEXTURE_ADDRESS_MODE(s.address[1] as i32),
                    AddressW: D3D12_TEXTURE_ADDRESS_MODE(s.address[2] as i32),
                    MipLODBias: s.mip_lod_bias,
                    MaxAnisotropy: s.max_anisotropy,
                    ComparisonFunc: D3D12_COMPARISON_FUNC(s.comparison as i32),
                    BorderColor: D3D12_STATIC_BORDER_COLOR(s.border as i32),
                    MinLOD: s.min_lod,
                    MaxLOD: s.max_lod,
                    ShaderRegister: n as u32,
                    RegisterSpace: SAMPLER_SPACE,
                    ShaderVisibility: visibility,
                });
            }
            bound.push(stage);
        }
        let (root, root_blob) =
            match root_signature(&host.device, &params, &samplers, compute.is_some()) {
                Ok(root) => root,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host root signature refused");
                    return None;
                }
            };
        let code = |i: usize| {
            stages.get(i).and_then(|(s, _)| s.translated()).map_or(
                D3D12_SHADER_BYTECODE::default(),
                |t| D3D12_SHADER_BYTECODE {
                    pShaderBytecode: t.host.dxil.as_ptr().cast(),
                    BytecodeLength: t.host.dxil.len(),
                },
            )
        };
        if let Some(cs) = compute {
            // A program that depends on the lane layout declares the
            // console's wave size. A host of narrower waves does not run it:
            // Apple's converter (behind D3DMetal, 32-lane waves) fails
            // ("Wave size is not supported"), D3DMetal hands back a pipeline
            // all the same, and a submission that dispatches with it never
            // finishes.
            let needs = cs.translated().and_then(|t| wave_lanes(&t.host.dxil));
            if let Some(needs) = needs
                && needs > host.wave_lanes()
            {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(
                        ?shaders,
                        needs,
                        host = host.wave_lanes(),
                        "no host pipeline for compute programs of wider waves than the host's \
                         (logged once; VICTORY_SPLIT_WAVES=1 translates them for 32-lane waves)"
                    )
                });
                WIDE_WAVE_REFUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return refuse("the compute program needs wider waves than the host GPU runs", shaders);
            }
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                // SAFETY: a borrowed pointer the description never releases.
                pRootSignature: unsafe { std::mem::transmute_copy(&root) },
                CS: code(0),
                ..Default::default()
            };
            let name = super::library::name("compute", &root_blob, &[&code_bytes(&code(0))], "");
            let made = super::library::pipeline(
                &name,
                // SAFETY: the description points at live data.
                |library, name| unsafe { library.LoadComputePipeline(name, &desc) },
                // SAFETY: as above.
                || unsafe { host.device.CreateComputePipelineState(&desc) },
            );
            let pso = match made {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: true,
                mesh: false,
                pso,
                rects: None,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                state: String::new(),
                slots: 0,
                inputs: Vec::new(),
            });
        }
        let Some(g) = g else {
            return refuse("graphics state not kept (plain description)", shaders);
        };
        let state = g.describe();
        if let Some(ms) = mesh {
            let bytes = ms.translated().and_then(|t| metal_mesh_bytes(&t.host.dxil));
            if let (Some(limit), Some(bytes)) = (host.mesh_output_limit(), bytes)
                && bytes > limit
            {
                tracing::warn!(?shaders, bytes, limit, "no host pipeline: mesh outputs over the host's limit");
                return refuse(
                    "the mesh shader's outputs take more than the host GPU allows",
                    shaders,
                );
            }
            // "triangles": the stream names its topology since it did not
            // load from the library without one (see `mesh_pipeline`); a
            // new name keeps such entries, stored before, from being
            // looked up again.
            let name = super::library::name(
                "mesh triangles",
                &root_blob,
                &[&code_bytes(&code(0)), &code_bytes(&code(1))],
                &state,
            );
            let pso = match mesh_pipeline(&host.device, &root, code(0), code(1), g, &name) {
                Ok(pso) => pso,
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host mesh pipeline refused");
                    host.log_messages();
                    return None;
                }
            };
            tracing::info!(?shaders, "host mesh pipeline");
            crate::host::name(&pso, || format!("{shaders:?}"));
            return Some(HostPipeline {
                compute: false,
                mesh: true,
                pso,
                rects: None,
                root,
                stages: bound,
                name: format!("{shaders:?}"),
                state: g.short(),
                slots: 0,
                inputs: Vec::new(),
            });
        }

        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            VS: code(0),
            PS: code(1),
            BlendState: g.blend,
            SampleMask: g.sample_mask,
            RasterizerState: g.rasterizer,
            DepthStencilState: g.depth,
            InputLayout: D3D12_INPUT_LAYOUT_DESC {
                pInputElementDescs: g.inputs.as_ptr(),
                NumElements: g.inputs.len() as u32,
            },
            IBStripCutValue: g.strip_cut,
            // The console's rectangle and quad types draw as triangles on
            // the host (see `draw`).
            PrimitiveTopologyType: match g.topology.0 {
                RECTANGLES | QUADS => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                _ => g.topology,
            },
            NumRenderTargets: g.targets.NumRenderTargets,
            RTVFormats: g.targets.RTFormats,
            DSVFormat: g.depth_format,
            SampleDesc: g.samples,
            ..Default::default()
        };
        let name = super::library::name(
            "graphics",
            &root_blob,
            &[&code_bytes(&code(0)), &code_bytes(&code(1))],
            &state,
        );
        let made = super::library::pipeline(
            &name,
            // SAFETY: the description points at live data.
            |library, name| unsafe { library.LoadGraphicsPipeline(name, &desc) },
            // SAFETY: as above.
            || unsafe { host.device.CreateGraphicsPipelineState(&desc) },
        );
        let pso = match made {
            Ok(pso) => pso,
            Err(e) => {
                tracing::warn!(error = %e, ?shaders, "host pipeline refused");
                host.log_messages();
                return None;
            }
        };
        tracing::info!(?shaders, "host pipeline");
        crate::host::name(&pso, || format!("{shaders:?}"));
        tracing::debug!(
            ?shaders,
            depth = ?g.depth,
            dsv = g.depth_format.0,
            targets = ?&g.targets.RTFormats[..g.targets.NumRenderTargets.min(8) as usize],
            blend = ?g.blend.RenderTarget[0],
            cull = g.rasterizer.CullMode.0,
            "host graphics state"
        );
        // Rectangle lists: the variant whose geometry shader draws their
        // fourth corner (`crate::rects`), for the draws the CPU cannot
        // expand.
        let rects = (g.topology.0 == RECTANGLES)
            .then(|| {
                stages
                    .first()
                    .and_then(|(vs, _)| vs.translated())
                    .and_then(|t| crate::rects::geometry_shader(&t.host.dxil))
            })
            .flatten()
        .and_then(|gs| {
            // SAFETY: a second description of the same borrowed pointers
            // (neither releases them).
            let mut desc: D3D12_GRAPHICS_PIPELINE_STATE_DESC = unsafe { std::ptr::read(&desc) };
            desc.GS = D3D12_SHADER_BYTECODE {
                pShaderBytecode: gs.as_ptr().cast(),
                BytecodeLength: gs.len(),
            };
            // A rectangle has no facing: whichever way its three corners
            // wind, both triangles of the strip draw.
            desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            // Diagnostics: these draws without their depth test, to tell a
            // test that fails from a draw that covers nothing.
            #[cfg(feature = "diagnostics")]
            if std::env::var_os("VICTORY_RECTS_UNTESTED").is_some() {
                desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
            }
            // SAFETY: the description points at live data.
            match unsafe { host.device.CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc) } {
                Ok(pso) => {
                    tracing::info!(?shaders, inputs = g.inputs.len(), "host pipeline for rectangles");
                    crate::host::name(&pso, || format!("{shaders:?} rectangles"));
                    Some(pso)
                }
                Err(e) => {
                    tracing::warn!(error = %e, ?shaders, "host rectangle pipeline refused");
                    host.log_messages();
                    None
                }
            }
        });
        Some(HostPipeline {
            compute: false,
            mesh: false,
            pso,
            rects,
            root,
            stages: bound,
            name: format!("{shaders:?}"),
            state: g.short(),
            slots: g.slots(),
            inputs: g.elements(),
        })
    }
}

/// Compute pipelines refused because their program needs wider waves than
/// the host runs.
pub(crate) static WIDE_WAVE_REFUSED: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// The wave size a translated program requires (`PSV0`'s minimum wave lane
/// count), if it requires one.
pub(crate) fn wave_lanes(dxil: &[u8]) -> Option<u32> {
    let u32_at = |b: &[u8], at: usize| {
        b.get(at..at + 4)
            .map(|v| u32::from_le_bytes(v.try_into().expect("4 bytes")) as usize)
    };
    let info = (0..u32_at(dxil, 28)?).find_map(|i| {
        let at = u32_at(dxil, 32 + 4 * i)?;
        (dxil.get(at..at + 4)? == b"PSV0")
            .then(|| dxil.get(at + 8..at + 8 + u32_at(dxil, at + 4)?))
            .flatten()
    })?;
    // The runtime information's size, then the information: minimum and
    // maximum wave lane count at 16 and 20 (0 and all ones: any).
    let (min, max) = (u32_at(info, 4 + 16)? as u32, u32_at(info, 4 + 20)? as u32);
    (min != 0 && min == max).then_some(min)
}

/// The bytes Apple's Metal counts for a translated mesh shader's outputs
/// (measured on its own pipelines, 43 of this title's and synthetic ones):
/// per vertex, the clip and cull distances as floats, the position, then
/// every other output in signature order, each placed at the next multiple
/// of its own size (4 bytes a component) with nothing after the last;
/// per primitive, its outputs the same way; each times the declared
/// maximum. `None` if the container does not read as ours.
pub(crate) fn metal_mesh_bytes(dxil: &[u8]) -> Option<u32> {
    let u32_at = |b: &[u8], at: usize| {
        b.get(at..at + 4)
            .map(|v| u32::from_le_bytes(v.try_into().expect("4 bytes")) as usize)
    };
    let part = |name: &[u8; 4]| {
        (0..u32_at(dxil, 28)?).find_map(|i| {
            let at = u32_at(dxil, 32 + 4 * i)?;
            (dxil.get(at..at + 4)? == name)
                .then(|| dxil.get(at + 8..at + 8 + u32_at(dxil, at + 4)?))
                .flatten()
        })
    };
    // Signature elements: (system value, components) in order.
    let elements = |name: &[u8; 4]| -> Option<Vec<(usize, u32)>> {
        let Some(data) = part(name) else {
            return Some(Vec::new());
        };
        (0..u32_at(data, 0)?)
            .map(|i| {
                let at = 8 + 32 * i;
                let mask = *data.get(at + 24)?;
                Some((u32_at(data, at + 12)?, mask.count_ones()))
            })
            .collect()
    };
    const POSITION: usize = 1;
    const DISTANCES: [usize; 2] = [2, 3];
    let layout = |sizes: &mut dyn Iterator<Item = u32>| {
        sizes
            .filter(|&size| size > 0)
            .fold(0, |end: u32, size| end.next_multiple_of(size) + size)
    };
    let vertex = elements(b"OSG1")?;
    let distances = vertex
        .iter()
        .filter(|(kind, _)| DISTANCES.contains(kind))
        .map(|&(_, n)| n)
        .sum::<u32>();
    let vertex_bytes = layout(
        &mut std::iter::repeat_n(4, distances as usize)
            .chain([16])
            .chain(vertex.iter().filter(|e| e.0 != POSITION).map(|e| 4 * e.1)),
    );
    let primitive_bytes = layout(&mut elements(b"PSG1")?.iter().map(|e| 4 * e.1));
    // `PSV0`: the runtime information's size, then the information.
    let info = part(b"PSV0")?.get(4..)?;
    let maximum = |at: usize| Some(u32::from(u16::from_le_bytes(info.get(at..at + 2)?.try_into().ok()?)));
    Some(maximum(12)? * vertex_bytes + maximum(14)? * primitive_bytes)
}

/// A mesh pipeline from the title's graphics state: the pipeline-state
/// stream form (no input layout).
///
/// The stream names a triangle topology, which D3D12 does not need for a
/// mesh shader: without it the pipeline library refuses to load a stored
/// pipeline with the very stream it was made from (`E_INVALIDARG`; debug
/// layer message 972, "does not match the one used to create the PSO"), so
/// every mesh pipeline was compiled again, and its store then refused (970,
/// "already exists"). The translation makes triangle mesh shaders only
/// (`mesh_prologue` refuses others), and the debug layer wants the two to
/// agree (1323).
fn mesh_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    ms: D3D12_SHADER_BYTECODE,
    ps: D3D12_SHADER_BYTECODE,
    g: &GraphicsState,
    name: &str,
) -> windows::core::Result<ID3D12PipelineState> {
    use windows::core::Interface;
    /// A stream subobject: its type, then its description, pointer-aligned.
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        blend: Sub<D3D12_BLEND_DESC>,
        mask: Sub<u32>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
        topology: Sub<D3D12_PRIMITIVE_TOPOLOGY_TYPE>,
    }
    let mut stream = Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps),
        blend: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, g.blend),
        mask: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
            g.sample_mask,
        ),
        raster: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, g.rasterizer),
        depth: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, g.depth),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            g.targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            g.depth_format,
        ),
        samples: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, g.samples),
        topology: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
            D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        ),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device = device.cast::<ID3D12Device2>()?;
    super::library::pipeline(
        name,
        // SAFETY: the stream holds whole subobjects pointing at live data.
        |library, name| unsafe { library.LoadPipeline(name, &desc) },
        // SAFETY: as above.
        || unsafe { device.CreatePipelineState(&desc) },
    )
}

/// The bytes of shader code (none for an absent stage).
fn code_bytes(code: &D3D12_SHADER_BYTECODE) -> Vec<u8> {
    if code.pShaderBytecode.is_null() {
        return Vec::new();
    }
    // SAFETY: a translation's own bytes, alive while its shader is.
    unsafe { std::slice::from_raw_parts(code.pShaderBytecode.cast::<u8>(), code.BytecodeLength) }
        .to_vec()
}

/// Whether root SRVs and UAVs are bound as one-view tables of the buffer's
/// host copy (reads past its end return zeros, writes past it are dropped,
/// as the console's buffer descriptors bound them) rather than root
/// descriptors, which nothing bounds: a shader that runs past the host copy
/// reaches unmapped or freed host memory and the GPU faults (#493 for root
/// SRVs; root UAVs fault the same way on writes). `VICTORY_ROOT_VIEW_DIRECT=1`
/// goes back to root descriptors.
pub(crate) fn bounded_root_views() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ROOT_VIEW_DIRECT").is_none())
}

/// Adds or reuses a root view parameter for `(kind, slot)`, widening its
/// visibility to `ALL` when shared across multiple shader stages so shared
/// root descriptors do not consume extra root signature dwords. With
/// [`bounded_root_views`], an SRV or UAV becomes a one-view table whose
/// range lives in `ranges` (kept until the root signature is made).
fn push_root_view(
    params: &mut Vec<D3D12_ROOT_PARAMETER1>,
    ranges: &mut Vec<Box<D3D12_DESCRIPTOR_RANGE1>>,
    seen: &mut Vec<((i32, u8), u32)>,
    kind: D3D12_ROOT_PARAMETER_TYPE,
    slot: u8,
    visibility: D3D12_SHADER_VISIBILITY,
) -> u32 {
    if let Some(&(_, existing)) = seen.iter().find(|&&(k, _)| k == (kind.0, slot)) {
        if params[existing as usize].ShaderVisibility != visibility {
            params[existing as usize].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        return existing;
    }
    let index = params.len() as u32;
    seen.push(((kind.0, slot), index));
    let table = match kind {
        D3D12_ROOT_PARAMETER_TYPE_SRV if bounded_root_views() => Some(D3D12_DESCRIPTOR_RANGE_TYPE_SRV),
        D3D12_ROOT_PARAMETER_TYPE_UAV if bounded_root_views() => Some(D3D12_DESCRIPTOR_RANGE_TYPE_UAV),
        _ => None,
    };
    if let Some(range_type) = table {
        ranges.push(Box::new(D3D12_DESCRIPTOR_RANGE1 {
            RangeType: range_type,
            NumDescriptors: 1,
            BaseShaderRegister: u32::from(slot),
            RegisterSpace: ROOT_CBV_SPACE,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE
                | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        }));
        // The box does not move when `ranges` grows.
        let range: *const D3D12_DESCRIPTOR_RANGE1 = &**ranges.last().expect("just pushed");
        params.push(D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: range,
                },
            },
            ShaderVisibility: visibility,
        });
        return index;
    }
    params.push(D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: u32::from(slot),
                RegisterSpace: ROOT_CBV_SPACE,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
            },
        },
        ShaderVisibility: visibility,
    });
    index
}

fn root_signature(
    device: &ID3D12Device,
    params: &[D3D12_ROOT_PARAMETER1],
    samplers: &[D3D12_STATIC_SAMPLER_DESC],
    compute: bool,
) -> windows::core::Result<(ID3D12RootSignature, Vec<u8>)> {
    let input = if compute {
        D3D12_ROOT_SIGNATURE_FLAG_NONE
    } else {
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
    };
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: windows::Win32::Graphics::Direct3D12::D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: samplers.len() as u32,
                pStaticSamplers: samplers.as_ptr(),
                Flags: input
                    | D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED
                    | D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    // SAFETY: the description points at live arrays; the blob is read in
    // full before it is released.
    unsafe {
        let mut blob: Option<ID3DBlob> = None;
        let mut error: Option<ID3DBlob> = None;
        if let Err(e) = D3D12SerializeVersionedRootSignature(&desc, &mut blob, Some(&mut error)) {
            if let Some(error) = error {
                let text = std::slice::from_raw_parts(
                    error.GetBufferPointer().cast::<u8>(),
                    error.GetBufferSize(),
                );
                tracing::warn!("root signature: {}", String::from_utf8_lossy(text));
            }
            return Err(e);
        }
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        Ok((device.CreateRootSignature(0, bytes)?, bytes.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::Ordering;

    use super::*;

    /// A triangle mesh shader of our own and a pixel shader for it.
    const MS: &str = "struct V { float4 p : SV_Position; };
        [outputtopology(\"triangle\")] [numthreads(1, 1, 1)]
        void main(out vertices V v[3], out indices uint3 t[1]) {
            SetMeshOutputCounts(3, 1);
            v[0].p = float4(0, 0, 0, 1);
            v[1].p = float4(1, 0, 0, 1);
            v[2].p = float4(0, 1, 0, 1);
            t[0] = uint3(0, 1, 2);
        }";
    const PS: &str = "float4 main() : SV_Target { return 1; }";

    /// The PC `dxc.exe` (`VICTORY_PC_DXC`, else the newest Windows SDK's).
    fn dxc() -> Option<PathBuf> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => PathBuf::from(p),
            None => victory_shader::emit::validator::Validator::default_path()?
                .with_file_name("dxc.exe"),
        };
        dxc.exists().then_some(dxc)
    }

    fn compile(dxc: &Path, dir: &Path, src: &str, profile: &str) -> Vec<u8> {
        let (hlsl, out) = (
            dir.join(format!("{profile}.hlsl")),
            dir.join(format!("{profile}.dxil")),
        );
        std::fs::write(&hlsl, src).unwrap();
        let status = std::process::Command::new(dxc)
            .args(["-nologo", "-T", profile, "-E", "main", "-Fo"])
            .arg(&out)
            .arg(&hlsl)
            .status()
            .unwrap();
        assert!(status.success(), "dxc failed for {profile}");
        std::fs::read(out).unwrap()
    }

    fn code(bytes: &[u8]) -> D3D12_SHADER_BYTECODE {
        D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        }
    }

    /// A whole-resource transition, for a test that records its own list.
    fn transition(
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) -> D3D12_RESOURCE_BARRIER {
        D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: a borrowed resource pointer; the barrier never releases it.
                    pResource: unsafe { std::mem::transmute_copy(resource) },
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        }
    }

    /// A mesh pipeline made once is loaded from the library the next time,
    /// not compiled again (a stream without a topology never loads).
    #[test]
    fn a_mesh_pipeline_loads_from_the_library() {
        let Some(host) = host() else {
            return; // no host GPU here
        };
        let Some(dxc) = dxc() else {
            eprintln!("no PC dxc.exe; skipped");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();
        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        let name = super::super::library::name("mesh test", &root_blob, &[&ms, &ps], &g.describe());
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, &name).expect("made");
        let loaded = super::super::library::LOADED.load(Ordering::Relaxed);
        mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, &name).expect("made again");
        assert!(
            super::super::library::LOADED.load(Ordering::Relaxed) > loaded,
            "the mesh pipeline was compiled again instead of loaded from the library"
        );
    }

    /// The production mesh pipeline draws its one triangle into an offscreen
    /// target, and the result is read back: a smoke test of the mesh path
    /// without any title data or console compiler.
    ///
    /// The triangle covers half of the upper-right quadrant (1/8 of the
    /// target), with centroid at (2/3 width, 1/3 height). The test checks
    /// that coverage is substantial but not a full-screen fill, that an
    /// interior pixel and the coverage centroid are white, and that a pixel outside
    /// the triangle stays black.
    #[test]
    fn mesh_pipeline_draws_a_triangle() {
        use windows::Win32::Foundation::RECT;
        use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        use windows::core::Interface;

        const W: u32 = 64;
        const H: u32 = 64;

        let Some(host) = host() else {
            eprintln!("SKIP mesh_pipeline_draws_a_triangle: no host D3D12 device");
            return;
        };
        if !host.mesh_shaders() {
            eprintln!("SKIP mesh_pipeline_draws_a_triangle: no mesh shaders at SM 6.6");
            return;
        }
        let Some(dxc) = dxc() else {
            eprintln!(
                "SKIP mesh_pipeline_draws_a_triangle: no PC dxc.exe (set VICTORY_PC_DXC, or install the Windows SDK)"
            );
            return;
        };

        let dir = tempfile::tempdir().unwrap();
        let ms = compile(&dxc, dir.path(), MS, "ms_6_5");
        let ps = compile(&dxc, dir.path(), PS, "ps_6_5");
        let (root, root_blob) = root_signature(&host.device, &[], &[], false).unwrap();

        let mut g = GraphicsState::default();
        g.targets.NumRenderTargets = 1;
        g.targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // No depth target is bound below; no culling, so the triangle's
        // winding cannot turn this into a coverage pass that draws nothing.
        g.depth.DepthEnable = false.into();
        g.depth.StencilEnable = false.into();
        g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
        let name =
            super::super::library::name("mesh draw smoke", &root_blob, &[&ms, &ps], &g.describe());
        let pso = mesh_pipeline(&host.device, &root, code(&ms), code(&ps), &g, &name)
            .expect("host mesh pipeline created");

        let device = &host.device;
        // SAFETY: D3D12 calls with descriptions that outlive them; every
        // resource and object stays alive until the fence below completes.
        unsafe {
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("command queue");
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("command allocator");
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .expect("command list");
            let list6: ID3D12GraphicsCommandList6 = match list.cast() {
                Ok(list) => list,
                Err(_) => {
                    eprintln!(
                        "SKIP mesh_pipeline_draws_a_triangle: ID3D12GraphicsCommandList6 unavailable"
                    );
                    return;
                }
            };

            let target_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(W),
                Height: H,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let clear = D3D12_CLEAR_VALUE {
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                Anonymous: D3D12_CLEAR_VALUE_0 {
                    Color: [0.0, 0.0, 0.0, 0.0],
                },
            };
            let mut target: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &target_desc,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    Some(&clear as *const D3D12_CLEAR_VALUE),
                    &mut target,
                )
                .expect("render target");
            let target = target.expect("render target");

            let rtv_heap: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })
                .expect("render target view heap");
            let rtv = rtv_heap.GetCPUDescriptorHandleForHeapStart();
            device.CreateRenderTargetView(&target, None, rtv);

            // The root signature allows directly indexed heaps; the shaders
            // read none, but bind empty shader-visible heaps so the draw
            // cannot depend on an unbound-heap validation rule.
            let views: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("view heap");
            let samplers: ID3D12DescriptorHeap = device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .expect("sampler heap");
            list.SetDescriptorHeaps(&[Some(views.clone()), Some(samplers.clone())]);
            list.SetPipelineState(&pso);
            list.SetGraphicsRootSignature(&root);
            list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: W as f32,
                Height: H as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[RECT {
                left: 0,
                top: 0,
                right: W as i32,
                bottom: H as i32,
            }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.ClearRenderTargetView(rtv, &[0.0, 0.0, 0.0, 0.0], None);
            list6.DispatchMesh(1, 1, 1);
            list.ResourceBarrier(&[transition(
                &target,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);

            let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0;
            device.GetCopyableFootprints(
                &target_desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total),
            );
            let readback_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: total,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut readback: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_READBACK,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &readback_desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .expect("read-back buffer");
            let readback = readback.expect("read-back buffer");

            let to = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed resource pointer; the location never releases it.
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: footprint,
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                // SAFETY: a borrowed resource pointer; the location never releases it.
                pResource: std::mem::transmute_copy(&target),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
            list.Close().expect("close command list");

            let submitted: ID3D12CommandList = list.cast().expect("command list");
            queue.ExecuteCommandLists(&[Some(submitted)]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
            queue.Signal(&fence, 1).expect("signal fence");
            let started = std::time::Instant::now();
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = device.GetDeviceRemovedReason() {
                    panic!("device removed while drawing the mesh triangle: {e}");
                }
                if started.elapsed() > std::time::Duration::from_secs(30) {
                    panic!("mesh triangle draw did not complete within 30 s");
                }
                std::thread::yield_now();
            }
            if let Err(e) = device.GetDeviceRemovedReason() {
                panic!("device removed after drawing the mesh triangle: {e}");
            }

            let mut data = std::ptr::null_mut();
            readback
                .Map(0, None, Some(&mut data))
                .expect("map read-back");
            let pitch = footprint.Footprint.RowPitch as usize;
            let bytes = std::slice::from_raw_parts(data.cast::<u8>(), pitch * H as usize);
            let pixel = |x: u32, y: u32| -> [u8; 4] {
                let at = y as usize * pitch + x as usize * 4;
                bytes[at..at + 4].try_into().unwrap()
            };

            let mut covered = 0usize;
            let mut sum_x = 0u64;
            let mut sum_y = 0u64;
            for y in 0..H {
                for x in 0..W {
                    let p = pixel(x, y);
                    if p[0].max(p[1]).max(p[2]) > 200 {
                        covered += 1;
                        sum_x += u64::from(x);
                        sum_y += u64::from(y);
                    }
                }
            }
            let total_pixels = (W * H) as usize;
            let center_x = if covered == 0 {
                0
            } else {
                (sum_x as f32 / covered as f32).round() as u32
            };
            let center_y = if covered == 0 {
                0
            } else {
                (sum_y as f32 / covered as f32).round() as u32
            };
            let center = pixel(center_x.min(W - 1), center_y.min(H - 1));
            let middle = pixel(W * 5 / 8, H * 3 / 8);
            let outside = pixel(W / 8, H * 7 / 8);
            readback.Unmap(0, None);
            eprintln!(
                "mesh smoke: {covered}/{total_pixels} pixels covered, centre {center:?} at ({center_x}, {center_y}), middle {middle:?}, outside {outside:?}"
            );

            assert!(
                covered >= total_pixels / 10,
                "mesh triangle coverage too small: {covered} of {total_pixels} pixels"
            );
            assert!(
                covered <= total_pixels * 3 / 20,
                "mesh triangle covered too much (a full-screen fill?): {covered} of {total_pixels} pixels"
            );
            assert!(
                middle[0] >= 250 && middle[1] >= 250 && middle[2] >= 250 && middle[3] >= 250,
                "the interior pixel {middle:?} is not white"
            );
            assert!(
                center[0] >= 250 && center[1] >= 250 && center[2] >= 250 && center[3] >= 250,
                "the rendered triangle's centre {center:?} at ({center_x}, {center_y}) is not white"
            );
            assert!(
                outside[0] < 10 && outside[1] < 10 && outside[2] < 10,
                "a pixel outside the triangle is not black: {outside:?}"
            );
            if covered > 0 {
                let cx = sum_x as f32 / covered as f32;
                let cy = sum_y as f32 / covered as f32;
                let expected = (W as f32 * 2.0 / 3.0, H as f32 / 3.0);
                assert!(
                    (cx - expected.0).abs() <= 4.0 && (cy - expected.1).abs() <= 4.0,
                    "coverage centroid ({cx:.1}, {cy:.1}) is not the triangle's ({:.1}, {:.1})",
                    expected.0,
                    expected.1
                );
            }
        }
    }
}
