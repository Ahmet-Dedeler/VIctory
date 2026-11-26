//! Carry vertex values to the pixel shader as flat per-primitive outputs.
//!
//! D3DMetal cannot lower `attributeAtVertex`. A mesh shader knows all
//! three vertex indices, and can export the values the pixel shader needs.
//! Only the linked pair uses this layout; the shared stage translations
//! still use their ordinary signatures, resource bindings and cache keys.

use std::collections::BTreeSet;

use super::container::{MESH_OUTPUT_LIMIT, SigElement, mesh_output_bytes};
use super::{HostShader, Prepared, emit_program_with_varyings, signature};
use crate::io::{Interp, Io};
use crate::native::{MeshIo, StageIo};
use crate::{FourCC, ShaderError};

const SEMANTIC: &str = "VICTORY_VERTEX_DATA";
const SIGNATURE_ROWS: u32 = 32;
const SHARED_LIMIT: u32 = 28 * 1024;

#[derive(Clone, Copy)]
pub(super) struct Field {
    /// Signature element ID and its relative component.
    pub mesh: (usize, u8),
    pub pixel: (usize, u8),
}

pub(super) struct Layout {
    pub fields: Vec<Field>,
}

pub(super) fn interpolates(io: &Io, row: u8) -> bool {
    io.interps.iter().any(|(_, i)| match i {
        Interp::Interpolated { attribute, .. } | Interp::Weighted { attribute } => {
            attribute.row == row
        }
        Interp::Vertex { .. } => false,
    })
}

pub(super) fn per_vertex(io: &Io, row: u8) -> bool {
    io.interps.iter().any(|(_, i)| match *i {
        Interp::Weighted { attribute } => attribute.row == row,
        Interp::Vertex { attribute, vertex } => {
            attribute.row == row && vertex != 2 && interpolates(io, row)
        }
        Interp::Interpolated { .. } => false,
    })
}

fn cannot_link(what: &'static str) -> ShaderError {
    ShaderError::Unsupported { offset: 0, what }
}

fn signature_rows(elements: &[SigElement]) -> u32 {
    elements
        .iter()
        .filter_map(|e| e.row)
        .max()
        .map_or(0, |r| r.saturating_add(1))
}

impl Layout {
    pub fn new(
        ms: &MeshIo,
        outputs: &[SigElement],
        ps: &StageIo,
        io: &Io,
        inputs: &[SigElement],
    ) -> Result<Option<Self>, ShaderError> {
        let StageIo::Ps {
            inputs: attributes, ..
        } = ps
        else {
            return Ok(None);
        };
        if ms.topology != 2
            || !(1..=128).contains(&ms.threads)
            || !(1..=256).contains(&ms.max_vertices)
            || !(1..=256).contains(&ms.max_primitives)
        {
            return Ok(None);
        }
        let needed: BTreeSet<_> = io
            .interps
            .iter()
            .filter_map(|(_, i)| {
                let a = match *i {
                    Interp::Vertex { vertex: 2, .. } => return None,
                    Interp::Vertex { attribute, .. }
                    | Interp::Weighted { attribute }
                    | Interp::Interpolated { attribute, .. } => attribute,
                };
                per_vertex(io, a.row).then_some((a.row, a.channel))
            })
            .collect();
        if needed.is_empty() {
            return Ok(None);
        }
        let mut fields = Vec::with_capacity(needed.len());
        for (row, channel) in needed {
            if attributes.iter().any(|a| a.row == row && a.fp16) {
                return Err(cannot_link("a packed 16-bit per-vertex pixel input"));
            }
            let (ps_id, input) = inputs
                .iter()
                .enumerate()
                .find(|(_, e)| {
                    e.row == Some(u32::from(row)) && (e.col..e.col + e.cols).contains(&channel)
                })
                .ok_or_else(|| {
                    cannot_link("a per-vertex channel missing from the pixel signature")
                })?;
            let column = channel - input.col;
            let matches =
                |e: &SigElement| e.name.eq_ignore_ascii_case(&input.name) && e.index == input.index;
            let mut matching_outputs = outputs.iter().enumerate().filter(|(_, e)| matches(e));
            let (ms_id, output) = matching_outputs.next().ok_or_else(|| {
                cannot_link("a per-vertex channel missing from the mesh signature")
            })?;
            if matching_outputs.next().is_some()
                || input.kind != 0
                || input.comp != 9
                || input.half
                || output.kind != 0
                || output.comp != 9
                || output.half
                || column >= output.cols
            {
                return Err(cannot_link(
                    "a per-vertex channel that is not an unambiguous f32 varying",
                ));
            }
            fields.push(Field {
                mesh: (ms_id, column),
                pixel: (ps_id, column),
            });
        }
        Ok(Some(Self { fields }))
    }

    pub fn shared_dwords(&self, ms: &MeshIo) -> u32 {
        // One value per possible vertex per channel, followed by one packed
        // index word per primitive. The original mesh LDS is separate.
        self.fields.len() as u32 * ms.max_vertices + ms.max_primitives
    }

    pub fn rows(&self) -> usize {
        (self.fields.len() * 3).div_ceil(4)
    }

    pub fn extend_signatures(
        &self,
        stage: &StageIo,
        inputs: &mut Vec<SigElement>,
        outputs: &[SigElement],
        prims: &mut Vec<SigElement>,
    ) -> Result<(), ShaderError> {
        let channels = (self.fields.len() as u32) * 3;
        let (target, rows_elsewhere) = match stage {
            StageIo::Ps { .. } => (inputs, 0),
            StageIo::Ms(_) => (&mut *prims, signature_rows(outputs)),
            _ => {
                return Err(cannot_link(
                    "a linked varying outside mesh and pixel stages",
                ));
            }
        };
        let start = signature_rows(target);
        if start
            .saturating_add(channels.div_ceil(4))
            .saturating_add(rows_elsewhere)
            > SIGNATURE_ROWS
            || target
                .iter()
                .chain(outputs)
                .any(|e| e.name.eq_ignore_ascii_case(SEMANTIC))
        {
            return Err(cannot_link(
                "a linked varying exceeds or conflicts with the signature layout",
            ));
        }
        for row in 0..channels.div_ceil(4) {
            target.push(SigElement {
                name: SEMANTIC.into(),
                index: row,
                kind: 0,
                comp: 9,
                interp: 1,
                row: Some(start + row),
                col: 0,
                cols: (channels - row * 4).min(4) as u8,
                usage: 0,
                half: false,
            });
        }
        if let StageIo::Ms(ms) = stage {
            if mesh_output_bytes(outputs, prims, ms.max_vertices, ms.max_primitives)
                > MESH_OUTPUT_LIMIT
                || ms
                    .lds_bytes
                    .div_ceil(4)
                    .saturating_add(self.shared_dwords(ms))
                    > SHARED_LIMIT / 4
            {
                return Err(cannot_link(
                    "a linked varying exceeds the mesh output or shared-memory budget",
                ));
            }
        }
        Ok(())
    }
}

/// Build a private mesh/pixel pair for a pipeline that needs per-vertex
/// attributes. The caller checks that the linked pair preserves its stage
/// bindings before using it with the ordinary root signature.
pub fn link_mesh_primitive_varyings(
    mesh: &[u8],
    pixel: &[u8],
) -> Result<Option<(HostShader, HostShader)>, ShaderError> {
    let ps = Prepared::parse(pixel)?;
    // Most pixel stages need no special values. Avoid analysing their mesh.
    if !ps.io.interps.iter().any(|(_, i)| {
        matches!(
            i,
            Interp::Weighted { .. } | Interp::Vertex { vertex: 0 | 1, .. }
        )
    }) {
        return Ok(None);
    }
    let ms = Prepared::parse(mesh)?;
    link_pair(ms, ps)
}

fn link_pair(ms: Prepared, ps: Prepared) -> Result<Option<(HostShader, HostShader)>, ShaderError> {
    let StageIo::Ms(mesh_io) = &ms.native.io else {
        return Ok(None);
    };
    let outputs = ms
        .container
        .part(FourCC::OSG1)
        .map(|p| signature(p.data))
        .transpose()?
        .unwrap_or_default();
    let inputs = ps
        .container
        .part(FourCC::ISG1)
        .map(|p| signature(p.data))
        .transpose()?
        .unwrap_or_default();
    let Some(layout) = Layout::new(mesh_io, &outputs, &ps.native.io, &ps.io, &inputs)? else {
        return Ok(None);
    };
    let emit = |p: Prepared| {
        emit_program_with_varyings(
            &p.container,
            &p.native,
            &p.ir,
            &p.structured,
            &p.io,
            &p.resources,
            Some(&layout),
        )
        .map(|(shader, _)| shader)
    };
    Ok(Some((emit(ms)?, emit(ps)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dxbc::Container;
    use crate::native::{HwStage, NativeProgram, PsInput, Usage, VsOutput};

    fn element(name: &str, kind: u8, row: u32, cols: u8) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols,
            usage: 0,
            half: false,
        }
    }

    /// Vertex n exports value n. A triangle using vertices 0, 1 and 2
    /// covers the target. The pixel's computed weights take half of each
    /// delta from vertex 0, so every covered pixel must equal 1.5.
    fn fixture(indices: u32) -> (HostShader, HostShader) {
        let mesh_words = [
            0xD765_0001u32,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x3604_0281,
            0x3606_0282, // v2 = lane & 1, v3 = lane & 2
            0x7E04_0D02,
            0x7E06_0D03, // convert positions to f32
            0x1008_04FF,
            4f32.to_bits(),
            0x0608_08F3, // x = v2 * 4 - 1
            0x100A_06FF,
            2f32.to_bits(),
            0x060A_0AF3, // y = v3 * 2 - 1
            0x7E0C_02F0,
            0x7E0E_02F2, // z = 0.5, w = 1
            0x7E12_02FF,
            indices,     // v_mov_b32 v9, indices
            0x7E14_0D01, // v_cvt_f32_u32 v10, v1 (varying)
            0xF800_0941,
            0x0000_0009, // exp prim v9
            0xF800_0201,
            0x0000_000A, // exp param0 v10
            0xF800_08CF,
            0x0706_0504, // exp pos0 v4, v5, v6, v7
            0xBF81_0000,
        ];
        let mesh = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 11,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(MeshIo {
                outputs: vec![VsOutput {
                    usage: Usage::Generic,
                    row: 1,
                    param: 0,
                    fp16: false,
                    pack2: false,
                }],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 3,
                max_primitives: 1,
                topology: 2,
                lds_bytes: 0,
            }),
            code: mesh_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let outputs = super::super::container::signature(
            &[
                element("SV_Position", 3, 0, 4),
                element("TEXCOORD", 0, 1, 1),
            ],
            false,
        );
        let mesh_bytes = crate::dxbc::build(&[(FourCC::OSG1, &outputs)]);
        let mesh = Prepared::from_native(Container::parse(&mesh_bytes).unwrap(), mesh).unwrap();

        let pixel_words = [
            0x7E06_02F0u32, // v_mov_b32 v3, 0.5 (computed i and j)
            0xC808_0003,
            0xC809_0003, // v_interp_p1/p2_f32 v2, v3, attr0.x
            0xF800_080Fu32,
            0x0202_0202, // exp mrt0, v2 splatted
            0xBF81_0000,
        ];
        let pixel = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 1,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 2,
                db_shader_control: 0,
                patches: vec![],
            },
            code: pixel_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let pixel_inputs =
            super::super::container::signature(&[element("TEXCOORD", 0, 0, 1)], true);
        let pixel_outputs =
            super::super::container::signature(&[element("SV_Target", 16, 0, 4)], false);
        let pixel_bytes = crate::dxbc::build(&[
            (FourCC::ISG1, &pixel_inputs),
            (FourCC::OSG1, &pixel_outputs),
        ]);
        let pixel = Prepared::from_native(Container::parse(&pixel_bytes).unwrap(), pixel).unwrap();

        link_pair(mesh, pixel).unwrap().expect("linked fixture")
    }

    #[test]
    fn linked_varying_signatures_match() {
        let (mesh, pixel) = fixture((1 << 10) | (2 << 20));
        let mesh = Container::parse(&mesh.dxil).unwrap();
        let pixel = Container::parse(&pixel.dxil).unwrap();
        let payload = |container: &Container, part| {
            signature(container.part(part).unwrap().data)
                .unwrap()
                .into_iter()
                .find(|e| e.name == SEMANTIC)
                .unwrap()
        };
        assert_eq!(payload(&mesh, FourCC::PSG1).cols, 3);
        assert_eq!(payload(&pixel, FourCC::ISG1).cols, 3);
        let flags = pixel.part(FourCC::SFI0).unwrap().data;
        assert_eq!(
            u64::from_le_bytes(flags.try_into().unwrap())
                & super::super::shader::flags::BARYCENTRICS,
            0
        );
    }

    #[cfg(windows)]
    #[test]
    fn linked_varyings_draw_with_original_culling() {
        use super::super::host::{Draw, Host, Root};
        use super::super::validator::Validator;
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
            D3D12_MESH_SHADER_TIER_NOT_SUPPORTED,
        };
        let _one = super::super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else {
            eprintln!("no hardware D3D12 device; skipped");
            return;
        };
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
        // SAFETY: valid output buffer of the size required by this feature query.
        let queried = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS7,
                (&raw mut options).cast(),
                size_of_val(&options) as u32,
            )
        };
        if queried.is_err() || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
            eprintln!("no hardware mesh shaders; skipped");
            return;
        }
        let mut validator =
            Validator::default_path().map(|path| Validator::load(Some(&path)).unwrap());
        for (indices, expected) in [
            ((1 << 10) | (2 << 20), 1.5),
            ((1 << 10) | (2 << 20) | (1 << 31), 0.0),
        ] {
            let (mesh, pixel) = fixture(indices);
            if let Some(validator) = &mut validator {
                for shader in [&mesh, &pixel] {
                    validator
                        .validate(&mut shader.dxil.clone())
                        .unwrap_or_else(|e| panic!("{:?}: {e}", shader.stage));
                }
            }
            let (root, pipeline) = host.mesh_pipeline(&mesh, &pixel).unwrap();
            let pixels = host
                .draw(
                    &root,
                    &pipeline,
                    &Draw {
                        roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
                        textures: &[],
                        samplers: &[],
                        vertices: &[],
                        stride: 0,
                        size: (16, 16),
                        mesh_groups: 1,
                    },
                )
                .unwrap();
            assert!(
                pixels.iter().all(|pixel| pixel[0] == expected),
                "{indices:#x}: expected red {expected}, got {:?}",
                &pixels[..8]
            );
        }
        eprintln!("hardware readback verified 256 weighted pixels and 256 culled pixels");
    }
}
