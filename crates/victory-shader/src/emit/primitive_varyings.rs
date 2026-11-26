//! Carry vertex values to the pixel shader as flat per-primitive outputs.
//!
//! D3DMetal cannot lower `attributeAtVertex`. A mesh shader knows all
//! three vertex indices, and can export the values the pixel shader needs.
//! Only the linked pair uses this layout; the shared stage translations
//! still use their ordinary signatures, resource bindings and cache keys.

use super::container::{MESH_OUTPUT_LIMIT, SigElement, mesh_output_bytes};
use super::{Emitted, HostShader, Prepared, emit_program_with_varyings, lower, signature};
use crate::io::{Interp, Io};
use crate::native::{MeshIo, StageIo};
use crate::{Container, FourCC, ShaderError};

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

#[derive(Clone, Copy)]
pub(super) enum Mode<'a> {
    Ordinary,
    Discover,
    Linked(&'a Layout),
}

impl<'a> Mode<'a> {
    pub fn layout(self) -> Option<&'a Layout> {
        match self {
            Self::Linked(layout) => Some(layout),
            _ => None,
        }
    }
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
        inputs: &[SigElement],
        needed: &[(usize, u8)],
    ) -> Result<Option<Self>, ShaderError> {
        if ms.topology != 2
            || !(1..=128).contains(&ms.threads)
            || !(1..=256).contains(&ms.max_vertices)
            || !(1..=256).contains(&ms.max_primitives)
        {
            return Ok(None);
        }
        if needed.is_empty() {
            return Ok(None);
        }
        let mut fields = Vec::with_capacity(needed.len());
        for &(ps_id, column) in needed {
            let input = inputs
                .get(ps_id)
                .filter(|element| column < element.cols)
                .ok_or_else(|| {
                    cannot_link("a per-vertex channel missing from the pixel signature")
                })?;
            let matches =
                |e: &SigElement| e.name.eq_ignore_ascii_case(&input.name) && e.index == input.index;
            let mut matching_outputs = outputs.iter().enumerate().filter(|(_, e)| matches(e));
            let (ms_id, output) = matching_outputs.next().ok_or_else(|| {
                cannot_link("a per-vertex channel missing from the mesh signature")
            })?;
            if matching_outputs.next().is_some()
                || input.kind != 0
                || input.comp != 9
                || output.kind != 0
                || output.comp != 9
                || column >= output.cols
            {
                return Err(cannot_link(
                    "a per-vertex channel without an unambiguous float varying",
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
/// attributes. Both linked stages must use exactly the ordinary stage's
/// bindings so the pipeline can retain its existing root signature.
pub fn link_mesh_primitive_varyings(
    mesh: (&[u8], &HostShader),
    pixel: (&[u8], &HostShader),
) -> Result<Option<(HostShader, HostShader)>, ShaderError> {
    let host_pixel = Container::parse(&pixel.1.dxil)?;
    let Some(flags) = host_pixel
        .part(FourCC::SFI0)
        .and_then(|part| part.data.get(..8))
    else {
        return Ok(None);
    };
    let flags = u64::from_le_bytes(flags.try_into().unwrap());
    if flags & super::shader::FEATURE_BARYCENTRICS == 0 {
        return Ok(None);
    }
    let ps = Prepared::parse(pixel.0)?;
    let ms = Prepared::parse(mesh.0)?;
    let Some((linked_mesh, linked_pixel)) = link_pair(ms, ps)? else {
        return Ok(None);
    };
    if !same_bindings(mesh.1, &linked_mesh) || !same_bindings(pixel.1, &linked_pixel) {
        return Err(cannot_link(
            "linking the mesh and pixel stages changes a resource binding",
        ));
    }
    Ok(Some((linked_mesh, linked_pixel)))
}

fn same_bindings(a: &HostShader, b: &HostShader) -> bool {
    let HostShader {
        stage,
        dxil: _,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
        half_views,
    } = a;
    *stage == b.stage
        && *user_data == b.user_data
        && *samplers == b.samplers
        && *static_samplers == b.static_samplers
        && *root_cbvs == b.root_cbvs
        && *root_srvs == b.root_srvs
        && *root_uavs == b.root_uavs
        && *root_indexed_views == b.root_indexed_views
        && *half_views == b.half_views
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
    // Discovery runs the same expansion as ordinary pixel emission. For a
    // packed f16 attribute, GX register halves can map to different host
    // signature elements and columns. Carry the *expanded* producer values
    // and let the existing pixel code do the same f16 conversion as before.
    let discovery = emit(&ps, Mode::Discover)?;
    let Some(layout) = Layout::new(mesh_io, &outputs, &inputs, &discovery.vertex_fetches)? else {
        return Ok(None);
    };
    let mode = Mode::Linked(&layout);
    Ok(Some((emit(&ms, mode)?.host, emit(&ps, mode)?.host)))
}

fn emit(p: &Prepared, mode: Mode<'_>) -> Result<Emitted, ShaderError> {
    let lower = || {
        emit_program_with_varyings(
            &p.container,
            &p.native,
            &p.ir,
            &p.structured,
            &p.io,
            &p.resources,
            mode,
        )
    };
    match lower() {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::FIRST_HALF => {
            lower::with_first_half_pairs(lower)
        }
        result => result,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Original signatures describe packed halves with component type 6.
    /// The emitter's signature writer describes already-expanded host floats.
    fn fixture_signature(elements: &[SigElement], input: bool) -> Vec<u8> {
        let mut bytes = super::super::container::signature(elements, input);
        for (id, _) in elements.iter().enumerate().filter(|(_, e)| e.half) {
            let at = 8 + id * 32 + 16;
            bytes[at..at + 4].copy_from_slice(&6u32.to_le_bytes());
        }
        bytes
    }

    fn fixture(first: u32, second: Option<u32>) -> (HostShader, HostShader, HostShader) {
        let packed = second.is_some();
        // The two-triangle fixture is a quad. The one-triangle fixture
        // doubles each leg to cover the same render target.
        let x_scale: f32 = if packed { 2.0 } else { 4.0 };
        let y_scale: f32 = if packed { 1.0 } else { 2.0 };
        let mut mesh_words = vec![
            0xD765_0001u32,
            0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x3604_0281,
            0x3606_0282, // v2 = lane & 1, v3 = lane & 2
            0x7E04_0D02,
            0x7E06_0D03, // convert positions to f32
            0x1008_04FF,
            x_scale.to_bits(),
            0x0608_08F3, // x = v2 * x_scale - 1
            0x100A_06FF,
            y_scale.to_bits(),
            0x060A_0AF3, // y = v3 * y_scale - 1
            0x7E0C_02F0,
            0x7E0E_02F2, // z = 0.5, w = 1
            0x7E12_02FF,
            first,       // v_mov_b32 v9, first triangle
            0x7E14_0D01, // v_cvt_f32_u32 v10, v1 (lane)
        ];
        if let Some(second) = second {
            mesh_words.extend([
                0x7E10_02FF,
                second,      // v_mov_b32 v8, second triangle
                0x7D84_0280, // v_cmp_eq_u32 vcc, 0, v1
                0x0212_1308, // v_cndmask_b32 v9, v8, v9, vcc
                0x1010_14FF,
                0.25f32.to_bits(), // v_mul_f32 v8, 0.25, v10
                0x0610_10F2,       // v_add_f32 v8, 1, v8 (low half = 1 + lane / 4)
                0x0614_14F6,       // v_add_f32 v10, 4, v10 (high half = 4 + lane)
                0x5E14_1508,       // v_cvt_pkrtz_f16_f32 v10, v8, v10
            ]);
        }
        mesh_words.extend([
            0xF800_0941,
            0x0000_0009, // exp prim v9
            0xF800_0201,
            0x0000_000A, // exp param0 v10
            0xF800_08CF,
            0x0706_0504, // exp pos0 v4, v5, v6, v7
            0xBF81_0000,
        ]);
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
                    fp16: packed,
                    pack2: packed,
                }],
                clip_cntl: 0,
                threads: 32,
                max_vertices: if packed { 4 } else { 3 },
                max_primitives: if packed { 2 } else { 1 },
                topology: 2,
                lds_bytes: 0,
            }),
            code: mesh_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let uv = |row| SigElement {
            half: packed,
            ..element("TEXCOORD", 0, row, if packed { 2 } else { 1 })
        };
        let outputs = fixture_signature(&[element("SV_Position", 3, 0, 4), uv(1)], false);
        let mesh_bytes = crate::dxbc::build(&[(FourCC::OSG1, &outputs)]);
        let mesh = Prepared::from_native(Container::parse(&mesh_bytes).unwrap(), mesh).unwrap();

        let pixel_words: &[u32] = if packed {
            &[
                0x7E06_02F0, // v_mov_b32 v3, 0.5 (computed i)
                0xD742_0002,
                0x0002_0600, // v_interp_p1ll_f16 v2, v3, attr0.x low
                0xD742_0004,
                0x0002_0700, // v_interp_p1ll_f16 v4, v3, attr0.x high
                0xF800_080F,
                0x0202_0402, // exp mrt0, v2, v4, v2, v2
                0xBF81_0000,
            ]
        } else {
            // P0 + 1.0 (P1 - P0) + 0.5 (P2 - P0). The asymmetry
            // makes the index order observable.
            &[
                0x7E06_02F2, // v_mov_b32 v3, 1.0 (computed i)
                0x7E08_02F0, // v_mov_b32 v4, 0.5 (computed j)
                0xC808_0003,
                0xC809_0004, // v_interp_p1 v2, v3 and p2 v2, v4, attr0.x
                0xF800_080F,
                0x0202_0202, // exp mrt0, v2 splatted
                0xBF81_0000,
            ]
        };
        let pixel = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 5,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: if packed { 3 } else { 1 },
                    attr: 0,
                    flat: false,
                    fp16: packed,
                    pack2: packed,
                    row2: 0,
                }],
                input_addr: 2,
                db_shader_control: 0,
                patches: vec![],
            },
            code: pixel_words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let pixel_inputs = fixture_signature(&[uv(0)], true);
        let pixel_outputs = fixture_signature(&[element("SV_Target", 16, 0, 4)], false);
        let pixel_bytes = crate::dxbc::build(&[
            (FourCC::ISG1, &pixel_inputs),
            (FourCC::OSG1, &pixel_outputs),
        ]);
        let pixel = Prepared::from_native(Container::parse(&pixel_bytes).unwrap(), pixel).unwrap();

        let ordinary = emit(&pixel, Mode::Discover).unwrap();
        let expected: &[(usize, u8)] = if packed { &[(0, 0), (0, 1)] } else { &[(0, 0)] };
        assert_eq!(ordinary.vertex_fetches, expected);
        let (mesh, pixel) = link_pair(mesh, pixel).unwrap().expect("linked fixture");
        (mesh, pixel, ordinary.host)
    }

    #[test]
    fn linked_varying_signatures_match() {
        let payload = |container: &Container, part| {
            signature(container.part(part).unwrap().data)
                .unwrap()
                .into_iter()
                .filter(|e| e.name == SEMANTIC)
                .map(|e| (e.index, e.cols))
                .collect::<Vec<_>>()
        };
        let check_barycentrics = |container: &Container| {
            let flags = container.part(FourCC::SFI0).unwrap().data;
            u64::from_le_bytes(flags.try_into().unwrap())
                & super::super::shader::FEATURE_BARYCENTRICS
        };
        for (second, expected) in [
            (None, vec![(0, 3)]),
            (Some(2 | (1 << 10) | (3 << 20)), vec![(0, 4), (1, 2)]),
        ] {
            let (mesh, pixel, ordinary) = fixture((1 << 10) | (2 << 20), second);
            let mesh = Container::parse(&mesh.dxil).unwrap();
            let pixel = Container::parse(&pixel.dxil).unwrap();
            assert_eq!(payload(&mesh, FourCC::PSG1), expected);
            assert_eq!(payload(&pixel, FourCC::ISG1), expected);
            assert_ne!(
                check_barycentrics(&Container::parse(&ordinary.dxil).unwrap()),
                0
            );
            assert_eq!(check_barycentrics(&pixel), 0);
        }
    }

    #[cfg(windows)]
    #[test]
    fn linked_varyings_hardware_readback() {
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
        let mut draw = |first, second| {
            let (mesh, pixel, _) = fixture(first, second);
            if let Some(validator) = &mut validator {
                for shader in [&mesh, &pixel] {
                    validator
                        .validate(&mut shader.dxil.clone())
                        .unwrap_or_else(|e| panic!("{:?}: {e}", shader.stage));
                }
            }
            let (root, pipeline) = host.mesh_pipeline(&mesh, &pixel).unwrap();
            host.draw(
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
            .unwrap()
        };
        for (indices, expected) in [
            ((1 << 10) | (2 << 20), 2.0),
            (2 | (1 << 20) | (1 << 9) | (1 << 19) | (1 << 29), -0.5),
            ((1 << 10) | (2 << 20) | (1 << 31), 0.0),
            ((1 << 10) | (256 << 20), 0.0),
        ] {
            let pixels = draw(indices, None);
            assert!(
                pixels.iter().all(|pixel| pixel[0] == expected),
                "{indices:#x}: expected red {expected}, got {:?}",
                &pixels[..8]
            );
        }
        // The two triangles share vertices 1 and 2. Each channel fetches
        // the same two vertex indices from a different physical half.
        let first = (1 << 10) | (2 << 20);
        let second = 2 | (1 << 10) | (3 << 20);
        let pixels = draw(first, Some(second));
        let bottom_left = [1.125, 4.5];
        let top_right = [1.375, 5.5];
        for (i, pixel) in pixels.iter().enumerate() {
            let (x, y) = (i % 16, i / 16);
            assert!(
                if x < y {
                    pixel[..2] == bottom_left
                } else if x > y {
                    pixel[..2] == top_right
                } else {
                    pixel[..2] == bottom_left || pixel[..2] == top_right
                },
                "packed varying at ({x}, {y}): {pixel:?}"
            );
        }
        eprintln!(
            "hardware readback verified 256 packed-half pixels on two shared-vertex triangles"
        );
        eprintln!("hardware readback verified 512 weighted pixels and 512 culled pixels");
    }
}
