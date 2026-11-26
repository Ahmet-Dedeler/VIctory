//! `VICTORY_F16_NATIVE=1` (with `VICTORY_F16_NEAREST=1`): the nearest-even rounding of
//! 16-bit float results as one native `fptrunc float to half` instead of the software
//! rounding (`lower.rs`, `Lower::f16_nearest_native`).
//!
//! These run on any host, without `dxil.dll`, a D3D12 device or Apple's IR converter: they
//! read the emitted container and its bitcode back with a small reader of their own (the
//! writer only writes unabbreviated records, `bitcode.rs`) and check
//!
//! - the instructions: `fptrunc` to `half`, `bitcast` to `i16`, `zext` to `i32`, chained;
//! - what the module declares: the `half` and `i16` types, the native 16-bit data layout,
//!   the `LowPrecisionPresent | UseNativeLowPrecision` shader flags and the `SFI0`
//!   `NativeLowPrecision` bit, shader model 6.2 or later; the same values DXC writes for
//!   the clean-room `reference-native.ll` at the top of the checkout;
//! - that nothing else changes: the switch alone (without `VICTORY_F16_NEAREST`), the
//!   software rounding, 32-bit float results and the exact wave sum of halves give the
//!   same bytes as before.
//!
//! What they cannot show is what a GPU computes: `STATUS.md` lists the hardware runs.

use super::bitcode::{LAYOUT_MIN_PRECISION, LAYOUT_NATIVE_16};
use super::shader::flags;
use super::{HostShader, lower};
use crate::dxbc::{Container, FourCC};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{dxil, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
/// `buffer_store_dword v0, off, s[8:11], 0`.
const STORE_V0: [u32; 2] = [0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16];

/// `v_add_f16 v0, v7, v6` of 0x3c03 and 0x4000: an exact sum halfway between two halves.
const ADD_F16: &[u32] = &[
    0x7E0E_02FF,
    0x0000_3C03, // v_mov_b32 v7, 0x3c03
    0x7E0C_02FF,
    0x0000_4000, // v_mov_b32 v6, 0x4000
    0x6400_0D07, // v_add_f16 v0, v7, v6
];
/// `v_cvt_f16_f32 v0, v7` of an f32 that is no half.
const CVT_F16_F32: &[u32] = &[
    0x7E0E_02FF,
    0x3F80_1000, // v_mov_b32 v7, 1.0 + 2^-11 (a tie: rounds to even, 1.0)
    0x7E00_1507, // v_cvt_f16_f32 v0, v7
];
/// `v_add_f32 v0, v7, v6`: no 16-bit result anywhere.
const ADD_F32: &[u32] = &[
    0x7E0E_02FF,
    0x3F80_0001, // v_mov_b32 v7, 1.0 + 2^-23
    0x7E0C_02FF,
    0x4000_0000, // v_mov_b32 v6, 2.0
    0x0600_0D07, // v_add_f32 v0, v7, v6
];

/// A compute program of `threads` threads (console waves of 64 with `wave64`) with 4 root
/// constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`, then a store of
/// v0 to dword 0 of the UAV.
fn program(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.extend_from_slice(&STORE_V0);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `native` translated with `VICTORY_F16_NEAREST` and `VICTORY_F16_NATIVE` as given (for this
/// thread only), on a host of `host_wave`-lane waves.
fn translate(
    native: &NativeProgram,
    nearest: bool,
    f16_native: bool,
    host_wave: u32,
) -> HostShader {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, native).expect("io::raise");
    let bytes = crate::dxbc::build(&[]);
    let c = Container::parse(&bytes).expect("an empty container");
    let was = (
        lower::TEST_F16_NEAREST.replace(Some(nearest)),
        lower::TEST_F16_NATIVE.replace(Some(f16_native)),
        lower::TEST_HOST_WAVE.replace(Some(host_wave)),
    );
    let shader = super::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_F16_NEAREST.set(was.0);
    lower::TEST_F16_NATIVE.set(was.1);
    lower::TEST_HOST_WAVE.set(was.2);
    shader.expect("emit")
}

/// The four settings of the two switches: (nearest, native).
const SWITCHES: [(bool, bool); 4] = [(false, false), (false, true), (true, false), (true, true)];

// ---------------------------------------------------------------------------------------
// A reader of what `bitcode.rs` writes: LLVM bitstream with unabbreviated records only.

/// A block: its id, its records (code, operands) and the blocks inside it, in order.
#[derive(Debug, Default)]
struct Block {
    id: u64,
    records: Vec<(u64, Vec<u64>)>,
    blocks: Vec<Block>,
}

impl Block {
    fn child(&self, id: u64) -> &Block {
        self.blocks
            .iter()
            .find(|b| b.id == id)
            .unwrap_or_else(|| panic!("no block {id} in block {}", self.id))
    }
}

struct Bits<'a> {
    data: &'a [u8],
    /// In bits.
    pos: usize,
}

impl Bits<'_> {
    fn read(&mut self, width: u32) -> u64 {
        let mut v = 0u64;
        for k in 0..width as usize {
            let bit = self.pos + k;
            let byte = *self.data.get(bit / 8).expect("bitcode ends inside a field");
            v |= u64::from(byte >> (bit % 8) & 1) << k;
        }
        self.pos += width as usize;
        v
    }

    fn vbr(&mut self, width: u32) -> u64 {
        let hi = 1u64 << (width - 1);
        let (mut v, mut shift) = (0u64, 0);
        loop {
            let chunk = self.read(width);
            v |= (chunk & (hi - 1)) << shift;
            if chunk & hi == 0 {
                return v;
            }
            shift += width - 1;
            assert!(shift < 64, "a VBR field past 64 bits");
        }
    }

    fn align32(&mut self) {
        self.pos = self.pos.next_multiple_of(32);
    }

    /// The rest of a block whose abbreviation width is `width`, up to its end.
    fn block(&mut self, id: u64, width: u32, end: Option<usize>) -> Block {
        let mut b = Block {
            id,
            ..Block::default()
        };
        loop {
            if end.is_none() && self.pos >= self.data.len() * 8 {
                return b;
            }
            match self.read(width) {
                // END_BLOCK
                0 => {
                    self.align32();
                    assert_eq!(
                        Some(self.pos),
                        end,
                        "block {id} ends where its length word says"
                    );
                    return b;
                }
                // ENTER_SUBBLOCK
                1 => {
                    let child = self.vbr(8);
                    let w = self.vbr(4) as u32;
                    self.align32();
                    let words = self.read(32) as usize;
                    let child_end = self.pos + 32 * words;
                    b.blocks.push(self.block(child, w, Some(child_end)));
                }
                // UNABBREV_RECORD
                3 => {
                    let code = self.vbr(6);
                    let n = self.vbr(6);
                    let ops = (0..n).map(|_| self.vbr(6)).collect();
                    b.records.push((code, ops));
                }
                a => panic!("abbreviation {a} in block {id}: the writer defines none"),
            }
        }
    }
}

const MODULE_BLOCK: u64 = 8;
const CONSTANTS_BLOCK: u64 = 11;
const FUNCTION_BLOCK: u64 = 12;
const VALUE_SYMTAB_BLOCK: u64 = 14;
const TYPE_BLOCK: u64 = 17;

/// LLVM type codes (`TYPE_CODE_*`).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Type {
    Void,
    Half,
    Float,
    Int(u64),
    Func(u64),
    Other(u64),
}

/// LLVM cast opcodes (`CAST_*`) the native rounding uses.
const CAST_ZEXT: u64 = 1;
const CAST_FPTRUNC: u64 = 7;
const CAST_BITCAST: u64 = 11;

/// An emitted shader read back.
struct Read {
    shader_model: (u8, u8),
    dxil_version: u32,
    sfi0: u64,
    layout: String,
    types: Vec<Type>,
    /// The `i64` constants (the shader flags are the only one in these programs).
    i64s: Vec<i64>,
    /// `main`'s instructions: (record code, operands, local value number if it has one).
    insts: Vec<(u64, Vec<u64>, Option<u64>)>,
    /// Names in the module's symbol table (`dx.op.*` declarations, `main`).
    names: Vec<String>,
}

impl Read {
    fn of(shader: &HostShader) -> Read {
        let c = Container::parse(&shader.dxil).expect("the container parses");
        let part = |f: FourCC| c.part(f).unwrap_or_else(|| panic!("no {f:?} part")).data;
        let program = dxil::DxilProgram::parse(part(FourCC::DXIL)).expect("the DXIL part parses");
        assert!(program.is_standard_layout());
        let sfi0 = u64::from_le_bytes(part(FourCC::SFI0).try_into().expect("8 bytes of SFI0"));
        let bc = program.bitcode;
        assert_eq!(&bc[..4], b"BC\xC0\xDE");
        let mut bits = Bits { data: bc, pos: 32 };
        let top = bits.block(u64::MAX, 2, None);
        assert_eq!(
            bits.pos.next_multiple_of(32),
            bc.len() * 8,
            "nothing after the module"
        );
        assert_eq!(top.blocks.len(), 1, "one module block");
        let module = top.child(MODULE_BLOCK);

        let types: Vec<Type> = module
            .child(TYPE_BLOCK)
            .records
            .iter()
            .filter(|(code, _)| *code != 1 && *code != 19) // NUMENTRY, STRUCT_NAME
            .map(|(code, ops)| match code {
                2 => Type::Void,
                3 => Type::Float,
                10 => Type::Half,
                7 => Type::Int(ops[0]),
                21 => Type::Func(ops[1]),
                c => Type::Other(*c),
            })
            .collect();
        let layout = module
            .records
            .iter()
            .find(|(code, _)| *code == 3)
            .map(|(_, ops)| ops.iter().map(|&c| c as u8 as char).collect())
            .expect("a data layout");

        let mut i64s = Vec::new();
        let mut current = None;
        for (code, ops) in &module.child(CONSTANTS_BLOCK).records {
            match code {
                1 => current = Some(ops[0] as usize), // SETTYPE
                4 if current.is_some_and(|t| types[t] == Type::Int(64)) => {
                    let v = ops[0];
                    let m = (v >> 1) as i64;
                    i64s.push(if v & 1 != 0 { -m } else { m });
                }
                _ => {}
            }
        }

        // main's instructions, numbered by the values they make (as the writer counts:
        // every instruction but a store or a call of a void function).
        let mut insts = Vec::new();
        let mut next = 0u64;
        for (code, ops) in &module.child(FUNCTION_BLOCK).records {
            let value = match code {
                1 | 10 | 11 | 44 => false, // DECLAREBLOCKS, RET, BR, STORE
                34 => {
                    !matches!(types[ops[2] as usize], Type::Func(r) if types[r as usize] == Type::Void)
                }
                _ => true,
            };
            insts.push((*code, ops.clone(), value.then_some(next)));
            if value {
                next += 1;
            }
        }

        let names = module
            .blocks
            .iter()
            .filter(|b| b.id == VALUE_SYMTAB_BLOCK)
            .flat_map(|b| &b.records)
            .map(|(_, ops)| ops[1..].iter().map(|&c| c as u8 as char).collect())
            .collect();

        Read {
            shader_model: program.shader_model,
            dxil_version: program.dxil_version,
            sfi0,
            layout,
            types,
            i64s,
            insts,
            names,
        }
    }

    fn has_type(&self, t: &Type) -> bool {
        self.types.contains(t)
    }

    /// The casts in `main`: (opcode, destination type, own value number, operand's; that one
    /// negative for a value from before `main`'s code, a constant).
    fn casts(&self) -> Vec<(u64, Type, i64, i64)> {
        self.insts
            .iter()
            .filter(|(code, ..)| *code == 3)
            .map(|(_, ops, id)| {
                let id = id.expect("a cast makes a value") as i64;
                (
                    ops[2],
                    self.types[ops[1] as usize].clone(),
                    id,
                    id - ops[0] as i64,
                )
            })
            .collect()
    }

    /// The `fptrunc` to `half` → `bitcast` to `i16` → `zext` to `i32` chains in `main`.
    fn native_roundings(&self) -> usize {
        let casts = self.casts();
        let by_id = |id: i64| casts.iter().find(|c| c.2 == id);
        casts
            .iter()
            .filter(|(op, ty, ..)| *op == CAST_ZEXT && *ty == Type::Int(32))
            .filter(|zext| {
                by_id(zext.3).is_some_and(|bc| {
                    bc.0 == CAST_BITCAST
                        && bc.1 == Type::Int(16)
                        && by_id(bc.3).is_some_and(|t| t.0 == CAST_FPTRUNC && t.1 == Type::Half)
                })
            })
            .count()
    }

    fn flags(&self) -> u64 {
        assert_eq!(
            self.i64s.len(),
            1,
            "one i64 constant, the shader flags: {:?}",
            self.i64s
        );
        self.i64s[0] as u64
    }

    fn calls(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }
}

/// What a module without 16-bit types declares: nothing of them.
fn assert_min_precision(r: &Read, what: &str) {
    assert_eq!(r.layout, LAYOUT_MIN_PRECISION, "{what}: the data layout");
    assert!(
        !r.has_type(&Type::Half) && !r.has_type(&Type::Int(16)),
        "{what}: 16-bit types {:?}",
        r.types
    );
    assert!(
        r.casts().iter().all(|c| c.0 != CAST_FPTRUNC),
        "{what}: an fptrunc"
    );
    assert_eq!(
        r.flags() & (flags::LOW_PRECISION | flags::NATIVE_LOW_PRECISION),
        0,
        "{what}: flags"
    );
    assert_eq!(r.sfi0 & 0x4_0000, 0, "{what}: SFI0 NativeLowPrecision");
}

// ---------------------------------------------------------------------------------------

#[test]
fn the_flag_values_are_the_ones_dxc_writes_for_native_halves() {
    // reference-native.ll: `!9 = !{i32 0, i64 8388656, ...}` for a compute shader with a
    // structured UAV and native `half` code.
    assert_eq!(
        flags::RAW_BUFFERS | flags::LOW_PRECISION | flags::NATIVE_LOW_PRECISION,
        8_388_656
    );
    assert_eq!(
        LAYOUT_NATIVE_16,
        "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
    );
    // The clean-room DXC output itself, where the checkout has it (it is not committed:
    // shader dumps are kept out of the repository, `docs/asset-policy.md`).
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../reference-native.ll");
    let Ok(ll) = std::fs::read_to_string(&path) else {
        eprintln!(
            "no {}; checked against the constants above only",
            path.display()
        );
        return;
    };
    assert!(
        ll.contains(&format!("target datalayout = \"{LAYOUT_NATIVE_16}\"")),
        "the layout"
    );
    assert!(ll.contains("i64 8388656"), "the shader flags");
    assert!(
        ll.contains("Use native low precision"),
        "the requirement note"
    );
    assert!(
        ll.lines()
            .any(|l| l.contains("= fptrunc float ") && l.ends_with(" to half")),
        "fptrunc float to half"
    );
    // `!{!"cs", i32 6, i32 2}`: native 16-bit types need shader model 6.2; ours is 6.6.
    assert!(ll.contains("!{!\"cs\", i32 6, i32 2}"), "shader model 6.2");
}

#[test]
fn nearest_rounding_with_the_native_switch_is_one_fptrunc_and_declares_native_halves() {
    for (name, body) in [("v_add_f16", ADD_F16), ("v_cvt_f16_f32", CVT_F16_F32)] {
        let native = program(body, false, 32);
        let soft = Read::of(&translate(&native, true, false, 32));
        let fast = Read::of(&translate(&native, true, true, 32));

        assert_eq!(
            fast.native_roundings(),
            1,
            "{name}: one fptrunc/bitcast/zext chain"
        );
        assert!(
            fast.has_type(&Type::Half) && fast.has_type(&Type::Int(16)),
            "{name}: {:?}",
            fast.types
        );
        assert_eq!(fast.layout, LAYOUT_NATIVE_16, "{name}: the data layout");
        assert_eq!(
            fast.flags(),
            soft.flags() | flags::LOW_PRECISION | flags::NATIVE_LOW_PRECISION,
            "{name}: the shader flags are the software's and the two 16-bit ones"
        );
        assert_eq!(
            fast.sfi0,
            soft.sfi0 | 0x4_0000,
            "{name}: SFI0 adds NativeLowPrecision"
        );
        assert!(
            fast.shader_model >= (6, 2) && fast.dxil_version >= 0x102,
            "{name}: SM 6.2+ / DXIL 1.2+"
        );
        assert_eq!(
            (fast.shader_model, fast.dxil_version),
            (soft.shader_model, soft.dxil_version)
        );

        // The software rounding is gone: no host f32tof16 in the shader, and fewer
        // instructions than the software's dozen and more.
        assert!(
            soft.calls("dx.op.legacyF32ToF16"),
            "{name}: the software rounding uses f32tof16"
        );
        assert!(
            !fast.calls("dx.op.legacyF32ToF16"),
            "{name}: the native rounding needs none"
        );
        assert!(
            fast.insts.len() + 10 <= soft.insts.len(),
            "{name}: {} vs {} records",
            fast.insts.len(),
            soft.insts.len()
        );
        assert_min_precision(&soft, name);
    }
}

#[test]
fn the_native_switch_alone_changes_nothing() {
    // Without VICTORY_F16_NEAREST the host's conversion stays, byte for byte.
    for body in [ADD_F16, CVT_F16_F32, ADD_F32] {
        let native = program(body, false, 32);
        let off = translate(&native, false, false, 32);
        let alone = translate(&native, false, true, 32);
        assert_eq!(off.dxil, alone.dxil);
        assert_min_precision(&Read::of(&off), "host conversion");
    }
}

#[test]
fn f32_results_are_the_same_under_every_switch() {
    let native = program(ADD_F32, false, 32);
    let base = translate(&native, false, false, 32);
    for (nearest, f16_native) in SWITCHES {
        let s = translate(&native, nearest, f16_native, 32);
        assert_eq!(s.dxil, base.dxil, "nearest {nearest}, native {f16_native}");
    }
    assert_min_precision(&Read::of(&base), "v_add_f32");
}

#[test]
fn the_exact_wave_sum_of_halves_keeps_its_software_rounding() {
    // WaveActiveSum of 16-bit floats (`Lower::half_sum`), bit for bit the console's tree:
    // rounded by `f16_nearest` whatever the switches say (it is exact by default).
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| {
        if w >> 25 == 28 {
            w & 0x01FF_FFFF | 50 << 25
        } else {
            w
        }
    }));
    words.push(0x7E00_0207); // v_mov_b32 v0, s7
    let native = program(&words, true, 64);
    let soft = translate(&native, true, false, 64);
    let fast = translate(&native, true, true, 64);
    assert_eq!(soft.dxil, fast.dxil);
    let r = Read::of(&fast);
    assert_min_precision(&r, "the wave sum of halves");
    assert!(
        r.calls("dx.op.legacyF32ToF16"),
        "the sum still rounds in software"
    );
}

#[test]
fn a_module_declares_native_halves_only_where_code_uses_them() {
    use super::bitcode::{Cast, Module, Term};
    let build = |with_half: bool, reached: bool| {
        let mut m = Module::default();
        let (void, i32t) = (m.void(), m.int(32));
        let main_ty = m.func_ty(void, vec![]);
        m.ptr(main_ty);
        m.function("main", main_ty, None, true);
        let entry = m.new_block();
        let x = m.f32(1.5f32.to_bits());
        let other = m.new_block();
        let at = if reached { entry } else { other };
        if with_half {
            let (half, i16t) = (m.half(), m.int(16));
            let h = m.cast(at, Cast::FpTrunc, x, half);
            let b = m.cast(at, Cast::Bitcast, h, i16t);
            m.cast(at, Cast::ZExt, b, i32t);
        } else {
            m.cast(at, Cast::Bitcast, x, i32t);
        }
        m.terminate(entry, Term::Ret);
        m.terminate(other, Term::Ret);
        m
    };
    assert!(build(true, true).uses_16_bit());
    assert!(!build(false, true).uses_16_bit());
    // (made, but in a block nothing reaches: not written, not declared)
    assert!(!build(true, false).uses_16_bit());
}

// ---------------------------------------------------------------------------------------
// The hardware qualification: what the host GPU makes of special values both ways.

/// Lanes per sweep dispatch (one group, two console waves of 32).
const SWEEP_LANES: u32 = 64;

/// A sweep program: lane `t` converts the f32 with bits `s0 + (t << s1)` (two root
/// constants) with `v_cvt_f16_f32` and stores the half to dword `t` of the UAV and the
/// f32's bits to dword 64 + `t` (so the oracle works from what the GPU converted).
fn sweep_program() -> NativeProgram {
    const STORE_OFFEN: u32 = 0x38 << 26 | 0x1C << 18 | 1 << 12; // buffer_store_dword .. offen
    let words = [
        0x3404_0082, // v_lshlrev_b32 v2, 2, v0          (byte offset of lane t)
        0x3402_0001, // v_lshlrev_b32 v1, s1, v0
        0x4A02_0200, // v_add_nc_u32 v1, s0, v1          (the f32's bits)
        0x7E06_1501, // v_cvt_f16_f32 v3, v1
        STORE_OFFEN,
        0x80 << 24 | 2 << 16 | 3 << 8 | 2, // buffer_store_dword v3, v2, s[8:11], 0 offen
        STORE_OFFEN | (4 * SWEEP_LANES),
        0x80 << 24 | 2 << 16 | 1 << 8 | 2, // buffer_store_dword v1, v2, s[8:11], 0 offen offset:256
    ];
    let mut native = program(&words, false, SWEEP_LANES);
    // (`program` appends a store of v0 to dword 0: dropped, the sweep stores its own)
    let n = native.code.len();
    let end = native.code[n - 4..].to_vec();
    native.code.truncate(n - 12);
    native.code.extend(end);
    native
}

/// IEEE f32 to f16, round to nearest, ties to even, denormals kept; a NaN keeps its sign and
/// its payload's top bits, quieted (what a correct conversion gives; neither the console's nor
/// the host's NaN bits are known to be this).
fn f16_oracle(bits: u32) -> u16 {
    let sign = (bits >> 16 & 0x8000) as u16;
    let exp = (bits >> 23 & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp == 0xFF {
        return sign
            | if mant == 0 {
                0x7C00
            } else {
                0x7E00 | (mant >> 13) as u16
            };
    }
    if exp == 0 {
        return sign; // f32 denormals are below the smallest half denormal's half: zero
    }
    // The value is m * 2^(e) with m = 1.mant as a 24-bit integer.
    let m = u64::from(mant | 0x80_0000);
    let e = exp - 127 - 23;
    // In units of the smallest half denormal (2^-24): m * 2^(e + 24), rounded to nearest even.
    let shift = e + 24;
    let units = if shift >= 0 {
        m.checked_shl(shift as u32)
            .filter(|_| shift < 40)
            .unwrap_or(u64::MAX)
    } else if shift > -64 {
        let s = (-shift) as u32;
        let (q, r, half) = (m >> s, m & ((1u64 << s) - 1), 1u64 << (s - 1));
        if r > half || (r == half && q & 1 == 1) {
            q + 1
        } else {
            q
        }
    } else {
        0
    };
    if units < 1 << 10 {
        return sign | units as u16; // a denormal half (or zero)
    }
    // A normal half: renormalise to 11 significant bits, rounding again from m directly.
    let top = 63 - u64::leading_zeros(units) as i32; // position of the leading 1 in units
    let hexp = top - 10 + 1; // biased half exponent (units of 2^-24 → 2^(top-24) → bias 15)
    let s = top - 10 - shift; // bits of m below the half's last place
    let q = if s <= 0 {
        m << (-s) as u32
    } else {
        let s = s as u32;
        let (q, r, half) = (m >> s, m & ((1u64 << s) - 1), 1u64 << (s - 1));
        if r > half || (r == half && q & 1 == 1) {
            q + 1
        } else {
            q
        }
    };
    // (a carry out of the significand moves to the next exponent)
    let (q, hexp) = if q >> 11 != 0 {
        (q >> 1, hexp + 1)
    } else {
        (q, hexp)
    };
    if hexp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((hexp as u16) << 10 | (q as u16 & 0x3FF))
}

#[test]
fn the_oracle_rounds_as_ieee_says() {
    for (f, h) in [
        (0x3F80_0000, 0x3C00), // 1.0
        (0x3F80_1000, 0x3C00), // 1 + 2^-11: a tie, to even
        (0x3F80_3000, 0x3C02), // 1 + 3 * 2^-11: a tie, to even (up)
        (0x3F80_1001, 0x3C01), // just past the tie
        (0x477F_E000, 0x7BFF), // 65504
        (0x477F_EFFF, 0x7BFF), // just below 65520
        (0x477F_F000, 0x7C00), // 65520: a tie past the largest half, to infinity
        (0x7F80_0000, 0x7C00), // +inf
        (0xFF80_0000, 0xFC00), // -inf
        (0x3380_0000, 0x0001), // 2^-24: the smallest denormal
        (0x3300_0000, 0x0000), // 2^-25: a tie, to even (zero)
        (0x3300_0001, 0x0001), // just past it
        (0x3880_0000, 0x0400), // 2^-14: the smallest normal
        (0x387F_C000, 0x03FF), // 1023 * 2^-24: the largest denormal
        (0x387F_A000, 0x03FE), // 1022.5 * 2^-24: a tie, to even (down)
        (0x387F_E000, 0x0400), // 1023.5 * 2^-24: a tie with the smallest normal, to even
        (0x387F_F000, 0x0400), // 1023.75 * 2^-24
        (0x8000_0000, 0x8000), // -0
        (0x0000_0001, 0x0000), // an f32 denormal
        (0xBF80_1000, 0xBC00), // -(1 + 2^-11)
        (0xC000_0000, 0xC000), // -2
    ] {
        assert_eq!(f16_oracle(f), h, "{f:#010x}");
    }
}

/// The sweeps (s0, s1): lane `t` converts `s0 + (t << s1)`, and what each one is for.
const SWEEPS: &[(u32, u32, &str)] = &[
    (0x3F80_0FE0, 0, "ties around 1 + 2^-11 (normal)"),
    (0x3F80_1000, 13, "every half step from 1.0 (normal)"),
    (0xBF80_0FE0, 0, "negative ties around -(1 + 2^-11) (normal)"),
    (0xC000_0000, 12, "negative halves and half steps (normal)"),
    (
        0x477F_EFE0,
        0,
        "around 65520, where IEEE overflows to infinity",
    ),
    (0x477F_0000, 12, "65280 up past 65520 (overflow)"),
    (0x4780_0000, 20, "65536 and far above (overflow)"),
    (0xC77F_EFE0, 0, "around -65520 (overflow)"),
    (0x7F80_0000, 31, "+inf, -inf"),
    (0x0000_0000, 31, "+0, -0"),
    (0x7F80_0001, 16, "signalling NaNs, payloads"),
    (0x7FC0_0000, 15, "quiet NaNs, payloads"),
    (0xFFC0_0000, 15, "negative quiet NaNs"),
    (0x3300_0000, 18, "half denormals from 2^-25"),
    (
        0x387F_C000,
        9,
        "the largest half denormals into the smallest normals",
    ),
    (0xB300_0000, 18, "negative half denormals"),
    (0x0000_0001, 17, "f32 denormals"),
    (0x8000_0001, 17, "negative f32 denormals"),
];

/// What kind of case an f32 is for the comparison: normal cases must agree exactly.
fn class(f: u32) -> &'static str {
    let h = f16_oracle(f);
    if f & 0x7F80_0000 == 0x7F80_0000 && f & 0x7F_FFFF != 0 {
        "NaN"
    } else if f & 0x7F80_0000 == 0 && f & 0x7F_FFFF != 0 {
        "f32 denormal"
    } else if h & 0x7C00 == 0 && h & 0x3FF != 0 || (h & 0x7FFF == 0 && f & 0x7FFF_FFFF != 0) {
        "half denormal or underflow"
    } else {
        "normal"
    }
}

#[test]
fn the_sweep_program_lowers_to_one_rounding_either_way() {
    let native = sweep_program();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let names: Vec<&str> = p
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .map(|i| i.name())
        .collect();
    assert_eq!(
        names.iter().filter(|n| **n == "v_cvt_f16_f32").count(),
        1,
        "{names:?}"
    );
    assert_eq!(
        names.iter().filter(|n| **n == "buffer_store_dword").count(),
        2,
        "{names:?}"
    );
    let fast = Read::of(&translate(&native, true, true, 32));
    let soft = Read::of(&translate(&native, true, false, 32));
    assert_eq!(fast.native_roundings(), 1);
    assert_eq!(soft.native_roundings(), 0);
    assert_min_precision(&soft, "the software sweep");
    // Every sweep's inputs are of the kinds it says (normal sweeps stay normal).
    for &(base, shift, what) in SWEEPS {
        for t in 0..SWEEP_LANES {
            let f = base.wrapping_add(t.checked_shl(shift).unwrap_or(0));
            if what.ends_with("(normal)") {
                assert_eq!(class(f), "normal", "{what}: lane {t}, {f:#010x}");
            }
        }
    }
}

/// On the host GPU (D3D12; with D3DMetal on a Mac under the Windows build): every sweep
/// converted by the software rounding and by the native one. Normal cases, overflow and
/// infinities included, must give the IEEE result both ways; NaNs must stay NaNs of the same
/// sign; half denormals and f32 denormals are reported, with every difference, for a person
/// to judge (`STATUS.md`). Skipped without a device that takes native 16-bit programs.
#[cfg(windows)]
#[test]
fn on_the_card_native_rounding_matches_the_software_rounding() {
    use super::host::{Dispatch, Host, Root, View};
    let native = sweep_program();
    let soft = translate(&native, true, false, 32);
    let fast = translate(&native, true, true, 32);
    if let Ok(mut v) =
        super::validator::Validator::load(super::validator::Validator::default_path().as_deref())
    {
        for s in [&soft, &fast] {
            let mut copy = s.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
    }
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (Ok(soft_pso), Ok(fast_pso)) = (host.compute_pipeline(&soft), host.compute_pipeline(&fast))
    else {
        eprintln!("the device does not take the programs (native 16-bit types?); not run");
        return;
    };
    let run = |(root, pso): &(_, _), base: u32, shift: u32| -> (Vec<u32>, Vec<u16>) {
        let mut ud = vec![0u32; soft.user_data_dwords().max(2) as usize];
        (ud[0], ud[1]) = (base, shift);
        let out = host
            .dispatch(
                root,
                pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(2 * SWEEP_LANES)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let n = SWEEP_LANES as usize;
        (
            out[0][n..2 * n].to_vec(),
            out[0][..n].iter().map(|&h| h as u16).collect(),
        )
    };
    let mut failures = Vec::new();
    let mut report = std::collections::BTreeMap::<&str, (u32, u32, u32, u32, Vec<String>)>::new();
    for &(base, shift, what) in SWEEPS {
        let (inputs, soft_h) = run(&soft_pso, base, shift);
        let (inputs_again, fast_h) = run(&fast_pso, base, shift);
        assert_eq!(inputs, inputs_again);
        for t in 0..SWEEP_LANES as usize {
            let expected = base.wrapping_add((t as u32).checked_shl(shift).unwrap_or(0));
            assert_eq!(
                inputs[t], expected,
                "{what}: the root constants reach s0 and s1"
            );
            let (f, s, n, o) = (inputs[t], soft_h[t], fast_h[t], f16_oracle(inputs[t]));
            let line =
                format!("{what}: f32 {f:#010x} oracle {o:#06x} software {s:#06x} native {n:#06x}");
            let k = class(f);
            let e = report.entry(k).or_default();
            e.0 += 1;
            e.1 += u32::from(n != s);
            e.2 += u32::from(n != o);
            e.3 += u32::from(s != o);
            if n != s && e.4.len() < 8 {
                e.4.push(line.clone());
            }
            let is_nan = |h: u16| h & 0x7C00 == 0x7C00 && h & 0x3FF != 0;
            let ok = match k {
                "normal" => n == o && s == o,
                "NaN" => is_nan(n) && is_nan(s) && n & 0x8000 == s & 0x8000,
                _ => true,
            };
            if !ok {
                failures.push(line);
            }
        }
    }
    for (k, (count, ns, no, so, examples)) in &report {
        eprintln!(
            "{k}: {count} cases; native != software {ns}, native != IEEE {no}, software != IEEE {so}"
        );
        for e in examples {
            eprintln!("    {e}");
        }
    }
    assert!(
        failures.is_empty(),
        "{} cases differ where they must not:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
