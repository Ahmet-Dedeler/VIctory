//! Instructions of the console's CPU (Zen 2) that the host does not run.
//!
//! A title is compiled for one CPU and uses what it has without asking.
//! An Intel host lacks AMD's own instructions (CLZERO, MONITORX/MWAITX,
//! RDPRU, SSE4a's EXTRQ/INSERTQ/MOVNTSS/MOVNTSD); a translator (Apple's
//! Rosetta 2, under Wine on a Mac) lacks those and also ADX, RDSEED, RDPID
//! and CLWB. Each arrives as an illegal-instruction exception. The vectored
//! handler here ([`install`], ahead of the crash reporter's) emulates the
//! instruction against the faulting thread's registers and steps over it.
//!
//! A trap costs tens of microseconds under a translator, so the loops the
//! title is known to run over cache lines (`clzero`/`clwb`, then
//! `add reg, 0x40; cmp; jb`) are finished within the one trap.
//!
//! [`decode`] and [`exec`] do not depend on Windows and are tested on any
//! host.
//!
//! shortcut: the SHA extensions are not emulated (Grand Theft Auto VI's
//! executable has none); the handler leaves them to the crash reporter,
//! which names the site.

/// The registers an instruction reads and writes. GPR numbers are the
/// hardware's: 0 RAX, 1 RCX, 2 RDX, 3 RBX, 4 RSP, 5 RBP, 6 RSI, 7 RDI,
/// 8-15 R8-R15.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cpu {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub flags: u32,
    /// XMM registers as (low, high) quadwords.
    pub xmm: [(u64, u64); 16],
}

const CF: u32 = 1 << 0;
const PF: u32 = 1 << 2;
const AF: u32 = 1 << 4;
const ZF: u32 = 1 << 6;
const SF: u32 = 1 << 7;
const OF: u32 = 1 << 11;
const ARITHMETIC: u32 = CF | PF | AF | ZF | SF | OF;

/// The r/m operand of an instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rm {
    Reg(u8),
    /// base + index * 2^scale + disp, or the next instruction's address +
    /// disp (`rip`).
    Mem {
        base: Option<u8>,
        index: Option<u8>,
        scale: u8,
        disp: i32,
        rip: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// `0F 01 FC`: zero the 64-byte line holding RAX.
    Clzero,
    /// `0F 01 FA`: arm a wait on the line at RAX. Nothing to do: [`Op::Mwaitx`]
    /// does not wait for a write.
    Monitorx,
    /// `0F 01 FB`: wait for a write to the monitored line, or a timer. Every
    /// user checks its condition again afterwards, so a short wait is one
    /// valid outcome.
    Mwaitx,
    /// `0F 01 FD`: ECX 0 MPERF / 1 APERF into EDX:EAX; both count like the
    /// time-stamp counter here.
    Rdpru,
    /// `F3 0F C7 /7`: the processor number into a register.
    Rdpid(u8),
    /// `0F C7 /7`: a random number into a register of `size` bytes, CF set.
    Rdseed { reg: u8, size: u8 },
    /// `66 0F AE /6`: write a cache line back. Nothing to do.
    Clwb(Rm),
    /// `F2 0F 78 /r ib ib` and `F2 0F 79 /r`: the low `len` bits of `src`
    /// into `dst`'s low quadword at bit `idx` (`imm` None: both from
    /// `src`'s high quadword).
    Insertq {
        dst: u8,
        src: u8,
        imm: Option<(u8, u8)>,
    },
    /// `66 0F 78 /0 ib ib` and `66 0F 79 /r`: `len` bits from bit `idx` of
    /// `dst`'s low quadword, moved down (`src` Some: both from its low
    /// quadword).
    Extrq {
        dst: u8,
        src: Option<u8>,
        imm: (u8, u8),
    },
    /// `F3 0F 2B /r` (4 bytes) and `F2 0F 2B /r` (8): store the low bytes
    /// of an XMM register.
    Movnts { src: u8, to: Rm, bytes: u8 },
    /// `66 0F 38 F6 /r` (carry) and `F3 0F 38 F6 /r` (`overflow`):
    /// dst = dst + src + flag, changing only that flag.
    Adx {
        dst: u8,
        src: Rm,
        wide: bool,
        overflow: bool,
    },
}

impl Op {
    pub fn name(&self) -> &'static str {
        match self {
            Op::Clzero => "CLZERO",
            Op::Monitorx => "MONITORX",
            Op::Mwaitx => "MWAITX",
            Op::Rdpru => "RDPRU",
            Op::Rdpid(_) => "RDPID",
            Op::Rdseed { .. } => "RDSEED",
            Op::Clwb(_) => "CLWB",
            Op::Insertq { .. } => "INSERTQ",
            Op::Extrq { .. } => "EXTRQ",
            Op::Movnts { .. } => "MOVNTSS/SD",
            Op::Adx { .. } => "ADCX/ADOX",
        }
    }
}

/// A ModRM operand at `c[0]`: the reg field (unextended), the operand and
/// the bytes it takes.
fn modrm(c: &[u8], rex: u8) -> Option<(u8, Rm, usize)> {
    let m = *c.first()?;
    let (md, reg, rm) = (m >> 6, (m >> 3) & 7, m & 7);
    let b = (rex & 1) << 3;
    if md == 3 {
        return Some((reg, Rm::Reg(rm | b), 1));
    }
    let mut at = 1;
    let (mut base, mut index, mut scale, mut rip) = (Some(rm | b), None, 0, false);
    let mut disp_len = match md {
        1 => 1,
        2 => 4,
        _ => 0,
    };
    if rm == 4 {
        let sib = *c.get(at)?;
        at += 1;
        scale = sib >> 6;
        let i = ((sib >> 3) & 7) | ((rex & 2) << 2);
        index = (i != 4).then_some(i);
        base = Some((sib & 7) | b);
        if sib & 7 == 5 && md == 0 {
            base = None;
            disp_len = 4;
        }
    } else if rm == 5 && md == 0 {
        base = None;
        rip = true;
        disp_len = 4;
    }
    let disp = match disp_len {
        1 => i32::from(*c.get(at)? as i8),
        4 => i32::from_le_bytes([*c.get(at)?, *c.get(at + 1)?, *c.get(at + 2)?, *c.get(at + 3)?]),
        _ => 0,
    };
    Some((
        reg,
        Rm::Mem {
            base,
            index,
            scale,
            disp,
            rip,
        },
        at + disp_len,
    ))
}

/// Decodes the instruction at the start of `c` if it is one of ours: what
/// it is and its length. FS/GS overrides and 32-bit addressing are
/// declined (a wrong address is worse than the unhandled fault).
pub fn decode(c: &[u8]) -> Option<(Op, usize)> {
    let (mut i, mut p66, mut rep) = (0, false, 0u8);
    loop {
        match *c.get(i)? {
            0x66 => p66 = true,
            // The last of F2/F3 decides.
            0xF2 | 0xF3 => rep = c[i],
            0x26 | 0x2E | 0x36 | 0x3E => {} // no effect in 64-bit mode
            _ => break,
        }
        i += 1;
        if i > 4 {
            return None;
        }
    }
    let mut rex = 0;
    if c[i] & 0xF0 == 0x40 {
        rex = c[i];
        i += 1;
    }
    if *c.get(i)? != 0x0F {
        return None;
    }
    let wide = rex & 8 != 0;
    let r = (rex & 4) << 1;
    let op = *c.get(i + 1)?;
    let rest = c.get(i + 2..)?;
    let done = |op: Op, n: usize| Some((op, i + 2 + n));
    match (op, rep, p66) {
        (0x01, 0, false) => match *rest.first()? {
            0xFC => done(Op::Clzero, 1),
            0xFA => done(Op::Monitorx, 1),
            0xFB => done(Op::Mwaitx, 1),
            0xFD => done(Op::Rdpru, 1),
            _ => None,
        },
        (0xC7, 0xF3 | 0, _) => {
            let (reg, rm, n) = modrm(rest, rex)?;
            let Rm::Reg(to) = rm else { return None };
            if reg != 7 {
                return None;
            }
            if rep == 0xF3 {
                return done(Op::Rdpid(to), n);
            }
            let size = match (wide, p66) {
                (true, _) => 8,
                (false, true) => 2,
                _ => 4,
            };
            done(Op::Rdseed { reg: to, size }, n)
        }
        (0xAE, 0, true) => {
            let (reg, rm, n) = modrm(rest, rex)?;
            (reg == 6 && matches!(rm, Rm::Mem { .. })).then_some(())?;
            done(Op::Clwb(rm), n)
        }
        (0x78, 0xF2, _) => {
            let (reg, rm, n) = modrm(rest, rex)?;
            let Rm::Reg(src) = rm else { return None };
            let imm = (*rest.get(n)?, *rest.get(n + 1)?);
            done(
                Op::Insertq {
                    dst: reg | r,
                    src,
                    imm: Some(imm),
                },
                n + 2,
            )
        }
        (0x79, 0xF2, _) => {
            let (reg, rm, n) = modrm(rest, rex)?;
            let Rm::Reg(src) = rm else { return None };
            done(
                Op::Insertq {
                    dst: reg | r,
                    src,
                    imm: None,
                },
                n,
            )
        }
        (0x78, 0, true) => {
            let (reg, rm, n) = modrm(rest, rex)?;
            let Rm::Reg(dst) = rm else { return None };
            if reg != 0 {
                return None;
            }
            let imm = (*rest.get(n)?, *rest.get(n + 1)?);
            done(
                Op::Extrq {
                    dst,
                    src: None,
                    imm,
                },
                n + 2,
            )
        }
        (0x79, 0, true) => {
            let (reg, rm, n) = modrm(rest, rex)?;
            let Rm::Reg(src) = rm else { return None };
            done(
                Op::Extrq {
                    dst: reg | r,
                    src: Some(src),
                    imm: (0, 0),
                },
                n,
            )
        }
        (0x2B, 0xF2 | 0xF3, _) => {
            let (reg, rm, n) = modrm(rest, rex)?;
            matches!(rm, Rm::Mem { .. }).then_some(())?;
            let bytes = if rep == 0xF3 { 4 } else { 8 };
            done(
                Op::Movnts {
                    src: reg | r,
                    to: rm,
                    bytes,
                },
                n,
            )
        }
        (0x38, 0xF3, _) | (0x38, 0, true) if *rest.first()? == 0xF6 => {
            let (reg, rm, n) = modrm(&rest[1..], rex)?;
            done(
                Op::Adx {
                    dst: reg | r,
                    src: rm,
                    wide,
                    overflow: rep == 0xF3,
                },
                1 + n,
            )
        }
        _ => None,
    }
}

/// A field length of SSE4a's six bits: 0 is 64.
fn field(len: u8) -> u32 {
    match len & 63 {
        0 => 64,
        n => u32::from(n),
    }
}

fn ones(len: u32) -> u64 {
    if len >= 64 { u64::MAX } else { (1 << len) - 1 }
}

fn extrq(value: u64, len: u8, idx: u8) -> u64 {
    (value >> (idx & 63)) & ones(field(len))
}

fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (mask, idx) = (ones(field(len)), u32::from(idx & 63));
    (dst & !(mask << idx)) | ((src & mask) << idx)
}

/// The flags of `cmp a, b` (64-bit), other flags of `old` kept.
fn compare(a: u64, b: u64, old: u32) -> u32 {
    let r = a.wrapping_sub(b);
    let mut f = old & !ARITHMETIC;
    if a < b {
        f |= CF;
    }
    if r == 0 {
        f |= ZF;
    }
    if r >> 63 != 0 {
        f |= SF;
    }
    if ((a ^ b) & (a ^ r)) >> 63 != 0 {
        f |= OF;
    }
    if (a ^ b ^ r) & 0x10 != 0 {
        f |= AF;
    }
    if (r as u8).count_ones() % 2 == 0 {
        f |= PF;
    }
    f
}

impl Cpu {
    /// The address of a memory operand of an instruction ending at `next`.
    fn address(&self, rm: Rm, next: u64) -> Option<u64> {
        let Rm::Mem {
            base,
            index,
            scale,
            disp,
            rip,
        } = rm
        else {
            return None;
        };
        let mut at = if rip { next } else { 0 };
        if let Some(b) = base {
            at = at.wrapping_add(self.gpr[usize::from(b)]);
        }
        if let Some(i) = index {
            at = at.wrapping_add(self.gpr[usize::from(i)] << scale);
        }
        Some(at.wrapping_add(disp as i64 as u64))
    }
}

/// `add reg, 0x40` at `c[0]` (REX.W 83 /0 40): the register.
fn add_line(c: &[u8]) -> Option<u8> {
    let &[rex, 0x83, m, 0x40, ..] = c else {
        return None;
    };
    (rex & 0xFA == 0x48 && m & 0xF8 == 0xC0).then_some((m & 7) | ((rex & 1) << 3))
}

/// `cmp a, b` on registers at `c[0]` (REX.W 39 /r): (a, b).
fn cmp_regs(c: &[u8]) -> Option<(u8, u8)> {
    let &[rex, 0x39, m, ..] = c else {
        return None;
    };
    (rex & 0xF8 == 0x48 && m >> 6 == 3)
        .then_some(((m & 7) | ((rex & 1) << 3), ((m >> 3) & 7) | ((rex & 4) << 1)))
}

/// What the caller still has to do for an instruction [`exec`] ran.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Done,
    /// Give up the processor for a moment (MWAITX).
    Wait,
}

/// Where the instruction's non-register inputs come from.
pub struct Host {
    /// A counter like the time-stamp counter.
    pub ticks: fn() -> u64,
    pub processor: fn() -> u32,
    pub random: fn() -> u64,
}

/// Runs the instruction `op` of `len` bytes at `cpu.rip`, whose bytes and
/// the ones after it are `code` and the ones before it `before` (both for
/// the loops finished in one go), and moves `rip` past it (past the loop).
///
/// # Safety
/// Memory the instruction writes (the line at RAX for CLZERO, the operand
/// of MOVNTSS/SD) must be writable, and a memory source of ADX readable, as
/// for the real instruction.
pub unsafe fn exec(
    cpu: &mut Cpu,
    op: Op,
    len: usize,
    code: &[u8],
    before: &[u8],
    host: &Host,
) -> Outcome {
    let next = cpu.rip.wrapping_add(len as u64);
    let after = code.get(len..).unwrap_or_default();
    let mut outcome = Outcome::Done;
    cpu.rip = next;
    match op {
        Op::Clzero => {
            // SAFETY: the caller's contract, for each line.
            let zero = |at: u64| unsafe { ((at & !63) as *mut u8).write_bytes(0, 64) };
            zero(cpu.gpr[0]);
            // `clzero; add rax, 0x40; cmp rax, end; jb clzero`
            if len == 3
                && add_line(after) == Some(0)
                && let Some((0, end)) = cmp_regs(&after[4..])
                && after[7..].starts_with(&[0x72, 0xF4])
            {
                let end = cpu.gpr[usize::from(end)];
                loop {
                    cpu.gpr[0] = cpu.gpr[0].wrapping_add(64);
                    if cpu.gpr[0] >= end {
                        break;
                    }
                    zero(cpu.gpr[0]);
                }
                cpu.flags = compare(cpu.gpr[0], end, cpu.flags);
                cpu.rip = next + 9;
            }
            // `mov rax, at; add at, 0x40; clzero; cmp at, end; jb mov`
            else if len == 3
                && let Some((at, end)) = cmp_regs(after)
                && after[3..].starts_with(&[0x72, 0xF1])
                && at != 0
                && end != 0
                && end != at
                && let [.., rex, 0x89, m, a0, a1, a2, a3] = *before
                && rex & 0xFB == 0x48
                && m == 0xC0 | (at & 7) << 3
                && rex & 4 == (at & 8) >> 1
                && add_line(&[a0, a1, a2, a3]) == Some(at)
            {
                let (at, end) = (usize::from(at), cpu.gpr[usize::from(end)]);
                while cpu.gpr[at] < end {
                    cpu.gpr[0] = cpu.gpr[at];
                    cpu.gpr[at] = cpu.gpr[at].wrapping_add(64);
                    zero(cpu.gpr[0]);
                }
                cpu.flags = compare(cpu.gpr[at], end, cpu.flags);
                cpu.rip = next + 5;
            }
        }
        Op::Monitorx => {}
        Op::Mwaitx => outcome = Outcome::Wait,
        Op::Rdpru => {
            let t = (host.ticks)();
            cpu.gpr[0] = t & 0xFFFF_FFFF;
            cpu.gpr[2] = t >> 32;
            cpu.flags &= !CF;
        }
        Op::Rdpid(reg) => cpu.gpr[usize::from(reg)] = u64::from((host.processor)()),
        Op::Rdseed { reg, size } => {
            let (old, new) = (cpu.gpr[usize::from(reg)], (host.random)());
            cpu.gpr[usize::from(reg)] = match size {
                8 => new,
                4 => new & 0xFFFF_FFFF,
                _ => (old & !0xFFFF) | (new & 0xFFFF),
            };
            cpu.flags = (cpu.flags & !ARITHMETIC) | CF;
        }
        Op::Clwb(rm) => {
            // `clwb [at]; add at, 0x40; cmp at, last; jbe clwb`
            if let Rm::Mem {
                base: Some(base),
                index: None,
                disp: 0,
                rip: false,
                ..
            } = rm
                && add_line(after) == Some(base)
                && let Some((at, last)) = cmp_regs(&after[4..])
                && at == base
                && last != base
                && after[7..].first() == Some(&0x76)
                && after.get(8).is_some_and(|&rel| rel as i8 as i64 == -(len as i64 + 9))
            {
                let (at, last) = (usize::from(at), cpu.gpr[usize::from(last)]);
                loop {
                    cpu.gpr[at] = cpu.gpr[at].wrapping_add(64);
                    if cpu.gpr[at] > last {
                        break;
                    }
                }
                cpu.flags = compare(cpu.gpr[at], last, cpu.flags);
                cpu.rip = next + 9;
            }
        }
        Op::Insertq { dst, src, imm } => {
            let s = cpu.xmm[usize::from(src)];
            let (len, idx) = imm.unwrap_or((s.1 as u8, (s.1 >> 8) as u8));
            let d = &mut cpu.xmm[usize::from(dst)];
            // The high quadword is undefined; the hardware leaves zero.
            *d = (insertq(d.0, s.0, len, idx), 0);
        }
        Op::Extrq { dst, src, imm } => {
            let (len, idx) = match src {
                Some(s) => {
                    let s = cpu.xmm[usize::from(s)].0;
                    (s as u8, (s >> 8) as u8)
                }
                None => imm,
            };
            let d = &mut cpu.xmm[usize::from(dst)];
            *d = (extrq(d.0, len, idx), 0);
        }
        Op::Movnts { src, to, bytes } => {
            let value = cpu.xmm[usize::from(src)].0.to_le_bytes();
            if let Some(at) = cpu.address(to, next) {
                // SAFETY: the caller's contract.
                unsafe {
                    (at as *mut u8).copy_from_nonoverlapping(value.as_ptr(), usize::from(bytes))
                };
            }
        }
        Op::Adx {
            dst,
            src,
            wide,
            overflow,
        } => {
            let b = match src {
                Rm::Reg(r) => cpu.gpr[usize::from(r)],
                // SAFETY: the caller's contract.
                m => cpu.address(m, next).map_or(0, |at| unsafe {
                    if wide {
                        (at as *const u64).read_unaligned()
                    } else {
                        u64::from((at as *const u32).read_unaligned())
                    }
                }),
            };
            let flag = if overflow { OF } else { CF };
            let carry = u64::from(cpu.flags & flag != 0);
            let a = cpu.gpr[usize::from(dst)];
            let (sum, out) = if wide {
                let (s, c1) = a.overflowing_add(b);
                let (s, c2) = s.overflowing_add(carry);
                (s, c1 | c2)
            } else {
                let s = (a & 0xFFFF_FFFF) + (b & 0xFFFF_FFFF) + carry;
                (s & 0xFFFF_FFFF, s >> 32 != 0)
            };
            cpu.gpr[usize::from(dst)] = sum;
            cpu.flags = (cpu.flags & !flag) | if out { flag } else { 0 };
        }
    }
    outcome
}

#[cfg(windows)]
mod host {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};

    use windows_sys::Win32::Foundation::EXCEPTION_ILLEGAL_INSTRUCTION;
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcessorNumber, SwitchToThread};

    use super::{Cpu, Host, Outcome, decode, exec};

    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    /// Sites seen (instruction address) and how often, for the log.
    const SITES: usize = 64;
    static SITE: [AtomicUsize; SITES] = [const { AtomicUsize::new(0) }; SITES];
    static COUNT: [AtomicU64; SITES] = [const { AtomicU64::new(0) }; SITES];

    /// Counts a trap at `at`; true when it is worth a log line (the first,
    /// then every tenfold).
    fn count(at: usize) -> Option<u64> {
        for (site, count) in SITE.iter().zip(&COUNT) {
            let seen = site.load(Relaxed);
            if seen == at
                || (seen == 0 && site.compare_exchange(0, at, Relaxed, Relaxed).is_ok())
                || site.load(Relaxed) == at
            {
                let n = count.fetch_add(1, Relaxed) + 1;
                let mut step = 1;
                while step < n {
                    step *= 10;
                }
                return (step == n).then_some(n);
            }
        }
        None
    }

    fn ticks() -> u64 {
        // SAFETY: plain instruction, present on every x86-64 CPU.
        unsafe { core::arch::x86_64::_rdtsc() }
    }

    fn processor() -> u32 {
        // SAFETY: plain query.
        unsafe { GetCurrentProcessorNumber() }
    }

    fn random() -> u64 {
        use std::hash::{BuildHasher, Hasher};
        // Keys from the system's generator, a new pair per call.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(ticks());
        h.finish()
    }

    static HOST: Host = Host {
        ticks,
        processor,
        random,
    };

    fn load(c: &CONTEXT) -> Cpu {
        // SAFETY: the legacy save area is valid in an exception's context.
        let x = unsafe { &c.Anonymous.FltSave.XmmRegisters };
        Cpu {
            gpr: [
                c.Rax, c.Rcx, c.Rdx, c.Rbx, c.Rsp, c.Rbp, c.Rsi, c.Rdi, c.R8, c.R9, c.R10, c.R11,
                c.R12, c.R13, c.R14, c.R15,
            ],
            rip: c.Rip,
            flags: c.EFlags,
            xmm: std::array::from_fn(|i| (x[i].Low, x[i].High as u64)),
        }
    }

    fn store(cpu: &Cpu, c: &mut CONTEXT) {
        let g = &cpu.gpr;
        (c.Rax, c.Rcx, c.Rdx, c.Rbx) = (g[0], g[1], g[2], g[3]);
        (c.Rsp, c.Rbp, c.Rsi, c.Rdi) = (g[4], g[5], g[6], g[7]);
        (c.R8, c.R9, c.R10, c.R11) = (g[8], g[9], g[10], g[11]);
        (c.R12, c.R13, c.R14, c.R15) = (g[12], g[13], g[14], g[15]);
        c.Rip = cpu.rip;
        c.EFlags = cpu.flags;
        // SAFETY: as in `load`.
        let x = unsafe { &mut c.Anonymous.FltSave.XmmRegisters };
        for (to, from) in x.iter_mut().zip(&cpu.xmm) {
            (to.Low, to.High) = (from.0, from.1 as i64);
        }
    }

    unsafe extern "system" fn on_illegal(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let (record, context) = unsafe { (&*(*info).ExceptionRecord, &mut *(*info).ContextRecord) };
        if record.ExceptionCode != EXCEPTION_ILLEGAL_INSTRUCTION {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = context.Rip as usize;
        // An instruction is at most 15 bytes; the loops finished in one
        // trap need up to 13 more. Code never ends within 32 bytes of the
        // end of its last page here: images end in other sections.
        // SAFETY: the bytes the processor just fetched and what follows.
        let code = unsafe { std::slice::from_raw_parts(at as *const u8, 32) };
        let Some((op, len)) = decode(code) else {
            return EXCEPTION_CONTINUE_SEARCH;
        };
        if let Some(n) = count(at) {
            eprintln!(
                "VIctory: {} at {} emulated ({n} so far)",
                op.name(),
                crate::locate(at)
            );
        }
        // SAFETY: the instructions that led here, in the same function.
        let before = unsafe { std::slice::from_raw_parts((at - 8) as *const u8, 8) };
        let mut cpu = load(context);
        // SAFETY: the title's own instruction, with its own operands.
        let outcome = unsafe { exec(&mut cpu, op, len, code, before, &HOST) };
        store(&cpu, context);
        if outcome == Outcome::Wait {
            // SAFETY: plain yield.
            unsafe { SwitchToThread() };
        }
        EXCEPTION_CONTINUE_EXECUTION
    }

    /// Installs the handler, ahead of every handler installed before it.
    pub fn install() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        // SAFETY: the handler only reads the faulting code and changes the
        // faulting thread's registers.
        ONCE.call_once(|| unsafe {
            AddVectoredExceptionHandler(1, Some(on_illegal));
        });
    }
}

#[cfg(windows)]
pub use host::install;

#[cfg(test)]
mod tests {
    use super::*;

    static HOST: Host = Host {
        ticks: || 0x1_0000_0002,
        processor: || 5,
        random: || 0x1122_3344_5566_7788,
    };

    fn run(cpu: &mut Cpu, code: &[u8]) -> Outcome {
        run_after(cpu, &[], code)
    }

    fn run_after(cpu: &mut Cpu, before: &[u8], code: &[u8]) -> Outcome {
        let (op, len) = decode(code).expect("one of ours");
        // SAFETY: the tests point memory operands at their own buffers.
        unsafe { exec(cpu, op, len, code, before, &HOST) }
    }

    #[test]
    fn decodes_what_the_title_uses() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((Op::Clzero, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((Op::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((Op::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((Op::Rdpru, 3)));
        assert_eq!(
            decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]),
            Some((
                Op::Insertq {
                    dst: 0,
                    src: 1,
                    imm: Some((16, 16))
                },
                6
            ))
        );
        assert_eq!(
            decode(&[0x66, 0x0F, 0x78, 0xC2, 0x04, 0x08]),
            Some((
                Op::Extrq {
                    dst: 2,
                    src: None,
                    imm: (4, 8)
                },
                6
            ))
        );
        assert!(matches!(
            decode(&[0x66, 0x0F, 0xAE, 0x31]),
            Some((Op::Clwb(_), 4))
        ));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((Op::Rdpid(9), 5)));
        assert_eq!(
            decode(&[0x48, 0x0F, 0xC7, 0xF8]),
            Some((Op::Rdseed { reg: 0, size: 8 }, 4))
        );
        // Not ours: clflushopt (/7), VMREAD (no prefix), a plain mov, cut short.
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x39]), None);
        assert_eq!(decode(&[0x0F, 0x78, 0xC1, 0x08, 0x10]), None);
        assert_eq!(decode(&[0x48, 0x89, 0xC8]), None);
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08]), None);
    }

    #[test]
    fn bit_fields_move_as_on_the_hardware() {
        let mut cpu = Cpu::default();
        cpu.xmm[0] = (0xAAAA_BBBB_CCCC_DDDD, 7);
        cpu.xmm[1] = (0x1234_5678, 9);
        run(&mut cpu, &[0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]);
        assert_eq!(cpu.xmm[0], (0xAAAA_BBBB_5678_DDDD, 0));
        assert_eq!(cpu.rip, 6);
        run(&mut cpu, &[0x66, 0x0F, 0x78, 0xC0, 0x08, 0x10]);
        assert_eq!(cpu.xmm[0], (0x78, 0));
        assert_eq!(insertq(0, u64::MAX, 0, 0), u64::MAX);
        assert_eq!(extrq(0xFF00, 8, 8), 0xFF);
    }

    #[test]
    fn clzero_loops_finish_in_one_step() {
        #[repr(align(64))]
        struct Lines([u8; 64 * 6]);
        let mut lines = Lines([0xEE; 64 * 6]);
        let base = lines.0.as_mut_ptr() as u64;
        // clzero; add rax, 0x40; cmp rax, r10; jb -12
        let code = [
            0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x4C, 0x39, 0xD0, 0x72, 0xF4,
        ];
        let mut cpu = Cpu::default();
        cpu.rip = 0x1000;
        cpu.gpr[0] = base + 64;
        cpu.gpr[10] = base + 64 * 4;
        run(&mut cpu, &code);
        assert_eq!((cpu.rip, cpu.gpr[0]), (0x100C, base + 64 * 4));
        assert_eq!(cpu.flags & (CF | ZF), ZF);
        assert!(lines.0[..64].iter().all(|&b| b == 0xEE));
        assert!(lines.0[64..64 * 4].iter().all(|&b| b == 0));
        assert!(lines.0[64 * 4..].iter().all(|&b| b == 0xEE));
        // (mov rax, r9; add r9, 0x40;) clzero; cmp r9, r10; jb -15
        let code = [0x0F, 0x01, 0xFC, 0x4D, 0x39, 0xD1, 0x72, 0xF1];
        lines.0.fill(0xEE);
        cpu.gpr[0] = base;
        cpu.gpr[9] = base + 64;
        cpu.gpr[10] = base + 64 * 3;
        cpu.rip = 0x2000;
        run_after(&mut cpu, &[0x4C, 0x89, 0xC8, 0x49, 0x83, 0xC1, 0x40], &code);
        assert_eq!((cpu.rip, cpu.gpr[9]), (0x2008, base + 64 * 3));
        assert!(lines.0[..64 * 3].iter().all(|&b| b == 0));
        assert!(lines.0[64 * 3..].iter().all(|&b| b == 0xEE));
        // A lone clzero zeroes the line holding RAX.
        lines.0.fill(0xEE);
        cpu.gpr[0] = base + 64 * 5 + 17;
        run(&mut cpu, &[0x0F, 0x01, 0xFC, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90]);
        assert!(lines.0[..64 * 5].iter().all(|&b| b == 0xEE));
        assert!(lines.0[64 * 5..].iter().all(|&b| b == 0));
    }

    #[test]
    fn clwb_loops_and_counters() {
        // clwb [rax]; add rax, 0x40; cmp rax, rcx; jbe -13
        let code = [
            0x66, 0x0F, 0xAE, 0x30, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xC8, 0x76, 0xF3,
        ];
        let mut cpu = Cpu::default();
        cpu.gpr[0] = 0x1000;
        cpu.gpr[1] = 0x10FF;
        run(&mut cpu, &code);
        assert_eq!((cpu.rip, cpu.gpr[0]), (13, 0x1100));
        assert_eq!(cpu.flags & (CF | ZF), 0);
        assert_eq!(run(&mut cpu, &[0x0F, 0x01, 0xFB]), Outcome::Wait);
        run(&mut cpu, &[0x0F, 0x01, 0xFD]);
        assert_eq!((cpu.gpr[0], cpu.gpr[2]), (2, 1));
        run(&mut cpu, &[0xF3, 0x0F, 0xC7, 0xFB]);
        assert_eq!(cpu.gpr[3], 5);
        run(&mut cpu, &[0x0F, 0xC7, 0xFE]);
        assert_eq!((cpu.gpr[6], cpu.flags & CF), (0x5566_7788, CF));
    }

    #[test]
    fn adx_and_scalar_stores() {
        let mut cpu = Cpu::default();
        cpu.gpr[0] = u64::MAX;
        cpu.gpr[1] = 1;
        cpu.flags = CF | ZF;
        // adcx rax, rcx
        run(&mut cpu, &[0x66, 0x48, 0x0F, 0x38, 0xF6, 0xC1]);
        assert_eq!((cpu.gpr[0], cpu.flags), (1, CF | ZF));
        // adox eax, ecx: no overflow flag in, none out
        run(&mut cpu, &[0xF3, 0x0F, 0x38, 0xF6, 0xC1]);
        assert_eq!((cpu.gpr[0], cpu.flags), (2, CF | ZF));
        let mut out = [0u8; 16];
        cpu.gpr[3] = out.as_mut_ptr() as u64;
        cpu.xmm[2] = (0x0102_0304_0506_0708, 0);
        // movntsd [rbx+8], xmm2; movntss [rbx], xmm2
        run(&mut cpu, &[0xF2, 0x0F, 0x2B, 0x53, 0x08]);
        run(&mut cpu, &[0xF3, 0x0F, 0x2B, 0x13]);
        assert_eq!(out[8..], 0x0102_0304_0506_0708u64.to_le_bytes());
        assert_eq!(out[..8], [8, 7, 6, 5, 0, 0, 0, 0]);
    }
}
