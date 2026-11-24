//! `clzero` sites turned into calls of ordinary code, so that each site
//! raises one exception instead of one per execution.
//!
//! A title built for Zen 2 clears its work areas with `clzero` loops. On a
//! CPU without the instruction every execution is an illegal-instruction
//! exception our handler answers; that costs about a microsecond on
//! Windows, but 110-300 us under a binary translator below Wine (Rosetta 2
//! on a Mac: a signal, Wine's dispatch, our handler, a context restore).
//! Grand Theft Auto VI clears 29 KiB a line at a time in every job its
//! worker threads start: 464 exceptions, 50 ms and more for what takes the
//! console a microsecond, and its frames waited for those jobs.
//!
//! `clzero` is three bytes, a call five, so a site takes the instruction
//! after it along: the compilers' loops go on with `add rax, 0x40` or with
//! a compare of two registers. On the first exception at a site whose next
//! instruction is one we can move (two registers or a register and an
//! immediate, not the stack pointer), a helper is written near the site:
//!
//! ```text
//! helper:  push rax; pushfq
//!          and  rax, -64
//!          mov  qword [rax], 0   (eight of them)
//!          popfq; pop rax
//! moved:   <the instruction after clzero>
//!          ret
//! ```
//!
//! and the site becomes `call helper` padded with `nop`. The thread that
//! raised the exception has the line zeroed by the handler and goes on at
//! `moved` with the return address pushed, as if it had made the call: so
//! does every thread that raises one at the site later, whatever the
//! site's bytes are by then (another thread may have written the call
//! while this one's exception was on its way).
//!
//! The call is written only once the site has been known for a while
//! ([`GRACE_MS`]): a thread the handler sent back to the next instruction
//! before the site was known has long executed it by then, so no thread is
//! inside the bytes that change. Nothing but an exception return ever
//! lands on that instruction: a site is left alone (and answered at every
//! exception, as before) if any branch in its function could.
//!
//! The helpers have unwind data, so a fault inside one (the title clearing
//! memory it does not own) unwinds and reports like one at the site.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, FlushInstructionCache, IMAGE_RUNTIME_FUNCTION_ENTRY, RtlAddFunctionTable,
    RtlLookupFunctionEntry,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READWRITE, VirtualAlloc,
    VirtualProtect, VirtualQuery,
};
use windows_sys::Win32::System::SystemInformation::GetTickCount64;
use windows_sys::Win32::System::Threading::GetCurrentProcess;

/// A site is rewritten this long after it was first seen, at the earliest.
const GRACE_MS: u64 = 50;
/// Sites remembered (a title has a handful).
const SITES: usize = 256;
/// Bytes a helper and its unwind data take in the pool.
const SLOT: usize = 160;
const POOL: usize = 64 << 10;

struct Site {
    /// The address of the `clzero`; 0 while the slot is free. Written last.
    at: AtomicUsize,
    /// Where a thread goes on after the handler zeroed the line: the moved
    /// instruction in the helper. 0 for a site that is answered in place.
    moved: AtomicUsize,
    /// The address after the bytes the call replaces.
    after: AtomicUsize,
    /// The helper's first instruction.
    helper: AtomicUsize,
    /// When the site was first seen (tick count, ms).
    seen: AtomicU64,
    /// The site's first eight bytes when first seen.
    bytes: AtomicU64,
    written: AtomicBool,
}

static TABLE: [Site; SITES] = [const {
    Site {
        at: AtomicUsize::new(0),
        moved: AtomicUsize::new(0),
        after: AtomicUsize::new(0),
        helper: AtomicUsize::new(0),
        seen: AtomicU64::new(0),
        bytes: AtomicU64::new(0),
        written: AtomicBool::new(false),
    }
}; SITES];
static COUNT: AtomicUsize = AtomicUsize::new(0);
/// Sites whose call has been written, for the log.
pub(crate) static WRITTEN: AtomicUsize = AtomicUsize::new(0);

/// Helper memory: (base, bytes used), each pool within reach of the sites
/// it serves.
static POOLS: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_ISA_PATCH").as_deref() != Ok("0"))
}

fn find(at: usize) -> Option<&'static Site> {
    let n = COUNT.load(Ordering::Acquire);
    TABLE[..n].iter().find(|s| s.at.load(Ordering::Acquire) == at)
}

/// The length of the instruction at the start of `code` if it is one a
/// helper can hold: an operation on two general registers or on a register
/// and an immediate, without the stack pointer.
fn movable(code: &[u8]) -> Option<usize> {
    let (rex, at) = match *code.first()? {
        r @ 0x40..=0x4f => (r, 1),
        _ => (0, 0),
    };
    let opcode = *code.get(at)?;
    let modrm = *code.get(at + 1)?;
    if modrm >> 6 != 3 {
        return None;
    }
    let reg = (modrm >> 3) & 7 | (rex & 4) << 1;
    let rm = modrm & 7 | (rex & 1) << 3;
    let immediate = match opcode {
        // add, or, adc, sbb, and, sub, xor, cmp, test, mov: r/m and r.
        0x01 | 0x03 | 0x09 | 0x0b | 0x11 | 0x13 | 0x19 | 0x1b | 0x21 | 0x23 | 0x29 | 0x2b
        | 0x31 | 0x33 | 0x39 | 0x3b | 0x85 | 0x89 | 0x8b => {
            if reg == 4 {
                return None;
            }
            0
        }
        // The same group with an immediate: the reg field picks the operation.
        0x83 => 1,
        0x81 => 4,
        _ => return None,
    };
    if rm == 4 {
        return None;
    }
    let len = at + 2 + immediate;
    (len <= code.len()).then_some(len)
}

/// Whether some branch in `code` (which starts at address `base`) may land
/// strictly inside `[from, to)`: every byte offset is read as the start of
/// a relative jump or call, so data and operands that look like one count
/// too (a site is then left alone).
fn branched_into(code: &[u8], base: usize, from: usize, to: usize) -> bool {
    let inside = |target: usize| target > from && target < to;
    for (i, &b) in code.iter().enumerate() {
        let at = base + i;
        let rel8 = |skip: usize| {
            code.get(i + skip - 1)
                .map(|&d| (at + skip).wrapping_add_signed(isize::from(d as i8)))
        };
        let rel32 = |skip: usize| {
            code.get(i + skip - 4..i + skip).map(|d| {
                let d = i32::from_le_bytes(d.try_into().unwrap_or([0; 4]));
                (at + skip).wrapping_add_signed(d as isize)
            })
        };
        let target = match b {
            0x70..=0x7f | 0xe0..=0xe3 | 0xeb => rel8(2),
            0xe8 | 0xe9 => rel32(5),
            0x0f if matches!(code.get(i + 1), Some(0x80..=0x8f)) => rel32(6),
            _ => None,
        };
        if target.is_some_and(inside) {
            return true;
        }
    }
    false
}

/// The code a branch into the site could come from: its function by the
/// unwind tables, or the committed memory within 512 bytes of it.
fn surroundings(at: usize) -> Option<(usize, &'static [u8])> {
    let mut image = 0;
    // SAFETY: looks up the function table entry covering the site, if any.
    let entry = unsafe { RtlLookupFunctionEntry(at as u64, &mut image, std::ptr::null_mut()) };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    if unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) } == 0
        || info.State != MEM_COMMIT
    {
        return None;
    }
    let (low, high) = (info.BaseAddress as usize, info.BaseAddress as usize + info.RegionSize);
    let (from, to) = if entry.is_null() {
        (at.saturating_sub(512), at + 512)
    } else {
        // SAFETY: a live table entry of a loaded image.
        let e = unsafe { &*entry };
        (image as usize + e.BeginAddress as usize, image as usize + e.EndAddress as usize)
    };
    let (from, to) = (from.max(low), to.min(high));
    if at < from || at + 16 > to {
        return None;
    }
    // SAFETY: committed memory of one region holding the site's code.
    Some((from, unsafe { std::slice::from_raw_parts(from as *const u8, to - from) }))
}

/// Room for a helper within a call's reach of `site`: from a pool that
/// has it, or a new pool placed near the site.
fn room(pools: &mut Vec<(usize, usize)>, site: usize) -> Option<usize> {
    const REACH: usize = 0x7000_0000;
    for (base, used) in pools.iter_mut() {
        if base.abs_diff(site) < REACH && *used + SLOT <= POOL {
            *used += SLOT;
            return Some(*base + *used - SLOT);
        }
    }
    // Free address space near the site, in steps of 16 MiB, below first.
    let start = site & !0xffff;
    for step in 1..=96usize {
        for candidate in [start.checked_sub(step << 24), start.checked_add(step << 24)] {
            let Some(candidate) = candidate else { continue };
            // SAFETY: asks for fresh memory at an address; refused if taken.
            let base = unsafe {
                VirtualAlloc(
                    candidate as *const _,
                    POOL,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_EXECUTE_READWRITE,
                )
            } as usize;
            if base != 0 {
                pools.push((base, SLOT));
                return Some(base);
            }
        }
    }
    None
}

/// Writes the helper for a site followed by the instruction `next` at
/// `slot`, with its unwind data, and returns the address of the moved
/// instruction in it.
///
/// # Safety
/// `slot` is [`SLOT`] writable, executable bytes nobody else uses.
unsafe fn write_helper(slot: usize, next: &[u8]) -> usize {
    let mut code = Vec::with_capacity(SLOT);
    code.extend_from_slice(&[0x50, 0x9c]); // push rax; pushfq
    code.extend_from_slice(&[0x48, 0x83, 0xe0, 0xc0]); // and rax, -64
    code.extend_from_slice(&[0x48, 0xc7, 0x00, 0, 0, 0, 0]); // mov qword [rax], 0
    for offset in (8u8..64).step_by(8) {
        code.extend_from_slice(&[0x48, 0xc7, 0x40, offset, 0, 0, 0, 0]);
    }
    code.extend_from_slice(&[0x9d, 0x58]); // popfq; pop rax
    let moved = code.len();
    code.extend_from_slice(next);
    code.push(0xc3);
    // The part that holds two more words on the stack, as a function of its
    // own: two one-byte pushes. What follows is a leaf (no entry).
    let table = code.len().next_multiple_of(4);
    code.resize(table, 0xcc);
    let unwind = table + size_of::<IMAGE_RUNTIME_FUNCTION_ENTRY>();
    // The table's addresses are relative to the pool's slot.
    code.extend_from_slice(&0u32.to_le_bytes());
    code.extend_from_slice(&(moved as u32).to_le_bytes());
    code.extend_from_slice(&(unwind as u32).to_le_bytes());
    // UNWIND_INFO: version 1, a 2-byte prologue, 2 codes, no frame register;
    // UWOP_ALLOC_SMALL of 8 bytes after offsets 2 and 1.
    code.extend_from_slice(&[0x01, 0x02, 0x02, 0x00, 0x02, 0x02, 0x01, 0x02]);
    debug_assert!(code.len() <= SLOT);
    // SAFETY: the caller's slot; the table inside it stays for good.
    unsafe {
        std::ptr::copy_nonoverlapping(code.as_ptr(), slot as *mut u8, code.len());
        FlushInstructionCache(GetCurrentProcess(), slot as *const _, code.len());
        RtlAddFunctionTable((slot + table) as *const _, 1, slot as u64);
    }
    slot + moved
}

/// Notes the site at `at` (a `clzero` that just raised an exception; `code`
/// is what follows from it on, within its page) and makes its helper if it
/// can have one.
fn learn(at: usize, code: &[u8]) -> Option<&'static Site> {
    let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(site) = find(at) {
        return Some(site);
    }
    let n = COUNT.load(Ordering::Acquire);
    let site = TABLE.get(n)?;
    let next = code.get(3..).and_then(|rest| Some(&rest[..movable(rest)?]));
    let helper = next.filter(|_| on()).and_then(|next| {
        let len = 3 + next.len();
        let (base, around) = surroundings(at)?;
        if len < 5 || len > 8 || code.len() < 8 || branched_into(around, base, at, at + len) {
            return None;
        }
        let slot = room(&mut pools, at)?;
        // SAFETY: a fresh slot of a pool of ours.
        let moved = unsafe { write_helper(slot, next) };
        Some((slot, moved, at + len))
    });
    if let Some((slot, moved, after)) = helper {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&code[..8]);
        site.helper.store(slot, Ordering::Relaxed);
        site.moved.store(moved, Ordering::Relaxed);
        site.after.store(after, Ordering::Relaxed);
        site.bytes.store(u64::from_le_bytes(bytes), Ordering::Relaxed);
        // SAFETY: a plain query.
        site.seen.store(unsafe { GetTickCount64() }, Ordering::Relaxed);
    }
    site.at.store(at, Ordering::Release);
    COUNT.store(n + 1, Ordering::Release);
    Some(site)
}

/// Writes the call over a site known for long enough, once.
fn write_call(site: &Site) {
    // SAFETY: a plain query.
    let now = unsafe { GetTickCount64() };
    if site.written.load(Ordering::Relaxed)
        || now.saturating_sub(site.seen.load(Ordering::Relaxed)) < GRACE_MS
    {
        return;
    }
    let _pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
    if site.written.swap(true, Ordering::Relaxed) {
        return;
    }
    let at = site.at.load(Ordering::Relaxed);
    let len = site.after.load(Ordering::Relaxed) - at;
    let target = at as *mut u64;
    // The site's eight bytes as one value: the call, the padding, and
    // whatever follows the replaced instructions as it is.
    let old = site.bytes.load(Ordering::Relaxed).to_le_bytes();
    let mut new = old;
    let distance = site.helper.load(Ordering::Relaxed) as i64 - (at as i64 + 5);
    let Ok(distance) = i32::try_from(distance) else {
        return;
    };
    new[0] = 0xe8;
    new[1..5].copy_from_slice(&distance.to_le_bytes());
    new[5..len].fill(0x90);
    let mut protection = 0;
    // SAFETY: eight bytes of code that still hold what the helper was made
    // for (checked under the lock); the page's protection is put back and
    // the processor told its code changed.
    unsafe {
        if VirtualProtect(target.cast(), 8, PAGE_EXECUTE_READWRITE, &mut protection) == 0 {
            return;
        }
        if target.read_unaligned().to_le_bytes() == old {
            target.write_unaligned(u64::from_le_bytes(new));
            WRITTEN.fetch_add(1, Ordering::Relaxed);
        }
        VirtualProtect(target.cast(), 8, protection, &mut protection);
        FlushInstructionCache(GetCurrentProcess(), target.cast(), 8);
    }
}

/// Answers an illegal-instruction exception at a `clzero` or at a site
/// that was one: zeroes the line at `rax` and sends the thread on. `code`
/// is the code from the faulting address on, within its page. False if the
/// exception is neither.
///
/// # Safety
/// `context` is the faulting thread's, from the exception being
/// dispatched, and `code` what its instruction pointer points at.
pub(crate) unsafe fn clzero(context: &mut CONTEXT, code: &[u8]) -> bool {
    let at = context.Rip as usize;
    let is_clzero = code.get(..3) == Some(&[0x0f, 0x01, 0xfc]);
    let site = match find(at) {
        Some(site) => Some(site),
        None if is_clzero => learn(at, code),
        None => return false,
    };
    let moved = site.map_or(0, |s| s.moved.load(Ordering::Relaxed));
    if moved == 0 && !is_clzero {
        return false;
    }
    // SAFETY: what the instruction does: the title's own line.
    unsafe { ((context.Rax & !63) as *mut u8).write_bytes(0, 64) };
    let Some(site) = site.filter(|_| moved != 0) else {
        context.Rip += 3;
        return true;
    };
    write_call(site);
    // As the call does: the return address pushed, on to the moved
    // instruction (the helper's first part is what the handler just did).
    context.Rsp -= 8;
    // SAFETY: the thread's stack below its pointer, as a call would write it.
    unsafe { (context.Rsp as *mut u64).write(site.after.load(Ordering::Relaxed) as u64) };
    context.Rip = moved as u64;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movable_instructions() {
        // The title's: add rax, 0x40; cmp r10, r11; cmp rdx, r8.
        assert_eq!(movable(&[0x48, 0x83, 0xc0, 0x40, 0x4c]), Some(4));
        assert_eq!(movable(&[0x4d, 0x39, 0xda, 0x72]), Some(3));
        assert_eq!(movable(&[0x4c, 0x39, 0xc2]), Some(3));
        // add eax, 0x12345678; mov ecx, edx.
        assert_eq!(movable(&[0x81, 0xc0, 0x78, 0x56, 0x34, 0x12]), Some(6));
        assert_eq!(movable(&[0x89, 0xd1]), Some(2));
        // The stack pointer in either place; memory operands; a jump; cut off.
        assert_eq!(movable(&[0x48, 0x83, 0xc4, 0x40]), None); // add rsp, 0x40
        assert_eq!(movable(&[0x48, 0x89, 0xe0]), None); // mov rax, rsp
        assert_eq!(movable(&[0x48, 0x39, 0xc4]), None); // cmp rsp, rax
        assert_eq!(movable(&[0x48, 0x89, 0x08]), None); // mov [rax], rcx
        assert_eq!(movable(&[0x48, 0x8b, 0x05, 0, 0, 0, 0]), None); // mov rax, [rip]
        assert_eq!(movable(&[0x72, 0xf4]), None);
        assert_eq!(movable(&[0x48, 0x83, 0xc0]), None);
        assert_eq!(movable(&[]), None);
        // r12 is not the stack pointer: add r12, 0x40.
        assert_eq!(movable(&[0x49, 0x83, 0xc4, 0x40]), Some(4));
    }

    #[test]
    fn branches_into_a_site_are_seen() {
        // clzero; add rax, 0x40; cmp rax, r10; jb -12 (to the clzero).
        let lp = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x4c, 0x39, 0xd0, 0x72, 0xf4];
        assert!(!branched_into(&lp, 0x1000, 0x1000, 0x1007));
        // The same loop jumping back to the add.
        let mut inner = lp;
        inner[11] = 0xf7;
        assert!(branched_into(&inner, 0x1000, 0x1000, 0x1007));
        // A near jump and a call from further away, to the add and past it.
        let mut far = vec![0x90; 0x40];
        far.extend_from_slice(&lp);
        far.extend_from_slice(&[0xe9, 0, 0, 0, 0, 0x0f, 0x84, 0, 0, 0, 0]);
        let site = 0x2040;
        let jmp = far.len() - 11;
        let to = |from: usize, skip: usize, target: usize| {
            ((target as i64 - (0x2000 + from + skip) as i64) as i32).to_le_bytes()
        };
        far[jmp + 1..jmp + 5].copy_from_slice(&to(jmp, 5, site + 3));
        far[jmp + 7..jmp + 11].copy_from_slice(&to(jmp + 5, 6, site + 7));
        assert!(branched_into(&far, 0x2000, site, site + 7));
        far[jmp + 1..jmp + 5].copy_from_slice(&to(jmp, 5, site));
        assert!(!branched_into(&far, 0x2000, site, site + 7));
        far[jmp + 7..jmp + 11].copy_from_slice(&to(jmp + 5, 6, site + 5));
        assert!(branched_into(&far, 0x2000, site, site + 7));
    }

    /// Runs `code` (a function of the Windows convention taking a start and
    /// an end address) on memory of ours.
    fn run(code: &[u8], calls: usize, lines: usize) -> (*const u8, Vec<u8>) {
        #[repr(align(64))]
        struct Lines([u8; 64 * 64]);
        // SAFETY: fresh executable memory holding `code`.
        let f = unsafe {
            let at = VirtualAlloc(
                std::ptr::null(),
                4096,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            assert!(!at.is_null());
            std::ptr::copy_nonoverlapping(code.as_ptr(), at, code.len());
            at
        };
        let mut memory = Box::new(Lines([0xa5; 64 * 64]));
        // SAFETY: the code clears whole lines of `memory` between the two
        // addresses it is given.
        let clear: extern "system" fn(*mut u8, *mut u8) -> u64 = unsafe { std::mem::transmute(f) };
        let mut last = 0;
        for _ in 0..calls {
            memory.0.fill(0xa5);
            let start = memory.0.as_mut_ptr();
            // SAFETY: inside the lines.
            last = clear(unsafe { start.add(64) }, unsafe { start.add(64 + 64 * lines) });
            assert_eq!(last as usize, start as usize + 64 + 64 * lines);
        }
        let _ = last;
        (f.cast_const(), memory.0.to_vec())
    }

    fn cleared(memory: &[u8], lines: usize) {
        assert!(memory[..64].iter().all(|&b| b == 0xa5));
        assert!(memory[64..64 + 64 * lines].iter().all(|&b| b == 0), "lines not zeroed");
        assert!(memory[64 + 64 * lines..].iter().all(|&b| b == 0xa5), "zeroed too far");
    }

    #[test]
    fn a_loop_of_the_titles_shape_is_rewritten_and_still_clears() {
        crate::crash::report_crashes();
        // mov rax, rcx; top: clzero; add rax, 0x40; cmp rax, rdx; jb top; ret
        let code = [
            0x48, 0x89, 0xc8, 0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x72,
            0xf4, 0xc3,
        ];
        let before = crate::crash::TRAPS[0].load(Ordering::Relaxed);
        let (f, memory) = run(&code, 1, 40);
        cleared(&memory, 40);
        let native = crate::crash::TRAPS[0].load(Ordering::Relaxed) == before;
        if native {
            // A CPU with the instruction: nothing to rewrite.
            return;
        }
        // Past the grace period the next exception writes the call; from
        // then on the loop raises none.
        std::thread::sleep(std::time::Duration::from_millis(2 * GRACE_MS));
        let (_, memory) = {
            // SAFETY: the function `run` made.
            let clear: extern "system" fn(*mut u8, *mut u8) -> u64 =
                unsafe { std::mem::transmute(f) };
            #[repr(align(64))]
            struct Lines([u8; 64 * 64]);
            let mut m = Box::new(Lines([0xa5; 64 * 64]));
            for round in 0..3 {
                m.0.fill(0xa5);
                let start = m.0.as_mut_ptr();
                let traps = crate::crash::TRAPS[0].load(Ordering::Relaxed);
                // SAFETY: inside the lines.
                let end = clear(unsafe { start.add(64) }, unsafe { start.add(64 + 64 * 40) });
                assert_eq!(end as usize, start as usize + 64 + 64 * 40);
                let raised = crate::crash::TRAPS[0].load(Ordering::Relaxed) - traps;
                // The first round raises the one exception that writes the call.
                assert!(raised <= u64::from(round == 0), "round {round}: {raised} exceptions");
            }
            (f, m.0.to_vec())
        };
        cleared(&memory, 40);
        // SAFETY: the function's code.
        let site = unsafe { std::slice::from_raw_parts(f.add(3), 7) };
        assert_eq!(site[0], 0xe8, "the site is a call now");
        assert_eq!(site[5..], [0x90, 0x90]);
    }

    #[test]
    fn a_site_a_branch_lands_inside_stays_as_it_is() {
        crate::crash::report_crashes();
        // mov rax, rcx; jmp top; (padding) top: clzero; mid: add rax, 0x40;
        // cmp rax, rdx; jb top; ret; and a jump to mid nothing takes.
        let code = [
            0x48, 0x89, 0xc8, 0xeb, 0x02, 0x90, 0x90, // 0..7
            0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x72, 0xf4, 0xc3, // 7..20
            0xeb, 0xf4, // 20: jmp to 10 (the add)
        ];
        let before = crate::crash::TRAPS[0].load(Ordering::Relaxed);
        let (f, memory) = run(&code, 1, 8);
        cleared(&memory, 8);
        if crate::crash::TRAPS[0].load(Ordering::Relaxed) == before {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(2 * GRACE_MS));
        // SAFETY: the function `run` made.
        let clear: extern "system" fn(*mut u8, *mut u8) -> u64 = unsafe { std::mem::transmute(f) };
        let mut m = vec![0xa5u8; 64 * 16 + 64];
        let start = m.as_mut_ptr().wrapping_add(m.as_ptr().align_offset(64));
        let traps = crate::crash::TRAPS[0].load(Ordering::Relaxed);
        // SAFETY: eight whole lines inside `m`.
        clear(start, unsafe { start.add(64 * 8) });
        assert_eq!(crate::crash::TRAPS[0].load(Ordering::Relaxed) - traps, 8);
        // SAFETY: the function's code.
        assert_eq!(unsafe { std::slice::from_raw_parts(f.add(7), 3) }, [0x0f, 0x01, 0xfc]);
    }

    #[test]
    fn threads_racing_on_a_site_all_clear_their_lines() {
        crate::crash::report_crashes();
        // mov rax, rcx; top: mov r10, rax; add r10, 0x40; clzero; cmp r10, rdx;
        // mov rax, r10; jb top; ret  (the compare form: clzero, then cmp).
        let code = [
            0x48, 0x89, 0xc8, // mov rax, rcx
            0x49, 0x89, 0xc2, // top: mov r10, rax
            0x49, 0x83, 0xc2, 0x40, // add r10, 0x40
            0x0f, 0x01, 0xfc, // clzero
            0x49, 0x39, 0xd2, // cmp r10, rdx
            0x4c, 0x89, 0xd0, // mov rax, r10
            0x72, 0xee, // jb top
            0xc3,
        ];
        // SAFETY: fresh executable memory holding `code`.
        let f = unsafe {
            let at = VirtualAlloc(
                std::ptr::null(),
                4096,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_EXECUTE_READWRITE,
            ) as *mut u8;
            assert!(!at.is_null());
            std::ptr::copy_nonoverlapping(code.as_ptr(), at, code.len());
            at as usize
        };
        let threads: Vec<_> = (0..6)
            .map(|_| {
                std::thread::spawn(move || {
                    // SAFETY: the code above.
                    let clear: extern "system" fn(*mut u8, *mut u8) -> u64 =
                        unsafe { std::mem::transmute(f) };
                    let mut m = vec![0xa5u8; 64 * 34];
                    let start = m.as_mut_ptr().wrapping_add(m.as_ptr().align_offset(64));
                    let started = std::time::Instant::now();
                    let mut rounds = 0;
                    while started.elapsed() < std::time::Duration::from_millis(4 * GRACE_MS) {
                        // SAFETY: 32 whole lines inside `m`.
                        let lines = unsafe { std::slice::from_raw_parts_mut(start, 64 * 32) };
                        lines.fill(0xa5);
                        let end = clear(start, unsafe { start.add(64 * 32) });
                        assert_eq!(end as usize, start as usize + 64 * 32);
                        assert!(lines.iter().all(|&b| b == 0), "round {rounds}");
                        rounds += 1;
                    }
                    rounds
                })
            })
            .collect();
        for t in threads {
            assert!(t.join().unwrap() > 0);
        }
    }
}
