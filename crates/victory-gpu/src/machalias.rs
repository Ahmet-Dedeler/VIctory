//! Title memory as the card's own memory on macOS (`VICTORY_ALIAS=1`, Wine on Darwin only).
//!
//! D3DMetal does not implement `OpenExistingHeapFromAddress[1]` / `OpenExistingHeapFromFileMapping`
//! (E_NOTIMPL / E_FAIL, measured with GPTK 4.0b3), so [`crate::inplace`]'s heaps over title memory
//! cannot be made there. The other way round works: a buffer D3DMetal makes itself in the PC's
//! memory ([`crate::host::Host::shared_buffer`], custom heap, write-back pages, pool L0) and keeps
//! mapped is the very memory the GPU reads (Apple silicon has one memory). Pointing the title's
//! addresses at that buffer's pages with `mach_vm_remap(copy = FALSE, VM_FLAGS_FIXED |
//! VM_FLAGS_OVERWRITE)` makes title memory and the buffer one and the same: what the title writes
//! the GPU reads, and what shaders write the title reads, with no copy, refresh or copy-back.
//!
//! Measured in a probe under CrossOver 26.3 / GPTK 4.0b3 / macOS 27.3 (`~/Code/gta-push/uma-probe`),
//! 64 MiB and 4 GiB: coherent both ways (copies and a compute shader through a root UAV), 8 of 8
//! rounds; Wine's `VirtualQuery` and `VirtualProtect` (with a write fault into a vectored handler)
//! keep working on the alias; no sync at submit (a tiny submit costs the same after every page was
//! written); CPU speed through the alias equals plain memory.
//!
//! When: at [`title_memory_aliasable`], called by the runtime right after the title's
//! `XMemVirtualAlloc` committed fresh memory and before the title has its address, so nothing can
//! write it between our copy and the remap (the pages are zero; nothing is copied). A title buffer
//! or window inside such memory is then placed on the alias by [`crate::inplace::place_for`].
//!
//! The Mach calls are raw traps from this PE code: Wine on macOS does not filter `syscall`, and
//! Rosetta 2 translates x86-64 Mach traps (class 1: `rax = 0x0100_0000 | n`). `mach_vm_remap` is a
//! MIG request to the task port; it must go through `mach_msg2_trap` (-47) with
//! `MACH64_SEND_KOBJECT_CALL`: the old `mach_msg_trap` (-31) gets the process killed on macOS 27
//! (EXC_GUARD "mach_msg_trap() called with msgh_id 4813. The trap is not allowed on this platform").
//! Message layouts follow `<mach/mach_vm.h>`'s generated code (`#pragma pack(4)`) and
//! `osfmk/mach/message.h` (`mach_msg2`).
//!
//! shortcut: the title freeing or decommitting part of an aliased allocation lets the whole
//! allocation's alias go (its shadows are let go, the table forgets it); the buffers stay alive
//! while lists hold them. The title's heaps are allocated once and kept, so this is rare.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

/// `VICTORY_ALIAS`: 0 off (the default); 1 alias committed title allocations of at least
/// `VICTORY_ALIAS_MIN_KB` (1024) under Wine on macOS, and place title buffers and windows in them.
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let asked = std::env::var("VICTORY_ALIAS").is_ok_and(|v| v.trim() == "1");
        let on = asked && crate::host::is_darwin();
        if asked && !on {
            tracing::warn!("VICTORY_ALIAS=1 needs Wine on macOS: off");
        }
        on
    })
}

fn env_mb(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(default)
}

/// Smallest allocation aliased, in bytes (`VICTORY_ALIAS_MIN_KB`, 1024).
fn min_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_mb("VICTORY_ALIAS_MIN_KB", 1024) << 10)
}

/// Most title memory aliased at once (`VICTORY_ALIAS_MB`, 24576).
fn most_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_mb("VICTORY_ALIAS_MB", 24576) << 20)
}

/// Largest single buffer (`VICTORY_ALIAS_CHUNK_MB`, 4096); bigger allocations are cut into
/// chunks of this size from their start, and a range across two chunks gets a copy as before.
fn chunk_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| (env_mb("VICTORY_ALIAS_CHUNK_MB", 4096) << 20).max(1 << 20))
}

/// One buffer the title's addresses `[start, start + len)` show.
struct Chunk {
    len: u64,
    buffer: ID3D12Resource,
    /// The buffer's GPU address (of `start`).
    gpu: u64,
}

/// Aliased allocations by start: their length and chunks by start.
struct Table {
    allocations: BTreeMap<u64, (u64, BTreeMap<u64, Chunk>)>,
}

static TABLE: Mutex<Table> = Mutex::new(Table { allocations: BTreeMap::new() });

/// Allocations aliased, bytes aliased, allocations turned down (too small, over the cap, host
/// refused), remap failures, placements handed out, allocations let go.
static STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn table() -> std::sync::MutexGuard<'static, Table> {
    TABLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title's `XMemVirtualAlloc` just committed `[at, at + len)` read-write, fresh (all zero),
/// and has not handed the address out yet: point it at buffers of the card's.
pub fn title_memory_aliasable(at: usize, len: usize) {
    if !enabled() || at == 0 {
        return;
    }
    let (at, len) = (at as u64, len as u64);
    if len < min_bytes() || at % (16 << 10) != 0 || len % (16 << 10) != 0 {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    if STATS[1].load(Ordering::Relaxed) + len > most_bytes() {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    }
    let Some(host) = crate::host::host() else {
        STATS[2].fetch_add(1, Ordering::Relaxed);
        return;
    };
    let mut chunks = BTreeMap::new();
    let mut offset = 0;
    while offset < len {
        let part = (len - offset).min(chunk_bytes());
        let Some((buffer, data)) = host.shared_buffer(part) else {
            STATS[2].fetch_add(1, Ordering::Relaxed);
            return; // the chunks made so far are dropped; the memory stays plain
        };
        // SAFETY: `data` is the mapping of a live buffer of `part` bytes (page aligned: Metal's
        // allocations are), and `[at + offset, + part)` is title memory just committed whose
        // address nobody else has yet; the remap replaces those pages with the buffer's.
        let r = unsafe { remap_shared(at + offset, data as u64, part) };
        if r.msg != 0 || r.kr != 0 || r.address != at + offset {
            if STATS[3].fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::warn!(
                    at = format_args!("{:#x}", at + offset),
                    part,
                    msg = format_args!("{:#x}", r.msg),
                    kr = r.kr,
                    "title memory not aliased onto the card's buffer: copies as before"
                );
            }
            // Parts already remapped stay pointed at their buffers' pages (the pages live as long
            // as either mapping does), so the memory is still correct; it is just not placed.
            return;
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        crate::host::name(&buffer, || format!("title memory {:#x}+{part:#x} aliased", at + offset));
        chunks.insert(at + offset, Chunk { len: part, buffer, gpu });
        offset += part;
    }
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[1].fetch_add(len, Ordering::Relaxed);
    if STATS[0].load(Ordering::Relaxed) <= 32 {
        tracing::info!(at = format_args!("{at:#x}"), mb = len >> 20, chunks = chunks.len(), "title memory aliased onto the card's buffer (no copies for it)");
    }
    table().allocations.insert(at, (len, chunks));
}

/// Buffers of allocations let go while some of their pages may still be mapped at title
/// addresses (a decommit of part of an allocation): kept for good, so those pages never belong
/// to a buffer D3DMetal has freed.
static KEPT: Mutex<Vec<ID3D12Resource>> = Mutex::new(Vec::new());

/// The title frees (`release`: the whole allocation at `at`, whose mappings go with it) or
/// decommits `[at, at + len)`: every aliased allocation it touches is let go.
pub fn title_memory_alias_freed(at: usize, len: usize, release: bool) {
    if !enabled() {
        return;
    }
    let (at, len) = (at as u64, len.max(1) as u64);
    let gone: Vec<(u64, u64, BTreeMap<u64, Chunk>)> = {
        let mut t = table();
        let hit: Vec<u64> = t
            .allocations
            .range(..at + len)
            .filter(|(start, (bytes, _))| **start + *bytes > at)
            .map(|(start, _)| *start)
            .collect();
        hit.into_iter().filter_map(|s| t.allocations.remove(&s).map(|(bytes, chunks)| (s, bytes, chunks))).collect()
    };
    for (start, bytes, chunks) in gone {
        STATS[1].fetch_sub(bytes, Ordering::Relaxed);
        STATS[5].fetch_add(1, Ordering::Relaxed);
        crate::inplace::title_memory_freeing(start as usize, bytes as usize);
        let whole = release && start == at;
        if !whole {
            KEPT.lock().unwrap_or_else(|e| e.into_inner()).extend(chunks.into_values().map(|c| c.buffer));
        }
        tracing::info!(at = format_args!("{start:#x}"), mb = bytes >> 20, whole, "aliased title memory let go");
    }
}

/// The buffer that `[start, start + len)` shows if it lies in one chunk of an aliased allocation:
/// the buffer, the GPU address of `start`, and `start`'s offset in the buffer.
pub(crate) fn placed(start: u64, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
    if !enabled() {
        return None;
    }
    let t = table();
    let (a_start, (a_len, chunks)) = t.allocations.range(..=start).next_back()?;
    if start + len.max(1) > a_start + a_len {
        return None;
    }
    let (c_start, chunk) = chunks.range(..=start).next_back()?;
    if start + len.max(1) > c_start + chunk.len {
        return None;
    }
    STATS[4].fetch_add(1, Ordering::Relaxed);
    let bias = start - c_start;
    Some((chunk.buffer.clone(), chunk.gpu + bias, bias))
}

/// Whether `[start, start + len)` lies in aliased memory.
pub(crate) fn covers(start: u64, len: u64) -> bool {
    enabled() && {
        let t = table();
        t.allocations.range(..=start).next_back().is_some_and(|(s, (l, _))| start + len.max(1) <= s + l)
    }
}

/// The line of what is aliased, for [`crate::inplace`]'s ten-second report.
pub(crate) fn report() -> Option<String> {
    if !enabled() {
        return None;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect();
    Some(format!(
        "title memory aliased onto the card's buffers (VICTORY_ALIAS): {} allocations {} MB; turned down {}; remap failures {}; placements {}; let go {}",
        s[0],
        s[1] >> 20,
        s[2],
        s[3],
        s[4],
        s[5]
    ))
}

// The Mach calls.

use core::arch::asm;

const MACH: u64 = 0x0100_0000;
const TRAP_REPLY_PORT: u64 = 26;
const TRAP_TASK_SELF: u64 = 28;
const TRAP_MSG2: u64 = 47;
/// `mach_msg2` option bits (osfmk/mach/message.h): a kernel-object call must carry this one and
/// be a combined send and receive.
const MACH64_SEND_KOBJECT_CALL: u64 = 0x0000_0002_0000_0000;
const VM_FLAGS_OVERWRITE: i32 = 0x4000;

/// A Mach trap: six arguments in registers, a seventh and eighth read by the kernel from the user
/// stack at `rsp + 8` and `rsp + 16` (x86-64 XNU's munger, which Rosetta follows).
unsafe fn trap(n: u64, a: [u64; 6], a7: u64, a8: u64) -> u64 {
    let ret: u64;
    // SAFETY: the caller picks a trap whose arguments are valid; syscall clobbers rcx and r11.
    unsafe {
        asm!(
            "sub rsp, 32",
            "mov qword ptr [rsp], 0",
            "mov qword ptr [rsp + 8], {a7}",
            "mov qword ptr [rsp + 16], {a8}",
            "syscall",
            "add rsp, 32",
            a7 = in(reg) a7, a8 = in(reg) a8,
            inlateout("rax") MACH | n => ret,
            in("rdi") a[0], in("rsi") a[1], in("rdx") a[2], in("r10") a[3], in("r8") a[4], in("r9") a[5],
            lateout("rcx") _, lateout("r11") _,
        );
    }
    ret
}

/// The task's port and a reply port, made once (a reply port per call would leak names).
fn ports() -> (u32, u32) {
    static PORTS: std::sync::OnceLock<(u32, u32)> = std::sync::OnceLock::new();
    // SAFETY: argument-less traps.
    *PORTS.get_or_init(|| unsafe { (trap(TRAP_TASK_SELF, [0; 6], 0, 0) as u32, trap(TRAP_REPLY_PORT, [0; 6], 0, 0) as u32) })
}

/// One `mach_vm_remap` at a time: they share the reply port.
static REMAP: Mutex<()> = Mutex::new(());

#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct Header {
    bits: u32,
    size: u32,
    remote: u32,
    local: u32,
    voucher: u32,
    id: i32,
}

/// `mach_vm_remap`'s request (msgh_id 4813).
#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct RemapRequest {
    head: Header,
    descriptors: u32,
    // mach_msg_port_descriptor_t: name, pad1, (pad2:16 | disposition:8 | type:8)
    src_task: u32,
    pad1: u32,
    disp_type: u32,
    ndr: [u8; 8],
    target_address: u64,
    size: u64,
    mask: u64,
    flags: i32,
    src_address: u64,
    copy: u32,
    inheritance: u32,
}

#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct RemapReply {
    head: Header,
    ndr: [u8; 8],
    ret: i32,
    target_address: u64,
    cur_protection: i32,
    max_protection: i32,
    trailer: [u8; 128],
}

struct Remapped {
    /// `mach_msg2_trap`'s own result (0: sent and received).
    msg: u32,
    /// The routine's `kern_return_t` (0: KERN_SUCCESS).
    kr: i32,
    address: u64,
}

/// Maps `[src, src + size)` of this task again at `target`, replacing what is there, sharing the
/// pages (`copy = FALSE`).
///
/// # Safety
/// Replaces whatever is mapped at `target`; both ranges page aligned and owned by the caller.
unsafe fn remap_shared(target: u64, src: u64, size: u64) -> Remapped {
    let _one = REMAP.lock().unwrap_or_else(|e| e.into_inner());
    let (task, reply) = ports();
    let req = RemapRequest {
        head: Header {
            // MACH_MSGH_BITS(COPY_SEND = 19, MAKE_SEND_ONCE = 21) | MACH_MSGH_BITS_COMPLEX
            bits: 19 | (21 << 8) | 0x8000_0000,
            size: size_of::<RemapRequest>() as u32,
            remote: task,
            local: reply,
            voucher: 0,
            id: 4813,
        },
        descriptors: 1,
        src_task: task,
        pad1: 0,
        disp_type: 19 << 16, // disposition COPY_SEND, type PORT_DESCRIPTOR (0)
        ndr: [0, 0, 0, 0, 1, 0, 0, 0],
        target_address: target,
        size,
        mask: 0,
        flags: VM_FLAGS_OVERWRITE, // | VM_FLAGS_FIXED (0)
        src_address: src,
        copy: 0,
        inheritance: 1, // VM_INHERIT_COPY, the default
    };
    let mut msg = [0u8; size_of::<RemapReply>()];
    // SAFETY: plain bytes of the right sizes.
    unsafe { std::ptr::copy_nonoverlapping((&raw const req).cast::<u8>(), msg.as_mut_ptr(), size_of::<RemapRequest>()) };
    let pack = |lo: u32, hi: u32| (hi as u64) << 32 | lo as u64;
    // mach_msg2(data, SEND | RCV | KOBJECT_CALL, bits | send_size << 32, remote | local << 32,
    //           voucher | id << 32, descriptors | rcv_name << 32, rcv_size | priority << 32, timeout)
    let args = [
        msg.as_mut_ptr() as u64,
        3 | MACH64_SEND_KOBJECT_CALL,
        pack(req.head.bits, size_of::<RemapRequest>() as u32),
        pack(task, reply),
        pack(0, 4813),
        pack(1, reply),
    ];
    // SAFETY: `msg` holds the request and room for the reply and a trailer.
    let r = unsafe { trap(TRAP_MSG2, args, pack(msg.len() as u32, 0), 0) } as u32;
    // SAFETY: the kernel wrote a reply of at most `msg.len()` bytes (zeros if it did not).
    let rep: RemapReply = unsafe { std::ptr::read_unaligned(msg.as_ptr().cast()) };
    Remapped { msg: r, kr: rep.ret, address: rep.target_address }
}

/// A check of the whole path without the title: `VICTORY_ALIAS=1 victory alias-check` under Wine on
/// macOS (a CrossOver bottle with the D3DMetal backend). Panics with what failed.
pub mod check {
    use super::*;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
    use windows::core::Interface;
    use windows_sys::Win32::System::Memory as mem;

    fn committed(device: &ID3D12Device, kind: D3D12_HEAP_TYPE, size: u64, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: plain creation.
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
        r.unwrap()
    }

    /// Records one copy, runs it on a queue of its own and waits.
    fn copy(device: &ID3D12Device, dst: &ID3D12Resource, dst_off: u64, src: &ID3D12Resource, src_off: u64, n: u64) {
        // SAFETY: objects of our own, one run, waited for.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let alloc: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None).unwrap();
            list.CopyBufferRegion(dst, dst_off, src, src_off, n);
            list.Close().unwrap();
            queue.ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let event = CreateEventW(None, false, false, None).unwrap();
            queue.Signal(&fence, 1).unwrap();
            fence.SetEventOnCompletion(1, event).unwrap();
            assert_eq!(WaitForSingleObject(event, 10_000).0, 0, "GPU copy timed out");
        }
    }

    /// Aliases 64 MiB the way the title's allocation is, places a "title buffer" in it through
    /// `crate::inplace::place_for`, and checks 8 rounds each way that CPU and GPU see one memory.
    pub fn run() {
        assert!(enabled(), "run with VICTORY_ALIAS=1 under Wine on macOS");
        const LEN: usize = 64 << 20;
        const AT: usize = 1 << 20;
        let device = crate::host::host().expect("host device").device.clone();
        // SAFETY: memory of the test's own, LEN bytes; every access lies inside it.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), LEN, mem::MEM_RESERVE | mem::MEM_COMMIT | 0x20_0000, mem::PAGE_READWRITE).cast::<u8>();
            assert!(!base.is_null());
            title_memory_aliasable(base as usize, LEN);
            assert!(covers(base as u64 + AT as u64, 4096), "allocation not aliased: {:?}", report());
            // What a title buffer at base + 1 MiB gets from the in-place path.
            let placed = crate::inplace::place_for(base as u64 + AT as u64, 4096, false).expect("placed");
            assert_eq!(placed.bias, AT as u64);
            assert!(placed.heap.is_none());
            for round in 0..8u8 {
                // CPU writes at the title's address; the GPU reads them through the placed buffer.
                let fill: Vec<u8> = (0..4096).map(|i| (i as u8).wrapping_mul(31) ^ round).collect();
                std::ptr::copy_nonoverlapping(fill.as_ptr(), base.add(AT), 4096);
                let back = committed(&device, D3D12_HEAP_TYPE_READBACK, 4096, D3D12_RESOURCE_STATE_COPY_DEST);
                copy(&device, &back, 0, &placed.buffer, placed.bias, 4096);
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).unwrap();
                assert_eq!(std::slice::from_raw_parts(p.cast::<u8>(), 4096), &fill[..], "GPU did not see the CPU's write, round {round}");
                back.Unmap(0, None);
                // The GPU writes the placed buffer; the CPU reads it at the title's address.
                let new: Vec<u8> = (0..4096).map(|i| (i as u8) ^ round ^ 0xa5).collect();
                let up = committed(&device, D3D12_HEAP_TYPE_UPLOAD, 4096, D3D12_RESOURCE_STATE_GENERIC_READ);
                up.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping(new.as_ptr(), p.cast::<u8>(), 4096);
                up.Unmap(0, None);
                copy(&device, &placed.buffer, placed.bias + 65536, &up, 0, 4096);
                assert_eq!(std::slice::from_raw_parts(base.add(AT + 65536), 4096), &new[..], "CPU did not see the GPU's write, round {round}");
            }
            // The title's VirtualProtect still works on aliased memory.
            let mut old = 0;
            assert_ne!(mem::VirtualProtect(base.add(AT).cast(), 65536, mem::PAGE_READONLY, &mut old), 0);
            assert_ne!(mem::VirtualProtect(base.add(AT).cast(), 65536, mem::PAGE_READWRITE, &mut old), 0);
            title_memory_alias_freed(base as usize, 0, true);
            assert!(!covers(base as u64 + AT as u64, 4096));
            assert_ne!(mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE), 0);
            assert!(device.GetDeviceRemovedReason().is_ok());
            println!("{}", report().unwrap_or_default());
        }
    }
}
