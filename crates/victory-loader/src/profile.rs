//! Where the time of a stretch of the run goes, thread by thread.
//! `VICTORY_PROFILE=<file>[,<seconds>[,<interval ms>]]`: whenever `<file>`
//! appears, it is removed and every thread is sampled for `<seconds>`
//! (default 20) every `<interval>` (default 40 ms): its unwound stack, as
//! wall-clock samples, so a waiting thread shows what it waits in. The
//! result goes to `<file>.<n>.txt`: per thread its CPU time over the
//! stretch and its stacks as `count frame;frame;...` lines (outermost
//! first, `module+rva`), then the exceptions the stretch raised by kind
//! and site. Symbols are put to the addresses afterwards, outside the run.
//!
//! Every five seconds it also logs the exceptions raised since the last
//! line, and at its start what one emulated instruction costs here.

use std::collections::HashMap;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Duration, Instant};

use windows_sys::Wdk::System::Threading::{NtQueryInformationThread, ThreadBasicInformation};
use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Diagnostics::Debug::{CONTEXT_FULL_AMD64, GetThreadContext};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThreadId, GetThreadTimes, OpenThread, ResumeThread,
    SuspendThread, THREAD_GET_CONTEXT, THREAD_QUERY_INFORMATION, THREAD_SUSPEND_RESUME,
};

use crate::crash::{MAX_FRAMES, TRAP_NAMES, TRAP_SITES, TRAPS, locate, unwind_into};
use crate::stacks::{Aligned, BasicInfo, name_of, start_of};

/// Reads `VICTORY_PROFILE` and, if set, starts the thread that waits for
/// the file and samples.
pub fn profile_on_request() {
    let Ok(spec) = std::env::var("VICTORY_PROFILE") else {
        return;
    };
    let mut parts = spec.split(',');
    let Some(file) = parts.next().map(std::path::PathBuf::from) else {
        return;
    };
    let seconds: f64 = parts.next().and_then(|t| t.parse().ok()).unwrap_or(20.0);
    let interval: u64 = parts.next().and_then(|t| t.parse().ok()).unwrap_or(40);
    let _ = std::thread::Builder::new()
        .name("victory-profile".into())
        .spawn(move || {
            trap_cost();
            let mut last = trap_counts();
            let mut said = Instant::now();
            let mut n = 0;
            loop {
                std::thread::sleep(Duration::from_millis(250));
                if said.elapsed() >= Duration::from_secs(5) {
                    let now = trap_counts();
                    let secs = said.elapsed().as_secs_f64();
                    let line: Vec<String> = (0..now.len())
                        .map(|i| format!("{} {:.0}", TRAP_NAMES[i], (now[i] - last[i]) as f64 / secs))
                        .collect();
                    eprintln!("VIctory: exceptions per second: {}", line.join(", "));
                    (last, said) = (now, Instant::now());
                }
                if file.exists() {
                    let _ = std::fs::remove_file(&file);
                    n += 1;
                    let out = format!("{}.{n}.txt", file.display());
                    let text = sample(Duration::from_secs_f64(seconds), Duration::from_millis(interval));
                    if let Err(e) = std::fs::write(&out, text) {
                        eprintln!("VIctory: profile not written to {out}: {e}");
                    } else {
                        eprintln!("VIctory: profile written to {out}");
                    }
                }
            }
        });
}

fn trap_counts() -> [u64; 5] {
    [0, 1, 2, 3, 4].map(|i| TRAPS[i].load(Relaxed))
}

/// Times `clzero` on a line of our own: on a CPU (or under a translator)
/// without it, each is one exception round trip through our handler.
fn trap_cost() {
    #[repr(align(64))]
    struct Line([u8; 64]);
    let mut line = Line([1; 64]);
    const N: u32 = 200;
    let before = TRAPS[0].load(Relaxed);
    let start = Instant::now();
    for _ in 0..N {
        // SAFETY: zeroes the 64-byte line at rax, which is ours; the
        // handler emulates it where the CPU has no such instruction.
        unsafe {
            std::arch::asm!(".byte 0x0f, 0x01, 0xfc", in("rax") line.0.as_mut_ptr(), options(nostack));
        }
    }
    let each = start.elapsed().as_secs_f64() * 1e6 / f64::from(N);
    let trapped = TRAPS[0].load(Relaxed) - before;
    TRAPS[0].fetch_sub(trapped, Relaxed);
    eprintln!(
        "VIctory: clzero takes {each:.2} us here ({trapped} of {N} raised an exception we emulated); zeroed {}",
        line.0.iter().all(|&b| b == 0)
    );
}

struct Sampled {
    id: u32,
    handle: HANDLE,
    low: usize,
    high: usize,
    cpu: u64,
    stacks: HashMap<Vec<u64>, u32>,
    samples: u32,
}

fn cpu_of(thread: HANDLE) -> u64 {
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut a, mut b, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: four out structures of the system's layout.
    if unsafe { GetThreadTimes(thread, &mut a, &mut b, &mut kernel, &mut user) } == 0 {
        return 0;
    }
    let ticks = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
    ticks(kernel) + ticks(user)
}

fn threads() -> Vec<Sampled> {
    // SAFETY: plain queries and a snapshot of this process's threads.
    let (me, process, snapshot) = unsafe {
        (
            GetCurrentThreadId(),
            GetCurrentProcessId(),
            CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0),
        )
    };
    let mut out = Vec::new();
    if snapshot == INVALID_HANDLE_VALUE {
        return out;
    }
    // SAFETY: zeroed is a valid THREADENTRY32; its size is set before use.
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    // SAFETY: `snapshot` is open; `entry` is sized.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == process && entry.th32ThreadID != me {
            let id = entry.th32ThreadID;
            // SAFETY: opens a thread of our own process.
            let handle = unsafe {
                OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, 0, id)
            };
            let mut info = BasicInfo {
                exit_status: 0,
                teb: std::ptr::null(),
                client_id: [0; 2],
                affinity: 0,
                priority: 0,
                base_priority: 0,
            };
            // SAFETY: writes `info`, whose layout and size are the system's.
            let status = unsafe {
                NtQueryInformationThread(
                    handle,
                    ThreadBasicInformation,
                    (&raw mut info).cast(),
                    size_of::<BasicInfo>() as u32,
                    std::ptr::null_mut(),
                )
            };
            if !handle.is_null() && status == 0 && !info.teb.is_null() {
                // SAFETY: the TEB of a live thread: stack base at +8, limit at +0x10.
                let (high, low) = unsafe { (info.teb.add(1).read(), info.teb.add(2).read()) };
                out.push(Sampled {
                    id,
                    handle,
                    low,
                    high,
                    cpu: cpu_of(handle),
                    stacks: HashMap::new(),
                    samples: 0,
                });
            } else if !handle.is_null() {
                // SAFETY: we opened it.
                unsafe { CloseHandle(handle) };
            }
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: we opened it.
    unsafe { CloseHandle(snapshot) };
    out
}

/// Samples every thread alive now for `length`, one round per `interval`.
fn sample(length: Duration, interval: Duration) -> String {
    use std::fmt::Write;
    let mut threads = threads();
    let traps = trap_counts();
    let sites: Vec<(u64, u64)> =
        TRAP_SITES.iter().map(|s| (s[0].load(Relaxed), s[1].load(Relaxed))).collect();
    let start = Instant::now();
    let mut frames: Vec<u64> = Vec::with_capacity(MAX_FRAMES);
    // SAFETY: zeroed is a valid CONTEXT.
    let mut context = Aligned(unsafe { std::mem::zeroed() });
    let (mut rounds, mut busy) = (0u32, Duration::ZERO);
    while start.elapsed() < length {
        let round = Instant::now();
        for t in &mut threads {
            frames.clear();
            context.0.ContextFlags = CONTEXT_FULL_AMD64;
            // SAFETY: suspends, reads and resumes a thread that is not this
            // one; nothing in between allocates (`frames` has its capacity).
            unsafe {
                if SuspendThread(t.handle) == u32::MAX {
                    continue;
                }
                if GetThreadContext(t.handle, &mut context.0) != 0 {
                    unwind_into(context.0, t.low, t.high, &mut frames);
                }
                ResumeThread(t.handle);
            }
            if !frames.is_empty() {
                t.samples += 1;
                *t.stacks.entry(frames.clone()).or_insert(0) += 1;
            }
        }
        rounds += 1;
        busy += round.elapsed();
        std::thread::sleep(interval.saturating_sub(round.elapsed()));
    }
    let wall = start.elapsed().as_secs_f64();
    let mut text = String::new();
    let _ = writeln!(
        text,
        "profile: {wall:.2} s, {rounds} rounds, {} threads, {:.1} ms a round sampling",
        threads.len(),
        busy.as_secs_f64() * 1e3 / f64::from(rounds.max(1))
    );
    let now = trap_counts();
    for i in 0..now.len() {
        let _ = writeln!(text, "exceptions {}: {}", TRAP_NAMES[i], now[i] - traps[i]);
    }
    for (i, slot) in TRAP_SITES.iter().enumerate() {
        let (at, n) = (slot[0].load(Relaxed), slot[1].load(Relaxed));
        let before = if sites[i].0 == at { sites[i].1 } else { 0 };
        if at != 0 && n > before {
            let _ = writeln!(text, "exception site {}: {}", locate(at as usize), n - before);
        }
    }
    let mut names: HashMap<u64, String> = HashMap::new();
    threads.sort_by_key(|t| std::cmp::Reverse(cpu_of(t.handle).saturating_sub(t.cpu)));
    for t in &threads {
        let cpu = cpu_of(t.handle).saturating_sub(t.cpu) as f64 / 1e4;
        let _ = writeln!(
            text,
            "thread {} \"{}\" start {} cpu {cpu:.0} ms samples {}",
            t.id,
            name_of(t.handle),
            locate(start_of(t.handle)),
            t.samples
        );
        let mut stacks: Vec<_> = t.stacks.iter().collect();
        stacks.sort_by_key(|&(_, n)| std::cmp::Reverse(*n));
        for (stack, n) in stacks {
            let line: Vec<String> = stack
                .iter()
                .rev()
                .map(|&pc| names.entry(pc).or_insert_with(|| locate(pc as usize)).clone())
                .collect();
            let _ = writeln!(text, "  {n} {}", line.join(";"));
        }
        // SAFETY: we opened it.
        unsafe { CloseHandle(t.handle) };
    }
    text
}
