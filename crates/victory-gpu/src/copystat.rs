//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only).

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Of texture refreshes, microseconds asking whether the memory is
    /// committed, hashing it and detiling it.
    TexQueryUs,
    TexHashUs,
    TexDetileUs,
    /// After a submission: bytes of what its shaders may have written
    /// copied back into title memory, the microseconds that took, and the
    /// microseconds of the rest (the title's bottom-of-pipe writes, the
    /// host's messages); the bytes its lists declared as written (the same
    /// ranges many times).
    BackBytes,
    BackUs,
    AfterUs,
    BackDeclared,
    /// Of the copies: microseconds in copies into title memory (their
    /// own, without the refreshes they make run first), in refreshes, and
    /// after a read met an unmapped page (finding the mapped parts and
    /// reading them); queries for the state of memory made there.
    TitleCopyUs,
    FlushUs,
    FallbackUs,
    FallbackQueries,
    /// With `VICTORY_COPY_DIFF=1`: bytes of refreshes compared with what
    /// the host copy held, and the 4 KiB pages and 64 KiB blocks of them
    /// that differed (what write tracking would have had to report, and
    /// each such page one write fault of the title's).
    DiffCompared,
    DiffPages,
    DiffBlocks,
    /// Of the time on the GPU: microseconds in the host's call that takes
    /// the lists, and waiting for them after it; submissions whose
    /// shaders may write host copies (their copy-back cannot be put off).
    HostCallUs,
    HostWaitUs,
    WritingSubmissions,
    /// Of copy-back: microseconds comparing the copy with its bytes before
    /// the submission, reading title memory where they differ, merging,
    /// and writing title memory; the 64 KiB pieces read and written.
    BackCompareUs,
    BackReadUs,
    BackMergeUs,
    BackWriteUs,
    BackReads,
    BackWrites,
}
const COUNTERS: usize = 45;

/// `VICTORY_COPY_DIFF=1`: refreshes of host copies compare before they
/// copy (twice the memory traffic), to count what changed.
pub(crate) fn diff() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_COPY_DIFF").as_deref() == Ok("1"))
}
static COUNT: [AtomicU64; COUNTERS] = [const { AtomicU64::new(0) }; COUNTERS];

pub(crate) fn add(c: C, n: u64) {
    COUNT[c as usize].fetch_add(n, Relaxed);
}

/// Entry points whose wall time is summed (title threads call them, several
/// at once, so the sums can exceed the frame): running submissions,
/// presenting, making host pipelines, bringing a host list up to date for
/// a draw or dispatch (that includes making its pipeline).
#[derive(Clone, Copy)]
pub(crate) enum T {
    Execute,
    Present,
    Pipeline,
    Prepare,
}
const TIMES: usize = 4;
/// Per entry point: microseconds, calls, longest call in microseconds.
static TIME: [[AtomicU64; 3]; TIMES] = [const { [const { AtomicU64::new(0) }; 3] }; TIMES];

/// Times an entry point from now until it is dropped.
pub(crate) struct Span(T, std::time::Instant);

pub(crate) fn span(t: T) -> Span {
    Span(t, std::time::Instant::now())
}

impl Drop for Span {
    fn drop(&mut self) {
        let us = self.1.elapsed().as_micros() as u64;
        let t = &TIME[self.0 as usize];
        t[0].fetch_add(us, Relaxed);
        t[1].fetch_add(1, Relaxed);
        t[2].fetch_max(us, Relaxed);
    }
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame over `frames` frames, and the ranges that
/// took most bytes.
pub(crate) fn log(frames: f64) {
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = COUNT
        .iter()
        .map(|a| a.swap(0, Relaxed) as f64 / frames)
        .collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
    );
    tracing::info!(
        "copies per frame, where the time goes: {:.1} ms refreshing host copies, {:.1} ms copying \
         into title memory, {:.1} ms (of both) after reads that met unmapped pages, in {:.0} \
         queries for the state of memory",
        ms(c[C::FlushUs as usize]),
        ms(c[C::TitleCopyUs as usize]),
        ms(c[C::FallbackUs as usize]),
        c[C::FallbackQueries as usize],
    );
    tracing::info!(
        "host submissions per frame: {:.1} ms handing the lists over, {:.1} ms waiting for them; \
         {:.1} of them with ranges shaders may write",
        ms(c[C::HostCallUs as usize]),
        ms(c[C::HostWaitUs as usize]),
        c[C::WritingSubmissions as usize],
    );
    if diff() {
        tracing::info!(
            "copies per frame, what changed: of {:.1} MB compared, {:.0} pages of 4 KiB ({:.1} MB) \
             in {:.0} blocks of 64 KiB ({:.1} MB) differed from the host copy",
            mb(c[C::DiffCompared as usize]),
            c[C::DiffPages as usize],
            c[C::DiffPages as usize] * 4096.0 / 1e6,
            c[C::DiffBlocks as usize],
            c[C::DiffBlocks as usize] * 65536.0 / 1e6,
        );
    }
    tracing::info!(
        "texture refreshes per frame: {:.2} ms asking for committed memory, {:.2} ms hashing, {:.2} ms detiling",
        ms(c[C::TexQueryUs as usize]),
        ms(c[C::TexHashUs as usize]),
        ms(c[C::TexDetileUs as usize]),
    );
    let t: Vec<[f64; 3]> = TIME
        .iter()
        .map(|t| [0, 1, 2].map(|i| t[i].swap(0, Relaxed) as f64))
        .collect();
    let per = |k: T| {
        let [us, calls, longest] = t[k as usize];
        format!("{:.1} ms in {:.0} calls (longest {:.0} ms)", us / 1e3 / frames, calls / frames, longest / 1e3)
    };
    tracing::info!(
        "time per frame in our entry points: submissions {}; present {}; host pipelines {}; \
         draw and dispatch preparation {}",
        per(T::Execute),
        per(T::Present),
        per(T::Pipeline),
        per(T::Prepare),
    );
    tracing::info!(
        "after submissions, per frame: {:.0} ranges declared as written by shaders, gone through \
         once per host copy: {:.1} ms comparing, {:.1} ms reading title memory ({:.0} pieces), \
         {:.1} ms merging, {:.1} ms writing title memory ({:.0} pieces)",
        c[C::BackDeclared as usize],
        ms(c[C::BackCompareUs as usize]),
        ms(c[C::BackReadUs as usize]),
        c[C::BackReads as usize],
        ms(c[C::BackMergeUs as usize]),
        ms(c[C::BackWriteUs as usize]),
        c[C::BackWrites as usize],
    );
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
        let kind = info.Type;
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}
