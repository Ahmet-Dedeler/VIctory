//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// The copy's bytes before the submission ran, from an offset on,
    /// covering the ranges its shaders write: what the GPU wrote differs.
    before: Mutex<Before>,
}

#[derive(Default)]
struct Before {
    snapshot: Option<(usize, Vec<u8>)>,
    unchanged: UnchangedPages,
}

/// Pages already compared in full against this submission's snapshot.
#[derive(Default)]
struct UnchangedPages {
    start: usize,
    end: usize,
    pages: Vec<bool>,
}

impl UnchangedPages {
    fn begin(&mut self, start: usize, end: usize) {
        if self.pages.is_empty() {
            self.start = start;
            self.end = end;
            self.pages.resize(end.div_ceil(PAGE) - start / PAGE, false);
        }
    }

    fn differs(&mut self, address: usize, current: &[u8], before: &[u8]) -> bool {
        if self.known(address) {
            return false;
        }
        self.compared(address, current.len(), current != before)
    }

    /// Bytes of `[from, to)` (title addresses) in pages not yet known
    /// unchanged, counted in whole pages.
    fn unknown(&self, from: usize, to: usize) -> usize {
        let first = from / PAGE - self.start / PAGE;
        let last = (to - 1) / PAGE - self.start / PAGE;
        self.pages[first..=last].iter().filter(|&&known| !known).count() * PAGE
    }

    /// Whether the page of `address` was compared in full already and
    /// found as it was.
    fn known(&self, address: usize) -> bool {
        self.pages[address / PAGE - self.start / PAGE]
    }

    /// Notes the comparison of `len` bytes at `address` (inside one page)
    /// with the snapshot, which found them different or not; returns
    /// whether they differ and the page was not known unchanged.
    fn compared(&mut self, address: usize, len: usize, differs: bool) -> bool {
        let page = address / PAGE;
        let unchanged = &mut self.pages[page - self.start / PAGE];
        if *unchanged {
            return false;
        }
        if differs {
            return true;
        }
        // A later range can touch bytes a partial comparison missed. At
        // the edges, only bytes covered by the snapshot can be compared.
        *unchanged = address == (page * PAGE).max(self.start)
            && address + len == ((page + 1) * PAGE).min(self.end);
        false
    }

    fn clear(&mut self) {
        self.pages.clear();
    }
}

/// For [`Shadow::copy_back`]: `host` is the copy's bytes from title address
/// `address` on and `before` the same bytes before the submission. Per
/// 64 KiB piece, in the order copy-back goes through them, whether the two
/// differ there (false for pages already known unchanged, which are not
/// compared again). Compared in parts on the pool.
fn compare_ahead(host: &[u8], address: usize, before: &[u8], unchanged: &UnchangedPages) -> Vec<bool> {
    let mut pieces = Vec::with_capacity(host.len() / PAGE + 2);
    let mut at = 0;
    while at < host.len() {
        let n = (PAGE - (address + at) % PAGE).min(host.len() - at);
        pieces.push((at, n));
        at += n;
    }
    let mut found = vec![false; pieces.len()];
    let per = pieces.len().div_ceil(crate::host::PARTS).max(1);
    let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = pieces
        .chunks(per)
        .zip(found.chunks_mut(per))
        .map(|(pieces, found)| {
            Box::new(move || {
                for (&(at, n), found) in pieces.iter().zip(found) {
                    *found = !unchanged.known(address + at) && host[at..at + n] != before[at..at + n];
                }
            }) as Box<dyn FnOnce() + Send + '_>
        })
        .collect();
    victory_formats::texture::run_parallel(jobs);
    found
}

/// Appends `bytes` to a snapshot that has room for them (see [`pooled`]),
/// big ones in parts on the pool.
fn extend(to: &mut Vec<u8>, bytes: &[u8]) {
    if bytes.len() < crate::host::PARALLEL_FROM || to.capacity() - to.len() < bytes.len() {
        to.extend_from_slice(bytes);
        return;
    }
    let at = to.len();
    let dst = to.as_mut_ptr() as usize + at;
    let part = bytes.len().div_ceil(crate::host::PARTS);
    let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = bytes
        .chunks(part)
        .enumerate()
        .map(|(i, piece)| {
            Box::new(move || {
                // SAFETY: the vector's spare room, each part its own bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(piece.as_ptr(), (dst + i * part) as *mut u8, piece.len());
                }
            }) as Box<dyn FnOnce() + Send + '_>
        })
        .collect();
    victory_formats::texture::run_parallel(jobs);
    // SAFETY: the parts wrote exactly these bytes, within the capacity.
    unsafe { to.set_len(at + bytes.len()) };
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one.
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool (at most 8 kept, the largest).
fn give_back(v: Vec<u8>) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    if pool.len() > 8 {
        pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
        pool.truncate(8);
    }
}

const NOT_WATCHED: u8 = 2;

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start
    }

    /// The copy's bytes at `[address, address + len)` (inside the buffer).
    #[cfg(test)]
    pub(crate) fn host_bytes(&self, address: u64, len: usize) -> Vec<u8> {
        // SAFETY: inside the mapping; no submission runs while tests read.
        unsafe { std::slice::from_raw_parts(self.data.add(self.offset(address) as usize), len) }.to_vec()
    }

    #[cfg(test)]
    pub(crate) fn read_u32(&self, offset: usize) -> u32 {
        assert!(offset + 4 <= self.len as usize);
        // SAFETY: `offset..offset + 4` lies within the mapped host buffer.
        unsafe { self.data.add(offset).cast::<u32>().read_unaligned() }
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            before: Mutex::default(),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, grown to cover every range the
    /// submission declares (the world declares overlapping views of its
    /// 174 MB heap dozens of times), in reused memory (fresh snapshots cost
    /// ~2.3 million page faults per world frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        self.pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((address, address + len));
        let (from, to) = self.inside(address, len);
        // SAFETY: inside the mapping.
        let host =
            |a: usize, b: usize| unsafe { std::slice::from_raw_parts(self.data.add(a), b - a) };
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let before = &mut before.snapshot;
        let (start, end) = match &*before {
            Some((at, bytes)) => ((*at).min(from), (at + bytes.len()).max(to)),
            None => (from, to),
        };
        if let Some((at, bytes)) = &*before
            && (*at, at + bytes.len()) == (start, end)
        {
            return;
        }
        // Parts newly covered from the copy, the rest as it was taken.
        let mut grown = pooled(end - start);
        match before.take() {
            Some((at, bytes)) => {
                extend(&mut grown, host(start, at));
                extend(&mut grown, &bytes);
                extend(&mut grown, host(at + bytes.len(), end));
                give_back(bytes);
            }
            None => extend(&mut grown, host(start, end)),
        }
        *before = Some((start, grown));
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written.
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        let written = self.copy_back_range(address, len);
        self.copied_back(&[(address, address + len)]);
        written
    }

    /// [`Shadow::copy_back`] for all the ranges a submission declared on
    /// this copy (`(address, length)` each, in any order, overlapping as
    /// they like): what they cover is gone through once. The world's
    /// passes declare hundreds of views of one heap a submission, each
    /// from its own address to the heap's end; going through each on its
    /// own compared the pages shaders had written, and read title memory
    /// there, once per view (0.4-4 s of a busy frame on a Mac for 20-40 MB
    /// written).
    pub(crate) fn copy_back_all(&self, declared: &[(usize, usize)]) -> usize {
        let mut ranges: Vec<(usize, usize)> =
            declared.iter().map(|&(address, len)| (address, address + len)).collect();
        ranges.sort_unstable();
        let mut written = 0;
        let mut merged: Option<(usize, usize)> = None;
        for &(from, to) in &ranges {
            match &mut merged {
                Some((_, end)) if from <= *end => *end = (*end).max(to),
                _ => {
                    if let Some((a, e)) = merged.replace((from, to)) {
                        written += self.copy_back_range(a, e - a);
                    }
                }
            }
        }
        if let Some((a, e)) = merged {
            written += self.copy_back_range(a, e - a);
        }
        self.copied_back(&ranges);
        written
    }

    /// The submission is done with the declared ranges `done` (start and
    /// end): they are no longer pending, and the snapshot goes with the
    /// last of them.
    fn copied_back(&self, done: &[(usize, usize)]) {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        match done {
            [one] => pending.retain(|r| r != one),
            _ => {
                let done: std::collections::HashSet<(usize, usize)> = done.iter().copied().collect();
                pending.retain(|r| !done.contains(r));
            }
        }
        if pending.is_empty()
            && let Some((_, bytes)) = before.snapshot.take()
        {
            before.unchanged.clear();
            give_back(bytes);
        }
    }

    /// The comparing and copying of [`Shadow::copy_back`] for one range.
    fn copy_back_range(&self, address: usize, len: usize) -> usize {
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Before {
            snapshot,
            unchanged,
        } = &mut *before;
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        if let Some((taken, before)) = &*snapshot
            && *taken <= from
            && to <= taken + before.len()
            && from < to
        {
            unchanged.begin(start + taken, start + taken + before.len());
            let before = &before[from - taken..to - taken];
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission, title memory read only where the GPU wrote.
            // An unchanged page is never modified by copy-back, so later
            // overlapping ranges can skip it for the rest of the submission.
            let mut title = None;
            let mut synced = Vec::new();
            // A big range is compared ahead on several threads (the world
            // declares 100-170 MB a submission and its shaders write a
            // tenth of that); the pieces are then gone through in order.
            // (Only where that much is still to compare: a submission's
            // later ranges mostly lie in pages its first ones went through.)
            #[cfg(feature = "diagnostics")]
            let (clock, mut reading, mut merging, mut writing) =
                (std::time::Instant::now(), 0u64, 0u64, 0u64);
            #[cfg(feature = "diagnostics")]
            let since = |t: std::time::Instant| t.elapsed().as_micros() as u64;
            let ahead = (unchanged.unknown(start + from, start + to) >= crate::host::PARALLEL_FROM).then(|| {
                // SAFETY: inside the mapping, which only this thread writes.
                let host = unsafe { std::slice::from_raw_parts(self.data.add(from), to - from) };
                compare_ahead(host, start + from, before, unchanged)
            });
            let mut piece = 0;
            let mut at = from;
            while at < to {
                let n = (PAGE - (start + at) % PAGE).min(to - at);
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let differs = match &ahead {
                    Some(found) => unchanged.compared(start + at, n, found[piece]),
                    None => unchanged.differs(start + at, host, old),
                };
                piece += 1;
                #[cfg(feature = "diagnostics")]
                let t0 = std::time::Instant::now();
                if differs
                    && let title = &mut title.get_or_insert_with(|| vec![0u8; PAGE])[..n]
                    && {
                        let read = crate::host::read_into((start + at) as *const u8, title);
                        #[cfg(feature = "diagnostics")]
                        {
                            reading += since(t0);
                            crate::copystat::add(crate::copystat::C::BackReads, 1);
                        }
                        read
                    }
                {
                    // Title memory with the GPU's bytes put in; the copy
                    // takes the title's bytes in turn.
                    #[cfg(feature = "diagnostics")]
                    let t1 = std::time::Instant::now();
                    let mut changed = false;
                    for ((t, h), o) in title.iter_mut().zip(host.iter_mut()).zip(old) {
                        if *h != *o && *t != *h {
                            *t = *h;
                            changed = true;
                        }
                        *h = *t;
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        merging += since(t1);
                    }
                    if changed {
                        #[cfg(feature = "diagnostics")]
                        let t2 = std::time::Instant::now();
                        crate::watch::writing(start + at, n);
                        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                        written += n;
                        #[cfg(feature = "diagnostics")]
                        {
                            writing += since(t2);
                            crate::copystat::add(crate::copystat::C::BackWrites, 1);
                        }
                    }
                    synced.push((start + at) / PAGE);
                }
                at += n;
            }
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                add(C::BackCompareUs, since(clock).saturating_sub(reading + merging + writing));
                add(C::BackReadUs, reading);
                add(C::BackMergeUs, merging);
                add(C::BackWriteUs, writing);
            }
            // Pages made equal to title memory are current in the copy, so
            // our own write is not taken for the title's; others keep what
            // they had (a title write there shows at the next refresh).
            let (first, last) = ((start + from).div_ceil(PAGE), (start + to) / PAGE);
            if first < last
                && !synced.is_empty()
                && self.watched.load(Ordering::Relaxed) != NOT_WATCHED
                && let Some((now, _)) = crate::watch::writes(first * PAGE, last * PAGE, true)
            {
                let base = start / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in synced.into_iter().filter(|p| (first..last).contains(p)) {
                    copied[page - base] = now;
                }
            }
        }
        written
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        self.copy(from, to);
        to - from
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        // A page only partly in the range would stay stale for the rest and
        // be copied again by every later range in it (directly indexed SRVs
        // ask for many small ranges per page, each submission). Where no
        // shader of a submission may be writing the rest of an edge page,
        // the range takes in the whole page, within the buffer.
        let (from, to) = self.whole_edge_pages(from, to);
        let base = self.start as usize / PAGE;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut writes = written.iter().peekable();
        let (mut bytes, mut run) = (0, None);
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                // A page only partly in the range is copied in part: it
                // stays stale for the rest.
                let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                bytes += self.copy((first * PAGE).max(from), (page * PAGE).min(to));
            }
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some(first) = run {
            bytes += self.copy((first * PAGE).max(from), ((last + 1) * PAGE).min(to));
        }
        bytes
    }

    /// `[from, to)` widened to its first and last pages' bounds (cut to the
    /// buffer), each edge only where no pending range lies in what it adds.
    fn whole_edge_pages(&self, from: usize, to: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let lo = (from / PAGE * PAGE).max(start);
        let hi = (to.div_ceil(PAGE) * PAGE).min(end);
        if lo == from && hi == to {
            return (from, to);
        }
        let pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        let free = |a: usize, e: usize| a >= e || pending.iter().all(|&(p, q)| q <= a || p >= e);
        (
            if free(lo, from) { lo } else { from },
            if free(to, hi) { hi } else { to },
        )
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        to - from
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use.
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    if address < 0x10000 {
        return None;
    }
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(1 << 16))?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        entry.shadow = Some(Arc::new(Shadow::new(buffer, data, gpu, key.0, len)));
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (Grand
/// Theft Auto VI's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// shortcut: copies of windows are never freed; key them to the title's
/// allocations if memory use grows.
const WINDOW: u64 = 4 << 20;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64) -> Option<Arc<Shadow>> {
    let start = address & !(WINDOW - 1);
    let end = end.max(address.saturating_add(64 << 10));
    let len = end.checked_next_multiple_of(WINDOW)? - start;
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(shadow) = map.get(&(start, len)) {
        return Some(shadow.clone());
    }
    if !crate::texture::committed(address as *const u8, 1) {
        return None;
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, start, len));
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    #[test]
    fn partial_match_leaves_unchecked_bytes_visible() {
        let mut unchanged = UnchangedPages::default();
        let before = vec![0; PAGE];
        let mut host = before.clone();
        host[PAGE - 1] = 1;
        unchanged.begin(PAGE, 2 * PAGE);

        assert!(!unchanged.differs(PAGE, &host[..4], &before[..4]));
        assert!(!unchanged.pages[0]);
        assert!(unchanged.differs(PAGE, &host, &before));
        assert!(!unchanged.pages[0]);
    }

    #[test]
    fn snapshot_edges_are_cacheable_and_survive_later_ranges() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 3, &[1; 3], &[1; 3]));
        assert!(!unchanged.differs(PAGE, &[2; 4], &[2; 4]));
        assert_eq!(unchanged.pages, [true, true]);

        unchanged.begin(PAGE - 3, PAGE + 4);
        assert!(!unchanged.differs(PAGE - 2, &[1; 2], &[1; 2]));
        assert!(!unchanged.differs(PAGE + 1, &[2; 2], &[2; 2]));
        assert_eq!(unchanged.pages, [true, true]);
    }

    #[test]
    fn next_submission_checks_the_same_page_again() {
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, 4);
        assert!(!unchanged.differs(0, &[0; 4], &[0; 4]));
        unchanged.clear();
        unchanged.begin(0, 4);
        assert!(unchanged.differs(0, &[0, 0, 0, 1], &[0; 4]));
    }

    // Both versions walk exactly the same declared ranges, in their
    // original order. Only the page comparison differs.
    fn scan(
        host: &[u8],
        before: &[u8],
        ranges: &[Range<usize>],
        mut differs: impl FnMut(usize, &[u8], &[u8]) -> bool,
    ) -> usize {
        let mut changed = 0;
        for range in ranges {
            let mut at = range.start;
            while at < range.end {
                let n = (PAGE - at % PAGE).min(range.end - at);
                changed += usize::from(differs(at, &host[at..at + n], &before[at..at + n]));
                at += n;
            }
        }
        changed
    }

    #[test]
    fn overlapping_ranges_match_uncached_comparisons() {
        let before = vec![0; 4 * PAGE];
        let mut host = before.clone();
        for at in [2, PAGE - 1, PAGE + 5, 3 * PAGE + 7] {
            host[at] = 1;
        }
        let ranges = [
            0..4,
            2..2 * PAGE + 3,
            PAGE - 2..4 * PAGE,
            2 * PAGE..3 * PAGE,
        ];
        let uncached = scan(&host, &before, &ranges, |_, host, before| host != before);
        let mut unchanged = UnchangedPages::default();
        unchanged.begin(0, before.len());
        let cached = scan(&host, &before, &ranges, |at, host, before| {
            unchanged.differs(at, host, before)
        });
        assert_eq!(uncached, 6);
        assert_eq!(cached, uncached);
        assert_eq!(unchanged.pages, [false, false, true, false]);
    }

    #[test]
    fn comparing_ahead_finds_what_piece_by_piece_finds() {
        // A range that starts and ends inside pages, changes in some
        // pieces, and one page known unchanged from an earlier range.
        let (address, len) = (5 * PAGE + 123, 200 * PAGE + 4567);
        let before: Vec<u8> = (0..len).map(|i| (i / 7) as u8).collect();
        let mut host = before.clone();
        for at in [0, 5, PAGE - 124, PAGE - 123, 17 * PAGE, 100 * PAGE + 1, len - 1] {
            host[at] ^= 0x40;
        }
        let fresh = || {
            let mut unchanged = UnchangedPages::default();
            unchanged.begin(address, address + len);
            // The page 30 pages in was compared in full by an earlier range.
            let at = 30 * PAGE - 123;
            assert!(!unchanged.differs(address + at, &before[at..at + PAGE], &before[at..at + PAGE]));
            unchanged
        };
        let mut one = fresh();
        let mut expected = Vec::new();
        let mut at = 0;
        while at < len {
            let n = (PAGE - (address + at) % PAGE).min(len - at);
            expected.push(one.differs(address + at, &host[at..at + n], &before[at..at + n]));
            at += n;
        }
        let mut other = fresh();
        let ahead = compare_ahead(&host, address, &before, &other);
        assert_eq!(ahead.len(), expected.len());
        let mut at = 0;
        let got: Vec<bool> = ahead
            .iter()
            .map(|&found| {
                let n = (PAGE - (address + at) % PAGE).min(len - at);
                let differs = other.compared(address + at, n, found);
                at += n;
                differs
            })
            .collect();
        assert_eq!(got, expected);
        // All 201 pages of the range were unknown but the one; now all are
        // known but those of the five changed pieces.
        assert_eq!(fresh().unknown(address, address + len), 200 * PAGE);
        assert_eq!(other.unknown(address, address + len), 5 * PAGE);
        // Five pieces hold the seven changed bytes (three are in the first).
        assert_eq!(got.iter().filter(|&&d| d).count(), 5);
        assert_eq!(other.pages, one.pages);
    }

    #[test]
    fn big_snapshots_are_appended_whole() {
        let bytes: Vec<u8> = (0..crate::host::PARALLEL_FROM + 12_345).map(|i| (i * 31 >> 3) as u8).collect();
        let mut to = Vec::with_capacity(bytes.len() + 10);
        to.extend_from_slice(b"head");
        extend(&mut to, &bytes);
        assert_eq!(&to[..4], b"head");
        assert!(to[4..] == bytes[..]);
        // No room: the plain way.
        let mut small = Vec::new();
        extend(&mut small, &bytes);
        assert!(small == bytes);
    }

    // cargo test --release -p victory-gpu benchmark_overlapping_copy_back -- --ignored --nocapture
    #[test]
    #[ignore = "synthetic CPU release benchmark"]
    fn benchmark_overlapping_copy_back() {
        use std::hint::black_box;
        use std::time::{Duration, Instant};

        const SIZE: usize = 128 << 20;
        let before = vec![0xA5; SIZE];
        let mut host = before.clone();
        for page in (0..SIZE / PAGE).step_by(64) {
            host[page * PAGE + PAGE / 2] ^= 0xFF;
        }
        let ranges: Vec<_> = (0..24).map(|i| i * PAGE + i..SIZE - i * 13).collect();
        let mut baseline = [Duration::ZERO; 5];
        let mut cached = [Duration::ZERO; 5];
        let mut unchanged = UnchangedPages::default();
        for (baseline, cached) in baseline.iter_mut().zip(&mut cached) {
            let started = Instant::now();
            let expected = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |_, host, before| host != before,
            );
            *baseline = started.elapsed();

            unchanged.clear();
            let started = Instant::now();
            unchanged.begin(0, before.len());
            let actual = scan(
                black_box(&host),
                black_box(&before),
                black_box(&ranges),
                |at, host, before| unchanged.differs(at, host, before),
            );
            *cached = started.elapsed();
            assert_eq!(actual, expected);
        }
        baseline.sort_unstable();
        cached.sort_unstable();
        println!(
            "median of five runs, 24 overlapping ranges across 128 MiB: {:?} -> {:?} ({:.1}x)",
            baseline[2],
            cached[2],
            baseline[2].as_secs_f64() / cached[2].as_secs_f64(),
        );
    }

    // PR #436's tests.
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    /// A write-watched title allocation of two pages, registered as a
    /// buffer, with its copy.
    struct Watched {
        base: usize,
        shadow: Arc<Shadow>,
    }

    impl Watched {
        fn new() -> Option<Watched> {
            const MEM_WRITE_WATCH: u32 = 0x20_0000;
            crate::host::host()?;
            // SAFETY: a new committed, write-watched allocation the test owns.
            let at = unsafe {
                VirtualAlloc(
                    std::ptr::null(),
                    2 * PAGE,
                    MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                    PAGE_READWRITE,
                )
            };
            assert!(!at.is_null());
            let base = at as usize;
            register(base, base as u64, 2 * PAGE as u64);
            let shadow = find(base as u64, 2 * PAGE as u64).unwrap();
            Some(Watched { base, shadow })
        }
    }

    impl Drop for Watched {
        fn drop(&mut self) {
            unregister(self.base, self.base as u64);
            // SAFETY: allocated with VirtualAlloc in `new`.
            unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// Small ranges in one page (as directly indexed SRVs ask for them): the
    /// first copies the page, the others nothing until the title writes it.
    #[test]
    fn small_ranges_in_one_page_copy_it_once() {
        let Some(t) = Watched::new() else {
            return;
        };
        let page = t.base + PAGE;
        // SAFETY: inside the test's allocation.
        unsafe { (page as *mut u32).write(1) };
        assert_eq!(t.shadow.update(page + 16, 16), PAGE);
        for offset in (256..PAGE).step_by(4096) {
            assert_eq!(t.shadow.update(page + offset, 16), 0);
        }
        // SAFETY: inside the test's allocation.
        unsafe { ((page + 8192) as *mut u32).write(500) };
        assert_eq!(t.shadow.update(page + 16, 16), PAGE);
        // SAFETY: the copy's mapping covers the allocation.
        let copied = unsafe { t.shadow.data.add(PAGE + 8192).cast::<u32>().read() };
        assert_eq!(copied, 500);
    }

    /// Where shaders of a submission may be writing the rest of the page,
    /// the range is copied as asked and the copy there is left alone.
    #[test]
    fn edge_pages_stay_cut_next_to_pending_writes() {
        let Some(t) = Watched::new() else {
            return;
        };
        let page = t.base + PAGE;
        t.shadow.writing(page + 1024, 64);
        // SAFETY: inside the copy's mapping and the test's allocation.
        unsafe {
            t.shadow.data.add(PAGE + 1024).cast::<u32>().write(7);
            ((page + 1024) as *mut u32).write(13);
        }
        assert_eq!(t.shadow.update(page + 16, 16), 32);
        // SAFETY: inside the copy's mapping.
        let kept = unsafe { t.shadow.data.add(PAGE + 1024).cast::<u32>().read() };
        assert_eq!(kept, 7);
    }
}
