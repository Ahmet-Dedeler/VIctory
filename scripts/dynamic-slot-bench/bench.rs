use std::collections::BTreeMap;
use std::hint::black_box;
use std::ops::Bound::{Excluded, Unbounded};
use std::sync::Mutex;
use std::sync::atomic::AtomicU32;
use std::time::{Duration, Instant};

// Let the extracted methods compile without loading any GPU or tracing code.
extern crate self as tracing;
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {{}};
}

const TITLE_SLOTS: u32 = 1_048_576;
const PORTABLE_DYNAMIC: u32 = 1 << 18;
const MAC_DYNAMIC_CEILING: u32 = 1 << 20;

struct HostHeap {
    sampler: bool,
    dynamic: u32,
    count: u32,
    next: Mutex<(u32, Vec<bool>)>,
    free: AtomicU32,
    too_long: AtomicU32,
}

include!("baseline.rs");

trait Allocator {
    fn new(capacity: u32) -> Self;
    fn allocate(&self, width: u32) -> Option<u32>;
    fn give_back(&self, first: u32, width: u32);
}

impl Allocator for HostHeap {
    fn new(capacity: u32) -> Self {
        Self {
            sampler: false,
            dynamic: capacity,
            count: TITLE_SLOTS,
            next: Mutex::new((0, vec![false; capacity as usize])),
            free: AtomicU32::new(capacity),
            too_long: AtomicU32::new(u32::MAX),
        }
    }

    fn allocate(&self, width: u32) -> Option<u32> {
        self.dynamic(width)
    }

    fn give_back(&self, first: u32, width: u32) {
        self.release(first, width);
    }
}

struct Indexed {
    capacity: u32,
    state: Mutex<FreeRuns>,
}

struct FreeRuns {
    /// Maximal, nonempty free intervals keyed by exclusive end.
    runs: BTreeMap<u32, u32>,
    cursor: u32,
    available: u32,
    too_long: u32,
}

impl FreeRuns {
    fn find(&self, width: u32) -> Option<(u32, u32)> {
        self.runs
            .range((Excluded(self.cursor), Unbounded))
            .find_map(|(&end, &start)| {
                let first = start.max(self.cursor);
                (end - first >= width).then_some((first, end))
            })
            .or_else(|| {
                // The original wrap search includes starts below the cursor
                // even if the chosen run extends beyond it.
                self.runs
                    .iter()
                    .take_while(|(_, start)| **start < self.cursor)
                    .find_map(|(&end, &start)| (end - start >= width).then_some((start, end)))
            })
    }

    fn take(&mut self, width: u32, capacity: u32) -> Option<u32> {
        if width > self.available || width >= self.too_long {
            return None;
        }
        let Some((first, end)) = self.find(width) else {
            self.too_long = width;
            return None;
        };
        let after = first + width;
        let start = self.runs[&end];
        if after == end {
            self.runs.remove(&end);
        } else {
            *self.runs.get_mut(&end).unwrap() = after;
        }
        if start < first {
            self.runs.insert(first, start);
        }
        self.available -= width;
        self.cursor = if after == capacity { 0 } else { after };
        Some(first)
    }

    fn release(&mut self, start: u32, end: u32) {
        self.too_long = u32::MAX;
        if start == end {
            return;
        }
        let mut joined_start = start;
        let mut joined_end = end;
        let mut given = end - start;
        while let Some((&known_end, &known_start)) = self.runs.range(start..).next() {
            if known_start > joined_end {
                break;
            }
            given -= known_end.min(end).saturating_sub(known_start.max(start));
            joined_start = joined_start.min(known_start);
            joined_end = joined_end.max(known_end);
            self.runs.remove(&known_end);
        }
        self.runs.insert(joined_end, joined_start);
        self.available += given;
    }
}

impl Allocator for Indexed {
    fn new(capacity: u32) -> Self {
        Self {
            capacity,
            state: Mutex::new(FreeRuns {
                runs: BTreeMap::from([(capacity, 0)]),
                cursor: 0,
                available: capacity,
                too_long: u32::MAX,
            }),
        }
    }

    fn allocate(&self, width: u32) -> Option<u32> {
        if width == 0 || width > self.capacity {
            return None;
        }
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take(width, self.capacity)
            .map(|first| first + TITLE_SLOTS)
    }

    fn give_back(&self, first: u32, width: u32) {
        let Some(start) = first.checked_sub(TITLE_SLOTS) else {
            return;
        };
        if start <= self.capacity {
            let end = (start + width).min(self.capacity);
            self.state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .release(start, end);
        }
    }
}

struct Packed {
    capacity: u32,
    state: Mutex<PackedSlots>,
}

struct PackedSlots {
    taken: Vec<u64>,
    cursor: u32,
    available: u32,
    too_long: u32,
}

fn low_bits(width: u32) -> u64 {
    u64::MAX >> (u64::BITS - width)
}

impl PackedSlots {
    fn first_run(&self, start: u32, end: u32, width: u32) -> Option<u32> {
        let mut suffix = 0;
        let mut at = start;
        while at < end {
            let offset = at % u64::BITS;
            let valid = (u64::BITS - offset).min(end - at);
            let free = !(self.taken[(at / u64::BITS) as usize] >> offset) & low_bits(valid);
            let prefix = free.trailing_ones();
            if suffix + prefix >= width {
                return Some(at - suffix);
            }
            if prefix == valid {
                suffix += valid;
            } else {
                if width <= valid && free != 0 {
                    // Each doubling keeps starts with twice as many free bits.
                    // Two overlapping power-of-two runs then cover `width`.
                    let mut starts = free;
                    let mut cover = 1;
                    while cover * 2 <= width {
                        starts &= starts >> cover;
                        cover *= 2;
                    }
                    starts &= starts >> (width - cover);
                    if starts != 0 {
                        return Some(at + starts.trailing_zeros());
                    }
                }
                suffix = (free << (u64::BITS - valid)).leading_ones();
            }
            at += valid;
        }
        None
    }

    fn update<const TAKE: bool>(&mut self, mut at: u32, end: u32) -> u32 {
        let mut given = 0;
        while at < end {
            let offset = at % u64::BITS;
            let valid = (u64::BITS - offset).min(end - at);
            let mask = low_bits(valid) << offset;
            let bits = &mut self.taken[(at / u64::BITS) as usize];
            if TAKE {
                *bits |= mask;
            } else {
                given += (*bits & mask).count_ones();
                *bits &= !mask;
            }
            at += valid;
        }
        given
    }
}

impl Allocator for Packed {
    fn new(capacity: u32) -> Self {
        Self {
            capacity,
            state: Mutex::new(PackedSlots {
                taken: vec![0; capacity.div_ceil(u64::BITS) as usize],
                cursor: 0,
                available: capacity,
                too_long: u32::MAX,
            }),
        }
    }

    fn allocate(&self, width: u32) -> Option<u32> {
        if width == 0 || width > self.capacity {
            return None;
        }
        let mut heap = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if width > heap.available || width >= heap.too_long {
            return None;
        }
        let second_end = (heap.cursor + width - 1).min(self.capacity);
        let first = heap
            .first_run(heap.cursor, self.capacity, width)
            .or_else(|| heap.first_run(0, second_end, width));
        let Some(first) = first else {
            heap.too_long = width;
            return None;
        };
        let after = first + width;
        heap.update::<true>(first, after);
        heap.available -= width;
        heap.cursor = if after == self.capacity { 0 } else { after };
        Some(TITLE_SLOTS + first)
    }

    fn give_back(&self, first: u32, width: u32) {
        let Some(start) = first.checked_sub(TITLE_SLOTS) else {
            return;
        };
        if start <= self.capacity {
            let mut heap = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let end = (start + width).min(self.capacity);
            heap.available += heap.update::<false>(start, end);
            heap.too_long = u32::MAX;
        }
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Take(u32, Option<u32>),
    Release(u32, u32),
}

struct Trace {
    baseline: HostHeap,
    operations: Vec<Operation>,
}

impl Trace {
    fn new(capacity: u32) -> Self {
        Self {
            baseline: HostHeap::new(capacity),
            operations: Vec::new(),
        }
    }

    fn allocate(&mut self, width: u32) -> Option<u32> {
        let first = self.baseline.allocate(width);
        self.operations.push(Operation::Take(width, first));
        first
    }

    fn give_back(&mut self, first: u32, width: u32) {
        self.baseline.give_back(first, width);
        self.operations.push(Operation::Release(first, width));
    }

    fn chunk(&mut self) -> Vec<Operation> {
        std::mem::take(&mut self.operations)
    }
}

struct Workload {
    name: &'static str,
    capacity: u32,
    setup: Vec<Operation>,
    frames: Vec<Vec<Operation>>,
}

struct Random(u64);

impl Random {
    fn below(&mut self, bound: u32) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 32) as u32) % bound
    }
}

fn render_load(
    name: &'static str,
    capacity: u32,
    requests: usize,
    frames_held: usize,
    samples: usize,
) -> Workload {
    const GROUPS: usize = 32;
    // Widths are synthetic. Widths 12 and 13 occur in the archived warnings;
    // width 1 represents individual root descriptor allocations.
    const WIDTHS: [u32; 12] = [1, 1, 1, 1, 3, 4, 6, 8, 10, 12, 13, 16];
    let mut random = Random(0xda7a);
    let mut trace = Trace::new(capacity);
    let mut held = vec![vec![Vec::new(); GROUPS]; frames_held];
    let mut setup = Vec::new();
    let mut frames = Vec::new();
    for frame in 0..frames_held + samples {
        let bank = &mut held[frame % frames_held];
        for (group_index, group) in bank.iter_mut().enumerate() {
            for (first, width) in group.drain(..).rev() {
                trace.give_back(first, width);
            }
            let count = requests / GROUPS + usize::from(group_index < requests % GROUPS);
            for _ in 0..count {
                let width = WIDTHS[random.below(WIDTHS.len() as u32) as usize];
                if let Some(first) = trace.allocate(width) {
                    group.push((first, width));
                }
            }
        }
        if frame < frames_held {
            setup.extend(trace.chunk());
        } else {
            frames.push(trace.chunk());
        }
    }
    Workload {
        name,
        capacity,
        setup,
        frames,
    }
}

fn fragmented(name: &'static str, gap: u32, step: u32, samples: usize) -> Workload {
    let capacity = PORTABLE_DYNAMIC;
    let mut trace = Trace::new(capacity);
    let mut holes = Vec::new();
    for _ in 0..capacity / step {
        holes.push(trace.allocate(gap).unwrap());
        trace.allocate(step - gap).unwrap();
    }
    for first in holes {
        trace.give_back(first, gap);
    }
    let setup = trace.chunk();
    let mut frames = Vec::new();
    for _ in 0..samples {
        for _ in 0..68 {
            // Free space is ample in aggregate. Givebacks invalidate the
            // too-long shortcut, so each repeated miss exercises a search.
            assert_eq!(trace.allocate(13), None);
            let first = trace.allocate(1).unwrap();
            trace.give_back(first, 1);
        }
        frames.push(trace.chunk());
    }
    Workload {
        name,
        capacity,
        setup,
        frames,
    }
}

fn replay<A: Allocator, const VERIFY: bool>(allocator: &A, operations: &[Operation]) {
    for &op in operations {
        match op {
            Operation::Take(width, expected) => {
                let actual = allocator.allocate(width);
                if VERIFY {
                    assert_eq!(actual, expected, "width {width}");
                } else {
                    black_box(actual);
                }
            }
            Operation::Release(first, width) => allocator.give_back(first, width),
        }
    }
}

fn verify_small_cases() {
    for capacity in 1..=128 {
        let mut random = Random(u64::from(capacity));
        let mut trace = Trace::new(capacity);
        // Cover full, sparse, duplicate releases, invalid widths, overlaps,
        // and wraparound against the actual baseline methods.
        for _ in 0..2048 {
            let width = random.below(capacity + 2);
            if random.below(4) == 0 {
                let first = TITLE_SLOTS - 1 + random.below(capacity + 3);
                trace.give_back(first, width);
            } else {
                trace.allocate(width);
            }
        }
        replay::<Indexed, true>(&Indexed::new(capacity), &trace.operations);
        replay::<Packed, true>(&Packed::new(capacity), &trace.operations);
    }
    println!("Verified 262144 mixed operations across 128 small heap sizes");
}

fn verify_workload<A: Allocator>(w: &Workload) {
    let allocator = A::new(w.capacity);
    replay::<A, true>(&allocator, &w.setup);
    for frame in &w.frames {
        replay::<A, true>(&allocator, frame);
    }
}

fn measure<A: Allocator>(allocator: &A, frame: &[Operation]) -> Duration {
    let start = Instant::now();
    replay::<A, false>(allocator, frame);
    start.elapsed()
}

fn report(label: &str, times: &mut [Duration]) {
    times.sort_unstable();
    let rank = |p: usize| times[(times.len() * p).div_ceil(100) - 1].as_secs_f64() * 1000.0;
    println!(
        "{label:<8} p50 {:>8.3} ms | p95 {:>8.3} ms",
        rank(50),
        rank(95)
    );
}

fn benchmark(w: &Workload) {
    verify_workload::<Indexed>(w);
    verify_workload::<Packed>(w);
    let scan = HostHeap::new(w.capacity);
    let index = Indexed::new(w.capacity);
    let packed = Packed::new(w.capacity);
    replay::<HostHeap, false>(&scan, &w.setup);
    replay::<Indexed, false>(&index, &w.setup);
    replay::<Packed, false>(&packed, &w.setup);
    let mut scan_times = Vec::new();
    let mut index_times = Vec::new();
    let mut packed_times = Vec::new();
    for (sample, frame) in w.frames.iter().enumerate() {
        for position in 0..3 {
            match (sample + position) % 3 {
                0 => scan_times.push(measure(&scan, frame)),
                1 => index_times.push(measure(&index, frame)),
                _ => packed_times.push(measure(&packed, frame)),
            }
        }
    }
    let mut asks = 0;
    let mut refused = 0;
    let mut releases = 0;
    for op in w.frames.iter().flatten() {
        match op {
            Operation::Take(_, found) => {
                asks += 1;
                refused += usize::from(found.is_none());
            }
            Operation::Release(..) => releases += 1,
        }
    }
    let frames = w.frames.len();
    println!(
        "{}: {} slots, {frames} paired frames, ~{} allocations, ~{} releases, {:.1} refused per frame",
        w.name,
        w.capacity,
        asks / frames,
        releases / frames,
        refused as f64 / frames as f64,
    );
    report("base", &mut scan_times);
    report("indexed", &mut index_times);
    report("packed", &mut packed_times);
}

fn main() {
    let samples: usize = std::env::args().nth(1).unwrap().parse().unwrap();
    assert!(samples > 0);
    verify_small_cases();
    benchmark(&render_load("steady", PORTABLE_DYNAMIC, 16_682, 2, samples));
    benchmark(&render_load(
        "high load",
        MAC_DYNAMIC_CEILING,
        33_337,
        4,
        samples,
    ));
    benchmark(&fragmented("fragmented, 256 holes", 11, 1024, samples));
    benchmark(&fragmented("fragmented, 65536 holes", 1, 4, samples));
}
