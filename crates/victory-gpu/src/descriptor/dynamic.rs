//! The occupancy of the host heap's dynamic slots. Every mutation is under
//! the host heap's allocator lock and takes effect before the next request.

pub(crate) struct DynamicSlots {
    capacity: u32,
    storage: Storage,
    cursor: u32,
    available: u32,
    /// A request at least this large cannot fit until a release.
    too_long: u32,
}

enum Storage {
    Bytes(Vec<bool>),
    Packed(Vec<u64>),
}

impl DynamicSlots {
    pub(crate) fn new(capacity: u32, packed: bool) -> Self {
        let storage = if packed {
            Storage::Packed(vec![0; capacity.div_ceil(u64::BITS) as usize])
        } else {
            Storage::Bytes(vec![false; capacity as usize])
        };
        Self {
            capacity,
            storage,
            cursor: 0,
            available: capacity,
            too_long: u32::MAX,
        }
    }

    /// Takes a physically contiguous run. When requested, the count
    /// includes slots inspected in byte storage or words loaded in packed
    /// storage. Skipped searches count zero.
    pub(crate) fn allocate<const COUNT: bool>(&mut self, width: u32) -> (Option<u32>, u32) {
        if width == 0 || width > self.available || width >= self.too_long {
            return (None, 0);
        }
        let cursor = self.cursor;
        let end = self.capacity;
        // Permit a run starting below the cursor and ending just above it.
        // The end of the heap itself never wraps into the start.
        let wrap_end = cursor.saturating_add(width - 1).min(end);
        let found = match &self.storage {
            Storage::Bytes(taken) => bytes_first(taken, cursor, end, width)
                .or_else(|| bytes_first(taken, 0, wrap_end, width)),
            Storage::Packed(taken) => packed_first(taken, cursor, end, width)
                .or_else(|| packed_first(taken, 0, wrap_end, width)),
        };
        let examined = if COUNT {
            let units = |start: u32, end: u32| match &self.storage {
                Storage::Bytes(_) => end - start,
                Storage::Packed(_) if start == end => 0,
                Storage::Packed(_) => end.div_ceil(u64::BITS) - start / u64::BITS,
            };
            match found {
                Some(first) if first >= cursor => units(cursor, first + width),
                Some(first) => units(cursor, end) + units(0, first + width),
                None => units(cursor, end) + units(0, wrap_end),
            }
        } else {
            0
        };
        let Some(first) = found else {
            self.too_long = width;
            return (None, examined);
        };
        let after = first + width;
        match &mut self.storage {
            Storage::Bytes(taken) => taken[first as usize..after as usize].fill(true),
            Storage::Packed(taken) => {
                update::<true>(taken, first, after);
            }
        }
        self.available -= width;
        self.cursor = if after == self.capacity { 0 } else { after };
        (Some(first), examined)
    }

    pub(crate) fn release(&mut self, first: u32, width: u32) {
        if first > self.capacity {
            return;
        }
        let after = (first + width).min(self.capacity);
        let given = match &mut self.storage {
            Storage::Bytes(taken) => {
                let old = &mut taken[first as usize..after as usize];
                let given = old.iter().filter(|taken| **taken).count() as u32;
                old.fill(false);
                given
            }
            Storage::Packed(taken) => update::<false>(taken, first, after),
        };
        self.available += given;
        self.too_long = u32::MAX;
    }

    #[cfg(any(feature = "diagnostics", test))]
    pub(crate) fn taken(&self, index: u32) -> Option<bool> {
        if index >= self.capacity {
            return None;
        }
        match &self.storage {
            Storage::Bytes(taken) => Some(taken[index as usize]),
            Storage::Packed(taken) => {
                let word = taken[(index / u64::BITS) as usize];
                Some(word >> (index % u64::BITS) & 1 != 0)
            }
        }
    }
}

fn bytes_first(taken: &[bool], start: u32, end: u32, width: u32) -> Option<u32> {
    let mut run = 0;
    for at in start..end {
        run = if taken[at as usize] { 0 } else { run + 1 };
        if run == width {
            return Some(at + 1 - width);
        }
    }
    None
}

fn low_bits(width: u32) -> u64 {
    u64::MAX >> (u64::BITS - width)
}

fn packed_first(taken: &[u64], start: u32, end: u32, width: u32) -> Option<u32> {
    let mut suffix = 0;
    let mut at = start;
    while at < end {
        let offset = at % u64::BITS;
        let valid = (u64::BITS - offset).min(end - at);
        let free = !(taken[(at / u64::BITS) as usize] >> offset) & low_bits(valid);
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

fn update<const TAKE: bool>(taken: &mut [u64], mut at: u32, end: u32) -> u32 {
    let mut given = 0;
    while at < end {
        let offset = at % u64::BITS;
        let valid = (u64::BITS - offset).min(end - at);
        let mask = low_bits(valid) << offset;
        let bits = &mut taken[(at / u64::BITS) as usize];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_run_crossing_words_and_the_cursor() {
        for packed in [false, true] {
            let mut slots = DynamicSlots::new(129, packed);
            assert_eq!(slots.allocate::<false>(62).0, Some(0));
            assert_eq!(slots.allocate::<false>(3).0, Some(62));
            assert_eq!(slots.allocate::<false>(5).0, Some(65));
            assert_eq!(slots.allocate::<false>(59).0, Some(70));
            slots.release(62, 3);
            assert_eq!(slots.allocate::<false>(3).0, Some(62));
            slots.release(62, 8);
            assert_eq!(
                slots.allocate::<true>(8),
                (Some(62), if packed { 4 } else { 134 })
            );
            assert_eq!(slots.taken(62), Some(true));
            assert_eq!(slots.taken(69), Some(true));
            assert_eq!(slots.taken(129), None);
        }
    }

    #[test]
    fn failed_search_is_cached_until_a_release() {
        for packed in [false, true] {
            let mut slots = DynamicSlots::new(130, packed);
            assert_eq!(slots.allocate::<false>(130).0, Some(0));
            slots.release(0, 1);
            slots.release(129, 1);
            assert_eq!(
                slots.allocate::<true>(2),
                (None, if packed { 4 } else { 131 })
            );
            assert_eq!(slots.allocate::<true>(2), (None, 0));
            slots.release(0, 1);
            assert_eq!(slots.available, 2);
            assert_eq!(
                slots.allocate::<true>(2),
                (None, if packed { 4 } else { 131 })
            );
            slots.release(1, 1);
            assert_eq!(
                slots.allocate::<true>(2),
                (Some(0), if packed { 1 } else { 2 })
            );
        }
    }
}
