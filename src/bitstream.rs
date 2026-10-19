//! The packed `BlockStates` bit stream, read and written through one set of
//! abstractions.
//!
//! A litematic region stores palette indices as a continuous LSB-first bit
//! stream carried by big-endian longs: entry `i` starts at bit `i * width` and
//! may straddle two words. [`BitStream`] is the immutable view, [`BlockIter`]
//! walks it, and [`Packer`] produces it. Keeping both directions here means
//! the width rule and the word layout are stated once, next to each other.

use crate::error::Result;

// Width
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The litematica bit width for `len` palette entries: the smallest `n >= 2`
/// with `2^n >= len`.
///
/// Both readers and the writer derive their width here, so a file written by
/// this crate cannot disagree with one read by it.
pub(crate) fn bits_for(len: usize) -> u32 {
    (2..32).find(|n| (1u64 << n) >= len as u64).unwrap_or(32)
}

// Unpacking
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The packed `BlockStates` payload viewed as fixed-width entries.
///
/// The payload is a run of big-endian longs, but entries are not aligned to
/// them: entry `i` starts at bit `i * bits` of one continuous stream, least
/// significant bit first, and may straddle two longs. [`BitCursor`] performs
/// that decode for both sequential and random access, so the layout of the
/// stream is stated once.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BitStream<'a> {
    longs: &'a [[u8; 8]],
    bits: u32,
    mask: u64,
}

impl<'a> BitStream<'a> {
    /// Views `bytes` as `bits`-wide entries.
    pub(crate) fn new(bytes: &'a [u8], bits: u32) -> Self {
        // A long array payload is a whole number of 8-byte entries, so the
        // remainder is empty for well-formed input and ignored otherwise.
        let (longs, _remainder) = bytes.as_chunks::<8>();
        Self {
            longs,
            bits,
            mask: (1u64 << bits) - 1,
        }
    }

    /// Iterates over `volume` entries in storage order.
    pub(crate) fn iter(self, volume: u64) -> BlockIter<'a> {
        BlockIter::new(self, volume)
    }

    /// Entry `index`, or `None` when the payload does not reach it.
    pub(crate) fn entry(self, index: u64) -> Option<u32> {
        let bit = u128::from(index) * u128::from(self.bits);
        let long = usize::try_from(bit / 64).ok()?;
        let shift = (bit % 64) as u32;
        let first = u64::from_be_bytes(*self.longs.get(long)?);
        let mut source = SliceLongs {
            stream: self,
            next: long + 1,
        };
        let mut cursor = BitCursor::with_remainder(first >> shift, 64 - shift);
        cursor.next(self.bits, self.mask, &mut source).ok()
    }

    /// The `index`th long, zero-filled when the payload is shorter.
    fn long(self, index: usize) -> u64 {
        self.longs
            .get(index)
            .copied()
            .map(u64::from_be_bytes)
            .unwrap_or(0)
    }

    /// Number of payload bytes this stream windows, longs only.
    pub(crate) fn byte_len(self) -> usize {
        self.longs.len() * 8
    }
}

/// A source of 64-bit words for [`BitCursor`].
///
/// The slice-backed random reader and the streaming reader supply words from
/// different places, but decode entries through the same cursor.
pub(crate) trait LongSource {
    /// The next word, least significant byte of the stream first.
    fn next_long(&mut self) -> Result<u64>;
}

/// A forward cursor over a continuous bit stream, buffering one word.
///
/// Sequential iteration is the cursor's natural use: each step consumes the
/// low `bits` of the buffer and refills only when the next entry crosses a
/// long boundary.
#[derive(Debug, Default)]
pub(crate) struct BitCursor {
    cur: u64,
    cur_bits: u32,
}

impl BitCursor {
    /// Creates a cursor with no buffered word.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    fn with_remainder(cur: u64, cur_bits: u32) -> Self {
        Self { cur, cur_bits }
    }

    /// Decodes one entry, pulling words from `source` as needed.
    #[inline]
    pub(crate) fn next(
        &mut self,
        bits: u32,
        mask: u64,
        source: &mut impl LongSource,
    ) -> Result<u32> {
        if self.cur_bits >= bits {
            // Fast path: enough buffered bits for this entry.
            let value = (self.cur & mask) as u32;
            self.cur >>= bits;
            self.cur_bits -= bits;
            return Ok(value);
        }
        // Slow path: the entry straddles the word boundary, low bits from
        // the buffer, the rest from the next word.
        let low = self.cur_bits;
        let mut value = self.cur & mask;
        self.cur = source.next_long()?;
        self.cur_bits = 64;
        value |= (self.cur & (mask >> low)) << low;
        self.cur >>= bits - low;
        self.cur_bits -= bits - low;
        Ok(value as u32)
    }
}

/// A [`LongSource`] reading words borrowed from a [`BitStream`].
#[derive(Debug)]
struct SliceLongs<'a> {
    stream: BitStream<'a>,
    next: usize,
}

impl LongSource for SliceLongs<'_> {
    fn next_long(&mut self) -> Result<u64> {
        let word = self.stream.long(self.next);
        self.next += 1;
        Ok(word)
    }
}

// Block iterator
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// An [`Iterator`] over the palette indices of a region, decoded on the fly
/// from the `BlockStates` bit stream.
///
/// The whole long array is treated as one continuous bit stream (entries may
/// cross long boundaries). Within a big-endian long, bit 0 is the least
/// significant bit; the first bit in the stream is the lowest bit of the
/// first entry. This matches the packing used by the Litematica mod and
/// rustmatica.
#[derive(Debug)]
#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct BlockIter<'a> {
    cursor: BitCursor,
    source: SliceLongs<'a>,
    bits: u32,
    mask: u64,
    remaining: u64,
}

impl<'a> BlockIter<'a> {
    fn new(stream: BitStream<'a>, volume: u64) -> Self {
        Self {
            cursor: BitCursor::new(),
            source: SliceLongs { stream, next: 0 },
            bits: stream.bits,
            mask: stream.mask,
            remaining: volume,
        }
    }
}

impl Iterator for BlockIter<'_> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.cursor
            .next(self.bits, self.mask, &mut self.source)
            .ok()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.len();
        (n, Some(n))
    }
}

impl ExactSizeIterator for BlockIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        usize::try_from(self.remaining).unwrap_or(usize::MAX)
    }
}

/// Decodes every palette index of a packed payload, calling `f` once per
/// voxel.
///
/// This is the owned-buffer counterpart of [`BlockIter`], used by the
/// streaming reader. Both routes share the same cursor, so they decode
/// identically.
pub(crate) fn decode_payload(
    bytes: &[u8],
    bits: u32,
    volume: u64,
    f: impl FnMut(u32) -> Result<()>,
) -> Result<()> {
    BitStream::new(bytes, bits).iter(volume).try_for_each(f)
}

// Packing
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// Packs palette indices into the `BlockStates` bit stream.
///
/// The mirror of [`BlockIter`]: one continuous LSB-first stream of
/// `bits`-wide entries, big-endian longs, entries straddling long boundaries.
/// A long is returned the moment 64 bits have accumulated;
/// [`Packer::finish`] flushes the trailing partial long, zero-padded.
pub(crate) struct Packer {
    acc: u128,
    have: u32,
    bits: u32,
}

impl Packer {
    /// Creates a packer emitting `bits`-wide entries.
    pub(crate) fn new(bits: u32) -> Self {
        Self {
            acc: 0,
            have: 0,
            bits,
        }
    }

    /// Packs one palette index, returning a completed long when one fills.
    #[inline]
    pub(crate) fn push(&mut self, id: u32) -> Option<u64> {
        self.acc |= u128::from(id) << self.have;
        self.have += self.bits;
        if self.have >= 64 {
            let word = self.acc as u64;
            self.acc >>= 64;
            self.have -= 64;
            Some(word)
        } else {
            None
        }
    }

    /// Flushes the pending partial long, zero-padded to 64 bits.
    pub(crate) fn finish(&mut self) -> Option<u64> {
        if self.have == 0 {
            return None;
        }
        let word = self.acc as u64;
        self.acc = 0;
        self.have = 0;
        Some(word)
    }
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::{bits_for, BitStream, BlockIter, Packer};

    #[test]
    fn bits_for_matches_litematica() {
        assert_eq!(bits_for(1), 2);
        assert_eq!(bits_for(2), 2);
        assert_eq!(bits_for(3), 2);
        assert_eq!(bits_for(4), 2);
        assert_eq!(bits_for(5), 3);
        assert_eq!(bits_for(8), 3);
        assert_eq!(bits_for(9), 4);
        assert_eq!(bits_for(16), 4);
        assert_eq!(bits_for(17), 5);
        assert_eq!(bits_for(64), 6);
        assert_eq!(bits_for(65), 7);
        assert_eq!(bits_for(100), 7);
    }

    #[test]
    fn iter_decodes_lsb_first_stream() {
        // 3 bits per entry, values 1, 2, 6, 5, 7 -> bits cross long boundary.
        let values = [1u32, 2, 6, 5, 7];
        let mut acc = 0u64;
        let mut filled = 0u32;
        for &v in &values {
            for k in 0..3 {
                acc |= u64::from((v >> k) & 1) << filled;
                filled += 1;
            }
        }
        let bytes = acc.to_be_bytes();
        let iter = BlockIter::new(BitStream::new(&bytes, 3), values.len() as u64);
        assert_eq!(iter.len(), values.len());
        let decoded: Vec<u32> = iter.collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn iter_entries_crossing_long_boundaries() {
        // 5 bits per entry, 20 entries -> 100 bits, crossing 64-bit borders.
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let bytes = pack(&values, 5);
        let decoded: Vec<u32> =
            BlockIter::new(BitStream::new(&bytes, 5), values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn random_entries_match_the_sequential_stream() {
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let bytes = pack(&values, 5);
        let stream = BitStream::new(&bytes, 5);
        for (i, &want) in values.iter().enumerate() {
            assert_eq!(stream.entry(i as u64), Some(want), "entry {i}");
        }
        assert_eq!(stream.entry(100), None);
    }

    #[test]
    fn packer_round_trips_width_3_through_block_iter() {
        let values = [1u32, 2, 6, 5, 7]; // 15 bits -> one partial long
        let bytes = pack(&values, 3);
        let decoded: Vec<u32> =
            BlockIter::new(BitStream::new(&bytes, 3), values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn packer_round_trips_crossing_long_boundaries() {
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect(); // 100 bits
        let bytes = pack(&values, 5);
        let decoded: Vec<u32> =
            BlockIter::new(BitStream::new(&bytes, 5), values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn packer_exact_multiple_flushes_no_tail() {
        let mut packer = Packer::new(2);
        let mut longs = Vec::new();
        for _ in 0..32 {
            if let Some(word) = packer.push(3) {
                longs.push(word);
            }
        }
        assert_eq!(longs.len(), 1); // 64 bits exactly
        assert!(packer.finish().is_none());
    }

    #[test]
    fn iter_zero_volume_is_empty() {
        let mut iter = BitStream::new(&[], 2).iter(0);
        assert_eq!(iter.len(), 0);
        assert!(iter.next().is_none());
    }

    /// Packs `values` with `bits`-wide entries into big-endian bytes.
    fn pack(values: &[u32], bits: u32) -> Vec<u8> {
        let mut packer = Packer::new(bits);
        let mut bytes = Vec::new();
        for &value in values {
            if let Some(word) = packer.push(value) {
                bytes.extend_from_slice(&word.to_be_bytes());
            }
        }
        if let Some(word) = packer.finish() {
            bytes.extend_from_slice(&word.to_be_bytes());
        }
        bytes
    }
}
