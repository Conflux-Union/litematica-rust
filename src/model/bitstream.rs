//! The packed `BlockStates` bit stream, read and written through one set of
//! abstractions.
//!
//! A litematic region stores palette identifiers as a continuous LSB-first bit
//! stream carried by big-endian longs: entry `i` starts at bit `i * width` and
//! may straddle two words. [`BlockIter`] walks that stream, while the private
//! `BitStream` and `Packer` implement the two directions. Keeping both
//! directions here means the word layout is stated once, next to itself.

use std::sync::Arc;

use crate::model::palette::PaletteId;

// Unpacking
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The bytes behind a [`BitStream`].
///
/// The random-access reader borrows the decompressed file; the streaming
/// reader owns the one region payload it is decoding. `Arc` keeps a clone of
/// the owning form cheap, so an iterator never copies the payload.
#[derive(Debug, Clone)]
enum Bytes<'a> {
    Borrowed(&'a [u8]),
    Owned(Arc<[u8]>),
}

impl Bytes<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }
}

impl AsRef<[u8]> for Bytes<'_> {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// The packed `BlockStates` payload viewed as fixed-width entries.
///
/// The payload is a run of big-endian longs, but entries are not aligned to
/// them: entry `i` starts at bit `i * bits` of one continuous stream, least
/// significant bit first, and may straddle two longs. [`BitCursor`] performs
/// that decode for both sequential and random access, so the layout of the
/// stream is stated once.
#[derive(Debug, Clone)]
pub(crate) struct BitStream<'a> {
    bytes: Bytes<'a>,
    bits: u32,
    mask: u64,
}

impl<'a> BitStream<'a> {
    /// Views `bytes` borrowed from the file as `bits`-wide entries.
    pub(crate) fn borrowed(bytes: &'a [u8], bits: u32) -> Self {
        Self::with_bytes(Bytes::Borrowed(bytes), bits)
    }

    /// Takes ownership of a region payload and views it as `bits`-wide
    /// entries.
    pub(crate) fn owned(bytes: Vec<u8>, bits: u32) -> Self {
        Self::with_bytes(Bytes::Owned(Arc::from(bytes)), bits)
    }

    /// Sets the width once the palette that defines it has been read.
    pub(crate) fn with_bits(mut self, bits: u32) -> Self {
        self.bits = bits;
        self.mask = (1u64 << bits) - 1;
        self
    }

    fn with_bytes(bytes: Bytes<'a>, bits: u32) -> Self {
        Self {
            bytes,
            bits,
            mask: (1u64 << bits) - 1,
        }
    }

    /// Iterates over `volume` entries in storage order.
    pub(crate) fn iter(self, volume: u64) -> BlockIter<'a> {
        BlockIter::new(self, volume)
    }

    /// Entry `index`, or `None` when the payload does not reach it.
    pub(crate) fn entry(&self, index: u64) -> Option<PaletteId> {
        let bit = u128::from(index) * u128::from(self.bits);
        let long = usize::try_from(bit / 64).ok()?;
        let shift = (bit % 64) as u32;
        let first = self.long(long)?;
        let mut source = Longs {
            bytes: self.bytes.as_slice(),
            next: long + 1,
        };
        let mut cursor = BitCursor::with_remainder(first >> shift, 64 - shift);
        Some(cursor.next(self.bits, self.mask, &mut source))
    }

    /// The `index`th long, zero-filled when the payload is shorter.
    fn long(&self, index: usize) -> Option<u64> {
        let start = index.checked_mul(8)?;
        let word = self.bytes.as_slice().get(start..start + 8)?;
        Some(u64::from_be_bytes(word.try_into().ok()?))
    }

    /// Number of payload bytes this stream windows, longs only.
    pub(crate) fn byte_len(&self) -> usize {
        self.bytes.as_slice().len() / 8 * 8
    }
}

/// A source of 64-bit words for [`BitCursor`].
///
/// The slice-backed random reader and the streaming reader supply words from
/// different places, but decode entries through the same cursor.
pub(crate) trait LongSource {
    /// The next word, least significant byte of the stream first.
    fn next_long(&mut self) -> u64;
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
    pub(crate) fn next(&mut self, bits: u32, mask: u64, source: &mut impl LongSource) -> PaletteId {
        if self.cur_bits >= bits {
            // Fast path: enough buffered bits for this entry.
            let value = (self.cur & mask) as PaletteId;
            self.cur >>= bits;
            self.cur_bits -= bits;
            return value;
        }
        // Slow path: the entry straddles the word boundary, low bits from
        // the buffer, the rest from the next word.
        let low = self.cur_bits;
        let mut value = self.cur & mask;
        self.cur = source.next_long();
        self.cur_bits = 64;
        value |= (self.cur & (mask >> low)) << low;
        self.cur >>= bits - low;
        self.cur_bits -= bits - low;
        value as PaletteId
    }
}

/// A [`LongSource`] over any byte container: a borrow for random access, an
/// owned payload for sequential iteration.
#[derive(Debug)]
struct Longs<B> {
    bytes: B,
    next: usize,
}

impl<B: AsRef<[u8]>> LongSource for Longs<B> {
    fn next_long(&mut self) -> u64 {
        // A payload shorter than the declared volume decodes missing words as
        // zero; `entry` reports the truncation separately.
        let start = self.next.saturating_mul(8);
        self.next += 1;
        self.bytes
            .as_ref()
            .get(start..start.saturating_add(8))
            .and_then(|word| word.try_into().ok())
            .map(u64::from_be_bytes)
            .unwrap_or(0)
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
    source: Longs<Bytes<'a>>,
    bits: u32,
    mask: u64,
    remaining: u64,
}

impl<'a> BlockIter<'a> {
    fn new(stream: BitStream<'a>, volume: u64) -> Self {
        Self {
            cursor: BitCursor::new(),
            source: Longs {
                bytes: stream.bytes,
                next: 0,
            },
            bits: stream.bits,
            mask: stream.mask,
            remaining: volume,
        }
    }
}

impl Iterator for BlockIter<'_> {
    type Item = PaletteId;

    #[inline]
    fn next(&mut self) -> Option<PaletteId> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.cursor.next(self.bits, self.mask, &mut self.source))
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
    use super::{BitStream, BlockIter, Packer};
    use crate::model::palette::PaletteId;

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
        let iter = BlockIter::new(BitStream::borrowed(&bytes, 3), values.len() as u64);
        assert_eq!(iter.len(), values.len());
        let decoded: Vec<u32> = iter.collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn iter_entries_crossing_long_boundaries() {
        // 5 bits per entry, 20 entries -> 100 bits, crossing 64-bit borders.
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let bytes = pack(&values, 5);
        let decoded: Vec<PaletteId> =
            BlockIter::new(BitStream::borrowed(&bytes, 5), values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn random_entries_match_the_sequential_stream() {
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let bytes = pack(&values, 5);
        let stream = BitStream::borrowed(&bytes, 5);
        for (i, &want) in values.iter().enumerate() {
            assert_eq!(stream.entry(i as u64), Some(want), "entry {i}");
        }
        assert_eq!(stream.entry(100), None);
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
        let mut iter = BitStream::borrowed(&[], 2).iter(0);
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
