//! The packed `BlockStates` bit stream, read and written through one set of
//! abstractions.
//!
//! A litematic region stores palette identifiers as a continuous LSB-first bit
//! stream carried by big-endian longs: entry `i` starts at bit `i * width` and
//! may straddle two words. [`PaletteIdIter`] walks that stream, while the private
//! `BitStream` and `Packer` implement the two directions. Keeping both
//! directions here means the word layout is stated once, next to itself.

use crate::model::palette::{BitsPerEntry, PaletteId};
use std::sync::Arc;

// Unpacking
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The packed `BlockStates` bytes behind a [`BitStream`].
///
/// The random-access reader borrows the decompressed file; the streaming
/// reader owns the one region payload it is decoding. `Arc` keeps a clone of
/// the owning form cheap, so an iterator never copies the payload.
#[derive(Debug, Clone)]
pub(crate) enum Payload<'a> {
    /// Bytes borrowed from a buffer that outlives the region.
    Borrowed(&'a [u8]),
    /// Bytes owned by the region that decodes them.
    Owned(Arc<[u8]>),
}

impl<'a> Payload<'a> {
    /// Borrows a payload from a buffer the caller keeps alive.
    pub(crate) fn borrowed(bytes: &'a [u8]) -> Self {
        Self::Borrowed(bytes)
    }

    /// Takes ownership of a payload.
    pub(crate) fn owned(bytes: Vec<u8>) -> Self {
        Self::Owned(Arc::from(bytes))
    }

    /// Number of payload bytes.
    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Owned(bytes) => bytes,
        }
    }
}

impl AsRef<[u8]> for Payload<'_> {
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
    payload: Payload<'a>,
    width: BitsPerEntry,
}

impl<'a> BitStream<'a> {
    /// Views `payload` as entries of `width` bits.
    pub(crate) fn new(payload: Payload<'a>, width: BitsPerEntry) -> Self {
        Self { payload, width }
    }

    /// Iterates over `volume` entries in storage order.
    pub(crate) fn iter(self, volume: u64) -> PaletteIdIter<'a> {
        PaletteIdIter::new(self, volume)
    }

    /// Entry `index`, or `None` when the payload does not contain every bit
    /// of it. A partial trailing entry is absent, never zero-filled.
    pub(crate) fn entry(&self, index: u64) -> Option<PaletteId> {
        let bit = u128::from(index) * u128::from(self.width.bits());
        if bit + u128::from(self.width.bits()) > self.byte_len() as u128 * 8 {
            return None;
        }
        let long = usize::try_from(bit / 64).ok()?;
        let shift = (bit % 64) as u32;
        let first = self.long(long)?;
        let mut source = Longs {
            bytes: self.payload.as_slice(),
            next: long + 1,
        };
        let mut cursor = BitCursor::with_remainder(first >> shift, 64 - shift);
        Some(cursor.next(self.width, &mut source))
    }

    /// The `index`th long, or `None` when the payload ends before it.
    fn long(&self, index: usize) -> Option<u64> {
        let start = index.checked_mul(8)?;
        let word = self.payload.as_slice().get(start..start + 8)?;
        Some(u64::from_be_bytes(word.try_into().ok()?))
    }

    /// Number of payload bytes this stream windows, longs only.
    pub(crate) fn byte_len(&self) -> usize {
        self.payload.len() / 8 * 8
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
    pub(crate) fn next(&mut self, width: BitsPerEntry, source: &mut impl LongSource) -> PaletteId {
        let (bits, mask) = (width.bits(), width.mask());
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
pub struct PaletteIdIter<'a> {
    cursor: BitCursor,
    source: Longs<Payload<'a>>,
    width: BitsPerEntry,
    remaining: u64,
}

impl<'a> PaletteIdIter<'a> {
    fn new(stream: BitStream<'a>, volume: u64) -> Self {
        Self {
            cursor: BitCursor::new(),
            source: Longs {
                bytes: stream.payload,
                next: 0,
            },
            width: stream.width,
            remaining: volume,
        }
    }
}

impl Iterator for PaletteIdIter<'_> {
    type Item = PaletteId;

    #[inline]
    fn next(&mut self) -> Option<PaletteId> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        Some(self.cursor.next(self.width, &mut self.source))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match usize::try_from(self.remaining) {
            Ok(n) => (n, Some(n)),
            Err(_) => (usize::MAX, None),
        }
    }
}

// Packing
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// Packs palette indices into the `BlockStates` bit stream.
///
/// The mirror of [`PaletteIdIter`]: one continuous LSB-first stream of
/// `bits`-wide entries, big-endian longs, entries straddling long boundaries.
/// A long is returned the moment 64 bits have accumulated;
/// [`Packer::finish`] flushes the trailing partial long, zero-padded.
pub(crate) struct Packer {
    acc: u128,
    have: u32,
    width: BitsPerEntry,
}

impl Packer {
    /// Creates a packer emitting entries of `width` bits.
    pub(crate) fn new(width: BitsPerEntry) -> Self {
        Self {
            acc: 0,
            have: 0,
            width,
        }
    }

    /// Packs one palette index, returning a completed long when one fills.
    ///
    /// The identifier must fit this packer's width; the writer obtains it from
    /// the same palette that supplied the width.
    #[inline]
    pub(crate) fn push(&mut self, id: u32) -> Option<u64> {
        debug_assert!(u64::from(id) <= self.width.mask(), "id {id} does not fit");
        self.acc |= u128::from(id) << self.have;
        self.have += self.width.bits();
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
    use super::{BitStream, Packer, PaletteIdIter, Payload};
    use crate::model::palette::{BitsPerEntry, PaletteId};

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
        let width = BitsPerEntry::of_len(values.len());
        let iter = PaletteIdIter::new(
            BitStream::new(Payload::borrowed(&bytes), width),
            values.len() as u64,
        );
        assert_eq!(iter.size_hint(), (values.len(), Some(values.len())));
        let decoded: Vec<u32> = iter.collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn iter_entries_crossing_long_boundaries() {
        // 5 bits per entry, 20 entries -> 100 bits, crossing 64-bit borders.
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let width = BitsPerEntry::of_len(values.len());
        let bytes = pack(&values, width);
        let decoded: Vec<PaletteId> = PaletteIdIter::new(
            BitStream::new(Payload::borrowed(&bytes), width),
            values.len() as u64,
        )
        .collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn random_entries_match_the_sequential_stream() {
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect();
        let width = BitsPerEntry::of_len(values.len());
        let bytes = pack(&values, width);
        let stream = BitStream::new(Payload::borrowed(&bytes), width);
        for (i, &want) in values.iter().enumerate() {
            assert_eq!(stream.entry(i as u64), Some(want), "entry {i}");
        }
        assert_eq!(stream.entry(100), None);
    }

    #[test]
    fn entry_rejects_a_partial_trailing_entry() {
        let width = BitsPerEntry::of_len(1000); // 10 bits per entry
        let word = [0u8; 8];
        let stream = BitStream::new(Payload::borrowed(&word), width);
        assert_eq!(stream.entry(5), Some(0), "entry 5 ends at bit 60");
        assert_eq!(stream.entry(6), None, "entry 6 needs bits past the payload");
    }

    #[test]
    fn packer_exact_multiple_flushes_no_tail() {
        let mut packer = Packer::new(BitsPerEntry::of_len(3));
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
        let mut iter = BitStream::new(Payload::borrowed(&[]), BitsPerEntry::of_len(0)).iter(0);
        assert_eq!(iter.size_hint().0, 0);
        assert!(iter.next().is_none());
    }

    /// Packs `values` with entries of `width` bits into big-endian bytes.
    fn pack(values: &[u32], width: BitsPerEntry) -> Vec<u8> {
        let mut packer = Packer::new(width);
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
