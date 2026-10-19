//! The bridge between block states and the compact identifiers a litematic
//! stores.
//!
//! A schematic has two readings of the same data: as a file it names blocks,
//! and as storage it carries small integers. A [`Palette`] is the join between
//! them, and its order is the authority — every packed identifier is an index
//! into it.
//!
//! The two readings are not symmetric, so they get two types. A file palette
//! preserves the order it was read in and only translates identifiers into
//! states. An [`OrderedPalette`] is a file palette that additionally promises a
//! canonical order, which makes the reverse lookup possible. Keeping the
//! promise in the type means a lookup cannot be attempted on an order that
//! was never sorted.
//!
//! This module owns the format's naming conventions: plain air occupies
//! identifier zero in a written palette, and the bit width an identifier needs
//! is derived from the palette length. Neither rule belongs to Minecraft's
//! vocabulary; both belong to the file format.

use crate::knowledge::mc::BlockState;
use std::ops::Deref;

// Palette & Ordered palette
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The identifier a palette assigns to one block state: the integer packed
/// into a region's `BlockStates` stream.
///
/// Identifiers are meaningless outside the palette that issued them, so they
/// travel with a [`Palette`] rather than as bare integers.
pub type PaletteId = u32;

/// The block states a region stores, in the order its packed identifiers refer
/// to.
///
/// A palette is a reading of an existing file: the order is part of the data,
/// so it is never sorted or deduplicated. Lookups therefore run one way only,
/// from identifier to state. A palette that can also answer the reverse
/// direction is an [`OrderedPalette`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
    states: Vec<BlockState>,
}

impl Palette {
    /// Wraps the states read from a file, in file order.
    ///
    /// The order is part of the data: every decoded identifier indexes this
    /// slice, so it must never be sorted or deduplicated.
    pub fn from_file(states: Vec<BlockState>) -> Self {
        Self { states }
    }

    /// The states in palette order; identifiers index this slice.
    pub fn states(&self) -> &[BlockState] {
        &self.states
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the palette has no entries at all.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Bits per packed identifier, matching the litematic width rule: the
    /// smallest `n >= 2` with `2^n >= len`.
    pub fn bits_per_entry(&self) -> u32 {
        self.width().bits()
    }

    /// The width and mask the file format assigns to this palette.
    pub(crate) fn width(&self) -> BitsPerEntry {
        BitsPerEntry::of_len(self.len())
    }

    /// The state with identifier `id`, or `None` when the palette is shorter.
    pub fn state_at(&self, id: PaletteId) -> Option<&BlockState> {
        self.states.get(id as usize)
    }
}

/// A palette whose order is canonical: air first, then the remaining states
/// sorted and deduplicated.
///
/// This is the palette a writer emits. The canonical order makes reverse
/// lookup a binary search, and gives a byte-for-byte deterministic file for a
/// given set of states. Over-approximating the declared set is harmless:
/// unused entries simply travel along.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedPalette(Palette);

impl OrderedPalette {
    /// Builds the palette a writer emits: air at 0, the remaining states
    /// sorted and deduplicated.
    pub fn new(declared: impl IntoIterator<Item = BlockState>) -> Self {
        let mut states: Vec<BlockState> = declared
            .into_iter()
            .filter(|state| state != &BlockState::AIR)
            .collect();
        states.sort();
        states.dedup();
        states.insert(0, BlockState::AIR);
        Self(Palette::from_file(states))
    }

    /// The identifier of `state`, or `None` when it is not covered.
    ///
    /// The search relies on the canonical order this type promises, so it is
    /// a binary search rather than a scan.
    pub fn id_of(&self, state: &BlockState) -> Option<PaletteId> {
        if state == &BlockState::AIR {
            return Some(0);
        }
        let rest = self.states().get(1..)?;
        rest.binary_search(state)
            .ok()
            .map(|index| index as PaletteId + 1)
    }
}

/// An ordered palette is readable as a plain palette: every lookup that only
/// follows the file order is available without a conversion.
impl Deref for OrderedPalette {
    type Target = Palette;

    fn deref(&self) -> &Palette {
        &self.0
    }
}

/// The packed width of one palette identifier, in bits, together with the mask
/// that isolates it.
///
/// A litematic entry is never narrower than two bits, and is only as wide as
/// the smallest power of two that can address every entry of its palette. The
/// two values are constructed together, so a decoder can never read a stream
/// with one width and mask it with another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BitsPerEntry {
    bits: u32,
    mask: u64,
}

impl BitsPerEntry {
    /// The width the litematic format assigns to a palette of `len` entries.
    pub(crate) fn of_len(len: usize) -> Self {
        let bits = (2..32).find(|n| (1u64 << n) >= len as u64).unwrap_or(32);
        Self {
            bits,
            mask: (1u64 << bits) - 1,
        }
    }

    /// The number of bits one entry occupies.
    pub(crate) const fn bits(self) -> u32 {
        self.bits
    }

    /// The mask that isolates one entry.
    pub(crate) const fn mask(self) -> u64 {
        self.mask
    }

    /// How many distinct identifiers this width can carry.
    pub(crate) const fn capacity(self) -> u64 {
        self.mask + 1
    }
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::{OrderedPalette, Palette};
    use crate::knowledge::mc::BlockState;

    fn st(name: &str) -> BlockState {
        BlockState {
            name: name.to_string().into(),
            properties: Default::default(),
        }
    }

    #[test]
    fn ordered_palette_pins_air_to_zero_and_sorts_rest() {
        let declared = [st("c"), st("minecraft:air"), st("a"), st("c")];
        let palette = OrderedPalette::new(declared);
        let air = st("minecraft:air");
        assert_eq!(palette.len(), 3);
        assert_eq!(palette.states(), [air.clone(), st("a"), st("c")]);
        assert_eq!(palette.bits_per_entry(), 2);
        assert_eq!(palette.id_of(&air), Some(0));
        assert_eq!(palette.id_of(&st("a")), Some(1));
        assert_eq!(palette.id_of(&st("c")), Some(2));
        assert_eq!(palette.id_of(&st("minecraft:dirt")), None);
    }

    #[test]
    fn ordered_palette_keeps_cave_air_as_regular_entry() {
        let declared = [st("minecraft:cave_air"), st("minecraft:air")];
        let palette = OrderedPalette::new(declared);
        assert_eq!(palette.len(), 2);
        assert_eq!(
            palette.states(),
            [st("minecraft:air"), st("minecraft:cave_air")]
        );
        assert_eq!(palette.id_of(&st("minecraft:cave_air")), Some(1));
    }

    #[test]
    fn empty_ordered_palette_is_air_only() {
        let palette = OrderedPalette::new(std::iter::empty());
        assert_eq!(palette.len(), 1);
        assert_eq!(palette.states(), [st("minecraft:air")]);
        assert_eq!(palette.bits_per_entry(), 2);
    }

    #[test]
    fn file_palette_keeps_order_and_only_decodes() {
        let palette = Palette::from_file(vec![st("b"), st("minecraft:air"), st("a")]);
        assert_eq!(palette.states(), [st("b"), st("minecraft:air"), st("a")]);
        assert_eq!(palette.state_at(2), Some(&st("a")));
        assert_eq!(palette.bits_per_entry(), 2);
    }

    #[test]
    fn width_starts_at_two_bits_and_grows_at_powers_of_two() {
        let width = |len| Palette::from_file(vec![st("x"); len]).bits_per_entry();
        assert_eq!(width(0), 2);
        assert_eq!(width(4), 2);
        assert_eq!(width(5), 3);
        assert_eq!(width(8), 3);
        assert_eq!(width(9), 4);
    }
}
