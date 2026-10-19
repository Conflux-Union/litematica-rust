//! The bridge between block states and the compact identifiers a litematic
//! stores.
//!
//! A schematic has two readings of the same data: as a file it names blocks,
//! and as storage it carries small integers. A [`Palette`] is the join between
//! them, and its order is the authority — every packed identifier is an index
//! into it. The two directions are not symmetric: decoding always follows the
//! file's order, while encoding needs a canonical order that can be searched.
//! The constructor that builds a palette decides which of the two readings it
//! supports.
//!
//! This module owns the format's naming conventions: plain air occupies
//! identifier zero in a written palette, and the bit width an identifier needs
//! is derived from the palette length. Neither rule belongs to Minecraft's
//! vocabulary; both belong to the file format.

use crate::knowledge::mc::BlockState;

// Palette
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The identifier a palette assigns to one block state: the integer packed
/// into a region's `BlockStates` stream.
///
/// Identifiers are meaningless outside the palette that issued them, so they
/// travel with a [`Palette`] rather than as bare integers.
pub type PaletteId = u32;

/// The block states a region stores, in the order its packed identifiers refer
/// to, together with the bit width that order implies.
///
/// The format has two orderings, and one type serves both: a file palette
/// preserves the order of the file so decoded identifiers keep their meaning,
/// while a palette being written pins plain air to id 0 and sorts the rest so
/// an identifier can be resolved by binary search. The constructor that
/// produced the palette decides which reading applies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Palette {
    states: Vec<BlockState>,
    bits: u32,
    /// Whether the states after index 0 are sorted, and therefore whether a
    /// state can be resolved back to an identifier.
    searchable: bool,
}

impl Palette {
    /// Wraps the states read from a file, in file order.
    ///
    /// The order is part of the data: every decoded identifier indexes this
    /// slice, so it must never be sorted or deduplicated. A file palette is
    /// translated identifier to state, never the other way.
    pub fn from_file(states: Vec<BlockState>) -> Self {
        Self::with_states(states, false)
    }

    /// Builds the palette a writer emits: air at 0, the remaining states
    /// sorted and deduplicated.
    ///
    /// Filtering air first and sorting the rest makes identifier resolution a
    /// binary search, and gives a byte-for-byte deterministic file for a given
    /// set of states. Over-approximating the declared set is harmless: unused
    /// entries simply travel along.
    pub fn from_layout(declared: impl IntoIterator<Item = BlockState>) -> Self {
        let mut states: Vec<BlockState> = declared
            .into_iter()
            .filter(|state| state != &BlockState::AIR)
            .collect();
        states.sort();
        states.dedup();
        states.insert(0, BlockState::AIR);
        Self::with_states(states, true)
    }

    /// The states in palette order; identifiers index this slice.
    pub fn states(&self) -> &[BlockState] {
        &self.states
    }

    /// Number of entries, air included when the palette is a written one.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether the palette has no entries at all.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Bits per packed identifier, matching the litematic width rule: the
    /// smallest `n >= 2` with `2^n >= len`.
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// The state with identifier `id`, or `None` when the palette is shorter.
    pub fn state_at(&self, id: PaletteId) -> Option<&BlockState> {
        self.states.get(id as usize)
    }

    /// The identifier of `state` in a palette built by
    /// [`Palette::from_layout`], or `None` when it is not covered.
    ///
    /// A file palette answers `None` for every state: its order serves
    /// decoding, and identifiers are resolved by the other direction.
    pub fn id_of(&self, state: &BlockState) -> Option<PaletteId> {
        if !self.searchable {
            return None;
        }
        if state == &BlockState::AIR {
            return Some(0);
        }
        let rest = self.states.get(1..)?;
        rest.binary_search(state)
            .ok()
            .map(|index| index as PaletteId + 1)
    }

    /// The one place the width rule is stated.
    fn with_states(states: Vec<BlockState>, searchable: bool) -> Self {
        let bits = (2..32)
            .find(|n| (1u64 << n) >= states.len() as u64)
            .unwrap_or(32);
        Self {
            states,
            bits,
            searchable,
        }
    }
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::Palette;
    use crate::knowledge::mc::BlockState;

    fn st(name: &str) -> BlockState {
        BlockState {
            name: name.to_string().into(),
            properties: Default::default(),
        }
    }

    #[test]
    fn layout_palette_pins_air_to_zero_and_sorts_rest() {
        let declared = [st("c"), st("minecraft:air"), st("a"), st("c")];
        let palette = Palette::from_layout(declared);
        let air = st("minecraft:air");
        assert_eq!(palette.len(), 3);
        assert_eq!(palette.states(), [air.clone(), st("a"), st("c")]);
        assert_eq!(palette.bits(), 2);
        assert_eq!(palette.id_of(&air), Some(0));
        assert_eq!(palette.id_of(&st("a")), Some(1));
        assert_eq!(palette.id_of(&st("c")), Some(2));
        assert_eq!(palette.id_of(&st("minecraft:dirt")), None);
    }

    #[test]
    fn layout_palette_keeps_cave_air_as_regular_entry() {
        let declared = [st("minecraft:cave_air"), st("minecraft:air")];
        let palette = Palette::from_layout(declared);
        assert_eq!(palette.len(), 2);
        assert_eq!(
            palette.states(),
            [st("minecraft:air"), st("minecraft:cave_air")]
        );
        assert_eq!(palette.id_of(&st("minecraft:cave_air")), Some(1));
    }

    #[test]
    fn empty_layout_palette_is_air_only() {
        let palette = Palette::from_layout(std::iter::empty());
        assert_eq!(palette.len(), 1);
        assert_eq!(palette.states(), [st("minecraft:air")]);
        assert_eq!(palette.bits(), 2);
    }

    #[test]
    fn file_palette_keeps_order_and_is_not_searchable() {
        let palette = Palette::from_file(vec![st("b"), st("minecraft:air"), st("a")]);
        assert_eq!(palette.states(), [st("b"), st("minecraft:air"), st("a")]);
        assert_eq!(palette.id_of(&st("b")), None);
        assert_eq!(palette.state_at(2), Some(&st("a")));
    }
}
