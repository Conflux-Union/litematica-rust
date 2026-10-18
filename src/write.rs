//! Streaming, low-memory serialization of `.litematic` schematics.
//!
//! The writer is the reader's mirror image: blocks are pulled from a
//! declarative [`Layout`] in storage order and packed straight into the
//! gzip-compressed NBT stream, so memory stays O(palette) no matter how
//! large the schematic is. The palette must be known before block data is
//! emitted (the litematic bit width depends on it); [`Layout::palette`] may
//! over-approximate, and its default implementation discovers the palette
//! with one extra scan.

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::io::Write as IoWrite;
use std::time::{SystemTime, UNIX_EPOCH};

use self::encode::{empty_list, Encode};

use flate2::write::GzEncoder;
use flate2::Compression;

use crate::error::{Error, Result};
use crate::nbt::{
    Writer, TAG_COMPOUND, TAG_END, TAG_INT, TAG_LIST, TAG_LONG, TAG_LONG_ARRAY, TAG_STRING,
};
use crate::{mcdata::BlockState, region::num_bits};

/// Litematica format version written by this crate (`Version`).
const FORMAT_VERSION: i32 = 6;
/// Litematica format sub-version written by this crate (`SubVersion`).
const FORMAT_SUB_VERSION: i32 = 1;

// Palette
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// True for the plain `minecraft:air` state, the palette entry the
/// litematic format pins to index 0.
fn is_air(state: &BlockState) -> bool {
    state.name == "minecraft:air" && state.properties.is_empty()
}

/// The file-side block state palette of a written schematic.
///
/// Built from the states a [`crate::Layout`] declares: air entries collapse
/// into the format-mandated index 0, the rest are sorted and deduplicated so
/// that [`Palette::id_of`] can answer with a binary search.
pub(crate) struct Palette {
    /// Non-air states in file order; plain `minecraft:air` is implicit at
    /// index 0 and never stored.
    states: Vec<BlockState>,
    /// Bits per `BlockStates` entry for this palette length.
    bits: u32,
}

impl Palette {
    /// Builds the file palette from the states a layout declared.
    ///
    /// Over-approximations are fine: duplicate entries collapse, and unused
    /// entries simply travel along in the file. Air is never stored: the
    /// format pins plain `minecraft:air` to index 0, and the palette list is
    /// written with it first, as Litematica does.
    pub(crate) fn build(declared: impl IntoIterator<Item = BlockState>) -> Self {
        let mut states: Vec<BlockState> = declared.into_iter().filter(|s| !is_air(s)).collect();
        states.sort();
        states.dedup();
        let bits = num_bits(states.len() + 1);
        Self { states, bits }
    }

    /// The file palette index of `state`, or `None` if the declared palette
    /// does not cover it.
    pub(crate) fn id_of(&self, state: &BlockState) -> Option<u32> {
        if is_air(state) {
            return Some(0);
        }
        self.states.binary_search(state).ok().map(|i| i as u32 + 1)
    }

    /// The non-air states in file order; index 0 is implicit air.
    pub(crate) fn states(&self) -> &[BlockState] {
        &self.states
    }

    /// Number of entries in the written `BlockStatePalette`, air included.
    pub(crate) fn len(&self) -> usize {
        self.states.len() + 1
    }

    /// Bits per `BlockStates` entry, matching the reader's width rule.
    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }
}

impl Encode for Palette {
    const TAG: u8 = TAG_LIST;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.u8(TAG_COMPOUND)?;
        out.i32(self.len() as i32)?;
        air().body(out)?;
        for state in self.states() {
            state.body(out)?;
        }
        Ok(())
    }
}

/// The implicit air entry written at palette index 0.
fn air() -> BlockState {
    BlockState {
        name: Cow::Borrowed("minecraft:air"),
        properties: BTreeMap::new(),
    }
}

// Layout
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A declarative, query-based source of the blocks to serialize.
///
/// Instead of handing the writer a materialized volume, a layout answers
/// point queries: what block sits at `pos`. This suits procedural
/// generators (which can compute a block from its coordinates) and keeps
/// memory at O(palette) for arbitrarily large schematics.
pub trait Layout {
    /// Absolute dimensions per axis. Every position handed to
    /// [`Layout::block_at`] satisfies `pos[axis] < size()[axis]`.
    fn size(&self) -> [u32; 3];

    /// The block state at `pos`, or `None` for air.
    ///
    /// Every returned reference must be structurally equal (name and
    /// properties) to some entry of [`Layout::palette`]; a violation surfaces
    /// as [`Error::BlockNotInPalette`] during serialization.
    fn block_at(&self, pos: [u32; 3]) -> Option<&BlockState>;

    /// Every state [`Layout::block_at`] may return, as a set.
    ///
    /// Returning a set lets composite layouts merge their children's
    /// palettes with cheap unions instead of copying and deduplicating
    /// sequences. Over-approximation is allowed: unused entries simply
    /// travel along in the written file. Implementors that can derive their
    /// state set analytically should override this: the default discovers it
    /// with a full scan of the bounding box, which costs one extra pass over
    /// every voxel.
    fn palette(&self) -> HashSet<BlockState> {
        let [sx, sy, sz] = self.size();
        let mut seen = HashSet::new();
        for y in 0..sy {
            for z in 0..sz {
                for x in 0..sx {
                    if let Some(state) = self.block_at([x, y, z]) {
                        seen.insert(state.clone());
                    }
                }
            }
        }
        seen
    }
}

// Metadata
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// Options controlling the metadata and compression of a written schematic.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Schematic name; also used as the region name (`Metadata.Name`).
    pub name: String,
    /// Free-form description (`Metadata.Description`).
    pub description: String,
    /// Author (`Metadata.Author`).
    pub author: String,
    /// Target Minecraft data version (`MinecraftDataVersion`), e.g. 4189
    /// for 1.21.4. Not defaulted on purpose: the writer cannot know which
    /// game version the blocks are meant for.
    pub data_version: i32,
    /// Creation time in milliseconds since the Unix epoch
    /// (`Metadata.TimeCreated`).
    pub time_created: i64,
    /// Last-modified time in milliseconds since the Unix epoch
    /// (`Metadata.TimeModified`).
    pub time_modified: i64,
    /// Gzip compression level of the output stream.
    pub compression: Compression,
}

impl WriteOptions {
    /// Creates options for a schematic named `name` targeting Minecraft
    /// data version `data_version`. Timestamps default to now, description
    /// and author to empty, compression to the default level.
    pub fn new(data_version: i32, name: impl Into<String>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);
        Self {
            name: name.into(),
            description: String::new(),
            author: String::new(),
            data_version,
            time_created: now,
            time_modified: now,
            compression: Compression::default(),
        }
    }
}

// litematic
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// Serializes `layout` as a single-region `.litematic` file into `sink`.
///
/// The output is a complete gzip member that any litematic reader (this
/// crate, rustmatica, the Litematica mod) accepts; blocks are pulled from
/// the layout in storage order and streamed through a bounded packer, so
/// peak memory is the palette plus a few words, independent of the volume.
///
/// Layout bounds are validated before any byte is written. If the layout's
/// declared palette misses a state that [`Layout::block_at`] returns, the
/// error surfaces at that voxel with its coordinates.
///
/// # Example
///
/// ```no_run
/// # use litematica::{Layout, WriteOptions, write};
/// # struct MyLayout;
/// # impl Layout for MyLayout {
/// #     fn size(&self) -> [u32; 3] { [16, 16, 16] }
/// #     fn block_at(&self, pos: [u32; 3]) -> Option<&litematica::BlockState> { None }
/// # }
/// # fn main() -> Result<(), litematica::Error> {
/// let layout = MyLayout;
/// let options = WriteOptions::new(4189, "my build");
/// let file = std::fs::File::create("my_build.litematic")?;
/// write(&layout, &options, file)?;
/// # Ok(())
/// # }
/// ```
pub fn write<L: Layout, W: IoWrite>(layout: &L, options: &WriteOptions, sink: W) -> Result<()> {
    let size = layout.size();
    for &d in &size {
        if d > i32::MAX as u32 {
            return Err(Error::DimensionOverflow {
                what: "size axis",
                value: u128::from(d),
            });
        }
    }
    let volume = u128::from(size[0]) * u128::from(size[1]) * u128::from(size[2]);
    let volume_i32 = i32::try_from(volume).map_err(|_| Error::DimensionOverflow {
        what: "volume",
        value: volume,
    })?;
    let size_i32 = [size[0] as i32, size[1] as i32, size[2] as i32];

    let palette = Palette::build(layout.palette());
    let bits = palette.bits();
    // `volume <= i32::MAX` and `bits <= 32` bound this well below i32::MAX;
    // the checked conversion keeps the written length prefix honest.
    let longs = (volume * u128::from(bits)).div_ceil(64);
    let longs_i32 = i32::try_from(longs).map_err(|_| Error::DimensionOverflow {
        what: "BlockStates length",
        value: longs,
    })?;

    let total_blocks = Cell::new(0i64);
    let schematic = Schematic {
        layout,
        options,
        palette,
        size: size_i32,
        volume: volume_i32,
        longs: longs_i32,
        total_blocks: &total_blocks,
    };

    let mut gz = GzEncoder::new(sink, options.compression);
    let mut nbt = Writer::new(&mut gz);
    nbt.root()?;
    schematic.body(&mut nbt)?;
    nbt.into_inner()
        .map_err(|source| Error::WriteIo { source })?;
    gz.finish()
        .map_err(|source| Error::WriteIo { source })
        .map(|_| ())
}

/// The root litematic compound: one region and the metadata describing it.
struct Schematic<'a, L: Layout> {
    layout: &'a L,
    options: &'a WriteOptions,
    palette: Palette,
    size: [i32; 3],
    volume: i32,
    longs: i32,
    total_blocks: &'a Cell<i64>,
}

impl<L: Layout> Encode for Schematic<'_, L> {
    const TAG: u8 = TAG_COMPOUND;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.named(TAG_COMPOUND, "Regions")?;
        let region = RegionNbt {
            layout: self.layout,
            palette: &self.palette,
            size: self.size,
            longs: self.longs,
            total_blocks: self.total_blocks,
        };
        region.encode(&self.options.name, out)?;
        out.end()?; // Regions
        self.options
            .data_version
            .encode("MinecraftDataVersion", out)?;
        FORMAT_VERSION.encode("Version", out)?;
        FORMAT_SUB_VERSION.encode("SubVersion", out)?;
        let header = Header {
            options: self.options,
            size: self.size,
            volume: self.volume,
            total_blocks: self.total_blocks,
        };
        header.encode("Metadata", out)?;
        out.end() // root
    }
}

/// One region: its box, palette, block states, and the empty entity lists.
struct RegionNbt<'a, L: Layout> {
    layout: &'a L,
    palette: &'a Palette,
    size: [i32; 3],
    longs: i32,
    total_blocks: &'a Cell<i64>,
}

impl<L: Layout> Encode for RegionNbt<'_, L> {
    const TAG: u8 = TAG_COMPOUND;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        [0, 0, 0].encode("Position", out)?;
        self.size.encode("Size", out)?;
        self.palette.encode("BlockStatePalette", out)?;
        let states = BlockStates {
            layout: self.layout,
            palette: self.palette,
            longs: self.longs,
            total_blocks: self.total_blocks,
        };
        states.encode("BlockStates", out)?;
        // An absent `TileEntities` is rejected by rustmatica, so emit it.
        empty_list(out, "TileEntities")?;
        out.end()
    }
}

/// The packed block states of a region, pulled one long at a time.
struct BlockStates<'a, L: Layout> {
    layout: &'a L,
    palette: &'a Palette,
    longs: i32,
    total_blocks: &'a Cell<i64>,
}

impl<L: Layout> Encode for BlockStates<'_, L> {
    const TAG: u8 = TAG_LONG_ARRAY;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.i32(self.longs)?;
        let [sx, sy, sz] = self.layout.size();
        let sx = u64::from(sx);
        let sz = u64::from(sz);
        let mut packer = Packer::new(self.palette.bits());
        for index in 0..sx * u64::from(sy) * sz {
            let x = (index % sx) as u32;
            let rest = index / sx;
            let z = (rest % sz) as u32;
            let y = (rest / sz) as u32;
            let id = match self.layout.block_at([x, y, z]) {
                None => 0,
                Some(state) => {
                    let Some(id) = self.palette.id_of(state) else {
                        return Err(Error::BlockNotInPalette {
                            name: state.name.to_string(),
                            x,
                            y,
                            z,
                        });
                    };
                    if id != 0 {
                        self.total_blocks.set(self.total_blocks.get() + 1);
                    }
                    id
                }
            };
            if let Some(word) = packer.push(id) {
                out.i64(word as i64)?;
            }
        }
        if let Some(word) = packer.finish() {
            out.i64(word as i64)?;
        }
        Ok(())
    }
}

/// The metadata compound of the written schematic.
struct Header<'a> {
    options: &'a WriteOptions,
    size: [i32; 3],
    volume: i32,
    total_blocks: &'a Cell<i64>,
}

impl Encode for Header<'_> {
    const TAG: u8 = TAG_COMPOUND;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        let options = self.options;
        options.name.encode("Name", out)?;
        options.description.encode("Description", out)?;
        options.author.encode("Author", out)?;
        1i32.encode("RegionCount", out)?;
        self.volume.encode("TotalVolume", out)?;
        let total_blocks =
            i32::try_from(self.total_blocks.get()).map_err(|_| Error::DimensionOverflow {
                what: "TotalBlocks",
                value: self.total_blocks.get() as u128,
            })?;
        total_blocks.encode("TotalBlocks", out)?;
        options.time_created.encode("TimeCreated", out)?;
        options.time_modified.encode("TimeModified", out)?;
        // The mod's enclosing-size formula reports `max(axis, 1)`: a zero-size
        // axis still spans one block. Match it so our files' declared bounds
        // agree with what readers compute from the region.
        let enclosing = [
            self.size[0].max(1),
            self.size[1].max(1),
            self.size[2].max(1),
        ];
        enclosing.encode("EnclosingSize", out)?;
        out.end()
    }
}

// Encode
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The NBT leaf types and the trait that serializes them.
///
/// Scalars, vectors, and block states write themselves without owning a
/// schema node, so they live together here and leave the top level for the
/// document structure. The low-level sink stays in [`crate::nbt`], which is
/// also why this module is named `encode` rather than `nbt`.
mod encode {
    use super::{BlockState, IoWrite, Result, Writer};
    use super::{TAG_COMPOUND, TAG_END, TAG_INT, TAG_LIST, TAG_LONG, TAG_STRING};

    /// A value that writes itself as NBT under a given name.
    ///
    /// Each schema node owns its shape: scalars write a payload, compounds
    /// open their fields and close themselves, and lists declare their element
    /// type. The document then reads top-down instead of as one long sequence
    /// of tag/name/value calls.
    pub(super) trait Encode {
        /// The tag this value is written as.
        const TAG: u8;

        /// Writes this tag's payload, without a tag id or a name.
        fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()>;

        /// Writes this value as the named tag `name`.
        fn encode<W: IoWrite>(&self, name: &str, out: &mut Writer<W>) -> Result<()> {
            out.named(Self::TAG, name)?;
            self.body(out)
        }
    }

    impl Encode for i32 {
        const TAG: u8 = TAG_INT;

        fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
            out.i32(*self)
        }
    }

    impl Encode for i64 {
        const TAG: u8 = TAG_LONG;

        fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
            out.i64(*self)
        }
    }

    impl Encode for str {
        const TAG: u8 = TAG_STRING;

        fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
            out.string(self)
        }
    }

    impl Encode for String {
        const TAG: u8 = TAG_STRING;

        fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
            out.string(self)
        }
    }

    impl Encode for [i32; 3] {
        const TAG: u8 = TAG_COMPOUND;

        fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
            self[0].encode("x", out)?;
            self[1].encode("y", out)?;
            self[2].encode("z", out)?;
            out.end()
        }
    }

    impl Encode for BlockState {
        const TAG: u8 = TAG_COMPOUND;

        fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
            self.name.as_ref().encode("Name", out)?;
            if !self.properties.is_empty() {
                out.named(TAG_COMPOUND, "Properties")?;
                for (key, value) in &self.properties {
                    value.encode(key.as_ref(), out)?;
                }
                out.end()?;
            }
            out.end()
        }
    }

    /// Writes an empty named list, used for the entity lists the schema expects.
    pub(super) fn empty_list<W: IoWrite>(out: &mut Writer<W>, name: &str) -> Result<()> {
        out.named(TAG_LIST, name)?;
        out.u8(TAG_END)?;
        out.i32(0)
    }
}

// Packing
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// Packs palette indices into the `BlockStates` bit stream.
///
/// The mirror of [`crate::region::BlockIter`]: one continuous LSB-first
/// stream of `bits`-wide entries, big-endian longs, entries straddling long
/// boundaries. A long is returned the moment 64 bits have accumulated;
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
    use super::Palette;
    use crate::mcdata::BlockState;

    fn st(name: &str, props: &[(&str, &str)]) -> BlockState {
        BlockState {
            name: name.to_string().into(),
            properties: props
                .iter()
                .map(|(k, v)| (k.to_string().into(), v.to_string().into()))
                .collect(),
        }
    }

    fn air() -> BlockState {
        st("minecraft:air", &[])
    }

    #[test]
    fn packer_round_trips_width_3_through_block_iter() {
        let values = [1u32, 2, 6, 5, 7]; // 15 bits -> one partial long
        let mut packer = super::Packer::new(3);
        let mut longs = Vec::new();
        for &v in &values {
            if let Some(w) = packer.push(v) {
                longs.push(w);
            }
        }
        if let Some(w) = packer.finish() {
            longs.push(w);
        }
        assert_eq!(longs.len(), 1); // ceil(15 / 64)
        let bytes: Vec<u8> = longs.iter().flat_map(|w| w.to_be_bytes()).collect();
        let decoded: Vec<u32> =
            crate::region::BlockIter::new(&bytes, 3, values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn packer_round_trips_crossing_long_boundaries() {
        let values: Vec<u32> = (0..20).map(|i| (i * 7 + 3) % 32).collect(); // 100 bits
        let mut packer = super::Packer::new(5);
        let mut longs = Vec::new();
        for &v in &values {
            if let Some(w) = packer.push(v) {
                longs.push(w);
            }
        }
        if let Some(w) = packer.finish() {
            longs.push(w);
        }
        assert_eq!(longs.len(), 2); // ceil(100 / 64)
        let bytes: Vec<u8> = longs.iter().flat_map(|w| w.to_be_bytes()).collect();
        let decoded: Vec<u32> =
            crate::region::BlockIter::new(&bytes, 5, values.len() as u64).collect();
        assert_eq!(decoded, values);
    }

    #[test]
    fn packer_exact_multiple_flushes_no_tail() {
        let mut packer = super::Packer::new(2);
        let mut longs = Vec::new();
        for _ in 0..32 {
            if let Some(w) = packer.push(3) {
                longs.push(w);
            }
        }
        assert_eq!(longs.len(), 1); // 64 bits exactly
        assert!(packer.finish().is_none());
    }

    #[test]
    fn palette_build_maps_air_to_zero_and_sorts_rest() {
        let declared = [st("c", &[]), air(), st("a", &[]), st("c", &[])];
        let p = Palette::build(declared);
        assert_eq!(p.len(), 3);
        assert_eq!(p.states(), [st("a", &[]), st("c", &[])]);
        assert_eq!(p.bits(), 2);
        assert_eq!(p.id_of(&air()), Some(0));
        assert_eq!(p.id_of(&st("a", &[])), Some(1));
        assert_eq!(p.id_of(&st("c", &[])), Some(2));
    }

    #[test]
    fn palette_build_keeps_cave_air_as_regular_entry() {
        let declared = [st("minecraft:cave_air", &[]), air()];
        let p = Palette::build(declared);
        assert_eq!(p.len(), 2);
        assert_eq!(p.states(), [st("minecraft:cave_air", &[])]);
        assert_eq!(p.id_of(&st("minecraft:cave_air", &[])), Some(1));
    }

    #[test]
    fn palette_build_empty_declared_is_air_only() {
        let p = Palette::build(std::iter::empty());
        assert_eq!(p.len(), 1);
        assert!(p.states().is_empty());
        assert_eq!(p.bits(), 2);
    }

    #[test]
    fn palette_bits_match_litematica_widths() {
        for len in [1usize, 2, 3, 4, 5, 8, 9, 16, 17] {
            let declared: Vec<_> = (0..len)
                .map(|i| st("litematica:p", &[("i", &i.to_string())]))
                .collect();
            let p = Palette::build(declared);
            let expect = crate::region::num_bits(len + 1);
            assert_eq!(p.bits(), expect, "palette of {len} entries + air");
        }
    }

    #[test]
    fn id_of_unknown_state_is_none() {
        let p = Palette::build([st("minecraft:stone", &[])]);
        assert_eq!(p.id_of(&st("minecraft:dirt", &[])), None);
    }
}
