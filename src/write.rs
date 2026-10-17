//! Streaming, low-memory serialization of `.litematic` schematics.
//!
//! The writer is the reader's mirror image: blocks are pulled from a
//! declarative [`Layout`] in storage order and packed straight into the
//! gzip-compressed NBT stream, so memory stays O(palette) no matter how
//! large the schematic is. The palette must be known before block data is
//! emitted (the litematic bit width depends on it); [`Layout::palette`] may
//! over-approximate, and its default implementation discovers the palette
//! with one extra scan.

use std::collections::HashSet;
use std::io::Write as IoWrite;
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::write::GzEncoder;
use flate2::Compression;

use crate::error::{Error, Result};
use crate::nbt::{
    Writer, TAG_COMPOUND, TAG_END, TAG_INT, TAG_LIST, TAG_LONG, TAG_LONG_ARRAY, TAG_STRING,
};
use crate::region::{num_bits, BlockState};

/// Litematica format version written by this crate (`Version`).
const FORMAT_VERSION: i32 = 6;
/// Litematica format sub-version written by this crate (`SubVersion`).
const FORMAT_SUB_VERSION: i32 = 1;

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
    /// The full file palette; `[0]` is always plain `minecraft:air`.
    states: Vec<BlockState>,
    /// Bits per `BlockStates` entry for this palette length.
    bits: u32,
}

impl Palette {
    /// Builds the file palette from the states a layout declared.
    ///
    /// Over-approximations are fine: duplicate entries collapse, and unused
    /// entries simply travel along in the file. The palette always contains
    /// at least air.
    pub(crate) fn build(declared: &[BlockState]) -> Self {
        let mut rest: Vec<BlockState> = declared.iter().filter(|s| !is_air(s)).cloned().collect();
        rest.sort();
        rest.dedup();
        let mut states = Vec::with_capacity(rest.len() + 1);
        states.push(BlockState {
            name: "minecraft:air".to_string(),
            properties: Vec::new(),
        });
        states.extend(rest);
        let bits = num_bits(states.len());
        Self { states, bits }
    }

    /// The file palette in file order; entry 0 is air.
    pub(crate) fn states(&self) -> &[BlockState] {
        &self.states
    }

    /// Number of palette entries.
    pub(crate) fn len(&self) -> usize {
        self.states.len()
    }

    /// Bits per `BlockStates` entry, matching the reader's width rule.
    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }

    /// The file palette index of `state`, or `None` if the declared palette
    /// does not cover it.
    pub(crate) fn id_of(&self, state: &BlockState) -> Option<u32> {
        if is_air(state) {
            return Some(0);
        }
        self.states[1..]
            .binary_search(state)
            .ok()
            .map(|i| i as u32 + 1)
    }
}

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
    /// properties, including property order) to some entry of
    /// [`Layout::palette`]; a violation surfaces as
    /// [`Error::BlockNotInPalette`] during serialization.
    fn block_at(&self, pos: [u32; 3]) -> Option<&BlockState>;

    /// Every state [`Layout::block_at`] may return, as a whole.
    ///
    /// Over-approximation is allowed and duplicates are fine; unused
    /// entries simply travel along in the written file. Implementors that
    /// can derive their state set analytically should override this: the
    /// default discovers it with a full scan of the bounding box, which
    /// costs one extra pass over every voxel.
    fn palette(&self) -> Vec<BlockState> {
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
        seen.into_iter().collect()
    }
}

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

    let declared = layout.palette();
    let palette = Palette::build(&declared);
    let bits = palette.bits();
    // `volume <= i32::MAX` and `bits <= 32` bound this well below i32::MAX;
    // the checked conversion keeps the written length prefix honest.
    let longs = (volume * u128::from(bits)).div_ceil(64);
    let longs_i32 = i32::try_from(longs).map_err(|_| Error::DimensionOverflow {
        what: "BlockStates length",
        value: longs,
    })?;

    let mut gz = GzEncoder::new(sink, options.compression);
    stream(
        layout, options, &palette, size_i32, volume_i32, longs_i32, &mut gz,
    )?;
    gz.finish()
        .map_err(|source| Error::WriteIo { source })
        .map(|_| ())
}

/// Emits the full NBT document into the gzip encoder.
///
/// Field order follows the reader's tolerance rather than the mod's output:
/// `Regions` first, `Metadata` last, so the non-air block count can be
/// accumulated while the `BlockStates` stream is written instead of
/// requiring a second pass over the layout.
fn stream<L: Layout, W: IoWrite>(
    layout: &L,
    options: &WriteOptions,
    palette: &Palette,
    size: [i32; 3],
    volume: i32,
    longs: i32,
    gz: &mut GzEncoder<W>,
) -> Result<()> {
    let mut nbt = Writer::new(gz);

    nbt.root()?;
    nbt.named(TAG_COMPOUND, "Regions")?;
    nbt.named(TAG_COMPOUND, &options.name)?;

    vec3(&mut nbt, "Position", [0, 0, 0])?;
    vec3(&mut nbt, "Size", size)?;

    nbt.named(TAG_LIST, "BlockStatePalette")?;
    nbt.u8(TAG_COMPOUND)?;
    nbt.i32(palette.len() as i32)?;
    for state in palette.states() {
        nbt.u8(TAG_STRING)?;
        nbt.string("Name")?;
        nbt.string(&state.name)?;
        if !state.properties.is_empty() {
            nbt.u8(TAG_COMPOUND)?;
            nbt.string("Properties")?;
            for (key, value) in &state.properties {
                nbt.u8(TAG_STRING)?;
                nbt.string(key)?;
                nbt.string(value)?;
            }
            nbt.end()?;
        }
        nbt.end()?;
    }

    nbt.named(TAG_LONG_ARRAY, "BlockStates")?;
    nbt.i32(longs)?;
    let mut non_air: u64 = 0;
    let mut packer = Packer::new(palette.bits());
    let [sx, sy, sz] = [size[0] as u32, size[1] as u32, size[2] as u32];
    for y in 0..sy {
        for z in 0..sz {
            for x in 0..sx {
                let id = match layout.block_at([x, y, z]) {
                    None => 0,
                    Some(state) => {
                        let Some(id) = palette.id_of(state) else {
                            return Err(Error::BlockNotInPalette {
                                name: state.name.clone(),
                                x,
                                y,
                                z,
                            });
                        };
                        if id != 0 {
                            non_air += 1;
                        }
                        id
                    }
                };
                if let Some(word) = packer.push(id) {
                    nbt.i64(word as i64)?;
                }
            }
        }
    }
    if let Some(word) = packer.finish() {
        nbt.i64(word as i64)?;
    }

    // Empty lists the schema treats as required readers-side; an absent
    // `TileEntities` is rejected by rustmatica, so emit it explicitly.
    nbt.named(TAG_LIST, "TileEntities")?;
    nbt.u8(TAG_END)?;
    nbt.i32(0)?;

    nbt.end()?; // region compound
    nbt.end()?; // Regions compound

    nbt.named(TAG_INT, "MinecraftDataVersion")?;
    nbt.i32(options.data_version)?;
    nbt.named(TAG_INT, "Version")?;
    nbt.i32(FORMAT_VERSION)?;
    nbt.named(TAG_INT, "SubVersion")?;
    nbt.i32(FORMAT_SUB_VERSION)?;

    nbt.named(TAG_COMPOUND, "Metadata")?;
    nbt.named(TAG_STRING, "Name")?;
    nbt.string(&options.name)?;
    nbt.named(TAG_STRING, "Description")?;
    nbt.string(&options.description)?;
    nbt.named(TAG_STRING, "Author")?;
    nbt.string(&options.author)?;
    nbt.named(TAG_INT, "RegionCount")?;
    nbt.i32(1)?;
    nbt.named(TAG_INT, "TotalVolume")?;
    nbt.i32(volume)?;
    nbt.named(TAG_INT, "TotalBlocks")?;
    nbt.i32(non_air as i32)?;
    nbt.named(TAG_LONG, "TimeCreated")?;
    nbt.i64(options.time_created)?;
    nbt.named(TAG_LONG, "TimeModified")?;
    nbt.i64(options.time_modified)?;
    // The mod's enclosing-size formula reports `max(axis, 1)`: a zero-size
    // axis still spans one block. Match it so our files' declared bounds
    // agree with what readers compute from the region.
    let enclosing = [size[0].max(1), size[1].max(1), size[2].max(1)];
    vec3(&mut nbt, "EnclosingSize", enclosing)?;
    nbt.end()?; // Metadata

    nbt.end()?; // root compound
    nbt.into_inner()
        .map_err(|source| Error::WriteIo { source })?;
    Ok(())
}

/// Writes an `{x, y, z}` compound of i32 values.
fn vec3<W: IoWrite>(nbt: &mut Writer<W>, name: &str, [x, y, z]: [i32; 3]) -> Result<()> {
    nbt.named(TAG_COMPOUND, name)?;
    for (axis, v) in [("x", x), ("y", y), ("z", z)] {
        nbt.named(TAG_INT, axis)?;
        nbt.i32(v)?;
    }
    nbt.end()
}

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

#[cfg(test)]
mod tests {
    use super::Palette;
    use crate::region::BlockState;

    fn st(name: &str, props: &[(&str, &str)]) -> BlockState {
        BlockState {
            name: name.to_string(),
            properties: props
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
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
        let p = Palette::build(&declared);
        assert_eq!(p.len(), 3);
        assert_eq!(p.states()[0], air());
        assert_eq!(p.states()[1], st("a", &[]));
        assert_eq!(p.states()[2], st("c", &[]));
        assert_eq!(p.bits(), 2);
        assert_eq!(p.id_of(&air()), Some(0));
        assert_eq!(p.id_of(&st("a", &[])), Some(1));
        assert_eq!(p.id_of(&st("c", &[])), Some(2));
    }

    #[test]
    fn palette_build_keeps_cave_air_as_regular_entry() {
        let declared = [st("minecraft:cave_air", &[]), air()];
        let p = Palette::build(&declared);
        assert_eq!(p.len(), 2);
        assert_eq!(p.id_of(&st("minecraft:cave_air", &[])), Some(1));
    }

    #[test]
    fn palette_build_empty_declared_is_air_only() {
        let p = Palette::build(&[]);
        assert_eq!(p.len(), 1);
        assert_eq!(p.states(), [air()]);
        assert_eq!(p.bits(), 2);
    }

    #[test]
    fn palette_bits_match_litematica_widths() {
        for len in [1usize, 2, 3, 4, 5, 8, 9, 16, 17] {
            let declared: Vec<_> = (0..len)
                .map(|i| st("litematica:p", &[("i", &i.to_string())]))
                .collect();
            let p = Palette::build(&declared);
            let expect = crate::region::num_bits(len + 1);
            assert_eq!(p.bits(), expect, "palette of {len} entries + air");
        }
    }

    #[test]
    fn id_of_unknown_state_is_none() {
        let p = Palette::build(&[st("minecraft:stone", &[])]);
        assert_eq!(p.id_of(&st("minecraft:dirt", &[])), None);
    }
}
