//! Streaming, low-memory serialization of `.litematic` schematics.
//!
//! The writer is the reader's mirror image: blocks are pulled from a
//! declarative [`Layout`] in storage order and packed straight into the
//! gzip-compressed NBT stream, so memory stays O(palette) no matter how
//! large the schematic is. The palette must be known before block data is
//! emitted (the litematic bit width depends on it); [`Layout::block_palette`]
//! may over-approximate, and its default implementation discovers the palette
//! with one extra scan.

use crate::bitstream::{bits_for, Packer};
use crate::error::{Error, Result};
use crate::mc::{BlockPos, BlockSize, BlockState};
use crate::nbt::{Encode, Tag, Writer};
use flate2::write::GzEncoder;
use flate2::Compression;
use std::cell::Cell;
use std::collections::HashSet;
use std::io::Write as IoWrite;
use std::time::{SystemTime, UNIX_EPOCH};

/// Litematica format version written by this crate (`Version`).
const FORMAT_VERSION: i32 = 6;
/// Litematica format sub-version written by this crate (`SubVersion`).
const FORMAT_SUB_VERSION: i32 = 1;

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
    /// Absolute dimensions per axis, never negative. The signed [`BlockPos`]
    /// shape mirrors the file's `Region.Size` field; the writer reads it as a
    /// span through [`BlockSize::of`]. Every position handed to
    /// [`Layout::block_at`] is contained by that span, as decided by
    /// [`BlockSize::contains`]. The value must stay stable throughout one
    /// [`write()`] call: it defines both the declared `Region.Size` and the
    /// voxels that get packed.
    fn size(&self) -> BlockPos;

    /// The block state at `pos`, or `None` for air; `pos` is assumed in bounds.
    ///
    /// Every returned state must be structurally equal (name and properties)
    /// to some entry of [`Layout::block_palette`]; a violation surfaces as
    /// [`Error::BlockNotInPalette`] during serialization. Air never needs an
    /// entry: both `None` and plain `minecraft:air` map to palette index 0.
    fn block_at(&self, pos: BlockPos) -> Option<BlockState>;

    /// Every state [`Layout::block_at`] may return, as a set, empty cells
    /// counted as air.
    ///
    /// Returning a set lets composite layouts merge their children's
    /// palettes with cheap unions instead of copying and deduplicating
    /// sequences. Over-approximation is allowed: unused entries simply
    /// travel along in the written file. Implementors that can derive their
    /// state set analytically should override this: the default discovers it
    /// with a full scan of the bounding box, which costs one extra pass over
    /// every voxel.
    fn block_palette(&self) -> HashSet<BlockState> {
        let positions = BlockSize::of(self.size()).cartesian_positions();
        positions.fold(HashSet::new(), |mut palette, pos| {
            palette.insert(self.block_at(pos).unwrap_or(BlockState::AIR));
            palette
        })
    }

    /// Writes this layout as a single-region `.litematic` file into `sink`.
    ///
    /// The same operation as the free [`write()`], reached from the value that
    /// describes the schematic.
    fn as_litematic<W: IoWrite>(&self, options: &WriteOptions, sink: W) -> Result<()>
    where
        Self: Sized,
    {
        write(self, options, sink)
    }
}

// Block values
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

impl Encode for BlockPos {
    const TAG: Tag = Tag::Compound;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        self.x.encode("x", out)?;
        self.y.encode("y", out)?;
        self.z.encode("z", out)?;
        out.end()
    }
}

impl Encode for BlockState {
    const TAG: Tag = Tag::Compound;

    /// Writes the entry body, omitting an empty `Properties` compound.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        self.name.as_ref().encode("Name", out)?;
        if self.properties.is_empty() {
            return out.end();
        }
        out.compound("Properties")?;
        for (key, value) in &self.properties {
            value.encode(key.as_ref(), out)?;
        }
        out.end()?;
        out.end()
    }
}

// Palette
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The file-side block state palette of a written schematic.
///
/// Built from the states a [`Layout`] declares: air entries collapse
/// into the format-mandated index 0, the rest are sorted and deduplicated so
/// that [`Palette::id_of`] can answer with a binary search.
pub(crate) struct Palette {
    /// Non-air states in file order; plain `minecraft:air` is implicit at
    /// index 0 and never stored.
    states: Vec<BlockState>,
}

impl Palette {
    /// Builds the file palette from the states a layout declared.
    ///
    /// Over-approximations are fine: duplicate entries collapse, and unused
    /// entries simply travel along in the file. Air is never stored: the
    /// format pins plain `minecraft:air` to index 0, and the palette list is
    /// written with it first, as Litematica does.
    pub(crate) fn build(declared: impl IntoIterator<Item = BlockState>) -> Self {
        let mut states: Vec<BlockState> = declared
            .into_iter()
            .filter(|state| state != &BlockState::AIR)
            .collect();
        states.sort();
        states.dedup();
        Self { states }
    }

    /// The file palette index of `state`, or `None` if the declared palette
    /// does not cover it.
    pub(crate) fn id_of(&self, state: &BlockState) -> Option<u32> {
        if state == &BlockState::AIR {
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
    pub(crate) fn num_bits(&self) -> u32 {
        bits_for(self.len())
    }

    /// Resolves the id written at `pos`: air is 0, and any other state must
    /// be covered by this palette.
    fn id_at<L: Layout>(&self, layout: &L, pos: BlockPos) -> Result<u32> {
        let Some(state) = layout.block_at(pos) else {
            return Ok(0);
        };
        self.id_of(&state).ok_or(Error::BlockNotInPalette {
            name: state.name.to_string(),
            x: pos.x,
            y: pos.y,
            z: pos.z,
        })
    }
}

impl Encode for Palette {
    const TAG: Tag = Tag::List;

    /// Writes air first, then the stored states, naming a length one larger.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        let len = i32::try_from(self.len()).map_err(|_| Error::DimensionOverflow {
            what: "BlockStatePalette length",
            value: self.len() as u128,
        })?;
        out.list(Tag::Compound, len)?;
        BlockState::AIR.body(out)?;
        self.states().iter().try_for_each(|state| state.body(out))
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
/// # use litematica::write::{write, Layout, WriteOptions};
/// # struct MyLayout;
/// # impl Layout for MyLayout {
/// #     fn size(&self) -> litematica::mc::BlockPos {
/// #         litematica::mc::BlockPos::new(16, 16, 16)
/// #     }
/// #     fn block_at(&self, pos: litematica::mc::BlockPos)
/// #         -> Option<litematica::mc::BlockState> { None }
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
    let total_blocks = Cell::new(0i64);
    let schematic = Schematic::new(layout, options, &total_blocks)?;
    let mut gz = GzEncoder::new(sink, options.compression);
    let mut nbt = Writer::new(&mut gz);
    schematic.write(&mut nbt)?;
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
    size: BlockPos,
    volume: i32,
    longs: i32,
    total_blocks: &'a Cell<i64>,
}

impl<'a, L: Layout> Schematic<'a, L> {
    /// Validates the layout's bounds and assembles the palette and lengths
    /// every later stage reads.
    fn new(layout: &'a L, options: &'a WriteOptions, total_blocks: &'a Cell<i64>) -> Result<Self> {
        let size = layout.size();
        let volume = Self::checked_volume(size)?;
        let palette = Palette::build(layout.block_palette());
        let bits = palette.num_bits();
        // `volume <= i32::MAX` and `bits <= 32` bound this well below i32::MAX;
        // the checked conversion keeps the written length prefix honest.
        let longs = (u128::from(volume.unsigned_abs()) * u128::from(bits)).div_ceil(64);
        let longs = i32::try_from(longs).map_err(|_| Error::DimensionOverflow {
            what: "BlockStates length",
            value: longs,
        })?;
        Ok(Self {
            layout,
            options,
            palette,
            size,
            volume,
            longs,
            total_blocks,
        })
    }

    /// Rejects negative axes, then returns the volume as the `i32` the format
    /// stores in `TotalVolume`.
    fn checked_volume(size: BlockPos) -> Result<i32> {
        let axes = [("x", size.x), ("y", size.y), ("z", size.z)];
        if let Some((axis, value)) = axes.into_iter().find(|(_, value)| *value < 0) {
            return Err(Error::NegativeSize { axis, value });
        }
        let volume = BlockSize::of(size).volume();
        i32::try_from(volume).map_err(|_| Error::DimensionOverflow {
            what: "volume",
            value: volume,
        })
    }

    /// Opens the root compound and writes the whole document into it.
    fn write<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.root()?;
        self.body(out)
    }
}

impl<L: Layout> Encode for Schematic<'_, L> {
    const TAG: Tag = Tag::Compound;

    /// Writes the root compound body: the region, versions, and metadata.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        out.compound("Regions")?;
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

/// One region: its box, palette, block states, and tile entities.
struct RegionNbt<'a, L: Layout> {
    layout: &'a L,
    palette: &'a Palette,
    size: BlockPos,
    longs: i32,
    total_blocks: &'a Cell<i64>,
}

impl<L: Layout> Encode for RegionNbt<'_, L> {
    const TAG: Tag = Tag::Compound;

    /// Writes the region body: its box, palette, blocks, and tile entities.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        BlockPos::ORIGIN.encode("Position", out)?;
        self.size.encode("Size", out)?;
        self.palette.encode("BlockStatePalette", out)?;
        let states = BlockStates {
            layout: self.layout,
            palette: self.palette,
            size: self.size,
            longs: self.longs,
            total_blocks: self.total_blocks,
        };
        states.encode("BlockStates", out)?;
        // rustmatica requires `TileEntities` to be present. `Entities` and the
        // pending-tick lists carry serde defaults, so omitting them is fine.
        out.empty_list("TileEntities")?;
        out.end()
    }
}

/// The packed block states of a region, pulled one long at a time.
struct BlockStates<'a, L: Layout> {
    layout: &'a L,
    palette: &'a Palette,
    size: BlockPos,
    longs: i32,
    total_blocks: &'a Cell<i64>,
}

impl<L: Layout> Encode for BlockStates<'_, L> {
    const TAG: Tag = Tag::LongArray;

    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        let packed = Packed::new(self.layout, self.palette, self.size, self.total_blocks);
        out.long_array(self.longs, packed)
    }
}

/// Packs palette ids into longs, one word per step, counting non-air blocks.
struct Packed<'a, L: Layout> {
    layout: &'a L,
    palette: &'a Palette,
    size: BlockSize,
    cursor: BlockPos,
    remaining: u64,
    packer: Packer,
    finished: bool,
    total_blocks: &'a Cell<i64>,
}

impl<'a, L: Layout> Packed<'a, L> {
    /// Prepares the stream for `layout`, walking `size` in storage order.
    fn new(
        layout: &'a L,
        palette: &'a Palette,
        size: BlockPos,
        total_blocks: &'a Cell<i64>,
    ) -> Self {
        let size = BlockSize::of(size);
        Self {
            layout,
            palette,
            size,
            cursor: BlockPos::ORIGIN,
            remaining: u64::try_from(size.volume()).unwrap_or(u64::MAX),
            packer: Packer::new(palette.num_bits()),
            finished: false,
            total_blocks,
        }
    }
}

impl<L: Layout> Packed<'_, L> {
    /// Advances the cursor in storage order (`x` fastest, then `z`, then `y`)
    /// and returns the position it just left. The size was validated
    /// non-negative, so every coordinate lies inside the box.
    fn next_position(&mut self) -> BlockPos {
        let pos = self.cursor;
        self.cursor.x += 1;
        if self.cursor.x == self.size.x as i32 {
            self.cursor.x = 0;
            self.cursor.z += 1;
            if self.cursor.z == self.size.z as i32 {
                self.cursor.z = 0;
                self.cursor.y += 1;
            }
        }
        pos
    }
}

impl<L: Layout> Iterator for Packed<'_, L> {
    type Item = Result<i64>;

    /// Produces one completed long at a time, pulling as many voxels as the
    /// current word needs. A palette miss ends the stream with that error.
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            if self.remaining == 0 {
                self.finished = true;
                return self.packer.finish().map(|word| Ok(word as i64));
            }
            self.remaining -= 1;
            let pos = self.next_position();
            let id = match self.palette.id_at(self.layout, pos) {
                Ok(id) => id,
                Err(error) => {
                    self.finished = true;
                    return Some(Err(error));
                }
            };
            if id != 0 {
                self.total_blocks.set(self.total_blocks.get() + 1);
            }
            if let Some(word) = self.packer.push(id) {
                return Some(Ok(word as i64));
            }
        }
    }
}

/// The metadata compound of the written schematic.
struct Header<'a> {
    options: &'a WriteOptions,
    size: BlockPos,
    volume: i32,
    /// Accumulated by `BlockStates` while the regions stream. Metadata is
    /// written after the regions, so the count is final when read here.
    total_blocks: &'a Cell<i64>,
}

impl Encode for Header<'_> {
    const TAG: Tag = Tag::Compound;

    /// Writes the metadata body.
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
        // `EnclosingSize` is a cached summary only: no reader validates or uses
        // it, and Litematica stores whatever value it last computed. Write the
        // declared region size verbatim. Do not clamp a zero axis to 1 to mimic
        // rustmatica's recomputation: that clamp is an artifact of supporting
        // negative sizes and would make the header disagree with `Region.Size`.
        self.size.encode("EnclosingSize", out)?;
        out.end()
    }
}

// Packing
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::{Layout, Palette};
    use crate::mc::{BlockPos, BlockState};

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
    fn palette_build_maps_air_to_zero_and_sorts_rest() {
        let declared = [st("c", &[]), air(), st("a", &[]), st("c", &[])];
        let p = Palette::build(declared);
        assert_eq!(p.len(), 3);
        assert_eq!(p.states(), [st("a", &[]), st("c", &[])]);
        assert_eq!(p.num_bits(), 2);
        assert_eq!(p.id_of(&air()), Some(0));
        assert_eq!(p.id_of(&st("a", &[])), Some(1));
        assert_eq!(p.id_of(&st("c", &[])), Some(2));
        assert_eq!(p.id_of(&st("minecraft:dirt", &[])), None);
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
        assert_eq!(p.num_bits(), 2);
    }

    #[test]
    fn default_palette_does_not_reserve_the_whole_box() {
        struct Empty(BlockPos);

        impl Layout for Empty {
            fn size(&self) -> BlockPos {
                self.0
            }

            fn block_at(&self, _pos: BlockPos) -> Option<BlockState> {
                None
            }
        }

        let palette = Empty(BlockPos::new(64, 64, 64)).block_palette();
        assert_eq!(palette, [air()].into());
        assert!(palette.capacity() < 1024, "palette reserved the whole box");
    }
}
