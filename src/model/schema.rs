//! Every NBT shape the litematic format defines, in one place.
//!
//! This module owns the codecs for the crate's vocabulary ([`BlockPos`],
//! [`BlockState`], [`Palette`]) and the accumulators for the root and metadata
//! compounds. Both read modes and the writer reach the same definitions here,
//! so a model type never has one codec per task.
//!
//! A file's root compound describes the schematic before and around the
//! regions: its format versions and its `Metadata` compound. Files are free to
//! order those fields however they like, so [`RootFields`] accepts one field
//! at a time. The random-access reader consumes it once at the end; the
//! streaming reader may ask it for the metadata after every field, including
//! ones that arrive after the regions.
//!
//! [`RootFields::read_field`] also reports when it reaches `Regions`, so a
//! caller learns about the boundary without this module owning the region
//! collection.

use crate::error::{Error, Quantity, Result};
use crate::knowledge::mc::{BlockPos, BlockSize, BlockState};
use crate::knowledge::nbt::{Decode, Encode, FieldEntry, FieldOutcome, FieldTable};
use crate::knowledge::nbt::{Reader, Slot, Source, Tag, Writer};
use crate::model::document::Metadata;
use crate::model::palette::{OrderedPalette, Palette};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::Write as IoWrite;

// Root fields
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The root fields that describe a schematic, accumulated in any order.
#[derive(Default)]
pub(crate) struct RootFields {
    minecraft_data_version: i32,
    version: i32,
    sub_version: Slot<i32>,
    metadata: Slot<MetadataFields>,
    /// The metadata assembled from the fields seen so far, kept current so a
    /// streaming caller can read it cheaply between fields.
    resolved: Option<Metadata>,
}

impl RootFields {
    /// Creates an empty accumulator; every field defaults until one is read.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The root members this accumulator recognizes, each paired with the way
    /// it deposits its value. `Regions` is deferred: it marks a compound the
    /// caller owns and must enter itself.
    fn fields<R: Source>() -> [FieldEntry<Self, R>; 5] {
        [
            FieldEntry::new("MinecraftDataVersion", |h, r, tag, depth| {
                h.minecraft_data_version = r.decode_field("MinecraftDataVersion", tag, depth)?;
                Ok(())
            }),
            FieldEntry::new("Version", |h, r, tag, depth| {
                h.version = r.decode_field("Version", tag, depth)?;
                Ok(())
            }),
            FieldEntry::new("SubVersion", |h, r, tag, depth| {
                h.sub_version.set(r.decode_field("SubVersion", tag, depth)?);
                Ok(())
            }),
            FieldEntry::new("Metadata", |h, r, tag, depth| {
                h.metadata.set(r.decode_field("Metadata", tag, depth)?);
                Ok(())
            }),
            FieldEntry::deferred("Regions", |_, r, tag, _| {
                r.expect("Regions", tag, Tag::Compound)
            }),
        ]
    }

    /// Consumes one root field whose name has already been read.
    ///
    /// `enclosing` is the depth of the root compound's fields. A deferred
    /// outcome means `Regions` was reached; its compound has not been entered,
    /// so the caller decides whether to read it or skip it.
    pub(crate) fn read_field<R: Source>(
        &mut self,
        r: &mut Reader<R>,
        name: &str,
        tag: Tag,
        enclosing: usize,
    ) -> Result<FieldOutcome> {
        let fields = Self::fields();
        let outcome = FieldTable::new(&fields).apply(self, r, name, tag, enclosing)?;
        self.resolve();
        Ok(outcome)
    }

    /// The header assembled so far, or `None` until `Metadata` has appeared.
    pub(crate) fn metadata(&self) -> Option<&Metadata> {
        self.resolved.as_ref()
    }

    /// Consumes the header, requiring a `Metadata` compound to have appeared.
    pub(crate) fn into_metadata(self, offset: usize) -> Result<Metadata> {
        self.resolved.ok_or(Error::MissingField {
            field: "Metadata",
            offset,
        })
    }

    /// Rebuilds the join of the metadata compound and the root-level versions.
    fn resolve(&mut self) {
        let (data_version, version) = (self.minecraft_data_version, self.version);
        let sub_version = self.sub_version.get().copied();
        let resolved = self
            .metadata
            .get()
            .map(|fields| fields.to_metadata(data_version, version, sub_version));
        self.resolved = resolved;
    }
}

// Vocabulary codecs
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

impl<S: Source> Decode<S> for BlockPos {
    const TAG: Tag = Tag::Compound;

    /// Reads the `{x, y, z}` compound that `Position`, `Size`, and
    /// `EnclosingSize` are all written as. Unknown members are skipped.
    ///
    /// A `Size`-shaped field stays signed here, exactly as declared; use
    /// [`BlockSize::of`] where it becomes a span.
    fn decode(r: &mut Reader<S>, field: &'static str, enclosing: usize) -> Result<Self> {
        /// The axis names in the order a position stores them.
        const AXES: [&str; 3] = ["x", "y", "z"];

        let depth = r.nested(enclosing)?;
        let mut values: [Slot<i32>; 3] = Default::default();
        while let Some((axis, tag)) = r.next_field()? {
            if let Some(index) = AXES.iter().position(|name| *name == axis) {
                values[index].set(r.decode_field(field, tag, depth)?);
            } else {
                r.skip_value(tag, depth)?;
            }
        }
        let [x, y, z] = values;
        let offset = r.pos();
        Ok(BlockPos::new(
            x.required(field, offset)?,
            y.required(field, offset)?,
            z.required(field, offset)?,
        ))
    }
}

impl Encode for BlockPos {
    const TAG: Tag = Tag::Compound;

    /// Writes the `{x, y, z}` compound shared by every position-shaped field.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        self.x.encode("x", out)?;
        self.y.encode("y", out)?;
        self.z.encode("z", out)?;
        out.end()
    }
}

impl Encode for BlockSize {
    const TAG: Tag = Tag::Compound;

    /// Writes a span as the signed triple the file expects, rejecting an axis
    /// that does not fit an `i32`.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        let axis = |value: u32| {
            i32::try_from(value).map_err(|_| Error::DimensionOverflow {
                quantity: Quantity::SizeAxis,
                value: u128::from(value),
            })
        };
        axis(self.x)?.encode("x", out)?;
        axis(self.y)?.encode("y", out)?;
        axis(self.z)?.encode("z", out)?;
        out.end()
    }
}

impl<S: Source> Decode<S> for BlockState {
    const TAG: Tag = Tag::Compound;

    /// Reads one palette entry: a compound with a `Name` and an optional
    /// `Properties` compound.
    fn decode(r: &mut Reader<S>, _field: &'static str, enclosing: usize) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let offset = r.pos();
        let fields = EntryFields::fields();
        let mut entry = EntryFields::default();
        while let Some((field, tag)) = r.next_field()? {
            FieldTable::new(&fields).apply(&mut entry, r, &field, tag, depth)?;
        }
        Ok(Self {
            name: Cow::Owned(entry.name.required("palette entry Name", offset)?),
            properties: entry.properties.or_default(),
        })
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

/// The members of a palette entry while they are still arriving.
///
/// `Properties` is optional, so it stays in a [`Slot`] until the compound
/// ends; `Name` is required and reports the entry's byte offset when missing.
#[derive(Default)]
struct EntryFields {
    name: Slot<String>,
    properties: Slot<BTreeMap<Cow<'static, str>, Cow<'static, str>>>,
}

impl EntryFields {
    /// The members of a palette entry and the way each one lands here.
    fn fields<R: Source>() -> [FieldEntry<Self, R>; 2] {
        [
            FieldEntry::new("Name", |entry, r, tag, depth| {
                entry
                    .name
                    .set(r.decode_field("palette entry Name", tag, depth)?);
                Ok(())
            }),
            FieldEntry::new("Properties", |entry, r, tag, depth| {
                entry.properties.set(decode_properties(r, tag, depth)?);
                Ok(())
            }),
        ]
    }
}

/// Reads a `Properties` compound as the keyed map the shared [`BlockState`]
/// orders by, so a file's member order cannot leak into state identity.
fn decode_properties<S: Source>(
    r: &mut Reader<S>,
    tag: Tag,
    enclosing: usize,
) -> Result<BTreeMap<Cow<'static, str>, Cow<'static, str>>> {
    r.expect("palette entry Properties", tag, Tag::Compound)?;
    let depth = r.nested(enclosing)?;
    let mut properties = BTreeMap::new();
    while let Some((key, tag)) = r.next_field()? {
        let value = r.decode_field("palette property value", tag, depth)?;
        properties.insert(Cow::Owned(key), Cow::Owned(value));
    }
    Ok(properties)
}

impl<S: Source> Decode<S> for Palette {
    const TAG: Tag = Tag::List;

    /// Reads a `BlockStatePalette`: the list of states whose order defines
    /// every packed identifier in the region.
    fn decode(r: &mut Reader<S>, _field: &'static str, enclosing: usize) -> Result<Self> {
        let elem = r.tag()?;
        r.expect("BlockStatePalette element", elem, Tag::Compound)?;
        let count = r.count()?;
        let entry_depth = r.nested(enclosing)?;
        // Cap the pre-reservation: a hostile count may be huge while the
        // payload is tiny; the payload bounds the real length.
        let mut states = Vec::with_capacity(count.min(256));
        for _ in 0..count {
            states.push(BlockState::decode(r, "palette entry", entry_depth)?);
        }
        Ok(Palette::from_file(states))
    }
}

impl Encode for Palette {
    const TAG: Tag = Tag::List;

    /// Writes the palette in its own order.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        let len = i32::try_from(self.len()).map_err(|_| Error::DimensionOverflow {
            quantity: Quantity::PaletteLength,
            value: self.len() as u128,
        })?;
        out.list(Tag::Compound, len)?;
        self.states().iter().try_for_each(|state| state.body(out))
    }
}

impl Encode for OrderedPalette {
    const TAG: Tag = Tag::List;

    /// Writes the canonical order; an ordered palette already begins with air.
    fn body<W: IoWrite>(&self, out: &mut Writer<W>) -> Result<()> {
        Encode::body(&**self, out)
    }
}

// Metadata
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The `Metadata` compound, before the root-level versions are joined to it.
#[derive(Default)]
pub(crate) struct MetadataFields {
    name: String,
    description: String,
    author: String,
    region_count: Option<i32>,
    total_volume: Option<i64>,
    total_blocks: Option<i64>,
    time_created: i64,
    time_modified: i64,
    enclosing_size: Option<BlockPos>,
}

impl MetadataFields {
    /// The members of a `Metadata` compound and the way each one lands in the
    /// accumulator. Absent members keep their defaults, so the scan needs no
    /// post-processing.
    fn fields<R: Source>() -> [FieldEntry<Self, R>; 9] {
        [
            FieldEntry::new("Name", |m, r, tag, depth| {
                m.name = r.decode_field("Metadata.Name", tag, depth)?;
                Ok(())
            }),
            FieldEntry::new("Description", |m, r, tag, depth| {
                m.description = r.decode_field("Metadata.Description", tag, depth)?;
                Ok(())
            }),
            FieldEntry::new("Author", |m, r, tag, depth| {
                m.author = r.decode_field("Metadata.Author", tag, depth)?;
                Ok(())
            }),
            FieldEntry::new("RegionCount", |m, r, tag, depth| {
                m.region_count = Some(r.decode_field("Metadata.RegionCount", tag, depth)?);
                Ok(())
            }),
            FieldEntry::new("TotalVolume", |m, r, tag, _| {
                r.expect("Metadata.TotalVolume", tag, Tag::Int)?;
                m.total_volume = Some(i64::from(r.i32()?));
                Ok(())
            }),
            FieldEntry::new("TotalBlocks", |m, r, tag, _| {
                m.total_blocks = Some(Self::decode_total_blocks(r, tag)?);
                Ok(())
            }),
            FieldEntry::new("TimeCreated", |m, r, tag, depth| {
                m.time_created = r.decode_field("Metadata.TimeCreated", tag, depth)?;
                Ok(())
            }),
            FieldEntry::new("TimeModified", |m, r, tag, depth| {
                m.time_modified = r.decode_field("Metadata.TimeModified", tag, depth)?;
                Ok(())
            }),
            FieldEntry::new("EnclosingSize", |m, r, tag, depth| {
                m.enclosing_size = Some(r.decode_field("Metadata.EnclosingSize", tag, depth)?);
                Ok(())
            }),
        ]
    }

    /// Reads `TotalBlocks`, which the Litematica mod writes as an Int and
    /// rustmatica writes as a Long. Both are accepted.
    fn decode_total_blocks<S: Source>(r: &mut Reader<S>, tag: Tag) -> Result<i64> {
        match tag {
            Tag::Int => Ok(i64::from(r.i32()?)),
            Tag::Long => r.i64(),
            _ => Err(Error::UnexpectedTag {
                field: "Metadata.TotalBlocks",
                tag: tag.id(),
                offset: r.pos(),
            }),
        }
    }

    /// Combines this compound with the versions declared beside it at the root.
    pub(crate) fn to_metadata(
        &self,
        minecraft_data_version: i32,
        version: i32,
        sub_version: Option<i32>,
    ) -> Metadata {
        Metadata {
            name: self.name.clone(),
            description: self.description.clone(),
            author: self.author.clone(),
            minecraft_data_version,
            version,
            sub_version,
            region_count: self.region_count,
            total_volume: self.total_volume,
            total_blocks: self.total_blocks,
            time_created: self.time_created,
            time_modified: self.time_modified,
            enclosing_size: self.enclosing_size,
        }
    }
}

impl<S: Source> Decode<S> for MetadataFields {
    const TAG: Tag = Tag::Compound;

    /// Reads the `Metadata` compound.
    fn decode(r: &mut Reader<S>, _field: &'static str, enclosing: usize) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let fields = Self::fields();
        let mut metadata = Self::default();
        while let Some((field, tag)) = r.next_field()? {
            FieldTable::new(&fields).apply(&mut metadata, r, &field, tag, depth)?;
        }
        Ok(metadata)
    }
}
