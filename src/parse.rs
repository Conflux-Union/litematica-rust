//! Single-pass parsing of the litematic NBT structure into a [`Document`].
//!
//! Each schema node reads itself: [`Document`] owns the root, [`MetadataFields`]
//! owns the header compound, and [`Regions`] owns the region collection, which
//! in turn asks [`Region`] to read one region. The only thing a parent supplies
//! is the name and depth its child sits at, so the shape of the file is stated
//! once, next to the type that mirrors it.
//!
//! The decompressed NBT buffer is scanned exactly once with a forward-only
//! [`Reader`]. Only the schema fields known to the litematic format are
//! decoded; everything else (entities, tile entities, pending ticks, preview
//! image data, ...) is skipped by length without allocation.

use crate::error::{Error, Result};
use crate::mc::BlockPos;
use crate::nbt::{Decode, Reader, SliceSource, Source, Tag};
use crate::region::Region;
use crate::Document;

// Document
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

impl<'a> Document<'a> {
    /// Parses a decompressed (gunzipped) litematic NBT buffer.
    ///
    /// The returned document borrows from `buf`: each region's block data is
    /// decoded lazily from a `&[u8]` slice into `buf` and is never copied or
    /// expanded, so `buf` must outlive the document.
    ///
    /// # Required fields
    ///
    /// Mirroring rustmatica, the root must contain a `Metadata` and a
    /// `Regions` compound, and every region must contain `Position`, `Size`,
    /// `BlockStatePalette`, and `BlockStates`. Unknown fields are tolerated
    /// and skipped. All errors report the byte offset at which they were
    /// detected.
    pub fn parse(buf: &'a [u8]) -> Result<Self> {
        let mut r = Reader::new(SliceSource::new(buf));
        let root_offset = r.pos();
        let root_tag = r.tag()?;
        if root_tag != Tag::Compound {
            return Err(Error::UnexpectedTag {
                field: "root",
                tag: root_tag.id(),
                offset: root_offset,
            });
        }
        // The root compound's name is arbitrary and ignored.
        r.string()?;
        Self::decode(&mut r, "root", 0)
    }
}

impl<'a> Decode<SliceSource<'a>> for Document<'a> {
    /// Reads the root compound's fields.
    fn decode(
        r: &mut Reader<SliceSource<'a>>,
        _field: &'static str,
        enclosing: usize,
    ) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut minecraft_data_version = 0i32;
        let mut version = 0i32;
        let mut sub_version: Option<i32> = None;
        let mut metadata: Option<MetadataFields> = None;
        let mut regions: Option<Regions<'a>> = None;

        loop {
            let tag = r.tag()?;
            if tag == Tag::End {
                break;
            }
            let name = r.string()?;
            match name.as_str() {
                "MinecraftDataVersion" => {
                    r.expect("MinecraftDataVersion", tag, Tag::Int)?;
                    minecraft_data_version = r.i32()?;
                }
                "Version" => {
                    r.expect("Version", tag, Tag::Int)?;
                    version = r.i32()?;
                }
                "SubVersion" => {
                    r.expect("SubVersion", tag, Tag::Int)?;
                    sub_version = Some(r.i32()?);
                }
                "Metadata" => {
                    r.expect("Metadata", tag, Tag::Compound)?;
                    metadata = Some(MetadataFields::decode(r, "Metadata", depth)?);
                }
                "Regions" => {
                    r.expect("Regions", tag, Tag::Compound)?;
                    regions = Some(Regions::decode(r, "Regions", depth)?);
                }
                _ => r.skip_value(tag, depth)?,
            }
        }

        let fields = metadata.ok_or(Error::MissingField {
            field: "Metadata",
            offset: r.pos(),
        })?;
        let regions = regions.ok_or(Error::MissingField {
            field: "Regions",
            offset: r.pos(),
        })?;

        Ok(Document {
            metadata: fields.to_metadata(minecraft_data_version, version, sub_version),
            regions: regions.0,
        })
    }
}

// Block geometry
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

impl<S: Source> Decode<S> for BlockPos {
    /// Reads the `{x, y, z}` compound that `Position`, `Size`, and
    /// `EnclosingSize` are all written as. Unknown members are skipped.
    ///
    /// A `Size`-shaped field stays signed here, exactly as declared; use
    /// [`crate::mc::BlockSize::of`] where it becomes a span.
    fn decode(r: &mut Reader<S>, field: &'static str, enclosing: usize) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut values: [Option<i32>; 3] = [None, None, None];
        loop {
            let tag = r.tag()?;
            if tag == Tag::End {
                break;
            }
            let axis = r.string()?;
            let slot = match axis.as_str() {
                "x" => 0,
                "y" => 1,
                "z" => 2,
                _ => {
                    r.skip_value(tag, depth)?;
                    continue;
                }
            };
            r.expect(field, tag, Tag::Int)?;
            values[slot] = Some(r.i32()?);
        }
        let [Some(x), Some(y), Some(z)] = values else {
            return Err(Error::MissingField {
                field,
                offset: r.pos(),
            });
        };
        Ok(BlockPos::new(x, y, z))
    }
}

// Metadata
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The `Metadata` compound, before the root-level versions are joined to it.
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
    ) -> crate::Metadata {
        crate::Metadata {
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
    /// Reads the `Metadata` compound.
    fn decode(r: &mut Reader<S>, _field: &'static str, enclosing: usize) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut name: Option<String> = None;
        let mut description: Option<String> = None;
        let mut author: Option<String> = None;
        let mut region_count: Option<i32> = None;
        let mut total_volume: Option<i64> = None;
        let mut total_blocks: Option<i64> = None;
        let mut time_created = 0i64;
        let mut time_modified = 0i64;
        let mut enclosing_size: Option<BlockPos> = None;

        loop {
            let tag = r.tag()?;
            if tag == Tag::End {
                break;
            }
            let field = r.string()?;
            match field.as_str() {
                "Name" => {
                    r.expect("Metadata.Name", tag, Tag::String)?;
                    name = Some(r.string()?);
                }
                "Description" => {
                    r.expect("Metadata.Description", tag, Tag::String)?;
                    description = Some(r.string()?);
                }
                "Author" => {
                    r.expect("Metadata.Author", tag, Tag::String)?;
                    author = Some(r.string()?);
                }
                "RegionCount" => {
                    r.expect("Metadata.RegionCount", tag, Tag::Int)?;
                    region_count = Some(r.i32()?);
                }
                "TotalVolume" => {
                    r.expect("Metadata.TotalVolume", tag, Tag::Int)?;
                    total_volume = Some(i64::from(r.i32()?));
                }
                "TotalBlocks" => total_blocks = Some(Self::decode_total_blocks(r, tag)?),
                "TimeCreated" => {
                    r.expect("Metadata.TimeCreated", tag, Tag::Long)?;
                    time_created = r.i64()?;
                }
                "TimeModified" => {
                    r.expect("Metadata.TimeModified", tag, Tag::Long)?;
                    time_modified = r.i64()?;
                }
                "EnclosingSize" => {
                    r.expect("Metadata.EnclosingSize", tag, Tag::Compound)?;
                    enclosing_size = Some(BlockPos::decode(r, "EnclosingSize", depth)?);
                }
                _ => r.skip_value(tag, depth)?,
            }
        }

        Ok(Self {
            name: name.unwrap_or_default(),
            description: description.unwrap_or_default(),
            author: author.unwrap_or_default(),
            region_count,
            total_volume,
            total_blocks,
            time_created,
            time_modified,
            enclosing_size,
        })
    }
}

// Regions
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The `Regions` compound: a map from region name to region.
struct Regions<'a>(Vec<Region<'a>>);

impl<'a> Decode<SliceSource<'a>> for Regions<'a> {
    /// Reads every region in file order, passing each its member name.
    fn decode(
        r: &mut Reader<SliceSource<'a>>,
        _field: &'static str,
        enclosing: usize,
    ) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut regions = Vec::new();
        loop {
            let offset = r.pos();
            let tag = r.tag()?;
            if tag == Tag::End {
                break;
            }
            let name = r.string()?;
            r.expect("region", tag, Tag::Compound)?;
            regions.push(Region::decode(r, name, offset, depth)?);
        }
        Ok(Self(regions))
    }
}
