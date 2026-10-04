//! Single-pass parsing of the litematic NBT structure into a [`Document`].
//!
//! The decompressed NBT buffer is scanned exactly once with a forward-only
//! [`Reader`]. Only the schema fields known to the litematic format are
//! decoded; everything else (entities, tile entities, pending ticks, preview
//! image data, ...) is skipped by length without allocation.

use crate::error::{Error, Result};
use crate::nbt::{
    Reader, TAG_COMPOUND, TAG_END, TAG_INT, TAG_LIST, TAG_LONG, TAG_LONG_ARRAY, TAG_STRING,
};
use crate::region::{BlockState, Region};
use crate::Document;

// Nesting depths of the schema levels, used only for the depth limit.
const ROOT_DEPTH: usize = 1;
const SCHEMA_DEPTH: usize = 2;
const REGION_DEPTH: usize = 3;
const REGION_FIELD_DEPTH: usize = 4;
const PALETTE_DEPTH: usize = 5;

fn expect_tag(r: &Reader<'_>, field: &'static str, actual: u8, expected: u8) -> Result<()> {
    if actual != expected {
        return Err(Error::UnexpectedTag {
            field,
            tag: actual,
            offset: r.pos(),
        });
    }
    Ok(())
}

/// Fields of the `Metadata` sub-compound of the file.
struct MetadataFields {
    name: String,
    description: String,
    author: String,
    region_count: Option<i32>,
    total_volume: Option<i64>,
    total_blocks: Option<i64>,
    time_created: i64,
    time_modified: i64,
    enclosing_size: Option<[i32; 3]>,
}

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
        let mut r = Reader::new(buf);
        let root_offset = r.pos();
        let root_tag = r.tag_id()?;
        if root_tag != TAG_COMPOUND {
            return Err(Error::UnexpectedTag {
                field: "root",
                tag: root_tag,
                offset: root_offset,
            });
        }
        // The root compound's name is arbitrary and ignored.
        let _root_name = r.string()?;

        let mut minecraft_data_version = 0i32;
        let mut version = 0i32;
        let mut sub_version: Option<i32> = None;
        let mut metadata: Option<MetadataFields> = None;
        let mut regions: Option<Vec<Region<'a>>> = None;

        loop {
            let tag = r.tag_id()?;
            if tag == TAG_END {
                break;
            }
            let name = r.string()?;
            match name.as_str() {
                "MinecraftDataVersion" => {
                    expect_tag(&r, "MinecraftDataVersion", tag, TAG_INT)?;
                    minecraft_data_version = r.i32()?;
                }
                "Version" => {
                    expect_tag(&r, "Version", tag, TAG_INT)?;
                    version = r.i32()?;
                }
                "SubVersion" => {
                    expect_tag(&r, "SubVersion", tag, TAG_INT)?;
                    sub_version = Some(r.i32()?);
                }
                "Metadata" => {
                    expect_tag(&r, "Metadata", tag, TAG_COMPOUND)?;
                    metadata = Some(parse_metadata(&mut r)?);
                }
                "Regions" => {
                    expect_tag(&r, "Regions", tag, TAG_COMPOUND)?;
                    regions = Some(parse_regions(&mut r)?);
                }
                _ => r.skip_value(tag, ROOT_DEPTH)?,
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
            metadata: crate::Metadata {
                name: fields.name,
                description: fields.description,
                author: fields.author,
                minecraft_data_version,
                version,
                sub_version,
                region_count: fields.region_count,
                total_volume: fields.total_volume,
                total_blocks: fields.total_blocks,
                time_created: fields.time_created,
                time_modified: fields.time_modified,
                enclosing_size: fields.enclosing_size,
            },
            regions,
        })
    }
}

fn parse_metadata(r: &mut Reader<'_>) -> Result<MetadataFields> {
    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut author: Option<String> = None;
    let mut region_count: Option<i32> = None;
    let mut total_volume: Option<i64> = None;
    let mut total_blocks: Option<i64> = None;
    let mut time_created = 0i64;
    let mut time_modified = 0i64;
    let mut enclosing_size: Option<[i32; 3]> = None;

    loop {
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let field = r.string()?;
        match field.as_str() {
            "Name" => {
                expect_tag(r, "Metadata.Name", tag, TAG_STRING)?;
                name = Some(r.string()?);
            }
            "Description" => {
                expect_tag(r, "Metadata.Description", tag, TAG_STRING)?;
                description = Some(r.string()?);
            }
            "Author" => {
                expect_tag(r, "Metadata.Author", tag, TAG_STRING)?;
                author = Some(r.string()?);
            }
            "RegionCount" => {
                expect_tag(r, "Metadata.RegionCount", tag, TAG_INT)?;
                region_count = Some(r.i32()?);
            }
            "TotalVolume" => {
                expect_tag(r, "Metadata.TotalVolume", tag, TAG_INT)?;
                total_volume = Some(i64::from(r.i32()?));
            }
            "TotalBlocks" => {
                // The Litematica mod writes an Int here; rustmatica writes a
                // Long. Accept both.
                match tag {
                    TAG_INT => total_blocks = Some(i64::from(r.i32()?)),
                    TAG_LONG => total_blocks = Some(r.i64()?),
                    _ => {
                        return Err(Error::UnexpectedTag {
                            field: "Metadata.TotalBlocks",
                            tag,
                            offset: r.pos(),
                        })
                    }
                }
            }
            "TimeCreated" => {
                expect_tag(r, "Metadata.TimeCreated", tag, TAG_LONG)?;
                time_created = r.i64()?;
            }
            "TimeModified" => {
                expect_tag(r, "Metadata.TimeModified", tag, TAG_LONG)?;
                time_modified = r.i64()?;
            }
            "EnclosingSize" => {
                expect_tag(r, "Metadata.EnclosingSize", tag, TAG_COMPOUND)?;
                enclosing_size = Some(parse_vec3(r, "EnclosingSize")?);
            }
            _ => r.skip_value(tag, SCHEMA_DEPTH)?,
        }
    }

    Ok(MetadataFields {
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

fn parse_regions<'a>(r: &mut Reader<'a>) -> Result<Vec<Region<'a>>> {
    let mut regions = Vec::new();
    loop {
        let offset = r.pos();
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let name = r.string()?;
        expect_tag(r, "region", tag, TAG_COMPOUND)?;
        regions.push(parse_region(r, name, offset)?);
    }
    Ok(regions)
}

fn parse_region<'a>(r: &mut Reader<'a>, name: String, region_offset: usize) -> Result<Region<'a>> {
    let mut position: Option<[i32; 3]> = None;
    let mut size: Option<[i32; 3]> = None;
    let mut palette: Option<Vec<BlockState>> = None;
    let mut states: Option<&[u8]> = None;
    let mut states_offset = 0usize;

    loop {
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let field = r.string()?;
        match field.as_str() {
            "Position" => {
                expect_tag(r, "Position", tag, TAG_COMPOUND)?;
                position = Some(parse_vec3(r, "Position")?);
            }
            "Size" => {
                expect_tag(r, "Size", tag, TAG_COMPOUND)?;
                size = Some(parse_vec3(r, "Size")?);
            }
            "BlockStatePalette" => {
                expect_tag(r, "BlockStatePalette", tag, TAG_LIST)?;
                palette = Some(parse_palette(r)?);
            }
            "BlockStates" => {
                expect_tag(r, "BlockStates", tag, TAG_LONG_ARRAY)?;
                states_offset = r.pos();
                let count = r.count()?;
                // Borrow the payload directly; never copy or expand it.
                states = Some(r.take(count as u64 * 8)?);
            }
            _ => r.skip_value(tag, REGION_DEPTH)?,
        }
    }

    let position = position.ok_or(Error::MissingField {
        field: "Position",
        offset: region_offset,
    })?;
    let size = size.ok_or(Error::MissingField {
        field: "Size",
        offset: region_offset,
    })?;
    let palette = palette.ok_or(Error::MissingField {
        field: "BlockStatePalette",
        offset: region_offset,
    })?;
    let states = states.ok_or(Error::MissingField {
        field: "BlockStates",
        offset: region_offset,
    })?;

    Region::from_parts(name, position, size, palette, states, states_offset)
}

/// Parses an `{x, y, z}` compound of i32 values. `field` names the compound
/// for error reporting. Unknown members are skipped.
fn parse_vec3(r: &mut Reader<'_>, field: &'static str) -> Result<[i32; 3]> {
    let mut values: [Option<i32>; 3] = [None, None, None];
    loop {
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let axis = r.string()?;
        match axis.as_str() {
            "x" => {
                expect_tag(r, field, tag, TAG_INT)?;
                values[0] = Some(r.i32()?);
            }
            "y" => {
                expect_tag(r, field, tag, TAG_INT)?;
                values[1] = Some(r.i32()?);
            }
            "z" => {
                expect_tag(r, field, tag, TAG_INT)?;
                values[2] = Some(r.i32()?);
            }
            _ => r.skip_value(tag, REGION_FIELD_DEPTH)?,
        }
    }
    if let [Some(x), Some(y), Some(z)] = values {
        Ok([x, y, z])
    } else {
        Err(Error::MissingField {
            field,
            offset: r.pos(),
        })
    }
}

/// Parses a `BlockStatePalette`: a TAG_List of compounds with a `Name`
/// string and an optional `Properties` compound of string values.
fn parse_palette(r: &mut Reader<'_>) -> Result<Vec<BlockState>> {
    let elem = r.tag_id()?;
    if elem != TAG_COMPOUND {
        return Err(Error::UnexpectedTag {
            field: "BlockStatePalette element",
            tag: elem,
            offset: r.pos(),
        });
    }
    let count = r.count()?;
    // Cap the pre-reservation: a hostile count may be huge while the payload
    // is tiny; the payload bounds the real length.
    let mut palette = Vec::with_capacity(count.min(256));
    for _ in 0..count {
        let entry_offset = r.pos();
        let mut name: Option<String> = None;
        let mut properties: Vec<(String, String)> = Vec::new();
        loop {
            let tag = r.tag_id()?;
            if tag == TAG_END {
                break;
            }
            let field = r.string()?;
            match field.as_str() {
                "Name" => {
                    expect_tag(r, "palette entry Name", tag, TAG_STRING)?;
                    name = Some(r.string()?);
                }
                "Properties" => {
                    expect_tag(r, "palette entry Properties", tag, TAG_COMPOUND)?;
                    loop {
                        let ptag = r.tag_id()?;
                        if ptag == TAG_END {
                            break;
                        }
                        let key = r.string()?;
                        expect_tag(r, "palette property value", ptag, TAG_STRING)?;
                        let value = r.string()?;
                        properties.push((key, value));
                    }
                }
                _ => r.skip_value(tag, PALETTE_DEPTH)?,
            }
        }
        let name = name.ok_or(Error::MissingField {
            field: "palette entry Name",
            offset: entry_offset,
        })?;
        palette.push(BlockState { name, properties });
    }
    Ok(palette)
}
