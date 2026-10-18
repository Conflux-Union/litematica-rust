//! Single-pass parsing of the litematic NBT structure into a [`Document`].
//!
//! The decompressed NBT buffer is scanned exactly once with a forward-only
//! [`Reader`]. Only the schema fields known to the litematic format are
//! decoded; everything else (entities, tile entities, pending ticks, preview
//! image data, ...) is skipped by length without allocation.

use std::borrow::Cow;

use crate::error::{Error, Result};
use crate::mcdata::{BlockState, Properties};
use crate::nbt::{Reader, TAG_COMPOUND, TAG_END, TAG_INT, TAG_LIST};
use crate::nbt::{TAG_LONG, TAG_LONG_ARRAY, TAG_STRING};
use crate::region::Region;
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

/// Reads an `i32` value after checking that its tag is `TAG_INT`.
fn read_int(r: &mut Reader<'_>, field: &'static str, tag: u8) -> Result<i32> {
    expect_tag(r, field, tag, TAG_INT)?;
    r.i32()
}

/// Reads a string value after checking that its tag is `TAG_STRING`.
fn read_string(r: &mut Reader<'_>, field: &'static str, tag: u8) -> Result<String> {
    expect_tag(r, field, tag, TAG_STRING)?;
    r.string()
}

/// Fields of the root compound of the file.
#[derive(Default)]
struct RootFields<'a> {
    data_version: i32,
    version: i32,
    sub_version: Option<i32>,
    metadata: Option<MetadataFields>,
    regions: Option<Vec<Region<'a>>>,
}

/// Fields of the `Metadata` sub-compound of the file.
#[derive(Default)]
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

/// Fields of one region compound, before the required ones are validated.
#[derive(Default)]
struct RegionFields<'a> {
    position: Option<[i32; 3]>,
    size: Option<[i32; 3]>,
    palette: Option<Vec<BlockState>>,
    states: Option<&'a [u8]>,
    states_offset: usize,
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

        let mut fields = RootFields::default();
        loop {
            let tag = r.tag_id()?;
            if tag == TAG_END {
                break;
            }
            let name = r.string()?;
            parse_root_field(&mut r, &name, tag, &mut fields)?;
        }

        let metadata = fields.metadata.ok_or(Error::MissingField {
            field: "Metadata",
            offset: r.pos(),
        })?;
        let regions = fields.regions.ok_or(Error::MissingField {
            field: "Regions",
            offset: r.pos(),
        })?;

        Ok(Document {
            metadata: crate::Metadata {
                name: metadata.name,
                description: metadata.description,
                author: metadata.author,
                minecraft_data_version: fields.data_version,
                version: fields.version,
                sub_version: fields.sub_version,
                region_count: metadata.region_count,
                total_volume: metadata.total_volume,
                total_blocks: metadata.total_blocks,
                time_created: metadata.time_created,
                time_modified: metadata.time_modified,
                enclosing_size: metadata.enclosing_size,
            },
            regions,
        })
    }
}

/// Decodes one root field into `fields`, skipping unknown ones.
fn parse_root_field<'a>(
    r: &mut Reader<'a>,
    field: &str,
    tag: u8,
    fields: &mut RootFields<'a>,
) -> Result<()> {
    match field {
        "MinecraftDataVersion" => {
            expect_tag(r, "MinecraftDataVersion", tag, TAG_INT)?;
            fields.data_version = r.i32()?;
        }
        "Version" => {
            expect_tag(r, "Version", tag, TAG_INT)?;
            fields.version = r.i32()?;
        }
        "SubVersion" => {
            expect_tag(r, "SubVersion", tag, TAG_INT)?;
            fields.sub_version = Some(r.i32()?);
        }
        "Metadata" => {
            expect_tag(r, "Metadata", tag, TAG_COMPOUND)?;
            fields.metadata = Some(parse_metadata(r)?);
        }
        "Regions" => {
            expect_tag(r, "Regions", tag, TAG_COMPOUND)?;
            fields.regions = Some(parse_regions(r)?);
        }
        _ => r.skip_value(tag, ROOT_DEPTH)?,
    }
    Ok(())
}

fn parse_metadata(r: &mut Reader<'_>) -> Result<MetadataFields> {
    let mut fields = MetadataFields::default();
    loop {
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let field = r.string()?;
        parse_metadata_field(r, &field, tag, &mut fields)?;
    }
    Ok(fields)
}

/// Decodes one `Metadata` field into `fields`, skipping unknown ones.
fn parse_metadata_field(
    r: &mut Reader<'_>,
    field: &str,
    tag: u8,
    fields: &mut MetadataFields,
) -> Result<()> {
    match field {
        "Name" => {
            expect_tag(r, "Metadata.Name", tag, TAG_STRING)?;
            fields.name = r.string()?;
        }
        "Description" => {
            expect_tag(r, "Metadata.Description", tag, TAG_STRING)?;
            fields.description = r.string()?;
        }
        "Author" => {
            expect_tag(r, "Metadata.Author", tag, TAG_STRING)?;
            fields.author = r.string()?;
        }
        "RegionCount" => {
            expect_tag(r, "Metadata.RegionCount", tag, TAG_INT)?;
            fields.region_count = Some(r.i32()?);
        }
        "TotalVolume" => {
            expect_tag(r, "Metadata.TotalVolume", tag, TAG_INT)?;
            fields.total_volume = Some(i64::from(r.i32()?));
        }
        "TotalBlocks" => fields.total_blocks = Some(read_total_blocks(r, tag)?),
        "TimeCreated" => {
            expect_tag(r, "Metadata.TimeCreated", tag, TAG_LONG)?;
            fields.time_created = r.i64()?;
        }
        "TimeModified" => {
            expect_tag(r, "Metadata.TimeModified", tag, TAG_LONG)?;
            fields.time_modified = r.i64()?;
        }
        "EnclosingSize" => {
            expect_tag(r, "Metadata.EnclosingSize", tag, TAG_COMPOUND)?;
            fields.enclosing_size = Some(parse_vec3(r, "EnclosingSize")?);
        }
        _ => r.skip_value(tag, SCHEMA_DEPTH)?,
    }
    Ok(())
}

/// Reads `TotalBlocks`, which may be written as an Int or as a Long.
fn read_total_blocks(r: &mut Reader<'_>, tag: u8) -> Result<i64> {
    match tag {
        TAG_INT => Ok(i64::from(r.i32()?)),
        TAG_LONG => r.i64(),
        _ => Err(Error::UnexpectedTag {
            field: "Metadata.TotalBlocks",
            tag,
            offset: r.pos(),
        }),
    }
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

fn parse_region<'a>(r: &mut Reader<'a>, name: String, offset: usize) -> Result<Region<'a>> {
    let mut fields = RegionFields::default();
    loop {
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let field = r.string()?;
        parse_region_field(r, &field, tag, &mut fields)?;
    }

    let position = fields.position.ok_or(Error::MissingField {
        field: "Position",
        offset,
    })?;
    let size = fields.size.ok_or(Error::MissingField {
        field: "Size",
        offset,
    })?;
    let palette = fields.palette.ok_or(Error::MissingField {
        field: "BlockStatePalette",
        offset,
    })?;
    let states = fields.states.ok_or(Error::MissingField {
        field: "BlockStates",
        offset,
    })?;
    Region::from_parts(name, position, size, palette, states, fields.states_offset)
}

/// Decodes one region field into `fields`, skipping unknown ones.
fn parse_region_field<'a>(
    r: &mut Reader<'a>,
    field: &str,
    tag: u8,
    fields: &mut RegionFields<'a>,
) -> Result<()> {
    match field {
        "Position" => {
            expect_tag(r, "Position", tag, TAG_COMPOUND)?;
            fields.position = Some(parse_vec3(r, "Position")?);
        }
        "Size" => {
            expect_tag(r, "Size", tag, TAG_COMPOUND)?;
            fields.size = Some(parse_vec3(r, "Size")?);
        }
        "BlockStatePalette" => {
            expect_tag(r, "BlockStatePalette", tag, TAG_LIST)?;
            fields.palette = Some(parse_palette(r)?);
        }
        "BlockStates" => {
            expect_tag(r, "BlockStates", tag, TAG_LONG_ARRAY)?;
            fields.states_offset = r.pos();
            let count = r.count()?;
            // Borrow the payload directly; never copy or expand it.
            fields.states = Some(r.take(count as u64 * 8)?);
        }
        _ => r.skip_value(tag, REGION_DEPTH)?,
    }
    Ok(())
}

/// Parses an `{x, y, z}` compound of i32 values.
///
/// `field` names the compound for error reporting; unknown members are
/// skipped.
fn parse_vec3(r: &mut Reader<'_>, field: &'static str) -> Result<[i32; 3]> {
    let mut values: [Option<i32>; 3] = [None, None, None];
    loop {
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let axis = r.string()?;
        match axis.as_str() {
            "x" => values[0] = Some(read_int(r, field, tag)?),
            "y" => values[1] = Some(read_int(r, field, tag)?),
            "z" => values[2] = Some(read_int(r, field, tag)?),
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
        palette.push(parse_palette_entry(r)?);
    }
    Ok(palette)
}

/// Parses one palette compound: its `Name` and optional `Properties`.
fn parse_palette_entry(r: &mut Reader<'_>) -> Result<BlockState> {
    let offset = r.pos();
    let mut name: Option<String> = None;
    let mut properties = Properties::new();
    loop {
        let tag = r.tag_id()?;
        if tag == TAG_END {
            break;
        }
        let field = r.string()?;
        match field.as_str() {
            "Name" => name = Some(read_string(r, "palette entry Name", tag)?),
            "Properties" => read_properties(r, tag, &mut properties)?,
            _ => r.skip_value(tag, PALETTE_DEPTH)?,
        }
    }
    let name = name.ok_or(Error::MissingField {
        field: "palette entry Name",
        offset,
    })?;
    Ok(BlockState {
        name: name.into(),
        properties,
    })
}

/// Reads a `Properties` compound into `properties`.
fn read_properties(r: &mut Reader<'_>, tag: u8, properties: &mut Properties) -> Result<()> {
    expect_tag(r, "palette entry Properties", tag, TAG_COMPOUND)?;
    loop {
        let value_tag = r.tag_id()?;
        if value_tag == TAG_END {
            break;
        }
        let key = r.string()?;
        let value = read_string(r, "palette property value", value_tag)?;
        properties.insert(Cow::Owned(key), Cow::Owned(value));
    }
    Ok(())
}
