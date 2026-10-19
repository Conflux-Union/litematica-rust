//! Single-pass parsing of the litematic NBT structure into a [`Document`].
//!
//! Each schema node reads itself: [`Document`] owns the root, the shared
//! [`RootFields`] owns the version and metadata fields, and [`Regions`] owns the
//! region collection, which in turn asks [`Region`] to read one region. The
//! only thing a parent supplies is the name and depth its child sits at, so
//! the shape of the file is stated once, next to the type that mirrors it.
//!
//! The decompressed NBT buffer is scanned exactly once with a forward-only
//! [`Reader`]. Only the schema fields known to the litematic format are
//! decoded; everything else (entities, tile entities, pending ticks, preview
//! image data, ...) is skipped by length without allocation.

use crate::document::Document;
use crate::error::{Error, Result};
use crate::nbt::{Decode, FieldOutcome, Reader, SliceSource, Slot, Tag};
use crate::region::Region;
use crate::schema::RootFields;

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
    const TAG: Tag = Tag::Compound;

    /// Reads the root compound's fields.
    fn decode(
        r: &mut Reader<SliceSource<'a>>,
        _field: &'static str,
        enclosing: usize,
    ) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut header = RootFields::new();
        let mut regions = Slot::default();

        while let Some((name, tag)) = r.next_field()? {
            if let FieldOutcome::Deferred = header.read_field(r, &name, tag, depth)? {
                regions.set(Regions::decode(r, "Regions", depth)?);
            }
        }

        let metadata = header.into_metadata(r.pos())?;
        let regions = regions.required("Regions", r.pos())?;
        Ok(Document {
            metadata,
            regions: regions.0,
        })
    }
}

// Regions
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The `Regions` compound: a map from region name to region.
struct Regions<'a>(Vec<Region<'a>>);

impl<'a> Decode<SliceSource<'a>> for Regions<'a> {
    const TAG: Tag = Tag::Compound;

    /// Reads every region in file order, passing each its member name.
    fn decode(
        r: &mut Reader<SliceSource<'a>>,
        _field: &'static str,
        enclosing: usize,
    ) -> Result<Self> {
        let depth = r.nested(enclosing)?;
        let mut regions = Vec::new();
        while let Some((name, tag)) = r.next_field()? {
            let offset = r.pos();
            r.expect("region", tag, Tag::Compound)?;
            regions.push(Region::decode(r, name, offset, depth)?);
        }
        Ok(Self(regions))
    }
}
