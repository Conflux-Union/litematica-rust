//! Sequential, bounded-memory reading of `.litematic` files.
//!
//! The counterpart of [`crate::model::document::Document`]: instead of
//! borrowing a fully decompressed buffer, a [`StreamReader`] pulls the gzip member through a
//! small buffer and hands out one [`Region`] at a time. A region's palette is
//! materialized (it is small) and its packed `BlockStates` payload is held
//! only while that region is alive, then released. Peak memory is
//! O(buffer + palette + one region payload), independent of the file's volume.
//!
//! NBT compounds are unordered and real files disagree about field order, so
//! a streaming reader must not assume one. Holding the current region's
//! payload while its header is still arriving is what makes order irrelevant
//! without seeking: `Size`, `Position`, and `BlockStatePalette` may come
//! before or after `BlockStates`, and `Metadata` may trail `Regions`. The
//! cost is that the payload of the region being read is resident; nothing
//! larger is.
//!
//! Random access remains the job of [`crate::model::document::Document`]. The
//! two modes share the same [`Region`] type, NBT decoders, header accumulators,
//! geometry validation, and bit-level cursor, so they agree on every decoded
//! value; only the storage strategy differs.

use std::io::{BufReader, Read};

use flate2::read::GzDecoder;

use crate::error::{Error, Result};
use crate::knowledge::nbt::{FieldOutcome, IoSource, Reader, Slot, Source, Tag};
use crate::model::document::Metadata;
use crate::model::region::{Region, RegionHeader, RegionParts};
use crate::model::schema::RootFields;

/// A buffered gzip stream, the byte source a [`StreamReader`] consumes.
type GzipSource<R> = IoSource<BufReader<GzDecoder<R>>>;

// Stream reader
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A sequential reader for gzip-compressed `.litematic` files.
///
/// Opening the stream scans root fields until `Regions` is reached, collecting
/// header fields whenever they appear. [`StreamReader::metadata`] returns
/// `None` until `Metadata` has been scanned, which may be after the last
/// region. Each call to [`StreamReader::next_region`] parses one region
/// completely and returns it.
pub struct StreamReader<R: Read> {
    reader: Reader<GzipSource<R>>,
    root_depth: usize,
    regions_depth: usize,
    header: RootFields,
    done: bool,
}

impl<R: Read> std::fmt::Debug for StreamReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamReader")
            .field("metadata", &self.header.metadata())
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

impl<R: Read> StreamReader<R> {
    /// Opens a gzip-compressed litematic and scans the root up to `Regions`.
    ///
    /// `limit` caps the decompressed byte count; exceeding it fails with
    /// [`Error::SizeLimitExceeded`] while inflating, so an untrusted file
    /// cannot force unbounded memory or CPU use. Trailing bytes after the
    /// first gzip member are ignored.
    pub fn open(inner: R, limit: u64) -> Result<Self> {
        let decoder = BufReader::with_capacity(64 * 1024, GzDecoder::new(inner));
        let mut reader = Reader::new(IoSource::new(decoder, limit));

        let tag = reader.tag()?;
        reader.expect("root", tag, Tag::Compound)?;
        // The root compound's name is arbitrary and ignored.
        reader.string()?;
        let root_depth = reader.nested(0)?;

        let mut stream = Self {
            reader,
            root_depth,
            regions_depth: 0,
            header: RootFields::new(),
            done: false,
        };
        stream.scan_to_regions()?;
        Ok(stream)
    }

    /// The header as declared by the file, once its `Metadata` compound has
    /// been scanned.
    ///
    /// Files may place `Metadata` before or after `Regions`; when it trails
    /// the regions, this returns `None` until the stream is exhausted.
    pub fn metadata(&self) -> Option<&Metadata> {
        self.header.metadata()
    }

    /// Advances to the next region, parsing it in full.
    ///
    /// The region's header and packed payload are read before the call
    /// returns, so field order inside the region compound does not matter.
    /// The payload stays with the returned [`Region`] until it is dropped. At
    /// the end of `Regions`, the remaining root fields are scanned so that a
    /// trailing `Metadata` is still collected.
    pub fn next_region(&mut self) -> Result<Option<Region<'static>>> {
        if self.done {
            return Ok(None);
        }
        let Some((name, tag)) = self.reader.next_field()? else {
            self.finish_root()?;
            self.done = true;
            return Ok(None);
        };
        self.reader.expect("region", tag, Tag::Compound)?;
        let region_depth = self.reader.nested(self.regions_depth)?;
        self.read_region(name, region_depth).map(Some)
    }

    /// Reads one region's fields and assembles it from the pieces found.
    ///
    /// `BlockStates` is copied aside rather than decoded immediately, because
    /// the palette and size it depends on may still be ahead in the compound.
    fn read_region(&mut self, name: String, region_depth: usize) -> Result<Region<'static>> {
        let region_offset = self.reader.pos();
        let mut header = RegionHeader::new();
        let mut payload = Slot::default();

        while let Some((field, tag)) = self.reader.next_field()? {
            if let FieldOutcome::Deferred =
                header.read_field(&mut self.reader, &field, tag, region_depth)?
            {
                let longs = self.reader.count()?;
                payload.set(read_payload(&mut self.reader, longs)?);
            }
        }

        let parts: RegionParts = header.into_parts(region_offset)?;
        let payload = payload.required("BlockStates", region_offset)?;
        Region::from_owned(name, parts, payload, region_offset)
    }

    /// Scans root fields until `Regions` opens.
    fn scan_to_regions(&mut self) -> Result<()> {
        loop {
            match self.read_root_field()? {
                Some(FieldOutcome::Deferred) => {
                    self.regions_depth = self.reader.nested(self.root_depth)?;
                    return Ok(());
                }
                Some(FieldOutcome::Value) => {}
                None => {
                    return Err(Error::MissingField {
                        field: "Regions",
                        offset: self.reader.pos(),
                    });
                }
            }
        }
    }

    /// Consumes root fields after `Regions`, collecting a trailing header.
    fn finish_root(&mut self) -> Result<()> {
        loop {
            match self.read_root_field()? {
                None => return Ok(()),
                Some(FieldOutcome::Deferred) => {
                    // A duplicate `Regions` is not part of the schema; skip it.
                    self.reader.skip_value(Tag::Compound, self.root_depth)?;
                }
                Some(FieldOutcome::Value) => {}
            }
        }
    }

    /// Consumes one root-level field, handing it to the shared accumulator.
    fn read_root_field(&mut self) -> Result<Option<FieldOutcome>> {
        let Some((name, tag)) = self.reader.next_field()? else {
            return Ok(None);
        };
        self.header
            .read_field(&mut self.reader, &name, tag, self.root_depth)
            .map(Some)
    }
}

impl<R: Read> Iterator for StreamReader<R> {
    type Item = Result<Region<'static>>;

    /// Yields regions until the end of `Regions`.
    ///
    /// An error ends the iteration: the stream cannot be resumed after a parse
    /// failure, so every later call returns `None`. Once exhausted, a trailing
    /// `Metadata` has been scanned and is available through
    /// [`StreamReader::metadata`].
    fn next(&mut self) -> Option<Self::Item> {
        let region = self.next_region();
        if region.is_err() {
            self.done = true;
        }
        region.transpose()
    }
}

impl<R: Read> std::iter::FusedIterator for StreamReader<R> {}

/// Reads a `BlockStates` long array into owned bytes.
///
/// The buffer grows in chunks and is never pre-allocated at the claimed
/// length, so a hostile count cannot force a huge allocation before the
/// payload (and the decompression limit) proves it real.
fn read_payload<S: Source>(reader: &mut Reader<S>, longs: usize) -> Result<Vec<u8>> {
    const CHUNK: usize = 64 * 1024;
    let total = longs.checked_mul(8).ok_or_else(|| Error::PayloadTooLarge {
        longs,
        offset: reader.pos(),
    })?;
    let mut bytes = Vec::with_capacity(total.min(CHUNK));
    while bytes.len() < total {
        let start = bytes.len();
        bytes.resize((start + CHUNK).min(total), 0);
        reader.read_exact(&mut bytes[start..])?;
    }
    Ok(bytes)
}
