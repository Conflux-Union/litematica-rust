//! Sequential, bounded-memory reading of `.litematic` files.
//!
//! The counterpart of [`crate::Document`]: instead of borrowing a fully
//! decompressed buffer, a [`StreamReader`] pulls the gzip member through a
//! small buffer and hands out one region at a time. A region's palette is
//! materialized (it is small) and its packed `BlockStates` payload is held
//! only until the region's other fields have been read, then decoded voxel by
//! voxel and released. Peak memory is O(buffer + palette + one region
//! payload), independent of the file's volume.
//!
//! NBT compounds are unordered and real files disagree about field order, so
//! a streaming reader must not assume one. Holding the current region's
//! payload while its header is still arriving is what makes order irrelevant
//! without seeking: `Size`, `Position`, and `BlockStatePalette` may come
//! before or after `BlockStates`, and `Metadata` may trail `Regions`. The
//! cost is that the payload of the region being read is resident; nothing
//! larger is.
//!
//! Random access remains the job of [`crate::Document`]. The two modes share
//! the same NBT decoders, geometry validation, and bit-level cursor, so they
//! agree on every decoded value; only the storage strategy differs.

use std::io::{BufReader, Read};

use flate2::read::GzDecoder;

use crate::error::{Error, Result};
use crate::mc::{BlockPos, BlockSize};
use crate::nbt::{Decode, IoSource, Reader, Source, Tag};
use crate::parse::MetadataFields;
use crate::region::{decode_payload, region_geometry, BlockState, Palette};
use crate::Metadata;

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
    metadata_fields: Option<MetadataFields>,
    metadata: Option<Metadata>,
    minecraft_data_version: i32,
    version: i32,
    sub_version: Option<i32>,
    done: bool,
}

impl<R: Read> std::fmt::Debug for StreamReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamReader")
            .field("metadata", &self.metadata)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}

/// One root-level field consumed by [`StreamReader::read_root_field`].
enum RootField {
    /// The root compound ended.
    End,
    /// `Regions` was reached; its compound has not been entered yet.
    Regions,
    /// A header field was collected or an unknown field was skipped.
    Other,
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
            metadata_fields: None,
            metadata: None,
            minecraft_data_version: 0,
            version: 0,
            sub_version: None,
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
        self.metadata.as_ref()
    }

    /// Advances to the next region, parsing it in full.
    ///
    /// The region's header and packed payload are read before the call
    /// returns, so field order inside the region compound does not matter.
    /// The payload stays with the returned [`StreamRegion`] until it is
    /// dropped. At the end of `Regions`, the remaining root fields are
    /// scanned so that a trailing `Metadata` is still collected.
    pub fn next_region(&mut self) -> Result<Option<StreamRegion>> {
        if self.done {
            return Ok(None);
        }
        let tag = self.reader.tag()?;
        if tag == Tag::End {
            self.finish_root()?;
            self.done = true;
            return Ok(None);
        }
        let name = self.reader.string()?;
        self.reader.expect("region", tag, Tag::Compound)?;
        let region_depth = self.reader.nested(self.regions_depth)?;
        self.read_region(name, region_depth).map(Some)
    }

    /// Reads one region's fields and assembles it from the pieces found.
    ///
    /// `BlockStates` is copied aside rather than decoded immediately, because
    /// the palette and size it depends on may still be ahead in the compound.
    fn read_region(&mut self, name: String, region_depth: usize) -> Result<StreamRegion> {
        let region_offset = self.reader.pos();
        let mut position = None;
        let mut size = None;
        let mut palette = None;
        let mut payload = None;

        loop {
            let tag = self.reader.tag()?;
            if tag == Tag::End {
                break;
            }
            let field = self.reader.string()?;
            match field.as_str() {
                "Position" => {
                    self.reader.expect("Position", tag, Tag::Compound)?;
                    position = Some(BlockPos::decode(
                        &mut self.reader,
                        "Position",
                        region_depth,
                    )?);
                }
                "Size" => {
                    self.reader.expect("Size", tag, Tag::Compound)?;
                    size = Some(BlockPos::decode(&mut self.reader, "Size", region_depth)?);
                }
                "BlockStatePalette" => {
                    self.reader.expect("BlockStatePalette", tag, Tag::List)?;
                    palette = Some(Palette::decode(
                        &mut self.reader,
                        "BlockStatePalette",
                        region_depth,
                    )?);
                }
                "BlockStates" => {
                    self.reader.expect("BlockStates", tag, Tag::LongArray)?;
                    let longs = self.reader.count()?;
                    let bytes = read_payload(&mut self.reader, longs)?;
                    payload.get_or_insert(bytes);
                }
                _ => self.reader.skip_value(tag, region_depth)?,
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
        let payload = payload.ok_or(Error::MissingField {
            field: "BlockStates",
            offset: region_offset,
        })?;

        let geometry = region_geometry(size, &palette, payload.len() as u64, region_offset)?;
        Ok(StreamRegion {
            name,
            position,
            size,
            dims: geometry.dims,
            palette,
            bits: geometry.bits,
            volume: geometry.volume,
            payload,
            blocks_done: false,
        })
    }

    /// Scans root fields until `Regions` opens.
    fn scan_to_regions(&mut self) -> Result<()> {
        loop {
            match self.read_root_field()? {
                RootField::Regions => {
                    self.regions_depth = self.reader.nested(self.root_depth)?;
                    return Ok(());
                }
                RootField::End => {
                    return Err(Error::MissingField {
                        field: "Regions",
                        offset: self.reader.pos(),
                    });
                }
                RootField::Other => {}
            }
        }
    }

    /// Consumes root fields after `Regions`, collecting a trailing header.
    fn finish_root(&mut self) -> Result<()> {
        loop {
            match self.read_root_field()? {
                RootField::End => return Ok(()),
                RootField::Regions => {
                    // A duplicate `Regions` is not part of the schema; skip it.
                    self.reader.skip_value(Tag::Compound, self.root_depth)?;
                }
                RootField::Other => {}
            }
        }
    }

    /// Consumes one root-level field, updating the header as it is found.
    fn read_root_field(&mut self) -> Result<RootField> {
        let tag = self.reader.tag()?;
        if tag == Tag::End {
            return Ok(RootField::End);
        }
        let name = self.reader.string()?;
        match name.as_str() {
            "MinecraftDataVersion" => {
                self.reader.expect("MinecraftDataVersion", tag, Tag::Int)?;
                self.minecraft_data_version = self.reader.i32()?;
                self.refresh_metadata();
            }
            "Version" => {
                self.reader.expect("Version", tag, Tag::Int)?;
                self.version = self.reader.i32()?;
                self.refresh_metadata();
            }
            "SubVersion" => {
                self.reader.expect("SubVersion", tag, Tag::Int)?;
                self.sub_version = Some(self.reader.i32()?);
                self.refresh_metadata();
            }
            "Metadata" => {
                self.reader.expect("Metadata", tag, Tag::Compound)?;
                let fields = MetadataFields::decode(&mut self.reader, "Metadata", self.root_depth)?;
                self.metadata_fields = Some(fields);
                self.refresh_metadata();
            }
            "Regions" => {
                self.reader.expect("Regions", tag, Tag::Compound)?;
                return Ok(RootField::Regions);
            }
            _ => self.reader.skip_value(tag, self.root_depth)?,
        }
        Ok(RootField::Other)
    }

    /// Rebuilds [`StreamReader::metadata`] from every header field seen so far.
    fn refresh_metadata(&mut self) {
        let (data_version, version, sub_version) =
            (self.minecraft_data_version, self.version, self.sub_version);
        let metadata = self
            .metadata_fields
            .as_ref()
            .map(|fields| fields.to_metadata(data_version, version, sub_version));
        self.metadata = metadata;
    }
}

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

// Stream region
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// One region yielded by a [`StreamReader`], fully parsed.
///
/// The header is materialized, palette included, and the packed payload is
/// retained until the region is dropped. Block data is emitted once, in
/// storage order, by [`StreamRegion::read_blocks`].
pub struct StreamRegion {
    name: String,
    position: BlockPos,
    size: BlockPos,
    dims: BlockSize,
    palette: Palette,
    bits: u32,
    volume: u64,
    payload: Vec<u8>,
    blocks_done: bool,
}

impl std::fmt::Debug for StreamRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamRegion")
            .field("name", &self.name)
            .field("position", &self.position)
            .field("size", &self.size)
            .field("palette", &self.palette.states())
            .finish_non_exhaustive()
    }
}

impl StreamRegion {
    /// Name of this region within `Regions`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Position of the region in schematic coordinates.
    pub fn position(&self) -> BlockPos {
        self.position
    }

    /// Size as stored in the file. Components may be negative, which means the
    /// region extends in the negative axis direction; the box actually
    /// addressed is [`StreamRegion::dims`].
    pub fn size(&self) -> BlockPos {
        self.size
    }

    /// The absolute size per axis, with any negative direction folded away.
    pub fn dims(&self) -> BlockSize {
        self.dims
    }

    /// The block state palette in file order; emitted indices refer to it.
    pub fn palette(&self) -> &[BlockState] {
        self.palette.states()
    }

    /// Bits used per block entry in the packed payload.
    pub fn bits_per_entry(&self) -> u32 {
        self.bits
    }

    /// Number of voxels in this region.
    pub fn volume(&self) -> u64 {
        self.volume
    }

    /// Emits every palette index in storage order, calling `f` once per voxel.
    ///
    /// Indices are decoded as they are handed to `f` and are never stored.
    /// Returning an error from `f` stops the read and propagates the error;
    /// the region counts as consumed, so a second call is a no-op.
    pub fn read_blocks(&mut self, f: impl FnMut(u32) -> Result<()>) -> Result<()> {
        if self.blocks_done {
            return Ok(());
        }
        self.blocks_done = true;
        decode_payload(&self.payload, self.bits, self.volume, f)
    }
}
