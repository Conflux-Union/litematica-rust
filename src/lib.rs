//! Streaming, low-memory parser for Minecraft `.litematic` schematic files.
//!
//! Tree-based parsers (such as rustmatica) materialize the full NBT structure
//! and then expand the packed block data, which peaks at ~15 GiB of RSS for a
//! large schematic (e.g. a 566×256×1423 bridge with 206 million voxels). This
//! crate takes a different route:
//!
//! - the gzip member is decompressed once into a bounded buffer
//!   ([`task::decompress::gzip_with_limit`]);
//! - the NBT payload is scanned by a hand-written forward-only cursor that
//!   never builds a tree ([`model::document::Document::parse`]);
//! - block state palettes are the only block data that is materialized;
//!   the packed `BlockStates` payload is borrowed from the input buffer and
//!   decoded on demand, either voxel by voxel by
//!   [`model::region::Region::iter_palette_ids`] or one entry at a time by
//!   [`model::region::Region::state_at`].
//!
//! Two read modes share that decoder:
//!
//! - [`model::document::Document`] decompresses the whole file once and gives
//!   random access to every region, including
//!   [`model::region::Region::state_at`] in O(1). Memory is the decompressed
//!   payload plus the palettes.
//! - [`task::stream::Reader`] keeps memory bounded by the gzip buffer,
//!   the current palette, and one region's packed payload, independent of the
//!   schematic's volume. NBT field order does not matter: a region's payload
//!   is held only until its other fields have been read, then decoded voxel by
//!   voxel and released. `Metadata` may appear before or after `Regions`, so
//!   [`task::stream::Reader::metadata`] returns `None` until it has been
//!   scanned.
//!
//! Unknown fields (`Entities`, `TileEntities`, pending ticks, preview image
//! data, ...) are skipped in O(length) without allocation, so parsing cost is
//! a single linear scan of the decompressed bytes plus lazy iteration.
//!
//! The crate also writes schematics with the same discipline: a
//! [`task::writer::Layout`] answers point queries (`size`, `block_at`), and
//! [`task::writer::write()`] serializes it
//! as a single-region file straight into any [`std::io::Write`] (file or
//! network stream), with memory bounded by the palette rather than the
//! volume — writing a 566x256x1423 layout (206 million voxels) peaks at a
//! few MiB of RSS.
//!
//! Every parse error reports the byte offset at which it was detected, and
//! hostile inputs (huge claimed sizes, negative lengths, over-deep nesting)
//! are rejected instead of being trusted.
//!
//! # Architecture
//!
//! The crate is split into three layers, each depending only on the ones
//! before it:
//!
//! - knowledge ([`knowledge::mc`] and the private `nbt` protocol): the
//!   Minecraft vocabulary and the tag encoding, with no litematic knowledge;
//! - model ([`model::palette::Palette`], [`model::region::Region`],
//!   [`model::document::Document`]): the litematic types and their invariants,
//!   with no I/O;
//! - tasks ([`model::document::Document::parse`],
//!   [`task::stream::Reader`], [`task::writer`]): byte movement,
//!   decompression, and serialization.
//!
//! Model types are defined once and shared by every task; no task carries a
//! parallel definition of a position, block state, palette, or region. A
//! single [`error::Error`] vocabulary spans all three layers, so every failure
//! is reported in the same terms.
//!
//! # Example
//!
//! ```no_run
//! use litematica::model::document::Document;
//! use litematica::task::decompress::gzip_with_limit;
//!
//! # fn main() -> Result<(), litematica::error::Error> {
//! let raw = std::fs::read("bridge.litematic")?;
//! let nbt = gzip_with_limit(&raw, 512 * 1024 * 1024)?;
//! let doc = Document::parse(&nbt)?;
//!
//! println!("{} region(s), {} voxels total", doc.regions.len(), doc.volume());
//! for region in &doc.regions {
//!     for index in region.iter_palette_ids() {
//!         // Indices are not validated against the palette; bounds-check.
//!         if let Some(state) = region.palette().states().get(index as usize) {
//!             // inspect `state.name` and `state.properties`
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_code)]

pub mod error;
pub mod knowledge;
pub mod model;
pub mod task;

/// Compiles every example in `README.md` as a doctest, so the documented API
/// cannot drift from the real one.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeExamples;
