//! The parsed model of a litematic file: its metadata and its regions.
//!
//! A [`Document`] is what the random-access reader produces and what both read
//! tasks describe: the declarations a file makes about itself, plus the region
//! models that carry the block data. This module owns those definitions and
//! the queries derived from them; parsing them from NBT is the job of the
//! reader tasks.
//!
//! Metadata is kept exactly as declared. In particular the file's
//! `EnclosingSize` is a cached summary written by the authoring tool, so it may
//! disagree with the regions; [`Document::enclosing_size`] computes the box
//! the regions actually span instead.

use crate::mc::{BlockPos, BlockSize};
use crate::region::Region;

// Metadata
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// Metadata of a litematic schematic, as declared by the file.
///
/// Optional fields are `None` when absent; string fields default to empty and
/// integer fields to `0`, so a stripped-down file still parses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metadata {
    /// Schematic name (`Metadata.Name`).
    pub name: String,
    /// Free-form description (`Metadata.Description`).
    pub description: String,
    /// Author (`Metadata.Author`).
    pub author: String,
    /// Minecraft data version (`MinecraftDataVersion`), e.g. 4189 for 1.21.4.
    pub minecraft_data_version: i32,
    /// Litematica format version (`Version`), 6 for current files.
    pub version: i32,
    /// Optional litematica format sub-version (`SubVersion`).
    pub sub_version: Option<i32>,
    /// Region count claimed by the file (`Metadata.RegionCount`).
    pub region_count: Option<i32>,
    /// Total volume claimed by the file (`Metadata.TotalVolume`).
    pub total_volume: Option<i64>,
    /// Non-air block count claimed by the file (`Metadata.TotalBlocks`).
    pub total_blocks: Option<i64>,
    /// Creation time in milliseconds since the Unix epoch
    /// (`Metadata.TimeCreated`).
    pub time_created: i64,
    /// Last-modified time in milliseconds since the Unix epoch
    /// (`Metadata.TimeModified`).
    pub time_modified: i64,
    /// Enclosing size claimed by the file (`Metadata.EnclosingSize`).
    pub enclosing_size: Option<BlockPos>,
}

// Document
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A parsed litematic document.
///
/// Regions borrow their packed block data from the buffer that
/// [`Document::parse`] was given; see the method documentation for details.
#[derive(Debug)]
pub struct Document<'a> {
    /// The schematic metadata.
    pub metadata: Metadata,
    /// The regions of the schematic, in file order.
    pub regions: Vec<Region<'a>>,
}

impl Document<'_> {
    /// Total voxel count across all regions (sum of [`Region::volume`]).
    ///
    /// Each region's volume is validated during parsing to fit a `u64`; the
    /// sum saturates rather than wrapping, which in practice can only happen
    /// for inputs far beyond any real file.
    pub fn total_volume(&self) -> u64 {
        self.regions
            .iter()
            .map(Region::volume)
            .fold(0u64, u64::saturating_add)
    }

    /// Size of the axis-aligned box enclosing all regions, per axis.
    ///
    /// The result is a span, so it is non-negative per axis even when regions
    /// extend in negative directions. This is distinct from
    /// [`Metadata::enclosing_size`], which mirrors the value a file declares.
    ///
    /// This replicates the Litematica mod / rustmatica formula: for each
    /// region and axis, `mn = min(pos, pos + size + 1)` and
    /// `mx = max(pos, pos + size - 1)`; the document bounds start at the
    /// origin and fold every region's `mn`/`mx` in, and each axis reports
    /// `|mx - mn + 1|`.
    ///
    /// Returns `None` if the document has no regions.
    pub fn enclosing_size(&self) -> Option<BlockSize> {
        if self.regions.is_empty() {
            return None;
        }
        let (mut lo, mut hi) = (BlockPos::ORIGIN, BlockPos::ORIGIN);
        for region in &self.regions {
            let p = region.position;
            let s = region.declared_size;
            lo = lo.min(p.min(p.saturating_add(s).saturating_add(BlockPos::ONE)));
            hi = hi.max(p.max(p.saturating_add(s).saturating_sub(BlockPos::ONE)));
        }
        Some(BlockSize::of(
            hi.saturating_sub(lo).saturating_add(BlockPos::ONE),
        ))
    }
}
