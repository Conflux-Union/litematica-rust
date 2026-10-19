# litematica

Streaming, low-memory reader and writer for Minecraft `.litematic` schematic files (format version 6).

## Design goals

The crate serves three distinct use cases: pure streaming reads whose memory does not grow with file size, compact random access over decompressed data, and streaming writes from procedural or composite layouts. All three share one set of format semantics and safety boundaries: input is never trusted unconditionally, no read expands the full volume, and errors point at the byte where they were detected.

The design separates where data comes from from how it is stored: the read side promises only to produce the next block or the block at a coordinate, and the write side asks a layout for its bounds, a single block, and the set of states it may produce, so volume never enters the memory budget. Positions, spans, and block states each have a distinct type, and the format invariants live on those types; composite layouts assemble by unioning state sets instead of materializing intermediate results.

## Installation

```sh
cargo add litematica
```

Requires Rust 1.88 or newer.

## Basic types

Three types carry the Minecraft vocabulary. `BlockState` is a namespaced id plus a property map, cheap to clone and deterministically ordered, so it can key a palette. `BlockPos` is a signed world position and supports vector arithmetic, while `BlockSize` is a non-negative cell count per axis, anchored at the origin and half-open, and owns storage-order indexing. Keeping position and span apart stops a signed file size from being used as an index bound, and lets either value move through a query without a conversion type.

```rust
use litematica::knowledge::mc::{BlockPos, BlockSize, BlockState};

let stone: BlockState = "minecraft:stone".into();

let pos = BlockPos::new(3, 2, 5);
let size = BlockSize::of(BlockPos::new(16, -4, 8)); // negative axes fold into spans

assert_eq!(BlockState::AIR.name, "minecraft:air");
assert_eq!(stone.name, "minecraft:stone");
assert_eq!(size.volume(), 16 * 4 * 8);
assert!(size.contains(pos));
assert_eq!(size.index_of(pos), Some((2 * 8 + 5) * 16 + 3));
assert_eq!(size.position_at(0), Some(BlockPos::ORIGIN));
```

## Pure streaming reads

Use `Reader` when the file is much larger than available memory. It inflates the gzip stream through a fixed-size buffer and yields one region at a time, keeping a region's packed payload only until that region is consumed. Peak memory is independent of file size: the buffer, the palette, and one region. NBT field order does not matter; a `Metadata` compound that trails the regions becomes available once the stream is exhausted.

```rust
use litematica::task::stream::Reader;

let file = std::fs::File::open("bridge.litematic")?;
let mut reader = Reader::open(file, 512 * 1024 * 1024)?;

for region in &mut reader {
    let region = region?;
    let air = region
        .iter_states()
        .filter(|state| state.name == "minecraft:air")
        .count();
    println!("{}: {} air of {} voxels", region.name, air, region.volume());
}

if let Some(metadata) = reader.metadata() {
    println!("{}", metadata.name);
}
```

## Compact random access

Use `Document` when the data can reside in memory as a whole but blocks should not be expanded per voxel. It keeps only the decompressed bytes and the palettes, decoding a block at query time, so a lookup does not depend on a region's volume. This suits repeated random access to one schematic; the tradeoff is holding the full decompressed buffer, which must outlive the document.

```rust
use litematica::knowledge::mc::BlockPos;
use litematica::model::document::Document;
use litematica::task::decompress::gzip_with_limit;

let raw = std::fs::read("map.litematic")?;
let nbt = gzip_with_limit(&raw, 512 * 1024 * 1024)?;
let doc = Document::parse(&nbt)?;

for region in &doc.regions {
    let pos = BlockPos::new(3, 4, 5);
    let state = region.state_at(pos);
    println!("{} {:?}", region.name, state);
}

println!("enclosing size: {:?}", doc.bounding_size());
```

## Composite streaming writes

The writer never requires a materialized volume; it treats a `Layout` as a queryable source: declare the bounding box, return a block per coordinate, and declare the set of states that may occur. Procedural generation, overlays, and template stitching all fit the same interface; a composite layout only has to union its children's state sets, after which the writer pulls voxels in storage order and streams compressed output. Peak memory is the palette, independent of volume. Output is currently a single region without entities or block entities.

```rust
use litematica::knowledge::mc::{BlockPos, BlockSize, BlockState};
use litematica::task::writer::{write, Layout, WriteOptions};
use std::collections::HashSet;

/// One layer of stone on the floor, air everywhere else.
struct Floor {
    state: BlockState,
}

impl Layout for Floor {
    fn size(&self) -> BlockSize {
        BlockSize::new(16, 16, 16)
    }

    fn block_at(&self, pos: BlockPos) -> Option<BlockState> {
        (pos.y == 0).then(|| self.state.clone())
    }

    fn block_palette(&self) -> HashSet<BlockState> {
        // Declaring the set analytically skips the full-box discovery scan;
        // over-approximation is allowed.
        HashSet::from([BlockState::AIR, self.state.clone()])
    }
}

fn main() -> litematica::error::Result<()> {
    let layout = Floor {
        state: "minecraft:stone".into(),
    };

    let options = WriteOptions::new("floor", 4189); // 4189 = 1.21.4
    write(&layout, &options, std::fs::File::create("floor.litematic")?)?;
    Ok(())
}
```
