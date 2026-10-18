# litematica

Streaming, low-memory reader and writer for Minecraft [Litematica](https://github.com/masa-tools/litematica)
`.litematic` schematic files (format version 6).

## Why

Tree-based parsers such as [rustmatica](https://crates.io/crates/rustmatica)
deserialize the whole NBT structure and then expand the packed block data into
a `Vec` with one entry per voxel. For a large schematic (for example a
566×256×1423 bridge: 206 million voxels, 232 MB of decompressed NBT) that
peaks at roughly **15.4 GiB of RSS**, which rules them out for server-side
use.

This crate takes a different route:

- the gzip member is inflated once into a caller-bounded buffer;
- the decompressed NBT is scanned by a hand-written, forward-only cursor that
  never builds a tree — unknown fields (`Entities`, `TileEntities`, pending
  ticks, `PreviewImageData`, ...) are skipped by length in O(1);
- only the block state palettes are materialized (they are tiny);
- the packed `BlockStates` payload is borrowed as a `&[u8]` slice from the
  input buffer and decoded voxel by voxel on demand, so block data is never
  copied or expanded.

Peak RSS is therefore roughly *decompressed file size + epsilon*, and parse
errors carry byte offsets for diagnosing hostile files.

## Installation

```toml
[dependencies]
litematica = "0.1"
```

Requires Rust 1.88 or newer.

## Usage

```rust
use litematica::{decompress_gzip_with_limit, Document};

let raw = std::fs::read("bridge.litematic")?;
let nbt = decompress_gzip_with_limit(&raw, 512 * 1024 * 1024)?; // hard cap
let doc = Document::parse(&nbt)?;

println!("{} region(s), {} voxels", doc.regions.len(), doc.total_volume());
for region in &doc.regions {
    for index in region.iter_blocks() {
        // Indices are not validated against the palette; bounds-check.
        if let Some(state) = region.palette().get(index as usize) {
            // inspect `state.name` and `state.properties`
        }
    }
}
```

## Writing

Serialization is declarative: implement `Layout` with two point queries and
the crate streams the file out with memory bounded by the palette, not the
volume — suited to procedural generation and network sinks alike.

```rust
use litematica::mcdata::util::BlockPos;
use litematica::mcdata::BlockState;
use litematica::{write, Layout, WriteOptions};

struct MyLayout {
    stone: BlockState,
}

impl Layout for MyLayout {
    fn size(&self) -> BlockPos {
        BlockPos::new(16, 16, 16)
    }

    fn block_at(&self, pos: BlockPos) -> Option<BlockState> {
        // None means air; return your own state storage.
        (pos.y == 0).then(|| self.stone.clone())
    }
}

# fn main() -> Result<(), litematica::Error> {
let options = WriteOptions::new(4189, "my build"); // data version 4189 = 1.21.4
let mut file = std::io::Cursor::new(Vec::new()); // any `std::io::Write` works
write(&MyLayout { stone: BlockState { name: "minecraft:stone".into(), properties: Default::default() } }, &options, &mut file)?;
# Ok(())
# }
```

Design notes:

- `Layout::block_at` may return `None` (air) or one of the states
  `Layout::block_palette` declares; the default `block_palette`
  implementation discovers it with one extra scan, and over-approximation is
  allowed.
- The writer enforces the format invariants itself: air is pinned to palette
  index 0, duplicate states collapse, and the bit width matches the reader's
  rule. A state not covered by the declared palette fails with
  `Error::BlockNotInPalette` naming the block and position.
- v1 writing scope is a single region without entities or tile entities.

## Format notes

`.litematic` files are gzipped, big-endian Java NBT with a root compound
containing `MinecraftDataVersion`, `Version`, `SubVersion`, a `Metadata`
compound, and a `Regions` compound. Each region carries `Position`,
`Size` (components may be negative), `BlockStatePalette` (a list of
`{ Name, Properties }` compounds), and `BlockStates`, a long array.

The long array is one continuous bit stream: within a big-endian `i64`, bit 0
is the least significant bit, and the first bit in the stream is the lowest
bit of the first entry. Entries use `n` bits each, where `n` is the smallest
value ≥ 2 with `2^n >= palette.len()`; entries may straddle long boundaries.
The voxel at local coordinates `(x, y, z)` is entry
`y*(sx*sz) + z*sx + x` (x fastest, y slowest), which yields exactly
`|size.x| * |size.y| * |size.z|` entries. This matches the bit layout used by
the Litematica mod and rustmatica.

## API overview

- `decompress_gzip_with_limit(input, limit)` — single-member gzip inflation
  with an enforced output cap (fails *while* decompressing, not after).
- `Document::parse(buf)` — one-pass scan; borrows `buf`.
- `Document::total_volume()`, `Document::enclosing_size()` — computed with
  the same formulas as Litematica/rustmatica.
- `Metadata` — name/description/author, data and format versions, declared
  counts and sizes, timestamps.
- `Region` — `name`, `position`, `size` as stored (possibly negative),
  `palette()`, `dims()`, `volume()`, `bits_per_entry()`, and
  `iter_blocks() -> impl Iterator<Item = u32> + ExactSizeIterator`.
- `BlockState` — `name` plus `properties` in file order.
- `Layout` — `size()` + `block_at(pos) -> Option<BlockState>`, plus
  `get_block`, `get_block_or_air`, `contains`, and a default `block_palette()`
  discovery scan; the declarative source for writing.
- `write(layout, &options, sink)` — single-region streaming serialization
  into any `Write`; `WriteOptions` carries name/author/description, the
  Minecraft data version, timestamps, and the gzip level.
- Errors: a `thiserror` enum, parse variants carry byte offsets;
  `Result<T>` alias included.

## Performance

Measured with `examples/bench.rs` on a real 566×256×1423 bridge schematic:
7.4 MiB gzipped, 221 MiB of decompressed NBT, 1 region, 206,187,008 voxels.
Peak RSS is read from `VmHWM`.

| stage | wall time |
| --- | --- |
| decompress (single-member gzip, capped) | 193 ms |
| parse (tree-less scan) | 0 ms (below timer resolution) |
| iterate all 206M voxels | 223 ms |
| total | 417 ms |

Peak RSS: **231 MiB** (242,167,808 bytes) — roughly the decompressed size
plus epsilon.

### Comparison with rustmatica

`examples/diff_real.rs` runs this crate and
[rustmatica](https://crates.io/crates/rustmatica) 0.5.2 side by side on the
same real file and asserts voxel-exact equality: both report 1 region,
volume 206,187,008, and block checksum `11145073651595385200`. Results from
one differential run (same machine):

| | this crate | rustmatica 0.5.2 |
| --- | --- | --- |
| wall time | 411 ms | 10,316 ms (~25× slower) |
| peak RSS | 231 MiB | 16.0 GiB (~71× more) |

A repeat of the differential run on the same machine gave 15.44 GiB /
12.45 s for rustmatica; absolute numbers move a little between runs, but
the gap of roughly two orders of magnitude in both time and memory does
not.

### Writing

`examples/write_bench.rs` serializes a procedurally generated hollow sphere
in the same 566×256×1423 envelope (206,187,008 voxels), then the file is
read back by this crate with exact volume agreement:

| stage | wall time |
| --- | --- |
| stream-write 206M voxels (queries + packing + gzip 6) | 12.2 s |
| read back (decompress + parse + iterate) | 89 ms |

Peak RSS while writing: **2.5 MiB** — the rust runtime baseline plus the
palette. Writer memory is O(palette) regardless of volume.

## Status

Reading is complete and production-ready. Writing is a newer addition:
single-region, declarative `Layout` input, no entities or tile entities yet.

## License

Apache-2.0.
