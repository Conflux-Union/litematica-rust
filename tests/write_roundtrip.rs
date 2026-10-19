//! Round-trip and error-path tests for the streaming writer.
//!
//! Layouts are hand-built tables (a grid of palette indices), serialized
//! with `litematica::task::write`, then parsed back with `Document::parse` and
//! checked field by field: metadata, region layout, palette order
//! (air pinned to index 0, rest sorted), and the voxel-by-voxel stream.

use std::collections::HashSet;

use litematica::error::Error;
use litematica::knowledge::mc::{BlockPos, BlockSize, BlockState};
use litematica::model::document::Document;
use litematica::task::decompress::decompress_gzip_with_limit;
use litematica::task::write::{write, Layout, WriteOptions};

fn st(name: &str, props: &[(&str, &str)]) -> BlockState {
    BlockState {
        name: name.to_string().into(),
        properties: props
            .iter()
            .map(|(k, v)| (k.to_string().into(), v.to_string().into()))
            .collect(),
    }
}

fn air() -> BlockState {
    st("minecraft:air", &[])
}

/// A palette entry as `(name, properties)`, the shape both sides share.
fn read_view(state: &BlockState) -> (String, Vec<(String, String)>) {
    let props = state
        .properties
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    (state.name.to_string(), props)
}

fn write_view(state: &BlockState) -> (String, Vec<(String, String)>) {
    read_view(state)
}

const AIR: u8 = 255;

/// A table-backed layout: `grid` holds indices into `states`, `AIR` is air.
struct Table {
    size: BlockPos,
    states: Vec<BlockState>,
    grid: Vec<u8>,
    /// Overrides `palette()` when set; may over-approximate.
    declared: Option<Vec<BlockState>>,
}

impl Table {
    /// Builds a table filled by `fill(pos)` returning a state index or AIR.
    fn filled(size: BlockPos, states: Vec<BlockState>, fill: impl Fn(BlockPos) -> u8) -> Self {
        let mut grid = Vec::with_capacity(BlockSize::of(size).volume() as usize);
        for y in 0..size.y {
            for z in 0..size.z {
                for x in 0..size.x {
                    grid.push(fill(BlockPos::new(x, y, z)));
                }
            }
        }
        Self {
            size,
            states,
            grid,
            declared: None,
        }
    }

    fn voxel_index(&self, pos: BlockPos) -> usize {
        BlockSize::of(self.size).index_of(pos).unwrap() as usize
    }
}

impl Layout for Table {
    fn size(&self) -> BlockSize {
        BlockSize::of(self.size)
    }

    fn block_at(&self, pos: BlockPos) -> Option<BlockState> {
        match self.grid[self.voxel_index(pos)] {
            AIR => None,
            i => Some(self.states[i as usize].clone()),
        }
    }

    fn block_palette(&self) -> HashSet<BlockState> {
        match &self.declared {
            Some(d) => d.iter().cloned().collect(),
            None => std::iter::once(air())
                .chain(self.states.iter().cloned())
                .collect(),
        }
    }
}

/// Serializes `table` and decompresses the output again; parse on demand.
struct RoundTrip {
    nbt: Vec<u8>,
}

impl RoundTrip {
    fn run<L: Layout>(table: &L, opts: &WriteOptions) -> Self {
        let mut out = Vec::new();
        write(table, opts, &mut out).expect("serialize");
        let nbt = decompress_gzip_with_limit(&out, 1 << 30).expect("decompress our own output");
        Self { nbt }
    }

    fn doc(&self) -> Document<'_> {
        Document::parse(&self.nbt).expect("parse our own output")
    }
}

/// Smallest `n >= 2` with `2^n >= len` — the litematic bit-width rule.
fn num_bits(len: usize) -> u32 {
    let mut n = 2u32;
    while (1u64 << n) < len as u64 {
        n += 1;
    }
    n
}

/// The file palette the table must produce: air at 0, rest sorted + deduped.
fn expected_file_palette(table: &Table) -> Vec<BlockState> {
    let mut rest: Vec<BlockState> = table
        .grid
        .iter()
        .filter(|&&i| i != AIR)
        .map(|&i| table.states[i as usize].clone())
        .collect();
    rest.sort();
    rest.dedup();
    let mut p = vec![air()];
    p.extend(rest);
    p
}

fn assert_voxels_match(table: &Table, doc: &Document<'_>) {
    let region = &doc.regions[0];
    let expected_palette = expected_file_palette(table);
    let id_of = |s: &BlockState| -> u32 {
        if s == &air() {
            0
        } else {
            expected_palette[1..].binary_search(s).ok().unwrap() as u32 + 1
        }
    };
    let expect: Vec<u32> = table
        .grid
        .iter()
        .map(|&i| match i {
            AIR => 0,
            i => id_of(&table.states[i as usize]),
        })
        .collect();
    let got: Vec<u32> = region.iter_blocks().collect();
    assert_eq!(got, expect, "voxel stream");

    // Random access must agree with the sequential stream entry by entry.
    let positions = BlockSize::of(table.size).cartesian_positions();
    for (i, pos) in positions.enumerate() {
        assert_eq!(
            region.palette_id_at(pos),
            Some(expect[i]),
            "index at {pos:?}"
        );
        assert_eq!(
            read_view(region.state_at(pos).expect("in-bounds position")),
            write_view(&expected_palette[expect[i] as usize]),
            "state at {pos:?}"
        );
    }
    let outside = BlockPos::new(table.size.x, 0, 0);
    assert_eq!(region.palette_id_at(outside), None);
    assert_eq!(region.state_at(outside), None);
}

#[test]
fn round_trip_full_layout() {
    let states = vec![
        st("minecraft:stone", &[]),
        st(
            "minecraft:redstone_wire",
            &[("east", "side"), ("power", "15")],
        ),
        st("minecraft:oak_log", &[("axis", "y")]),
        st("litematica:x", &[("a", "1"), ("b", "2")]),
    ];
    let size = BlockPos::new(5, 3, 7);
    let mut opts = WriteOptions::new(4189, "多区域 schematic ✓");
    opts.description = "round trip".into();
    opts.author = "litematica-rust".into();
    opts.time_created = 1_700_000_000_123;
    opts.time_modified = 1_700_000_000_456;

    let table = Table::filled(size, states, |pos| {
        ((pos.x + 2 * pos.y + 3 * pos.z) % 4) as u8
    });
    let non_air = table.grid.iter().filter(|&&i| i != AIR).count() as i64;
    let volume = (5 * 3 * 7) as i64;

    let rt = RoundTrip::run(&table, &opts);
    let doc = rt.doc();

    let m = &doc.metadata;
    assert_eq!(m.name, "多区域 schematic ✓");
    assert_eq!(m.description, "round trip");
    assert_eq!(m.author, "litematica-rust");
    assert_eq!(m.minecraft_data_version, 4189);
    assert_eq!(m.version, 6);
    assert_eq!(m.sub_version, Some(1));
    assert_eq!(m.region_count, Some(1));
    assert_eq!(m.total_volume, Some(volume));
    assert_eq!(m.total_blocks, Some(non_air));
    assert_eq!(m.time_created, 1_700_000_000_123);
    assert_eq!(m.time_modified, 1_700_000_000_456);
    assert_eq!(m.enclosing_size, Some(BlockPos::new(5, 3, 7)));

    assert_eq!(doc.regions.len(), 1);
    let region = &doc.regions[0];
    assert_eq!(region.name, "多区域 schematic ✓");
    assert_eq!(region.position, BlockPos::new(0, 0, 0));
    assert_eq!(region.declared_size, BlockPos::new(5, 3, 7));
    assert_eq!(region.volume(), 105);
    let expected_palette = expected_file_palette(&table);
    let got: Vec<_> = region.palette().iter().map(read_view).collect();
    let want: Vec<_> = expected_palette.iter().map(write_view).collect();
    assert_eq!(got, want);
    assert_eq!(region.bits_per_entry(), num_bits(expected_palette.len()));

    // Properties survive keyed by name, in the map's sorted order.
    let wire = region
        .palette()
        .iter()
        .find(|s| s.name == "minecraft:redstone_wire")
        .expect("wire in palette");
    assert_eq!(wire.name, "minecraft:redstone_wire");
    let props: Vec<(&str, &str)> = wire
        .properties
        .iter()
        .map(|(k, v)| (k.as_ref(), v.as_ref()))
        .collect();
    assert_eq!(props, vec![("east", "side"), ("power", "15")]);

    assert_voxels_match(&table, &doc);
}

#[test]
fn round_trip_over_approximated_palette_keeps_unused_entries() {
    let states = vec![st("minecraft:stone", &[])];
    let mut table = Table::filled(BlockPos::new(4, 2, 3), states, |pos| {
        if pos.x % 2 == 0 {
            0
        } else {
            AIR
        }
    });
    table.declared = Some(vec![
        st("minecraft:unused_entry", &[("never", "emitted")]),
        st("minecraft:stone", &[]),
        st("minecraft:stone", &[]), // duplicate collapses
    ]);
    let rt = RoundTrip::run(&table, &WriteOptions::new(4189, "over-approx"));
    let doc = rt.doc();

    let region = &doc.regions[0];
    // air + sorted {stone, unused_entry}; the unused entry travels along.
    assert_eq!(region.palette().len(), 3);
    assert_eq!(read_view(&region.palette()[0]), write_view(&air()));
    assert_eq!(
        read_view(&region.palette()[1]),
        write_view(&st("minecraft:stone", &[]))
    );
    assert_eq!(
        read_view(&region.palette()[2]),
        write_view(&st("minecraft:unused_entry", &[("never", "emitted")]))
    );
    assert_voxels_match(&table, &doc);
}

#[test]
fn round_trip_degenerate_volumes() {
    // A zero axis has no voxels; `EnclosingSize` mirrors `Region.Size`
    // verbatim, zero axes included.
    let table = Table::filled(BlockPos::new(4, 0, 2), vec![], |_| AIR);
    let rt = RoundTrip::run(&table, &WriteOptions::new(4189, "flat"));
    let doc = rt.doc();

    let region = &doc.regions[0];
    assert_eq!(region.volume(), 0);
    assert_eq!(region.palette().len(), 1);
    assert_eq!(read_view(&region.palette()[0]), write_view(&air()));
    assert_eq!(region.iter_blocks().len(), 0);
    assert_eq!(doc.metadata.total_blocks, Some(0));
    assert_eq!(doc.metadata.total_volume, Some(0));
    assert_eq!(doc.metadata.enclosing_size, Some(BlockPos::new(4, 0, 2)));

    // All air still uses the 2-bit floor rather than a zero-width palette.
    let table = Table::filled(BlockPos::new(3, 3, 3), vec![], |_| AIR);
    let rt = RoundTrip::run(&table, &WriteOptions::new(4189, "air only"));
    let doc = rt.doc();
    let region = &doc.regions[0];
    assert_eq!(region.bits_per_entry(), 2);
    assert_eq!(region.palette().len(), 1);
    assert_eq!(read_view(&region.palette()[0]), write_view(&air()));
    assert_eq!(doc.metadata.total_blocks, Some(0));
}

/// A lazy layout answering every query with one state; for overflow tests
/// whose sizes must never be materialized.
struct Sparse {
    size: BlockPos,
    state: BlockState,
}

impl Layout for Sparse {
    fn size(&self) -> BlockSize {
        BlockSize::of(self.size)
    }

    fn block_at(&self, _pos: BlockPos) -> Option<BlockState> {
        Some(self.state.clone())
    }

    fn block_palette(&self) -> HashSet<BlockState> {
        HashSet::from([self.state.clone()])
    }
}

fn sparse(size: BlockPos) -> Sparse {
    Sparse {
        size,
        state: st("minecraft:stone", &[]),
    }
}

#[test]
fn write_rejections() {
    // A state outside the declared palette reports its position.
    let states = vec![st("minecraft:stone", &[]), st("litematica:missing", &[])];
    let mut table = Table::filled(BlockPos::new(2, 1, 2), states, |pos| (pos.x % 2) as u8);
    table.declared = Some(vec![st("minecraft:stone", &[])]); // misses index 1
    let err = write(&table, &WriteOptions::new(4189, "bad palette"), Vec::new()).unwrap_err();
    match err {
        Error::BlockNotInPalette { state, pos } => {
            assert_eq!(state.name, "litematica:missing");
            assert_eq!(pos, BlockPos::new(1, 0, 0));
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // A volume beyond the format's i32 limit and a negative axis.
    let err = write(
        &sparse(BlockPos::new(65536, 65536, 1)),
        &WriteOptions::new(4189, "huge"),
        Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::DimensionOverflow { .. }));
}

#[test]
fn parsed_region_serializes_itself() {
    let states = vec![st("minecraft:stone", &[]), st("minecraft:glass", &[])];
    let size = BlockPos::new(3, 2, 4);
    let table = Table::filled(size, states, |pos| ((pos.x + pos.z) % 2) as u8);
    let source = RoundTrip::run(&table, &WriteOptions::new(4189, "source"));
    let doc = source.doc();

    let mut out = Vec::new();
    let opts = WriteOptions::new(4189, "source");
    write(&doc.regions[0], &opts, &mut out).expect("rewrite a parsed region");
    let nbt = decompress_gzip_with_limit(&out, 1 << 30).expect("decompress");
    let copy = Document::parse(&nbt).expect("parse the rewrite");

    assert_eq!(copy.regions.len(), 1);
    let region = &copy.regions[0];
    assert_eq!(region.dims(), BlockSize::of(size));
    assert_eq!(copy.metadata.total_blocks, doc.metadata.total_blocks);
    assert_eq!(
        region.iter_blocks().count(),
        BlockSize::of(size).volume() as usize
    );
    // The palette is rewritten canonically, so indices may move; states must not.
    for pos in BlockSize::of(size).cartesian_positions() {
        assert_eq!(
            read_view(region.state_at(pos).expect("in bounds")),
            read_view(doc.regions[0].state_at(pos).expect("in bounds")),
            "state at {pos:?}"
        );
    }
}

#[test]
fn default_palette_discovers_only_used_states() {
    // A layout that does NOT override block_palette(): the default discovery
    // scan must find exactly the states block_at actually returns.
    struct Bare {
        stone: BlockState,
        glass: BlockState,
    }
    impl Layout for Bare {
        fn size(&self) -> BlockSize {
            BlockSize::new(4, 2, 2)
        }
        fn block_at(&self, pos: BlockPos) -> Option<BlockState> {
            if pos.y % 2 == 0 {
                Some(self.stone.clone())
            } else {
                Some(self.glass.clone())
            }
        }
    }
    let bare = Bare {
        stone: st("minecraft:stone", &[]),
        glass: st("minecraft:glass", &[]),
    };
    let rt = RoundTrip::run(&bare, &WriteOptions::new(4189, "bare"));
    let doc = rt.doc();
    let region = &doc.regions[0];
    let got: Vec<_> = region.palette().iter().map(read_view).collect();
    let want: Vec<_> = [
        air(),
        st("minecraft:glass", &[]),
        st("minecraft:stone", &[]),
    ]
    .iter()
    .map(write_view)
    .collect();
    assert_eq!(got, want);
    assert_eq!(region.bits_per_entry(), 2);
    assert_eq!(doc.metadata.total_blocks, Some(16));
}
