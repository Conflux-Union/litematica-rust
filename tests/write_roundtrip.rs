//! Round-trip and error-path tests for the streaming writer.
//!
//! Layouts are hand-built tables (a grid of palette indices), serialized
//! with `litematica::write`, then parsed back with `Document::parse` and
//! checked field by field: metadata, region layout, palette order
//! (air pinned to index 0, rest sorted), and the voxel-by-voxel stream.

use std::collections::HashSet;

use litematica::{
    decompress_gzip_with_limit, write, BlockState, Document, Error, Layout, WriteOptions,
};

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

const AIR: u8 = 255;

/// A table-backed layout: `grid` holds indices into `states`, `AIR` is air.
struct Table {
    size: [u32; 3],
    states: Vec<BlockState>,
    grid: Vec<u8>,
    /// Overrides `palette()` when set; may over-approximate.
    declared: Option<Vec<BlockState>>,
}

impl Table {
    /// Builds a table filled by `fill(pos)` returning a state index or AIR.
    fn filled(size: [u32; 3], states: Vec<BlockState>, fill: impl Fn([u32; 3]) -> u8) -> Self {
        let [sx, sy, sz] = size;
        let mut grid = Vec::with_capacity((sx * sy * sz) as usize);
        for y in 0..sy {
            for z in 0..sz {
                for x in 0..sx {
                    grid.push(fill([x, y, z]));
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

    fn voxel_index(&self, [x, y, z]: [u32; 3]) -> usize {
        let [sx, _, sz] = self.size;
        ((y * sz + z) * sx + x) as usize
    }
}

impl Layout for Table {
    fn size(&self) -> [u32; 3] {
        self.size
    }

    fn block_at(&self, pos: [u32; 3]) -> Option<&BlockState> {
        match self.grid[self.voxel_index(pos)] {
            AIR => None,
            i => Some(&self.states[i as usize]),
        }
    }

    fn palette(&self) -> HashSet<BlockState> {
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
    let size = [5u32, 3, 7];
    let mut opts = WriteOptions::new(4189, "多区域 schematic ✓");
    opts.description = "round trip".into();
    opts.author = "litematica-rust".into();
    opts.time_created = 1_700_000_000_123;
    opts.time_modified = 1_700_000_000_456;

    let table = Table::filled(size, states, |[x, y, z]| ((x + 2 * y + 3 * z) % 4) as u8);
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
    assert_eq!(m.enclosing_size, Some([5, 3, 7]));

    assert_eq!(doc.regions.len(), 1);
    let region = &doc.regions[0];
    assert_eq!(region.name, "多区域 schematic ✓");
    assert_eq!(region.position, [0, 0, 0]);
    assert_eq!(region.size, [5, 3, 7]);
    assert_eq!(region.volume(), 105);
    let expected_palette = expected_file_palette(&table);
    assert_eq!(region.palette(), expected_palette.as_slice());
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
    let mut table = Table::filled(
        [4, 2, 3],
        states,
        |[x, _, _]| if x % 2 == 0 { 0 } else { AIR },
    );
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
    assert_eq!(region.palette()[0], air());
    assert_eq!(region.palette()[1], st("minecraft:stone", &[]));
    assert_eq!(
        region.palette()[2],
        st("minecraft:unused_entry", &[("never", "emitted")])
    );
    assert_voxels_match(&table, &doc);
}

#[test]
fn round_trip_zero_axis_writes_empty_block_states() {
    let table = Table::filled([4, 0, 2], vec![], |_| AIR);
    let rt = RoundTrip::run(&table, &WriteOptions::new(4189, "flat"));
    let doc = rt.doc();

    let region = &doc.regions[0];
    assert_eq!(region.volume(), 0);
    assert_eq!(region.palette(), [air()].as_slice());
    assert_eq!(region.iter_blocks().len(), 0);
    assert_eq!(doc.metadata.total_blocks, Some(0));
    assert_eq!(doc.metadata.total_volume, Some(0));
    // The enclosing-size formula reports 1 for a zero axis.
    assert_eq!(doc.metadata.enclosing_size, Some([4, 1, 2]));
}

#[test]
fn round_trip_all_air_uses_two_bit_entries() {
    let table = Table::filled([3, 3, 3], vec![], |_| AIR);
    let rt = RoundTrip::run(&table, &WriteOptions::new(4189, "air only"));
    let doc = rt.doc();
    let region = &doc.regions[0];
    assert_eq!(region.bits_per_entry(), 2);
    assert_eq!(region.palette(), [air()].as_slice());
    assert_eq!(doc.metadata.total_blocks, Some(0));
}

#[test]
fn write_errors_when_block_not_in_declared_palette() {
    let states = vec![st("minecraft:stone", &[]), st("litematica:missing", &[])];
    let mut table = Table::filled([2, 1, 2], states, |[x, _, _]| (x % 2) as u8);
    table.declared = Some(vec![st("minecraft:stone", &[])]); // misses index 1
    let err = write(&table, &WriteOptions::new(4189, "bad palette"), Vec::new()).unwrap_err();
    match err {
        Error::BlockNotInPalette { name, x, y, z } => {
            assert_eq!(name, "litematica:missing");
            assert_eq!([x, y, z], [1, 0, 0]);
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

/// A lazy layout answering every query with one state; for overflow tests
/// whose sizes must never be materialized.
struct Sparse {
    size: [u32; 3],
    state: BlockState,
}

impl Layout for Sparse {
    fn size(&self) -> [u32; 3] {
        self.size
    }

    fn block_at(&self, _pos: [u32; 3]) -> Option<&BlockState> {
        Some(&self.state)
    }

    fn palette(&self) -> HashSet<BlockState> {
        HashSet::from([self.state.clone()])
    }
}

fn sparse(size: [u32; 3]) -> Sparse {
    Sparse {
        size,
        state: st("minecraft:stone", &[]),
    }
}

#[test]
fn write_errors_when_volume_exceeds_i32() {
    let err = write(
        &sparse([65536, 65536, 1]),
        &WriteOptions::new(4189, "huge"),
        Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::DimensionOverflow { .. }));
}

#[test]
fn write_errors_when_axis_exceeds_i32() {
    let err = write(
        &sparse([0x8000_0000, 1, 1]),
        &WriteOptions::new(4189, "wide"),
        Vec::new(),
    )
    .unwrap_err();
    assert!(matches!(err, Error::DimensionOverflow { .. }));
}

#[test]
fn default_palette_discovers_only_used_states() {
    // A layout that does NOT override palette(): the default discovery
    // scan must find exactly the states block_at actually returns.
    struct Bare {
        stone: BlockState,
        glass: BlockState,
    }
    impl Layout for Bare {
        fn size(&self) -> [u32; 3] {
            [4, 2, 2]
        }
        fn block_at(&self, pos: [u32; 3]) -> Option<&BlockState> {
            if pos[1].is_multiple_of(2) {
                Some(&self.stone)
            } else {
                Some(&self.glass)
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
    assert_eq!(
        region.palette(),
        [
            air(),
            st("minecraft:glass", &[]),
            st("minecraft:stone", &[])
        ]
        .as_slice()
    );
    assert_eq!(region.bits_per_entry(), 2);
    assert_eq!(doc.metadata.total_blocks, Some(16));
}
