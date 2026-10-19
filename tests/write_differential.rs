//! Differential tests for the streaming writer against `rustmatica` 0.5 as
//! an independent oracle.
//!
//! Random layouts are serialized with `litematica::task::writer`; the resulting
//! bytes must be readable by both rustmatica and this crate's own parser,
//! and every consumer must agree on metadata, palette order, bit width and
//! the voxel-by-voxel state stream.

use std::collections::{HashMap, HashSet};

use litematica::knowledge::mc::{BlockPos, BlockSize, BlockState};
use litematica::model::document::Document;
use litematica::task::decompress::gzip_with_limit;
use litematica::task::writer::{write, Layout, WriteOptions};
use rustmatica::{Litematic, Region};

type Schematic = Litematic<CowStrState, CowEntity, CowBlockEntity>;
type OracleRegion = Region<CowStrState, CowEntity, CowBlockEntity>;

use mcdata::{GenericBlockEntity, GenericBlockState, GenericEntity};

type CowStrState = GenericBlockState;
type CowEntity = GenericEntity;
type CowBlockEntity = GenericBlockEntity;

/// Deterministic xorshift64* PRNG (no external crates), as in differential.rs.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        self.next_u64() % n
    }
}

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

/// Table-backed layout, as in write_roundtrip.rs, with a default `palette`.
struct Table {
    size: BlockPos,
    states: Vec<BlockState>,
    grid: Vec<u8>,
}

impl Table {
    fn random(size: BlockPos, distinct: usize, rng: &mut Rng) -> Self {
        let states: Vec<BlockState> = (0..distinct)
            .map(|i| {
                st(
                    if i % 3 == 0 {
                        "minecraft:fuzz_block"
                    } else {
                        "litematica:fuzz"
                    },
                    &[("gen", &i.to_string()), ("case", "w")],
                )
            })
            .collect();
        let mut grid = Vec::with_capacity(BlockSize::of(size).volume() as usize);
        for y in 0..size.y {
            for z in 0..size.z {
                for x in 0..size.x {
                    let _ = (x, y, z);
                    grid.push(if rng.below(4) == 0 {
                        AIR
                    } else {
                        rng.below(distinct as u64) as u8
                    });
                }
            }
        }
        Self { size, states, grid }
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
        std::iter::once(air())
            .chain(self.states.iter().cloned())
            .collect()
    }
}

/// Expected file palette: air at 0, then sorted + deduped table states.
fn expected_file_palette(table: &Table) -> Vec<BlockState> {
    let mut rest: Vec<BlockState> = table.states.clone();
    rest.retain(|s| *s != air());
    rest.sort();
    rest.dedup();
    let mut p = vec![air()];
    p.extend(rest);
    p
}

/// A palette entry as `(name, properties)`, the shape both sides share.
fn read_view(state: &litematica::knowledge::mc::BlockState) -> (String, Vec<(String, String)>) {
    let props = state
        .properties
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    (state.name.to_string(), props)
}

fn sorted_props(state: &BlockState) -> Vec<(String, String)> {
    state
        .properties
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn oracle_sorted_props(state: &GenericBlockState) -> Vec<(String, String)> {
    let mut props: Vec<(String, String)> = state
        .properties
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    props.sort();
    props
}

fn assert_written_file_matches(table: &Table, options: &WriteOptions) {
    let mut out = Vec::new();
    write(table, options, &mut out).expect("serialize");

    // Oracle: rustmatica reads the raw gzip member.
    let oracle: Schematic = Litematic::from_bytes(&out).expect("rustmatica parses our output");
    // Our own reader agrees on the same bytes.
    let nbt = gzip_with_limit(&out, 1 << 30).expect("decompress our output");
    let doc = Document::parse(&nbt).expect("our parse");

    // Metadata.
    assert_eq!(oracle.metadata.name.to_string(), options.name);
    assert_eq!(oracle.metadata.description.to_string(), options.description);
    assert_eq!(oracle.metadata.author.to_string(), options.author);
    assert_eq!(
        oracle.metadata.minecraft_data_version,
        options.minecraft_data_version
    );
    assert_eq!(oracle.metadata.version, 6);
    assert_eq!(oracle.metadata.sub_version, Some(1));
    assert_eq!(
        oracle.metadata.time_created, options.time_created,
        "time created"
    );
    assert_eq!(
        oracle.metadata.time_modified, options.time_modified,
        "time modified"
    );
    assert_eq!(doc.metadata.region_count, Some(1));
    assert_eq!(oracle.regions.len(), 1);
    let non_air = table.grid.iter().filter(|&&i| i != AIR).count() as i64;
    assert_eq!(oracle.total_blocks(), non_air, "total blocks");
    let size = table.size;
    let (sx, sy, sz) = (size.x, size.y, size.z);
    let volume = i64::from(sx) * i64::from(sy) * i64::from(sz);
    assert_eq!(
        doc.metadata.total_volume,
        Some(i32::try_from(volume).expect("volume fits an i32")),
        "total volume"
    );
    assert_eq!(doc.metadata.total_blocks, Some(non_air));
    // The oracle recomputes the bounding box and clamps zero axes to 1.
    let enclosing = oracle.enclosing_size();
    assert_eq!(
        [enclosing.x, enclosing.y, enclosing.z],
        [sx.max(1), sy.max(1), sz.max(1)],
        "oracle enclosing size follows the mod formula"
    );
    // We store the region size verbatim; the field is never validated.
    assert_eq!(doc.metadata.enclosing_size, Some(BlockPos::new(sx, sy, sz)));

    // Region layout.
    let oregion: &OracleRegion = &oracle.regions[0];
    let region = &doc.regions[0];
    assert_eq!(oregion.name.to_string(), options.name);
    assert_eq!(region.name, options.name);
    assert_eq!(
        (oregion.position.x, oregion.position.y, oregion.position.z),
        (0, 0, 0)
    );
    assert_eq!(region.position, BlockPos::new(0, 0, 0));
    assert_eq!(
        (oregion.size.x, oregion.size.y, oregion.size.z),
        (sx, sy, sz)
    );

    // Palette: file order (air first, sorted rest) and bit width.
    let expected = expected_file_palette(table);
    let opalette = oregion.block_palette();
    assert_eq!(opalette.len(), expected.len(), "oracle palette length");
    for (i, (ostate, estate)) in opalette.iter().zip(&expected).enumerate() {
        assert_eq!(ostate.name.to_string(), estate.name, "palette[{i}] name");
        assert_eq!(
            oracle_sorted_props(ostate),
            sorted_props(estate),
            "palette[{i}] properties"
        );
    }
    let got: Vec<_> = region.palette().states().iter().map(read_view).collect();
    let want: Vec<_> = expected
        .iter()
        .map(|s| (s.name.to_string(), sorted_props(s)))
        .collect();
    assert_eq!(got, want, "our palette order");
    let mut n = 2u32;
    while (1u64 << n) < expected.len() as u64 {
        n += 1;
    }
    assert_eq!(region.bits_per_entry(), n);

    // Voxel-by-voxel: both readers must yield the expected state stream.
    let mut ptr_to_idx: HashMap<*const GenericBlockState, u32> =
        HashMap::with_capacity(opalette.len());
    for (i, b) in opalette.iter().enumerate() {
        ptr_to_idx.insert(b as *const _, i as u32);
    }
    let mut ours = region.iter_palette_ids();
    for (i, (&cell, (_pos, block))) in table.grid.iter().zip(oregion.blocks()).enumerate() {
        let expected_id = match cell {
            AIR => 0,
            c => {
                expected[1..]
                    .binary_search(&table.states[c as usize])
                    .ok()
                    .unwrap() as u32
                    + 1
            }
        };
        let oidx = *ptr_to_idx
            .get(&(block as *const _))
            .expect("blocks() yields palette entries");
        assert_eq!(oidx, expected_id, "oracle voxel {i}");
        let midx = ours
            .next()
            .unwrap_or_else(|| panic!("our stream ended early at {i}"));
        assert_eq!(midx, expected_id, "our voxel {i}");
    }
    assert!(
        ours.next().is_none(),
        "our stream is longer than the oracle's"
    );
}

#[test]
fn random_layouts_match_oracle() {
    let mut seed = 0x0ddb_a11a_dead_beef_u64;
    for (sx, sy, sz, distinct) in [
        (5i32, 3i32, 7i32, 5usize),
        (13, 2, 11, 1),
        (1, 1, 1, 1),
        (9, 4, 3, 17),
        (16, 16, 16, 64),
        (2, 2, 2, 8),
        (7, 1, 13, 4),
    ] {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let table = Table::random(BlockPos::new(sx, sy, sz), distinct, &mut Rng::new(seed));
        let mut options = WriteOptions::new("差分 write ✓", 4189);
        options.description = "written by litematica-rust".into();
        options.author = "litematica-rust".into();
        options.time_created = 1_760_000_000_000;
        options.time_modified = 1_760_000_000_001;
        assert_written_file_matches(&table, &options);
    }
}

#[test]
fn palette_boundary_sizes_match_oracle() {
    for distinct in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 33, 64] {
        let table = Table::random(
            BlockPos::new(8, 4, 4),
            distinct,
            &mut Rng::new(0x9e37_79b9_7f4a_7c15 ^ distinct as u64),
        );
        let options = WriteOptions::new(format!("palette {distinct}"), 4189);
        assert_written_file_matches(&table, &options);
    }
}

#[test]
fn zero_axis_layout_matches_oracle() {
    let mut table = Table::random(BlockPos::new(4, 0, 2), 1, &mut Rng::new(1));
    table.grid.clear(); // volume zero: nothing to query
    let options = WriteOptions::new("flat", 4189);
    assert_written_file_matches(&table, &options);
}
