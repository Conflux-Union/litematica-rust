//! Differential tests against `rustmatica` 0.5 as the oracle.
//!
//! Fixtures are built with rustmatica itself (region + `set_block` +
//! `to_bytes`), then parsed by both this crate and rustmatica. Everything is
//! required to agree: metadata, region layout, palettes (in file order, with
//! properties compared sorted because rustmatica serializes them from a
//! `HashMap`), bit-packing width, and the full voxel-by-voxel index stream.
//!
//! On the oracle side, `blocks()` yields references into its palette, so a
//! pointer -> index map gives its palette indices without re-hashing states.

use std::borrow::Cow;
use std::collections::HashMap;

use litematica::{decompress_gzip_with_limit, Document};
use mcdata::util::BlockPos;
use mcdata::{GenericBlockEntity, GenericBlockState, GenericEntity};
use rustmatica::{Litematic, Region};

type Schematic = Litematic<GenericBlockState, GenericEntity, GenericBlockEntity>;
type OracleRegion = Region<GenericBlockState, GenericEntity, GenericBlockEntity>;

type CowStr = Cow<'static, str>;

/// Deterministic xorshift64* PRNG (no external crates).
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

fn st(name: &str, props: &[(&str, &str)]) -> GenericBlockState {
    GenericBlockState {
        name: CowStr::from(name.to_string()),
        properties: props
            .iter()
            .map(|(k, v)| (CowStr::from(k.to_string()), CowStr::from(v.to_string())))
            .collect(),
    }
}

/// Smallest `n >= 2` with `2^n >= palette_len`, mirroring
/// `rustmatica::Region::num_bits` (region.rs:151-157).
fn oracle_num_bits(palette_len: usize) -> u32 {
    let mut n = 2u32;
    while (1usize << n) < palette_len {
        n += 1;
    }
    n
}

fn sorted_props(state: &litematica::BlockState) -> Vec<(String, String)> {
    let mut props = state.properties.clone();
    props.sort();
    props
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

fn base_schematic(name: String) -> Schematic {
    let mut schem = Litematic::new(name, "differential test fixture", "litematica-rust");
    schem.metadata.minecraft_data_version = 4189;
    schem
}

/// Overwrites every voxel of `region` with a state picked from `states`.
fn fill_region(region: &mut OracleRegion, rng: &mut Rng, states: &[GenericBlockState]) {
    let sx = region.size.x.unsigned_abs() as i32;
    let sy = region.size.y.unsigned_abs() as i32;
    let sz = region.size.z.unsigned_abs() as i32;
    for y in 0..sy {
        for z in 0..sz {
            for x in 0..sx {
                let state = &states[rng.below(states.len() as u64) as usize];
                region.set_block(BlockPos::new(x, y, z), state.clone());
            }
        }
    }
}

fn assert_matches_oracle(schem: &Schematic) {
    let bytes = schem.to_bytes().expect("serialize fixture");
    let nbt = decompress_gzip_with_limit(&bytes, 1 << 40).expect("decompress fixture");
    let doc = Document::parse(&nbt).expect("our parse");
    let oracle: Schematic = Litematic::from_bytes(&bytes).expect("oracle parse");

    // Metadata: all declared fields must round-trip identically.
    assert_eq!(doc.metadata.name, oracle.metadata.name.to_string());
    assert_eq!(
        doc.metadata.description,
        oracle.metadata.description.to_string()
    );
    assert_eq!(doc.metadata.author, oracle.metadata.author.to_string());
    assert_eq!(doc.metadata.version, oracle.metadata.version);
    assert_eq!(doc.metadata.sub_version, oracle.metadata.sub_version);
    assert_eq!(
        doc.metadata.minecraft_data_version,
        oracle.metadata.minecraft_data_version
    );
    assert_eq!(doc.metadata.time_created, oracle.metadata.time_created);
    assert_eq!(doc.metadata.time_modified, oracle.metadata.time_modified);

    // Values written by rustmatica's serializer.
    assert_eq!(doc.metadata.region_count, Some(schem.regions.len() as i32));
    assert_eq!(doc.metadata.total_blocks, Some(oracle.total_blocks()));
    assert_eq!(
        doc.metadata.total_volume,
        Some(i64::from(oracle.total_volume()))
    );
    let enclosing = oracle.enclosing_size();
    let enclosing = [enclosing.x, enclosing.y, enclosing.z];
    assert_eq!(doc.metadata.enclosing_size, Some(enclosing));

    // Computed document values must match the oracle's formulas.
    assert_eq!(
        doc.total_volume(),
        oracle
            .regions
            .iter()
            .map(|r| u64::from(r.size.volume()))
            .sum::<u64>()
    );
    assert_eq!(doc.enclosing_size(), Some(enclosing));

    // Regions: same set of names, same layout, same palettes, same streams.
    assert_eq!(doc.regions.len(), oracle.regions.len());
    let ours: HashMap<&str, &litematica::Region> =
        doc.regions.iter().map(|r| (r.name.as_str(), r)).collect();
    assert_eq!(ours.len(), doc.regions.len(), "duplicate region names");

    for oregion in &oracle.regions {
        let name = oregion.name.to_string();
        let mregion = *ours
            .get(name.as_str())
            .unwrap_or_else(|| panic!("region {name} missing in our parse"));
        assert_eq!(
            mregion.position,
            [oregion.position.x, oregion.position.y, oregion.position.z],
            "position of {name}"
        );
        assert_eq!(
            mregion.size,
            [oregion.size.x, oregion.size.y, oregion.size.z],
            "size of {name}"
        );

        let opalette = oregion.block_palette();
        assert_eq!(
            mregion.palette().len(),
            opalette.len(),
            "palette length of {name}"
        );
        for (i, (mstate, ostate)) in mregion.palette().iter().zip(opalette).enumerate() {
            assert_eq!(
                mstate.name,
                ostate.name.to_string(),
                "palette[{i}] name of {name}"
            );
            assert_eq!(
                sorted_props(mstate),
                oracle_sorted_props(ostate),
                "palette[{i}] properties of {name}"
            );
        }
        assert_eq!(
            mregion.bits_per_entry(),
            oracle_num_bits(opalette.len()),
            "bits per entry of {name}"
        );
        assert_eq!(
            mregion.volume(),
            u64::from(oregion.size.volume()),
            "volume of {name}"
        );

        // Voxel-by-voxel index stream comparison.
        let mut ptr_to_idx: HashMap<*const GenericBlockState, u32> =
            HashMap::with_capacity(opalette.len());
        for (i, b) in opalette.iter().enumerate() {
            ptr_to_idx.insert(b as *const _, i as u32);
        }
        let mut ours_iter = mregion.iter_blocks();
        assert_eq!(ours_iter.len() as u64, mregion.volume());
        for (i, (_pos, block)) in oregion.blocks().enumerate() {
            let oidx = *ptr_to_idx
                .get(&(block as *const _))
                .expect("blocks() yields palette entries");
            let midx = ours_iter
                .next()
                .unwrap_or_else(|| panic!("our stream ended early in {name} at voxel {i}"));
            assert_eq!(u64::from(midx), u64::from(oidx), "voxel {i} of {name}");
        }
        assert!(
            ours_iter.next().is_none(),
            "our stream is longer than the oracle's in {name}"
        );
    }
}

#[test]
fn palette_size_sweep() {
    for want in [1usize, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 100] {
        let mut schem = base_schematic(format!("palette sweep {want}"));
        let mut region = Region::new("sweep", BlockPos::new(0, 0, 0), BlockPos::new(16, 2, 8));
        for i in 0..want.saturating_sub(1) {
            let pos = BlockPos::new((i % 16) as i32, ((i / 16) % 2) as i32, (i / 32) as i32);
            region.set_block(
                pos,
                st(
                    "litematica:sweep_block",
                    &[("index", &i.to_string()), ("flavor", "sweep")],
                ),
            );
        }
        assert_eq!(region.block_palette().len(), want);
        schem.regions.push(region);
        assert_matches_oracle(&schem);
    }
}

#[test]
fn odd_dimensions_cross_long_boundaries() {
    let mut seed = 0x1234_5678_9abc_def1u64;
    for (sx, sy, sz) in [
        (5, 3, 7),
        (13, 2, 11),
        (1, 1, 1),
        (9, 4, 3),
        (7, 1, 13),
        (2, 2, 2),
    ] {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut schem = base_schematic(format!("odd {sx}x{sy}x{sz}"));
        let states: Vec<_> = (0..5)
            .map(|i| st("litematica:odd_block", &[("v", &i.to_string())]))
            .collect();
        let mut region = Region::new(
            "odd",
            BlockPos::new(-(sx / 2), 0, -(sz / 2)),
            BlockPos::new(sx, sy, sz),
        );
        fill_region(&mut region, &mut Rng::new(seed), &states);
        schem.regions.push(region);
        assert_matches_oracle(&schem);
    }
}

#[test]
fn multi_region_unicode_names_and_negative_offsets() {
    let mut schem = base_schematic("多区域 schematic ✓".to_string());
    schem.metadata.minecraft_data_version = 2975;

    let states: Vec<_> = (0..4)
        .map(|i| st("litematica:multi", &[("layer", &i.to_string())]))
        .collect();

    let mut r1 = Region::new(
        "主区域",
        BlockPos::new(-10, -5, -7),
        BlockPos::new(-6, 4, -5),
    );
    fill_region(&mut r1, &mut Rng::new(0x0bad_c0de_dead_beef), &states);
    let mut r2 = Region::new(
        "über-region ✓",
        BlockPos::new(3, 64, -100),
        BlockPos::new(8, -3, 2),
    );
    fill_region(&mut r2, &mut Rng::new(0x9e37_79b9_7f4a_7c15), &states);
    let mut r3 = Region::new(
        "région trois",
        BlockPos::new(0, 0, 0),
        BlockPos::new(5, 5, 5),
    );
    fill_region(&mut r3, &mut Rng::new(0x5851_f42d_4c95_7f2d), &states);

    schem.regions.push(r1);
    schem.regions.push(r2);
    schem.regions.push(r3);
    assert_matches_oracle(&schem);
}

#[test]
fn property_states_round_trip() {
    let mut schem = base_schematic("properties".to_string());
    let mut region = Region::new("props", BlockPos::new(0, 0, 0), BlockPos::new(9, 2, 3));
    region.set_block(
        BlockPos::new(0, 0, 0),
        st(
            "minecraft:redstone_wire",
            &[
                ("east", "side"),
                ("north", "none"),
                ("south", "up"),
                ("west", "side"),
                ("power", "15"),
            ],
        ),
    );
    region.set_block(
        BlockPos::new(1, 0, 0),
        st(
            "minecraft:redstone_wire",
            &[
                ("east", "none"),
                ("north", "none"),
                ("south", "side"),
                ("west", "none"),
                ("power", "7"),
            ],
        ),
    );
    for (i, facing) in ["east", "south", "west", "north"].iter().enumerate() {
        region.set_block(
            BlockPos::new(2 + i as i32, 0, 0),
            st("minecraft:furnace", &[("facing", facing), ("lit", "false")]),
        );
    }
    for (i, axis) in ["y", "x", "z"].iter().enumerate() {
        region.set_block(
            BlockPos::new(2 + i as i32, 1, 1),
            st("minecraft:oak_log", &[("axis", axis)]),
        );
    }
    region.set_block(
        BlockPos::new(6, 0, 2),
        st("minecraft:snow", &[("layers", "3")]),
    );
    region.set_block(
        BlockPos::new(7, 0, 2),
        st("minecraft:snow", &[("layers", "8")]),
    );
    schem.regions.push(region);
    assert_matches_oracle(&schem);
}

#[test]
fn zero_axis_regions() {
    let mut schem = base_schematic("zero axis".to_string());
    // No blocks can be set in a region with a zero axis; the palette stays at
    // just air and BlockStates is empty.
    schem.regions.push(Region::new(
        "x zero",
        BlockPos::new(1, 2, 3),
        BlockPos::new(0, 4, 6),
    ));
    schem.regions.push(Region::new(
        "y zero",
        BlockPos::new(-2, -3, -4),
        BlockPos::new(5, 0, 3),
    ));
    schem.regions.push(Region::new(
        "z zero",
        BlockPos::new(7, 8, 9),
        BlockPos::new(5, 4, 0),
    ));
    let mut live = Region::new("live", BlockPos::new(-1, -1, -1), BlockPos::new(3, 3, 3));
    live.set_block(BlockPos::new(1, 1, 1), st("minecraft:stone", &[]));
    schem.regions.push(live);
    assert_matches_oracle(&schem);
}

#[test]
fn fuzz_random_schematics() {
    for case in 0..12u64 {
        let mut rng = Rng::new(0x9e37_79b9_7f4a_7c15 ^ (case + 1));
        let mut schem = base_schematic(format!("fuzz {case}"));
        let region_count = 1 + rng.below(2) as usize;
        for ri in 0..region_count {
            let sx = 1 + rng.below(40) as i32;
            let sy = 1 + rng.below(24) as i32;
            let sz = 1 + rng.below(32) as i32;
            let size = BlockPos::new(
                if rng.below(4) == 0 { -sx } else { sx },
                if rng.below(4) == 0 { -sy } else { sy },
                if rng.below(4) == 0 { -sz } else { sz },
            );
            let position = BlockPos::new(
                rng.below(200) as i32 - 100,
                rng.below(33) as i32 - 16,
                rng.below(200) as i32 - 100,
            );
            let mut region = Region::new(format!("fuzz {case}/{ri}"), position, size);
            let volume = (sx as u64) * (sy as u64) * (sz as u64);
            let distinct = 1 + rng.below(volume.min(63)) as usize;
            let states: Vec<_> = (0..distinct)
                .map(|i| {
                    st(
                        "litematica:fuzz_block",
                        &[("gen", &format!("{case}-{i}")), ("case", &case.to_string())],
                    )
                })
                .collect();
            fill_region(&mut region, &mut rng, &states);
            schem.regions.push(region);
        }
        assert_matches_oracle(&schem);
    }
}
