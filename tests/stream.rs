//! Streaming-reader tests against hand-built litematic NBT.
//!
//! Fixtures are assembled byte by byte (as in `hostile.rs`) rather than
//! written by this crate, so the tests can vary field order freely. NBT
//! compounds are unordered and real files disagree about placement, so every
//! combination of "metadata before or after regions" and "block data before
//! or after its header" is exercised.
//!
//! Each case compares the streaming path with `Document`, which shares the
//! same schema decoders but keeps the whole payload for random access, so the
//! two modes are checked against each other as well as against the fixture.

use std::io::{Cursor, Write as _};

use flate2::{write::GzEncoder, Compression};
use litematica::knowledge::mc::{BlockPos, BlockSize};
use litematica::model::document::Document;
use litematica::task::stream::StreamReader;

// NBT fixture builder
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

fn u16b(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn i32b(v: i32) -> [u8; 4] {
    v.to_be_bytes()
}

fn i64b(v: i64) -> [u8; 8] {
    v.to_be_bytes()
}

/// u16 length prefix plus UTF-8 bytes, as NBT writes strings.
fn name_bytes(s: &str) -> Vec<u8> {
    let mut v = u16b(s.len() as u16).to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// A full named-tag header plus a raw payload.
fn field(tag: u8, field_name: &str, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![tag];
    v.extend(name_bytes(field_name));
    v.extend_from_slice(payload);
    v
}

fn string_payload(s: &str) -> Vec<u8> {
    name_bytes(s)
}

fn vec3_payload(x: i32, y: i32, z: i32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend(field(3, "x", &i32b(x)));
    v.extend(field(3, "y", &i32b(y)));
    v.extend(field(3, "z", &i32b(z)));
    v.push(0);
    v
}

type Props<'p> = &'p [(&'p str, &'p str)];
type Entry<'p> = (&'p str, Props<'p>);

fn palette_payload(entries: &[Entry<'_>]) -> Vec<u8> {
    let mut v = vec![10u8]; // list element tag: compound
    v.extend(i32b(entries.len() as i32));
    for (name, props) in entries {
        v.extend(field(8, "Name", &string_payload(name)));
        if !props.is_empty() {
            let mut p = Vec::new();
            for (k, value) in *props {
                p.extend(field(8, k, &string_payload(value)));
            }
            p.push(0);
            v.extend(field(10, "Properties", &p));
        }
        v.push(0);
    }
    v
}

fn long_array_payload(longs: &[i64]) -> Vec<u8> {
    let mut v = i32b(longs.len() as i32).to_vec();
    for l in longs {
        v.extend(i64b(*l));
    }
    v
}

/// Smallest `n >= 2` with `2^n >= len`, the litematic width rule.
fn bits_for(len: usize) -> u32 {
    (2..32).find(|n| (1u64 << n) >= len as u64).unwrap_or(32)
}

/// Packs palette indices LSB-first into a big-endian long array, exactly the
/// litematica bit layout.
fn pack(indices: &[u32], nbits: u32) -> Vec<i64> {
    let mut longs = Vec::new();
    let mut acc = 0u64;
    let mut filled = 0u32;
    for &idx in indices {
        for k in 0..nbits {
            acc |= u64::from((idx >> k) & 1) << filled;
            filled += 1;
            if filled == 64 {
                longs.push(acc as i64);
                acc = 0;
                filled = 0;
            }
        }
    }
    if filled > 0 {
        longs.push(acc as i64);
    }
    longs
}

struct RegionSpec<'p> {
    name: &'p str,
    position: BlockPos,
    size: BlockPos,
    palette: Vec<Entry<'p>>,
    indices: Vec<u32>,
}

impl RegionSpec<'_> {
    fn bits(&self) -> u32 {
        bits_for(self.palette.len())
    }

    fn volume(&self) -> u64 {
        BlockSize::of(self.size).volume() as u64
    }
}

/// A region with the usual field order and an unknown member appended to
/// prove it is skipped.
fn region_bytes(spec: &RegionSpec<'_>) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend(field(
        10,
        "Position",
        &vec3_payload(spec.position.x, spec.position.y, spec.position.z),
    ));
    v.extend(field(
        10,
        "Size",
        &vec3_payload(spec.size.x, spec.size.y, spec.size.z),
    ));
    v.extend(field(
        9,
        "BlockStatePalette",
        &palette_payload(&spec.palette),
    ));
    let states = pack(&spec.indices, spec.bits());
    v.extend(field(12, "BlockStates", &long_array_payload(&states)));
    v.extend(field(8, "UnknownRegionMember", &string_payload("ignored")));
    v.push(0);
    v
}

/// A region whose block data precedes the fields needed to decode it.
fn region_bytes_states_first(spec: &RegionSpec<'_>) -> Vec<u8> {
    let states = pack(&spec.indices, spec.bits());
    let mut v = field(12, "BlockStates", &long_array_payload(&states));
    v.extend(field(
        10,
        "Position",
        &vec3_payload(spec.position.x, spec.position.y, spec.position.z),
    ));
    v.extend(field(
        10,
        "Size",
        &vec3_payload(spec.size.x, spec.size.y, spec.size.z),
    ));
    v.extend(field(
        9,
        "BlockStatePalette",
        &palette_payload(&spec.palette),
    ));
    v.push(0);
    v
}

fn regions_field(regions: &[Vec<u8>], names: &[&str]) -> Vec<u8> {
    let mut regs = Vec::new();
    for (name, bytes) in names.iter().zip(regions) {
        regs.extend(field(10, name, bytes));
    }
    regs.push(0);
    field(10, "Regions", &regs)
}

fn metadata_field(specs: &[RegionSpec<'_>]) -> Vec<u8> {
    let volume: u64 = specs.iter().map(RegionSpec::volume).sum();
    let blocks: u64 = specs
        .iter()
        .flat_map(|spec| spec.indices.iter())
        .filter(|&&index| index != 0)
        .count() as u64;
    let mut m = Vec::new();
    m.extend(field(8, "Name", &string_payload("流式 ✓")));
    m.extend(field(8, "Author", &string_payload("tester")));
    m.extend(field(8, "Description", &string_payload("stream fixture")));
    m.extend(field(3, "RegionCount", &i32b(specs.len() as i32)));
    m.extend(field(3, "TotalVolume", &i32b(volume as i32)));
    m.extend(field(4, "TotalBlocks", &i64b(blocks as i64)));
    m.extend(field(4, "TimeCreated", &i64b(1_700_000_000_123)));
    m.extend(field(4, "TimeModified", &i64b(1_700_000_000_456)));
    m.extend(field(10, "EnclosingSize", &vec3_payload(16, 64, 16)));
    m.push(0);
    field(10, "Metadata", &m)
}

fn root(fields: &[Vec<u8>]) -> Vec<u8> {
    let mut v = vec![10u8]; // root TAG_Compound
    v.extend(name_bytes("")); // arbitrary root name
    for f in fields {
        v.extend(f);
    }
    v.push(0);
    v
}

/// A complete decompressed litematic payload with configurable ordering.
fn document_bytes(specs: &[RegionSpec<'_>], regions_first: bool, states_first: bool) -> Vec<u8> {
    let names: Vec<&str> = specs.iter().map(|spec| spec.name).collect();
    let regions: Vec<Vec<u8>> = specs
        .iter()
        .map(|spec| {
            if states_first {
                region_bytes_states_first(spec)
            } else {
                region_bytes(spec)
            }
        })
        .collect();
    let regions = regions_field(&regions, &names);
    let metadata = metadata_field(specs);
    let mut fields = vec![
        field(3, "MinecraftDataVersion", &i32b(4189)),
        field(3, "Version", &i32b(6)),
        field(3, "SubVersion", &i32b(1)),
    ];
    if regions_first {
        fields.push(regions);
        fields.push(metadata);
    } else {
        fields.push(metadata);
        fields.push(regions);
    }
    fields.push(field(8, "UnknownRootMember", &string_payload("ignored")));
    root(&fields)
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).expect("gzip write");
    enc.finish().expect("gzip finish")
}

fn open(bytes: &[u8]) -> StreamReader<Cursor<Vec<u8>>> {
    StreamReader::open(Cursor::new(bytes.to_vec()), 1 << 30).expect("open fixture")
}

// Fixtures
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

fn wire_palette() -> Vec<Entry<'static>> {
    vec![
        ("minecraft:air", &[]),
        (
            "minecraft:redstone_wire",
            &[("east", "side"), ("power", "15")],
        ),
        ("minecraft:oak_log", &[("axis", "y")]),
        ("litematica:x", &[("a", "1"), ("b", "2")]),
    ]
}

/// Three regions: a dense orthogonal one, a negative-size one, and an empty
/// one. Together they cover long-boundary crossing (`bits == 2`), negative
/// axes, and zero volume.
fn three_regions() -> Vec<RegionSpec<'static>> {
    let dense_indices: Vec<u32> = (0..105).map(|i| (i * 7 % 4) as u32).collect();
    let negative_indices: Vec<u32> = (0..24).map(|i| (i % 2) as u32).collect();
    vec![
        RegionSpec {
            name: "dense",
            position: BlockPos::new(0, 0, 0),
            size: BlockPos::new(5, 3, 7),
            palette: wire_palette(),
            indices: dense_indices,
        },
        RegionSpec {
            name: "negative",
            position: BlockPos::new(-4, 64, -3),
            size: BlockPos::new(-4, 2, 3),
            palette: vec![("minecraft:air", &[]), ("minecraft:stone", &[])],
            indices: negative_indices,
        },
        RegionSpec {
            name: "empty",
            position: BlockPos::new(7, 8, 9),
            size: BlockPos::new(4, 0, 2),
            palette: vec![("minecraft:air", &[])],
            indices: Vec::new(),
        },
    ]
}

fn palette_view(
    palette: &[litematica::knowledge::mc::BlockState],
) -> Vec<(String, Vec<(String, String)>)> {
    palette
        .iter()
        .map(|state| {
            let props = state
                .properties
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            (state.name.to_string(), props)
        })
        .collect()
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[test]
fn stream_matches_document_region_by_region() {
    let specs = three_regions();
    let raw = document_bytes(&specs, false, false);
    let doc = Document::parse(&raw).expect("fixture parses with Document");

    let mut reader = open(&gzip(&raw));
    assert_eq!(reader.metadata(), Some(&doc.metadata));

    for (spec, doc_region) in specs.iter().zip(&doc.regions) {
        let region = reader
            .next_region()
            .expect("region header")
            .expect("region present");

        assert_eq!(region.name, spec.name);
        assert_eq!(region.position, spec.position);
        assert_eq!(region.declared_size, spec.size);
        assert_eq!(region.dims(), BlockSize::of(spec.size));
        assert_eq!(region.volume(), spec.volume());
        assert_eq!(region.bits_per_entry(), spec.bits());
        assert_eq!(
            palette_view(region.palette()),
            palette_view(doc_region.palette())
        );

        let streamed: Vec<u32> = region.iter_blocks().collect();
        assert_eq!(streamed, spec.indices, "voxel stream of {}", spec.name);
        let random: Vec<u32> = doc_region.iter_blocks().collect();
        assert_eq!(streamed, random, "stream vs Document for {}", spec.name);
    }

    assert!(reader.next_region().expect("end of regions").is_none());
}

#[test]
fn field_order_is_irrelevant() {
    let specs = three_regions();
    for (regions_first, states_first) in
        [(false, false), (true, false), (false, true), (true, true)]
    {
        let raw = document_bytes(&specs, regions_first, states_first);
        let doc = Document::parse(&raw).expect("fixture parses with Document");
        let mut reader = open(&gzip(&raw));

        if regions_first {
            assert!(
                reader.metadata().is_none(),
                "a trailing Metadata is not known before the regions"
            );
        } else {
            assert_eq!(reader.metadata(), Some(&doc.metadata));
        }

        for (spec, doc_region) in specs.iter().zip(&doc.regions) {
            let region = reader.next_region().expect("region").expect("present");
            assert_eq!(region.position, spec.position);
            assert_eq!(region.declared_size, spec.size);
            assert_eq!(
                palette_view(region.palette()),
                palette_view(doc_region.palette())
            );

            let streamed: Vec<u32> = region.iter_blocks().collect();
            assert_eq!(streamed, spec.indices, "voxel stream of {}", spec.name);
        }

        assert!(reader.next_region().expect("end of regions").is_none());
        assert_eq!(
            reader.metadata(),
            Some(&doc.metadata),
            "metadata is collected after the regions are exhausted"
        );
    }
}

#[test]
fn dropping_an_unread_region_advances_the_stream() {
    let specs = three_regions();
    let raw = document_bytes(&specs, false, false);
    let mut reader = open(&gzip(&raw));

    let first = reader
        .next_region()
        .expect("first header")
        .expect("present");
    assert_eq!(first.name, "dense");
    drop(first);

    let second = reader
        .next_region()
        .expect("second header")
        .expect("present");
    assert_eq!(second.name, "negative");
    let streamed: Vec<u32> = second.iter_blocks().collect();
    assert_eq!(streamed, specs[1].indices);
}

#[test]
fn iterating_a_streamed_region_is_repeatable() {
    let specs = three_regions();
    let mut reader = open(&gzip(&document_bytes(&specs, false, false)));

    let region = reader.next_region().expect("header").expect("present");
    let first: Vec<u32> = region.iter_blocks().collect();
    let second: Vec<u32> = region.iter_blocks().collect();
    assert_eq!(first, specs[0].indices);
    assert_eq!(first, second);
    drop(region);

    let next = reader.next_region().expect("stream continues");
    assert_eq!(next.expect("second region present").name, "negative");
}

#[test]
fn truncated_and_corrupt_inputs_error_without_panicking() {
    let bytes = gzip(&document_bytes(&three_regions(), false, false));
    // The final gzip trailer is never read, so a cut inside it is not an
    // error; every cut inside the deflate stream must be.
    for cut in [0, bytes.len() / 4, bytes.len() / 2, bytes.len() * 3 / 4] {
        let input = bytes[..cut].to_vec();
        let outcome = StreamReader::open(Cursor::new(input), 1 << 30).and_then(|mut reader| {
            while let Some(region) = reader.next_region()? {
                region.iter_blocks().for_each(drop);
            }
            Ok(())
        });
        assert!(outcome.is_err(), "cut at {cut} must fail");
    }

    let garbage = vec![0xFFu8; 64];
    assert!(StreamReader::open(Cursor::new(garbage), 1 << 30).is_err());
}
