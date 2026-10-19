//! Adversarial-input tests using a hand-written NBT byte builder.
//!
//! Every case must produce an `Err` (or, for the two documented tolerant
//! cases, an `Ok` without panic/OOM). No case may panic or allocate memory
//! proportional to any claimed-but-absent size.

use std::io::{Cursor, Write as _};

use flate2::{write::GzEncoder, Compression};
use litematica::error::Error;
use litematica::knowledge::mc::{BlockPos, BlockSize};
use litematica::model::document::Document;
use litematica::task::decompress::decompress_gzip_with_limit;
use litematica::task::stream::StreamReader;

// ---------------------------------------------------------------- builder

fn u16b(v: u16) -> [u8; 2] {
    v.to_be_bytes()
}

fn i32b(v: i32) -> [u8; 4] {
    v.to_be_bytes()
}

fn i64b(v: i64) -> [u8; 8] {
    v.to_be_bytes()
}

/// u16 length prefix + UTF-8 bytes, as NBT writes strings.
fn name_bytes(s: &str) -> Vec<u8> {
    let mut v = u16b(s.len() as u16).to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// A full NBT named-tag header + raw payload.
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

type PaletteEntry<'p> = (&'static str, &'p [(&'static str, &'static str)]);

fn palette_payload(entries: &[PaletteEntry<'_>]) -> Vec<u8> {
    let mut v = vec![10u8]; // list element tag: compound
    v.extend(i32b(entries.len() as i32));
    for (name, props) in entries {
        v.extend(field(8, "Name", &string_payload(name)));
        if !props.is_empty() {
            let mut p = Vec::new();
            for (k, val) in *props {
                p.extend(field(8, k, &string_payload(val)));
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
    name: &'static str,
    position: BlockPos,
    size: BlockPos,
    palette: Vec<PaletteEntry<'p>>,
    states: Vec<i64>,
}

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
    v.extend(field(12, "BlockStates", &long_array_payload(&spec.states)));
    v.push(0);
    v
}

fn metadata_field() -> Vec<u8> {
    let mut m = Vec::new();
    m.extend(field(8, "Name", &string_payload("hostile")));
    m.extend(field(8, "Author", &string_payload("tester")));
    m.extend(field(8, "Description", &string_payload("minimal")));
    m.extend(field(3, "RegionCount", &i32b(1)));
    m.extend(field(3, "TotalVolume", &i32b(4)));
    m.extend(field(3, "TotalBlocks", &i32b(2))); // the mod writes an Int here
    m.extend(field(4, "TimeCreated", &i64b(1700000000123)));
    m.extend(field(4, "TimeModified", &i64b(1700000000456)));
    m.extend(field(10, "EnclosingSize", &vec3_payload(4, 1, 1)));
    m.push(0);
    field(10, "Metadata", &m)
}

fn regions_field(specs: &[RegionSpec<'_>]) -> Vec<u8> {
    let mut regs = Vec::new();
    for spec in specs {
        regs.extend(field(10, spec.name, &region_bytes(spec)));
    }
    regs.push(0);
    field(10, "Regions", &regs)
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

fn standard_root_fields() -> Vec<Vec<u8>> {
    vec![
        field(3, "MinecraftDataVersion", &i32b(4189)),
        field(3, "Version", &i32b(6)),
        field(3, "SubVersion", &i32b(1)),
        metadata_field(),
    ]
}

fn minimal_valid() -> Vec<u8> {
    let spec = RegionSpec {
        name: "main",
        position: BlockPos::new(0, 0, 0),
        size: BlockPos::new(4, 1, 1),
        palette: vec![("minecraft:air", &[]), ("minecraft:stone", &[])],
        states: pack(&[0, 1, 0, 1], 2),
    };
    let mut fields = standard_root_fields();
    fields.push(regions_field(&[spec]));
    root(&fields)
}

fn unit_region() -> RegionSpec<'static> {
    RegionSpec {
        name: "main",
        position: BlockPos::new(0, 0, 0),
        size: BlockPos::new(1, 1, 1),
        palette: vec![("minecraft:air", &[])],
        states: vec![0],
    }
}

fn parse_err(bytes: &[u8]) -> Error {
    Document::parse(bytes).expect_err("expected parse error")
}

fn gzip(data: &[u8]) -> Vec<u8> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data).expect("gzip write");
    enc.finish().expect("gzip finish")
}

// ------------------------------------------------------------------ tests

#[test]
fn control_minimal_file_parses() {
    let bytes = minimal_valid();
    let doc = Document::parse(&bytes).expect("valid file must parse");
    assert_eq!(doc.metadata.name, "hostile");
    assert_eq!(doc.metadata.author, "tester");
    assert_eq!(doc.metadata.description, "minimal");
    assert_eq!(doc.metadata.version, 6);
    assert_eq!(doc.metadata.sub_version, Some(1));
    assert_eq!(doc.metadata.minecraft_data_version, 4189);
    assert_eq!(doc.metadata.time_created, 1700000000123);
    assert_eq!(doc.metadata.time_modified, 1700000000456);
    assert_eq!(doc.metadata.region_count, Some(1));
    assert_eq!(doc.metadata.total_volume, Some(4));
    assert_eq!(doc.metadata.total_blocks, Some(2));
    assert_eq!(doc.metadata.enclosing_size, Some(BlockPos::new(4, 1, 1)));
    assert_eq!(doc.regions.len(), 1);
    let region = &doc.regions[0];
    assert_eq!(region.name, "main");
    assert_eq!(region.position, BlockPos::new(0, 0, 0));
    assert_eq!(region.declared_size, BlockPos::new(4, 1, 1));
    assert_eq!(region.volume(), 4);
    assert_eq!(region.bits_per_entry(), 2);
    assert_eq!(region.palette().len(), 2);
    assert_eq!(region.palette()[1].name, "minecraft:stone");
    let stream: Vec<u32> = region.iter_blocks().collect();
    assert_eq!(stream, vec![0, 1, 0, 1]);
    assert_eq!(doc.total_volume(), 4);
    assert_eq!(doc.enclosing_size(), Some(BlockSize::new(4, 1, 1)));
}

#[test]
fn every_truncated_prefix_fails() {
    let bytes = minimal_valid();
    for cut in 0..bytes.len() {
        let err = Document::parse(&bytes[..cut]).expect_err("prefix must fail");
        assert!(
            matches!(
                err,
                Error::UnexpectedEof { .. } | Error::InvalidTagId { .. }
            ),
            "cut at {cut}: unexpected error {err:?}"
        );
    }
}

#[test]
fn well_formedness_rejections() {
    // Bogus tag ids at the root and inside a compound; a non-compound root.
    let err = Document::parse(&[13, 0, 0, 0]).expect_err("must fail");
    assert!(matches!(err, Error::InvalidTagId { id: 13, .. }), "{err:?}");

    let mut bytes = vec![10u8];
    bytes.extend(name_bytes(""));
    bytes.push(200);
    let err = Document::parse(&bytes).expect_err("must fail");
    assert!(
        matches!(err, Error::InvalidTagId { id: 200, .. }),
        "{err:?}"
    );

    let mut bytes = vec![3u8];
    bytes.extend(name_bytes(""));
    bytes.extend(i32b(7));
    let err = Document::parse(&bytes).expect_err("must fail");
    assert!(
        matches!(err, Error::UnexpectedTag { field: "root", .. }),
        "{err:?}"
    );

    // Negative lengths: a BlockStates array, an unknown byte array, and an
    // unknown list count.
    let mut region = Vec::new();
    region.extend(field(10, "Position", &vec3_payload(0, 0, 0)));
    region.extend(field(10, "Size", &vec3_payload(4, 1, 1)));
    region.extend(field(
        9,
        "BlockStatePalette",
        &palette_payload(&[("minecraft:air", &[])]),
    ));
    region.extend(field(12, "BlockStates", &i32b(-1)));
    region.push(0);
    let mut fields = standard_root_fields();
    let mut regs = field(10, "main", &region);
    regs.push(0);
    fields.push(field(10, "Regions", &regs));
    let err = parse_err(&root(&fields));
    assert!(
        matches!(err, Error::NegativeLength { length: -1, .. }),
        "{err:?}"
    );

    let mut fields = standard_root_fields();
    fields.push(field(7, "EvilBytes", &i32b(-7)));
    fields.push(regions_field(&[unit_region()]));
    let err = parse_err(&root(&fields));
    assert!(
        matches!(err, Error::NegativeLength { length: -7, .. }),
        "{err:?}"
    );

    let mut fields = standard_root_fields();
    let mut list = vec![10u8];
    list.extend(i32b(-3));
    fields.push(field(9, "Entities", &list));
    fields.push(regions_field(&[unit_region()]));
    let err = parse_err(&root(&fields));
    assert!(
        matches!(err, Error::NegativeLength { length: -3, .. }),
        "{err:?}"
    );

    // A region with volume but an empty palette can never resolve an index.
    let spec = RegionSpec {
        size: BlockPos::new(2, 1, 1),
        palette: vec![],
        ..unit_region()
    };
    let mut fields = standard_root_fields();
    fields.push(regions_field(&[spec]));
    let err = parse_err(&root(&fields));
    assert!(matches!(err, Error::EmptyPalette { .. }), "{err:?}");

    // Missing required fields: root Metadata, root Regions, then every
    // required region member in turn.
    let mut fields = vec![
        field(3, "MinecraftDataVersion", &i32b(4189)),
        field(3, "Version", &i32b(6)),
    ];
    fields.push(regions_field(&[unit_region()]));
    let err = parse_err(&root(&fields));
    assert!(
        matches!(
            err,
            Error::MissingField {
                field: "Metadata",
                ..
            }
        ),
        "{err:?}"
    );

    let err = parse_err(&root(&standard_root_fields()));
    assert!(
        matches!(
            err,
            Error::MissingField {
                field: "Regions",
                ..
            }
        ),
        "{err:?}"
    );

    for missing in ["Position", "Size", "BlockStatePalette", "BlockStates"] {
        let mut region = Vec::new();
        if missing != "Position" {
            region.extend(field(10, "Position", &vec3_payload(0, 0, 0)));
        }
        if missing != "Size" {
            region.extend(field(10, "Size", &vec3_payload(1, 1, 1)));
        }
        if missing != "BlockStatePalette" {
            region.extend(field(
                9,
                "BlockStatePalette",
                &palette_payload(&[("minecraft:air", &[])]),
            ));
        }
        if missing != "BlockStates" {
            region.extend(field(12, "BlockStates", &long_array_payload(&[0])));
        }
        region.push(0);
        let mut fields = standard_root_fields();
        let mut regs = field(10, "main", &region);
        regs.push(0);
        fields.push(field(10, "Regions", &regs));
        let err = parse_err(&root(&fields));
        assert!(
            matches!(err, Error::MissingField { field, .. } if field == missing),
            "missing {missing}: {err:?}"
        );
    }

    // Nesting deeper than the 64-level limit.
    let mut payload = field(3, "leaf", &i32b(1));
    payload.push(0);
    for _ in 0..100 {
        let mut outer = field(10, "n", &payload);
        outer.push(0);
        payload = outer;
    }
    let mut fields = standard_root_fields();
    fields.push(regions_field(&[unit_region()]));
    fields.push(field(10, "Deep", &payload));
    let err = parse_err(&root(&fields));
    assert!(matches!(err, Error::DepthLimitExceeded { .. }), "{err:?}");
}

#[test]
fn malicious_sizes_are_not_trusted() {
    // Declared sizes far beyond the payload available: the short payload is
    // reported as truncation rather than driving allocation.
    for size in [
        BlockPos::new(1_000_000, 1_000_000, 1_000_000),
        BlockPos::new(i32::MAX, i32::MAX, i32::MAX),
        BlockPos::new(i32::MIN, i32::MIN, i32::MIN),
    ] {
        let spec = RegionSpec {
            size,
            palette: vec![("minecraft:air", &[]), ("minecraft:stone", &[])],
            states: vec![5], // one long, far too short
            ..unit_region()
        };
        let mut fields = standard_root_fields();
        fields.push(regions_field(&[spec]));
        let err = parse_err(&root(&fields));
        assert!(
            matches!(err, Error::TruncatedBlockStates { have: 8, .. }),
            "size {size:?}: {err:?}"
        );
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    // Claimed lengths with nothing behind them must fail as EOF, not OOM.
    let mut region = Vec::new();
    region.extend(field(10, "Position", &vec3_payload(0, 0, 0)));
    region.extend(field(10, "Size", &vec3_payload(1, 1, 1)));
    region.extend(field(
        9,
        "BlockStatePalette",
        &palette_payload(&[("minecraft:air", &[])]),
    ));
    region.extend(field(12, "BlockStates", &i32b(i32::MAX)));
    let mut fields = standard_root_fields();
    let mut regs = field(10, "main", &region);
    regs.push(0);
    fields.push(field(10, "Regions", &regs));
    let bytes = root(&fields);
    let err = Document::parse(&bytes[..bytes.len() - 1]).expect_err("must fail");
    assert!(matches!(err, Error::UnexpectedEof { .. }), "{err:?}");

    // A palette list claiming 2^31-1 compound entries with no payload.
    let mut region = Vec::new();
    region.extend(field(10, "Position", &vec3_payload(0, 0, 0)));
    region.extend(field(10, "Size", &vec3_payload(1, 1, 1)));
    let mut list = vec![10u8];
    list.extend(i32b(i32::MAX));
    region.extend(field(9, "BlockStatePalette", &list));
    let mut fields = standard_root_fields();
    let mut regs = field(10, "main", &region);
    regs.push(0);
    fields.push(field(10, "Regions", &regs));
    let bytes = root(&fields);
    // Strip the three trailing END tags (region, regions, root) so the buffer
    // ends right after the palette list header.
    let err = Document::parse(&bytes[..bytes.len() - 3]).expect_err("must fail");
    assert!(matches!(err, Error::UnexpectedEof { .. }), "{err:?}");
}

#[test]
fn decompression_guards() {
    // Not gzip at all.
    let err = decompress_gzip_with_limit(b"definitely not gzip", 1 << 20).expect_err("must fail");
    assert!(matches!(err, Error::CorruptGzip(_)), "{err:?}");

    // Corrupted body of an otherwise valid stream, and a truncated stream.
    let mut gz = gzip(&minimal_valid());
    let mid = gz.len() / 2;
    gz[mid] ^= 0xff;
    let err = decompress_gzip_with_limit(&gz, 1 << 20).expect_err("must fail");
    assert!(matches!(err, Error::CorruptGzip(_)), "{err:?}");

    let gz = gzip(&minimal_valid());
    let err = decompress_gzip_with_limit(&gz[..gz.len() / 2], 1 << 20).expect_err("must fail");
    assert!(matches!(err, Error::CorruptGzip(_)), "{err:?}");

    // Output beyond the limit, including a zero limit; exactly the limit is
    // allowed.
    let data = vec![0u8; 1000];
    let gz = gzip(&data);
    let err = decompress_gzip_with_limit(&gz, 10).expect_err("must fail");
    assert!(
        matches!(err, Error::SizeLimitExceeded { limit: 10 },),
        "{err:?}"
    );
    assert!(err.to_string().contains("10"), "{err}");

    let err = decompress_gzip_with_limit(&gz, 0).expect_err("must fail");
    assert!(
        matches!(err, Error::SizeLimitExceeded { limit: 0 }),
        "{err:?}"
    );

    let out = decompress_gzip_with_limit(&gz, 1000).expect("must succeed");
    assert_eq!(out.len(), 1000);

    // The streaming reader applies the same limit while opening.
    let err = StreamReader::open(Cursor::new(gz), 0).expect_err("limit must bite");
    assert!(
        matches!(err, Error::SizeLimitExceeded { limit: 0 }),
        "{err:?}"
    );
}

#[test]
fn out_of_range_indices_yield_without_panic() {
    let spec = RegionSpec {
        name: "main",
        position: BlockPos::new(0, 0, 0),
        size: BlockPos::new(2, 1, 1),
        palette: vec![("minecraft:air", &[])], // palette len 1 -> nbits 2
        states: pack(&[3, 3], 2),              // index 3 >= palette len
    };
    let mut fields = standard_root_fields();
    fields.push(regions_field(&[spec]));
    let bytes = root(&fields);
    let doc = Document::parse(&bytes).expect("parse must succeed");
    let stream: Vec<u32> = doc.regions[0].iter_blocks().collect();
    assert_eq!(stream, vec![3, 3]);
    // The resolved-state iterator drops identifiers with no palette entry.
    assert_eq!(doc.regions[0].iter_states().count(), 0);
    // Bounds-checking is the caller's job.
    assert!(doc.regions[0].palette().get(3).is_none());
    // The checked accessor treats the dangling index as absent.
    assert_eq!(doc.regions[0].state_at(BlockPos::new(0, 0, 0)), None);
}

#[test]
fn zero_volume_tolerates_empty_palette_and_states() {
    let spec = RegionSpec {
        name: "main",
        position: BlockPos::new(0, 0, 0),
        size: BlockPos::new(0, 3, 4),
        palette: vec![],
        states: vec![],
    };
    let mut fields = standard_root_fields();
    fields.push(regions_field(&[spec]));
    let bytes = root(&fields);
    let doc = Document::parse(&bytes).expect("zero volume must parse");
    assert_eq!(doc.regions[0].volume(), 0);
    assert_eq!(doc.regions[0].iter_blocks().count(), 0);
    assert_eq!(doc.total_volume(), 0);
    assert_eq!(doc.enclosing_size(), Some(BlockSize::new(1, 3, 4)));
}

#[test]
fn parse_error_messages_carry_offsets() {
    let samples: Vec<Error> = vec![
        {
            let mut fields = standard_root_fields();
            let spec = RegionSpec {
                name: "main",
                position: BlockPos::new(0, 0, 0),
                size: BlockPos::new(1000, 1000, 1000),
                palette: vec![("minecraft:air", &[])],
                states: vec![0],
            };
            fields.push(regions_field(&[spec]));
            parse_err(&root(&fields))
        },
        Document::parse(&[13, 0, 0]).expect_err("must fail"),
        Document::parse(&[]).expect_err("must fail"),
    ];
    for err in samples {
        let msg = err.to_string();
        assert!(msg.contains("offset"), "message lacks offset: {msg}");
    }
}
