//! Parse and fully iterate a real `.litematic` file.
//!
//! Ignored by default: point the `LITEMATICA_TEST_FILE` environment variable
//! at a real file to run it. Without the variable the test returns silently.
//! Set `LITEMATICA_DIFF_RUSTMATICA=1` to additionally run a full
//! voxel-by-voxel comparison against rustmatica — this materializes the whole
//! schematic in memory (~15 GiB for a large file) and is intended for manual
//! runs only.
//!
//! ```text
//! LITEMATICA_TEST_FILE=/path/to/file.litematic \
//!     cargo test --release --test real_file -- --ignored --nocapture
//! ```

use std::collections::HashMap;
use std::time::Instant;

use litematica::{decompress_gzip_with_limit, Document};

#[test]
#[ignore = "set LITEMATICA_TEST_FILE to a real .litematic file to run this"]
fn real_file_parse_and_iterate() {
    let Some(path) = std::env::var_os("LITEMATICA_TEST_FILE") else {
        eprintln!("LITEMATICA_TEST_FILE not set; skipping");
        return;
    };
    let raw = std::fs::read(&path).expect("read test file");

    let t0 = Instant::now();
    let nbt = decompress_gzip_with_limit(&raw, 8 << 30).expect("decompress");
    let t1 = Instant::now();
    let doc = Document::parse(&nbt).expect("parse");
    let t2 = Instant::now();

    let mut volume = 0u64;
    for region in &doc.regions {
        for _ in region.iter_blocks() {
            volume += 1;
        }
    }
    let t3 = Instant::now();

    println!("file={path:?}");
    println!(
        "bytes_in={} bytes_out={} regions={}",
        raw.len(),
        nbt.len(),
        doc.regions.len()
    );
    println!(
        "volume={} total_volume={} declared_total_volume={:?}",
        volume,
        doc.total_volume(),
        doc.metadata.total_volume
    );
    println!(
        "wall_ms_decompress={} wall_ms_parse={} wall_ms_iterate={}",
        (t1 - t0).as_millis(),
        (t2 - t1).as_millis(),
        (t3 - t2).as_millis()
    );
    assert_eq!(volume, doc.total_volume());

    if std::env::var_os("LITEMATICA_DIFF_RUSTMATICA").is_some() {
        diff_against_rustmatica(&raw, &doc);
    }
}

/// Full per-voxel comparison against the rustmatica oracle. Memory hungry:
/// rustmatica expands the entire bit stream into `Vec<usize>`.
fn diff_against_rustmatica(raw: &[u8], doc: &Document<'_>) {
    use mcdata::GenericBlockState;
    use rustmatica::Litematic;

    let schem: Litematic = Litematic::from_bytes(raw).expect("rustmatica parse");
    assert_eq!(schem.regions.len(), doc.regions.len());

    for oregion in &schem.regions {
        let mregion = doc
            .regions
            .iter()
            .find(|r| r.name == *oregion.name)
            .unwrap_or_else(|| panic!("region {} missing in our parse", oregion.name));
        let palette = oregion.block_palette();
        let mut ptr_to_idx: HashMap<*const GenericBlockState, u32> =
            HashMap::with_capacity(palette.len());
        for (i, b) in palette.iter().enumerate() {
            ptr_to_idx.insert(b as *const _, i as u32);
        }
        let mut iter = mregion.iter_blocks();
        let mut count = 0u64;
        for (_pos, block) in oregion.blocks() {
            let oidx = *ptr_to_idx
                .get(&(block as *const _))
                .expect("blocks() yields palette entries");
            let midx = iter.next().expect("matching stream length");
            assert_eq!(
                midx, oidx,
                "voxel {count} of region {} differs",
                oregion.name
            );
            count += 1;
        }
        assert!(iter.next().is_none(), "our stream is longer");
        println!(
            "region {}: {} voxels, streams identical",
            oregion.name, count
        );
    }
    println!("rustmatica diff: MATCH");
}
