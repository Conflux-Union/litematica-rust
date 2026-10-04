//! Full differential run against rustmatica on a real `.litematic` file.
//!
//! First runs this crate end-to-end (decompress + parse + full iteration with
//! an FNV-1a checksum), then — after keeping our document alive so the
//! comparison is truly voxel-by-voxel — parses the same file with rustmatica
//! and iterates it fully. rustmatica expands the whole schematic into memory
//! (~15.5 GiB peak for a large file); that is expected.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example diff_real -- <file.litematic>
//! ```
//!
//! Prints `VERDICT: MATCH` and exits 0 when every voxel agrees, otherwise
//! prints `VERDICT: MISMATCH` and exits 1.

use std::collections::HashMap;
use std::time::Instant;

use litematica::{decompress_gzip_with_limit, Document};
use mcdata::GenericBlockState;
use rustmatica::Litematic;

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Peak resident set size in bytes (Linux only, from VmHWM). Note that
/// `VmHWM` is the process-lifetime maximum, so the value printed after the
/// rustmatica phase is the max over both phases.
fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line
        .strip_prefix("VmHWM:")?
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kb * 1024)
}

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: diff_real <file.litematic>");
        std::process::exit(2);
    };
    let raw = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    });

    // ---- our side ------------------------------------------------------
    let t0 = Instant::now();
    let nbt = decompress_gzip_with_limit(&raw, 8 << 30).expect("our decompress");
    let doc = Document::parse(&nbt).expect("our parse");
    let mut our_checksum = FNV_OFFSET;
    let mut our_volume: u64 = 0;
    for region in &doc.regions {
        for index in region.iter_blocks() {
            our_volume += 1;
            our_checksum ^= u64::from(index);
            our_checksum = our_checksum.wrapping_mul(FNV_PRIME);
        }
    }
    let our_wall = t0.elapsed();
    // VmHWM up to this point covers exactly our phase.
    let our_peak = peak_rss_bytes().unwrap_or(0);
    println!("our_regions={}", doc.regions.len());
    println!("our_volume={our_volume}");
    println!("our_checksum={our_checksum}");
    println!("our_wall_ms={}", our_wall.as_millis());
    println!("our_peak_rss_bytes={our_peak}");

    // ---- oracle side ---------------------------------------------------
    // rustmatica is expected to allocate ~15.5 GiB here; our document stays
    // alive because the comparison below iterates both sides in lockstep.
    let t1 = Instant::now();
    let schem: Litematic = Litematic::from_bytes(&raw).expect("rustmatica parse");
    let mut oracle_checksum = FNV_OFFSET;
    let mut mismatches: u64 = 0;
    let mut first_mismatch: Option<String> = None;
    let mut oracle_volume: u64 = 0;
    for oregion in &schem.regions {
        let Some(mregion) = doc.regions.iter().find(|r| r.name == *oregion.name) else {
            eprintln!("region {} missing in our parse", oregion.name);
            std::process::exit(1);
        };
        let palette = oregion.block_palette();
        let mut ptr_to_idx: HashMap<*const GenericBlockState, u32> =
            HashMap::with_capacity(palette.len());
        for (i, b) in palette.iter().enumerate() {
            ptr_to_idx.insert(b as *const _, i as u32);
        }
        let mut iter = mregion.iter_blocks();
        let mut count: u64 = 0;
        for (_pos, block) in oregion.blocks() {
            let oidx = *ptr_to_idx
                .get(&(block as *const _))
                .expect("blocks() yields palette entries");
            let Some(midx) = iter.next() else {
                mismatches += 1;
                first_mismatch.get_or_insert_with(|| {
                    format!("our stream ended early in region {}", oregion.name)
                });
                break;
            };
            if u64::from(midx) != u64::from(oidx) {
                mismatches += 1;
                first_mismatch.get_or_insert_with(|| {
                    format!(
                        "region {} voxel {count}: ours={} rustmatica={}",
                        oregion.name, midx, oidx
                    )
                });
            }
            oracle_checksum ^= u64::from(oidx);
            oracle_checksum = oracle_checksum.wrapping_mul(FNV_PRIME);
            count += 1;
        }
        if iter.next().is_some() {
            mismatches += 1;
            first_mismatch
                .get_or_insert_with(|| format!("our stream is longer in region {}", oregion.name));
        }
        oracle_volume += count;
    }
    let oracle_wall = t1.elapsed();
    let oracle_peak = peak_rss_bytes().unwrap_or(0);

    println!("rustmatica_regions={}", schem.regions.len());
    println!("rustmatica_volume={oracle_volume}");
    println!("rustmatica_checksum={oracle_checksum}");
    println!("rustmatica_wall_ms={}", oracle_wall.as_millis());
    println!("rustmatica_peak_rss_bytes={oracle_peak}");

    if mismatches == 0 && our_checksum == oracle_checksum && our_volume == oracle_volume {
        println!("VERDICT: MATCH");
    } else {
        if let Some(detail) = first_mismatch {
            eprintln!("first mismatch: {detail}");
        }
        eprintln!("mismatched_voxels={mismatches}");
        println!("VERDICT: MISMATCH");
        std::process::exit(1);
    }
}
