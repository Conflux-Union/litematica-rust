//! Benchmark: decompress + parse + full iteration of a `.litematic` file.
//!
//! Prints one `key=value` per line, ending with the process peak RSS read
//! from `/proc/self/status` (`VmHWM`).
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example bench -- <file.litematic> [--limit BYTES]
//! ```
//!
//! The default decompression limit is 8 GiB.

use std::time::Instant;

use litematica::{decompress_gzip_with_limit, Document};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Peak resident set size in bytes (Linux only, from VmHWM).
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
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: bench <file.litematic> [--limit BYTES]");
        std::process::exit(2);
    };
    let mut limit: u64 = 8 << 30;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--limit" => {
                let Some(value) = args.next() else {
                    eprintln!("--limit requires a byte count");
                    std::process::exit(2);
                };
                limit = value.parse().unwrap_or_else(|_| {
                    eprintln!("invalid limit: {value}");
                    std::process::exit(2);
                });
            }
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
    }

    let bytes_in = std::fs::read(&path).unwrap_or_else(|e| {
        eprintln!("cannot read {path}: {e}");
        std::process::exit(2);
    });

    let total_start = Instant::now();

    let t0 = Instant::now();
    let nbt = decompress_gzip_with_limit(&bytes_in, limit).unwrap_or_else(|e| {
        eprintln!("decompress failed: {e}");
        std::process::exit(1);
    });
    let wall_decompress = t0.elapsed();

    let t1 = Instant::now();
    let doc = Document::parse(&nbt).unwrap_or_else(|e| {
        eprintln!("parse failed: {e}");
        std::process::exit(1);
    });
    let wall_parse = t1.elapsed();

    let t2 = Instant::now();
    let mut checksum = FNV_OFFSET;
    let mut volume: u64 = 0;
    let mut regions = 0u64;
    for region in &doc.regions {
        regions += 1;
        for index in region.iter_blocks() {
            volume += 1;
            checksum ^= u64::from(index);
            checksum = checksum.wrapping_mul(FNV_PRIME);
        }
    }
    let wall_iterate = t2.elapsed();
    let wall_total = total_start.elapsed();

    println!("bytes_in={}", bytes_in.len());
    println!("bytes_out={}", nbt.len());
    println!("regions={regions}");
    println!("volume={volume}");
    println!("checksum={checksum}");
    println!("wall_ms_decompress={}", wall_decompress.as_millis());
    println!("wall_ms_parse={}", wall_parse.as_millis());
    println!("wall_ms_iterate={}", wall_iterate.as_millis());
    println!("wall_ms_total={}", wall_total.as_millis());
    println!(
        "peak_rss_bytes={}",
        peak_rss_bytes().unwrap_or_else(|| {
            eprintln!("warning: could not read VmHWM, reporting 0");
            0
        })
    );
}
