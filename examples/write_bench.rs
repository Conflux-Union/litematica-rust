//! Benchmark: stream-write a large procedurally generated layout.
//!
//! Builds a hollow sphere inside the same envelope as the README's real
//! bridge file (566x256x1423 = 206M voxels) and serializes it without any
//! materialized volume, demonstrating that writer memory is O(palette) at
//! any scale. Prints one `key=value` per line, ending with the process
//! peak RSS read from `/proc/self/status` (`VmHWM`).
//!
//! Usage:
//!
//! ```text
//! cargo run --release --example write_bench -- [out.litematic]
//! ```

use std::time::Instant;

use litematica::knowledge::mc::{BlockPos, BlockSize, BlockState};
use litematica::task::write::{write, Layout, WriteOptions};

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

/// A hollow sphere inside an air box; blocks are computed per query, so
/// nothing about the volume is ever stored.
struct HollowSphere {
    size: BlockPos,
    shell: BlockState,
    filler: BlockState,
}

impl Layout for HollowSphere {
    fn size(&self) -> BlockSize {
        BlockSize::of(self.size)
    }

    fn block_at(&self, pos: BlockPos) -> Option<BlockState> {
        let size = self.size;
        let c = [
            f64::from(size.x - 1) / 2.0,
            f64::from(size.y - 1) / 2.0,
            f64::from(size.z - 1) / 2.0,
        ];
        let d = (f64::from(pos.x) - c[0])
            .hypot(f64::from(pos.y) - c[1])
            .hypot(f64::from(pos.z) - c[2]);
        let r = c[0].min(c[1]).min(c[2]);
        if d > r {
            None
        } else if d >= r - 3.0 {
            Some(self.shell.clone())
        } else {
            Some(self.filler.clone())
        }
    }
}

fn main() -> Result<(), litematica::error::Error> {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "write_bench.litematic".to_string());

    let layout = HollowSphere {
        size: BlockPos::new(566, 256, 1423),
        shell: "minecraft:stone".into(),
        filler: "minecraft:glass".into(),
    };
    let options = WriteOptions::new(4189, "write bench");

    let t0 = Instant::now();
    let file = std::fs::File::create(&out_path)?;
    let bytes = write(&layout, &options, file)
        .map(|_| std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0))?;
    let t1 = Instant::now();

    println!("out_path={out_path}");
    println!("voxels={}", 566u64 * 256 * 1423);
    println!("bytes_out={bytes}");
    println!("wall_ms_write={}", (t1 - t0).as_millis());
    if let Some(rss) = peak_rss_bytes() {
        println!("peak_rss={rss}");
    }
    Ok(())
}
