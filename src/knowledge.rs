//! The vocabulary every other layer is written in.
//!
//! Knowledge is independent of the litematic format: [`mc`] names the
//! Minecraft concepts a schematic talks about, while the private `nbt` module
//! speaks the tag protocol those concepts are serialized with. Nothing here
//! knows about regions, files, palettes, or tasks.

pub mod mc;

pub(crate) mod nbt;
