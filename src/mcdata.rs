//! Minecraft data types shared by the reader and the writer.

use std::borrow::Cow;
use std::collections::BTreeMap;

/// The property map of a [`BlockState`]: property name to value.
pub type Properties = BTreeMap<Cow<'static, str>, Cow<'static, str>>;

/// A single block state from a region palette.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockState {
    /// Namespaced block id, e.g. `minecraft:redstone_wire`.
    pub name: Cow<'static, str>,
    /// Block state properties, keyed by property name.
    pub properties: Properties,
}
