//! The writer's block and coordinate types.

pub mod util;
use std::borrow::Cow;
use std::collections::BTreeMap;

/// A single block state from a region palette.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockState {
    /// Namespaced block id, e.g. `minecraft:redstone_wire`.
    pub name: Cow<'static, str>,
    /// Block state properties, keyed by property name.
    pub properties: BTreeMap<Cow<'static, str>, Cow<'static, str>>,
}

impl BlockState {
    /// `minecraft:air`, the empty block state; always palette index 0.
    pub const AIR: Self = Self {
        name: Cow::Borrowed("minecraft:air"),
        properties: BTreeMap::new(),
    };

    /// A fresh copy of [`BlockState::AIR`], for callers that prefer a method.
    pub fn air() -> Self {
        Self::AIR.clone()
    }
}
