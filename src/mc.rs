//! The Minecraft vocabulary a schematic is built from: where a block sits and
//! what that block is.
//!
//! [`BlockState`] is one half of that vocabulary: a namespaced block id
//! together with its properties. It is cheap to clone and orders
//! deterministically, which is what makes it usable as a palette key in both
//! the writer and downstream layout combinators.
//!
//! The other half is split by role. [`BlockPos`] is a point in the world, a
//! signed coordinate on every axis; it answers "where". [`BlockSize`] is the
//! span of a box, a non-negative cell count per axis; it answers "how much".
//! A file declares both with the same signed triple, because a region may
//! extend along a negative direction, so [`BlockSize::of`] is the single place
//! where a declaration becomes a span. Keeping them apart means a position can
//! never silently bound an index, and a span can never be mistaken for a point.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// BlockState
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A single block state from a region palette.
///
/// The id is namespaced, as in `minecraft:redstone_wire`. Properties are keyed
/// by name, so equality and ordering are independent of the order a file
/// happened to list them in; this is what lets a palette deduplicate states
/// found anywhere in a volume and still order them deterministically.
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
}

// BlockPos
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// A Minecraft block coordinate: a point in the world, signed per axis.
///
/// This is the "where" of the vocabulary; the span of a box lives in
/// [`BlockSize`] instead. Components are signed because Minecraft's coordinate
/// space is signed and a region's declared size may run along a negative axis.
/// The direction of that run is meaningful here, while [`BlockSize`] folds it
/// away. A position therefore never acts as an index bound on its own.
///
/// Arithmetic follows vector intuition: `+` and `-` are component-wise, `*` and
/// `/` with an `i32` scale every axis, and `*` between two positions is the
/// cross product. [`BlockPos::min`], [`BlockPos::max`] and
/// [`BlockPos::component_mul`] are the other, component-wise readings.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockPos {
    /// The horizontal axis, positive to the east.
    pub x: i32,
    /// The vertical axis, positive upward.
    pub y: i32,
    /// The horizontal axis, positive to the south.
    pub z: i32,
}

impl BlockPos {
    /// The position at (0, 0, 0).
    ///
    /// [`BlockSize`] anchors every box here, so this constant is also the
    /// lower corner shared by all spans.
    pub const ORIGIN: Self = Self::new(0, 0, 0);

    /// Creates a position from its three components.
    ///
    /// Any signed triple is a valid world position. When the value is meant as
    /// the size a file declares, read it through [`BlockSize::of`] instead of
    /// treating the position itself as a bound.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// A position with the absolute value of each component.
    ///
    /// The result always lies in the positive octant. This is a plain
    /// coordinate operation, not the way to obtain a span; use
    /// [`BlockSize::of`] to read a signed size as the box it covers.
    pub const fn abs(&self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
            z: self.z.abs(),
        }
    }

    /// Dot product of `self` and `v`, both read as vectors.
    ///
    /// Layout code uses it to project a position onto a direction and decide
    /// which side of a plane it falls on. The component-wise product is
    /// [`BlockPos::component_mul`]; the `*` operator is reserved for the cross
    /// product.
    pub const fn dot(self, v: Self) -> i32 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// Component-wise minimum of `self` and `other`.
    ///
    /// Two positions combine axis by axis without unpacking the triple; the
    /// result is still a position.
    pub const fn min(self, other: Self) -> Self {
        Self {
            x: if self.x < other.x { self.x } else { other.x },
            y: if self.y < other.y { self.y } else { other.y },
            z: if self.z < other.z { self.z } else { other.z },
        }
    }

    /// Component-wise maximum of `self` and `other`.
    ///
    /// The natural companion to [`BlockPos::min`], again without unpacking the
    /// triple.
    pub const fn max(self, other: Self) -> Self {
        Self {
            x: if self.x > other.x { self.x } else { other.x },
            y: if self.y > other.y { self.y } else { other.y },
            z: if self.z > other.z { self.z } else { other.z },
        }
    }

    /// Component-wise product of `self` and `other`.
    ///
    /// The `*` operator is the cross product, matching the geometry
    /// convention; this method is the other reading, useful for scaling a
    /// position per axis.
    pub const fn component_mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
        }
    }
}

impl fmt::Debug for BlockPos {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}

impl Add for BlockPos {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl AddAssign for BlockPos {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for BlockPos {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl SubAssign for BlockPos {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<i32> for BlockPos {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl MulAssign<i32> for BlockPos {
    fn mul_assign(&mut self, rhs: i32) {
        *self = *self * rhs;
    }
}

impl Div<i32> for BlockPos {
    type Output = Self;

    fn div(self, rhs: i32) -> Self {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl DivAssign<i32> for BlockPos {
    fn div_assign(&mut self, rhs: i32) {
        *self = *self / rhs;
    }
}

impl Mul for BlockPos {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl MulAssign for BlockPos {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for BlockPos {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

// BlockSize
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

/// The extent of a box: a non-negative cell count per axis.
///
/// A schematic declares sizes as signed triples, because a region may extend
/// along a negative axis. What storage actually addresses is the box those
/// signed values span, and every count in that box is non-negative. Making
/// the distinction a type means a signed value cannot silently be used as an
/// index bound, and a span cannot be misread as a world position.
///
/// The box is anchored at [`BlockPos::ORIGIN`] and half-open, the same
/// convention every coordinate walk in this crate shares. Storage order is
/// `x` fastest, then `z`, then `y`, matching the litematic bit stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSize {
    /// Number of cells along the x axis.
    pub x: u32,
    /// Number of cells along the y axis.
    pub y: u32,
    /// Number of cells along the z axis.
    pub z: u32,
}

impl BlockSize {
    /// Creates an extent from its three cell counts.
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    /// The extent of the box a signed size spans, sign folded away per axis.
    ///
    /// This is the bridge from a file's declared `Size` to the range storage
    /// indexes: the direction a negative component describes is irrelevant
    /// once the cell count is known.
    pub const fn of(pos: BlockPos) -> Self {
        Self {
            x: pos.x.unsigned_abs(),
            y: pos.y.unsigned_abs(),
            z: pos.z.unsigned_abs(),
        }
    }

    /// Number of cells in the box.
    ///
    /// The result is exact even for triples near `i32::MAX`, whose product
    /// does not fit a `u64`; callers that need a smaller type must convert
    /// explicitly and decide what overflow means for them.
    pub const fn volume(self) -> u128 {
        self.x as u128 * self.y as u128 * self.z as u128
    }

    /// Whether `pos` lies inside the box, half-open from the origin.
    ///
    /// The origin is inside and the far corner is not, so adjacent boxes of
    /// the same extent neither overlap nor leave gaps. A negative component
    /// is outside by construction.
    pub const fn contains(self, pos: BlockPos) -> bool {
        pos.x >= 0
            && (pos.x as u32) < self.x
            && pos.y >= 0
            && (pos.y as u32) < self.y
            && pos.z >= 0
            && (pos.z as u32) < self.z
    }

    /// The storage index of `pos`, or `None` when it lies outside the box.
    ///
    /// The formula `(y * sz + z) * sx + x` is the litematic stream order.
    pub const fn index_of(self, pos: BlockPos) -> Option<u64> {
        if !self.contains(pos) {
            return None;
        }
        let index =
            (pos.y as u128 * self.z as u128 + pos.z as u128) * self.x as u128 + pos.x as u128;
        if index > u64::MAX as u128 {
            return None;
        }
        Some(index as u64)
    }

    /// The position stored at `index`, or `None` when the index is past the
    /// last cell or any axis is empty. The inverse of [`BlockSize::index_of`].
    pub const fn position_at(self, index: u64) -> Option<BlockPos> {
        let (sx, sz) = (self.x as u64, self.z as u64);
        if sx == 0 || sz == 0 {
            return None;
        }
        let (x, rest) = (index % sx, index / sx);
        let (y, z) = (rest / sz, rest % sz);
        if y >= self.y as u64 {
            return None;
        }
        Some(BlockPos::new(x as i32, y as i32, z as i32))
    }

    /// Every position in the box, in storage order: `x` fastest, then `z`,
    /// then `y`. An empty axis yields no items. Items are produced lazily.
    pub fn cartesian_positions(self) -> impl Iterator<Item = BlockPos> {
        (0..self.y)
            .flat_map(move |y| (0..self.z).map(move |z| (y, z)))
            .flat_map(move |(y, z)| {
                (0..self.x).map(move |x| BlockPos::new(x as i32, y as i32, z as i32))
            })
    }
}

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::{BlockPos, BlockSize};

    fn pos(x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos::new(x, y, z)
    }

    #[test]
    fn position_is_a_signed_point() {
        assert_eq!(pos(-2, 3, -4).abs(), pos(2, 3, 4));
        assert_eq!(format!("{:?}", pos(1, -2, 3)), "(1, -2, 3)");
    }

    #[test]
    fn vector_helpers_are_componentwise() {
        let (a, b) = (pos(1, 2, 3), pos(4, 5, 6));
        assert_eq!(a.dot(b), 32);
        assert_eq!(a.min(b), a);
        assert_eq!(a.max(b), b);
        assert_eq!(a.component_mul(b), pos(4, 10, 18));
    }

    #[test]
    fn block_size_folds_sign_and_indexes_in_storage_order() {
        let size = BlockSize::of(pos(-2, 3, -4));
        assert_eq!(size.volume(), 24);
        assert!(size.contains(pos(0, 0, 0)));
        assert!(size.contains(pos(1, 2, 3)));
        assert!(!size.contains(pos(2, 0, 0)));
        assert!(!size.contains(pos(-1, 0, 0)));
        assert_eq!(size.index_of(pos(1, 2, 3)), Some(1 + (2 * 4 + 3) * 2));
        assert_eq!(size.index_of(pos(2, 0, 0)), None);
        assert_eq!(size.position_at(23), Some(pos(1, 2, 3)));
        assert_eq!(size.position_at(24), None);
    }

    #[test]
    fn block_size_enumerates_x_fastest_then_z_then_y() {
        let size = BlockSize::new(2, 2, 2);
        let positions: Vec<BlockPos> = size.cartesian_positions().collect();
        assert_eq!(positions.len(), 8);
        assert_eq!(positions[0], pos(0, 0, 0));
        assert_eq!(positions[1], pos(1, 0, 0));
        assert_eq!(positions[2], pos(0, 0, 1));
        assert_eq!(positions[4], pos(0, 1, 0));
        assert_eq!(positions[7], pos(1, 1, 1));
        assert_eq!(BlockSize::new(0, 2, 2).cartesian_positions().count(), 0);
    }

    #[test]
    fn operators_follow_vector_conventions() {
        let (a, b) = (pos(1, 2, 3), pos(4, 5, 6));
        assert_eq!(a + b, pos(5, 7, 9));
        assert_eq!(b - a, pos(3, 3, 3));
        assert_eq!(-a, pos(-1, -2, -3));
        assert_eq!(a * 2, pos(2, 4, 6));
        assert_eq!(pos(2, -4, 6) / 2, pos(1, -2, 3));
        assert_eq!(pos(1, 0, 0) * pos(0, 1, 0), pos(0, 0, 1));

        let mut p = a;
        p += b;
        p -= b;
        p *= 2;
        p /= 2;
        p *= pos(1, 0, 0);
        assert_eq!(p, pos(0, 3, -2));
    }
}
