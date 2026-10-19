//! The Minecraft vocabulary a schematic is built from: where a block sits and
//! what that block is.
//!
//! [`BlockState`] is the first half of the vocabulary: a namespaced block id
//! together with its properties. It is cheap to clone and orders
//! deterministically, which is what makes it usable as a palette key in both
//! the writer and downstream layout combinators.
//!
//! Schematic code thinks in two quantities that happen to share one shape: a
//! point in the world, and the span of a box. [`BlockPos`] gives both the same
//! signed triple, so a value moves between the two roles without a conversion
//! type, and a bounding box can be passed wherever a position is expected. The
//! convention that keeps the roles readable is the origin anchor: extent
//! operations measure from [`BlockPos::ORIGIN`], so a size is the corner
//! opposite the origin rather than a free-floating vector.

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

/// A Minecraft block coordinate, three signed components.
///
/// Positions and sizes are deliberately the same type. A layout author can
/// carry a value from a query into arithmetic and back without ceremony, and a
/// schematic's bounding box can be handed to code that expects a position.
/// Extent-oriented methods therefore document how they read the origin: a size
/// is anchored at [`BlockPos::ORIGIN`], not a directionless vector.
///
/// Components are signed because Minecraft's coordinate space is signed and a
/// region's declared size may run along a negative axis. The box queries pin
/// down which combinations they accept, so invalid extents cannot silently
/// masquerade as valid ones.
///
/// Arithmetic follows vector intuition: `+` and `-` are component-wise, `*` and
/// `/` with an `i32` scale every axis, and `*` between two positions is the
/// cross product. Use [`BlockPos::component_mul`] for the component-wise
/// reading.
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
    /// Every box extent is anchored here, so this constant is also the lower
    /// corner shared by all sizes.
    pub const ORIGIN: Self = Self::new(0, 0, 0);

    /// Creates a position from its three components.
    ///
    /// Any signed triple is accepted. Negative components are ordinary world
    /// positions, and as an extent they describe a box that reaches into the
    /// negative axes.
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    /// Number of cells in the box between [`BlockPos::ORIGIN`] and `self`.
    ///
    /// Sign is ignored, so the result answers how many cells the span covers
    /// even when it runs along a negative axis. A zero on any axis makes the
    /// box empty.
    pub const fn volume(&self) -> u32 {
        self.x.unsigned_abs() * self.y.unsigned_abs() * self.z.unsigned_abs()
    }

    /// A position with the absolute value of each component.
    ///
    /// This is how a negative-direction extent becomes the size of the box it
    /// spans: the result always lies in the positive octant.
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

    /// Whether `pos` lies inside the box between [`BlockPos::ORIGIN`] and
    /// `self`, with `self` treated as the box extent.
    ///
    /// The box is half-open: the origin is inside and the far corner is not.
    /// Half-open bounds tile space cleanly — adjacent boxes of the same extent
    /// neither overlap nor leave gaps — which is the convention every
    /// coordinate walk in this crate shares.
    pub const fn contains(&self, pos: Self) -> bool {
        pos.x >= 0 && pos.x < self.x && pos.y >= 0 && pos.y < self.y && pos.z >= 0 && pos.z < self.z
    }

    /// Every position inside the box between [`BlockPos::ORIGIN`] and `self`,
    /// in writing order: `x` fastest, then `z`, then `y`.
    ///
    /// The order is the one schematic storage uses, so a box can be walked
    /// without maintaining a separate index-to-coordinate mapping. Items are
    /// produced lazily, which keeps the helper usable for boxes far too large
    /// to materialize.
    ///
    /// # Panics
    ///
    /// Panics if any component of `self` is negative, since such an extent
    /// describes no box to enumerate.
    pub fn cartesian_positions(self) -> impl Iterator<Item = Self> {
        assert!(
            self.x >= 0 && self.y >= 0 && self.z >= 0,
            "box extent {self:?} has a negative component"
        );
        (0..self.y)
            .flat_map(move |y| (0..self.z).map(move |z| (y, z)))
            .flat_map(move |(y, z)| (0..self.x).map(move |x| Self::new(x, y, z)))
    }

    /// Component-wise minimum of `self` and `other`.
    ///
    /// Both roles support it: two boxes can be intersected corner by corner,
    /// or two positions combined axis by axis, without unpacking the triple.
    pub const fn min(self, other: Self) -> Self {
        Self {
            x: if self.x < other.x { self.x } else { other.x },
            y: if self.y < other.y { self.y } else { other.y },
            z: if self.z < other.z { self.z } else { other.z },
        }
    }

    /// Component-wise maximum of `self` and `other`.
    ///
    /// The natural companion to [`BlockPos::min`]: boxes are unioned by taking
    /// the maximum corner and the minimum origin, again without unpacking the
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
    /// convention; this method is the other reading, useful for scaling an
    /// extent per axis.
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

// Tests
//
// ++++++++++++============++++++++++++============++++++++++++============++++++++++++============

#[cfg(test)]
mod tests {
    use super::BlockPos;

    fn pos(x: i32, y: i32, z: i32) -> BlockPos {
        BlockPos::new(x, y, z)
    }

    #[test]
    fn extent_methods_anchor_at_the_origin() {
        let size = pos(2, 3, 4);
        assert_eq!(size.volume(), 24);
        assert_eq!(pos(-2, 3, -4).abs(), pos(2, 3, 4));
        assert!(size.contains(pos(0, 0, 0)));
        assert!(size.contains(pos(1, 2, 3)));
        assert!(!size.contains(pos(2, 0, 0)));
        assert!(!size.contains(pos(-1, 0, 0)));
        assert_eq!(format!("{:?}", pos(1, -2, 3)), "(1, -2, 3)");
    }

    #[test]
    fn cartesian_positions_walk_x_fastest_then_z_then_y() {
        let positions: Vec<BlockPos> = pos(2, 2, 2).cartesian_positions().collect();
        assert_eq!(positions.len(), 8);
        assert_eq!(positions[0], pos(0, 0, 0));
        assert_eq!(positions[1], pos(1, 0, 0));
        assert_eq!(positions[2], pos(0, 0, 1));
        assert_eq!(positions[4], pos(0, 1, 0));
        assert_eq!(BlockPos::ORIGIN.cartesian_positions().count(), 0);
    }

    #[test]
    #[should_panic(expected = "negative component")]
    fn cartesian_positions_rejects_negative_extent() {
        let _ = pos(-1, 2, 3).cartesian_positions();
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
