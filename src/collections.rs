//! Collection type aliases that switch between gxhash and std collections
//! based on feature flags. This allows building on systems without specific
//! CPU intrinsics (AES-NI, SSE2) while providing better performance on systems
//! that support these features.

#[cfg(feature = "gxhash")]
pub use gxhash::{
    HashMap as GxHashMap, HashMapExt, HashSet as GxHashSet, HashSetExt, GxBuildHasher,
};

#[cfg(not(feature = "gxhash"))]
use std::collections::{HashMap as StdHashMap, HashSet as StdHashSet};

/// Type alias for HashMap that uses gxhash when available, std otherwise
#[cfg(feature = "gxhash")]
pub type HashMap<K, V> = GxHashMap<K, V>;

/// Type alias for HashMap that uses gxhash when available, std otherwise
#[cfg(not(feature = "gxhash"))]
pub type HashMap<K, V> = StdHashMap<K, V>;

/// Type alias for HashSet that uses gxhash when available, std otherwise
#[cfg(feature = "gxhash")]
pub type HashSet<T> = GxHashSet<T>;

/// Type alias for HashSet that uses gxhash when available, std otherwise
#[cfg(not(feature = "gxhash"))]
pub type HashSet<T> = StdHashSet<T>;

/// Extension trait for creating HashMap instances
#[cfg(not(feature = "gxhash"))]
pub trait HashMapExt {
    /// Creates a new HashMap with default capacity
    fn new() -> Self;
    
    /// Creates a new HashMap with specified capacity
    fn with_capacity(capacity: usize) -> Self;
}

#[cfg(not(feature = "gxhash"))]
impl<K, V> HashMapExt for StdHashMap<K, V> {
    fn new() -> Self {
        StdHashMap::new()
    }
    
    fn with_capacity(capacity: usize) -> Self {
        StdHashMap::with_capacity(capacity)
    }
}

/// Extension trait for creating HashSet instances
#[cfg(not(feature = "gxhash"))]
pub trait HashSetExt {
    /// Creates a new HashSet with default capacity
    fn new() -> Self;
    
    /// Creates a new HashSet with specified capacity
    fn with_capacity(capacity: usize) -> Self;
}

#[cfg(not(feature = "gxhash"))]
impl<T> HashSetExt for StdHashSet<T> {
    fn new() -> Self {
        StdHashSet::new()
    }
    
    fn with_capacity(capacity: usize) -> Self {
        StdHashSet::with_capacity(capacity)
    }
}

/// Hasher type that uses GxBuildHasher when available, std otherwise
#[cfg(not(feature = "gxhash"))]
pub type GxBuildHasher = std::hash::RandomState; 