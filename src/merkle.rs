//! Merkle tree implementation for cryptographic file verification in Titor
//!
//! This module provides a robust Merkle tree implementation that enables efficient
//! and secure verification of file integrity across checkpoints. The implementation
//! supports both individual file verification and batch verification of entire
//! directory structures.
//!
//! ## Overview
//!
//! A Merkle tree is a binary tree where each leaf node represents a hash of a file's
//! content and metadata, and each internal node represents the hash of its children.
//! This structure allows for:
//!
//! - **Efficient Verification**: Verify individual files without downloading entire datasets
//! - **Tamper Detection**: Any modification to files changes the root hash
//! - **Batch Operations**: Verify multiple files simultaneously
//! - **Cryptographic Security**: Based on SHA-256 cryptographic hashing
//!
//! ## Tree Structure
//!
//! ```text
//!        Root Hash
//!       /          \
//!    H(A,B)       H(C,D)
//!   /      \     /      \
//!  H(A)   H(B) H(C)   H(D)
//!   |      |    |      |
//! File A File B File C File D
//! ```
//!
//! Each file contributes a leaf node containing:
//! - Content hash (SHA-256 of file content)
//! - Metadata hash (SHA-256 of permissions, timestamps, etc.)
//! - Combined hash (SHA-256 of content hash + metadata hash)
//!
//! ## Usage Examples
//!
//! ### Basic Tree Construction
//!
//! ```rust,ignore
//! use crate::merkle::{MerkleTree, FileEntryHashBuilder};
//! use titor::types::FileEntry;
//! use std::path::PathBuf;
//! use chrono::Utc;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create file entries with proper hashes
//! let mut builder = FileEntryHashBuilder::new();
//! let content = b"Hello, world!";
//! let content_hash = builder.hash_content(content);
//! let metadata_hash = builder.hash_metadata(0o644, &Utc::now());
//! let combined_hash = builder.combined_hash(&content_hash, &metadata_hash);
//!
//! let entry = FileEntry {
//!     path: PathBuf::from("hello.txt"),
//!     content_hash,
//!     metadata_hash,
//!     combined_hash,
//!     size: content.len() as u64,
//!     permissions: 0o644,
//!     modified: Utc::now(),
//!     is_compressed: false,
//!     is_symlink: false,
//!     symlink_target: None,
//!     is_directory: false,
//! };
//!
//! // Build Merkle tree
//! let tree = MerkleTree::from_entries(&[entry])?;
//! let root_hash = tree.root_hash().expect("Tree should have root hash");
//! println!("Root hash: {}", root_hash);
//! # Ok(())
//! # }
//! ```
//!
//! ### Verification Workflow
//!
//! ```rust,ignore
//! use crate::merkle::MerkleTree;
//! use titor::types::FileEntry;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let entries: Vec<FileEntry> = vec![]; // Assume we have file entries
//! // Build tree from current files
//! let current_tree = MerkleTree::from_entries(&entries)?;
//!
//! // Compare with previous checkpoint
//! let previous_root = "previous_root_hash";
//! let current_root = current_tree.root_hash().unwrap_or_default();
//!
//! if current_root == previous_root {
//!     println!("Files are unchanged");
//! } else {
//!     println!("Files have been modified");
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Security Properties
//!
//! - **Collision Resistance**: Based on SHA-256, making hash collisions computationally infeasible
//! - **Tamper Detection**: Any modification to files changes the root hash
//! - **Integrity Verification**: Verify file integrity without accessing original content
//! - **Cryptographic Strength**: Uses industry-standard SHA-256 hashing
//!
//! ## Performance Characteristics
//!
//! - **Time Complexity**: O(n) to build tree, O(log n) for verification
//! - **Space Complexity**: O(n) for tree storage
//! - **Optimization**: Caches intermediate hashes for efficient proof generation
//! - **Scalability**: Handles large file sets efficiently through tree structure
//!
//! ## Implementation Notes
//!
//! - Trees are balanced by duplicating odd nodes during construction
//! - Leaf nodes contain file-specific hashes for content and metadata
//! - Internal nodes are computed from child hashes using SHA-256
//! - Empty trees are supported and return `None` for root hash

use crate::error::{Result, TitorError};
use crate::types::FileEntry;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use crate::collections::{HashMap, HashMapExt};
use tracing::trace;

/// A node in the Merkle tree structure
///
/// Represents both internal nodes and leaf nodes in the tree. Internal nodes
/// contain hashes computed from their children, while leaf nodes contain
/// hashes computed from file content and metadata.
///
/// # Structure
///
/// - **Internal nodes**: Have both left and right children, hash computed from children
/// - **Leaf nodes**: Have no children, hash computed from file data
///
/// # Example
///
    /// ```rust,ignore
    /// use crate::merkle::MerkleNode;
///
/// // Create leaf nodes
/// let leaf1 = MerkleNode {
///     hash: "hash_of_file1".to_string(),
///     left: None,
///     right: None,
/// };
/// let leaf2 = MerkleNode {
///     hash: "hash_of_file2".to_string(),
///     left: None,
///     right: None,
/// };
///
/// // Create internal node from children
/// let internal = MerkleNode::internal(leaf1, leaf2);
/// ```
#[derive(Debug, Clone)]
pub struct MerkleNode {
    /// SHA-256 hash of this node (computed from children for internal nodes)
    pub hash: String,
    /// Left child node (None for leaf nodes)
    #[allow(dead_code)]
    pub left: Option<Box<MerkleNode>>,
    /// Right child node (None for leaf nodes)
    #[allow(dead_code)]
    pub right: Option<Box<MerkleNode>>,
}

impl MerkleNode {
    /// Create an internal node from two child nodes
    ///
    /// Constructs a new internal node by computing the hash from the
    /// concatenation of the two child node hashes using SHA-256.
    ///
    /// # Arguments
    ///
    /// * `left` - Left child node
    /// * `right` - Right child node
    ///
    /// # Returns
    ///
    /// Returns a new internal node with the computed hash and child references.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::merkle::MerkleNode;
    ///
    /// let left = MerkleNode {
    ///     hash: "left_hash".to_string(),
    ///     left: None,
    ///     right: None,
    /// };
    /// let right = MerkleNode {
    ///     hash: "right_hash".to_string(),
    ///     left: None,
    ///     right: None,
    /// };
    ///
    /// let internal = MerkleNode::internal(left, right);
    /// assert_eq!(internal.hash.len(), 64); // SHA-256 hex length
    /// ```
    pub fn internal(left: MerkleNode, right: MerkleNode) -> Self {
        let hash = compute_internal_hash(&left.hash, &right.hash);
        Self {
            hash,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }
}

/// Leaf node in the Merkle tree representing a file
///
/// Each leaf node represents a single file in the filesystem and contains
/// the combined hash of both the file's content and metadata. This ensures
/// that any changes to either the file content or its metadata (permissions,
/// timestamps, etc.) will be detected.
///
/// # Hash Computation
///
/// The combined hash is computed as:
/// ```text
/// combined_hash = SHA-256(content_hash + metadata_hash)
/// ```
///
/// Where:
/// - `content_hash` = SHA-256 of the file's content
/// - `metadata_hash` = SHA-256 of the file's metadata (permissions, timestamps)
///
/// # Example
///
    /// ```rust,ignore
    /// use crate::merkle::{MerkleLeaf, FileEntryHashBuilder};
/// use chrono::Utc;
///
/// let mut builder = FileEntryHashBuilder::new();
/// let content = b"Hello, world!";
/// let content_hash = builder.hash_content(content);
/// let metadata_hash = builder.hash_metadata(0o644, &Utc::now());
/// let combined_hash = builder.combined_hash(&content_hash, &metadata_hash);
///
/// let leaf = MerkleLeaf { combined_hash };
/// ```
#[derive(Debug, Clone)]
pub struct MerkleLeaf {
    /// Combined hash of file content and metadata
    pub combined_hash: String,
}

/// Merkle tree for cryptographic file verification
///
/// A complete Merkle tree implementation that enables efficient verification
/// of file integrity across large datasets. The tree is built from file entries
/// and provides cryptographic guarantees about the integrity of the entire
/// file set through a single root hash.
///
/// # Features
///
/// - **Efficient Construction**: O(n) time complexity for tree building
/// - **Cryptographic Security**: Based on SHA-256 hashing
/// - **Proof Generation**: Cached intermediate hashes for efficient proofs
/// - **Path Indexing**: Fast lookup of files by path
/// - **Tamper Detection**: Any file modification changes the root hash
///
/// # Structure
///
/// The tree maintains:
/// - Root node with the top-level hash
/// - All leaf nodes representing individual files
/// - Path index for fast file lookup
/// - Cache of intermediate node hashes
///
/// # Example
///
/// ```rust,ignore
/// use crate::merkle::MerkleTree;
/// use titor::types::FileEntry;
/// use std::path::PathBuf;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let file_entries: Vec<FileEntry> = vec![]; // Assume we have file entries
/// // Build tree from file entries
/// let tree = MerkleTree::from_entries(&file_entries)?;
///
/// // Get root hash for verification
/// if let Some(root_hash) = tree.root_hash() {
///     println!("Root hash: {}", root_hash);
///     // Store this hash for later verification
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// - Tree construction: O(n) where n is the number of files
/// - Root hash computation: O(1) after construction
/// - Memory usage: O(n) for tree storage
/// - Proof generation: O(log n) with caching
#[derive(Debug)]
pub struct MerkleTree {
    /// Root node of the tree (None if empty)
    pub root: Option<MerkleNode>,
    /// All leaf nodes representing individual files
    pub leaves: Vec<MerkleLeaf>,
    /// Map from file path to leaf index for fast lookup
    path_index: HashMap<PathBuf, usize>,
    /// Cache of intermediate node hashes for efficient proof generation
    /// Maps from (level, index) to hash
    node_cache: HashMap<(usize, usize), String>,
}

impl MerkleTree {
    /// Create a new empty Merkle tree
    ///
    /// Creates an empty tree with no root node and no leaves. This is useful
    /// as a starting point before building a tree from file entries.
    ///
    /// # Returns
    ///
    /// Returns an empty `MerkleTree` with no files.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::merkle::MerkleTree;
    ///
    /// let tree = MerkleTree::new();
    /// assert!(tree.root_hash().is_none());
    /// assert_eq!(tree.leaves.len(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            root: None,
            leaves: Vec::new(),
            path_index: HashMap::new(),
            node_cache: HashMap::new(),
        }
    }
    
    /// Build a Merkle tree from file entries
    ///
    /// Constructs a complete Merkle tree from a collection of file entries.
    /// Each entry contributes a leaf node to the tree, and the tree is built
    /// bottom-up with internal nodes computed from child hashes.
    ///
    /// # Arguments
    ///
    /// * `entries` - Slice of file entries to build the tree from
    ///
    /// # Returns
    ///
    /// Returns a constructed `MerkleTree` with computed root hash.
    ///
    /// # Errors
    ///
    /// - [`TitorError::Internal`] if tree construction fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
/// use crate::merkle::{MerkleTree, FileEntryHashBuilder};
    /// use titor::types::FileEntry;
    /// use std::path::PathBuf;
    /// use chrono::Utc;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create file entries
    /// let mut builder = FileEntryHashBuilder::new();
    /// let content = b"Hello, world!";
    /// let content_hash = builder.hash_content(content);
    /// let metadata_hash = builder.hash_metadata(0o644, &Utc::now());
    /// let combined_hash = builder.combined_hash(&content_hash, &metadata_hash);
    ///
    /// let entry = FileEntry {
    ///     path: PathBuf::from("hello.txt"),
    ///     content_hash,
    ///     metadata_hash,
    ///     combined_hash,
    ///     size: content.len() as u64,
    ///     permissions: 0o644,
    ///     modified: Utc::now(),
    ///     is_compressed: false,
    ///     is_symlink: false,
    ///     symlink_target: None,
    ///     is_directory: false,
    /// };
    ///
    /// let tree = MerkleTree::from_entries(&[entry])?;
    /// assert!(tree.root_hash().is_some());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n) where n is the number of entries
    /// - Space complexity: O(n) for tree storage
    /// - Intermediate hashes are cached for efficient proof generation
    pub fn from_entries(entries: &[FileEntry]) -> Result<Self> {
        if entries.is_empty() {
            return Ok(Self::new());
        }
        
        let mut tree = Self::new();
        
        // Create leaves
        for (index, entry) in entries.iter().enumerate() {
            let leaf = MerkleLeaf {
                combined_hash: entry.combined_hash.clone(),
            };
            
            tree.path_index.insert(entry.path.clone(), index);
            tree.leaves.push(leaf);
        }
        
        // Build tree from leaves
        let nodes: Vec<MerkleNode> = tree.leaves
            .iter()
            .map(|leaf| MerkleNode {
                hash: leaf.combined_hash.clone(),
                left: None,
                right: None,
            })
            .collect();
        
        tree.root = Some(tree.build_tree_with_cache(nodes)?);
        
        trace!("Built Merkle tree with {} leaves", tree.leaves.len());
        Ok(tree)
    }
    
    /// Build tree recursively from nodes, caching intermediate hashes
    fn build_tree_with_cache(&mut self, nodes: Vec<MerkleNode>) -> Result<MerkleNode> {
        self.build_tree_level(nodes, 0)
    }
    
    /// Build a single level of the tree
    fn build_tree_level(&mut self, mut nodes: Vec<MerkleNode>, level: usize) -> Result<MerkleNode> {
        if nodes.is_empty() {
            return Err(TitorError::internal("Cannot build tree from empty nodes"));
        }
        
        if nodes.len() == 1 {
            return Ok(nodes.pop().unwrap());
        }
        
        let mut next_level = Vec::new();
        
        // Pair up nodes
        let mut i = 0;
        while i < nodes.len() {
            if i + 1 < nodes.len() {
                // Pair exists
                let left = nodes[i].clone();
                let right = nodes[i + 1].clone();
                let internal_node = MerkleNode::internal(left, right);
                
                // Cache the node hash
                let index = next_level.len();
                self.node_cache.insert((level + 1, index), internal_node.hash.clone());
                
                next_level.push(internal_node);
                i += 2;
            } else {
                // Odd node - duplicate it
                let node = nodes[i].clone();
                let internal_node = MerkleNode::internal(node.clone(), node);
                
                // Cache the node hash
                let index = next_level.len();
                self.node_cache.insert((level + 1, index), internal_node.hash.clone());
                
                next_level.push(internal_node);
                i += 1;
            }
        }
        
        // Recursively build next level
        self.build_tree_level(next_level, level + 1)
    }
    
    /// Get the root hash of the tree
    ///
    /// Returns the SHA-256 hash of the root node, which represents the
    /// cryptographic digest of the entire file set. This hash changes
    /// if any file in the tree is modified, added, or removed.
    ///
    /// # Returns
    ///
    /// - `Some(String)` - The root hash as a 64-character hexadecimal string
    /// - `None` - If the tree is empty
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::merkle::MerkleTree;
    /// use titor::types::FileEntry;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let entries: Vec<FileEntry> = vec![]; // Assume we have entries
    /// let tree = MerkleTree::from_entries(&entries)?;
    /// 
    /// match tree.root_hash() {
    ///     Some(hash) => {
    ///         println!("Root hash: {}", hash);
    ///         assert_eq!(hash.len(), 64); // SHA-256 hex length
    ///     }
    ///     None => println!("Tree is empty"),
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Usage
    ///
    /// The root hash is typically used for:
    /// - Verifying file integrity across checkpoints
    /// - Detecting changes in file sets
    /// - Cryptographic verification of data integrity
    pub fn root_hash(&self) -> Option<String> {
        self.root.as_ref().map(|r| r.hash.clone())
    }
    

}

/// Compute hash for an internal node
///
/// Computes the SHA-256 hash of an internal node by concatenating the
/// hashes of its left and right children and hashing the result.
///
/// # Arguments
///
/// * `left` - Hash of the left child node
/// * `right` - Hash of the right child node
///
/// # Returns
///
/// Returns the SHA-256 hash as a 64-character hexadecimal string.
///
/// # Example
///
/// ```rust,ignore
/// # use crate::merkle::compute_internal_hash;
/// let left_hash = "abc123..."; // 64-char hex string
/// let right_hash = "def456..."; // 64-char hex string
/// let internal_hash = compute_internal_hash(left_hash, right_hash);
/// assert_eq!(internal_hash.len(), 64);
/// ```
fn compute_internal_hash(left: &str, right: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(left);
    hasher.update(right);
    hex::encode(hasher.finalize())
}

/// Builder for creating file entries with proper hashes
///
/// A utility for computing the various hashes needed for file entries in
/// the Merkle tree. This builder handles the complexity of computing
/// content hashes, metadata hashes, and combined hashes correctly.
///
/// # Hash Types
///
/// The builder computes three types of hashes:
/// 1. **Content Hash**: SHA-256 of the file's content
/// 2. **Metadata Hash**: SHA-256 of file metadata (permissions, timestamps)
/// 3. **Combined Hash**: SHA-256 of content_hash + metadata_hash
///
/// # Example
///
/// ```rust,ignore
/// use crate::merkle::FileEntryHashBuilder;
/// use chrono::Utc;
///
/// let mut builder = FileEntryHashBuilder::new();
/// let content = b"Hello, world!";
///
/// // Compute individual hashes
/// let content_hash = builder.hash_content(content);
/// let metadata_hash = builder.hash_metadata(0o644, &Utc::now());
/// let combined_hash = builder.combined_hash(&content_hash, &metadata_hash);
///
/// // All hashes are 64-character hex strings
/// assert_eq!(content_hash.len(), 64);
/// assert_eq!(metadata_hash.len(), 64);
/// assert_eq!(combined_hash.len(), 64);
/// ```
///
/// # Thread Safety
///
/// The builder is not thread-safe and should not be shared between threads.
/// Create a new builder instance for each thread.
#[derive(Debug)]
pub struct FileEntryHashBuilder {
    hasher: Sha256,
}

impl FileEntryHashBuilder {
    /// Create a new hash builder
    ///
    /// Creates a new instance of the hash builder with a fresh SHA-256 hasher.
    ///
    /// # Returns
    ///
    /// Returns a new `FileEntryHashBuilder` ready for use.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::merkle::FileEntryHashBuilder;
    ///
    /// let builder = FileEntryHashBuilder::new();
    /// // Ready to compute hashes
    /// ```
    pub fn new() -> Self {
        Self {
            hasher: Sha256::new(),
        }
    }
    
    /// Hash file content
    #[allow(dead_code)]
    pub fn hash_content(&mut self, content: &[u8]) -> String {
        self.hasher.update(content);
        hex::encode(self.hasher.finalize_reset())
    }
    
    /// Hash file metadata
    ///
    /// Computes the SHA-256 hash of file metadata including permissions
    /// and modification time. This ensures that changes to file metadata
    /// are detected in the Merkle tree.
    ///
    /// # Arguments
    ///
    /// * `permissions` - Unix-style file permissions (e.g., 0o644)
    /// * `modified` - Last modification timestamp in UTC
    ///
    /// # Returns
    ///
    /// Returns the SHA-256 hash as a 64-character hexadecimal string.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::merkle::FileEntryHashBuilder;
    /// use chrono::Utc;
    ///
    /// let mut builder = FileEntryHashBuilder::new();
    /// let metadata_hash = builder.hash_metadata(0o644, &Utc::now());
    /// assert_eq!(metadata_hash.len(), 64);
    /// ```
    pub fn hash_metadata(
        &mut self,
        permissions: u32,
        modified: &chrono::DateTime<chrono::Utc>,
    ) -> String {
        self.hasher.update(&permissions.to_le_bytes());
        self.hasher.update(modified.to_rfc3339().as_bytes());
        hex::encode(self.hasher.finalize_reset())
    }
    
    /// Compute combined hash
    ///
    /// Combines the content hash and metadata hash into a single hash
    /// that represents the complete state of a file. This is the hash
    /// used in the Merkle tree leaf nodes.
    ///
    /// # Arguments
    ///
    /// * `content_hash` - SHA-256 hash of the file's content
    /// * `metadata_hash` - SHA-256 hash of the file's metadata
    ///
    /// # Returns
    ///
    /// Returns the SHA-256 hash as a 64-character hexadecimal string.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::merkle::FileEntryHashBuilder;
    /// use chrono::Utc;
    ///
    /// let mut builder = FileEntryHashBuilder::new();
    /// let content = b"Hello, world!";
    /// let content_hash = builder.hash_content(content);
    /// let metadata_hash = builder.hash_metadata(0o644, &Utc::now());
    /// let combined_hash = builder.combined_hash(&content_hash, &metadata_hash);
    /// 
    /// assert_eq!(combined_hash.len(), 64);
    /// // Combined hash changes if either content or metadata changes
    /// ```
    ///
    /// # Implementation
    ///
    /// The combined hash is computed as:
    /// ```text
    /// combined_hash = SHA-256(content_hash + metadata_hash)
    /// ```
    pub fn combined_hash(&mut self, content_hash: &str, metadata_hash: &str) -> String {
        self.hasher.update(content_hash);
        self.hasher.update(metadata_hash);
        hex::encode(self.hasher.finalize_reset())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    fn create_test_entry(path: &str, content: &str) -> FileEntry {
        let mut builder = FileEntryHashBuilder::new();
        let content_hash = builder.hash_content(content.as_bytes());
        let metadata_hash = builder.hash_metadata(0o644, &Utc::now());
        let combined_hash = builder.combined_hash(&content_hash, &metadata_hash);
        
        FileEntry {
            path: PathBuf::from(path),
            content_hash,
            size: content.len() as u64,
            permissions: 0o644,
            modified: Utc::now(),
            is_compressed: false,
            metadata_hash,
            combined_hash,
            is_symlink: false,
            symlink_target: None,
            is_directory: false,
        }
    }
    
    #[test]
    fn test_merkle_tree_empty() {
        let tree = MerkleTree::new();
        assert!(tree.root_hash().is_none());
        assert_eq!(tree.leaves.len(), 0);
    }
    
    #[test]
    fn test_merkle_tree_single_file() {
        let entries = vec![create_test_entry("file1.txt", "content1")];
        let tree = MerkleTree::from_entries(&entries).unwrap();
        
        assert!(tree.root_hash().is_some());
        assert_eq!(tree.leaves.len(), 1);
    }
    
    #[test]
    fn test_merkle_tree_multiple_files() {
        let entries = vec![
            create_test_entry("file1.txt", "content1"),
            create_test_entry("file2.txt", "content2"),
            create_test_entry("file3.txt", "content3"),
            create_test_entry("file4.txt", "content4"),
        ];
        
        let tree = MerkleTree::from_entries(&entries).unwrap();
        
        assert!(tree.root_hash().is_some());
        assert_eq!(tree.leaves.len(), 4);
    }
    
    #[test]
    fn test_hash_consistency() {
        let entry1 = create_test_entry("file.txt", "content");
        let entry2 = create_test_entry("file.txt", "content");
        
        // Same content should produce same content hash
        assert_eq!(entry1.content_hash, entry2.content_hash);
    }
} 