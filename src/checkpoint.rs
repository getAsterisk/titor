//! Checkpoint definitions and operations
//!
//! This module defines the checkpoint structure and provides functionality
//! for creating, verifying, and manipulating checkpoints.
//!
//! ## Overview
//!
//! A checkpoint represents an immutable snapshot of a directory tree at a specific
//! point in time. Each checkpoint contains:
//!
//! - **Metadata**: Information about files, sizes, and changes
//! - **Content Hash**: Merkle root of all file hashes for verification
//! - **State Hash**: Cryptographic hash of the checkpoint itself
//! - **Parent Reference**: Link to the previous checkpoint (if any)
//!
//! ## Checkpoint Integrity
//!
//! Checkpoints use multiple layers of integrity verification:
//!
//! 1. **Content Verification**: Merkle tree ensures file integrity
//! 2. **State Verification**: SHA-256 hash of checkpoint metadata
//! 3. **Chain Verification**: Parent references maintain timeline integrity
//!
//! ## Examples
//!
//! ```rust
//! use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
//!
//! // Create checkpoint metadata
//! let metadata = CheckpointMetadataBuilder::new()
//!     .file_count(150)
//!     .total_size(10_000_000)
//!     .compressed_size(3_000_000)
//!     .build();
//!
//! // Create a checkpoint
//! let checkpoint = Checkpoint::new(
//!     Some("parent-id".to_string()),
//!     Some("Added new feature".to_string()),
//!     metadata,
//!     "merkle-root-hash".to_string()
//! );
//!
//! // Verify integrity
//! assert!(checkpoint.verify_integrity().unwrap());
//! ```

use crate::error::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use gxhash::HashMap;

/// Represents a checkpoint in the timeline
///
/// A checkpoint is an immutable snapshot of a directory tree at a specific point
/// in time. It contains all the information needed to restore the directory to
/// that exact state.
///
/// # Structure
///
/// Each checkpoint contains:
/// - Unique identifier (UUID v4)
/// - Optional parent reference for timeline tracking
/// - Timestamp of creation
/// - Metadata about files and changes
/// - Cryptographic hashes for verification
///
/// # Examples
///
/// ```rust
/// use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
///
/// let metadata = CheckpointMetadataBuilder::new()
///     .file_count(100)
///     .total_size(1_000_000)
///     .build();
///
/// let checkpoint = Checkpoint::new(
///     None, // No parent (root checkpoint)
///     Some("Initial import".to_string()),
///     metadata,
///     "merkle-root-hash".to_string()
/// );
///
/// println!("Checkpoint ID: {}", checkpoint.short_id());
/// println!("Created at: {}", checkpoint.timestamp);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Unique identifier for this checkpoint
    pub id: String,
    /// Parent checkpoint ID (None for root checkpoint)
    pub parent_id: Option<String>,
    /// Creation timestamp
    pub timestamp: DateTime<Utc>,
    /// User-provided description
    pub description: Option<String>,
    /// Additional metadata
    pub metadata: CheckpointMetadata,
    /// SHA-256 hash of the entire checkpoint state
    pub state_hash: String,
    /// Merkle root of all file hashes
    pub content_merkle_root: String,
    /// Digital signature for tamper detection (optional)
    pub signature: Option<String>,
}

/// Metadata associated with a checkpoint
///
/// Contains statistics and information about the checkpoint including
/// file counts, sizes, changes from parent, and system information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Number of files in this checkpoint
    pub file_count: usize,
    /// Total size in bytes (uncompressed)
    pub total_size: u64,
    /// Compressed size in storage
    pub compressed_size: u64,
    /// Number of files changed from parent
    pub files_changed: usize,
    /// Bytes changed from parent
    pub bytes_changed: u64,
    /// User-defined tags
    pub tags: Vec<String>,
    /// Custom key-value metadata
    pub custom: HashMap<String, String>,
    /// Titor version that created this checkpoint
    pub titor_version: String,
    /// Host information
    pub host_info: HostInfo,
}

/// Information about the host system
///
/// Captures system information at the time of checkpoint creation
/// for debugging and auditing purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// Hostname
    pub hostname: String,
    /// Operating system
    pub os: String,
    /// Architecture
    pub arch: String,
    /// Username (if available)
    pub username: Option<String>,
}

impl Default for HostInfo {
    fn default() -> Self {
        Self {
            hostname: hostname::get()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|_| "unknown".to_string()),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            username: std::env::var("USER")
                .or_else(|_| std::env::var("USERNAME"))
                .ok(),
        }
    }
}

impl Checkpoint {
    /// Create a new checkpoint
    ///
    /// Creates a new checkpoint with a unique ID and computes its state hash.
    /// The state hash is automatically calculated from all checkpoint components
    /// to ensure integrity.
    ///
    /// # Arguments
    ///
    /// * `parent_id` - Optional ID of the parent checkpoint
    /// * `description` - Optional human-readable description
    /// * `metadata` - Checkpoint metadata including file statistics
    /// * `content_merkle_root` - Merkle root hash of all file contents
    ///
    /// # Returns
    ///
    /// A new `Checkpoint` instance with computed state hash.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
    /// let checkpoint = Checkpoint::new(
    ///     Some("parent-uuid".to_string()),
    ///     Some("Fixed bug #123".to_string()),
    ///     CheckpointMetadataBuilder::new().build(),
    ///     "merkle-hash".to_string()
    /// );
    /// ```
    pub fn new(
        parent_id: Option<String>,
        description: Option<String>,
        metadata: CheckpointMetadata,
        content_merkle_root: String,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now();
        
        let mut checkpoint = Self {
            id,
            parent_id,
            timestamp,
            description,
            metadata,
            state_hash: String::new(),
            content_merkle_root,
            signature: None,
        };
        
        // Compute state hash
        checkpoint.state_hash = checkpoint.compute_state_hash();
        checkpoint
    }
    
    /// Compute the state hash from all checkpoint components
    ///
    /// Calculates a SHA-256 hash of all checkpoint components to create
    /// a unique fingerprint. This hash can be used to verify that the
    /// checkpoint has not been tampered with.
    ///
    /// # Returns
    ///
    /// Hex-encoded SHA-256 hash string.
    ///
    /// # Implementation Details
    ///
    /// The hash includes:
    /// - Checkpoint ID
    /// - Parent ID (or empty string if none)
    /// - Timestamp (RFC3339 format)
    /// - Description (or empty string if none)
    /// - Serialized metadata
    /// - Content Merkle root
    pub fn compute_state_hash(&self) -> String {
        let mut hasher = Sha256::new();
        
        // Hash structural components
        hasher.update(&self.id);
        hasher.update(self.parent_id.as_deref().unwrap_or(""));
        hasher.update(self.timestamp.to_rfc3339());
        hasher.update(self.description.as_deref().unwrap_or(""));
        
        // Hash metadata
        if let Ok(metadata_bytes) = serde_json::to_vec(&self.metadata) {
            hasher.update(&metadata_bytes);
        }
        
        // Hash content merkle root
        hasher.update(&self.content_merkle_root);
        
        hex::encode(hasher.finalize())
    }
    
    /// Verify checkpoint integrity
    ///
    /// Verifies that the checkpoint's state hash matches the computed hash,
    /// ensuring the checkpoint has not been modified since creation.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` if the checkpoint is valid
    /// - `Ok(false)` if the checkpoint has been tampered with
    /// - `Err` if verification fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
    /// # let checkpoint = Checkpoint::new(None, None, CheckpointMetadataBuilder::new().build(), "".to_string());
    /// match checkpoint.verify_integrity() {
    ///     Ok(true) => println!("Checkpoint is valid"),
    ///     Ok(false) => println!("Checkpoint has been tampered with!"),
    ///     Err(e) => println!("Verification failed: {}", e),
    /// }
    /// ```
    ///
    /// # Notes
    ///
    /// This only verifies the checkpoint metadata integrity. Full verification
    /// including file contents requires access to the storage backend.
    pub fn verify_integrity(&self) -> Result<bool> {
        // Verify state hash
        let computed_hash = self.compute_state_hash();
        if computed_hash != self.state_hash {
            return Ok(false);
        }
        
        // Additional verification would be done by the Titor instance
        // which has access to storage and can verify merkle roots
        
        Ok(true)
    }
    
    /// Check if this checkpoint is a descendant of another
    ///
    /// Traverses the parent chain to determine if this checkpoint is
    /// descended from the specified ancestor.
    ///
    /// # Arguments
    ///
    /// * `ancestor_id` - ID of the potential ancestor checkpoint
    /// * `checkpoint_map` - Map of all checkpoints for traversal
    ///
    /// # Returns
    ///
    /// `true` if this checkpoint is a descendant of the specified ancestor.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
    /// # use gxhash::{HashMap, HashMapExt};
    /// # let checkpoint = Checkpoint::new(Some("parent".to_string()), None, CheckpointMetadataBuilder::new().build(), "".to_string());
    /// # let checkpoints = HashMap::new();
    /// if checkpoint.is_descendant_of("root-id", &checkpoints) {
    ///     println!("Checkpoint is on the main branch");
    /// }
    /// ```
    pub fn is_descendant_of(&self, ancestor_id: &str, checkpoint_map: &HashMap<String, Checkpoint>) -> bool {
        let mut current_id = self.parent_id.as_ref();
        
        while let Some(id) = current_id {
            if id == ancestor_id {
                return true;
            }
            
            // Move up the chain
            current_id = checkpoint_map
                .get(id)
                .and_then(|c| c.parent_id.as_ref());
        }
        
        false
    }
    
    /// Get the chain of parent checkpoints up to the root
    ///
    /// Returns a vector of checkpoint IDs representing the path from
    /// the root checkpoint to this checkpoint's parent.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_map` - Map of all checkpoints for traversal
    ///
    /// # Returns
    ///
    /// Vector of checkpoint IDs from root to parent (exclusive of self).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
    /// # use gxhash::{HashMap, HashMapExt};
    /// # let checkpoint = Checkpoint::new(Some("parent".to_string()), None, CheckpointMetadataBuilder::new().build(), "".to_string());
    /// # let checkpoints = HashMap::new();
    /// let chain = checkpoint.get_parent_chain(&checkpoints);
    /// println!("Checkpoint has {} ancestors", chain.len());
    /// ```
    pub fn get_parent_chain(&self, checkpoint_map: &HashMap<String, Checkpoint>) -> Vec<String> {
        let mut chain = Vec::new();
        let mut current_id = self.parent_id.as_ref();
        
        while let Some(id) = current_id {
            chain.push(id.clone());
            current_id = checkpoint_map
                .get(id)
                .and_then(|c| c.parent_id.as_ref());
        }
        
        chain.reverse();
        chain
    }
    
    /// Get a short ID for display (first 8 characters)
    ///
    /// Returns a truncated version of the checkpoint ID suitable for
    /// display in logs and user interfaces.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
    /// # let checkpoint = Checkpoint::new(None, None, CheckpointMetadataBuilder::new().build(), "".to_string());
    /// println!("Checkpoint: {}", checkpoint.short_id());
    /// // Output: "12345678" (first 8 chars of UUID)
    /// ```
    pub fn short_id(&self) -> &str {
        &self.id[..8.min(self.id.len())]
    }
    
    /// Format checkpoint for display
    pub fn display_format(&self) -> String {
        format!(
            "[{}] {} - {} files, {} bytes{}",
            self.short_id(),
            self.timestamp.format("%Y-%m-%d %H:%M:%S"),
            self.metadata.file_count,
            human_bytes(self.metadata.total_size),
            self.description
                .as_ref()
                .map(|d| format!(" - {}", d))
                .unwrap_or_default()
        )
    }
}

/// Format bytes in human-readable form
fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    if unit_idx == 0 {
        format!("{} {}", size as u64, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Builder for checkpoint metadata
///
/// Provides a fluent interface for constructing `CheckpointMetadata` instances
/// with sensible defaults. This is the recommended way to create checkpoint
/// metadata.
///
/// # Examples
///
/// ```rust
/// use titor::checkpoint::CheckpointMetadataBuilder;
///
/// let metadata = CheckpointMetadataBuilder::new()
///     .file_count(250)
///     .total_size(50_000_000)
///     .compressed_size(15_000_000)
///     .files_changed(10)
///     .bytes_changed(100_000)
///     .add_tag("release".to_string())
///     .add_tag("v1.2.0".to_string())
///     .add_custom("commit".to_string(), "abc123".to_string())
///     .build();
///
/// assert_eq!(metadata.file_count, 250);
/// assert_eq!(metadata.tags.len(), 2);
/// ```
#[derive(Debug, Default)]
pub struct CheckpointMetadataBuilder {
    file_count: usize,
    total_size: u64,
    compressed_size: u64,
    files_changed: usize,
    bytes_changed: u64,
    tags: Vec<String>,
    custom: HashMap<String, String>,
}

impl CheckpointMetadataBuilder {
    /// Create a new builder
    ///
    /// Initializes a new builder with default values:
    /// - All counts and sizes set to 0
    /// - Empty tags and custom metadata
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set file count
    ///
    /// # Arguments
    ///
    /// * `count` - Total number of files in the checkpoint
    pub fn file_count(mut self, count: usize) -> Self {
        self.file_count = count;
        self
    }
    
    /// Set total size
    ///
    /// # Arguments
    ///
    /// * `size` - Total uncompressed size of all files in bytes
    pub fn total_size(mut self, size: u64) -> Self {
        self.total_size = size;
        self
    }
    
    /// Set compressed size
    ///
    /// # Arguments
    ///
    /// * `size` - Total compressed size in storage in bytes
    pub fn compressed_size(mut self, size: u64) -> Self {
        self.compressed_size = size;
        self
    }
    
    /// Set files changed
    ///
    /// # Arguments
    ///
    /// * `count` - Number of files that changed from the parent checkpoint
    pub fn files_changed(mut self, count: usize) -> Self {
        self.files_changed = count;
        self
    }
    
    /// Set bytes changed
    ///
    /// # Arguments
    ///
    /// * `bytes` - Total bytes changed from the parent checkpoint
    pub fn bytes_changed(mut self, bytes: u64) -> Self {
        self.bytes_changed = bytes;
        self
    }
    
    /// Add a tag
    ///
    /// Tags can be used to categorize or label checkpoints for easier
    /// identification and filtering.
    ///
    /// # Arguments
    ///
    /// * `tag` - Tag string to add
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::checkpoint::CheckpointMetadataBuilder;
    /// let metadata = CheckpointMetadataBuilder::new()
    ///     .add_tag("release".to_string())
    ///     .add_tag("production".to_string())
    ///     .build();
    /// ```
    pub fn add_tag(mut self, tag: String) -> Self {
        self.tags.push(tag);
        self
    }
    
    /// Add custom metadata
    ///
    /// Allows storing arbitrary key-value pairs with the checkpoint
    /// for application-specific purposes.
    ///
    /// # Arguments
    ///
    /// * `key` - Metadata key
    /// * `value` - Metadata value
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::checkpoint::CheckpointMetadataBuilder;
    /// let metadata = CheckpointMetadataBuilder::new()
    ///     .add_custom("git_commit".to_string(), "abc123".to_string())
    ///     .add_custom("build_number".to_string(), "42".to_string())
    ///     .build();
    /// ```
    pub fn add_custom(mut self, key: String, value: String) -> Self {
        self.custom.insert(key, value);
        self
    }
    
    /// Build the metadata
    ///
    /// Consumes the builder and returns a complete `CheckpointMetadata` instance
    /// with the current Titor version and host information automatically populated.
    ///
    /// # Returns
    ///
    /// A new `CheckpointMetadata` instance with all configured values.
    pub fn build(self) -> CheckpointMetadata {
        CheckpointMetadata {
            file_count: self.file_count,
            total_size: self.total_size,
            compressed_size: self.compressed_size,
            files_changed: self.files_changed,
            bytes_changed: self.bytes_changed,
            tags: self.tags,
            custom: self.custom,
            titor_version: env!("CARGO_PKG_VERSION").to_string(),
            host_info: HostInfo::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gxhash::HashMapExt;
    
    #[test]
    fn test_checkpoint_creation() {
        let metadata = CheckpointMetadataBuilder::new()
            .file_count(100)
            .total_size(1_000_000)
            .build();
        
        let checkpoint = Checkpoint::new(
            None,
            Some("Initial checkpoint".to_string()),
            metadata,
            "merkle_root_hash".to_string(),
        );
        
        assert!(checkpoint.parent_id.is_none());
        assert_eq!(checkpoint.description, Some("Initial checkpoint".to_string()));
        assert_eq!(checkpoint.metadata.file_count, 100);
        assert!(!checkpoint.state_hash.is_empty());
    }
    
    #[test]
    fn test_checkpoint_integrity() {
        let metadata = CheckpointMetadataBuilder::new().build();
        let checkpoint = Checkpoint::new(
            None,
            None,
            metadata,
            "merkle_root".to_string(),
        );
        
        // Should pass integrity check
        assert!(checkpoint.verify_integrity().unwrap());
        
        // Tamper with checkpoint
        let mut tampered = checkpoint.clone();
        tampered.content_merkle_root = "tampered".to_string();
        
        // Should fail integrity check (state hash doesn't match)
        assert!(!tampered.verify_integrity().unwrap());
    }
    
    #[test]
    fn test_human_bytes() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.00 KB");
        assert_eq!(human_bytes(1536), "1.50 KB");
        assert_eq!(human_bytes(1_048_576), "1.00 MB");
        assert_eq!(human_bytes(1_073_741_824), "1.00 GB");
    }
    
    #[test]
    fn test_parent_chain() {
        let mut checkpoints = HashMap::new();
        
        // Create a chain: root -> checkpoint1 -> checkpoint2
        let root = Checkpoint::new(
            None,
            Some("Root".to_string()),
            CheckpointMetadataBuilder::new().build(),
            "root_merkle".to_string(),
        );
        let root_id = root.id.clone();
        checkpoints.insert(root_id.clone(), root);
        
        let checkpoint1 = Checkpoint::new(
            Some(root_id.clone()),
            Some("Checkpoint 1".to_string()),
            CheckpointMetadataBuilder::new().build(),
            "merkle1".to_string(),
        );
        let checkpoint1_id = checkpoint1.id.clone();
        checkpoints.insert(checkpoint1_id.clone(), checkpoint1);
        
        let checkpoint2 = Checkpoint::new(
            Some(checkpoint1_id.clone()),
            Some("Checkpoint 2".to_string()),
            CheckpointMetadataBuilder::new().build(),
            "merkle2".to_string(),
        );
        
        // Test parent chain
        let chain = checkpoint2.get_parent_chain(&checkpoints);
        assert_eq!(chain, vec![root_id.clone(), checkpoint1_id.clone()]);
        
        // Test descendant check
        assert!(checkpoint2.is_descendant_of(&root_id, &checkpoints));
        assert!(checkpoint2.is_descendant_of(&checkpoint1_id, &checkpoints));
        assert!(!checkpoint2.is_descendant_of(&checkpoint2.id, &checkpoints));
    }
} 