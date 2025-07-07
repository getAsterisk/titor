//! Core data types used throughout the Titor library
//!
//! This module contains fundamental data structures that are shared across
//! different components of the library.
//!
//! ## Overview
//!
//! The types in this module represent:
//! - **File System State**: `FileEntry`, `FileManifest` - representing files and their metadata
//! - **Operations**: `RestoreResult`, `CheckpointDiff`, `ChangeStats` - results of various operations
//! - **Configuration**: `TitorConfig`, `CheckpointOptions`, `RestoreOptions` - operation parameters
//! - **Storage**: `StorageObject`, `StorageMetadata` - internal storage representations
//! - **Hooks**: `CheckpointHook` - extensibility points for custom behavior
//!
//! ## Examples
//!
//! ```rust
//! use titor::types::{CheckpointOptions, FileEntry};
//! use std::path::PathBuf;
//! 
//! // Configure checkpoint creation
//! let options = CheckpointOptions {
//!     description: Some("Release v1.0".to_string()),
//!     tags: vec!["release".to_string(), "production".to_string()],
//!     ..Default::default()
//! };
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use gxhash::HashMap;
use std::sync::Arc;

/// Represents a file entry in a checkpoint manifest
///
/// Contains all necessary information to restore a file to its exact state,
/// including content hash, permissions, and metadata.
///
/// # Examples
///
/// ```rust
/// # use titor::types::FileEntry;
/// # use std::path::PathBuf;
/// # use chrono::Utc;
/// let entry = FileEntry {
///     path: PathBuf::from("src/main.rs"),
///     content_hash: "abc123...".to_string(),
///     size: 1024,
///     permissions: 0o644,
///     modified: Utc::now(),
///     is_compressed: true,
///     metadata_hash: "def456...".to_string(),
///     combined_hash: "ghi789...".to_string(),
///     is_symlink: false,
///     symlink_target: None,
///     is_directory: false,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FileEntry {
    /// Relative path from the root directory
    pub path: PathBuf,
    /// SHA-256 hash of file content
    pub content_hash: String,
    /// File size in bytes
    pub size: u64,
    /// Unix file permissions
    pub permissions: u32,
    /// Last modified timestamp
    pub modified: DateTime<Utc>,
    /// Whether the file is compressed in storage
    pub is_compressed: bool,
    /// Hash of file metadata (permissions, timestamps, etc.)
    pub metadata_hash: String,
    /// Combined hash of content and metadata
    pub combined_hash: String,
    /// Whether this is a symbolic link
    pub is_symlink: bool,
    /// Target of symbolic link (if is_symlink is true)
    pub symlink_target: Option<PathBuf>,
    /// Whether this is a directory
    pub is_directory: bool,
}

/// Manifest containing all files in a checkpoint
///
/// A complete record of all files and directories at the time of checkpoint
/// creation, including their hashes and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileManifest {
    /// Checkpoint ID this manifest belongs to
    pub checkpoint_id: String,
    /// List of all files in the checkpoint
    pub files: Vec<FileEntry>,
    /// Total size of all files (uncompressed)
    pub total_size: u64,
    /// Number of files
    pub file_count: usize,
    /// Merkle root of all file hashes
    pub merkle_root: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Statistics about changes between checkpoints
///
/// Provides detailed information about what changed between two states,
/// useful for understanding the impact of changes and optimizing operations.
///
/// # Examples
///
/// ```rust
/// # use titor::types::ChangeStats;
/// let stats = ChangeStats {
///     files_added: 10,
///     files_modified: 5,
///     files_deleted: 2,
///     bytes_added: 50000,
///     bytes_modified: 10000,
///     bytes_deleted: 5000,
///     changed_files: vec![],
/// };
/// 
/// assert_eq!(stats.total_operations(), 17);
/// assert_eq!(stats.net_size_change(), 55000);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChangeStats {
    /// Number of files added
    pub files_added: usize,
    /// Number of files modified
    pub files_modified: usize,
    /// Number of files deleted
    pub files_deleted: usize,
    /// Total size of added files
    pub bytes_added: u64,
    /// Total size of modified files (new size)
    pub bytes_modified: u64,
    /// Total size of deleted files
    pub bytes_deleted: u64,
    /// Files that changed
    pub changed_files: Vec<PathBuf>,
}

impl ChangeStats {
    /// Check if there are any changes
    ///
    /// Returns `true` if any files were added, modified, or deleted.
    pub fn has_changes(&self) -> bool {
        self.files_added > 0 || self.files_modified > 0 || self.files_deleted > 0
    }
    
    /// Get total number of file operations
    ///
    /// Returns the sum of added, modified, and deleted files.
    pub fn total_operations(&self) -> usize {
        self.files_added + self.files_modified + self.files_deleted
    }
    
    /// Get net size change in bytes
    ///
    /// Returns the difference between bytes added/modified and bytes deleted.
    /// Positive values indicate growth, negative values indicate shrinkage.
    pub fn net_size_change(&self) -> i64 {
        (self.bytes_added + self.bytes_modified) as i64 - self.bytes_deleted as i64
    }
}

/// Result of a restore operation
///
/// Contains statistics and information about a completed restore operation,
/// including any warnings that occurred during the process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestoreResult {
    /// ID of the checkpoint that was restored
    pub checkpoint_id: String,
    /// Number of files restored
    pub files_restored: usize,
    /// Number of files deleted (that existed before but not in checkpoint)
    pub files_deleted: usize,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total bytes deleted
    pub bytes_deleted: u64,
    /// Time taken for restoration in milliseconds
    pub duration_ms: u64,
    /// Any warnings during restoration
    pub warnings: Vec<String>,
}

/// Difference between two checkpoints
///
/// Represents the changes between two checkpoints, showing which files
/// were added, modified, or deleted.
///
/// # Examples
///
/// ```rust
/// # use titor::types::CheckpointDiff;
/// # use titor::Titor;
/// # use std::path::PathBuf;
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
/// let diff = titor.diff("checkpoint1", "checkpoint2")?;
/// 
/// for file in &diff.added_files {
///     println!("Added: {:?}", file.path);
/// }
/// 
/// for (old, new) in &diff.modified_files {
///     println!("Modified: {:?} ({} -> {} bytes)", 
///              old.path, old.size, new.size);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointDiff {
    /// Source checkpoint ID
    pub from_id: String,
    /// Target checkpoint ID
    pub to_id: String,
    /// Files added in target
    pub added_files: Vec<FileEntry>,
    /// Files modified between checkpoints (old, new)
    pub modified_files: Vec<(FileEntry, FileEntry)>,
    /// Files deleted in target
    pub deleted_files: Vec<FileEntry>,
    /// Change statistics
    pub stats: ChangeStats,
}

/// Statistics from garbage collection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GcStats {
    /// Number of objects examined
    pub objects_examined: usize,
    /// Number of objects deleted
    pub objects_deleted: usize,
    /// Bytes reclaimed
    pub bytes_reclaimed: u64,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Unreferenced objects found
    pub unreferenced_objects: Vec<String>,
}

/// Configuration for Titor instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TitorConfig {
    /// Root directory being tracked
    pub root_path: PathBuf,
    /// Storage location
    pub storage_path: PathBuf,
    /// Maximum file size to track (0 = unlimited)
    pub max_file_size: u64,
    /// Number of parallel workers for operations
    pub parallel_workers: usize,
    /// Ignore patterns (gitignore style)
    pub ignore_patterns: Vec<String>,
    /// Compression strategy name
    pub compression_strategy: String,
    /// Whether to follow symbolic links
    pub follow_symlinks: bool,
    /// Titor version that created this config
    pub version: String,
}

/// Metadata stored with the storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetadata {
    /// Version of storage format
    pub format_version: u32,
    /// Titor version that created the storage
    pub titor_version: String,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
    /// Configuration
    pub config: TitorConfig,
}

/// Progress callback for long-running operations
pub type ProgressCallback = Arc<dyn Fn(ProgressInfo) + Send + Sync>;

/// Information passed to progress callbacks
#[derive(Debug, Clone)]
pub struct ProgressInfo {
    /// Operation being performed
    pub operation: String,
    /// Current item being processed
    pub current_item: Option<String>,
    /// Items processed so far
    pub processed: usize,
    /// Total items to process (if known)
    pub total: Option<usize>,
    /// Bytes processed so far
    pub bytes_processed: u64,
    /// Total bytes to process (if known)
    pub total_bytes: Option<u64>,
}

impl ProgressInfo {
    /// Get progress as a percentage (0-100)
    pub fn percentage(&self) -> Option<f32> {
        match self.total {
            Some(total) if total > 0 => Some((self.processed as f32 / total as f32) * 100.0),
            _ => None,
        }
    }
}

/// Auto-checkpoint strategies
///
/// Defines when checkpoints should be created automatically.
/// This allows for hands-free operation while ensuring important
/// changes are captured.
///
/// # Examples
///
/// ```rust
/// use titor::types::AutoCheckpointStrategy;
/// use std::time::Duration;
///
/// // Checkpoint every 100 file operations
/// let strategy = AutoCheckpointStrategy::AfterOperations(100);
///
/// // Checkpoint every hour
/// let strategy = AutoCheckpointStrategy::TimeBased(Duration::from_secs(3600));
///
/// // Smart detection with custom thresholds
/// let strategy = AutoCheckpointStrategy::Smart {
///     min_files_changed: 10,
///     min_size_changed: 1_000_000, // 1MB
///     max_time_between: Duration::from_secs(86400), // 24 hours
/// };
/// ```
#[derive(Clone)]
pub enum AutoCheckpointStrategy {
    /// No automatic checkpoints
    Disabled,
    /// Checkpoint after N file operations
    AfterOperations(usize),
    /// Checkpoint after duration
    TimeBased(std::time::Duration),
    /// Smart detection of significant changes
    Smart {
        /// Minimum files changed to trigger
        min_files_changed: usize,
        /// Minimum size changed to trigger
        min_size_changed: u64,
        /// Maximum time between checkpoints
        max_time_between: std::time::Duration,
    },
    /// Custom strategy with callback
    Custom(Arc<dyn Fn(&ChangeStats) -> bool + Send + Sync>),
}

impl std::fmt::Debug for AutoCheckpointStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => write!(f, "Disabled"),
            Self::AfterOperations(n) => f.debug_tuple("AfterOperations").field(n).finish(),
            Self::TimeBased(duration) => f.debug_tuple("TimeBased").field(duration).finish(),
            Self::Smart { min_files_changed, min_size_changed, max_time_between } => f
                .debug_struct("Smart")
                .field("min_files_changed", min_files_changed)
                .field("min_size_changed", min_size_changed)
                .field("max_time_between", max_time_between)
                .finish(),
            Self::Custom(_) => write!(f, "Custom(Fn)"),
        }
    }
}

/// File operation for tracking changes
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOperation {
    /// File was added
    Added(PathBuf),
    /// File was modified
    Modified(PathBuf),
    /// File was deleted
    Deleted(PathBuf),
}

/// Options for checkpoint creation
#[derive(Clone, Default)]
pub struct CheckpointOptions {
    /// Description for the checkpoint
    pub description: Option<String>,
    /// Tags to associate with checkpoint
    pub tags: Vec<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
    /// Progress callback
    pub progress_callback: Option<ProgressCallback>,
}

impl std::fmt::Debug for CheckpointOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointOptions")
            .field("description", &self.description)
            .field("tags", &self.tags)
            .field("metadata", &self.metadata)
            .field("progress_callback", &self.progress_callback.is_some())
            .finish()
    }
}

/// Options for restore operations
#[derive(Clone, Default)]
pub struct RestoreOptions {
    /// Whether to verify hashes during restore
    pub verify_hashes: bool,
    /// Whether to preserve current file timestamps
    pub preserve_timestamps: bool,
    /// Files to exclude from restoration
    pub exclude_patterns: Vec<String>,
    /// Progress callback
    pub progress_callback: Option<ProgressCallback>,
    /// Whether to do a dry run (no actual changes)
    pub dry_run: bool,
}

impl std::fmt::Debug for RestoreOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RestoreOptions")
            .field("verify_hashes", &self.verify_hashes)
            .field("preserve_timestamps", &self.preserve_timestamps)
            .field("exclude_patterns", &self.exclude_patterns)
            .field("progress_callback", &self.progress_callback.is_some())
            .field("dry_run", &self.dry_run)
            .finish()
    }
}

/// Information about a storage object
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageObject {
    /// Hash of the object
    pub hash: String,
    /// Size in bytes (compressed)
    pub compressed_size: u64,
    /// Size in bytes (uncompressed)
    pub uncompressed_size: u64,
    /// Reference count
    pub ref_count: usize,
    /// Whether the object is compressed
    pub is_compressed: bool,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last accessed timestamp
    pub last_accessed: DateTime<Utc>,
}

/// File filter for operations
pub type FileFilter = Arc<dyn Fn(&FileEntry) -> bool + Send + Sync>;

/// Hook trait for checkpoint operations
///
/// Allows extending Titor with custom behavior at key points in the
/// checkpoint lifecycle. Implement this trait to add logging, notifications,
/// validation, or other custom logic.
///
/// # Examples
///
/// ```rust
/// use titor::types::{CheckpointHook, ChangeStats, RestoreResult};
/// use titor::checkpoint::Checkpoint;
/// use titor::Result;
///
/// struct NotificationHook {
///     webhook_url: String,
/// }
///
/// impl CheckpointHook for NotificationHook {
///     fn pre_checkpoint(&self, stats: &ChangeStats) -> Result<()> {
///         if stats.total_operations() > 100 {
///             println!("Large checkpoint coming: {} operations", stats.total_operations());
///         }
///         Ok(())
///     }
///     
///     fn post_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
///         println!("Checkpoint {} created", checkpoint.id);
///         // Send webhook notification here
///         Ok(())
///     }
///     
///     fn pre_restore(&self, from: &Checkpoint, to: &Checkpoint) -> Result<()> {
///         println!("Restoring from {} to {}", from.short_id(), to.short_id());
///         Ok(())
///     }
///     
///     fn post_restore(&self, result: &RestoreResult) -> Result<()> {
///         println!("Restore complete: {} files restored", result.files_restored);
///         Ok(())
///     }
/// }
/// ```
pub trait CheckpointHook: Send + Sync {
    /// Called before creating a checkpoint
    ///
    /// Use this to validate or prepare for checkpoint creation.
    /// Returning an error will cancel the checkpoint.
    ///
    /// # Arguments
    ///
    /// * `stats` - Statistics about changes since last checkpoint
    fn pre_checkpoint(&self, stats: &ChangeStats) -> crate::error::Result<()>;
    
    /// Called after checkpoint creation
    ///
    /// Use this for notifications, logging, or post-processing.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The newly created checkpoint
    fn post_checkpoint(&self, checkpoint: &crate::checkpoint::Checkpoint) -> crate::error::Result<()>;
    
    /// Called before restoration
    ///
    /// Use this to prepare for restoration or validate the operation.
    /// Returning an error will cancel the restore.
    ///
    /// # Arguments
    ///
    /// * `from` - Current checkpoint
    /// * `to` - Target checkpoint to restore
    fn pre_restore(&self, from: &crate::checkpoint::Checkpoint, to: &crate::checkpoint::Checkpoint) -> crate::error::Result<()>;
    
    /// Called after restoration
    ///
    /// Use this for cleanup, notifications, or verification.
    ///
    /// # Arguments
    ///
    /// * `result` - Statistics about the restore operation
    fn post_restore(&self, result: &RestoreResult) -> crate::error::Result<()>;
}

/// Default implementation of CheckpointHook that does nothing
#[derive(Debug)]
pub struct NoOpHook;

impl CheckpointHook for NoOpHook {
    fn pre_checkpoint(&self, _stats: &ChangeStats) -> crate::error::Result<()> {
        Ok(())
    }
    
    fn post_checkpoint(&self, _checkpoint: &crate::checkpoint::Checkpoint) -> crate::error::Result<()> {
        Ok(())
    }
    
    fn pre_restore(&self, _from: &crate::checkpoint::Checkpoint, _to: &crate::checkpoint::Checkpoint) -> crate::error::Result<()> {
        Ok(())
    }
    
    fn post_restore(&self, _result: &RestoreResult) -> crate::error::Result<()> {
        Ok(())
    }
}

/// State of the timeline persisted to disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineState {
    /// Current checkpoint ID
    pub current_checkpoint_id: Option<String>,
    /// Format version for future compatibility
    pub version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_change_stats() {
        let mut stats = ChangeStats::default();
        assert!(!stats.has_changes());
        
        stats.files_added = 5;
        stats.bytes_added = 1000;
        assert!(stats.has_changes());
        assert_eq!(stats.total_operations(), 5);
        assert_eq!(stats.net_size_change(), 1000);
    }
    
    #[test]
    fn test_progress_info() {
        let info = ProgressInfo {
            operation: "Scanning".to_string(),
            current_item: None,
            processed: 50,
            total: Some(100),
            bytes_processed: 0,
            total_bytes: None,
        };
        
        assert_eq!(info.percentage(), Some(50.0));
    }
} 