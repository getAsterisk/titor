//! # Titor - Time-traveling directory snapshots
//!
//! A high-performance checkpointing library that enables time-travel through
//! directory states with efficient storage and fast operations.
//!
//! ## Overview
//!
//! Titor provides a Git-like versioning system for entire directory trees, allowing you to:
//! - Create immutable snapshots (checkpoints) of directory states
//! - Restore directories to any previous checkpoint  
//! - Fork from existing checkpoints to create alternate timelines
//! - Efficiently store only changed files using content-addressable storage
//! - Verify checkpoint integrity using Merkle trees
//! - Navigate through checkpoint history with a timeline interface
//!
//! ## Architecture
//!
//! Titor uses several key components to achieve high performance:
//!
//! - **Content-Addressable Storage**: Files are stored by their content hash, enabling
//!   automatic deduplication across checkpoints
//! - **Merkle Trees**: Each checkpoint includes a Merkle tree for cryptographic verification
//!   of file integrity
//! - **Compression**: Multiple compression strategies (Fast, Balanced, Best) to optimize
//!   storage vs performance tradeoffs
//! - **Parallel Processing**: File scanning and storage operations use parallel processing
//!   for maximum throughput
//! - **Reference Counting**: Efficient garbage collection tracks which objects are still
//!   referenced by active checkpoints
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use titor::{Titor, TitorBuilder, CompressionStrategy};
//! use std::path::PathBuf;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize a new Titor instance
//! let mut titor = Titor::init(
//!     PathBuf::from("./my_project"),    // Directory to track
//!     PathBuf::from("./.titor_storage") // Where to store checkpoints
//! )?;
//!
//! // Create a checkpoint
//! let checkpoint = titor.checkpoint(Some("Initial state".to_string()))?;
//! println!("Created checkpoint: {}", checkpoint.id);
//!
//! // Make some changes to your files...
//!
//! // Create another checkpoint
//! let checkpoint2 = titor.checkpoint(Some("Added new feature".to_string()))?;
//!
//! // Restore to the first checkpoint
//! let result = titor.restore(&checkpoint.id)?;
//! println!("Restored {} files", result.files_restored);
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage
//!
//! ### Using TitorBuilder for Custom Configuration
//!
//! ```rust,no_run
//! use titor::{TitorBuilder, CompressionStrategy};
//! use std::path::PathBuf;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let titor = TitorBuilder::new()
//!     .compression_strategy(CompressionStrategy::Fast)
//!     .ignore_patterns(vec![
//!         "*.tmp".to_string(),
//!         "node_modules/**".to_string(),
//!         ".git/**".to_string()
//!     ])
//!     .max_file_size(100 * 1024 * 1024) // Skip files larger than 100MB
//!     .parallel_workers(8)
//!     .follow_symlinks(false)
//!     .build(
//!         PathBuf::from("./project"),
//!         PathBuf::from("./storage")
//!     )?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Forking Timelines
//!
//! ```rust,no_run
//! # use titor::Titor;
//! # use std::path::PathBuf;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let mut titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
//! // Fork from an existing checkpoint to create an alternate timeline
//! let fork = titor.fork("checkpoint_id", Some("Experimental branch".to_string()))?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Checkpoint Verification
//!
//! ```rust,no_run  
//! # use titor::Titor;
//! # use std::path::PathBuf;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
//! // Verify integrity of a specific checkpoint
//! let report = titor.verify_checkpoint("checkpoint_id")?;
//! if report.is_valid() {
//!     println!("Checkpoint is valid!");
//! } else {
//!     let invalid_files = report.file_checks.iter().filter(|f| !f.is_valid()).count();
//!     println!("Found {} invalid files", invalid_files);
//! }
//!
//! // Verify entire timeline
//! let timeline_report = titor.verify_timeline()?;
//! println!("Timeline verification: {:?}", timeline_report);
//! # Ok(())
//! # }
//! ```
//!
//! ## Key Concepts
//!
//! ### Checkpoints
//!
//! A checkpoint is an immutable snapshot of a directory at a specific point in time.
//! Each checkpoint contains:
//! - Unique identifier (UUID)
//! - Parent checkpoint reference (for timeline tracking)
//! - Timestamp and description
//! - Merkle root hash for verification
//! - Metadata about files and changes
//!
//! ### Timeline
//!
//! The timeline tracks the relationship between checkpoints, forming a directed acyclic
//! graph (DAG). This allows for:
//! - Linear history tracking
//! - Branching and forking
//! - Finding common ancestors
//! - Navigating checkpoint relationships
//!
//! ### Content-Addressable Storage
//!
//! Files are stored using their SHA-256 hash as the key. This provides:
//! - Automatic deduplication
//! - Integrity verification
//! - Efficient storage of similar files
//! - Immutable object store
//!
//! ### Merkle Trees
//!
//! Each checkpoint builds a Merkle tree from all tracked files, enabling:
//! - Efficient verification of checkpoint integrity
//! - Cryptographic proof of file inclusion
//! - Fast comparison between checkpoints
//!
//! ## Performance Considerations
//!
//! - **Parallel Scanning**: Directory scanning uses parallel workers for large directories
//! - **Compression**: Choose compression strategy based on your needs:
//!   - `Fast`: Best for frequently changing files
//!   - `Balanced`: Good general-purpose compression
//!   - `Best`: Maximum compression for archival
//! - **Memory Usage**: Large directories may require significant memory during scanning
//! - **Storage Growth**: Use `gc()` periodically to clean up unreferenced objects
//!
//! ## Security Considerations
//!
//! - Checkpoints are immutable once created
//! - File integrity is verified using SHA-256 hashes
//! - Merkle trees provide cryptographic verification
//! - Symlinks are stored as references (be cautious with external targets)
//! - File permissions are preserved but ownership is not
//!
//! ## Error Handling
//!
//! All operations return `Result<T, TitorError>` where `TitorError` provides
//! detailed error information including:
//! - Error category (Storage, Checkpoint, Timeline, etc.)
//! - Descriptive message
//! - Optional source error for debugging
//!
//! ## Module Organization
//!
//! - [`checkpoint`]: Checkpoint creation and metadata management
//! - [`compression`]: Compression strategies and engine
//! - [`timeline`]: Timeline structure and navigation
//! - `storage`: Content-addressable storage backend (internal module)
//! - [`verification`]: Integrity checking and verification
//! - [`types`]: Common types and data structures
//! - [`error`]: Error types and handling

// Public API modules
pub mod checkpoint;
pub mod compression;
pub mod error;
pub mod titor;
pub mod timeline;
pub mod types;
pub mod verification;

// Internal modules (not part of public API)
mod collections;
mod file_tracking;
mod merkle;
pub mod storage;
mod utils;

// Re-export main types for convenience
pub use checkpoint::{Checkpoint, CheckpointMetadata};
pub use compression::{CompressionEngine, CompressionStrategy};
pub use error::{Result, TitorError};
pub use storage::Storage;
pub use titor::{Titor, TitorBuilder};
pub use timeline::Timeline;
pub use types::*;
pub use verification::{CheckpointVerifier, TimelineVerificationReport, VerificationReport};

#[cfg(test)]
mod tests;


