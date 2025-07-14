//! Main Titor implementation
//!
//! This module provides the core Titor struct which is the main entry point
//! for all checkpoint operations including creating, restoring, forking, and
//! navigating through time.
//!
//! ## Overview
//!
//! The `Titor` struct manages the lifecycle of checkpoints and provides methods
//! for interacting with the checkpoint system. It coordinates between several
//! subsystems:
//!
//! - **Storage Backend**: Manages content-addressable storage of file objects
//! - **Timeline**: Tracks relationships between checkpoints
//! - **File Tracker**: Scans directories and detects changes
//! - **Compression Engine**: Handles file compression/decompression
//! - **Verification System**: Ensures checkpoint integrity
//!
//! ## Thread Safety
//!
//! `Titor` uses internal locking to ensure thread-safe operations. Multiple
//! operations can be performed concurrently, though some operations (like
//! checkpoint creation) may serialize access to maintain consistency.
//!
//! ## Examples
//!
//! ### Basic Usage
//!
//! ```rust,no_run
//! use titor::Titor;
//! use std::path::PathBuf;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize a new Titor instance
//! let mut titor = Titor::init(
//!     PathBuf::from("./my_project"),
//!     PathBuf::from("./.titor")
//! )?;
//!
//! // Create checkpoints
//! let cp1 = titor.checkpoint(Some("Initial commit".to_string()))?;
//! let cp2 = titor.checkpoint(Some("Added features".to_string()))?;
//!
//! // Restore to previous state
//! titor.restore(&cp1.id)?;
//! # Ok(())
//! # }
//! ```

use crate::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
use crate::compression::{CompressionEngine, CompressionStrategy};
use crate::error::{Result, TitorError};
use crate::file_tracking::{FileTracker, create_manifest, create_file_map, process_file_entry};
use crate::fs_watcher::FsWatcher;
use crate::index::CheckpointIndex;
use crate::merkle::{MerkleTree, FileEntryHashBuilder};
use crate::storage::Storage;
use crate::timeline::Timeline;
use crate::types::*;
use crate::utils;
use crate::verification::{CheckpointVerifier, TimelineVerificationReport, VerificationReport};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument, trace, warn};
use serde_json;

/// Main Titor struct for checkpoint operations
///
/// `Titor` is the primary interface for interacting with the checkpoint system.
/// It manages the storage backend, timeline tracking, and file operations.
///
/// # Examples
///
/// ```rust,no_run
/// use titor::{Titor, TitorBuilder};
/// use std::path::PathBuf;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Using direct initialization
/// let mut titor = Titor::init(
///     PathBuf::from("./project"),
///     PathBuf::from("./.titor")
/// )?;
///
/// // Using builder pattern for custom configuration
/// let mut titor = TitorBuilder::new()
///     .ignore_patterns(vec!["*.tmp".to_string()])
///     .build(
///         PathBuf::from("./project"),
///         PathBuf::from("./.titor")
///     )?;
/// # Ok(())
/// # }
/// ```
pub struct Titor {
    /// Root directory being tracked
    root_path: PathBuf,
    /// Storage backend
    storage: Arc<Storage>,
    /// Timeline structure
    timeline: Arc<RwLock<Timeline>>,
    /// Configuration
    config: TitorConfig,
    /// Auto-checkpoint strategy
    auto_checkpoint_strategy: Arc<Mutex<AutoCheckpointStrategy>>,
    /// Checkpoint hooks
    hooks: Arc<Mutex<Vec<Box<dyn CheckpointHook>>>>,
    /// File tracker
    file_tracker: FileTracker,
    /// File system watcher for real-time change tracking
    watcher: Option<FsWatcher>,
}

impl std::fmt::Debug for Titor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Titor")
            .field("root_path", &self.root_path)
            .field("storage", &self.storage)
            .field("timeline", &self.timeline)
            .field("config", &self.config)
            .field("auto_checkpoint_strategy", &self.auto_checkpoint_strategy)
            .field("hooks", &format!("<{} hooks>", self.hooks.lock().len()))
            .field("file_tracker", &self.file_tracker)
            .field("watcher", &self.watcher.is_some())
            .finish()
    }
}

impl Titor {
    /// Initialize Titor for a directory
    ///
    /// Creates a new Titor instance and initializes the storage backend. If the
    /// storage directory already exists with a different configuration, this will
    /// fail. Use [`Titor::open`] to open existing storage.
    ///
    /// # Arguments
    ///
    /// * `root_path` - The directory to track. Must exist and be readable.
    /// * `storage_path` - Where to store checkpoint data. Will be created if it doesn't exist.
    ///
    /// # Returns
    ///
    /// Returns a new `Titor` instance on success.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The root path does not exist
    /// - The storage path cannot be created
    /// - Storage initialization fails
    /// - Insufficient permissions
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use titor::Titor;
    /// use std::path::PathBuf;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut titor = Titor::init(
    ///     PathBuf::from("/home/user/project"),
    ///     PathBuf::from("/home/user/.titor")
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Panics
    ///
    /// This function should not panic under normal circumstances.
    #[instrument(skip(storage_path))]
    pub fn init(root_path: PathBuf, storage_path: PathBuf) -> Result<Self> {
        info!("Initializing Titor for {:?}", root_path);
        
        // Ensure root path exists
        if !root_path.exists() {
            return Err(TitorError::internal(format!(
                "Root path {:?} does not exist",
                root_path
            )));
        }
        
        // Create configuration
        let config = TitorConfig {
            root_path: root_path.clone(),
            storage_path: storage_path.clone(),
            max_file_size: 0,
            parallel_workers: num_cpus::get(),
            ignore_patterns: vec![],
            compression_strategy: "fast".to_string(),
            follow_symlinks: false,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        // Initialize storage
        let compression = CompressionEngine::new(CompressionStrategy::Fast);
        let storage = Storage::init_or_open(storage_path, config.clone(), compression)?;
        
        // Create file tracker
        let file_tracker = FileTracker::new(root_path.clone());
        
        // Ensure .titor is in .gitignore
        let gitignore_path = root_path.join(".gitignore");
        if let Err(e) = utils::ensure_gitignore_has_entry(&gitignore_path, ".titor") {
            warn!("Failed to update .gitignore: {}", e);
            // Continue anyway - this is not a critical error
        } else {
            debug!("Ensured .titor is in .gitignore");
        }
        
        // Initialize file system watcher
        let watcher = match FsWatcher::new(&root_path) {
            Ok(w) => {
                if let Err(e) = w.watch() {
                    warn!("Failed to start file system watcher: {}", e);
                    None
                } else {
                    info!("File system watcher initialized");
                    Some(w)
                }
            }
            Err(e) => {
                warn!("Failed to initialize file system watcher: {}", e);
                None
            }
        };
        
        Ok(Self {
            root_path,
            storage: Arc::new(storage),
            timeline: Arc::new(RwLock::new(Timeline::new())),
            config,
            auto_checkpoint_strategy: Arc::new(Mutex::new(AutoCheckpointStrategy::Disabled)),
            hooks: Arc::new(Mutex::new(Vec::new())),
            file_tracker,
            watcher,
        })
    }
    
    /// Open existing Titor storage
    ///
    /// Opens an existing Titor storage directory and loads the configuration
    /// and timeline. The root path does not need to match the original initialization
    /// path, allowing for relocated directories.
    ///
    /// # Arguments
    ///
    /// * `root_path` - The current location of the tracked directory
    /// * `storage_path` - Path to existing Titor storage
    ///
    /// # Returns
    ///
    /// Returns a `Titor` instance connected to the existing storage.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The storage path does not exist or is not a valid Titor storage
    /// - The storage is corrupted or incompatible
    /// - Configuration cannot be loaded
    /// - Timeline data is corrupted
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use titor::Titor;
    /// use std::path::PathBuf;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Open existing storage
    /// let mut titor = Titor::open(
    ///     PathBuf::from("./relocated_project"),
    ///     PathBuf::from("./.titor")
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(storage_path))]
    pub fn open(root_path: PathBuf, storage_path: PathBuf) -> Result<Self> {
        info!("Opening Titor storage at {:?}", storage_path);
        
        // Open storage
        let compression = CompressionEngine::new(CompressionStrategy::Fast);
        let storage = Storage::open(storage_path, compression)?;
        
        // Load configuration
        let config = {
            let metadata = storage.metadata().read();
            metadata.config.clone()
        };
        
        // Create file tracker
        let file_tracker = FileTracker::new(root_path.clone())
            .with_ignore_patterns(config.ignore_patterns.clone())
            .with_max_file_size(config.max_file_size)
            .with_follow_symlinks(config.follow_symlinks)
            .with_parallel_workers(config.parallel_workers);
        
        // Load timeline
        let timeline = Self::load_timeline(&storage)?;
        
        // Ensure .titor is in .gitignore
        let gitignore_path = root_path.join(".gitignore");
        if let Err(e) = utils::ensure_gitignore_has_entry(&gitignore_path, ".titor") {
            warn!("Failed to update .gitignore: {}", e);
            // Continue anyway - this is not a critical error
        } else {
            debug!("Ensured .titor is in .gitignore");
        }
        
        // Initialize file system watcher
        let watcher = match FsWatcher::new(&root_path) {
            Ok(w) => {
                if let Err(e) = w.watch() {
                    warn!("Failed to start file system watcher: {}", e);
                    None
                } else {
                    info!("File system watcher initialized");
                    Some(w)
                }
            }
            Err(e) => {
                warn!("Failed to initialize file system watcher: {}", e);
                None
            }
        };
        
        Ok(Self {
            root_path,
            storage: Arc::new(storage),
            timeline: Arc::new(RwLock::new(timeline)),
            config,
            auto_checkpoint_strategy: Arc::new(Mutex::new(AutoCheckpointStrategy::Disabled)),
            hooks: Arc::new(Mutex::new(Vec::new())),
            file_tracker,
            watcher,
        })
    }
    
    /// Create a new checkpoint
    ///
    /// Scans the tracked directory and creates an immutable snapshot of its current
    /// state. The checkpoint includes all file contents, metadata, and a Merkle tree
    /// for verification.
    ///
    /// # Arguments
    ///
    /// * `description` - Optional human-readable description of the checkpoint
    ///
    /// # Returns
    ///
    /// Returns the newly created `Checkpoint` containing its ID and metadata.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Directory scanning fails (permissions, I/O errors)
    /// - File reading fails
    /// - Storage write operations fail
    /// - Compression fails
    /// - Maximum file size is exceeded (if configured)
    ///
    /// # Performance
    ///
    /// Checkpoint creation performance depends on:
    /// - Number and size of files
    /// - Compression strategy
    /// - Storage backend performance
    /// - Available parallelism
    ///
    /// Files are processed in parallel for optimal performance.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::Titor;
    /// # use std::path::PathBuf;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// // Create checkpoint with description
    /// let checkpoint = titor.checkpoint(Some("Added login feature".to_string()))?;
    /// println!("Created checkpoint: {}", checkpoint.id);
    ///
    /// // Create checkpoint without description
    /// let checkpoint = titor.checkpoint(None)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// - Checkpoints are immutable once created
    /// - The current timeline position is updated to the new checkpoint
    /// - Only changed files are stored (deduplication)
    /// - Symlinks are preserved but their targets are not followed by default
    /// - Empty directories are preserved
    #[instrument(skip(self))]
    pub fn checkpoint(&mut self, description: Option<String>) -> Result<Checkpoint> {
        self.checkpoint_with_options(CheckpointOptions {
            description,
            ..Default::default()
        })
    }
    
    /// Create a new checkpoint with options
    ///
    /// This method provides the full checkpoint creation functionality with
    /// support for all optimization features including file system watching
    /// and index-based change detection.
    ///
    /// # Arguments
    ///
    /// * `options` - Checkpoint creation options including description, verify flag, etc.
    ///
    /// # Returns
    ///
    /// Returns the newly created `Checkpoint` containing its ID and metadata.
    #[instrument(skip(self, options))]
    pub fn checkpoint_with_options(&mut self, options: CheckpointOptions) -> Result<Checkpoint> {
        info!("Creating checkpoint: {:?}", options.description);
        let start = Instant::now();
        
        // Get current checkpoint for parent
        let parent_id = self.timeline.read().current_checkpoint_id.clone();
        
        // Load parent index if available
        let parent_index = if let Some(ref pid) = parent_id {
            match CheckpointIndex::new(&self.storage.path(), pid) {
                Ok(idx) => Some(idx),
                Err(e) => {
                    warn!("Failed to load parent index: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Check if we can use watcher-based optimization
        let use_watcher = self.watcher.as_ref()
            .map(|w| w.has_events() && !w.missed_events())
            .unwrap_or(false);
        
        let mut file_entries = if use_watcher && !options.verify.unwrap_or(false) {
            info!("Using file system watcher for optimized checkpoint");
            
            // Get events from watcher
            let events = self.watcher.as_ref().unwrap().drain()?;
            let mut entries = Vec::new();
            let mut deleted_paths = Vec::new();
            
            // Process each event
            for event in events {
                match event {
                    FsEvent::Add(path) | FsEvent::Modify(path) => {
                        // Process the file/directory
                        let full_path = self.root_path.join(&path);
                        if let Ok(Some(entry)) = process_file_entry(
                            &full_path,
                            &self.root_path,
                            self.config.max_file_size,
                            full_path.is_dir()
                        ) {
                            entries.push(entry);
                        }
                    }
                    FsEvent::Delete(path) => {
                        deleted_paths.push(path);
                    }
                }
            }
            
            // If we have a parent index, merge with unchanged files
            if let Some(ref parent_idx) = parent_index {
                let mut all_entries = Vec::new();
                
                // Add modified/new entries
                let modified_paths: std::collections::HashSet<_> = entries.iter()
                    .map(|e| e.path.clone())
                    .collect();
                all_entries.extend(entries);
                
                // Add unchanged entries from parent
                for result in parent_idx.iter() {
                    if let Ok((path, _)) = result {
                        if !modified_paths.contains(&path) && !deleted_paths.contains(&path) {
                            // Get the file entry from parent manifest
                            if let Ok(parent_manifest) = self.storage.load_manifest(parent_id.as_ref().unwrap()) {
                                if let Some(entry) = parent_manifest.files.iter().find(|e| e.path == path) {
                                    all_entries.push(entry.clone());
                                }
                            }
                        }
                    }
                }
                
                all_entries
            } else {
                entries
            }
        } else {
            // Fall back to full scan with index optimization
            debug!("Scanning directory with index optimization");
            self.file_tracker.scan_with_index(
                parent_index.as_ref(),
                options.verify.unwrap_or(false),
                Some(|info: ProgressInfo| {
                    trace!("Scanned: {:?}", info.current_item);
                })
            )?
        };
        
        // Store file contents in parallel using rayon
        debug!("Storing {} files", file_entries.len());
        let storage = Arc::clone(&self.storage);
        let root_path = self.root_path.clone();
        
        // Process files in parallel and collect results
        let processing_results: Vec<Result<(usize, u64)>> = file_entries
            .par_iter_mut()
            .enumerate()
            .map(|(idx, entry)| -> Result<(usize, u64)> {
                let file_path = root_path.join(&entry.path);
                let mut compressed_size = 0u64;
                
                if entry.is_directory {
                    // For directories, we don't store any content
                    // Just ensure it exists during scanning
                    if !file_path.exists() {
                        fs::create_dir_all(&file_path)?;
                    }
                    entry.size = 0;
                } else if entry.is_symlink {
                    // For symlinks, store the target path as content
                    if let Some(target) = &entry.symlink_target {
                        let content_str = target.to_string_lossy();
                        let content = content_str.as_bytes();
                        trace!("Storing symlink {:?} -> {:?}", entry.path, target);
                        let (_, comp_size) = storage.store_object(content, &entry.path)?;
                        // Size of symlink object is the target path length
                        entry.size = content.len() as u64;
                        compressed_size = comp_size;
                    }
                } else if file_path.exists() {
                    // Read and store file content
                    let content = fs::read(&file_path)?;
                    
                    // Only re-hash if size changed (indicates file was modified)
                    if content.len() as u64 != entry.size {
                        // File changed after initial scan, update hash
                        let actual_hash = utils::hash_data(&content);
                        entry.content_hash = actual_hash.clone();
                        let mut builder = FileEntryHashBuilder::new();
                        entry.combined_hash = builder.combined_hash(&entry.content_hash, &entry.metadata_hash);
                        entry.size = content.len() as u64;
                    }
                    
                    let (_, comp_size) = storage.store_object(&content, &entry.path)?;
                    compressed_size = comp_size;
                }
                
                Ok((idx, compressed_size))
            })
            .collect();
        
        // Check for errors and accumulate sizes
        let mut compressed_size = 0u64;
        for result in processing_results {
            let (_, comp_size) = result?;
            compressed_size += comp_size;
        }
        
        // Calculate total size
        let total_size: u64 = file_entries.iter().map(|e| e.size).sum();
        
        // Build Merkle tree using the finalised entries
        debug!("Building Merkle tree for {} files (post-storage)", file_entries.len());
        let merkle_tree = MerkleTree::from_entries(&file_entries)?;
        let merkle_root = merkle_tree.root_hash().unwrap_or_default();
        
        // Calculate change statistics
        let change_stats = if let Some(parent_id) = &parent_id {
            let parent_manifest = self.storage.load_manifest(parent_id)?;
            self.file_tracker.detect_changes(&parent_manifest)?
        } else {
            ChangeStats {
                files_added: file_entries.len(),
                bytes_added: file_entries.iter().map(|e| e.size).sum(),
                ..Default::default()
            }
        };
        
        // Call pre-checkpoint hooks
        for hook in self.hooks.lock().iter() {
            hook.pre_checkpoint(&change_stats)?;
        }
        
        // Create checkpoint metadata
        let metadata = CheckpointMetadataBuilder::new()
            .file_count(file_entries.len())
            .total_size(total_size)
            .compressed_size(compressed_size)
            .files_changed(change_stats.total_operations())
            .bytes_changed(change_stats.net_size_change() as u64)
            .build();
        
        // Create checkpoint
        let checkpoint = Checkpoint::new(
            parent_id,
            options.description,
            metadata,
            merkle_root.clone(),
        );
        
        // Store checkpoint and manifest
        self.storage.store_checkpoint(&checkpoint)?;
        let manifest = create_manifest(
            checkpoint.id.clone(),
            file_entries.clone(),
            merkle_root,
        );
        self.storage.store_manifest(&manifest)?;
        
        // Create and save index for this checkpoint
        debug!("Creating checkpoint index");
        let new_index = CheckpointIndex::new(&self.storage.path(), &checkpoint.id)?;
        for entry in &file_entries {
            // Get file metadata for index
            let file_path = self.root_path.join(&entry.path);
            if let Ok(metadata) = utils::get_file_metadata(&file_path) {
                let mtime_dt: chrono::DateTime<chrono::Utc> = metadata.modified.into();
                let index_entry = IndexEntry {
                    size: entry.size,
                    mtime_ns: mtime_dt.timestamp_nanos_opt().unwrap_or(0),
                    ctime_ns: 0, // We don't track creation time in FileMetadata
                    hash: entry.content_hash.clone(),
                    mode: entry.permissions,
                    is_dir: entry.is_directory,
                    dir_hash: None, // Will be computed later
                };
                new_index.insert(&entry.path, index_entry)?;
            }
        }
        
        // Compute directory hashes
        let mut directories: Vec<_> = file_entries.iter()
            .filter(|e| e.is_directory)
            .map(|e| e.path.clone())
            .collect();
        directories.sort_by(|a, b| b.components().count().cmp(&a.components().count())); // Process deepest first
        
        for dir_path in directories {
            new_index.update_dir_hash(&dir_path)?;
        }
        
        // Flush index to disk
        new_index.flush()?;
        
        // Flush reference count updates
        self.storage.flush_ref_counts()?;
        
        // Update timeline
        {
            let mut timeline = self.timeline.write();
            timeline.add_checkpoint(checkpoint.clone())?;
            // Mark the newly created checkpoint as current HEAD.
            // This mirrors typical VCS behaviour (e.g., git) and aligns with user expectations
            // that creating a checkpoint advances the timeline.
            timeline.set_current(&checkpoint.id)?;
        }

        // Persist timeline state
        self.save_timeline()?;
        
        // Reset watcher after checkpoint
        if let Some(ref mut watcher) = self.watcher {
            debug!("Resetting file system watcher");
            let _ = watcher.drain(); // Clear any remaining events
        }
        
        // Call post-checkpoint hooks
        for hook in self.hooks.lock().iter() {
            hook.post_checkpoint(&checkpoint)?;
        }
        
        let duration = start.elapsed();
        info!(
            "Created checkpoint {} in {:?} ({} files, {} bytes)",
            checkpoint.short_id(),
            duration,
            manifest.file_count,
            utils::format_bytes(total_size)
        );
        
        Ok(checkpoint)
    }
    
    /// Restore to a specific checkpoint
    ///
    /// Restores the tracked directory to match the exact state captured in the
    /// specified checkpoint. This operation will:
    /// - Delete files that didn't exist in the checkpoint
    /// - Restore files that existed in the checkpoint
    /// - Update file permissions to match the checkpoint
    /// - Preserve symlinks as they were
    ///
    /// # Arguments
    ///
    /// * `checkpoint_id` - The ID (or short ID prefix) of the checkpoint to restore
    ///
    /// # Returns
    ///
    /// Returns a `RestoreResult` containing statistics about the restore operation:
    /// - Number of files restored
    /// - Number of files deleted
    /// - Bytes written/deleted
    /// - Any warnings encountered
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The checkpoint ID is invalid or not found
    /// - File operations fail (permissions, disk space)
    /// - The checkpoint data is corrupted
    ///
    /// # Safety
    ///
    /// This operation modifies the filesystem and cannot be undone except by
    /// restoring to another checkpoint. Ensure you have a recent checkpoint
    /// before restoring to an older state.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::Titor;
    /// # use std::path::PathBuf;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// // Restore to a specific checkpoint
    /// let result = titor.restore("abc123")?;
    /// println!("Restored {} files, deleted {} files", 
    ///          result.files_restored, result.files_deleted);
    ///
    /// // Check for warnings
    /// if !result.warnings.is_empty() {
    ///     eprintln!("Warnings during restore:");
    ///     for warning in &result.warnings {
    ///         eprintln!("  - {}", warning);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// - Files are restored with their original permissions but not ownership
    /// - Symlinks are restored as they were (relative symlinks remain relative)
    /// - The timeline position is updated to the restored checkpoint
    /// - Restore operations are not atomic - interruption may leave partial state
    #[instrument(skip(self))]
    pub fn restore(&mut self, checkpoint_id: &str) -> Result<RestoreResult> {
        info!("Restoring to checkpoint {}", &checkpoint_id[..8.min(checkpoint_id.len())]);
        let start = Instant::now();
        
        // Load checkpoint
        let checkpoint = self.storage.load_checkpoint(checkpoint_id)?;
        let manifest = self.storage.load_manifest(checkpoint_id)?;
        
        // Get current checkpoint for hooks
        let current_checkpoint = self.timeline.read()
            .current_checkpoint()
            .cloned();
        
        // Call pre-restore hooks
        if let Some(current) = &current_checkpoint {
            for hook in self.hooks.lock().iter() {
                hook.pre_restore(current, &checkpoint)?;
            }
        }
        
        // Track restore statistics
        let mut files_restored = 0;
        let mut files_deleted = 0;
        let mut bytes_written = 0u64;
        let mut bytes_deleted = 0u64;
        let mut warnings = Vec::new();
        
        // Create map of target files
        let target_files = create_file_map(&manifest.files);
        
        // Scan current directory to find files to delete
        let current_files = self.file_tracker.scan_directory::<fn(ProgressInfo)>(None)?;
        let mut directories_to_check = std::collections::HashSet::new();
        
        for current_file in &current_files {
            if !target_files.contains_key(current_file.path.as_path()) {
                // Entry should be deleted (file or empty directory)
                let file_path = self.root_path.join(&current_file.path);
                if file_path.exists() {
                    // Track parent directories for cleanup
                    if let Some(parent) = file_path.parent() {
                        let mut parent = parent.to_path_buf();
                        while parent != self.root_path && parent.starts_with(&self.root_path) {
                            directories_to_check.insert(parent.clone());
                            if let Some(p) = parent.parent() {
                                parent = p.to_path_buf();
                            } else {
                                break;
                            }
                        }
                    }
                    
                    if current_file.is_directory {
                        // Remove empty directory entries without warning
                        if let Err(e) = utils::remove_dir_if_empty(&file_path) {
                            trace!("Could not remove directory {:?}: {}", file_path, e);
                        }
                    } else {
                        // Remove regular file or symlink
                        match fs::remove_file(&file_path) {
                            Ok(_) => {
                                files_deleted += 1;
                                bytes_deleted += current_file.size;
                                trace!("Deleted file: {:?}", current_file.path);
                            }
                            Err(e) => {
                                warnings.push(format!(
                                    "Failed to delete {:?}: {}",
                                    current_file.path, e
                                ));
                            }
                        }
                    }
                }
            }
        }
        
        // Clean up empty directories
        let mut dirs_to_check: Vec<_> = directories_to_check.into_iter().collect();
        dirs_to_check.sort_by(|a, b| b.components().count().cmp(&a.components().count())); // Sort deepest first
        
        for dir in dirs_to_check {
            if dir.exists() && dir != self.root_path {
                if let Err(e) = utils::remove_dir_if_empty(&dir) {
                    trace!("Could not remove directory {:?}: {}", dir, e);
                }
            }
        }
        
        // Restore files from checkpoint
        debug!("Restoring {} files", manifest.files.len());
        for entry in &manifest.files {
            let file_path = self.root_path.join(&entry.path);
            
            // Ensure parent directory exists
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent)?;
            }
            
            if entry.is_directory {
                // Restore directory
                if !file_path.exists() {
                    fs::create_dir_all(&file_path)?;
                    // Set directory permissions
                    utils::set_permissions(&file_path, entry.permissions)?;
                    files_restored += 1;
                }
            } else if entry.is_symlink {
                // Restore symbolic link
                if let Some(target) = &entry.symlink_target {
                    // Remove existing file/link if present
                    if file_path.exists() || file_path.symlink_metadata().is_ok() {
                        // Use symlink_metadata to check if it's a symlink even if broken
                        trace!("Removing existing file/symlink at {:?}", file_path);
                        fs::remove_file(&file_path).ok();
                    }
                    
                    // Ensure the target is relative to the symlink location for relative symlinks
                    // This handles cases where the working directory during checkpoint creation
                    // differs from restoration
                    let final_target = if target.is_relative() {
                        target.clone()
                    } else {
                        // For absolute paths, we keep them as-is but warn if they don't exist
                        if !target.exists() {
                            warnings.push(format!(
                                "Symlink target {:?} is absolute and does not exist",
                                target
                            ));
                        }
                        target.clone()
                    };
                    
                    trace!("Creating symlink {:?} -> {:?}", file_path, final_target);
                    match utils::create_symlink(&final_target, &file_path) {
                        Ok(_) => {
                            // Symlink successfully restored
                            files_restored += 1;
                            trace!("Successfully created symlink");
                        }
                        Err(e) => {
                            warnings.push(format!(
                                "Failed to create symlink {:?} -> {:?}: {}",
                                entry.path, final_target, e
                            ));
                        }
                    }
                }
            } else {
                // Restore regular file
                match self.storage.load_object(&entry.content_hash) {
                    Ok(content) => {
                        // Write file
                        fs::write(&file_path, &content)?;
                        
                        // Set permissions
                        utils::set_permissions(&file_path, entry.permissions)?;
                        
                        files_restored += 1;
                        bytes_written += content.len() as u64;
                    }
                    Err(e) => {
                        warnings.push(format!(
                            "Failed to restore {:?}: {}",
                            entry.path, e
                        ));
                    }
                }
            }
        }
        
        // Update current checkpoint
        self.timeline.write().set_current(checkpoint_id)?;
        self.save_timeline()?;
        
        // Create result
        let result = RestoreResult {
            checkpoint_id: checkpoint_id.to_string(),
            files_restored,
            files_deleted,
            bytes_written,
            bytes_deleted,
            duration_ms: start.elapsed().as_millis() as u64,
            warnings,
        };
        
        // Call post-restore hooks
        for hook in self.hooks.lock().iter() {
            hook.post_restore(&result)?;
        }
        
        info!(
            "Restored to checkpoint {} in {}ms ({} files restored, {} deleted)",
            &checkpoint_id[..8.min(checkpoint_id.len())],
            result.duration_ms,
            result.files_restored,
            result.files_deleted
        );
        
        Ok(result)
    }
    
    /// List all checkpoints
    ///
    /// Returns all checkpoints in the timeline, ordered by creation time (oldest first).
    /// This includes checkpoints from all branches if the timeline has been forked.
    ///
    /// # Returns
    ///
    /// A vector of all `Checkpoint` objects in the storage.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Timeline data cannot be accessed
    /// - Checkpoint data is corrupted
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::Titor;
    /// # use std::path::PathBuf;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// let checkpoints = titor.list_checkpoints()?;
    /// for cp in checkpoints {
    ///     println!("{}: {}", 
    ///              cp.timestamp.format("%Y-%m-%d %H:%M:%S"),
    ///              cp.description.as_ref().unwrap_or(&"No description".to_string()));
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn list_checkpoints(&self) -> Result<Vec<Checkpoint>> {
        let timeline = self.timeline.read();
        let mut checkpoints: Vec<_> = timeline.checkpoints.values().cloned().collect();
        checkpoints.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(checkpoints)
    }
    
    /// Get the timeline tree structure
    pub fn get_timeline(&self) -> Result<Timeline> {
        Ok(self.timeline.read().clone())
    }
    
    /// Fork from a checkpoint
    ///
    /// Creates a new checkpoint that branches from an existing checkpoint, allowing
    /// for alternate timelines. This is useful for experimenting with changes without
    /// affecting the main timeline.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_id` - The ID of the checkpoint to fork from
    /// * `description` - Optional description for the forked checkpoint
    ///
    /// # Returns
    ///
    /// Returns the newly created fork checkpoint.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The checkpoint ID is invalid or not found
    /// - Checkpoint creation fails
    /// - Storage operations fail
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::Titor;
    /// # use std::path::PathBuf;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let mut titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// // Fork from an existing checkpoint
    /// let fork = titor.fork("main-branch-cp-id", 
    ///                        Some("Experimental feature branch".to_string()))?;
    /// 
    /// // The fork becomes the current checkpoint
    /// // Make changes and create more checkpoints on this branch
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// - The fork operation doesn't modify files, it only creates a new checkpoint
    /// - The forked checkpoint becomes the current checkpoint
    /// - Multiple forks from the same checkpoint create sibling branches
    #[instrument(skip(self))]
    pub fn fork(&mut self, checkpoint_id: &str, description: Option<String>) -> Result<Checkpoint> {
        info!("Forking from checkpoint {}", &checkpoint_id[..8.min(checkpoint_id.len())]);
        
        // First restore to the checkpoint
        self.restore(checkpoint_id)?;
        
        // Then create a new checkpoint with the fork description
        let fork_description = description.or_else(|| {
            Some(format!("Fork from {}", &checkpoint_id[..8.min(checkpoint_id.len())]))
        });
        
        self.checkpoint(fork_description)
    }
    
    /// Compare two checkpoints (file-level differences)
    ///
    /// Computes the differences between two checkpoints, showing which files
    /// were added, modified, or deleted. This comparison is based on content
    /// hashes rather than timestamps.
    ///
    /// # Arguments
    ///
    /// * `from_id` - Source checkpoint ID (older)
    /// * `to_id` - Target checkpoint ID (newer)
    ///
    /// # Returns
    ///
    /// Returns a [`CheckpointDiff`] containing:
    /// - Lists of added, modified, and deleted files
    /// - Change statistics (file counts and sizes)
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Either checkpoint ID is not found
    /// - Checkpoint data cannot be loaded
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::Titor;
    /// # use std::path::PathBuf;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// let diff = titor.diff("checkpoint1", "checkpoint2")?;
    /// 
    /// println!("Files added: {}", diff.added_files.len());
    /// println!("Files modified: {}", diff.modified_files.len());
    /// println!("Files deleted: {}", diff.deleted_files.len());
    /// 
    /// // Show detailed changes
    /// for file in &diff.added_files {
    ///     println!("Added: {:?}", file.path);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// - Comparison is based on content hashes, not timestamps
    /// - Directory structure changes are tracked
    /// - The order of checkpoints matters for add/delete determination
    pub fn diff(&self, from_id: &str, to_id: &str) -> Result<CheckpointDiff> {
        debug!("Computing diff between {} and {}", 
               &from_id[..8.min(from_id.len())],
               &to_id[..8.min(to_id.len())]);
        
        // Load manifests
        let from_manifest = self.storage.load_manifest(from_id)?;
        let to_manifest = self.storage.load_manifest(to_id)?;
        
        // Create maps for efficient lookup
        let from_map = create_file_map(&from_manifest.files);
        let to_map = create_file_map(&to_manifest.files);
        
        let mut added_files = Vec::new();
        let mut modified_files = Vec::new();
        let mut deleted_files = Vec::new();
        let mut stats = ChangeStats::default();
        
        // Find added and modified files
        for (path, to_entry) in &to_map {
            match from_map.get(path) {
                Some(from_entry) => {
                    if to_entry.content_hash != from_entry.content_hash {
                        modified_files.push(((*from_entry).clone(), (*to_entry).clone()));
                        stats.files_modified += 1;
                        stats.bytes_modified += to_entry.size;
                        stats.changed_files.push((*path).to_path_buf());
                    }
                }
                None => {
                    added_files.push((*to_entry).clone());
                    stats.files_added += 1;
                    stats.bytes_added += to_entry.size;
                    stats.changed_files.push((*path).to_path_buf());
                }
            }
        }
        
        // Find deleted files
        for (path, from_entry) in &from_map {
            if !to_map.contains_key(path) {
                deleted_files.push((*from_entry).clone());
                stats.files_deleted += 1;
                stats.bytes_deleted += from_entry.size;
                stats.changed_files.push((*path).to_path_buf());
            }
        }
        
        Ok(CheckpointDiff {
            from_id: from_id.to_string(),
            to_id: to_id.to_string(),
            added_files,
            modified_files,
            deleted_files,
            stats,
        })
    }
    
    /// Compute line-level diff for a specific file between checkpoints
    ///
    /// This method provides detailed line-by-line differences for a text file
    /// between two checkpoints, similar to git diff output.
    ///
    /// # Arguments
    ///
    /// * `from_id` - Source checkpoint ID
    /// * `to_id` - Target checkpoint ID  
    /// * `file_path` - Path to the file to diff
    /// * `options` - Options controlling diff generation
    ///
    /// # Returns
    ///
    /// Returns a [`FileDiff`] containing:
    /// - Diff hunks with line-level changes
    /// - Line addition/deletion counts
    /// - Binary file detection
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Either checkpoint doesn't exist
    /// - The file doesn't exist in one or both checkpoints
    /// - The file content cannot be loaded
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::{Titor, types::DiffOptions};
    /// # use std::path::{Path, PathBuf};
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// let options = DiffOptions::default();
    /// let file_diff = titor.diff_file("checkpoint1", "checkpoint2", 
    ///                                  Path::new("src/main.rs"), options)?;
    /// 
    /// println!("Lines added: {}", file_diff.lines_added);
    /// println!("Lines deleted: {}", file_diff.lines_deleted);
    /// 
    /// for hunk in &file_diff.hunks {
    ///     println!("@@ -{},{} +{},{} @@", 
    ///              hunk.from_line, hunk.from_count,
    ///              hunk.to_line, hunk.to_count);
    ///     // Print the actual line changes...
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn diff_file(
        &self,
        from_id: &str,
        to_id: &str,
        file_path: &Path,
        options: crate::types::DiffOptions,
    ) -> Result<crate::types::FileDiff> {
        use crate::diff;
        
        debug!("Computing file diff for {:?} between {} and {}", 
               file_path,
               &from_id[..8.min(from_id.len())],
               &to_id[..8.min(to_id.len())]);
        
        // Load manifests
        let from_manifest = self.storage.load_manifest(from_id)?;
        let to_manifest = self.storage.load_manifest(to_id)?;
        
        // Find the file in both manifests
        let from_entry = from_manifest.files.iter()
            .find(|e| e.path == file_path)
            .ok_or_else(|| TitorError::internal(
                format!("File {:?} not found in checkpoint {}", file_path, from_id)
            ))?;
            
        let to_entry = to_manifest.files.iter()
            .find(|e| e.path == file_path)
            .ok_or_else(|| TitorError::internal(
                format!("File {:?} not found in checkpoint {}", file_path, to_id)
            ))?;
        
        // Check if the file actually changed
        if from_entry.content_hash == to_entry.content_hash {
            // No changes - return empty diff
            return Ok(crate::types::FileDiff {
                path: file_path.to_path_buf(),
                from_hash: from_entry.content_hash.clone(),
                to_hash: to_entry.content_hash.clone(),
                is_binary: false,
                hunks: vec![],
                lines_added: 0,
                lines_deleted: 0,
            });
        }
        
        // Check file size limits
        if from_entry.size > options.max_file_size || to_entry.size > options.max_file_size {
            return Err(TitorError::internal(
                format!("File {:?} exceeds maximum size for diff ({} bytes)", 
                        file_path, options.max_file_size)
            ));
        }
        
        // Load file contents
        let from_content = self.storage.load_object(&from_entry.content_hash)?;
        let to_content = self.storage.load_object(&to_entry.content_hash)?;
        
        // Create the file diff
        diff::create_file_diff(
            file_path,
            &from_entry.content_hash,
            &to_entry.content_hash,
            &from_content,
            &to_content,
            &options,
        )
    }
    
    /// Compare two checkpoints with detailed line-level differences
    ///
    /// This method extends the basic diff functionality by computing line-level
    /// differences for all modified text files, providing git-like diff output.
    ///
    /// # Arguments
    ///
    /// * `from_id` - Source checkpoint ID
    /// * `to_id` - Target checkpoint ID
    /// * `options` - Options controlling diff generation
    ///
    /// # Returns
    ///
    /// Returns a [`DetailedCheckpointDiff`] containing:
    /// - Basic file-level diff information
    /// - Line-level diffs for each modified text file
    /// - Total line addition/deletion counts
    ///
    /// # Notes
    ///
    /// - Binary files are detected and marked but not diffed
    /// - Large files may be skipped based on `options.max_file_size`
    /// - This operation may be memory intensive for large changesets
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::{Titor, types::DiffOptions};
    /// # use std::path::PathBuf;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// let options = DiffOptions {
    ///     context_lines: 3,
    ///     ignore_whitespace: false,
    ///     show_line_numbers: true,
    ///     max_file_size: 10 * 1024 * 1024, // 10MB
    /// };
    /// 
    /// let detailed_diff = titor.diff_detailed("checkpoint1", "checkpoint2", options)?;
    /// 
    /// println!("Total lines added: {}", detailed_diff.total_lines_added);
    /// println!("Total lines deleted: {}", detailed_diff.total_lines_deleted);
    /// 
    /// for file_diff in &detailed_diff.file_diffs {
    ///     if file_diff.is_binary {
    ///         println!("Binary file: {:?}", file_diff.path);
    ///     } else {
    ///         println!("Modified: {:?} (+{} -{})", 
    ///                  file_diff.path, 
    ///                  file_diff.lines_added,
    ///                  file_diff.lines_deleted);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn diff_detailed(
        &self,
        from_id: &str,
        to_id: &str,
        options: crate::types::DiffOptions,
    ) -> Result<crate::types::DetailedCheckpointDiff> {
        use crate::diff;
        
        debug!("Computing detailed diff between {} and {}", 
               &from_id[..8.min(from_id.len())],
               &to_id[..8.min(to_id.len())]);
        
        // First get the basic diff
        let basic_diff = self.diff(from_id, to_id)?;
        
        // Now compute line-level diffs for modified files
        let mut file_diffs = Vec::new();
        let mut total_lines_added = 0;
        let mut total_lines_deleted = 0;
        
        for (from_entry, to_entry) in &basic_diff.modified_files {
            // Skip files that are too large
            if from_entry.size > options.max_file_size || to_entry.size > options.max_file_size {
                debug!("Skipping large file {:?} for line diff", from_entry.path);
                continue;
            }
            
            // Load file contents
            match (
                self.storage.load_object(&from_entry.content_hash),
                self.storage.load_object(&to_entry.content_hash)
            ) {
                (Ok(from_content), Ok(to_content)) => {
                    match diff::create_file_diff(
                        &from_entry.path,
                        &from_entry.content_hash,
                        &to_entry.content_hash,
                        &from_content,
                        &to_content,
                        &options,
                    ) {
                        Ok(file_diff) => {
                            total_lines_added += file_diff.lines_added;
                            total_lines_deleted += file_diff.lines_deleted;
                            file_diffs.push(file_diff);
                        }
                        Err(e) => {
                            warn!("Failed to compute diff for {:?}: {}", from_entry.path, e);
                        }
                    }
                }
                (Err(e), _) | (_, Err(e)) => {
                    warn!("Failed to load content for {:?}: {}", from_entry.path, e);
                }
            }
        }
        
        Ok(crate::types::DetailedCheckpointDiff {
            basic_diff,
            file_diffs,
            total_lines_added,
            total_lines_deleted,
        })
    }
    
    /// Configure auto-checkpoint behavior
    pub fn set_auto_checkpoint(&mut self, strategy: AutoCheckpointStrategy) {
        *self.auto_checkpoint_strategy.lock() = strategy;
    }
    
    /// Garbage collect unreferenced content
    ///
    /// Removes objects from storage that are no longer referenced by any checkpoint.
    /// This helps reclaim disk space after checkpoints have been deleted or when
    /// content has been deduplicated.
    ///
    /// # Returns
    ///
    /// Returns `GcStats` containing:
    /// - Number of objects examined and deleted
    /// - Bytes reclaimed
    /// - List of deleted object hashes
    /// - Operation duration
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - Storage access fails
    /// - Object deletion fails (partial cleanup may occur)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::Titor;
    /// # use std::path::PathBuf;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
    /// // Run garbage collection
    /// let stats = titor.gc()?;
    /// println!("Reclaimed {} bytes by deleting {} objects",
    ///          stats.bytes_reclaimed, stats.objects_deleted);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Notes
    ///
    /// - This operation is irreversible - deleted objects cannot be recovered
    /// - Only unreferenced objects are deleted
    /// - Consider using `gc_analyze()` first to preview what would be deleted
    /// - Garbage collection may take significant time for large repositories
    #[instrument(skip(self))]
    pub fn gc(&self) -> Result<GcStats> {
        info!("Starting garbage collection");
        let start = Instant::now();
        
        let mut stats = GcStats::default();
        
        // Find unreferenced objects
        let unreferenced = self.storage.get_unreferenced_objects()?;
        stats.unreferenced_objects = unreferenced.clone();
        stats.objects_examined = self.storage.list_all_objects()?.len();
        
        // Delete unreferenced objects and track sizes
        for hash in &unreferenced {
            // Try to get object size before deletion
            match self.storage.get_object_size(hash) {
                Ok(size) => {
                    // Now delete the object
                    match self.storage.delete_object(hash) {
                        Ok(_) => {
                            stats.objects_deleted += 1;
                            stats.bytes_reclaimed += size;
                        }
                        Err(e) => {
                            warn!("Failed to delete object {}: {}", &hash[..8], e);
                        }
                    }
                }
                Err(e) => {
                    // Object might have been deleted already or is corrupted
                    warn!("Failed to get size for object {}: {}", &hash[..8], e);
                    // Try to delete anyway
                    if self.storage.delete_object(hash).is_ok() {
                        stats.objects_deleted += 1;
                    }
                }
            }
        }
        
        stats.duration_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "Garbage collection complete in {}ms: {} objects deleted, {} bytes reclaimed",
            stats.duration_ms,
            stats.objects_deleted,
            stats.bytes_reclaimed
        );
        
        Ok(stats)
    }
    
    /// Analyze garbage collection without actually deleting anything (dry run)
    #[instrument(skip(self))]
    pub fn gc_analyze(&self) -> Result<GcStats> {
        info!("Analyzing garbage collection (dry run)");
        let start = Instant::now();
        
        let mut stats = GcStats::default();
        
        // Find unreferenced objects
        let unreferenced = self.storage.get_unreferenced_objects()?;
        stats.unreferenced_objects = unreferenced.clone();
        stats.objects_examined = self.storage.list_all_objects()?.len();
        
        // Calculate sizes without deleting
        for hash in &unreferenced {
            match self.storage.get_object_size(hash) {
                Ok(size) => {
                    stats.bytes_reclaimed += size;
                }
                Err(e) => {
                    warn!("Failed to get size for object {}: {}", &hash[..8], e);
                }
            }
        }
        
        stats.duration_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "Garbage collection analysis complete in {}ms: {} objects would be deleted, {} bytes would be reclaimed",
            stats.duration_ms,
            unreferenced.len(),
            stats.bytes_reclaimed
        );
        
        Ok(stats)
    }
    
    /// Verify integrity of a specific checkpoint
    pub fn verify_checkpoint(&self, checkpoint_id: &str) -> Result<VerificationReport> {
        let checkpoint = self.storage.load_checkpoint(checkpoint_id)?;
        let verifier = CheckpointVerifier::new(&self.storage);
        verifier.verify_complete(&checkpoint)
    }
    
    /// Verify entire timeline integrity
    pub fn verify_timeline(&self) -> Result<TimelineVerificationReport> {
        let timeline = self.timeline.read();
        let verifier = CheckpointVerifier::new(&self.storage);
        verifier.verify_timeline(&timeline)
    }
    
    /// Compute merkle root for current state
    pub fn compute_current_merkle_root(&self) -> Result<String> {
        let entries = self.file_tracker.scan_directory::<fn(ProgressInfo)>(None)?;
        let tree = MerkleTree::from_entries(&entries)?;
        Ok(tree.root_hash().unwrap_or_default())
    }
    
    /// Add a checkpoint hook
    pub fn add_hook(&mut self, hook: Box<dyn CheckpointHook>) {
        self.hooks.lock().push(hook);
    }
    
    /// Load timeline from storage
    fn load_timeline(storage: &Storage) -> Result<Timeline> {
        let mut timeline = Timeline::new();
        
        // Load all checkpoints
        for checkpoint_id in storage.list_checkpoints()? {
            let checkpoint = storage.load_checkpoint(&checkpoint_id)?;
            timeline.add_checkpoint(checkpoint)?;
        }
        
        // Load current checkpoint from persistent storage
        let timeline_path = storage.root().join("timeline.json");
        if timeline_path.exists() {
            let timeline_data = fs::read_to_string(&timeline_path)?;
            if let Ok(timeline_state) = serde_json::from_str::<TimelineState>(&timeline_data) {
                timeline.current_checkpoint_id = timeline_state.current_checkpoint_id;
                debug!("Loaded current checkpoint: {:?}", timeline.current_checkpoint_id);
            }
        }
        
        Ok(timeline)
    }
    
    /// Save timeline to storage
    fn save_timeline(&self) -> Result<()> {
        // Save current checkpoint state
        let timeline_state = TimelineState {
            current_checkpoint_id: self.timeline.read().current_checkpoint_id.clone(),
            version: 1,
        };
        
        let timeline_path = self.storage.root().join("timeline.json");
        let timeline_json = serde_json::to_string_pretty(&timeline_state)?;
        utils::atomic_write(&timeline_path, timeline_json.as_bytes())?;
        
        debug!("Saved timeline state with current checkpoint: {:?}", timeline_state.current_checkpoint_id);
        Ok(())
    }
}

/// Builder pattern for Titor configuration
///
/// `TitorBuilder` provides a fluent interface for configuring Titor instances
/// with custom settings. This is the recommended way to create Titor instances
/// when you need non-default configuration.
///
/// # Examples
///
/// ```rust,no_run
/// use titor::{TitorBuilder, CompressionStrategy};
/// use std::path::PathBuf;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let titor = TitorBuilder::new()
///     .compression_strategy(CompressionStrategy::Fast)
///     .ignore_patterns(vec![
///         "*.log".to_string(),
///         "temp/**".to_string(),
///         ".git/**".to_string()
///     ])
///     .max_file_size(50 * 1024 * 1024) // 50MB
///     .parallel_workers(4)
///     .follow_symlinks(false)
///     .build(
///         PathBuf::from("./my_project"),
///         PathBuf::from("./.titor")
///     )?;
/// # Ok(())
/// # }
/// ```
///
/// # Default Values
///
/// - `compression_strategy`: `CompressionStrategy::Fast`
/// - `ignore_patterns`: Empty (but `.titor` is always ignored)
/// - `max_file_size`: 0 (no limit)
/// - `parallel_workers`: Number of CPU cores
/// - `follow_symlinks`: false
#[derive(Debug)]
pub struct TitorBuilder {
    compression_strategy: CompressionStrategy,
    ignore_patterns: Vec<String>,
    max_file_size: u64,
    parallel_workers: usize,
    follow_symlinks: bool,
}

impl TitorBuilder {
    /// Create a new builder with default settings
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::TitorBuilder;
    ///
    /// let builder = TitorBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self {
            compression_strategy: CompressionStrategy::default(),
            ignore_patterns: Vec::new(),
            max_file_size: 0,
            parallel_workers: num_cpus::get(),
            follow_symlinks: false,
        }
    }
    
    /// Set compression strategy
    ///
    /// Determines how aggressively files are compressed. Higher compression
    /// levels save storage space but increase CPU usage.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The compression strategy to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::{TitorBuilder, CompressionStrategy};
    ///
    /// let builder = TitorBuilder::new()
    ///     .compression_strategy(CompressionStrategy::Fast);
    /// ```
    pub fn compression_strategy(mut self, strategy: CompressionStrategy) -> Self {
        self.compression_strategy = strategy;
        self
    }
    
    /// Set ignore patterns
    ///
    /// Specifies glob patterns for files and directories to exclude from
    /// checkpoints. Patterns follow gitignore-style syntax.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Vector of glob patterns to ignore
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::TitorBuilder;
    ///
    /// let builder = TitorBuilder::new()
    ///     .ignore_patterns(vec![
    ///         "*.log".to_string(),
    ///         "node_modules/**".to_string(),
    ///         "target/**".to_string(),
    ///     ]);
    /// ```
    ///
    /// # Notes
    ///
    /// - The storage directory (`.titor`) is always ignored automatically
    /// - Patterns are relative to the root directory being tracked
    pub fn ignore_patterns(mut self, patterns: Vec<String>) -> Self {
        self.ignore_patterns = patterns;
        self
    }
    
    /// Set maximum file size
    ///
    /// Files larger than this size will be skipped during checkpoint creation.
    /// Use 0 for no limit.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum file size in bytes (0 = no limit)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::TitorBuilder;
    ///
    /// let builder = TitorBuilder::new()
    ///     .max_file_size(100 * 1024 * 1024); // Skip files larger than 100MB
    /// ```
    pub fn max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size;
        self
    }
    
    /// Set number of parallel workers
    ///
    /// Controls how many threads are used for parallel operations like
    /// directory scanning and file processing.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of parallel workers (minimum 1)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::TitorBuilder;
    ///
    /// let builder = TitorBuilder::new()
    ///     .parallel_workers(4); // Use 4 threads
    /// ```
    ///
    /// # Notes
    ///
    /// - Defaults to the number of CPU cores
    /// - Values less than 1 are automatically set to 1
    pub fn parallel_workers(mut self, count: usize) -> Self {
        self.parallel_workers = count.max(1);
        self
    }
    
    /// Set whether to follow symbolic links
    ///
    /// When enabled, symlinks are followed and their targets are included
    /// in checkpoints. When disabled (default), symlinks are preserved as
    /// symlinks.
    ///
    /// # Arguments
    ///
    /// * `follow` - Whether to follow symbolic links
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::TitorBuilder;
    ///
    /// let builder = TitorBuilder::new()
    ///     .follow_symlinks(false); // Preserve symlinks (default)
    /// ```
    ///
    /// # Security Considerations
    ///
    /// Following symlinks can lead to:
    /// - Including files outside the tracked directory
    /// - Circular references causing infinite loops
    /// - Increased checkpoint size
    pub fn follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }
    
    /// Build Titor instance
    ///
    /// Creates a new Titor instance with the configured settings. If the
    /// storage path already contains Titor data, it will be opened instead
    /// of initialized.
    ///
    /// # Arguments
    ///
    /// * `root_path` - Directory to track
    /// * `storage_path` - Where to store checkpoint data
    ///
    /// # Returns
    ///
    /// Returns a configured `Titor` instance.
    ///
    /// # Errors
    ///
    /// This function will return an error if:
    /// - The root path does not exist
    /// - Storage initialization/opening fails
    /// - Invalid configuration
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use titor::TitorBuilder;
    /// use std::path::PathBuf;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let titor = TitorBuilder::new()
    ///     .compression_strategy(titor::CompressionStrategy::Adaptive { min_size: 4096, skip_extensions: vec![] })
    ///     .build(
    ///         PathBuf::from("./my_project"),
    ///         PathBuf::from("./.titor")
    ///     )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self, root_path: PathBuf, storage_path: PathBuf) -> Result<Titor> {
        // Compute effective ignore patterns. Always exclude the internal storage directory (".titor")
        // so that Titor never attempts to snapshot its own repository. This prevents exponential
        // growth and potential corruption when restoring checkpoints.
        let mut effective_ignore_patterns = self.ignore_patterns.clone();
        // Ignore the storage directory itself and all its contents
        effective_ignore_patterns.push(".titor".to_string());
        effective_ignore_patterns.push(".titor/".to_string());

        // Check if storage exists by looking for metadata.json
        let storage_metadata_path = storage_path.join("metadata.json");
        
        if storage_metadata_path.exists() {
            // Open existing
            Titor::open(root_path, storage_path)
        } else {
            // For initialization, ensure the storage directory doesn't exist
            // (Storage::init requires this)
            if storage_path.exists() && storage_path.read_dir()?.next().is_none() {
                // Directory exists but is empty (common with TempDir), remove it
                std::fs::remove_dir(&storage_path).ok();
            }
            
            // Initialize new
            let mut titor = Titor::init(root_path.clone(), storage_path)?;
            
            // Update configuration
            titor.config.ignore_patterns = effective_ignore_patterns.clone();
            titor.config.max_file_size = self.max_file_size;
            titor.config.parallel_workers = self.parallel_workers;
            titor.config.follow_symlinks = self.follow_symlinks;
            
            // Update file tracker
            titor.file_tracker = FileTracker::new(root_path)
                .with_ignore_patterns(effective_ignore_patterns.clone())
                .with_max_file_size(self.max_file_size)
                .with_follow_symlinks(self.follow_symlinks)
                .with_parallel_workers(self.parallel_workers);
            
            // Update storage metadata
            titor.storage.update_metadata(|metadata| {
                metadata.config = titor.config.clone();
            })?;
            
            Ok(titor)
        }
    }
}

impl Default for TitorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    fn create_test_titor() -> (Titor, TempDir, TempDir) {
        let root_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let titor = TitorBuilder::new()
            .build(
                root_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        (titor, root_dir, storage_dir)
    }
    
    #[test]
    fn test_titor_init() {
        let root_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        // Remove the directory created by TempDir
        std::fs::remove_dir_all(storage_dir.path()).ok();
        
        let _titor = Titor::init(
            root_dir.path().to_path_buf(),
            storage_dir.path().to_path_buf(),
        ).unwrap();
        
        // Check storage structure was created
        assert!(storage_dir.path().join("metadata.json").exists());
        assert!(storage_dir.path().join("checkpoints").exists());
        assert!(storage_dir.path().join("objects").exists());
    }
    
    #[test]
    fn test_checkpoint_creation() {
        let (mut titor, root_dir, _storage_dir) = create_test_titor();
        
        // Create some files
        fs::write(root_dir.path().join("file1.txt"), "content1").unwrap();
        fs::write(root_dir.path().join("file2.txt"), "content2").unwrap();
        
        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Initial state".to_string())).unwrap();
        
        assert!(checkpoint.parent_id.is_none());
        assert_eq!(checkpoint.metadata.file_count, 2);
        assert!(checkpoint.metadata.total_size > 0);
    }
    
    #[test]
    fn test_checkpoint_restore() {
        let (mut titor, root_dir, _storage_dir) = create_test_titor();
        
        // Initial state
        fs::write(root_dir.path().join("file1.txt"), "version1").unwrap();
        let checkpoint1 = titor.checkpoint(Some("Version 1".to_string())).unwrap();
        
        // Modify files
        fs::write(root_dir.path().join("file1.txt"), "version2").unwrap();
        fs::write(root_dir.path().join("file2.txt"), "new file").unwrap();
        let _checkpoint2 = titor.checkpoint(Some("Version 2".to_string())).unwrap();
        
        // Restore to version 1
        let result = titor.restore(&checkpoint1.id).unwrap();
        
        assert_eq!(result.files_restored, 1);
        assert_eq!(result.files_deleted, 1);
        
        // Verify content
        let content = fs::read_to_string(root_dir.path().join("file1.txt")).unwrap();
        assert_eq!(content, "version1");
        assert!(!root_dir.path().join("file2.txt").exists());
    }
    
    #[test]
    fn test_diff() {
        let (mut titor, root_dir, _storage_dir) = create_test_titor();
        
        // Create checkpoints
        fs::write(root_dir.path().join("file1.txt"), "content1").unwrap();
        let checkpoint1 = titor.checkpoint(None).unwrap();
        
        fs::write(root_dir.path().join("file1.txt"), "modified").unwrap();
        fs::write(root_dir.path().join("file2.txt"), "new").unwrap();
        let checkpoint2 = titor.checkpoint(None).unwrap();
        
        // Compute diff
        let diff = titor.diff(&checkpoint1.id, &checkpoint2.id).unwrap();
        
        assert_eq!(diff.added_files.len(), 1);
        assert_eq!(diff.modified_files.len(), 1);
        assert_eq!(diff.deleted_files.len(), 0);
    }

    #[test]
    fn test_storage_dir_ignored() {
        use std::fs;
        // Create root directory with nested .titor dir (storage path)
        let root_dir = TempDir::new().unwrap();
        let storage_path = root_dir.path().join(".titor");

        // Ensure storage directory exists so builder recognises it
        fs::create_dir_all(&storage_path).unwrap();

        // Build Titor instance
        let mut titor = TitorBuilder::new()
            .build(root_dir.path().to_path_buf(), storage_path.clone())
            .unwrap();

        // Add a real file outside the storage directory
        fs::write(root_dir.path().join("data.txt"), "hello").unwrap();

        // Create checkpoint
        let checkpoint = titor.checkpoint(None).unwrap();

        // The checkpoint should only contain the user file, not the storage contents
        assert_eq!(checkpoint.metadata.file_count, 1);
        
        // Manifest should not include any path inside .titor
        let manifest = titor.storage.load_manifest(&checkpoint.id).unwrap();
        assert!(manifest.files.iter().all(|e| !e.path.starts_with(".titor")));
    }

    #[test]
    fn test_current_checkpoint_updates() {
        use std::fs;
        let (mut titor, root_dir, _storage_dir) = create_test_titor();

        // First checkpoint
        fs::write(root_dir.path().join("file1.txt"), "one").unwrap();
        let cp1 = titor.checkpoint(None).unwrap();
        assert_eq!(titor.get_timeline().unwrap().current_checkpoint_id, Some(cp1.id.clone()));

        // Second checkpoint
        fs::write(root_dir.path().join("file2.txt"), "two").unwrap();
        let cp2 = titor.checkpoint(None).unwrap();
        assert_eq!(titor.get_timeline().unwrap().current_checkpoint_id, Some(cp2.id.clone()));
    }

    #[test]
    fn test_special_character_filenames() {
        use std::fs;
        let (mut titor, root_dir, _storage_dir) = create_test_titor();

        // Create files with special characters
        let special_files = vec![
            ("file with spaces.txt", "content1"),
            ("file-with-dashes.txt", "content2"),
            ("file_with_underscores.txt", "content3"),
            ("file$with$dollar.txt", "content4"),
            ("file@with@at.txt", "content5"),
            ("file#with#hash.txt", "content6"),
            ("file(with)parens.txt", "content7"),
            ("file[with]brackets.txt", "content8"),
            ("file{with}braces.txt", "content9"),
            ("file'with'quotes.txt", "content10"),
            ("file\"with\"doublequotes.txt", "content11"),
            ("file🔥with🔥emoji.txt", "content12"),
            ("文件名.txt", "content13"), // Chinese characters
            ("файл.txt", "content14"), // Cyrillic characters
        ];

        // Create all special files
        for (filename, content) in &special_files {
            fs::write(root_dir.path().join(filename), content).unwrap();
        }

        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Special characters test".to_string())).unwrap();
        assert_eq!(checkpoint.metadata.file_count, special_files.len());

        // Delete all files
        for (filename, _) in &special_files {
            fs::remove_file(root_dir.path().join(filename)).unwrap();
        }

        // Restore checkpoint
        let result = titor.restore(&checkpoint.id).unwrap();
        assert_eq!(result.files_restored, special_files.len());
        assert!(result.warnings.is_empty(), "Warnings during restore: {:?}", result.warnings);

        // Verify all files were restored correctly
        for (filename, expected_content) in &special_files {
            let path = root_dir.path().join(filename);
            assert!(path.exists(), "File {} was not restored", filename);
            let content = fs::read_to_string(&path).unwrap();
            assert_eq!(content, *expected_content, "Content mismatch for {}", filename);
        }
    }

    #[test]
    fn test_symlink_restoration() {
        use std::fs;
        let (mut titor, root_dir, _storage_dir) = create_test_titor();

        // Create a regular file and a symlink to it
        let target_path = root_dir.path().join("target.txt");
        let symlink_path = root_dir.path().join("symlink.txt");
        
        fs::write(&target_path, "target content").unwrap();
        utils::create_symlink(&PathBuf::from("target.txt"), &symlink_path).unwrap();
        
        // Verify symlink was created correctly
        assert!(symlink_path.exists(), "Symlink was not created");
        assert!(symlink_path.symlink_metadata().unwrap().file_type().is_symlink(), "Created file is not a symlink");
        
        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Symlink test".to_string())).unwrap();
        println!("Created checkpoint with {} files", checkpoint.metadata.file_count);
        
        // Load and inspect manifest
        let manifest = titor.storage.load_manifest(&checkpoint.id).unwrap();
        for entry in &manifest.files {
            println!("Manifest entry: path={:?}, is_symlink={}, symlink_target={:?}",
                     entry.path, entry.is_symlink, entry.symlink_target);
        }
        
        // Delete symlink and target
        fs::remove_file(&symlink_path).unwrap();
        fs::remove_file(&target_path).unwrap();
        
        // Restore checkpoint
        let result = titor.restore(&checkpoint.id).unwrap();
        println!("Restore result: {} files restored, warnings: {:?}", result.files_restored, result.warnings);
        assert!(result.warnings.is_empty(), "Warnings during restore: {:?}", result.warnings);
        
        // List what files exist after restore
        for entry in fs::read_dir(root_dir.path()).unwrap() {
            let entry = entry.unwrap();
            let metadata = entry.metadata();
            let is_symlink = if let Ok(m) = &metadata {
                m.file_type().is_symlink()
            } else {
                // If metadata fails, try symlink_metadata
                entry.path().symlink_metadata()
                    .map(|m| m.file_type().is_symlink())
                    .unwrap_or(false)
            };
            println!("After restore: {:?} (symlink: {})", 
                     entry.file_name(), 
                     is_symlink);
        }
        
        // Verify symlink was restored correctly
        assert!(symlink_path.exists() || symlink_path.symlink_metadata().is_ok(), 
                "Symlink was not restored");
        assert!(symlink_path.symlink_metadata().unwrap().file_type().is_symlink(), 
                "Restored file is not a symlink");
        
        // Verify symlink points to correct target
        let restored_target = utils::read_symlink(&symlink_path).unwrap();
        assert_eq!(restored_target, PathBuf::from("target.txt"));
    }

    #[test]
    fn test_compression_size_tracking() {
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create files with different compression characteristics
        // Highly compressible text file
        let repetitive_content = "This is a test. ".repeat(10000); // ~160KB
        fs::write(temp_dir.path().join("repetitive.txt"), &repetitive_content).unwrap();
        
        // Less compressible binary data
        let binary_content: Vec<u8> = (0..10000).map(|i| (i % 256) as u8).collect();
        fs::write(temp_dir.path().join("binary.dat"), &binary_content).unwrap();
        
        // Small file (below compression threshold)
        let small_content = "Small file";
        fs::write(temp_dir.path().join("small.txt"), &small_content).unwrap();
        
        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Compression test".to_string())).unwrap();
        
        // Verify that compression is happening
        assert!(checkpoint.metadata.compressed_size < checkpoint.metadata.total_size,
                "Compressed size ({}) should be less than total size ({})",
                checkpoint.metadata.compressed_size,
                checkpoint.metadata.total_size);
        
        // The repetitive text should compress significantly
        let compression_ratio = 1.0 - (checkpoint.metadata.compressed_size as f64 / checkpoint.metadata.total_size as f64);
        assert!(compression_ratio > 0.1, // At least 10% compression
                "Compression ratio {:.2}% is too low", compression_ratio * 100.0);
        
        println!("Compression test results:");
        println!("  Total size: {} bytes", checkpoint.metadata.total_size);
        println!("  Compressed size: {} bytes", checkpoint.metadata.compressed_size);
        println!("  Compression ratio: {:.2}%", compression_ratio * 100.0);
    }
    
    #[test]
    fn test_empty_directory_preservation() {
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();

        // Create empty directories and directories with files
        fs::create_dir(temp_dir.path().join("empty_dir")).unwrap();
        fs::create_dir_all(temp_dir.path().join("nested/empty")).unwrap();
        fs::create_dir(temp_dir.path().join("dir_with_file")).unwrap();
        fs::write(temp_dir.path().join("dir_with_file/file.txt"), "content").unwrap();
        
        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Empty dirs test".to_string())).unwrap();
        
        // Delete all directories
        fs::remove_dir_all(temp_dir.path().join("empty_dir")).unwrap();
        fs::remove_dir_all(temp_dir.path().join("nested")).unwrap();
        fs::remove_dir_all(temp_dir.path().join("dir_with_file")).unwrap();
        
        // Restore checkpoint
        let result = titor.restore(&checkpoint.id).unwrap();
        assert!(result.warnings.is_empty(), "Warnings during restore: {:?}", result.warnings);
        
        // Verify empty directories were restored
        assert!(temp_dir.path().join("empty_dir").exists(), "Empty directory was not restored");
        assert!(temp_dir.path().join("nested/empty").exists(), "Nested empty directory was not restored");
        assert!(temp_dir.path().join("dir_with_file").exists(), "Directory with file was not restored");
        assert!(temp_dir.path().join("dir_with_file/file.txt").exists(), "File in directory was not restored");
    }
    
    #[test]
    fn test_line_diff_simple() {
        use crate::types::DiffOptions;
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create a file with initial content
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "line1\nline2\nline3\n").unwrap();
        let checkpoint1 = titor.checkpoint(Some("Initial".to_string())).unwrap();
        
        // Modify the file
        fs::write(&file_path, "line1\nline2 modified\nline3\nline4\n").unwrap();
        let checkpoint2 = titor.checkpoint(Some("Modified".to_string())).unwrap();
        
        // Get line diff
        let options = DiffOptions::default();
        let file_diff = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("test.txt"),
            options
        ).unwrap();
        
        // Verify diff results
        assert!(!file_diff.is_binary);
        assert_eq!(file_diff.lines_added, 2); // "line2 modified" and "line4"
        assert_eq!(file_diff.lines_deleted, 1); // "line2"
        assert!(file_diff.hunks.len() > 0);
    }
    
    #[test]
    fn test_line_diff_binary_file() {
        use crate::types::DiffOptions;
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create a binary file
        let file_path = temp_dir.path().join("binary.dat");
        let binary_content: Vec<u8> = vec![0, 1, 2, 3, 255, 254, 253, 0];
        fs::write(&file_path, &binary_content).unwrap();
        let checkpoint1 = titor.checkpoint(Some("Binary v1".to_string())).unwrap();
        
        // Modify the binary file
        let modified_binary: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 0];
        fs::write(&file_path, &modified_binary).unwrap();
        let checkpoint2 = titor.checkpoint(Some("Binary v2".to_string())).unwrap();
        
        // Get diff
        let options = DiffOptions::default();
        let file_diff = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("binary.dat"),
            options
        ).unwrap();
        
        // Verify binary detection
        assert!(file_diff.is_binary);
        assert_eq!(file_diff.hunks.len(), 0); // No hunks for binary files
        assert_eq!(file_diff.lines_added, 0);
        assert_eq!(file_diff.lines_deleted, 0);
    }
    
    #[test]
    fn test_line_diff_new_file() {
        use crate::types::DiffOptions;
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create initial checkpoint without the file
        fs::write(temp_dir.path().join("other.txt"), "other content").unwrap();
        let checkpoint1 = titor.checkpoint(Some("Before".to_string())).unwrap();
        
        // Add new file
        fs::write(temp_dir.path().join("new.txt"), "new line 1\nnew line 2\n").unwrap();
        let checkpoint2 = titor.checkpoint(Some("After".to_string())).unwrap();
        
        // Try to diff a file that doesn't exist in first checkpoint
        let options = DiffOptions::default();
        let result = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("new.txt"),
            options
        );
        
        // Should fail because file doesn't exist in first checkpoint
        assert!(result.is_err());
    }
    
    #[test]
    fn test_line_diff_context_lines() {
        use crate::types::{DiffOptions, LineChange};
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create a file with many lines
        let content = (1..=10).map(|i| format!("line{}", i)).collect::<Vec<_>>().join("\n");
        fs::write(temp_dir.path().join("context.txt"), &content).unwrap();
        let checkpoint1 = titor.checkpoint(Some("Original".to_string())).unwrap();
        
        // Modify one line in the middle
        let modified = content.replace("line5", "line5 modified");
        fs::write(temp_dir.path().join("context.txt"), &modified).unwrap();
        let checkpoint2 = titor.checkpoint(Some("Modified".to_string())).unwrap();
        
        // Get diff with custom context lines
        let mut options = DiffOptions::default();
        options.context_lines = 2;
        
        let file_diff = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("context.txt"),
            options
        ).unwrap();
        
        // Verify we have context lines
        assert_eq!(file_diff.hunks.len(), 1);
        let hunk = &file_diff.hunks[0];
        
        // Count context lines
        let context_count = hunk.changes.iter()
            .filter(|c| matches!(c, LineChange::Context(_, _)))
            .count();
        
        // Should have at least 4 context lines (2 before, 2 after)
        assert!(context_count >= 4);
    }
    
    #[test]
    fn test_detailed_diff() {
        use crate::types::DiffOptions;
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create multiple files
        fs::write(temp_dir.path().join("file1.txt"), "content1").unwrap();
        fs::write(temp_dir.path().join("file2.txt"), "content2").unwrap();
        fs::write(temp_dir.path().join("file3.txt"), "content3").unwrap();
        let checkpoint1 = titor.checkpoint(Some("Initial".to_string())).unwrap();
        
        // Modify files
        fs::write(temp_dir.path().join("file1.txt"), "content1\nmodified").unwrap();
        fs::write(temp_dir.path().join("file2.txt"), "completely different").unwrap();
        fs::remove_file(temp_dir.path().join("file3.txt")).unwrap();
        fs::write(temp_dir.path().join("file4.txt"), "new file").unwrap();
        let checkpoint2 = titor.checkpoint(Some("Changes".to_string())).unwrap();
        
        // Get detailed diff
        let options = DiffOptions::default();
        let detailed = titor.diff_detailed(&checkpoint1.id, &checkpoint2.id, options).unwrap();
        
        // Verify basic diff info
        assert_eq!(detailed.basic_diff.added_files.len(), 1); // file4.txt
        assert_eq!(detailed.basic_diff.modified_files.len(), 2); // file1.txt, file2.txt
        assert_eq!(detailed.basic_diff.deleted_files.len(), 1); // file3.txt
        
        // Verify line-level diffs were computed for modified files
        assert_eq!(detailed.file_diffs.len(), 2); // Should have diffs for 2 modified files
        assert!(detailed.total_lines_added > 0);
        assert!(detailed.total_lines_deleted > 0);
    }
    
    #[test]
    fn test_line_diff_whitespace_ignore() {
        use crate::types::DiffOptions;
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create file with whitespace
        fs::write(temp_dir.path().join("whitespace.txt"), "line1\nline2  \nline3").unwrap();
        let checkpoint1 = titor.checkpoint(Some("Original".to_string())).unwrap();
        
        // Modify whitespace only
        fs::write(temp_dir.path().join("whitespace.txt"), "line1\nline2\nline3  ").unwrap();
        let checkpoint2 = titor.checkpoint(Some("Whitespace changed".to_string())).unwrap();
        
        // Diff without ignoring whitespace
        let options_no_ignore = DiffOptions {
            ignore_whitespace: false,
            ..Default::default()
        };
        let diff_with_ws = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("whitespace.txt"),
            options_no_ignore
        ).unwrap();
        
        // Diff ignoring whitespace
        let options_ignore = DiffOptions {
            ignore_whitespace: true,
            ..Default::default()
        };
        let diff_without_ws = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("whitespace.txt"),
            options_ignore
        ).unwrap();
        
        // When not ignoring whitespace, we should see changes
        assert!(diff_with_ws.lines_added > 0 || diff_with_ws.lines_deleted > 0);
        
        // When ignoring whitespace, changes should be minimal or none
        // (This depends on the exact implementation of whitespace handling)
        assert!(diff_without_ws.lines_added <= diff_with_ws.lines_added);
        assert!(diff_without_ws.lines_deleted <= diff_with_ws.lines_deleted);
    }
    
    #[test] 
    fn test_line_diff_large_file() {
        use crate::types::DiffOptions;
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create a large file
        let large_content = "Large line\n".repeat(10000);
        fs::write(temp_dir.path().join("large.txt"), &large_content).unwrap();
        let checkpoint1 = titor.checkpoint(Some("Large file".to_string())).unwrap();
        
        // Modify it slightly
        let modified = format!("First line modified\n{}", &large_content[11..]);
        fs::write(temp_dir.path().join("large.txt"), &modified).unwrap();
        let checkpoint2 = titor.checkpoint(Some("Large file modified".to_string())).unwrap();
        
        // Try to diff with size limit
        let small_options = DiffOptions {
            max_file_size: 100, // Very small limit
            ..Default::default()
        };
        
        let result = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("large.txt"),
            small_options
        );
        
        // Should fail due to size limit
        assert!(result.is_err());
        
        // Try with larger limit
        let large_options = DiffOptions {
            max_file_size: 1024 * 1024 * 100, // 100MB
            ..Default::default()
        };
        let file_diff = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("large.txt"),
            large_options
        ).unwrap();
        
        // Should succeed and show the change
        assert_eq!(file_diff.lines_added, 1);
        assert_eq!(file_diff.lines_deleted, 1);
    }
    
    #[test]
    fn test_line_diff_unicode() {
        use crate::types::DiffOptions;
        
        let (mut titor, temp_dir, _storage_dir) = create_test_titor();
        
        // Create file with unicode content
        let unicode_content = "Hello 世界\n🦀 Rust\nΓειά σου κόσμε\n";
        fs::write(temp_dir.path().join("unicode.txt"), unicode_content).unwrap();
        let checkpoint1 = titor.checkpoint(Some("Unicode v1".to_string())).unwrap();
        
        // Modify with more unicode
        let modified = "Hello 世界\n🦀 Rust 🔥\nΓειά σου κόσμε\n新しい行\n";
        fs::write(temp_dir.path().join("unicode.txt"), modified).unwrap();
        let checkpoint2 = titor.checkpoint(Some("Unicode v2".to_string())).unwrap();
        
        // Get diff
        let options = DiffOptions::default();
        let file_diff = titor.diff_file(
            &checkpoint1.id,
            &checkpoint2.id,
            Path::new("unicode.txt"),
            options
        ).unwrap();
        
        // Verify unicode handling
        assert!(!file_diff.is_binary);
        assert_eq!(file_diff.lines_added, 2); // Modified line + new line
        assert_eq!(file_diff.lines_deleted, 1); // Original emoji line
    }
    
    #[test]
    fn test_gitignore_creation() {
        let root_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        // Remove the directory created by TempDir
        std::fs::remove_dir_all(storage_dir.path()).ok();
        
        let gitignore_path = root_dir.path().join(".gitignore");
        assert!(!gitignore_path.exists(), ".gitignore should not exist initially");
        
        // Initialize Titor
        let _titor = Titor::init(
            root_dir.path().to_path_buf(),
            storage_dir.path().to_path_buf(),
        ).unwrap();
        
        // Check that .gitignore was created
        assert!(gitignore_path.exists(), ".gitignore should be created");
        
        // Read .gitignore content
        let content = fs::read_to_string(&gitignore_path).unwrap();
        assert!(content.contains(".titor"), ".gitignore should contain .titor");
    }
    
    #[test]
    fn test_gitignore_existing_file() {
        let root_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        // Remove the directory created by TempDir
        std::fs::remove_dir_all(storage_dir.path()).ok();
        
        // Create .gitignore with existing content
        let gitignore_path = root_dir.path().join(".gitignore");
        fs::write(&gitignore_path, "*.log\nnode_modules/\n").unwrap();
        
        // Initialize Titor
        let _titor = Titor::init(
            root_dir.path().to_path_buf(),
            storage_dir.path().to_path_buf(),
        ).unwrap();
        
        // Read .gitignore content
        let content = fs::read_to_string(&gitignore_path).unwrap();
        
        // Check that existing content is preserved
        assert!(content.contains("*.log"), "Existing content should be preserved");
        assert!(content.contains("node_modules/"), "Existing content should be preserved");
        
        // Check that .titor was added
        assert!(content.contains(".titor"), ".gitignore should contain .titor");
        
        // Verify the content structure
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.contains(&"*.log"));
        assert!(lines.contains(&"node_modules/"));
        assert!(lines.contains(&".titor"));
    }
    
    #[test]
    fn test_gitignore_already_contains_titor() {
        let root_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        // Remove the directory created by TempDir
        std::fs::remove_dir_all(storage_dir.path()).ok();
        
        // Create .gitignore with .titor already in it
        let gitignore_path = root_dir.path().join(".gitignore");
        fs::write(&gitignore_path, "*.log\n.titor\nnode_modules/\n").unwrap();
        let original_content = fs::read_to_string(&gitignore_path).unwrap();
        
        // Initialize Titor
        let _titor = Titor::init(
            root_dir.path().to_path_buf(),
            storage_dir.path().to_path_buf(),
        ).unwrap();
        
        // Read .gitignore content after init
        let new_content = fs::read_to_string(&gitignore_path).unwrap();
        
        // Content should remain unchanged
        assert_eq!(original_content, new_content, ".gitignore should not be modified if .titor already exists");
    }
    
    #[test]
    fn test_gitignore_on_open() {
        let root_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        // Remove the directory created by TempDir
        std::fs::remove_dir_all(storage_dir.path()).ok();
        
        // Initialize Titor first
        let _titor = Titor::init(
            root_dir.path().to_path_buf(),
            storage_dir.path().to_path_buf(),
        ).unwrap();
        
        // Remove .gitignore
        let gitignore_path = root_dir.path().join(".gitignore");
        fs::remove_file(&gitignore_path).unwrap();
        assert!(!gitignore_path.exists());
        
        // Open existing Titor storage
        let _titor2 = Titor::open(
            root_dir.path().to_path_buf(),
            storage_dir.path().to_path_buf(),
        ).unwrap();
        
        // Check that .gitignore was recreated
        assert!(gitignore_path.exists(), ".gitignore should be created on open");
        let content = fs::read_to_string(&gitignore_path).unwrap();
        assert!(content.contains(".titor"), ".gitignore should contain .titor");
    }
} 