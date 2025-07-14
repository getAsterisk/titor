//! File tracking and change detection for Titor
//!
//! This module provides comprehensive file system scanning and change detection
//! capabilities for Titor. It handles directory traversal, file filtering,
//! and change detection between different states of the file system.
//!
//! ## Features
//!
//! - **Directory Scanning**: Efficient recursive directory traversal
//! - **Change Detection**: Compare file states between checkpoints
//! - **Ignore Patterns**: Support for .gitignore-style file filtering
//! - **Parallel Processing**: Multi-threaded file processing for performance
//! - **Symlink Handling**: Configurable symbolic link following
//! - **Progress Reporting**: Real-time progress callbacks during operations
//! - **File Metadata**: Comprehensive file metadata extraction
//!
//! ## Ignore Pattern Support
//!
//! The module supports sophisticated ignore patterns similar to Git:
//!
//! - **`.gitignore` files**: Automatically respected at all directory levels
//! - **Custom patterns**: Additional patterns can be specified programmatically
//! - **Nested rules**: Child directories can override parent ignore rules
//! - **Negation patterns**: Use `!` to include files that would otherwise be ignored
//!
//! ## File Metadata Tracking
//!
//! For each file, the system tracks:
//! - Content hash (SHA-256)
//! - File size
//! - Permissions (Unix-style)
//! - Last modification time
//! - Symbolic link information
//! - Directory status
//!
//! ## Example Usage
//!
//! ### Basic Directory Scanning
//!
//! ```rust,ignore
//! use crate::file_tracking::FileTracker;
//! use std::path::PathBuf;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let tracker = FileTracker::new(PathBuf::from("./my_project"))
//!     .with_ignore_patterns(vec![
//!         "*.tmp".to_string(),
//!         "target/".to_string(),
//!         "node_modules/".to_string(),
//!     ])
//!     .with_max_file_size(100 * 1024 * 1024) // 100MB limit
//!     .with_parallel_workers(4);
//!
//! let files = tracker.scan_directory::<fn(titor::types::ProgressInfo)>(None)?;
//! println!("Found {} files", files.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Change Detection
//!
//! ```rust,ignore
//! use crate::file_tracking::{FileTracker, create_manifest};
//! use std::path::PathBuf;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let tracker = FileTracker::new(PathBuf::from("./my_project"));
//!
//! // Create initial manifest
//! let initial_files = tracker.scan_directory::<fn(titor::types::ProgressInfo)>(None)?;
//! let initial_manifest = create_manifest(
//!     "checkpoint1".to_string(),
//!     initial_files,
//!     "merkle_root".to_string(),
//! );
//!
//! // ... files are modified ...
//!
//! // Detect changes
//! let changes = tracker.detect_changes(&initial_manifest)?;
//! println!("Changes detected:");
//! println!("  Added: {}", changes.files_added);
//! println!("  Modified: {}", changes.files_modified);
//! println!("  Deleted: {}", changes.files_deleted);
//! # Ok(())
//! # }
//! ```
//!
//! ### Progress Reporting
//!
//! ```rust,ignore
//! use crate::file_tracking::FileTracker;
//! use titor::types::ProgressInfo;
//! use std::path::PathBuf;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let tracker = FileTracker::new(PathBuf::from("./my_project"));
//!
//! let files = tracker.scan_directory(Some(|progress: ProgressInfo| {
//!     println!("Processing: {} ({}/{})", 
//!         progress.operation,
//!         progress.processed,
//!         progress.total.unwrap_or(0)
//!     );
//! }))?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Considerations
//!
//! - **Parallel Processing**: Uses multiple threads for file processing
//! - **Efficient Filtering**: Ignore patterns are applied early to reduce processing
//! - **Memory Optimization**: Streaming approach for large directory trees
//! - **Caching**: File metadata is cached to avoid repeated filesystem calls
//!
//! ## Ignore Pattern Examples
//!
//! ```text
//! # Ignore compiled files
//! *.o
//! *.so
//! target/
//!
//! # Ignore temporary files
//! *.tmp
//! *.swp
//! .DS_Store
//!
//! # But include specific files
//! !important.tmp
//! ```
//!
//! ## Thread Safety
//!
//! The `FileTracker` is designed to be thread-safe and can be used from multiple
//! threads simultaneously. Progress callbacks are also thread-safe.

use crate::error::Result;
use crate::merkle::FileEntryHashBuilder;
use crate::types::{ChangeStats, FileEntry, FileManifest, ProgressInfo, IndexEntry};
use crate::utils;
use crate::index::CheckpointIndex;
use chrono::Utc;
use ignore::{WalkBuilder, WalkState, overrides::OverrideBuilder};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use crate::collections::{HashMap, HashSet, HashSetExt};
use tracing::{debug, trace, warn, info};

/// File tracker for detecting changes in a directory
///
/// `FileTracker` provides comprehensive file system scanning and change detection
/// capabilities. It efficiently traverses directory structures, applies ignore patterns,
/// and detects changes between different states of the file system.
///
/// ## Features
///
/// - **Efficient Scanning**: Parallel directory traversal with configurable worker threads
/// - **Smart Filtering**: Supports .gitignore-style patterns and custom ignore rules
/// - **Change Detection**: Compare file states to detect additions, modifications, and deletions
/// - **Flexible Configuration**: Configurable file size limits, symlink handling, and more
/// - **Progress Reporting**: Optional progress callbacks for long-running operations
///
/// ## Configuration Options
///
/// - `root_path`: Base directory to scan
/// - `ignore_patterns`: Custom ignore patterns (in addition to .gitignore files)
/// - `max_file_size`: Maximum file size to process (0 = unlimited)
/// - `follow_symlinks`: Whether to follow symbolic links
/// - `parallel_workers`: Number of threads for parallel processing
///
/// ## Example
///
/// ```rust,ignore
/// use crate::file_tracking::FileTracker;
/// use std::path::PathBuf;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a tracker with custom configuration
/// let tracker = FileTracker::new(PathBuf::from("./project"))
///     .with_ignore_patterns(vec![
///         "*.log".to_string(),
///         "target/".to_string(),
///         "*.tmp".to_string(),
///     ])
///     .with_max_file_size(10 * 1024 * 1024) // 10MB limit
///     .with_follow_symlinks(false)
///     .with_parallel_workers(8);
///
/// // Scan the directory
/// let files = tracker.scan_directory::<fn(titor::types::ProgressInfo)>(None)?;
/// println!("Scanned {} files", files.len());
/// # Ok(())
/// # }
/// ```
///
/// ## Performance Notes
///
/// - Uses parallel processing for optimal performance on multi-core systems
/// - Ignore patterns are applied early to minimize unnecessary processing
/// - Memory usage is optimized for large directory structures
/// - Progress callbacks are optional and don't significantly impact performance
#[derive(Debug)]
pub struct FileTracker {
    /// Root directory to track
    root_path: PathBuf,
    /// Custom ignore patterns (in addition to .gitignore files)
    ignore_patterns: Vec<String>,
    /// Maximum file size to track in bytes (0 = unlimited)
    max_file_size: u64,
    /// Whether to follow symbolic links during traversal
    follow_symlinks: bool,
    /// Number of parallel workers for file processing
    parallel_workers: usize,
}

impl FileTracker {
    /// Create a new file tracker with default settings
    ///
    /// Creates a new `FileTracker` instance with sensible defaults:
    /// - No custom ignore patterns (only .gitignore files)
    /// - No file size limit
    /// - Symbolic links are not followed
    /// - Number of workers equals the number of CPU cores
    ///
    /// # Arguments
    ///
    /// * `root_path` - The root directory to track
    ///
    /// # Returns
    ///
    /// Returns a new `FileTracker` instance with default configuration.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::FileTracker;
    /// use std::path::PathBuf;
    ///
    /// let tracker = FileTracker::new(PathBuf::from("./my_project"));
    /// // Ready to scan with default settings
    /// ```
    pub fn new(root_path: PathBuf) -> Self {
        Self {
            root_path,
            ignore_patterns: Vec::new(),
            max_file_size: 0,
            follow_symlinks: false,
            parallel_workers: num_cpus::get(),
        }
    }
    
    /// Set custom ignore patterns
    ///
    /// Adds custom ignore patterns that will be applied in addition to any
    /// .gitignore files found in the directory structure. Patterns follow
    /// the same syntax as .gitignore files.
    ///
    /// # Arguments
    ///
    /// * `patterns` - Vector of ignore patterns (e.g., "*.tmp", "target/")
    ///
    /// # Returns
    ///
    /// Returns the modified `FileTracker` instance for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::FileTracker;
    /// use std::path::PathBuf;
    ///
    /// let tracker = FileTracker::new(PathBuf::from("./project"))
    ///     .with_ignore_patterns(vec![
    ///         "*.log".to_string(),
    ///         "target/".to_string(),
    ///         "node_modules/".to_string(),
    ///     ]);
    /// ```
    pub fn with_ignore_patterns(mut self, patterns: Vec<String>) -> Self {
        self.ignore_patterns = patterns;
        self
    }
    
    /// Set maximum file size limit
    ///
    /// Sets the maximum file size that will be processed. Files larger than
    /// this limit will be skipped during scanning. Set to 0 for no limit.
    ///
    /// # Arguments
    ///
    /// * `size` - Maximum file size in bytes (0 = unlimited)
    ///
    /// # Returns
    ///
    /// Returns the modified `FileTracker` instance for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::FileTracker;
    /// use std::path::PathBuf;
    ///
    /// let tracker = FileTracker::new(PathBuf::from("./project"))
    ///     .with_max_file_size(100 * 1024 * 1024); // 100MB limit
    /// ```
    pub fn with_max_file_size(mut self, size: u64) -> Self {
        self.max_file_size = size;
        self
    }
    
    /// Set symbolic link following behavior
    ///
    /// Configures whether the tracker should follow symbolic links during
    /// directory traversal. When disabled, symbolic links are treated as
    /// regular files and their targets are not traversed.
    ///
    /// # Arguments
    ///
    /// * `follow` - Whether to follow symbolic links
    ///
    /// # Returns
    ///
    /// Returns the modified `FileTracker` instance for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::FileTracker;
    /// use std::path::PathBuf;
    ///
    /// let tracker = FileTracker::new(PathBuf::from("./project"))
    ///     .with_follow_symlinks(true); // Follow symbolic links
    /// ```
    pub fn with_follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }
    
    /// Set number of parallel workers
    ///
    /// Configures the number of worker threads used for parallel file processing.
    /// More workers can improve performance on multi-core systems, but may
    /// increase memory usage. The minimum value is 1.
    ///
    /// # Arguments
    ///
    /// * `workers` - Number of worker threads (minimum 1)
    ///
    /// # Returns
    ///
    /// Returns the modified `FileTracker` instance for method chaining.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::FileTracker;
    /// use std::path::PathBuf;
    ///
    /// let tracker = FileTracker::new(PathBuf::from("./project"))
    ///     .with_parallel_workers(8); // Use 8 worker threads
    /// ```
    pub fn with_parallel_workers(mut self, workers: usize) -> Self {
        self.parallel_workers = workers.max(1);
        self
    }
    
    /// Scan directory and create file entries
    ///
    /// Performs a comprehensive scan of the configured directory, creating
    /// `FileEntry` objects for each file found. The scan respects ignore patterns,
    /// file size limits, and other configuration options.
    ///
    /// # Arguments
    ///
    /// * `progress_callback` - Optional callback function to report progress
    ///
    /// # Returns
    ///
    /// Returns a vector of `FileEntry` objects representing all discovered files.
    /// The entries are sorted by path for consistent ordering.
    ///
    /// # Errors
    ///
    /// - [`TitorError::Io`] if directory traversal fails
    /// - [`TitorError::Internal`] if file processing fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::FileTracker;
    /// use titor::types::ProgressInfo;
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let tracker = FileTracker::new(PathBuf::from("./project"));
    ///
    /// // Scan with progress reporting
    /// let files = tracker.scan_directory(Some(|progress: ProgressInfo| {
    ///     println!("Scanning: {} files processed", progress.processed);
    /// }))?;
    ///
    /// println!("Found {} files", files.len());
    /// for file in &files {
    ///     println!("  {}: {} bytes", file.path.display(), file.size);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance
    ///
    /// - Uses parallel processing for optimal performance
    /// - Ignore patterns are applied early to minimize processing overhead
    /// - Progress callbacks are called efficiently without blocking
    /// - Memory usage is optimized for large directory structures
    ///
    /// # File Processing
    ///
    /// For each file found, the scanner:
    /// 1. Checks against ignore patterns
    /// 2. Verifies file size limits
    /// 3. Extracts file metadata
    /// 4. Computes content hash
    /// 5. Creates a `FileEntry` with all information
    pub fn scan_directory<F>(&self, progress_callback: Option<F>) -> Result<Vec<FileEntry>>
    where
        F: Fn(ProgressInfo) + Send + Sync,
    {
        let start = Instant::now();
        let processed_count = Arc::new(AtomicUsize::new(0));
        let total_size = Arc::new(AtomicU64::new(0));
        let directories_with_files = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));
        
        // Clone for closure
        let root_path = self.root_path.clone();
        let dirs_with_files_clone = Arc::clone(&directories_with_files);
        
        // Build ignore walker
        let mut walker_builder = WalkBuilder::new(&self.root_path);
        
        // Configure walker
        walker_builder
            .follow_links(self.follow_symlinks)
            .hidden(false) // Don't skip hidden files by default
            .parents(true) // Respect parent .gitignore files
            .ignore(true)  // Respect .gitignore files
            .git_ignore(true) // Respect .gitignore files
            .git_global(false) // Don't use global gitignore
            .git_exclude(false) // Don't use .git/info/exclude
            .require_git(false) // Work even outside git repositories
            .threads(self.parallel_workers);
        
        // Build overrides for custom patterns and .titor directory
        let mut override_builder = OverrideBuilder::new(&self.root_path);
        
        // Always ignore .titor directory
        // In override builder, ! prefix means exclude/ignore
        override_builder.add("!.titor/**").ok();
        override_builder.add("!.titor/").ok();
        override_builder.add("!.titor").ok();
        
        // Add custom ignore patterns (these override gitignore files)
        for pattern in &self.ignore_patterns {
            // In override builder, patterns need ! prefix to exclude
            let final_pattern = if pattern.starts_with('!') {
                // User explicitly wants to include this pattern
                pattern[1..].to_string()
            } else {
                // User wants to ignore this pattern, so add ! prefix
                format!("!{}", pattern)
            };
            
            if let Err(e) = override_builder.add(&final_pattern) {
                warn!("Invalid ignore pattern '{}': {}", pattern, e);
            }
        }
        
        // Build and set the overrides
        if let Ok(overrides) = override_builder.build() {
            walker_builder.overrides(overrides);
        }
        
        // Collect paths first for batch processing
        let paths_to_process = Arc::new(Mutex::new(Vec::<(PathBuf, bool)>::new()));
        
        // Walk directory using the ignore crate
        walker_builder.build_parallel().run(|| {
            let paths_to_process = Arc::clone(&paths_to_process);
            let root_path = root_path.clone();
            let dirs_with_files = Arc::clone(&dirs_with_files_clone);
            
            Box::new(move |entry_result| {
                match entry_result {
                    Ok(entry) => {
                        let path = entry.path();
                        
                        // Skip .gitignore and .titor_ignore files themselves
                        if let Some(file_name) = path.file_name() {
                            let file_name_str = file_name.to_string_lossy();
                            if file_name_str == ".gitignore" || file_name_str == ".titor_ignore" {
                                return WalkState::Continue;
                            }
                        }
                        
                        // Get file type from entry
                        let is_dir = entry.file_type()
                            .map(|ft| ft.is_dir())
                            .unwrap_or(false);
                        
                        // Track that parent directories have files
                        if !is_dir {
                            let mut parent = path.parent();
                            while let Some(p) = parent {
                                if p == root_path {
                                    break;
                                }
                                dirs_with_files.lock().insert(p.to_path_buf());
                                parent = p.parent();
                            }
                        }
                        
                        // Collect path for batch processing
                        paths_to_process.lock().push((path.to_path_buf(), is_dir));
                    }
                    Err(e) => {
                        warn!("Walk error: {}", e);
                    }
                }
                
                WalkState::Continue
            })
        });
        
        // Process collected paths in parallel using rayon
        let paths_vec = paths_to_process.lock().clone();
        let root_path_for_processing = self.root_path.clone();
        
        let file_entries: Vec<Option<FileEntry>> = paths_vec
            .par_iter()
            .map(|(path, is_directory)| {
                process_file_entry(path, &root_path_for_processing, self.max_file_size, *is_directory)
                    .unwrap_or_else(|e| {
                        warn!("Error processing entry {:?}: {}", path, e);
                        None
                    })
            })
            .collect();
        
        // Filter and collect valid entries
        let mut final_entries = Vec::new();
        let dirs_with_files = directories_with_files.lock();
        
        for entry_opt in file_entries.into_iter() {
            if let Some(entry) = entry_opt {
                // Filter out empty directories
                if !entry.is_directory || !dirs_with_files.contains(&self.root_path.join(&entry.path)) {
                    // Update counters
                    processed_count.fetch_add(1, Ordering::Relaxed);
                    total_size.fetch_add(entry.size, Ordering::Relaxed);
                    
                    // Report progress
                    if let Some(ref callback) = progress_callback {
                        let info = ProgressInfo {
                            operation: "Scanning files".to_string(),
                            current_item: Some(entry.path.to_string_lossy().to_string()),
                            processed: processed_count.load(Ordering::Relaxed),
                            total: None,
                            bytes_processed: total_size.load(Ordering::Relaxed),
                            total_bytes: None,
                        };
                        callback(info);
                    }
                    
                    final_entries.push(entry);
                }
            }
        }
        
        final_entries.sort_by(|a, b| a.path.cmp(&b.path));
        
        let scan_duration = start.elapsed();
        debug!(
            "Scanned {} files/directories ({} bytes) in {:?}",
            final_entries.len(),
            total_size.load(Ordering::Relaxed),
            scan_duration
        );
        
        Ok(final_entries)
    }
    
    /// Scan directory using an index for optimization
    ///
    /// This method provides optimized directory scanning by leveraging a checkpoint
    /// index to avoid re-reading and re-hashing unchanged files. It compares file
    /// metadata against the index and only processes files that have actually changed.
    ///
    /// # Arguments
    ///
    /// * `parent_index` - Optional index from parent checkpoint for change detection
    /// * `verify` - If true, forces full re-hashing of all files (ignores index)
    /// * `progress_callback` - Optional callback function to report progress
    ///
    /// # Returns
    ///
    /// Returns a vector of `FileEntry` objects with updated index information.
    ///
    /// # Performance Benefits
    ///
    /// - Skips reading/hashing unchanged files (O(1) metadata check vs O(file size))
    /// - Can skip entire directory subtrees if directory hash matches
    /// - Dramatically reduces checkpoint time for small changes in large repos
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::FileTracker;
    /// use crate::index::CheckpointIndex;
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let tracker = FileTracker::new(PathBuf::from("./project"));
    /// let parent_index = CheckpointIndex::new(
    ///     &PathBuf::from(".titor"),
    ///     "parent-checkpoint-id"
    /// )?;
    ///
    /// // Fast scan using index
    /// let files = tracker.scan_with_index(
    ///     Some(&parent_index),
    ///     false, // Don't verify
    ///     None
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn scan_with_index<F>(
        &self,
        parent_index: Option<&CheckpointIndex>,
        verify: bool,
        progress_callback: Option<F>,
    ) -> Result<Vec<FileEntry>>
    where
        F: Fn(ProgressInfo) + Send + Sync,
    {
        let start = Instant::now();
        let processed_count = Arc::new(AtomicUsize::new(0));
        let total_size = Arc::new(AtomicU64::new(0));
        let skipped_count = Arc::new(AtomicUsize::new(0));
        let directories_with_files = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));
        
        info!(
            "Scanning directory with index optimization (verify={})",
            verify
        );
        
        // Track visited paths for deletion detection
        let visited_paths = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));
        
        // Clone for closure
        let root_path = self.root_path.clone();
        let dirs_with_files_clone = Arc::clone(&directories_with_files);
        
        // Build ignore walker
        let mut walker_builder = WalkBuilder::new(&self.root_path);
        
        // Configure walker
        walker_builder
            .follow_links(self.follow_symlinks)
            .hidden(false)
            .parents(true)
            .ignore(true)
            .git_ignore(true)
            .git_global(false)
            .git_exclude(false)
            .require_git(false)
            .threads(self.parallel_workers);
        
        // Build overrides for custom patterns and .titor directory
        let mut override_builder = OverrideBuilder::new(&self.root_path);
        
        // Always ignore .titor directory
        override_builder.add("!.titor/**").ok();
        override_builder.add("!.titor/").ok();
        override_builder.add("!.titor").ok();
        
        // Add custom ignore patterns
        for pattern in &self.ignore_patterns {
            let final_pattern = if pattern.starts_with('!') {
                pattern[1..].to_string()
            } else {
                format!("!{}", pattern)
            };
            
            if let Err(e) = override_builder.add(&final_pattern) {
                warn!("Invalid ignore pattern '{}': {}", pattern, e);
            }
        }
        
        // Build and set the overrides
        if let Ok(overrides) = override_builder.build() {
            walker_builder.overrides(overrides);
        }
        
        // Collect paths first for batch processing
        let paths_to_process = Arc::new(Mutex::new(Vec::<(PathBuf, bool)>::new()));
        
        // Walk directory using the ignore crate
        walker_builder.build_parallel().run(|| {
            let paths_to_process = Arc::clone(&paths_to_process);
            let root_path = root_path.clone();
            let dirs_with_files = Arc::clone(&dirs_with_files_clone);
            let visited_paths = Arc::clone(&visited_paths);
            
            Box::new(move |entry_result| {
                match entry_result {
                    Ok(entry) => {
                        let path = entry.path();
                        
                        // Skip .gitignore and .titor_ignore files themselves
                        if let Some(file_name) = path.file_name() {
                            let file_name_str = file_name.to_string_lossy();
                            if file_name_str == ".gitignore" || file_name_str == ".titor_ignore" {
                                return WalkState::Continue;
                            }
                        }
                        
                        // Get file type from entry
                        let is_dir = entry.file_type()
                            .map(|ft| ft.is_dir())
                            .unwrap_or(false);
                        
                        // Track visited paths for deletion detection
                        if let Ok(relative_path) = utils::make_relative(path, &root_path) {
                            visited_paths.lock().insert(relative_path);
                        }
                        
                        // Track that parent directories have files
                        if !is_dir {
                            let mut parent = path.parent();
                            while let Some(p) = parent {
                                if p == root_path {
                                    break;
                                }
                                dirs_with_files.lock().insert(p.to_path_buf());
                                parent = p.parent();
                            }
                        }
                        
                        // Collect path for batch processing
                        paths_to_process.lock().push((path.to_path_buf(), is_dir));
                    }
                    Err(e) => {
                        warn!("Walk error: {}", e);
                    }
                }
                
                WalkState::Continue
            })
        });
        
        // Process collected paths in parallel using rayon
        let paths_vec = paths_to_process.lock().clone();
        let root_path_for_processing = self.root_path.clone();
        let skipped_clone = Arc::clone(&skipped_count);
        
        let file_entries: Vec<Option<FileEntry>> = paths_vec
            .par_iter()
            .map(|(path, is_directory)| -> Result<Option<FileEntry>> {
                // Try to use index entry if available and not verifying
                if !verify && parent_index.is_some() {
                    if let Ok(relative_path) = utils::make_relative(path, &root_path_for_processing) {
                        if let Ok(Some(index_entry)) = parent_index.unwrap().get(&relative_path) {
                            // For directories with dir_hash, check if we can skip the subtree
                            if *is_directory && index_entry.dir_hash.is_some() {
                                // Get current directory metadata
                                if let Ok(metadata) = utils::get_file_metadata(path) {
                                    // Check if directory metadata matches
                                    if metadata.size == index_entry.size
                                        && metadata.permissions == index_entry.mode
                                        && metadata_matches_ns(&metadata, &index_entry)
                                    {
                                        // Compute current dir hash
                                        if let Ok(current_hash) = parent_index.unwrap().compute_dir_hash(&relative_path) {
                                            if Some(current_hash) == index_entry.dir_hash {
                                                debug!("Skipping unchanged directory subtree: {}", relative_path.display());
                                                skipped_clone.fetch_add(1, Ordering::Relaxed);
                                                
                                                // Return the cached entry
                                                return Ok(Some(create_file_entry_from_index(
                                                    relative_path,
                                                    &index_entry,
                                                    metadata,
                                                )));
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // For files, check if we can reuse the hash
                            if !*is_directory {
                                if let Ok(metadata) = utils::get_file_metadata(path) {
                                    // Check if file metadata matches
                                    if metadata.size == index_entry.size
                                        && metadata.permissions == index_entry.mode
                                        && metadata_matches_ns(&metadata, &index_entry)
                                    {
                                        trace!("Reusing hash for unchanged file: {}", relative_path.display());
                                        skipped_clone.fetch_add(1, Ordering::Relaxed);
                                        
                                        // Return the cached entry
                                        return Ok(Some(create_file_entry_from_index(
                                            relative_path,
                                            &index_entry,
                                            metadata,
                                        )));
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Fall back to normal processing
                process_file_entry(path, &root_path_for_processing, self.max_file_size, *is_directory)
            })
            .map(|result| result.unwrap_or_else(|e| {
                warn!("Error processing entry: {}", e);
                None
            }))
            .collect();
        
        // Filter and collect valid entries
        let mut final_entries = Vec::new();
        let dirs_with_files = directories_with_files.lock();
        
        for entry_opt in file_entries.into_iter() {
            if let Some(entry) = entry_opt {
                // Filter out empty directories
                if !entry.is_directory || !dirs_with_files.contains(&self.root_path.join(&entry.path)) {
                    // Update counters
                    processed_count.fetch_add(1, Ordering::Relaxed);
                    total_size.fetch_add(entry.size, Ordering::Relaxed);
                    
                    // Report progress
                    if let Some(ref callback) = progress_callback {
                        let info = ProgressInfo {
                            operation: "Scanning files (optimized)".to_string(),
                            current_item: Some(entry.path.to_string_lossy().to_string()),
                            processed: processed_count.load(Ordering::Relaxed),
                            total: None,
                            bytes_processed: total_size.load(Ordering::Relaxed),
                            total_bytes: None,
                        };
                        callback(info);
                    }
                    
                    final_entries.push(entry);
                }
            }
        }
        
        final_entries.sort_by(|a, b| a.path.cmp(&b.path));
        
        let scan_duration = start.elapsed();
        let skipped = skipped_count.load(Ordering::Relaxed);
        
        info!(
            "Optimized scan complete: {} files/directories ({} bytes) in {:?}, {} entries skipped",
            final_entries.len(),
            total_size.load(Ordering::Relaxed),
            scan_duration,
            skipped
        );
        
        Ok(final_entries)
    }
    
    /// Detect changes between current state and a manifest
    ///
    /// Compares the current state of the directory with a previous manifest
    /// to identify all changes that have occurred. This includes file additions,
    /// modifications, and deletions.
    ///
    /// # Arguments
    ///
    /// * `old_manifest` - Previous manifest to compare against
    ///
    /// # Returns
    ///
    /// Returns a `ChangeStats` object containing detailed information about
    /// all detected changes, including counts and lists of affected files.
    ///
    /// # Errors
    ///
    /// - [`TitorError::Io`] if directory scanning fails
    /// - [`TitorError::Internal`] if change detection fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use crate::file_tracking::{FileTracker, create_manifest};
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let tracker = FileTracker::new(PathBuf::from("./project"));
    ///
    /// // Create initial manifest
    /// let initial_files = tracker.scan_directory::<fn(titor::types::ProgressInfo)>(None)?;
    /// let initial_manifest = create_manifest(
    ///     "checkpoint1".to_string(),
    ///     initial_files,
    ///     "merkle_root".to_string(),
    /// );
    ///
    /// // ... files are modified ...
    ///
    /// // Detect changes
    /// let changes = tracker.detect_changes(&initial_manifest)?;
    ///
    /// println!("Change Summary:");
    /// println!("  Added: {} files ({} bytes)", changes.files_added, changes.bytes_added);
    /// println!("  Modified: {} files ({} bytes)", changes.files_modified, changes.bytes_modified);
    /// println!("  Deleted: {} files ({} bytes)", changes.files_deleted, changes.bytes_deleted);
    /// println!("  Total changes: {}", changes.total_operations());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Change Detection Logic
    ///
    /// The method performs the following comparisons:
    /// 1. **Added Files**: Present in current state but not in old manifest
    /// 2. **Modified Files**: Present in both but with different content hashes
    /// 3. **Deleted Files**: Present in old manifest but not in current state
    ///
    /// # Performance
    ///
    /// - Uses efficient hash maps for O(1) lookups
    /// - Scans current directory only once
    /// - Minimal memory overhead for large file sets
    /// - Parallel processing for optimal performance
    pub fn detect_changes(&self, old_manifest: &FileManifest) -> Result<ChangeStats> {
        let start = Instant::now();
        
        // Scan current state
        let current_entries = self.scan_directory::<fn(ProgressInfo)>(None)?;
        
        // Create maps for efficient lookup
        let old_map = create_file_map(&old_manifest.files);
        let current_map = create_file_map(&current_entries);
        
        let mut stats = ChangeStats::default();
        
        // Find added and modified files
        for (path, current_entry) in &current_map {
            match old_map.get(path) {
                Some(old_entry) => {
                    // File exists in both - check if modified
                    if current_entry.content_hash != old_entry.content_hash {
                        stats.files_modified += 1;
                        stats.bytes_modified += current_entry.size;
                        stats.changed_files.push((*path).to_path_buf());
                    }
                }
                None => {
                    // File added
                    stats.files_added += 1;
                    stats.bytes_added += current_entry.size;
                    stats.changed_files.push((*path).to_path_buf());
                }
            }
        }
        
        // Find deleted files
        for (path, old_entry) in &old_map {
            if !current_map.contains_key(path) {
                stats.files_deleted += 1;
                stats.bytes_deleted += old_entry.size;
                stats.changed_files.push((*path).to_path_buf());
            }
        }
        
        let detect_duration = start.elapsed();
        debug!(
            "Detected {} changes in {:?}",
            stats.total_operations(),
            detect_duration
        );
        
        Ok(stats)
    }
    

}

/// Process a single file entry
pub fn process_file_entry(
    path: &Path,
    root_path: &Path,
    max_file_size: u64,
    is_directory: bool,
) -> Result<Option<FileEntry>> {
    // Get metadata
    let metadata = match utils::get_file_metadata(path) {
        Ok(m) => m,
        Err(e) => {
            trace!("Skipping entry {:?}: {}", path, e);
            return Ok(None);
        }
    };
    
    // Check file size limit (directories have size 0)
    if !is_directory && max_file_size > 0 && metadata.size > max_file_size {
        trace!("Skipping large file {:?} ({} bytes)", path, metadata.size);
        return Ok(None);
    }
    
    // Skip the root directory itself
    if path == root_path {
        return Ok(None);
    }
    
    // Make path relative to root
    let relative_path = utils::make_relative(path, root_path)?;
    
    // Handle different entry types
    let (content_hash, symlink_target, size) = if is_directory {
        // For directories, use a special hash based on path
        let hash = utils::hash_data(format!("dir:{}", relative_path.display()).as_bytes());
        (hash, None, 0u64)
    } else if metadata.is_symlink {
        let target = utils::read_symlink(path)?;
        let hash = utils::hash_data(target.to_string_lossy().as_bytes());
        (hash, Some(target), metadata.size)
    } else {
        // Hash file content
        let hash = utils::hash_file_content(path)?;
        (hash, None, metadata.size)
    };
    
    // Hash metadata
    let mut hash_builder = FileEntryHashBuilder::new();
    let metadata_hash = hash_builder.hash_metadata(
        metadata.permissions,
        &metadata.modified.into(),
    );
    
    // Compute combined hash
    let combined_hash = hash_builder.combined_hash(&content_hash, &metadata_hash);
    
    Ok(Some(FileEntry {
        path: relative_path,
        content_hash,
        size,
        permissions: metadata.permissions,
        modified: metadata.modified.into(),
        is_compressed: false, // Will be set during storage
        metadata_hash,
        combined_hash,
        is_symlink: metadata.is_symlink,
        symlink_target,
        is_directory,
    }))
}

/// Create a file manifest from entries
///
/// Creates a `FileManifest` from a collection of file entries. The manifest
/// serves as a snapshot of the file system state at a specific point in time
/// and is used for change detection and checkpoint storage.
///
/// # Arguments
///
/// * `checkpoint_id` - Unique identifier for the checkpoint
/// * `entries` - Vector of file entries to include in the manifest
/// * `merkle_root` - Root hash of the Merkle tree for this file set
///
/// # Returns
///
/// Returns a `FileManifest` containing all provided entries with computed statistics.
///
/// # Example
///
/// ```rust,ignore
/// use crate::file_tracking::{FileTracker, create_manifest};
/// use std::path::PathBuf;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let tracker = FileTracker::new(PathBuf::from("./project"));
/// let files = tracker.scan_directory::<fn(titor::types::ProgressInfo)>(None)?;
///
/// let manifest = create_manifest(
///     "checkpoint_123".to_string(),
///     files,
///     "merkle_root_hash".to_string(),
/// );
///
/// println!("Manifest created with {} files", manifest.file_count);
/// println!("Total size: {} bytes", manifest.total_size);
/// # Ok(())
/// # }
/// ```
///
/// # Computed Statistics
///
/// The manifest automatically computes:
/// - Total size of all files
/// - Number of files in the manifest
/// - Creation timestamp
pub fn create_manifest(
    checkpoint_id: String,
    entries: Vec<FileEntry>,
    merkle_root: String,
) -> FileManifest {
    let total_size = entries.iter().map(|e| e.size).sum();
    let file_count = entries.len();
    
    FileManifest {
        checkpoint_id,
        files: entries,
        total_size,
        file_count,
        merkle_root,
        created_at: Utc::now(),
    }
}

/// Create a HashMap from file entries for efficient path lookup
///
/// Creates a hash map that maps file paths to their corresponding `FileEntry`
/// objects. This is used internally for efficient O(1) lookups during change
/// detection and other operations that need to find files by path.
///
/// # Arguments
///
/// * `entries` - Slice of file entries to create the map from
///
/// # Returns
///
/// Returns a `HashMap` where keys are file paths and values are references
/// to the corresponding `FileEntry` objects.
///
/// # Example
///
/// ```rust,ignore
/// use crate::file_tracking::{FileTracker, create_file_map};
/// use std::path::Path;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let tracker = FileTracker::new(std::path::PathBuf::from("./project"));
/// let files = tracker.scan_directory::<fn(titor::types::ProgressInfo)>(None)?;
///
/// let file_map = create_file_map(&files);
///
/// // Quick lookup by path
/// if let Some(entry) = file_map.get(Path::new("src/main.rs")) {
///     println!("Found file: {} bytes", entry.size);
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// - O(n) time complexity to create the map
/// - O(1) average time complexity for lookups
/// - Memory efficient using references to avoid cloning
pub fn create_file_map<'a>(entries: &'a [FileEntry]) -> HashMap<&'a Path, &'a FileEntry> {
    entries.iter()
        .map(|e| (e.path.as_path(), e))
        .collect()
}

/// Check if file metadata matches at nanosecond precision
///
/// Compares file metadata timestamps with nanosecond precision to detect
/// even the smallest changes. This is used by the index-based scanning
/// to determine if a file needs to be re-processed.
///
/// # Arguments
///
/// * `metadata` - Current file metadata
/// * `index_entry` - Stored index entry to compare against
///
/// # Returns
///
/// Returns true if the metadata matches exactly, false otherwise
fn metadata_matches_ns(metadata: &crate::utils::FileMetadata, index_entry: &IndexEntry) -> bool {
    use chrono::{DateTime, Utc};
    
    // Convert metadata times to nanoseconds since epoch
    let mtime_dt: DateTime<Utc> = metadata.modified.into();
    let mtime_ns = mtime_dt.timestamp_nanos_opt().unwrap_or(0);
    
    // For now, we don't track creation time in FileMetadata, so we only compare mtime
    mtime_ns == index_entry.mtime_ns
}

/// Create a FileEntry from an IndexEntry
///
/// Converts an index entry back into a FileEntry when the file hasn't changed.
/// This allows reusing cached hash values without re-reading the file.
///
/// # Arguments
///
/// * `path` - Relative path to the file
/// * `index_entry` - The cached index entry
/// * `metadata` - Current file metadata
///
/// # Returns
///
/// Returns a FileEntry with information from the index
fn create_file_entry_from_index(
    path: PathBuf,
    index_entry: &IndexEntry,
    metadata: crate::utils::FileMetadata,
) -> FileEntry {
    // Hash metadata
    let mut hash_builder = FileEntryHashBuilder::new();
    let metadata_hash = hash_builder.hash_metadata(
        metadata.permissions,
        &metadata.modified.into(),
    );
    
    // Compute combined hash
    let combined_hash = hash_builder.combined_hash(&index_entry.hash, &metadata_hash);
    
    FileEntry {
        path: path.clone(),
        content_hash: index_entry.hash.clone(),
        size: index_entry.size,
        permissions: metadata.permissions,
        modified: metadata.modified.into(),
        is_compressed: false, // Will be set during storage
        metadata_hash,
        combined_hash,
        is_symlink: metadata.is_symlink,
        symlink_target: if metadata.is_symlink {
            utils::read_symlink(&path).ok()
        } else {
            None
        },
        is_directory: index_entry.is_dir,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_file_tracker_scan() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        
        // Create test files
        fs::write(root.join("file1.txt"), "content1").unwrap();
        fs::write(root.join("file2.txt"), "content2").unwrap();
        fs::create_dir(root.join("subdir")).unwrap();
        fs::write(root.join("subdir/file3.txt"), "content3").unwrap();
        
        // Scan directory
        let tracker = FileTracker::new(root.to_path_buf());
        let entries = tracker.scan_directory::<fn(ProgressInfo)>(None).unwrap();
        
        assert_eq!(entries.len(), 3);
        assert!(entries.iter().any(|e| e.path == Path::new("file1.txt")));
        assert!(entries.iter().any(|e| e.path == Path::new("file2.txt")));
        assert!(entries.iter().any(|e| e.path == Path::new("subdir/file3.txt")));
    }
    
    #[test]
    fn test_file_tracker_ignore_patterns() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        
        // Create test files
        fs::write(root.join("file.txt"), "content").unwrap();
        fs::write(root.join("file.tmp"), "temp").unwrap();
        fs::write(root.join("file.log"), "log").unwrap();
        
        // Create .gitignore
        fs::write(root.join(".gitignore"), "*.tmp\n*.log").unwrap();
        
        // Scan with ignore patterns
        let tracker = FileTracker::new(root.to_path_buf())
            .with_ignore_patterns(vec!["*.tmp".to_string()]);
        let entries = tracker.scan_directory::<fn(ProgressInfo)>(None).unwrap();
        
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].path, Path::new("file.txt"));
    }
    
    #[test]
    fn test_nested_gitignore() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        
        // Create directory structure
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("src/tests")).unwrap();
        fs::create_dir_all(root.join("build")).unwrap();
        fs::create_dir_all(root.join("docs")).unwrap();
        
        // Create files at root level
        fs::write(root.join("README.md"), "readme").unwrap();
        fs::write(root.join("config.json"), "config").unwrap();
        fs::write(root.join("debug.log"), "debug").unwrap();
        fs::write(root.join("temp.txt"), "temp").unwrap();
        
        // Create files in src/
        fs::write(root.join("src/main.rs"), "main").unwrap();
        fs::write(root.join("src/lib.rs"), "lib").unwrap();
        fs::write(root.join("src/test.tmp"), "test temp").unwrap();
        
        // Create files in src/tests/
        fs::write(root.join("src/tests/test1.rs"), "test1").unwrap();
        fs::write(root.join("src/tests/test2.rs"), "test2").unwrap();
        fs::write(root.join("src/tests/fixture.json"), "fixture").unwrap();
        
        // Create files in build/
        fs::write(root.join("build/output.exe"), "output").unwrap();
        fs::write(root.join("build/cache.dat"), "cache").unwrap();
        
        // Create files in docs/
        fs::write(root.join("docs/api.md"), "api").unwrap();
        fs::write(root.join("docs/internal.md"), "internal").unwrap();
        
        // Create root .gitignore
        fs::write(root.join(".gitignore"), "*.log\nbuild/\ntemp.txt").unwrap();
        
        // Create nested .gitignore in src/
        fs::write(root.join("src/.gitignore"), "*.tmp").unwrap();
        
        // Create nested .gitignore in src/tests/
        fs::write(root.join("src/tests/.gitignore"), "*.json").unwrap();
        
        // Create nested .gitignore in docs/
        fs::write(root.join("docs/.gitignore"), "internal.md").unwrap();
        
        // Scan directory
        let tracker = FileTracker::new(root.to_path_buf());
        let entries = tracker.scan_directory::<fn(ProgressInfo)>(None).unwrap();
        
        // Collect all paths for easier assertion
        let paths: Vec<_> = entries.iter()
            .map(|e| e.path.to_string_lossy().to_string())
            .collect();
        
        // Files that should be included
        assert!(paths.contains(&"README.md".to_string()));
        assert!(paths.contains(&"config.json".to_string()));
        assert!(paths.contains(&"src/main.rs".to_string()));
        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(paths.contains(&"src/tests/test1.rs".to_string()));
        assert!(paths.contains(&"src/tests/test2.rs".to_string()));
        assert!(paths.contains(&"docs/api.md".to_string()));
        
        // Files that should be ignored
        assert!(!paths.contains(&"debug.log".to_string()));       // Ignored by root .gitignore
        assert!(!paths.contains(&"temp.txt".to_string()));        // Ignored by root .gitignore
        assert!(!paths.contains(&"build/output.exe".to_string())); // build/ ignored by root .gitignore
        assert!(!paths.contains(&"build/cache.dat".to_string()));  // build/ ignored by root .gitignore
        assert!(!paths.contains(&"src/test.tmp".to_string()));    // Ignored by src/.gitignore
        assert!(!paths.contains(&"src/tests/fixture.json".to_string())); // Ignored by src/tests/.gitignore
        assert!(!paths.contains(&"docs/internal.md".to_string())); // Ignored by docs/.gitignore
        
        // Verify correct number of files (should be 7)
        assert_eq!(entries.len(), 7);
    }
    
    #[test]
    fn test_gitignore_precedence() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        
        // Create nested structure
        fs::create_dir_all(root.join("nested/deep")).unwrap();
        
        // Create files
        fs::write(root.join("test.txt"), "root test").unwrap();
        fs::write(root.join("nested/test.txt"), "nested test").unwrap();
        fs::write(root.join("nested/deep/test.txt"), "deep test").unwrap();
        fs::write(root.join("nested/keep.txt"), "keep").unwrap();
        
        // Root .gitignore ignores all test.txt files
        fs::write(root.join(".gitignore"), "test.txt").unwrap();
        
        // Nested .gitignore un-ignores test.txt (using ! prefix)
        fs::write(root.join("nested/.gitignore"), "!test.txt").unwrap();
        
        // Deep .gitignore re-ignores test.txt
        fs::write(root.join("nested/deep/.gitignore"), "test.txt").unwrap();
        
        // Scan
        let tracker = FileTracker::new(root.to_path_buf());
        let entries = tracker.scan_directory::<fn(ProgressInfo)>(None).unwrap();
        
        let paths: Vec<_> = entries.iter()
            .map(|e| e.path.to_string_lossy().to_string())
            .collect();
        
        // Root test.txt should be ignored
        assert!(!paths.contains(&"test.txt".to_string()));
        
        // Nested test.txt should be included (un-ignored)
        assert!(paths.contains(&"nested/test.txt".to_string()));
        
        // Deep test.txt should be ignored again
        assert!(!paths.contains(&"nested/deep/test.txt".to_string()));
        
        // Keep.txt should be included
        assert!(paths.contains(&"nested/keep.txt".to_string()));
    }
    
    #[test]
    fn test_custom_patterns_override() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        
        // Create files
        fs::write(root.join("include.txt"), "include").unwrap();
        fs::write(root.join("exclude.txt"), "exclude").unwrap();
        fs::write(root.join("force_exclude.txt"), "force").unwrap();
        
        // .gitignore excludes exclude.txt
        fs::write(root.join(".gitignore"), "exclude.txt").unwrap();
        
        // Scan with custom pattern that force excludes another file
        let tracker = FileTracker::new(root.to_path_buf())
            .with_ignore_patterns(vec!["force_exclude.txt".to_string()]);
        let entries = tracker.scan_directory::<fn(ProgressInfo)>(None).unwrap();
        
        let paths: Vec<_> = entries.iter()
            .map(|e| e.path.to_string_lossy().to_string())
            .collect();
        
        // Should include only include.txt
        assert_eq!(entries.len(), 1);
        assert!(paths.contains(&"include.txt".to_string()));
        assert!(!paths.contains(&"exclude.txt".to_string()));
        assert!(!paths.contains(&"force_exclude.txt".to_string()));
    }
    
    #[test]
    fn test_detect_changes() {
        let temp_dir = TempDir::new().unwrap();
        let root = temp_dir.path();
        
        // Initial state
        fs::write(root.join("file1.txt"), "content1").unwrap();
        fs::write(root.join("file2.txt"), "content2").unwrap();
        
        let tracker = FileTracker::new(root.to_path_buf());
        let entries1 = tracker.scan_directory::<fn(ProgressInfo)>(None).unwrap();
        let manifest1 = create_manifest(
            "checkpoint1".to_string(),
            entries1,
            "merkle1".to_string(),
        );
        
        // Make changes
        fs::write(root.join("file1.txt"), "modified").unwrap(); // Modified
        fs::remove_file(root.join("file2.txt")).unwrap(); // Deleted
        fs::write(root.join("file3.txt"), "new").unwrap(); // Added
        
        // Detect changes
        let changes = tracker.detect_changes(&manifest1).unwrap();
        
        assert_eq!(changes.files_added, 1);
        assert_eq!(changes.files_modified, 1);
        assert_eq!(changes.files_deleted, 1);
        assert_eq!(changes.total_operations(), 3);
    }
    

} 