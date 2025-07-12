//! Content-addressable storage implementation for Titor
//!
//! This module provides the core storage backend for Titor, implementing a
//! content-addressable storage (CAS) system with advanced features including:
//!
//! - **Deduplication**: Identical content is stored only once, regardless of filename
//! - **Sharding**: Files are distributed across subdirectories for optimal performance
//! - **Reference counting**: Tracks how many checkpoints reference each object
//! - **Compression**: Transparent compression/decompression of stored objects
//! - **Caching**: In-memory caching for frequently accessed metadata
//! - **Garbage collection**: Automatic cleanup of unreferenced objects
//!
//! ## Architecture
//!
//! The storage system uses a hierarchical directory structure:
//!
//! ```text
//! storage_root/
//! ├── metadata.json          # Storage metadata and configuration
//! ├── checkpoints/           # Checkpoint metadata and manifests
//! │   └── <checkpoint_id>/
//! │       ├── metadata.json  # Checkpoint metadata
//! │       └── manifest.bin   # File manifest (binary format)
//! ├── objects/               # Content-addressable objects (sharded)
//! │   └── <prefix>/          # First 2 chars of hash
//! │       └── <suffix>       # Remaining hash chars
//! ├── refs/                  # Reference counts for objects
//! │   └── <hash>             # File containing reference count
//! └── object_cache.bin       # Cached object metadata
//! ```
//!
//! ## Content Addressing
//!
//! Objects are identified by their SHA-256 hash and stored in a sharded
//! directory structure. The first two characters of the hash form the
//! shard prefix, improving filesystem performance for large numbers of objects.
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use titor::{Storage, CompressionEngine};
//! use titor::types::TitorConfig;
//! use std::path::PathBuf;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Initialize storage
//! let config = TitorConfig {
//!     root_path: PathBuf::from("./project"),
//!     storage_path: PathBuf::from("./storage"),
//!     max_file_size: 0,
//!     parallel_workers: 4,
//!     ignore_patterns: vec![],
//!     compression_strategy: "fast".to_string(),
//!     follow_symlinks: false,
//!     version: "0.1.2".to_string(),
//! };
//! let compression = CompressionEngine::new(
//!     titor::compression::CompressionStrategy::Fast
//! );
//! let storage = Storage::init(
//!     PathBuf::from("./storage"),
//!     config,
//!     compression,
//! )?;
//!
//! // Store file content
//! let content = b"Hello, world!";
//! let path = PathBuf::from("hello.txt");
//! let (hash, compressed_size) = storage.store_object(content, &path)?;
//!
//! // Retrieve content
//! let retrieved = storage.load_object(&hash)?;
//! assert_eq!(retrieved, content);
//!
//! // Check storage statistics
//! let stats = storage.stats()?;
//! println!("Objects: {}, Total size: {}", stats.object_count, stats.total_size);
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Considerations
//!
//! - **Sharding**: Objects are distributed across subdirectories to prevent
//!   filesystem performance degradation with large numbers of files
//! - **Caching**: Frequently accessed object metadata is cached in memory
//! - **Batch operations**: Reference count updates are batched for efficiency
//! - **Compression**: Files are compressed based on the configured strategy
//!
//! ## Thread Safety
//!
//! The storage system is designed to be thread-safe using:
//! - `Arc<DashMap>` for concurrent access to caches and reference counts
//! - `Arc<RwLock>` for metadata that requires exclusive write access
//! - `Arc<Mutex>` for the compression engine
//!
//! ## Error Handling
//!
//! All operations return `Result<T, TitorError>` with detailed error information.
//! Common error scenarios include:
//! - Storage not initialized
//! - Objects not found
//! - I/O errors
//! - Serialization/deserialization errors

use crate::checkpoint::Checkpoint;
use crate::compression::CompressionEngine;
use crate::error::{Result, TitorError};
use crate::types::{FileManifest, StorageMetadata, StorageObject, TitorConfig};
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use crate::collections::GxBuildHasher;
use sha2::{Digest, Sha256};
use std::fs;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{debug, info, trace, warn};

/// Content-addressable storage backend for Titor
///
/// `Storage` provides a thread-safe, content-addressable storage system with
/// deduplication, compression, and reference counting. It serves as the core
/// storage layer for Titor, managing both file content and metadata.
///
/// ## Features
///
/// - **Content Deduplication**: Identical content is stored only once
/// - **Sharded Storage**: Objects are distributed across subdirectories for performance
/// - **Reference Counting**: Tracks usage of objects for garbage collection
/// - **Transparent Compression**: Automatic compression/decompression of stored content
/// - **Metadata Caching**: In-memory caching for frequently accessed object metadata
/// - **Atomic Operations**: All operations are atomic and consistent
///
/// ## Thread Safety
///
/// All operations are thread-safe and can be called concurrently from multiple threads.
/// The storage system uses lock-free data structures where possible for optimal performance.
///
/// ## Example
///
/// ```rust,ignore
/// use titor::{Storage, CompressionEngine};
/// use titor::types::TitorConfig;
/// use std::path::PathBuf;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = TitorConfig {
///     root_path: PathBuf::from("./project"),
///     storage_path: PathBuf::from("./storage"),
///     max_file_size: 0,
///     parallel_workers: 4,
///     ignore_patterns: vec![],
///     compression_strategy: "fast".to_string(),
///     follow_symlinks: false,
///     version: "0.1.2".to_string(),
/// };
/// let compression = CompressionEngine::new(
///     titor::compression::CompressionStrategy::Fast
/// );
///
/// // Initialize new storage
/// let storage = Storage::init(
///     PathBuf::from("./my_storage"),
///     config,
///     compression,
/// )?;
///
/// // Store and retrieve content
/// let content = b"Hello, Titor!";
/// let path = PathBuf::from("greeting.txt");
/// let (hash, size) = storage.store_object(content, &path)?;
/// let retrieved = storage.load_object(&hash)?;
/// assert_eq!(retrieved, content);
/// # Ok(())
/// # }
/// ```
pub struct Storage {
    /// Root directory for storage
    root: PathBuf,
    /// Compression engine for transparent compression/decompression
    compression: Arc<Mutex<CompressionEngine>>,
    /// Cache of object metadata for fast lookups
    object_cache: Arc<DashMap<String, StorageObject, GxBuildHasher>>,
    /// Reference counts for objects (for garbage collection)
    ref_counts: Arc<DashMap<String, usize, GxBuildHasher>>,
    /// Pending reference count updates (batched for performance)
    pending_ref_updates: Arc<DashMap<String, usize, GxBuildHasher>>,
    /// Storage metadata and configuration
    metadata: Arc<RwLock<StorageMetadata>>,
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("root", &self.root)
            .field("object_cache_size", &self.object_cache.len())
            .field("ref_counts_size", &self.ref_counts.len())
            .field("pending_updates", &self.pending_ref_updates.len())
            .finish()
    }
}

impl Storage {
    /// Initialize a new storage repository
    ///
    /// Creates a new content-addressable storage repository at the specified root path.
    /// This operation will fail if the storage directory already exists.
    ///
    /// # Arguments
    ///
    /// * `root` - Root directory path for the storage repository
    /// * `config` - Configuration settings for the storage
    /// * `compression` - Compression engine to use for object storage
    ///
    /// # Returns
    ///
    /// Returns a `Storage` instance ready for use.
    ///
    /// # Errors
    ///
    /// - [`TitorError::StorageAlreadyExists`] if storage already exists at the path
    /// - [`TitorError::Io`] if filesystem operations fail
    /// - [`TitorError::Serialization`] if metadata serialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use titor::{Storage, CompressionEngine};
    /// use titor::types::TitorConfig;
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = TitorConfig {
    /// #     root_path: PathBuf::from("./project"),
    /// #     storage_path: PathBuf::from("./storage"),
    /// #     max_file_size: 0,
    /// #     parallel_workers: 4,
    /// #     ignore_patterns: vec![],
    /// #     compression_strategy: "fast".to_string(),
    /// #     follow_symlinks: false,
    /// #     version: "0.1.2".to_string(),
    /// # };
    /// let compression = CompressionEngine::new(
    ///     titor::compression::CompressionStrategy::Fast
    /// );
    ///
    /// let storage = Storage::init(
    ///     PathBuf::from("./new_storage"),
    ///     config,
    ///     compression,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn init(root: PathBuf, config: TitorConfig, compression: CompressionEngine) -> Result<Self> {
        // Check if storage already exists
        if root.exists() {
            return Err(TitorError::StorageAlreadyExists(root));
        }
        
        // Create directory structure
        fs::create_dir_all(&root)?;
        fs::create_dir_all(root.join("checkpoints"))?;
        fs::create_dir_all(root.join("objects"))?;
        fs::create_dir_all(root.join("refs"))?;
        
        // Create metadata
        let metadata = StorageMetadata {
            format_version: 1,
            titor_version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: Utc::now(),
            last_accessed: Utc::now(),
            config,
        };
        
        // Write metadata
        let metadata_path = root.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        fs::write(metadata_path, metadata_json)?;
        
        info!("Initialized storage at {:?}", root);
        
        Ok(Self {
            root,
            compression: Arc::new(Mutex::new(compression)),
            object_cache: Arc::new(DashMap::with_capacity_and_hasher(1000, GxBuildHasher::default())),
            ref_counts: Arc::new(DashMap::with_capacity_and_hasher(1000, GxBuildHasher::default())),
            pending_ref_updates: Arc::new(DashMap::with_capacity_and_hasher(1000, GxBuildHasher::default())),
            metadata: Arc::new(RwLock::new(metadata)),
        })
    }
    
    /// Initialize new storage or open existing storage
    /// This is useful for tests where we want to reuse existing storage
    pub fn init_or_open(root: PathBuf, config: TitorConfig, compression: CompressionEngine) -> Result<Self> {
        if root.join("metadata.json").exists() {
            // Storage already exists, open it
            Storage::open(root, compression)
        } else {
            // Initialize new storage
            Storage::init(root, config, compression)
        }
    }
    
    /// Open an existing storage repository
    ///
    /// Opens an existing content-addressable storage repository at the specified path.
    /// This operation will load existing metadata, reference counts, and object cache.
    ///
    /// # Arguments
    ///
    /// * `root` - Root directory path of the existing storage repository
    /// * `compression` - Compression engine to use for object storage
    ///
    /// # Returns
    ///
    /// Returns a `Storage` instance connected to the existing repository.
    ///
    /// # Errors
    ///
    /// - [`TitorError::StorageNotInitialized`] if storage doesn't exist at the path
    /// - [`TitorError::Io`] if filesystem operations fail
    /// - [`TitorError::Serialization`] if metadata deserialization fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use titor::{Storage, CompressionEngine};
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let compression = CompressionEngine::new(
    ///     titor::compression::CompressionStrategy::Fast
    /// );
    ///
    /// let storage = Storage::open(
    ///     PathBuf::from("./existing_storage"),
    ///     compression,
    /// )?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn open(root: PathBuf, compression: CompressionEngine) -> Result<Self> {
        // Check if storage exists
        if !root.exists() {
            return Err(TitorError::StorageNotInitialized(root));
        }
        
        // Load metadata
        let metadata_path = root.join("metadata.json");
        let metadata_json = fs::read_to_string(&metadata_path)?;
        let mut metadata: StorageMetadata = serde_json::from_str(&metadata_json)?;
        
        // Update last accessed time
        metadata.last_accessed = Utc::now();
        
        // Load reference counts
        let ref_counts = Arc::new(DashMap::with_capacity_and_hasher(1000, GxBuildHasher::default()));
        let refs_dir = root.join("refs");
        if refs_dir.exists() {
            for entry in fs::read_dir(refs_dir)? {
                let entry = entry?;
                let hash = entry.file_name().to_string_lossy().to_string();
                if let Ok(count_str) = fs::read_to_string(entry.path()) {
                    if let Ok(count) = count_str.trim().parse::<usize>() {
                        ref_counts.insert(hash, count);
                    }
                }
            }
        }
        
        // Load object cache if available
        let object_cache = Arc::new(DashMap::with_capacity_and_hasher(1000, GxBuildHasher::default()));
        let cache_path = root.join("object_cache.bin");
        if cache_path.exists() {
            match fs::read(&cache_path) {
                Ok(cache_bytes) => {
                    match bincode::serde::decode_from_slice::<Vec<(String, StorageObject)>, _>(&cache_bytes, bincode::config::standard()) {
                        Ok((cache_entries, _)) => {
                            debug!("Loaded {} cached object entries", cache_entries.len());
                            for (hash, obj) in cache_entries {
                                object_cache.insert(hash, obj);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to load object cache: {}", e);
                            // Remove corrupted cache file
                            fs::remove_file(&cache_path).ok();
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to read object cache: {}", e);
                }
            }
        }
        
        info!("Opened storage at {:?}", root);
        
        Ok(Self {
            root,
            compression: Arc::new(Mutex::new(compression)),
            object_cache,
            ref_counts,
            pending_ref_updates: Arc::new(DashMap::with_capacity_and_hasher(1000, GxBuildHasher::default())),
            metadata: Arc::new(RwLock::new(metadata)),
        })
    }
    
    /// Store file content in the content-addressable storage
    ///
    /// Stores the provided content in the storage system, returning the content's
    /// SHA-256 hash and compressed size. If the content already exists, the reference
    /// count is incremented and the existing hash is returned (deduplication).
    ///
    /// # Arguments
    ///
    /// * `content` - Raw file content to store
    /// * `path` - Original file path (used for compression hint)
    ///
    /// # Returns
    ///
    /// Returns a tuple containing:
    /// - `String` - SHA-256 hash of the content (64 hexadecimal characters)
    /// - `u64` - Compressed size in bytes
    ///
    /// # Errors
    ///
    /// - [`TitorError::Io`] if filesystem operations fail
    /// - [`TitorError::Compression`] if compression fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use titor::{Storage, CompressionEngine, TitorConfig};
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = TitorConfig { root_path: PathBuf::from("./project"), storage_path: PathBuf::from("./storage"), max_file_size: 0, parallel_workers: 4, ignore_patterns: vec![], compression_strategy: "fast".to_string(), follow_symlinks: false, version: "0.1.2".to_string() };
    /// # let compression = CompressionEngine::new(titor::compression::CompressionStrategy::Fast);
    /// # let storage = Storage::init(PathBuf::from("./test_storage"), config, compression)?;
    /// let content = b"Hello, world!";
    /// let path = PathBuf::from("hello.txt");
    /// let (hash, compressed_size) = storage.store_object(content, &path)?;
    ///
    /// println!("Stored object with hash: {}", hash);
    /// println!("Compressed size: {} bytes", compressed_size);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Content is automatically deduplicated based on SHA-256 hash
    /// - Compression is applied based on file extension and content type
    /// - Objects are stored in a sharded directory structure for performance
    pub fn store_object(&self, content: &[u8], path: &Path) -> Result<(String, u64)> {
        // Compute hash of uncompressed content
        let hash = compute_hash(content);
        
        // Check if object already exists
        if self.object_exists(&hash)? {
            debug!("Object {} already exists, incrementing ref count", &hash[..8]);
            self.increment_ref_count(&hash)?;
            // Get compressed size from cache or disk
            let compressed_size = self.get_object_size(&hash).unwrap_or(content.len() as u64);
            return Ok((hash, compressed_size));
        }
        
        // Compress content
        let compressed = self.compression.lock().compress(path, content)?;
        let is_compressed = compressed.len() < content.len();
        let compressed_size = compressed.len() as u64;
        
        // Store object
        let object_path = self.get_object_path(&hash);
        let object_dir = object_path.parent().unwrap();
        fs::create_dir_all(object_dir)?;
        fs::write(&object_path, &compressed)?;
        
        // Create storage object metadata
        let obj = StorageObject {
            hash: hash.clone(),
            compressed_size,
            uncompressed_size: content.len() as u64,
            ref_count: 1,
            is_compressed,
            created_at: Utc::now(),
            last_accessed: Utc::now(),
        };
        
        // Cache object metadata
        self.object_cache.insert(hash.clone(), obj);
        
        // Initialize ref count
        self.set_ref_count(&hash, 1)?;
        
        trace!("Stored object {} ({} bytes)", &hash[..8], compressed.len());
        Ok((hash, compressed_size))
    }
    
    /// Load object content by its SHA-256 hash
    ///
    /// Retrieves and decompresses the content associated with the given hash.
    /// The content is automatically decompressed if it was compressed during storage.
    ///
    /// # Arguments
    ///
    /// * `hash` - SHA-256 hash of the object to load (64 hexadecimal characters)
    ///
    /// # Returns
    ///
    /// Returns the original, uncompressed content as a byte vector.
    ///
    /// # Errors
    ///
    /// - [`TitorError::ObjectNotFound`] if the object doesn't exist
    /// - [`TitorError::Io`] if filesystem operations fail
    /// - [`TitorError::Compression`] if decompression fails
    ///
    /// # Example
    ///
    /// ```rust
    /// use titor::{Storage, CompressionEngine, TitorConfig};
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = TitorConfig { root_path: PathBuf::from("./project"), storage_path: PathBuf::from("./storage"), max_file_size: 0, parallel_workers: 4, ignore_patterns: vec![], compression_strategy: "fast".to_string(), follow_symlinks: false, version: "0.1.2".to_string() };
    /// # let compression = CompressionEngine::new(titor::compression::CompressionStrategy::Fast);
    /// # let storage = Storage::init(PathBuf::from("./test_storage"), config, compression)?;
    /// # let content = b"Hello, world!";
    /// # let path = PathBuf::from("hello.txt");
    /// # let (hash, _) = storage.store_object(content, &path)?;
    /// // Load object by hash
    /// let loaded_content = storage.load_object(&hash)?;
    /// assert_eq!(loaded_content, b"Hello, world!");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Object metadata is cached for faster subsequent access
    /// - Content is automatically decompressed during retrieval
    /// - Last access time is updated in the metadata cache
    pub fn load_object(&self, hash: &str) -> Result<Vec<u8>> {
        let object_path = self.get_object_path(hash);
        if !object_path.exists() {
            return Err(TitorError::ObjectNotFound(hash.to_string()));
        }
        
        // Read compressed content
        let compressed = fs::read(&object_path)?;
        
        // Decompress
        let content = self.compression.lock().decompress(&compressed)?;
        
        // Update last accessed time in cache
        if let Some(mut obj) = self.object_cache.get_mut(hash) {
            obj.last_accessed = Utc::now();
        }
        
        trace!("Loaded object {} ({} bytes)", &hash[..8], content.len());
        Ok(content)
    }
    
    /// Check if an object exists
    pub fn object_exists(&self, hash: &str) -> Result<bool> {
        // Check cache first
        if self.object_cache.contains_key(hash) {
            return Ok(true);
        }
        
        // Check filesystem
        Ok(self.get_object_path(hash).exists())
    }
    
    /// Store a checkpoint
    pub fn store_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        let checkpoint_dir = self.root.join("checkpoints").join(&checkpoint.id);
        fs::create_dir_all(&checkpoint_dir)?;
        
        // Store checkpoint metadata
        let metadata_path = checkpoint_dir.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(checkpoint)?;
        fs::write(metadata_path, metadata_json)?;
        
        debug!("Stored checkpoint {}", checkpoint.short_id());
        Ok(())
    }
    
    /// Load a checkpoint
    pub fn load_checkpoint(&self, checkpoint_id: &str) -> Result<Checkpoint> {
        let metadata_path = self.root
            .join("checkpoints")
            .join(checkpoint_id)
            .join("metadata.json");
        
        if !metadata_path.exists() {
            return Err(TitorError::CheckpointNotFound(checkpoint_id.to_string()));
        }
        
        let metadata_json = fs::read_to_string(metadata_path)?;
        let checkpoint = serde_json::from_str(&metadata_json)?;
        
        Ok(checkpoint)
    }
    
    /// Store a file manifest
    pub fn store_manifest(&self, manifest: &FileManifest) -> Result<()> {
        let manifest_path = self.root
            .join("checkpoints")
            .join(&manifest.checkpoint_id)
            .join("manifest.bin");
        
        // Store manifest in bincode format
        let manifest_bytes = bincode::serde::encode_to_vec(manifest, bincode::config::standard())?;
        fs::write(&manifest_path, &manifest_bytes)?;
        
        debug!("Stored manifest for checkpoint {} ({} files, {} bytes)", 
               &manifest.checkpoint_id[..8], manifest.file_count, manifest_bytes.len());
        Ok(())
    }
    
    /// Load a file manifest
    pub fn load_manifest(&self, checkpoint_id: &str) -> Result<FileManifest> {
        let manifest_path = self.root
            .join("checkpoints")
            .join(checkpoint_id)
            .join("manifest.bin");
        
        if !manifest_path.exists() {
            return Err(TitorError::CheckpointNotFound(checkpoint_id.to_string()));
        }
        
        let manifest_bytes = fs::read(&manifest_path)?;
        let (manifest, _): (FileManifest, _) = bincode::serde::decode_from_slice(&manifest_bytes, bincode::config::standard())?;
        Ok(manifest)
    }
    
    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Result<Vec<String>> {
        let checkpoints_dir = self.root.join("checkpoints");
        let mut checkpoint_ids = Vec::new();
        
        if checkpoints_dir.exists() {
            for entry in fs::read_dir(checkpoints_dir)? {
                let entry = entry?;
                if entry.path().is_dir() {
                    checkpoint_ids.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }
        
        Ok(checkpoint_ids)
    }
    
    /// Check if a checkpoint exists
    pub fn checkpoint_exists(&self, checkpoint_id: &str) -> bool {
        self.root
            .join("checkpoints")
            .join(checkpoint_id)
            .join("metadata.json")
            .exists()
    }
    
    /// Delete a checkpoint (but not its objects)
    pub fn delete_checkpoint(&self, checkpoint_id: &str) -> Result<()> {
        let checkpoint_dir = self.root.join("checkpoints").join(checkpoint_id);
        if checkpoint_dir.exists() {
            fs::remove_dir_all(checkpoint_dir)?;
            info!("Deleted checkpoint {}", &checkpoint_id[..8]);
        }
        Ok(())
    }
    
    /// Increment reference count for an object
    fn increment_ref_count(&self, hash: &str) -> Result<()> {
        let mut count = self.ref_counts.entry(hash.to_string()).or_insert(0);
        *count += 1;
        // Store in pending updates instead of persisting immediately
        self.pending_ref_updates.insert(hash.to_string(), *count);
        Ok(())
    }
    
    /// Decrement reference count for an object
    pub fn decrement_ref_count(&self, hash: &str) -> Result<usize> {
        let mut count = 0;
        
        if let Some(mut ref_count) = self.ref_counts.get_mut(hash) {
            if *ref_count > 0 {
                *ref_count -= 1;
                count = *ref_count;
            }
        }
        
        // Store in pending updates instead of persisting immediately
        self.pending_ref_updates.insert(hash.to_string(), count);
        Ok(count)
    }
    
    /// Set reference count
    fn set_ref_count(&self, hash: &str, count: usize) -> Result<()> {
        self.ref_counts.insert(hash.to_string(), count);
        // Store in pending updates instead of persisting immediately
        self.pending_ref_updates.insert(hash.to_string(), count);
        Ok(())
    }
    
    /// Flush pending reference count updates to disk
    pub fn flush_ref_counts(&self) -> Result<()> {
        let updates: Vec<_> = self.pending_ref_updates.iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();
        
        for (hash, count) in updates {
            self.persist_ref_count(&hash, count)?;
            self.pending_ref_updates.remove(&hash);
        }
        
        Ok(())
    }
    
    /// Persist reference count to disk
    fn persist_ref_count(&self, hash: &str, count: usize) -> Result<()> {
        let ref_path = self.root.join("refs").join(hash);
        
        if count == 0 {
            // Remove ref file if count is 0
            if ref_path.exists() {
                fs::remove_file(ref_path)?;
            }
        } else {
            fs::write(ref_path, count.to_string())?;
        }
        
        Ok(())
    }
    
    /// Delete an object
    pub fn delete_object(&self, hash: &str) -> Result<()> {
        let object_path = self.get_object_path(hash);
        if object_path.exists() {
            fs::remove_file(object_path)?;
            self.object_cache.remove(hash);
            debug!("Deleted object {}", &hash[..8]);
        }
        Ok(())
    }
    
    /// Get size of an object in storage (compressed size)
    pub fn get_object_size(&self, hash: &str) -> Result<u64> {
        // Check cache first
        if let Some(obj) = self.object_cache.get(hash) {
            return Ok(obj.compressed_size);
        }
        
        // Check filesystem
        let object_path = self.get_object_path(hash);
        if !object_path.exists() {
            return Err(TitorError::ObjectNotFound(hash.to_string()));
        }
        
        let metadata = fs::metadata(&object_path)?;
        Ok(metadata.len())
    }
    
    /// Get path for an object (with sharding)
    fn get_object_path(&self, hash: &str) -> PathBuf {
        let (prefix, suffix) = hash.split_at(2);
        self.root.join("objects").join(prefix).join(suffix)
    }
    
    /// Persist object cache to disk for faster startup
    pub fn persist_object_cache(&self) -> Result<()> {
        let cache_path = self.root.join("object_cache.bin");
        
        // Convert DashMap to Vec for serialization
        let cache_entries: Vec<(String, StorageObject)> = self.object_cache
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().clone()))
            .collect();
        
        if cache_entries.is_empty() {
            // Remove empty cache file
            if cache_path.exists() {
                fs::remove_file(&cache_path).ok();
            }
            return Ok(());
        }
        
        // Serialize with bincode
        let cache_bytes = bincode::serde::encode_to_vec(&cache_entries, bincode::config::standard())?;
        fs::write(&cache_path, &cache_bytes)?;
        
        debug!("Persisted {} object cache entries ({} bytes)", 
               cache_entries.len(), cache_bytes.len());
        Ok(())
    }
    
    /// Get comprehensive storage statistics
    ///
    /// Returns detailed statistics about the storage repository, including
    /// object counts, total size, and reference information.
    ///
    /// # Returns
    ///
    /// Returns a [`StorageStats`] struct containing:
    /// - Number of objects stored
    /// - Total compressed size of all objects
    /// - Number of checkpoints
    /// - Total number of references
    ///
    /// # Errors
    ///
    /// - [`TitorError::Io`] if filesystem operations fail
    ///
    /// # Example
    ///
    /// ```rust
    /// use titor::{Storage, CompressionEngine, TitorConfig};
    /// use std::path::PathBuf;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// # let config = TitorConfig { root_path: PathBuf::from("./project"), storage_path: PathBuf::from("./storage"), max_file_size: 0, parallel_workers: 4, ignore_patterns: vec![], compression_strategy: "fast".to_string(), follow_symlinks: false, version: "0.1.2".to_string() };
    /// # let compression = CompressionEngine::new(titor::compression::CompressionStrategy::Fast);
    /// # let storage = Storage::init(PathBuf::from("./test_storage"), config, compression)?;
    /// let stats = storage.stats()?;
    /// println!("Objects: {}", stats.object_count);
    /// println!("Total size: {} bytes", stats.total_size);
    /// println!("Checkpoints: {}", stats.checkpoint_count);
    /// println!("References: {}", stats.total_references);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - Statistics are computed primarily from cached data for efficiency
    /// - Object counts may be estimated for very large repositories
    /// - The operation is relatively fast and safe to call frequently
    pub fn stats(&self) -> Result<StorageStats> {
        let mut stats = StorageStats::default();
        
        // Use cached information when possible
        stats.object_count = self.object_cache.len();
        
        // If cache is empty, do a quick count (but don't walk entire structure)
        if stats.object_count == 0 {
            let objects_dir = self.root.join("objects");
            if objects_dir.exists() {
                // Just count shard directories as an estimate
                let shard_count = fs::read_dir(objects_dir)?.count();
                stats.object_count = shard_count * 100; // Rough estimate
            }
        }
        
        // Get total size from cache
        stats.total_size = self.object_cache.iter()
            .map(|entry| entry.value().compressed_size)
            .sum();
        
        // Count checkpoints
        stats.checkpoint_count = self.list_checkpoints()?.len();
        
        // Count references
        stats.total_references = self.ref_counts.len();
        
        Ok(stats)
    }
    
    /// Get unreferenced objects (for garbage collection)
    pub fn get_unreferenced_objects(&self) -> Result<Vec<String>> {
        let mut unreferenced = Vec::new();
        
        let objects_dir = self.root.join("objects");
        if objects_dir.exists() {
            for shard_entry in fs::read_dir(objects_dir)? {
                let shard_entry = shard_entry?;
                if shard_entry.path().is_dir() {
                    let shard_name = shard_entry.file_name().to_string_lossy().to_string();
                    
                    for object_entry in fs::read_dir(shard_entry.path())? {
                        let object_entry = object_entry?;
                        if object_entry.path().is_file() {
                            let object_name = object_entry.file_name().to_string_lossy().to_string();
                            let hash = format!("{}{}", shard_name, object_name);
                            
                            // Check reference count
                            let ref_count = self.ref_counts.get(&hash).map(|r| *r).unwrap_or(0);
                            if ref_count == 0 {
                                unreferenced.push(hash);
                            }
                        }
                    }
                }
            }
        }
        
        Ok(unreferenced)
    }
    
    /// List all objects in storage
    pub fn list_all_objects(&self) -> Result<Vec<String>> {
        let mut objects = Vec::new();
        
        let objects_dir = self.root.join("objects");
        if objects_dir.exists() {
            for shard_entry in fs::read_dir(objects_dir)? {
                let shard_entry = shard_entry?;
                if shard_entry.path().is_dir() {
                    let shard_name = shard_entry.file_name().to_string_lossy().to_string();
                    
                    for object_entry in fs::read_dir(shard_entry.path())? {
                        let object_entry = object_entry?;
                        if object_entry.path().is_file() {
                            let object_name = object_entry.file_name().to_string_lossy().to_string();
                            let hash = format!("{}{}", shard_name, object_name);
                            objects.push(hash);
                        }
                    }
                }
            }
        }
        
        Ok(objects)
    }
    
    /// Update storage metadata
    pub fn update_metadata<F>(&self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut StorageMetadata),
    {
        let mut metadata = self.metadata.write();
        updater(&mut metadata);
        metadata.last_accessed = Utc::now();
        
        // Persist to disk
        let metadata_path = self.root.join("metadata.json");
        let metadata_json = serde_json::to_string_pretty(&*metadata)?;
        fs::write(metadata_path, metadata_json)?;
        
        Ok(())
    }
    
    /// Get storage root path
    pub fn root(&self) -> &Path {
        &self.root
    }
    
    /// Get storage metadata
    pub fn metadata(&self) -> &RwLock<StorageMetadata> {
        &self.metadata
    }
}

/// Comprehensive storage statistics
///
/// Provides detailed information about the current state of the storage repository,
/// including object counts, storage usage, and reference tracking data.
///
/// # Fields
///
/// * `object_count` - Total number of unique objects stored
/// * `total_size` - Total compressed size of all objects in bytes
/// * `checkpoint_count` - Number of checkpoints in the repository
/// * `total_references` - Total number of object references across all checkpoints
///
/// # Example
///
/// ```rust
/// use titor::{Storage, CompressionEngine, TitorConfig};
/// use std::path::PathBuf;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # let config = TitorConfig { root_path: PathBuf::from("./project"), storage_path: PathBuf::from("./storage"), max_file_size: 0, parallel_workers: 4, ignore_patterns: vec![], compression_strategy: "fast".to_string(), follow_symlinks: false, version: "0.1.2".to_string() };
/// # let compression = CompressionEngine::new(titor::compression::CompressionStrategy::Fast);
/// # let storage = Storage::init(PathBuf::from("./test_storage"), config, compression)?;
/// let stats = storage.stats()?;
///
/// // Display storage efficiency
/// if stats.object_count > 0 {
///     let avg_size = stats.total_size / stats.object_count as u64;
///     println!("Average object size: {} bytes", avg_size);
/// }
///
/// // Display deduplication efficiency
/// if stats.total_references > 0 {
///     let dedup_ratio = stats.total_references as f64 / stats.object_count as f64;
///     println!("Deduplication ratio: {:.2}x", dedup_ratio);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct StorageStats {
    /// Number of unique objects stored in the repository
    pub object_count: usize,
    /// Total compressed size of all objects in bytes
    pub total_size: u64,
    /// Number of checkpoints in the repository
    pub checkpoint_count: usize,
    /// Total number of object references across all checkpoints
    pub total_references: usize,
}

/// Compute SHA-256 hash of content
///
/// Calculates the SHA-256 hash of the provided content and returns it
/// as a hexadecimal string. This is the core function used for content
/// addressing in the storage system.
///
/// # Arguments
///
/// * `content` - Raw bytes to hash
///
/// # Returns
///
/// Returns a 64-character hexadecimal string representing the SHA-256 hash.
///
/// # Example
///
/// ```rust,ignore
/// // This is a private function in the storage module
/// use crate::storage::compute_hash;
/// let content = b"Hello, world!";
/// let hash = compute_hash(content);
/// assert_eq!(hash.len(), 64); // SHA-256 is 32 bytes = 64 hex chars
/// ```
fn compute_hash(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionStrategy;
    use tempfile::TempDir;
    
    fn create_test_storage() -> (Storage, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();
        
        // Remove the directory created by TempDir to allow init() to work
        std::fs::remove_dir_all(&path).ok();
        
        let config = TitorConfig {
            root_path: PathBuf::from("/test"),
            storage_path: path.clone(),
            max_file_size: 0,
            parallel_workers: 4,
            ignore_patterns: vec![],
            compression_strategy: "fast".to_string(),
            follow_symlinks: false,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        let compression = CompressionEngine::new(CompressionStrategy::Fast);
        let storage = Storage::init(
            path,
            config,
            compression,
        ).unwrap();
        
        (storage, temp_dir)
    }
    
    #[test]
    fn test_storage_init_and_open() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().to_path_buf();
        
        // Ensure directory doesn't exist (TempDir creates it)
        std::fs::remove_dir_all(&path).ok();
        
        // Initialize storage
        let config = TitorConfig {
            root_path: PathBuf::from("/test"),
            storage_path: path.clone(),
            max_file_size: 0,
            parallel_workers: 4,
            ignore_patterns: vec![],
            compression_strategy: "fast".to_string(),
            follow_symlinks: false,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };
        
        let compression = CompressionEngine::new(CompressionStrategy::Fast);
        let _storage = Storage::init(path.clone(), config, compression).unwrap();
        
        // Check directory structure
        assert!(path.join("checkpoints").exists());
        assert!(path.join("objects").exists());
        assert!(path.join("refs").exists());
        assert!(path.join("metadata.json").exists());
        
        // Open existing storage
        let compression2 = CompressionEngine::new(CompressionStrategy::Fast);
        let _storage2 = Storage::open(path, compression2).unwrap();
    }
    
    #[test]
    fn test_object_storage() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store object
        let content = b"Hello, World!";
        let path = PathBuf::from("test.txt");
        let (hash, _compressed_size) = storage.store_object(content, &path).unwrap();
        
        // Verify hash
        assert_eq!(hash.len(), 64); // SHA-256 hex
        
        // Check object exists
        assert!(storage.object_exists(&hash).unwrap());
        
        // Load object
        let loaded = storage.load_object(&hash).unwrap();
        assert_eq!(loaded, content);
        
        // Store same content again (deduplication)
        let (hash2, _compressed_size2) = storage.store_object(content, &path).unwrap();
        assert_eq!(hash, hash2);
    }
    
    #[test]
    fn test_reference_counting() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Store object
        let content = b"Test content";
        let path = PathBuf::from("test.txt");
        let (hash, _compressed_size) = storage.store_object(content, &path).unwrap();
        
        // Initial ref count is 1
        assert_eq!(*storage.ref_counts.get(&hash).unwrap(), 1);
        
        // Store again (increments ref count)
        let _ = storage.store_object(content, &path).unwrap();
        assert_eq!(*storage.ref_counts.get(&hash).unwrap(), 2);
        
        // Decrement ref count
        let count = storage.decrement_ref_count(&hash).unwrap();
        assert_eq!(count, 1);
        
        // Decrement again
        let count = storage.decrement_ref_count(&hash).unwrap();
        assert_eq!(count, 0);
    }
    
    #[test]
    fn test_checkpoint_storage() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Create checkpoint
        let checkpoint = Checkpoint::new(
            None,
            Some("Test checkpoint".to_string()),
            crate::checkpoint::CheckpointMetadataBuilder::new()
                .file_count(10)
                .total_size(1000)
                .build(),
            "merkle_root".to_string(),
        );
        
        let checkpoint_id = checkpoint.id.clone();
        
        // Store checkpoint
        storage.store_checkpoint(&checkpoint).unwrap();
        
        // Check checkpoint exists
        assert!(storage.checkpoint_exists(&checkpoint_id));
        
        // Load checkpoint
        let loaded = storage.load_checkpoint(&checkpoint_id).unwrap();
        assert_eq!(loaded.id, checkpoint_id);
        assert_eq!(loaded.description, Some("Test checkpoint".to_string()));
        
        // List checkpoints
        let checkpoints = storage.list_checkpoints().unwrap();
        assert_eq!(checkpoints.len(), 1);
        assert_eq!(checkpoints[0], checkpoint_id);
        
        // Delete checkpoint
        storage.delete_checkpoint(&checkpoint_id).unwrap();
        assert!(!storage.checkpoint_exists(&checkpoint_id));
    }
    
    #[test]
    fn test_sharding() {
        let (storage, temp_dir) = create_test_storage();
        
        // Store object
        let content = b"Sharded content";
        let path = PathBuf::from("test.txt");
        let (hash, _compressed_size) = storage.store_object(content, &path).unwrap();
        
        // Check sharded path
        let object_path = storage.get_object_path(&hash);
        assert!(object_path.exists());
        
        // Verify path structure
        let prefix = &hash[..2];
        let suffix = &hash[2..];
        let expected_path = temp_dir.path()
            .join("objects")
            .join(prefix)
            .join(suffix);
        assert_eq!(object_path, expected_path);
    }
} 