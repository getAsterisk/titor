//! Checkpoint index for fast change detection
//!
//! This module provides the `CheckpointIndex` struct which maintains a persistent
//! mapping of file paths to metadata and hashes. This enables fast change detection
//! by comparing metadata instead of re-reading and hashing files.
//!
//! The index also supports directory-level hashing for skipping entire unchanged
//! subtrees during scans.

use crate::error::{Result, TitorError};
use crate::types::IndexEntry;
use sled::Db;
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, trace};
use sha2::{Sha256, Digest};
use std::collections::BTreeMap;

/// Persistent index for a checkpoint
///
/// Stores file metadata and hashes to enable fast change detection
/// without reading file contents. Uses sled as an embedded key-value
/// database for persistence and performance.
pub struct CheckpointIndex {
    /// The sled database instance
    db: Db,
    /// Checkpoint ID this index belongs to
    checkpoint_id: String,
    /// Storage path for indexes
    storage_path: PathBuf,
}

impl CheckpointIndex {
    /// Create or open a checkpoint index
    ///
    /// # Arguments
    ///
    /// * `storage_path` - Base storage directory (usually .titor)
    /// * `checkpoint_id` - ID of the checkpoint this index belongs to
    ///
    /// # Returns
    ///
    /// A new `CheckpointIndex` instance
    pub fn new(storage_path: &Path, checkpoint_id: &str) -> Result<Self> {
        info!("Opening checkpoint index for: {}", checkpoint_id);
        
        // Create indexes directory
        let index_path = storage_path
            .join("indexes")
            .join(format!("{}.db", checkpoint_id));
        
        std::fs::create_dir_all(index_path.parent().unwrap())?;
        
        // Open or create the index database
        let db = sled::open(&index_path)
            .map_err(|e| TitorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to open index database: {}", e)
            )))?;
        
        Ok(CheckpointIndex {
            db,
            checkpoint_id: checkpoint_id.to_string(),
            storage_path: storage_path.to_path_buf(),
        })
    }
    
    /// Insert an entry into the index
    ///
    /// # Arguments
    ///
    /// * `path` - File path (relative to root)
    /// * `entry` - Index entry containing metadata and hash
    pub fn insert(&self, path: &Path, entry: IndexEntry) -> Result<()> {
        trace!("Inserting index entry for: {}", path.display());
        
        let key = path_to_key(path);
        let value = bincode::serde::encode_to_vec(&entry, bincode::config::standard())
            .map_err(|e| TitorError::internal(format!("Failed to serialize index entry: {}", e)))?;
        
        self.db.insert(key, value)?;
        
        Ok(())
    }
    
    /// Get an entry from the index
    ///
    /// # Arguments
    ///
    /// * `path` - File path to look up
    ///
    /// # Returns
    ///
    /// The index entry if found, or None
    pub fn get(&self, path: &Path) -> Result<Option<IndexEntry>> {
        let key = path_to_key(path);
        
        match self.db.get(&key)? {
            Some(value) => {
                let (entry, _) = bincode::serde::decode_from_slice(&value, bincode::config::standard())
                    .map_err(|e| TitorError::internal(format!("Failed to deserialize index entry: {}", e)))?;
                Ok(Some(entry))
            }
            None => Ok(None),
        }
    }
    
    /// Delete an entry from the index
    ///
    /// # Arguments
    ///
    /// * `path` - File path to delete
    pub fn delete(&self, path: &Path) -> Result<()> {
        trace!("Deleting index entry for: {}", path.display());
        
        let key = path_to_key(path);
        self.db.remove(&key)?;
        
        Ok(())
    }
    
    /// Iterate over all entries in the index
    ///
    /// # Returns
    ///
    /// An iterator over (PathBuf, IndexEntry) pairs
    pub fn iter(&self) -> impl Iterator<Item = Result<(PathBuf, IndexEntry)>> + '_ {
        self.db.iter().map(|result| {
            let (key, value) = result?;
            
            let path = key_to_path(&key)?;
            let (entry, _) = bincode::serde::decode_from_slice(&value, bincode::config::standard())
                .map_err(|e| TitorError::internal(format!("Failed to deserialize index entry: {}", e)))?;
            
            Ok((path, entry))
        })
    }
    
    /// Compute hash for a directory based on its children
    ///
    /// Creates a deterministic hash by sorting child paths and combining
    /// their hashes. This allows skipping entire directory trees if the
    /// directory hash hasn't changed.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory path
    ///
    /// # Returns
    ///
    /// SHA-256 hash of the directory contents
    pub fn compute_dir_hash(&self, dir: &Path) -> Result<String> {
        debug!("Computing directory hash for: {}", dir.display());
        
        // Collect all child entries
        let mut children = BTreeMap::new();
        
        for result in self.iter() {
            let (path, entry) = result?;
            
            // Check if this path is a direct child of the directory
            if let Some(parent) = path.parent() {
                if parent == dir {
                    children.insert(path, entry);
                }
            }
        }
        
        // Create hash from sorted children
        let mut hasher = Sha256::new();
        
        for (path, entry) in children {
            // Include path and hash in directory hash
            hasher.update(path.to_string_lossy().as_bytes());
            hasher.update(b"\0"); // Separator
            
            if entry.is_dir {
                // For subdirectories, use their dir_hash if available
                if let Some(ref dir_hash) = entry.dir_hash {
                    hasher.update(dir_hash.as_bytes());
                }
            } else {
                // For files, use content hash
                hasher.update(entry.hash.as_bytes());
            }
            hasher.update(b"\n"); // Entry separator
        }
        
        let hash = format!("{:x}", hasher.finalize());
        trace!("Directory hash for {}: {}", dir.display(), hash);
        
        Ok(hash)
    }
    
    /// Update directory hash for a given path
    ///
    /// Computes and updates the dir_hash field for a directory entry
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory path to update
    pub fn update_dir_hash(&self, dir: &Path) -> Result<()> {
        if let Some(mut entry) = self.get(dir)? {
            if entry.is_dir {
                let hash = self.compute_dir_hash(dir)?;
                entry.dir_hash = Some(hash);
                self.insert(dir, entry)?;
            }
        }
        
        Ok(())
    }
    
    /// Flush all pending writes to disk
    pub fn flush(&self) -> Result<()> {
        self.db.flush()?;
        Ok(())
    }
    
    /// Get the size of the index in bytes
    pub fn size(&self) -> Result<u64> {
        Ok(self.db.size_on_disk()?)
    }
    
    /// Clear all entries from the index
    pub fn clear(&self) -> Result<()> {
        self.db.clear()?;
        Ok(())
    }
}

impl Drop for CheckpointIndex {
    fn drop(&mut self) {
        // Ensure all data is written to disk
        if let Err(e) = self.flush() {
            error!("Failed to flush index on drop: {}", e);
        }
    }
}

/// Convert a path to a database key
fn path_to_key(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

/// Convert a database key to a path
fn key_to_path(key: &[u8]) -> Result<PathBuf> {
    let path_str = std::str::from_utf8(key)
        .map_err(|e| TitorError::internal(format!("Invalid UTF-8 in path: {}", e)))?;
    Ok(PathBuf::from(path_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_checkpoint_index() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path();
        
        // Create index
        let index = CheckpointIndex::new(storage_path, "test-checkpoint").unwrap();
        
        // Test insert and get
        let path = PathBuf::from("test.txt");
        let entry = IndexEntry {
            size: 1024,
            mtime_ns: 1234567890,
            ctime_ns: 1234567890,
            hash: "abc123".to_string(),
            mode: 0o644,
            is_dir: false,
            dir_hash: None,
        };
        
        index.insert(&path, entry.clone()).unwrap();
        
        let retrieved = index.get(&path).unwrap().unwrap();
        assert_eq!(retrieved.size, entry.size);
        assert_eq!(retrieved.hash, entry.hash);
        
        // Test delete
        index.delete(&path).unwrap();
        assert!(index.get(&path).unwrap().is_none());
    }
    
    #[test]
    fn test_directory_hash() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path();
        
        let index = CheckpointIndex::new(storage_path, "test-checkpoint").unwrap();
        
        // Add some files to a directory
        let dir = PathBuf::from("src");
        let file1 = PathBuf::from("src/file1.rs");
        let file2 = PathBuf::from("src/file2.rs");
        
        index.insert(&file1, IndexEntry {
            size: 100,
            mtime_ns: 1234567890,
            ctime_ns: 1234567890,
            hash: "hash1".to_string(),
            mode: 0o644,
            is_dir: false,
            dir_hash: None,
        }).unwrap();
        
        index.insert(&file2, IndexEntry {
            size: 200,
            mtime_ns: 1234567890,
            ctime_ns: 1234567890,
            hash: "hash2".to_string(),
            mode: 0o644,
            is_dir: false,
            dir_hash: None,
        }).unwrap();
        
        // Compute directory hash
        let hash1 = index.compute_dir_hash(&dir).unwrap();
        
        // Hash should be deterministic
        let hash2 = index.compute_dir_hash(&dir).unwrap();
        assert_eq!(hash1, hash2);
        
        // Changing a file should change the directory hash
        index.insert(&file1, IndexEntry {
            size: 100,
            mtime_ns: 1234567890,
            ctime_ns: 1234567890,
            hash: "hash1-modified".to_string(),
            mode: 0o644,
            is_dir: false,
            dir_hash: None,
        }).unwrap();
        
        let hash3 = index.compute_dir_hash(&dir).unwrap();
        assert_ne!(hash1, hash3);
    }
    
    #[test]
    fn test_index_iteration() {
        let temp_dir = TempDir::new().unwrap();
        let storage_path = temp_dir.path();
        
        let index = CheckpointIndex::new(storage_path, "test-checkpoint").unwrap();
        
        // Add multiple entries
        let paths = vec![
            PathBuf::from("file1.txt"),
            PathBuf::from("file2.txt"),
            PathBuf::from("dir/file3.txt"),
        ];
        
        for (i, path) in paths.iter().enumerate() {
            index.insert(path, IndexEntry {
                size: (i + 1) as u64 * 100,
                mtime_ns: 1234567890,
                ctime_ns: 1234567890,
                hash: format!("hash{}", i),
                mode: 0o644,
                is_dir: false,
                dir_hash: None,
            }).unwrap();
        }
        
        // Iterate and collect
        let mut collected = Vec::new();
        for result in index.iter() {
            let (path, _) = result.unwrap();
            collected.push(path);
        }
        
        assert_eq!(collected.len(), paths.len());
        
        // Check all paths are present (order may differ)
        for path in &paths {
            assert!(collected.contains(path));
        }
    }
}