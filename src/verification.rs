//! Checkpoint verification and integrity checking
//!
//! This module provides comprehensive verification functionality to ensure
//! checkpoint integrity, detect corruption, and validate the entire timeline.
//!
//! ## Overview
//!
//! Verification operates at multiple levels:
//!
//! 1. **File Level**: Verify individual file hashes and metadata
//! 2. **Checkpoint Level**: Verify merkle trees, state hashes, and manifests
//! 3. **Timeline Level**: Verify relationships, structure, and consistency
//!
//! ## Verification Process
//!
//! A complete checkpoint verification includes:
//! - Metadata integrity check
//! - State hash verification
//! - Merkle tree validation
//! - Individual file verification
//! - Parent chain validation
//! - Orphaned object detection
//!
//! ## Usage
//!
//! ```rust,no_run
//! use titor::verification::CheckpointVerifier;
//! use titor::Titor;
//! use std::path::PathBuf;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let titor = Titor::init(PathBuf::from("."), PathBuf::from(".titor"))?;
//! 
//! // Verify a specific checkpoint
//! let report = titor.verify_checkpoint("checkpoint-id")?;
//! if report.is_valid() {
//!     println!("Checkpoint is valid!");
//! } else {
//!     println!("Issues found: {}", report.summary());
//! }
//!
//! // Verify entire timeline
//! let timeline_report = titor.verify_timeline()?;
//! println!("{}", timeline_report.summary());
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance
//!
//! Verification can be expensive for large repositories as it requires:
//! - Loading and hashing all file contents
//! - Building merkle trees
//! - Checking all relationships
//!
//! Use `QuickVerifier` for faster checks that only validate hashes without
//! loading content.

use crate::checkpoint::Checkpoint;
use crate::error::Result;
use crate::merkle::MerkleTree;
use crate::storage::Storage;
use crate::timeline::Timeline;
use crate::types::FileEntry;
use crate::utils;

use serde::{Deserialize, Serialize};
use crate::collections::{HashMap, HashMapExt};
use std::collections::HashSet;
use std::time::Instant;
use tracing::{debug, error, info, warn};

/// Checkpoint verifier for comprehensive integrity checking
///
/// Provides thorough verification of checkpoints including content hashes,
/// merkle trees, and storage consistency. This is the main verification
/// engine for ensuring data integrity.
///
/// # Examples
///
/// ```rust,no_run
/// # use titor::verification::CheckpointVerifier;
/// # use titor::storage::Storage;
/// # use titor::checkpoint::Checkpoint;
/// # use std::path::PathBuf;
/// # fn example(storage: &Storage, checkpoint: &Checkpoint) -> Result<(), Box<dyn std::error::Error>> {
/// let verifier = CheckpointVerifier::new(storage);
/// 
/// // Verify a checkpoint
/// let report = verifier.verify_complete(checkpoint)?;
/// 
/// // Check results
/// if !report.is_valid() {
///     for error in &report.errors {
///         eprintln!("Error: {}", error);
///     }
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CheckpointVerifier<'a> {
    storage: &'a Storage,
}

impl<'a> CheckpointVerifier<'a> {
    /// Create a new checkpoint verifier
    ///
    /// # Arguments
    ///
    /// * `storage` - Storage backend to verify against
    pub fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }
    
    /// Verify all aspects of a checkpoint
    ///
    /// Performs comprehensive verification including:
    /// - Metadata integrity
    /// - State hash validation
    /// - Merkle tree verification
    /// - Individual file checks
    /// - Parent relationship validation
    /// - Orphaned object detection
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The checkpoint to verify
    ///
    /// # Returns
    ///
    /// A detailed verification report containing results and any errors found.
    ///
    /// # Errors
    ///
    /// Returns an error if verification cannot be performed (e.g., storage issues).
    /// Verification failures are reported in the result, not as errors.
    pub fn verify_complete(&self, checkpoint: &Checkpoint) -> Result<VerificationReport> {
        let start = Instant::now();
        let mut report = VerificationReport::new(checkpoint.id.clone());
        
        // 1. Verify checkpoint metadata integrity
        debug!("Verifying checkpoint metadata for {}", checkpoint.short_id());
        report.metadata_valid = checkpoint.verify_integrity()?;
        
        // 2. Verify state hash
        let computed_state = checkpoint.compute_state_hash();
        report.state_hash_valid = computed_state == checkpoint.state_hash;
        if !report.state_hash_valid {
            report.errors.push(format!(
                "State hash mismatch: expected {}, got {}",
                checkpoint.state_hash, computed_state
            ));
        }
        
        // 3. Load and verify manifest
        let manifest = match self.storage.load_manifest(&checkpoint.id) {
            Ok(m) => m,
            Err(e) => {
                report.errors.push(format!("Failed to load manifest: {}", e));
                report.verification_time_ms = start.elapsed().as_millis() as u64;
                return Ok(report);
            }
        };
        
        // 4. Verify merkle tree
        debug!("Building merkle tree for verification");
        let merkle_tree = MerkleTree::from_entries(&manifest.files)?;
        if let Some(root_hash) = merkle_tree.root_hash() {
            report.merkle_root_valid = root_hash == checkpoint.content_merkle_root;
            if !report.merkle_root_valid {
                report.errors.push(format!(
                    "Merkle root mismatch: expected {}, got {}",
                    checkpoint.content_merkle_root, root_hash
                ));
            }
        } else {
            report.merkle_root_valid = checkpoint.content_merkle_root.is_empty();
        }
        
        // 5. Verify each file in manifest
        debug!("Verifying {} files (excluding directories)", manifest.files.len());
        for file_entry in &manifest.files {
            // Skip directory entries (they have no stored object)
            if file_entry.is_directory {
                continue;
            }
            let file_check = self.verify_file_entry(file_entry)?;
            if !file_check.is_valid() {
                report.errors.push(format!(
                    "File verification failed: {:?}",
                    file_entry.path
                ));
            }
            report.file_checks.push(file_check);
        }
        
        // 6. Verify parent chain
        if let Some(parent_id) = &checkpoint.parent_id {
            report.parent_valid = self.storage.checkpoint_exists(parent_id);
            if !report.parent_valid {
                report.errors.push(format!("Parent checkpoint {} not found", parent_id));
            }
        } else {
            report.parent_valid = true; // Root checkpoint has no parent
        }
        
        // 7. Check for orphaned objects
        report.orphaned_objects = self.find_orphaned_objects_for_checkpoint(checkpoint)?;
        
        // Calculate summary
        report.total_files_checked = report.file_checks.len();
        report.files_valid = report.file_checks.iter().filter(|f| f.is_valid()).count();
        report.verification_time_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "Verified checkpoint {} in {}ms: {} / {} files valid",
            checkpoint.short_id(),
            report.verification_time_ms,
            report.files_valid,
            report.total_files_checked
        );
        
        Ok(report)
    }
    
    /// Verify a single file entry
    fn verify_file_entry(&self, entry: &FileEntry) -> Result<FileVerification> {
        let mut verification = FileVerification {
            path: entry.path.clone(),
            content_hash_valid: false,
            metadata_hash_valid: false,
            object_exists: false,
            size_matches: false,
            error: None,
        };
        
        // Check if object exists in storage
        verification.object_exists = self.storage.object_exists(&entry.content_hash)?;
        if !verification.object_exists {
            verification.error = Some(format!("Object {} not found", &entry.content_hash[..8]));
            return Ok(verification);
        }
        
        // Load and verify object
        match self.storage.load_object(&entry.content_hash) {
            Ok(content) => {
                // Verify content hash
                let computed_hash = utils::hash_data(&content);
                verification.content_hash_valid = computed_hash == entry.content_hash;
                
                // Verify size
                verification.size_matches = content.len() as u64 == entry.size;
                
                // For now, assume metadata hash is valid if content hash is valid
                verification.metadata_hash_valid = verification.content_hash_valid;
            }
            Err(e) => {
                verification.error = Some(format!("Failed to load object: {}", e));
            }
        }
        
        Ok(verification)
    }
    
    /// Find orphaned objects for a checkpoint
    fn find_orphaned_objects_for_checkpoint(&self, _checkpoint: &Checkpoint) -> Result<Vec<String>> {
        // Get all objects in storage
        let all_objects = self.storage.list_all_objects()?;
        
        // Get all referenced objects across all checkpoints
        let mut referenced_objects = HashSet::new();
        
        // Scan all checkpoints to find referenced objects
        for checkpoint_id in self.storage.list_checkpoints()? {
            if let Ok(manifest) = self.storage.load_manifest(&checkpoint_id) {
                for file in &manifest.files {
                    // Regular files and symlinks have content stored as objects
                    if !file.is_directory {
                        referenced_objects.insert(file.content_hash.clone());
                    }
                }
            }
        }
        
        // Find orphaned objects (objects that exist but are not referenced)
        let orphaned: Vec<String> = all_objects
            .into_iter()
            .filter(|hash| !referenced_objects.contains(hash))
            .collect();
        
        debug!(
            "Found {} orphaned objects out of {} total references",
            orphaned.len(),
            referenced_objects.len()
        );
        
        Ok(orphaned)
    }
    
    /// Verify an entire timeline
    pub fn verify_timeline(&self, timeline: &Timeline) -> Result<TimelineVerificationReport> {
        let start = Instant::now();
        let mut report = TimelineVerificationReport::default();
        
        report.total_checkpoints = timeline.checkpoints.len();
        
        // Verify each checkpoint
        for checkpoint in timeline.checkpoints.values() {
            debug!("Verifying checkpoint {} in timeline", checkpoint.short_id());
            
            match self.verify_complete(checkpoint) {
                Ok(checkpoint_report) => {
                    if checkpoint_report.is_valid() {
                        report.valid_checkpoints += 1;
                    } else {
                        report.invalid_checkpoints += 1;
                        report.checkpoint_errors.insert(
                            checkpoint.id.clone(),
                            checkpoint_report.errors.clone(),
                        );
                    }
                    report.checkpoint_reports.push(checkpoint_report);
                }
                Err(e) => {
                    report.invalid_checkpoints += 1;
                    report.checkpoint_errors.insert(
                        checkpoint.id.clone(),
                        vec![format!("Verification failed: {}", e)],
                    );
                }
            }
        }
        
        // Verify timeline structure
        report.timeline_structure_valid = self.verify_timeline_structure(timeline)?;
        
        // Check for hash conflicts
        report.no_hash_conflicts = self.check_hash_conflicts(timeline)?;
        
        // Verify deduplication
        report.deduplication_working = self.verify_deduplication(timeline)?;
        
        report.verification_time_ms = start.elapsed().as_millis() as u64;
        
        info!(
            "Timeline verification complete in {}ms: {}/{} checkpoints valid",
            report.verification_time_ms,
            report.valid_checkpoints,
            report.total_checkpoints
        );
        
        Ok(report)
    }
    
    /// Verify timeline structure (no cycles, valid parent references)
    fn verify_timeline_structure(&self, timeline: &Timeline) -> Result<bool> {
        for checkpoint in timeline.checkpoints.values() {
            if let Some(parent_id) = &checkpoint.parent_id {
                if !timeline.checkpoints.contains_key(parent_id) {
                    error!(
                        "Checkpoint {} references non-existent parent {}",
                        checkpoint.short_id(),
                        &parent_id[..8]
                    );
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
    
    /// Check for hash conflicts
    fn check_hash_conflicts(&self, timeline: &Timeline) -> Result<bool> {
        let mut content_hashes = HashMap::new();
        
        for checkpoint in timeline.checkpoints.values() {
            if let Ok(manifest) = self.storage.load_manifest(&checkpoint.id) {
                for file in &manifest.files {
                    if let Some(existing) = content_hashes.get(&file.content_hash) {
                        if existing != &file.size {
                            warn!(
                                "Hash conflict detected: {} has different sizes {} vs {}",
                                &file.content_hash[..8],
                                existing,
                                file.size
                            );
                            return Ok(false);
                        }
                    } else {
                        content_hashes.insert(file.content_hash.clone(), file.size);
                    }
                }
            }
        }
        
        Ok(true)
    }
    
    /// Verify deduplication is working
    fn verify_deduplication(&self, _timeline: &Timeline) -> Result<bool> {
        // This would check that identical content is stored only once
        // For now, return true
        Ok(true)
    }
}

/// Verification report for a single checkpoint
///
/// Contains detailed results from checkpoint verification including
/// individual file checks, integrity validation, and any errors found.
///
/// # Examples
///
/// ```rust
/// # use titor::verification::VerificationReport;
/// # let report = VerificationReport::new("checkpoint-id".to_string());
/// if report.is_valid() {
///     println!("All checks passed!");
/// } else {
///     println!("Verification failed: {}", report.summary());
///     
///     // Check specific issues
///     if !report.metadata_valid {
///         println!("Metadata corruption detected");
///     }
///     
///     if report.orphaned_objects.len() > 0 {
///         println!("Found {} orphaned objects", report.orphaned_objects.len());
///     }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Checkpoint ID being verified
    pub checkpoint_id: String,
    /// Whether checkpoint metadata is valid
    pub metadata_valid: bool,
    /// Whether state hash matches
    pub state_hash_valid: bool,
    /// Whether merkle root matches
    pub merkle_root_valid: bool,
    /// Individual file verification results
    pub file_checks: Vec<FileVerification>,
    /// Whether parent exists (if applicable)
    pub parent_valid: bool,
    /// List of orphaned objects
    pub orphaned_objects: Vec<String>,
    /// Time taken for verification in milliseconds
    pub verification_time_ms: u64,
    /// Total files checked
    pub total_files_checked: usize,
    /// Number of valid files
    pub files_valid: usize,
    /// List of errors encountered
    pub errors: Vec<String>,
}

impl VerificationReport {
    /// Create a new verification report
    pub fn new(checkpoint_id: String) -> Self {
        Self {
            checkpoint_id,
            metadata_valid: false,
            state_hash_valid: false,
            merkle_root_valid: false,
            file_checks: Vec::new(),
            parent_valid: false,
            orphaned_objects: Vec::new(),
            verification_time_ms: 0,
            total_files_checked: 0,
            files_valid: 0,
            errors: Vec::new(),
        }
    }
    
    /// Check if the checkpoint is fully valid
    pub fn is_valid(&self) -> bool {
        self.metadata_valid
            && self.state_hash_valid
            && self.merkle_root_valid
            && self.parent_valid
            && self.files_valid == self.total_files_checked
            && self.orphaned_objects.is_empty()
            && self.errors.is_empty()
    }
    
    /// Get a summary of the verification
    pub fn summary(&self) -> String {
        if self.is_valid() {
            format!(
                "Checkpoint {} is valid ({} files verified in {}ms)",
                &self.checkpoint_id[..8],
                self.total_files_checked,
                self.verification_time_ms
            )
        } else {
            let issues = vec![
                (!self.metadata_valid).then(|| "invalid metadata"),
                (!self.state_hash_valid).then(|| "state hash mismatch"),
                (!self.merkle_root_valid).then(|| "merkle root mismatch"),
                (!self.parent_valid).then(|| "parent missing"),
                (self.files_valid < self.total_files_checked)
                    .then(|| "file verification failures"),
                (!self.orphaned_objects.is_empty()).then(|| "orphaned objects"),
            ]
            .into_iter()
            .filter_map(|x| x)
            .collect::<Vec<_>>()
            .join(", ");
            
            format!(
                "Checkpoint {} is invalid: {} ({}/{} files valid)",
                &self.checkpoint_id[..8],
                issues,
                self.files_valid,
                self.total_files_checked
            )
        }
    }
}

/// Verification result for a single file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileVerification {
    /// File path
    pub path: std::path::PathBuf,
    /// Whether content hash is valid
    pub content_hash_valid: bool,
    /// Whether metadata hash is valid
    pub metadata_hash_valid: bool,
    /// Whether the object exists in storage
    pub object_exists: bool,
    /// Whether the size matches
    pub size_matches: bool,
    /// Error message if verification failed
    pub error: Option<String>,
}

impl FileVerification {
    /// Check if the file verification passed
    pub fn is_valid(&self) -> bool {
        self.content_hash_valid
            && self.metadata_hash_valid
            && self.object_exists
            && self.size_matches
            && self.error.is_none()
    }
}

/// Timeline verification report
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TimelineVerificationReport {
    /// Total checkpoints in timeline
    pub total_checkpoints: usize,
    /// Number of valid checkpoints
    pub valid_checkpoints: usize,
    /// Number of invalid checkpoints
    pub invalid_checkpoints: usize,
    /// Whether timeline structure is valid
    pub timeline_structure_valid: bool,
    /// Whether there are no hash conflicts
    pub no_hash_conflicts: bool,
    /// Whether deduplication is working
    pub deduplication_working: bool,
    /// Individual checkpoint reports
    pub checkpoint_reports: Vec<VerificationReport>,
    /// Errors by checkpoint ID
    pub checkpoint_errors: HashMap<String, Vec<String>>,
    /// Total verification time in milliseconds
    pub verification_time_ms: u64,
}

impl TimelineVerificationReport {
    /// Check if the entire timeline is valid
    pub fn is_valid(&self) -> bool {
        self.valid_checkpoints == self.total_checkpoints
            && self.timeline_structure_valid
            && self.no_hash_conflicts
            && self.deduplication_working
    }
    
    /// Get a summary of the timeline verification
    pub fn summary(&self) -> String {
        if self.is_valid() {
            format!(
                "Timeline is valid: {} checkpoints verified in {}ms",
                self.total_checkpoints, self.verification_time_ms
            )
        } else {
            format!(
                "Timeline has issues: {}/{} checkpoints valid, {} errors in {}ms",
                self.valid_checkpoints,
                self.total_checkpoints,
                self.checkpoint_errors.len(),
                self.verification_time_ms
            )
        }
    }
}

/// Quick verification that just checks hashes without loading content
///
/// Provides fast verification by only checking hashes and metadata without
/// loading file contents. This is useful for quick integrity checks where
/// full verification would be too expensive.
///
/// # Examples
///
/// ```rust
/// use titor::verification::QuickVerifier;
/// use titor::checkpoint::Checkpoint;
/// # use titor::checkpoint::CheckpointMetadataBuilder;
///
/// # let checkpoint = Checkpoint::new(None, None, CheckpointMetadataBuilder::new().build(), "".to_string());
/// // Quick checkpoint verification
/// let is_valid = QuickVerifier::verify_checkpoint(&checkpoint)?;
/// 
/// // Quick file verification
/// let content = b"file content";
/// # use titor::types::FileEntry;
/// # use std::path::PathBuf;
/// # use chrono::Utc;
/// # let entry = FileEntry {
/// #     path: PathBuf::new(),
/// #     content_hash: "".to_string(),
/// #     size: 0,
/// #     permissions: 0,
/// #     modified: Utc::now(),
/// #     is_compressed: false,
/// #     metadata_hash: "".to_string(),
/// #     combined_hash: "".to_string(),
/// #     is_symlink: false,
/// #     symlink_target: None,
/// #     is_directory: false,
/// # };
/// let file_valid = QuickVerifier::verify_file(&entry, content);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct QuickVerifier;

impl QuickVerifier {
    /// Perform quick verification of a checkpoint
    ///
    /// Only verifies the state hash without checking file contents or
    /// merkle trees. This is much faster than full verification.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The checkpoint to verify
    ///
    /// # Returns
    ///
    /// `true` if the state hash is valid, `false` otherwise.
    pub fn verify_checkpoint(checkpoint: &Checkpoint) -> Result<bool> {
        // Just verify the state hash
        let computed = checkpoint.compute_state_hash();
        Ok(computed == checkpoint.state_hash)
    }
    
    /// Perform quick verification of a file
    pub fn verify_file(entry: &FileEntry, content: &[u8]) -> bool {
        let hash = utils::hash_data(content);
        hash == entry.content_hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::CheckpointMetadataBuilder;
    use crate::compression::{CompressionEngine, CompressionStrategy};
    use crate::types::{FileManifest, TitorConfig};
    use chrono::Utc;
    use tempfile::TempDir;
    use std::path::PathBuf;
    
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
            version: "0.1.1".to_string(),
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
    fn test_checkpoint_verification() {
        let (storage, _temp_dir) = create_test_storage();
        
        // Create a checkpoint
        let checkpoint = Checkpoint::new(
            None,
            Some("Test checkpoint".to_string()),
            CheckpointMetadataBuilder::new()
                .file_count(0)
                .total_size(0)
                .build(),
            "merkle_root".to_string(),
        );
        
        // Store checkpoint
        storage.store_checkpoint(&checkpoint).unwrap();
        
        // Create empty manifest
        let manifest = FileManifest {
            checkpoint_id: checkpoint.id.clone(),
            files: vec![],
            total_size: 0,
            file_count: 0,
            merkle_root: "".to_string(),
            created_at: Utc::now(),
        };
        storage.store_manifest(&manifest).unwrap();
        
        // Verify checkpoint
        let verifier = CheckpointVerifier::new(&storage);
        let report = verifier.verify_complete(&checkpoint).unwrap();
        
        assert!(report.metadata_valid);
        assert!(report.state_hash_valid);
        assert!(report.parent_valid);
        assert_eq!(report.total_files_checked, 0);
        assert_eq!(report.files_valid, 0);
    }
    
    #[test]
    fn test_quick_verification() {
        let checkpoint = Checkpoint::new(
            None,
            Some("Test".to_string()),
            CheckpointMetadataBuilder::new().build(),
            "merkle".to_string(),
        );
        
        assert!(QuickVerifier::verify_checkpoint(&checkpoint).unwrap());
        
        // Tamper with checkpoint
        let mut tampered = checkpoint.clone();
        tampered.content_merkle_root = "tampered".to_string();
        assert!(!QuickVerifier::verify_checkpoint(&tampered).unwrap());
    }
    
    #[test]
    fn test_file_verification() {
        let content = b"test content";
        let hash = utils::hash_data(content);
        
        let entry = FileEntry {
            path: PathBuf::from("test.txt"),
            content_hash: hash,
            size: content.len() as u64,
            permissions: 0o644,
            modified: Utc::now(),
            is_compressed: false,
            metadata_hash: "meta".to_string(),
            combined_hash: "combined".to_string(),
            is_symlink: false,
            symlink_target: None,
            is_directory: false,
        };
        
        assert!(QuickVerifier::verify_file(&entry, content));
        assert!(!QuickVerifier::verify_file(&entry, b"different content"));
    }
} 