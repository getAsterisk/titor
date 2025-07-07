//! Property-based testing for Titor
//!
//! Uses proptest to verify invariants and properties across
//! randomly generated inputs and operations.

use ::titor::*;
use tempfile::TempDir;
use proptest::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::collections::BTreeMap;
use tracing::info;

/// Strategy for generating file operations
#[derive(Debug, Clone)]
pub enum FileOperation {
    Create { path: PathBuf, content: Vec<u8> },
    Modify { path: PathBuf, content: Vec<u8> },
    Delete { path: PathBuf },
}

/// Generate a random file operation
fn file_operation_strategy() -> impl Strategy<Value = FileOperation> {
    prop_oneof![
        // Create operation
        (path_strategy(), content_strategy()).prop_map(|(path, content)| {
            FileOperation::Create { path, content }
        }),
        // Modify operation
        (path_strategy(), content_strategy()).prop_map(|(path, content)| {
            FileOperation::Modify { path, content }
        }),
        // Delete operation
        path_strategy().prop_map(|path| FileOperation::Delete { path }),
    ]
}

/// Generate random file paths
fn path_strategy() -> impl Strategy<Value = PathBuf> {
    // Generate directory components (0-4 directories)
    let dir_strategy = prop::collection::vec(
        prop_oneof![
            "[a-z]{1,10}".prop_map(|s| s),
            "dir[0-9]{1,3}".prop_map(|s| s),
        ],
        0..=4
    );
    
    // Generate filenames (always non-empty)
    let filename_strategy = prop_oneof![
        "file[0-9]{1,3}\\.txt".prop_map(|s| s),
        "[a-z]{1,8}\\.(txt|rs|md)".prop_map(|s| s),
        "[a-z]{3,10}".prop_map(|s| s), // At least 3 chars to avoid empty
    ];
    
    // Combine directories and filename
    (dir_strategy, filename_strategy).prop_map(|(dirs, filename)| {
        let mut path = PathBuf::new();
        for dir in dirs {
            path = path.join(dir);
        }
        path.join(filename)
    })
}

/// Generate random file content
fn content_strategy() -> impl Strategy<Value = Vec<u8>> {
    prop_oneof![
        // Small text files
        "[a-zA-Z0-9 \n]{1,1000}".prop_map(|s| s.into_bytes()),
        // Binary data
        prop::collection::vec(any::<u8>(), 1..10000),
        // Repetitive patterns
        (any::<u8>(), 1..1000usize).prop_map(|(byte, count)| vec![byte; count]),
    ]
}

/// Apply a file operation to the filesystem
fn apply_operation(root: &Path, op: &FileOperation) -> anyhow::Result<()> {
    match op {
        FileOperation::Create { path, content } => {
            let full_path = root.join(path);
            if let Some(parent) = full_path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(full_path, content)?;
        }
        FileOperation::Modify { path, content } => {
            let full_path = root.join(path);
            if full_path.exists() {
                fs::write(full_path, content)?;
            }
        }
        FileOperation::Delete { path } => {
            let full_path = root.join(path);
            if full_path.exists() {
                fs::remove_file(full_path)?;
            }
        }
    }
    Ok(())
}

/// Compute a hash of the entire directory state
fn compute_directory_hash(root: &Path) -> anyhow::Result<String> {
    use sha2::{Sha256, Digest};
    use walkdir::WalkDir;
    
    let mut hasher = Sha256::new();
    let mut entries = Vec::new();
    
    // Collect all entries
    for entry in WalkDir::new(root).sort_by_file_name() {
        let entry = entry?;
        if entry.file_type().is_file() {
            let path = entry.path();
            let relative = path.strip_prefix(root)?;
            entries.push((relative.to_path_buf(), path.to_path_buf()));
        }
    }
    
    // Sort for deterministic hashing
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    
    // Hash each file
    for (relative, full) in entries {
        hasher.update(relative.to_string_lossy().as_bytes());
        hasher.update(b"\0");
        
        let content = fs::read(&full)?;
        hasher.update(&content);
        hasher.update(b"\0");
    }
    
    Ok(hex::encode(hasher.finalize()))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]
    
    /// Test that checkpoint and restore preserves exact state
    #[test]
    fn checkpoint_restore_identity(
        operations in prop::collection::vec(file_operation_strategy(), 1..100)
    ) {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let mut titor = TitorBuilder::new()
            .compression_strategy(CompressionStrategy::Fast)
            .build(
                temp_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        // Apply operations
        for op in &operations {
            apply_operation(temp_dir.path(), op).unwrap();
        }
        
        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Test checkpoint".to_string())).unwrap();
        let original_hash = compute_directory_hash(temp_dir.path()).unwrap();
        
        // Make more changes
        for i in 0..10 {
            let path = temp_dir.path().join(format!("extra_{}.txt", i));
            fs::write(&path, format!("Extra content {}", i)).unwrap();
        }
        
        // Restore
        titor.restore(&checkpoint.id).unwrap();
        let restored_hash = compute_directory_hash(temp_dir.path()).unwrap();
        
        // Verify identity
        prop_assert_eq!(original_hash, restored_hash);
    }
    
    /// Test that multiple checkpoints maintain independence
    #[test]
    fn checkpoint_independence(
        operation_sets in prop::collection::vec(
            prop::collection::vec(file_operation_strategy(), 1..20),
            2..10
        )
    ) {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let mut titor = TitorBuilder::new()
            .compression_strategy(CompressionStrategy::Fast)
            .build(
                temp_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        // Use BTreeMap for deterministic iteration order
        let mut checkpoint_hashes = BTreeMap::new();
        
        // Create checkpoints for each operation set
        for (idx, operations) in operation_sets.iter().enumerate() {
            // Apply operations
            for op in operations {
                apply_operation(temp_dir.path(), op).unwrap();
            }
            
            // Create checkpoint
            let checkpoint = titor.checkpoint(Some(format!("Checkpoint {}", idx))).unwrap();
            let hash = compute_directory_hash(temp_dir.path()).unwrap();
            checkpoint_hashes.insert(checkpoint.id.clone(), hash);
        }
        
        // Verify each checkpoint can be restored independently
        for (checkpoint_id, expected_hash) in &checkpoint_hashes {
            titor.restore(checkpoint_id).unwrap();
            let actual_hash = compute_directory_hash(temp_dir.path()).unwrap();
            
            if &actual_hash != expected_hash {
                // Debug output on failure
                eprintln!("Checkpoint restoration failed!");
                eprintln!("Checkpoint ID: {}", checkpoint_id);
                eprintln!("Expected hash: {}", expected_hash);
                eprintln!("Actual hash: {}", actual_hash);
                
                // List files in directory
                eprintln!("\nCurrent directory contents:");
                use walkdir::WalkDir;
                for entry in WalkDir::new(temp_dir.path()).sort_by_file_name() {
                    if let Ok(entry) = entry {
                        if entry.file_type().is_file() {
                            let path = entry.path();
                            if let Ok(relative) = path.strip_prefix(temp_dir.path()) {
                                if let Ok(content) = fs::read(path) {
                                    eprintln!("  {}: {} bytes", relative.display(), content.len());
                                }
                            }
                        }
                    }
                }
            }
            
            prop_assert_eq!(&actual_hash, expected_hash);
        }
    }
    
    /// Test that timeline operations maintain consistency
    #[test]
    fn timeline_consistency(
        operations in prop::collection::vec(
            prop_oneof![
                file_operation_strategy().prop_map(|op| TimelineOp::FileOp(op)),
                Just(TimelineOp::Checkpoint),
                Just(TimelineOp::Restore),
            ],
            1..50
        )
    ) {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let mut titor = TitorBuilder::new()
            .compression_strategy(CompressionStrategy::Fast)
            .build(
                temp_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        let mut checkpoints = Vec::new();
        
        // Apply timeline operations
        for op in operations {
            match op {
                TimelineOp::FileOp(file_op) => {
                    apply_operation(temp_dir.path(), &file_op).ok();
                }
                TimelineOp::Checkpoint => {
                    if let Ok(checkpoint) = titor.checkpoint(None) {
                        checkpoints.push(checkpoint.id);
                    }
                }
                TimelineOp::Restore => {
                    if !checkpoints.is_empty() {
                        let idx = checkpoints.len() / 2;
                        titor.restore(&checkpoints[idx]).ok();
                    }
                }
            }
        }
        
        // Verify timeline integrity
        let timeline_verification = titor.verify_timeline().unwrap();
        prop_assert!(timeline_verification.is_valid());
    }
    
    /// Test that compression doesn't affect correctness
    #[test]
    fn compression_correctness(
        content in prop::collection::vec(any::<u8>(), 0..100000),
        should_compress in any::<bool>()
    ) {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let strategy = if should_compress {
            CompressionStrategy::Fast
        } else {
            CompressionStrategy::None
        };
        
        let mut titor = TitorBuilder::new()
            .compression_strategy(strategy)
            .build(
                temp_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        // Write file
        let file_path = temp_dir.path().join("test_file.bin");
        fs::write(&file_path, &content).unwrap();
        
        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Compression test".to_string())).unwrap();
        
        // Modify file
        fs::write(&file_path, b"modified").unwrap();
        
        // Restore
        titor.restore(&checkpoint.id).unwrap();
        
        // Verify content
        let restored_content = fs::read(&file_path).unwrap();
        prop_assert_eq!(restored_content, content);
    }
    
    /// Test that garbage collection preserves all referenced data
    #[test]
    fn garbage_collection_safety(
        checkpoint_count in 2..20usize,
        files_per_checkpoint in 1..10usize
    ) {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let mut titor = TitorBuilder::new()
            .build(
                temp_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        let mut checkpoint_ids = Vec::new();
        
        // Create checkpoints
        for i in 0..checkpoint_count {
            for j in 0..files_per_checkpoint {
                let path = temp_dir.path().join(format!("file_{}_{}.txt", i, j));
                fs::write(&path, format!("Content {} {}", i, j)).unwrap();
            }
            
            let checkpoint = titor.checkpoint(Some(format!("Checkpoint {}", i))).unwrap();
            checkpoint_ids.push(checkpoint.id);
        }
        
        // Run garbage collection
        let gc_stats = titor.gc().unwrap();
        info!("GC collected {} objects", gc_stats.objects_deleted);
        
        // Verify all checkpoints can still be restored
        for checkpoint_id in &checkpoint_ids {
            titor.restore(checkpoint_id).unwrap();
            
            // Verify files exist
            for j in 0..files_per_checkpoint {
                let path = temp_dir.path().join(format!("file_{}_{}.txt", 
                    checkpoint_ids.iter().position(|id| id == checkpoint_id).unwrap(), j));
                prop_assert!(path.exists());
            }
        }
    }
    
    /// Test merkle tree verification
    #[test]
    fn merkle_tree_verification(
        file_count in 1..100usize,
        corrupt_file in any::<bool>()
    ) {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let mut titor = TitorBuilder::new()
            .build(
                temp_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        // Create files
        for i in 0..file_count {
            let path = temp_dir.path().join(format!("file_{}.txt", i));
            fs::write(&path, format!("Content {}", i)).unwrap();
        }
        
        // Create checkpoint
        let checkpoint = titor.checkpoint(Some("Merkle test".to_string())).unwrap();
        let original_merkle = checkpoint.content_merkle_root.clone();
        
        if corrupt_file && file_count > 0 {
            // Corrupt a file
            let path = temp_dir.path().join("file_0.txt");
            fs::write(&path, "corrupted content").unwrap();
            
            // Recompute merkle root
            let new_merkle = titor.compute_current_merkle_root().unwrap();
            
            // Should be different
            prop_assert_ne!(original_merkle, new_merkle);
        } else {
            // No corruption, merkle should match
            let new_merkle = titor.compute_current_merkle_root().unwrap();
            prop_assert_eq!(original_merkle, new_merkle);
        }
    }
}

#[derive(Debug, Clone)]
enum TimelineOp {
    FileOp(FileOperation),
    Checkpoint,
    Restore,
}

#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    proptest! {
        /// Test handling of empty directories
        #[test]
        fn empty_directory_handling(
            dir_count in 0..20usize
        ) {
            let temp_dir = TempDir::new().unwrap();
            let storage_dir = TempDir::new().unwrap();
            
            let mut titor = TitorBuilder::new()
                .build(
                    temp_dir.path().to_path_buf(),
                    storage_dir.path().to_path_buf(),
                )
                .unwrap();
            
            // Create empty directories
            for i in 0..dir_count {
                let dir_path = temp_dir.path().join(format!("empty_dir_{}", i));
                fs::create_dir_all(&dir_path).unwrap();
            }
            
            // Create checkpoint
            let checkpoint = titor.checkpoint(Some("Empty dirs".to_string())).unwrap();
            
            // Add files to directories
            for i in 0..dir_count {
                let file_path = temp_dir.path()
                    .join(format!("empty_dir_{}", i))
                    .join("new_file.txt");
                fs::write(&file_path, "content").unwrap();
            }
            
            // Restore should remove the new files
            titor.restore(&checkpoint.id).unwrap();
            
            // Verify directories are empty or don't exist
            // (TARDIS doesn't track empty directories, so they won't be restored)
            for i in 0..dir_count {
                let dir_path = temp_dir.path().join(format!("empty_dir_{}", i));
                if dir_path.exists() {
                    // If directory exists, it should be empty
                    let entries: Vec<_> = fs::read_dir(&dir_path)
                        .unwrap()
                        .collect();
                    prop_assert!(entries.is_empty());
                }
                // If directory doesn't exist, that's expected behavior for empty directories
            }
        }
        
        /// Test files with special characters
        #[test]
        fn special_character_filenames(
            special_chars in prop::collection::vec(
                prop_oneof![
                    Just(' '),
                    Just('!'),
                    Just('#'),
                    Just('$'),
                    Just('%'),
                    Just('&'),
                    Just('\''),
                    Just('('),
                    Just(')'),
                    Just('+'),
                    Just(','),
                    Just('-'),
                    Just('.'),
                    Just('='),
                    Just('@'),
                    Just('['),
                    Just(']'),
                    Just('^'),
                    Just('_'),
                    Just('`'),
                    Just('{'),
                    Just('}'),
                    Just('~'),
                ],
                1..10
            )
        ) {
            let temp_dir = TempDir::new().unwrap();
            let storage_dir = TempDir::new().unwrap();
            
            let mut titor = TitorBuilder::new()
                .build(
                    temp_dir.path().to_path_buf(),
                    storage_dir.path().to_path_buf(),
                )
                .unwrap();
            
            // Create filename with special characters
            let filename: String = special_chars.into_iter()
                .collect::<String>() + "_file.txt";
            
            // Skip if filename would be invalid on this OS
            let file_path = temp_dir.path().join(&filename);
            if let Err(_) = fs::write(&file_path, "test content") {
                // Invalid filename for this OS, skip test
                return Ok(());
            }
            
            // Create checkpoint
            let checkpoint = titor.checkpoint(Some("Special chars".to_string())).unwrap();
            
            // Delete file
            fs::remove_file(&file_path).unwrap();
            
            // Restore
            titor.restore(&checkpoint.id).unwrap();
            
            // Verify file exists with correct content
            let content = fs::read_to_string(&file_path).unwrap();
            prop_assert_eq!(content, "test content");
        }
        
        /// Test maximum path length handling
        #[test]
        fn max_path_length(
            depth in 10..50usize
        ) {
            let temp_dir = TempDir::new().unwrap();
            let storage_dir = TempDir::new().unwrap();
            
            let mut titor = TitorBuilder::new()
                .build(
                    temp_dir.path().to_path_buf(),
                    storage_dir.path().to_path_buf(),
                )
                .unwrap();
            
            // Create deep directory structure
            let mut current_path = temp_dir.path().to_path_buf();
            for i in 0..depth {
                current_path = current_path.join(format!("d{}", i));
                if let Err(_) = fs::create_dir(&current_path) {
                    // Path too long, stop here
                    break;
                }
            }
            
            // Try to create file at deepest level
            let file_path = current_path.join("deep.txt");
            match fs::write(&file_path, "deep content") {
                Ok(_) => {
                    // Successfully created, test checkpoint/restore
                    let checkpoint = titor.checkpoint(Some("Deep path".to_string())).unwrap();
                    
                    // Modify file
                    fs::write(&file_path, "modified").unwrap();
                    
                    // Restore
                    titor.restore(&checkpoint.id).unwrap();
                    
                    // Verify
                    let content = fs::read_to_string(&file_path).unwrap();
                    prop_assert_eq!(content, "deep content");
                }
                Err(_) => {
                    // Path too long for file creation, that's OK
                }
            }
        }
    }
} 