//! Integration tests for Titor
//!
//! This module contains comprehensive integration tests that verify
//! the entire system works correctly end-to-end.

#[cfg(test)]
mod integration_tests {
    use crate::*;
    use tempfile::TempDir;
    use std::fs;
    
    #[test]
    fn test_basic_workflow() {
        // Create test directories
        let root_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        // Initialize Titor
        let mut titor = TitorBuilder::new()
            .compression_strategy(CompressionStrategy::Fast)
            .build(
                root_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        // Create initial files
        fs::write(root_dir.path().join("README.md"), "# My Project").unwrap();
        fs::write(root_dir.path().join("main.rs"), "fn main() {}").unwrap();
        
        // Create first checkpoint
        let checkpoint1 = titor.checkpoint(Some("Initial commit".to_string())).unwrap();
        assert_eq!(checkpoint1.metadata.file_count, 2);
        
        // Modify files
        fs::write(root_dir.path().join("main.rs"), "fn main() { println!(\"Hello\"); }").unwrap();
        fs::write(root_dir.path().join("lib.rs"), "pub fn hello() {}").unwrap();
        
        // Create second checkpoint
        let checkpoint2 = titor.checkpoint(Some("Add hello function".to_string())).unwrap();
        assert_eq!(checkpoint2.parent_id, Some(checkpoint1.id.clone()));
        
        // Restore to first checkpoint
        let restore_result = titor.restore(&checkpoint1.id).unwrap();
        assert_eq!(restore_result.files_restored, 2);
        assert_eq!(restore_result.files_deleted, 1);
        
        // Verify files are restored
        let main_content = fs::read_to_string(root_dir.path().join("main.rs")).unwrap();
        assert_eq!(main_content, "fn main() {}");
        assert!(!root_dir.path().join("lib.rs").exists());
    }
} 