//! Performance tests for hybrid optimization

use titor::{Titor, CheckpointOptions};
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;
use std::time::Instant;

#[test]
#[ignore] // Run with: cargo test --test performance_test -- --ignored
fn test_performance_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();
    let storage = root.join(".titor_test");
    
    // Create a large directory structure (1000 files)
    println!("Creating test directory with 1000 files...");
    for i in 0..10 {
        let dir = root.join(format!("dir{}", i));
        fs::create_dir(&dir).unwrap();
        
        for j in 0..100 {
            let file = dir.join(format!("file{}.txt", j));
            fs::write(&file, format!("Content for file {} in dir {}", j, i)).unwrap();
        }
    }
    
    // Initialize Titor
    let mut titor = Titor::init(root.to_path_buf(), storage).unwrap();
    
    // Create initial checkpoint
    println!("Creating initial checkpoint...");
    let start = Instant::now();
    let checkpoint1 = titor.checkpoint(Some("Initial state".to_string())).unwrap();
    let initial_time = start.elapsed();
    println!("Initial checkpoint took: {:?}", initial_time);
    
    // Modify just one file
    fs::write(root.join("dir5/file50.txt"), "Modified content").unwrap();
    
    // Create second checkpoint (should be much faster)
    println!("Creating optimized checkpoint after 1 file change...");
    let start = Instant::now();
    let checkpoint2 = titor.checkpoint(Some("One file modified".to_string())).unwrap();
    let optimized_time = start.elapsed();
    println!("Optimized checkpoint took: {:?}", optimized_time);
    
    // The optimized checkpoint should be significantly faster
    assert!(optimized_time < initial_time / 2, 
        "Optimized checkpoint ({:?}) should be at least 2x faster than initial ({:?})",
        optimized_time, initial_time);
    
    println!("Performance improvement: {:.1}x faster", 
        initial_time.as_secs_f64() / optimized_time.as_secs_f64());
}

#[test]
fn test_verify_mode() {
    let temp_dir = TempDir::new().unwrap();
    let root = temp_dir.path();
    let storage = root.join(".titor_test");
    
    // Create test files
    for i in 0..10 {
        fs::write(root.join(format!("file{}.txt", i)), format!("Content {}", i)).unwrap();
    }
    
    // Initialize Titor
    let mut titor = Titor::init(root.to_path_buf(), storage).unwrap();
    
    // Create initial checkpoint
    let checkpoint1 = titor.checkpoint(Some("Initial".to_string())).unwrap();
    
    // Modify a file
    fs::write(root.join("file5.txt"), "Modified").unwrap();
    
    // Create checkpoint with verify mode
    let options = CheckpointOptions {
        description: Some("Verified checkpoint".to_string()),
        verify: Some(true),
        ..Default::default()
    };
    
    let checkpoint2 = titor.checkpoint_with_options(options).unwrap();
    
    // Verify that the checkpoint was created successfully
    assert_ne!(checkpoint1.id, checkpoint2.id);
}