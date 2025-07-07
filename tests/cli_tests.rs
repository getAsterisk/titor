use std::process::Command;
use tempfile::TempDir;
use std::fs;

#[test]
fn test_cli_verify_prefix() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Initialize repository via CLI
    let status = Command::new("cargo")
        .args(&["run", "--quiet", "--example", "titor_cli", "--", "--path", path, "init"] )
        .status()
        .expect("Failed to run init");
    assert!(status.success(), "CLI init failed");

    // Create a checkpoint
    let output = Command::new("cargo")
        .args(&["run", "--quiet", "--example", "titor_cli", "--", "--path", path, "checkpoint", "-m", "Initial"])
        .output()
        .expect("Failed to run checkpoint");
    assert!(output.status.success(), "CLI checkpoint failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let id = stdout
        .lines()
        .find(|l| l.contains("✓ Created checkpoint"))
        .and_then(|l| l.split_whitespace().last())
        .expect("Failed to parse checkpoint ID");

    // Verify using prefix ID
    let output = Command::new("cargo")
        .args(&["run", "--quiet", "--example", "titor_cli", "--", "--path", path, "verify", id])
        .output()
        .expect("Failed to run verify");
    assert!(output.status.success(), "CLI verify failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Verification Report:"), "Unexpected verify output: {}", stdout);
}

#[test]
fn test_cli_restore_prefix() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap();

    // Initialize repository via CLI
    let status = Command::new("cargo")
        .args(&["run", "--quiet", "--example", "titor_cli", "--", "--path", path, "init"] )
        .status()
        .expect("Failed to run init");
    assert!(status.success(), "CLI init failed");

    // Create a test file
    let file_path = tmp.path().join("test.txt");
    fs::write(&file_path, "hello").unwrap();

    // Create a checkpoint
    let output = Command::new("cargo")
        .args(&["run", "--quiet", "--example", "titor_cli", "--", "--path", path, "checkpoint", "-m", "Add test file"])
        .output()
        .expect("Failed to run checkpoint");
    assert!(output.status.success(), "CLI checkpoint failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let id = stdout
        .lines()
        .find(|l| l.contains("✓ Created checkpoint"))
        .and_then(|l| l.split_whitespace().last())
        .expect("Failed to parse checkpoint ID");

    // Remove the file
    fs::remove_file(&file_path).unwrap();

    // Restore using prefix ID
    let status = Command::new("cargo")
        .args(&["run", "--quiet", "--example", "titor_cli", "--", "--path", path, "restore", id])
        .status()
        .expect("Failed to run restore");
    assert!(status.success(), "CLI restore failed");

    // Verify file was restored
    let restored = fs::read_to_string(&file_path).unwrap();
    assert_eq!(restored, "hello", "Restored file content mismatch");
} 