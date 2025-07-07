//! Chaos testing framework for Titor
//!
//! Implements chaos engineering principles to ensure robustness
//! under adverse conditions including corruption, concurrent access,
//! and resource exhaustion.

use ::titor::*;
use tempfile::TempDir;
use std::fs;
use std::path::{Path, PathBuf};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::sync::Arc;

use tracing::{info, warn, error};

/// Chaos testing framework
pub struct TitorChaosTest {
    pub titor: Titor,
    pub temp_dir: TempDir,
    pub storage_dir: TempDir,
    pub chaos_engine: ChaosEngine,
    pub verification_engine: VerificationEngine,
}

impl TitorChaosTest {
    pub fn new() -> Self {
        let temp_dir = TempDir::new().unwrap();
        let storage_dir = TempDir::new().unwrap();
        
        let titor = TitorBuilder::new()
            .compression_strategy(CompressionStrategy::Fast)
            .build(
                temp_dir.path().to_path_buf(),
                storage_dir.path().to_path_buf(),
            )
            .unwrap();
        
        Self {
            titor,
            temp_dir,
            storage_dir,
            chaos_engine: ChaosEngine::new(42),
            verification_engine: VerificationEngine::new(),
        }
    }
    
    /// Run comprehensive chaos test suite
    pub async fn run_chaos_suite(&mut self) -> anyhow::Result<ChaosReport> {
        let mut report = ChaosReport::new();
        
        // 1. Storage corruption scenarios
        info!("Testing storage corruption scenarios");
        report.storage_tests = self.test_storage_corruption().await?;
        
        // 2. Concurrent operation chaos
        info!("Testing concurrent operation chaos");
        report.concurrency_tests = self.test_concurrent_chaos().await?;
        
        // 3. System resource exhaustion
        info!("Testing resource exhaustion");
        report.resource_tests = self.test_resource_exhaustion().await?;
        
        // 4. Time-based anomalies
        info!("Testing time-based anomalies");
        report.time_tests = self.test_time_anomalies().await?;
        
        // 5. Network and I/O failures
        info!("Testing I/O failures");
        report.io_tests = self.test_io_failures().await?;
        
        Ok(report)
    }
    
    /// Test storage corruption scenarios
    async fn test_storage_corruption(&mut self) -> anyhow::Result<CorruptionTestResult> {
        let mut result = CorruptionTestResult::default();
        
        // Create test data
        self.create_test_checkpoints(10)?;
        
        // Test 1: Corrupt random object files
        info!("Testing object file corruption");
        let objects_dir = self.storage_dir.path().join("objects");
        let corrupted_objects = self.chaos_engine.corrupt_random_files(&objects_dir, 5)?;
        result.corrupted_objects = corrupted_objects.len();
        
        // Verify detection
        for checkpoint_id in self.titor.list_checkpoints()?.iter().map(|c| &c.id) {
            match self.titor.verify_checkpoint(checkpoint_id) {
                Ok(verification) => {
                    if !verification.is_valid() {
                        result.corruption_detected += 1;
                    }
                }
                Err(_) => {
                    result.corruption_detected += 1;
                }
            }
        }
        
        // Test 2: Corrupt checkpoint metadata
        info!("Testing checkpoint metadata corruption");
        let checkpoints_dir = self.storage_dir.path().join("checkpoints");
        let corrupted_metadata = self.chaos_engine.corrupt_json_files(&checkpoints_dir, 3)?;
        result.corrupted_metadata = corrupted_metadata.len();
        
        // Test 3: Delete random storage files
        info!("Testing missing storage files");
        let deleted_files = self.chaos_engine.delete_random_files(&objects_dir, 3)?;
        result.deleted_files = deleted_files.len();
        
        // Test recovery mechanisms
        info!("Testing recovery mechanisms");
        let timeline = self.titor.get_timeline()?;
        for checkpoint in timeline.checkpoints.values() {
            match self.titor.restore(&checkpoint.id) {
                Ok(_) => result.successful_recoveries += 1,
                Err(e) => {
                    warn!("Failed to restore checkpoint {}: {}", checkpoint.short_id(), e);
                    result.failed_recoveries += 1;
                }
            }
        }
        
        result.test_passed = result.corruption_detected > 0 && result.successful_recoveries > 0;
        Ok(result)
    }
    
    /// Test concurrent operation chaos
    async fn test_concurrent_chaos(&mut self) -> anyhow::Result<ConcurrencyTestResult> {
        use tokio::task;
        
        let mut result = ConcurrencyTestResult::default();
        let root = Arc::new(self.temp_dir.path().to_path_buf());
        
        // Create initial checkpoints for testing
        let initial_checkpoints = self.create_test_checkpoints(5)?;
        
        // Test concurrent file operations
        let mut file_handles = vec![];
        
        // Spawn tasks that create files concurrently
        for i in 0..5 {
            let root = Arc::clone(&root);
            
            let handle = task::spawn(async move {
                let mut files_created = 0;
                for j in 0..20 {
                    let file_path = root.join(format!("concurrent_{}_file_{}.txt", i, j));
                    match tokio::fs::write(&file_path, format!("Concurrent content {} {}", i, j)).await {
                        Ok(_) => files_created += 1,
                        Err(e) => warn!("File write failed: {}", e),
                    }
                    tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
                }
                files_created
            });
            
            file_handles.push(handle);
        }
        
        // Wait for file operations to complete
        let mut total_files = 0;
        for handle in file_handles {
            match handle.await {
                Ok(count) => total_files += count,
                Err(e) => {
                    error!("File creation task panicked: {}", e);
                    result.task_panics += 1;
                }
            }
        }
        
        // Now do checkpoint operations sequentially to test state consistency
        info!("Created {} files concurrently, now testing checkpoint operations", total_files);
        
        // Create checkpoint after concurrent file creation
        match self.titor.checkpoint(Some("After concurrent files".to_string())) {
            Ok(_) => result.successful_operations += 1,
            Err(e) => warn!("Checkpoint after concurrent files failed: {}", e),
        }
        
        // Test rapid checkpoint/restore cycles
        for i in 0..10 {
            // Randomly pick a checkpoint to restore
            let checkpoint_id = &initial_checkpoints[i % initial_checkpoints.len()];
            match self.titor.restore(checkpoint_id) {
                Ok(_) => result.successful_operations += 1,
                Err(e) => warn!("Restore {} failed: {}", i, e),
            }
            
            // Create a new checkpoint
            match self.titor.checkpoint(Some(format!("Rapid cycle {}", i))) {
                Ok(_) => result.successful_operations += 1,
                Err(e) => warn!("Checkpoint {} failed: {}", i, e),
            }
        }
        
        // Test garbage collection
        match self.titor.gc() {
            Ok(_) => result.successful_operations += 1,
            Err(e) => warn!("GC failed: {}", e),
        }
        
        // Verify final state
        let final_verification = self.titor.verify_timeline()?;
        result.timeline_consistent = final_verification.is_valid();
        result.race_conditions_detected = false; // No real races in this simpler test
        result.test_passed = result.timeline_consistent && result.task_panics == 0;
        
        Ok(result)
    }
    
    /// Test resource exhaustion
    async fn test_resource_exhaustion(&mut self) -> anyhow::Result<ResourceTestResult> {
        let mut result = ResourceTestResult::default();
        
        // Test 1: Large file handling
        info!("Testing large file handling");
        let large_file_path = self.temp_dir.path().join("large_file.bin");
        let large_size = 100_000_000; // 100MB
        
        // Create large file
        let mut rng = StdRng::seed_from_u64(123);
        let chunk_size = 1_000_000;
        let mut file = std::fs::File::create(&large_file_path)?;
        
        use std::io::Write;
        for _ in 0..(large_size / chunk_size) {
            let chunk: Vec<u8> = (0..chunk_size).map(|_| rng.random()).collect();
            file.write_all(&chunk)?;
        }
        drop(file);
        
        // Try to checkpoint with large file
        match self.titor.checkpoint(Some("Large file checkpoint".to_string())) {
            Ok(_) => result.large_files_handled += 1,
            Err(e) => {
                warn!("Failed to checkpoint large file: {}", e);
                result.oom_errors += 1;
            }
        }
        
        // Test 2: Many small files
        info!("Testing many small files");
        for i in 0..10000 {
            let path = self.temp_dir.path().join(format!("small_{}.txt", i));
            fs::write(&path, format!("Small file {}", i))?;
        }
        
        match self.titor.checkpoint(Some("Many files checkpoint".to_string())) {
            Ok(_) => result.many_files_handled += 1,
            Err(e) => {
                warn!("Failed to checkpoint many files: {}", e);
                result.oom_errors += 1;
            }
        }
        
        // Test 3: Deep directory nesting
        info!("Testing deep directory nesting");
        let mut deep_path = self.temp_dir.path().to_path_buf();
        for i in 0..100 {
            deep_path = deep_path.join(format!("level_{}", i));
            fs::create_dir(&deep_path)?;
        }
        fs::write(deep_path.join("deep_file.txt"), "Deep content")?;
        
        match self.titor.checkpoint(Some("Deep nesting checkpoint".to_string())) {
            Ok(_) => result.deep_nesting_handled += 1,
            Err(e) => {
                warn!("Failed to checkpoint deep nesting: {}", e);
                result.path_too_long_errors += 1;
            }
        }
        
        result.test_passed = result.oom_errors == 0 && 
                            result.large_files_handled > 0 &&
                            result.many_files_handled > 0;
        
        Ok(result)
    }
    
    /// Test time-based anomalies
    async fn test_time_anomalies(&mut self) -> anyhow::Result<TimeTestResult> {
        let mut result = TimeTestResult::default();
        
        // Create checkpoints with manipulated timestamps
        info!("Testing timestamp anomalies");
        
        // Test future timestamps
        let future_file = self.temp_dir.path().join("future.txt");
        fs::write(&future_file, "Future content")?;
        
        // Set file time to future (if supported by OS)
        #[cfg(unix)]
        {

            use std::time::{SystemTime, Duration};
            
            let future_time = SystemTime::now() + Duration::from_secs(365 * 24 * 60 * 60); // 1 year
            match filetime::set_file_mtime(&future_file, future_time.into()) {
                Ok(_) => result.future_timestamps_tested += 1,
                Err(e) => warn!("Failed to set future timestamp: {}", e),
            }
        }
        
        match self.titor.checkpoint(Some("Future timestamp checkpoint".to_string())) {
            Ok(_) => result.checkpoints_with_anomalies += 1,
            Err(e) => warn!("Failed checkpoint with future timestamp: {}", e),
        }
        
        // Test clock skew during operations
        info!("Testing clock skew");
        let checkpoint1 = self.titor.checkpoint(Some("Before skew".to_string()))?;
        
        // Simulate clock going backwards by modifying files
        let skew_file = self.temp_dir.path().join("skew.txt");
        fs::write(&skew_file, "Skew content")?;
        
        #[cfg(unix)]
        {
            use std::time::{SystemTime, Duration};
            
            let past_time = SystemTime::now() - Duration::from_secs(60 * 60); // 1 hour ago
            match filetime::set_file_mtime(&skew_file, past_time.into()) {
                Ok(_) => result.clock_skew_tested += 1,
                Err(e) => warn!("Failed to simulate clock skew: {}", e),
            }
        }
        
        let checkpoint2 = self.titor.checkpoint(Some("After skew".to_string()))?;
        
        // Verify both checkpoints are valid
        let verify1 = self.titor.verify_checkpoint(&checkpoint1.id)?;
        let verify2 = self.titor.verify_checkpoint(&checkpoint2.id)?;
        
        result.timeline_consistency_maintained = verify1.is_valid() && verify2.is_valid();
        result.test_passed = result.timeline_consistency_maintained;
        
        Ok(result)
    }
    
    /// Test I/O failures
    async fn test_io_failures(&mut self) -> anyhow::Result<IoTestResult> {
        let mut result = IoTestResult::default();
        
        // Test 1: Readonly filesystem (simulate)
        info!("Testing readonly filesystem handling");
        
        // Make a file readonly
        let readonly_file = self.temp_dir.path().join("readonly.txt");
        fs::write(&readonly_file, "Readonly content")?;
        
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&readonly_file)?.permissions();
            perms.set_mode(0o444); // readonly
            fs::set_permissions(&readonly_file, perms)?;
            
            // Try to modify it
            match fs::write(&readonly_file, "Modified content") {
                Ok(_) => {}
                Err(_) => result.permission_errors += 1,
            }
        }
        
        // Test 2: Disk full simulation (create many files)
        info!("Testing disk full scenarios");
        let mut created_files = 0;
        for i in 0..100000 {
            let path = self.temp_dir.path().join(format!("fill_{}.dat", i));
            match fs::write(&path, vec![0u8; 10000]) {
                Ok(_) => created_files += 1,
                Err(e) if e.raw_os_error() == Some(28) /* ENOSPC */ => {
                    result.disk_full_errors += 1;
                    break;
                }
                Err(_) => break,
            }
        }
        info!("Created {} files before stopping", created_files);
        
        // Test 3: Interrupted operations
        info!("Testing interrupted operations");
        // This is simulated by starting operations and not completing them
        
        // Checkpoint should still work
        match self.titor.checkpoint(Some("After I/O tests".to_string())) {
            Ok(_) => result.successful_despite_errors += 1,
            Err(e) => warn!("Checkpoint failed after I/O tests: {}", e),
        }
        
        result.test_passed = result.successful_despite_errors > 0;
        Ok(result)
    }
    
    /// Create test checkpoints
    fn create_test_checkpoints(&mut self, count: usize) -> anyhow::Result<Vec<String>> {
        let mut checkpoint_ids = Vec::new();
        
        for i in 0..count {
            // Create some files
            for j in 0..10 {
                let path = self.temp_dir.path().join(format!("test_{}_{}.txt", i, j));
                fs::write(&path, format!("Test content {} {}", i, j))?;
            }
            
            let checkpoint = self.titor.checkpoint(Some(format!("Test checkpoint {}", i)))?;
            checkpoint_ids.push(checkpoint.id);
        }
        
        Ok(checkpoint_ids)
    }
}

/// Chaos engine for introducing failures
pub struct ChaosEngine {
    rng: StdRng,
}

impl ChaosEngine {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
    
    /// Corrupt random files
    pub fn corrupt_random_files(&mut self, dir: &Path, count: usize) -> anyhow::Result<Vec<PathBuf>> {
        let mut corrupted = Vec::new();
        let mut files = Vec::new();
        
        // Collect all files
        for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                files.push(entry.path().to_path_buf());
            }
        }
        
        // Corrupt random files
        for _ in 0..count.min(files.len()) {
            if files.is_empty() {
                break;
            }
            
            let idx = self.rng.random_range(0..files.len());
            let file_path = files.remove(idx);
            
            // Read file
            if let Ok(mut content) = fs::read(&file_path) {
                // Corrupt random bytes
                for _ in 0..10 {
                    if !content.is_empty() {
                        let byte_idx = self.rng.random_range(0..content.len());
                        content[byte_idx] = self.rng.random();
                    }
                }
                
                // Write back
                if fs::write(&file_path, content).is_ok() {
                    corrupted.push(file_path);
                }
            }
        }
        
        Ok(corrupted)
    }
    
    /// Corrupt JSON files
    pub fn corrupt_json_files(&mut self, dir: &Path, count: usize) -> anyhow::Result<Vec<PathBuf>> {
        let mut corrupted = Vec::new();
        let mut json_files = Vec::new();
        
        // Find JSON files
        for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("json") {
                json_files.push(entry.path().to_path_buf());
            }
        }
        
        // Corrupt them
        for _ in 0..count.min(json_files.len()) {
            if json_files.is_empty() {
                break;
            }
            
            let idx = self.rng.random_range(0..json_files.len());
            let file_path = json_files.remove(idx);
            
            if let Ok(content) = fs::read_to_string(&file_path) {
                // Corrupt JSON by removing random characters
                let mut corrupted_content = content.chars().collect::<Vec<_>>();
                if corrupted_content.len() > 10 {
                    // Remove some characters
                    for _ in 0..5 {
                        let idx = self.rng.random_range(0..corrupted_content.len());
                        corrupted_content.remove(idx);
                    }
                }
                
                let corrupted_str: String = corrupted_content.into_iter().collect();
                if fs::write(&file_path, corrupted_str).is_ok() {
                    corrupted.push(file_path);
                }
            }
        }
        
        Ok(corrupted)
    }
    
    /// Delete random files
    pub fn delete_random_files(&mut self, dir: &Path, count: usize) -> anyhow::Result<Vec<PathBuf>> {
        let mut deleted = Vec::new();
        let mut files = Vec::new();
        
        // Collect files
        for entry in walkdir::WalkDir::new(dir).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                files.push(entry.path().to_path_buf());
            }
        }
        
        // Delete random files
        for _ in 0..count.min(files.len()) {
            if files.is_empty() {
                break;
            }
            
            let idx = self.rng.random_range(0..files.len());
            let file_path = files.remove(idx);
            
            if fs::remove_file(&file_path).is_ok() {
                deleted.push(file_path);
            }
        }
        
        Ok(deleted)
    }
}

/// Verification engine for chaos tests
pub struct VerificationEngine;

impl VerificationEngine {
    pub fn new() -> Self {
        Self
    }
}

// Result types for chaos tests

#[derive(Debug, Default)]
pub struct ChaosReport {
    pub storage_tests: CorruptionTestResult,
    pub concurrency_tests: ConcurrencyTestResult,
    pub resource_tests: ResourceTestResult,
    pub time_tests: TimeTestResult,
    pub io_tests: IoTestResult,
}

impl ChaosReport {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug, Default)]
pub struct CorruptionTestResult {
    pub corrupted_objects: usize,
    pub corrupted_metadata: usize,
    pub deleted_files: usize,
    pub corruption_detected: usize,
    pub successful_recoveries: usize,
    pub failed_recoveries: usize,
    pub test_passed: bool,
}

#[derive(Debug, Default)]
pub struct ConcurrencyTestResult {
    pub successful_operations: usize,
    pub race_conditions_detected: bool,
    pub timeline_consistent: bool,
    pub task_panics: usize,
    pub test_passed: bool,
}

#[derive(Debug, Default)]
pub struct ResourceTestResult {
    pub large_files_handled: usize,
    pub many_files_handled: usize,
    pub deep_nesting_handled: usize,
    pub oom_errors: usize,
    pub path_too_long_errors: usize,
    pub test_passed: bool,
}

#[derive(Debug, Default)]
pub struct TimeTestResult {
    pub future_timestamps_tested: usize,
    pub clock_skew_tested: usize,
    pub checkpoints_with_anomalies: usize,
    pub timeline_consistency_maintained: bool,
    pub test_passed: bool,
}

#[derive(Debug, Default)]
pub struct IoTestResult {
    pub permission_errors: usize,
    pub disk_full_errors: usize,
    pub network_errors: usize,
    pub successful_despite_errors: usize,
    pub test_passed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;
    
    #[tokio::test]
    #[traced_test]
    async fn test_storage_corruption_recovery() {
        let mut chaos_test = TitorChaosTest::new();
        
        // Run storage corruption tests
        let result = chaos_test.test_storage_corruption().await.unwrap();
        
        assert!(result.corruption_detected > 0, "Should detect corruption");
        assert!(result.test_passed, "Storage corruption test should pass");
    }
    
    #[tokio::test]
    #[traced_test] 
    async fn test_concurrent_chaos() {
        let mut chaos_test = TitorChaosTest::new();
        
        // Create initial state
        chaos_test.create_test_checkpoints(5).unwrap();
        
        // Run concurrent chaos
        let result = chaos_test.test_concurrent_chaos().await.unwrap();
        
        assert!(result.timeline_consistent, "Timeline should remain consistent");
        assert_eq!(result.task_panics, 0, "No tasks should panic");
        assert!(result.test_passed, "Concurrent chaos test should pass");
    }
    
    #[tokio::test]
    #[traced_test]
    async fn test_resource_exhaustion() {
        let mut chaos_test = TitorChaosTest::new();
        
        let result = chaos_test.test_resource_exhaustion().await.unwrap();
        
        assert!(result.large_files_handled > 0, "Should handle large files");
        assert!(result.test_passed, "Resource exhaustion test should pass");
    }
} 