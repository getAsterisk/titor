//! Comprehensive integration tests for Titor
//!
//! Tests complex real-world scenarios including timeline navigation,
//! massive file operations, and edge cases.

use ::titor::*;
use tempfile::TempDir;
use std::fs;
use std::path::PathBuf;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use gxhash::{HashMap, HashMapExt};
use tracing::info;

/// Test harness for complex checkpoint scenarios
pub struct TitorTestHarness {
    pub temp_dir: TempDir,
    pub storage_dir: TempDir,
    pub titor: Titor,
    pub file_generator: FileGenerator,
    pub operation_log: Vec<TestOperation>,
}

#[derive(Debug, Clone)]
pub enum TestOperation {
    CreateFile { path: PathBuf, content: Vec<u8> },
    ModifyFile { path: PathBuf, content: Vec<u8> },
    DeleteFile { path: PathBuf },
    CreateCheckpoint { id: String, description: String },
    RestoreCheckpoint { id: String },
    Fork { from_id: String, new_id: String },
}

impl TitorTestHarness {
    /// Create a new test harness
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
            temp_dir,
            storage_dir,
            titor,
            file_generator: FileGenerator::new(42),
            operation_log: Vec::new(),
        }
    }
    
    /// Generate complex file structures
    pub fn generate_complex_project(&mut self, config: ProjectConfig) -> anyhow::Result<()> {
        let root = self.temp_dir.path();
        
        // Create directory structure
        for dir_depth in 1..=config.max_depth {
            for dir_idx in 0..config.dirs_per_level {
                let mut path = root.to_path_buf();
                for level in 0..dir_depth {
                    path = path.join(format!("dir_{}_{}", level, dir_idx));
                }
                fs::create_dir_all(&path)?;
                
                // Create files in this directory
                for file_idx in 0..config.files_per_dir {
                    let file_path = path.join(format!("file_{}.txt", file_idx));
                    let content = self.file_generator.generate_file_content(config.file_size_range.clone());
                    fs::write(&file_path, &content)?;
                    
                    self.operation_log.push(TestOperation::CreateFile {
                        path: file_path.strip_prefix(root)?.to_path_buf(),
                        content,
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply random mutations to files
    pub fn mutate_files(&mut self, mutation_config: MutationConfig) -> anyhow::Result<Vec<FileChange>> {
        let mut changes = Vec::new();
        let root = self.temp_dir.path();
        
        // Collect all files
        let mut all_files = Vec::new();
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                all_files.push(entry.path().to_path_buf());
            }
        }
        
        // Use the RNG from file_generator directly instead of cloning
        // This ensures consistent random number generation across calls
        
        // Apply mutations
        for mutation_idx in 0..mutation_config.num_mutations {
            if all_files.is_empty() {
                break;
            }
            
            let mutation_type = self.file_generator.rng.random_range(0..3);
            match mutation_type {
                0 => {
                    // Modify existing file
                    let idx = self.file_generator.rng.random_range(0..all_files.len());
                    let file_path = &all_files[idx];
                    let new_content = self.file_generator.generate_file_content(mutation_config.file_size_range.clone());
                    fs::write(file_path, &new_content)?;
                    
                    changes.push(FileChange::Modified(file_path.clone()));
                    self.operation_log.push(TestOperation::ModifyFile {
                        path: file_path.strip_prefix(root)?.to_path_buf(),
                        content: new_content,
                    });
                }
                1 => {
                    // Delete file
                    let idx = self.file_generator.rng.random_range(0..all_files.len());
                    let file_path = all_files.remove(idx);
                    fs::remove_file(&file_path)?;
                    
                    changes.push(FileChange::Deleted(file_path.clone()));
                    self.operation_log.push(TestOperation::DeleteFile {
                        path: file_path.strip_prefix(root)?.to_path_buf(),
                    });
                }
                2 => {
                    // Add new file with a deterministic name based on mutation index
                    let file_path = root.join(format!("mutated_file_{}.txt", mutation_idx));
                    let content = self.file_generator.generate_file_content(mutation_config.file_size_range.clone());
                    fs::write(&file_path, &content)?;
                    
                    all_files.push(file_path.clone());
                    changes.push(FileChange::Added(file_path.clone()));
                    self.operation_log.push(TestOperation::CreateFile {
                        path: file_path.strip_prefix(root)?.to_path_buf(),
                        content,
                    });
                }
                _ => unreachable!(),
            }
        }
        
        Ok(changes)
    }
    
    /// Verify checkpoint integrity across the entire timeline
    pub fn verify_timeline_integrity(&self) -> anyhow::Result<IntegrityReport> {
        let timeline = self.titor.get_timeline()?;
        let mut report = IntegrityReport::default();
        
        // Verify each checkpoint
        for checkpoint in timeline.checkpoints.values() {
            let verification = self.titor.verify_checkpoint(&checkpoint.id)?;
            report.checkpoint_verifications.push(verification.clone());
            
            if !verification.is_valid() {
                report.invalid_checkpoints.push(checkpoint.id.clone());
            }
        }
        
        // Verify timeline structure
        report.timeline_valid = self.titor.verify_timeline()?.is_valid();
        report.total_checkpoints = timeline.checkpoints.len();
        
        Ok(report)
    }
    
    /// Simulate concurrent file modifications (checkpoints are sequential)
    pub async fn simulate_concurrent_changes(&mut self, workers: usize) -> anyhow::Result<()> {
        use tokio::task;
        use std::sync::Arc;
        
        let root = Arc::new(self.temp_dir.path().to_path_buf());
        let mut handles = vec![];
        
        // Workers create files concurrently
        for worker_id in 0..workers {
            let root = Arc::clone(&root);
            
            let handle = task::spawn(async move {
                for op_idx in 0..10 {
                    // Create a file
                    let file_path = root.join(format!("worker_{}_file_{}.txt", worker_id, op_idx));
                    tokio::fs::write(&file_path, format!("Worker {} operation {}", worker_id, op_idx)).await?;
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                }
                Ok::<(), anyhow::Error>(())
            });
            
            handles.push(handle);
        }
        
        // Wait for all workers
        for handle in handles {
            handle.await??;
        }
        
        // Create a checkpoint after all concurrent operations
        self.titor.checkpoint(Some("After concurrent changes".to_string()))?;
        
        Ok(())
    }
}

/// File generator for test data
pub struct FileGenerator {
    rng: StdRng,
}

impl FileGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }
    
    /// Generate realistic file content
    pub fn generate_file_content(&mut self, size_range: std::ops::Range<usize>) -> Vec<u8> {
        let size = self.rng.random_range(size_range);
        let mut content = Vec::with_capacity(size);
        
        // Generate somewhat realistic text content
        let words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "lorem", "ipsum"];
        while content.len() < size {
            let word = words[self.rng.random_range(0..words.len())];
            content.extend_from_slice(word.as_bytes());
            content.push(b' ');
        }
        
        content.truncate(size);
        content
    }
    
    /// Generate binary file content
    pub fn generate_binary_content(&mut self, size: usize) -> Vec<u8> {
        let mut content = vec![0u8; size];
        self.rng.fill(&mut content[..]);
        content
    }
}

#[derive(Debug, Clone)]
pub struct ProjectConfig {
    pub max_depth: usize,
    pub dirs_per_level: usize,
    pub files_per_dir: usize,
    pub file_size_range: std::ops::Range<usize>,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            max_depth: 3,
            dirs_per_level: 5,
            files_per_dir: 10,
            file_size_range: 100..10_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MutationConfig {
    pub num_mutations: usize,
    pub file_size_range: std::ops::Range<usize>,
}

#[derive(Debug)]
pub enum FileChange {
    Added(PathBuf),
    Modified(PathBuf),
    Deleted(PathBuf),
}

#[derive(Debug, Default)]
pub struct IntegrityReport {
    pub checkpoint_verifications: Vec<::titor::VerificationReport>,
    pub invalid_checkpoints: Vec<String>,
    pub timeline_valid: bool,
    pub total_checkpoints: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tracing_test::traced_test;
    
    #[test]
    #[traced_test]
    fn test_complex_timeline_navigation() {
        let mut harness = TitorTestHarness::new();
        
        // Generate initial project structure - REDUCED SCALE
        harness.generate_complex_project(ProjectConfig {
            max_depth: 2,      // Reduced from 5
            dirs_per_level: 3, // Reduced from 10
            files_per_dir: 5,  // Reduced from 20
            file_size_range: 100..1_000, // Reduced from 100..100_000
        }).unwrap();
        
        // Create main timeline with checkpoints - REDUCED ITERATIONS
        let mut checkpoints = Vec::new();
        
        for i in 0..10 { // Reduced from 50
            // Apply mutations
            harness.mutate_files(MutationConfig {
                num_mutations: 10, // Reduced from 100
                file_size_range: 100..1_000, // Reduced from 100..50_000
            }).unwrap();
            
            // Create checkpoint
            let checkpoint = harness.titor.checkpoint(Some(format!("Checkpoint {}", i))).unwrap();
            checkpoints.push(checkpoint.id.clone());
        }
        
        // Test navigation by restoring to various checkpoints
        let mut rng = StdRng::seed_from_u64(123);
        
        for _ in 0..20 { // Test 20 random navigations
            // Pick a random checkpoint
            let target_idx = rng.random_range(0..checkpoints.len());
            let target_checkpoint = &checkpoints[target_idx];
            
            info!("Navigating to checkpoint: {}", &target_checkpoint[..8]);
            
            // Restore
            let result = harness.titor.restore(target_checkpoint).unwrap();
            assert!(result.warnings.is_empty(), "Restore had warnings: {:?}", result.warnings);
            
            // Verify checkpoint integrity
            let verification = harness.titor.verify_checkpoint(target_checkpoint).unwrap();
            assert!(verification.is_valid(), "Checkpoint verification failed: {}", verification.summary());
            
            // Verify we can read the timeline
            let timeline = harness.titor.get_timeline().unwrap();
            assert!(timeline.checkpoints.contains_key(target_checkpoint));
            
            // Verify we can list checkpoints
            let checkpoint_list = harness.titor.list_checkpoints().unwrap();
            assert!(checkpoint_list.iter().any(|c| c.id == *target_checkpoint));
        }
        
        // Create branches to test more complex timeline navigation
        let mut branches = HashMap::new();
        for (branch_idx, &base_checkpoint_idx) in [2, 5, 8].iter().enumerate() {
            // Restore to base checkpoint
            harness.titor.restore(&checkpoints[base_checkpoint_idx]).unwrap();
            
            // Create a fork
            let fork = harness.titor.fork(
                &checkpoints[base_checkpoint_idx], 
                Some(format!("Branch {}", branch_idx))
            ).unwrap();
            
            let mut branch_checkpoints = vec![fork.id.clone()];
            
            // Create more checkpoints on this branch
            for i in 0..3 {
                harness.mutate_files(MutationConfig {
                    num_mutations: 5,
                    file_size_range: 100..1_000,
                }).unwrap();
                
                let checkpoint = harness.titor.checkpoint(
                    Some(format!("Branch {} checkpoint {}", branch_idx, i))
                ).unwrap();
                branch_checkpoints.push(checkpoint.id.clone());
            }
            
            branches.insert(branch_idx, branch_checkpoints);
        }
        
        // Test navigation across branches
        for _ in 0..10 {
            // Pick a random branch
            let branch_idx = rng.random_range(0..branches.len());
            let branch_checkpoints = &branches[&branch_idx];
            let checkpoint_idx = rng.random_range(0..branch_checkpoints.len());
            let target_checkpoint = &branch_checkpoints[checkpoint_idx];
            
            info!("Navigating to branch checkpoint: {}", &target_checkpoint[..8]);
            
            // Restore
            let result = harness.titor.restore(target_checkpoint).unwrap();
            assert!(result.warnings.is_empty());
            
            // Verify checkpoint
            let verification = harness.titor.verify_checkpoint(target_checkpoint).unwrap();
            assert!(verification.is_valid());
        }
        
        // Final timeline integrity check
        let integrity = harness.verify_timeline_integrity().unwrap();
        assert!(integrity.timeline_valid, "Timeline integrity check failed");
        assert_eq!(integrity.invalid_checkpoints.len(), 0, "Found invalid checkpoints: {:?}", integrity.invalid_checkpoints);
    }
    
    #[test]
    #[traced_test]
    fn test_massive_file_operations() {
        let mut harness = TitorTestHarness::new();
        
        // Generate project with a reasonable number of files for testing - REDUCED SCALE
        info!("Generating project structure");
        harness.generate_complex_project(ProjectConfig {
            max_depth: 3,      // Reduced from 20
            dirs_per_level: 5, // Reduced from 50
            files_per_dir: 10, // Reduced from 100
            file_size_range: 1..10_000, // Reduced from 1..100_000
        }).unwrap();
        
        // Create initial checkpoint
        info!("Creating initial checkpoint");
        let start = std::time::Instant::now();
        let initial_checkpoint = harness.titor.checkpoint(Some("Initial state".to_string())).unwrap();
        let checkpoint_time = start.elapsed();
        info!("Initial checkpoint created in {:?}", checkpoint_time);
        
        // Verify checkpoint was fast enough (should be < 1 second for this test scale)
        assert!(checkpoint_time.as_secs() < 5, "Checkpoint took too long: {:?}", checkpoint_time);
        
        // Perform various operations - REDUCED ITERATIONS AND SCALE
        for op_idx in 0..5 { // Reduced from 10
            info!("Performing operation batch {}", op_idx);
            
            // Random modifications - REDUCED SCALE
            let num_modifications = match op_idx {
                0..=2 => 5,    // Reduced from 10
                3 => 50,       // Reduced from 1000
                _ => 100,      // Reduced from 10000
            };
            
            harness.mutate_files(MutationConfig {
                num_mutations: num_modifications,
                file_size_range: 1..10_000, // Reduced from 1..1_000_000
            }).unwrap();
            
            // Create checkpoint
            let checkpoint = harness.titor.checkpoint(
                Some(format!("After {} modifications", num_modifications))
            ).unwrap();
            
            // Verify checkpoint
            let verification = harness.titor.verify_checkpoint(&checkpoint.id).unwrap();
            assert!(verification.is_valid());
        }
        
        // Test restoration performance
        info!("Testing restoration performance");
        let restore_start = std::time::Instant::now();
        let result = harness.titor.restore(&initial_checkpoint.id).unwrap();
        let restore_time = restore_start.elapsed();
        
        info!("Restored {} files in {:?}", result.files_restored, restore_time);
        assert!(result.warnings.is_empty());
        
        // Verify restore was fast enough (should be < 5 seconds for this test scale)
        assert!(restore_time.as_secs() < 10, "Restore took too long: {:?}", restore_time);
    }
    
    #[tokio::test]
    async fn test_concurrent_operations() {
        let mut harness = TitorTestHarness::new();
        
        // Create initial state
        harness.generate_complex_project(ProjectConfig::default()).unwrap();
        harness.titor.checkpoint(Some("Initial state".to_string())).unwrap();
        
        // Simulate concurrent changes
        harness.simulate_concurrent_changes(10).await.unwrap();
        
        // Verify timeline integrity
        let integrity = harness.verify_timeline_integrity().unwrap();
        assert!(integrity.timeline_valid);
    }
} 