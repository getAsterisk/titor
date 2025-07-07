//! Performance benchmarks for Titor
//!
//! Tracks performance metrics including checkpoint creation time,
//! restoration time, memory usage, and I/O patterns.

#![cfg_attr(feature = "quick-bench", allow(dead_code))]

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use titor::compression::CompressionStrategy;
use titor::TitorBuilder;
use titor::types::{FileManifest, FileEntry};
use chrono::Utc;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Benchmark checkpoint creation with varying file counts
fn bench_checkpoint_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint_creation");
    // Reduce measurement time and adjust sample size based on operation complexity
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);
    
    // Use more reasonable file counts
    for file_count in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(file_count),
            file_count,
            |b, &file_count| {
                // Pre-create test environment once
                let temp_dir = TempDir::new().unwrap();
                let storage_dir = TempDir::new().unwrap();
                
                // Create files once before benchmarking
                let mut rng = StdRng::seed_from_u64(42);
                for i in 0..file_count {
                    let path = temp_dir.path().join(format!("file_{}.txt", i));
                    // Use smaller files for faster I/O
                    let size = rng.random_range(100..1000);
                    let content: Vec<u8> = (0..size).map(|_| rng.random()).collect();
                    fs::write(path, content).unwrap();
                }
                
                // Benchmark
                b.iter(|| {
                    let mut titor = TitorBuilder::new()
                        .compression_strategy(CompressionStrategy::Fast)
                        .build(
                            temp_dir.path().to_path_buf(),
                            storage_dir.path().to_path_buf(),
                        )
                        .unwrap();
                    
                    let checkpoint = titor.checkpoint(Some("Benchmark".to_string())).unwrap();
                    black_box(checkpoint);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark restoration with varying checkpoint sizes
fn bench_restoration(c: &mut Criterion) {
    let mut group = c.benchmark_group("restoration");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    
    for file_count in [10, 50, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(file_count),
            file_count,
            |b, &file_count| {
                // Setup - do this once outside the benchmark loop
                let temp_dir = TempDir::new().unwrap();
                let storage_dir = TempDir::new().unwrap();
                
                let mut titor = TitorBuilder::new()
                    .compression_strategy(CompressionStrategy::Fast)
                    .build(
                        temp_dir.path().to_path_buf(),
                        storage_dir.path().to_path_buf(),
                    )
                    .unwrap();
                
                // Create files and checkpoint once
                let mut rng = StdRng::seed_from_u64(42);
                for i in 0..file_count {
                    let path = temp_dir.path().join(format!("file_{}.txt", i));
                    let size = rng.random_range(100..1000);
                    let content: Vec<u8> = (0..size).map(|_| rng.random()).collect();
                    fs::write(path, content).unwrap();
                }
                
                let checkpoint = titor.checkpoint(Some("Benchmark".to_string())).unwrap();
                let checkpoint_id = checkpoint.id.clone();
                
                // Benchmark restoration only
                b.iter_batched(
                    || {
                        // Setup: Modify files before each iteration
                        for i in 0..file_count/2 {
                            let path = temp_dir.path().join(format!("file_{}.txt", i));
                            fs::write(path, b"modified").unwrap();
                        }
                    },
                    |_| {
                        // Benchmark: Restore operation
                        let result = titor.restore(&checkpoint_id).unwrap();
                        black_box(result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark compression strategies
fn bench_compression_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_strategies");
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);
    
    let strategies = vec![
        ("none", CompressionStrategy::None),
        ("fast", CompressionStrategy::Fast),
        ("adaptive", CompressionStrategy::Adaptive {
            min_size: 4096,
            skip_extensions: vec!["jpg".to_string(), "mp4".to_string()],
        }),
    ];
    
    for (name, strategy) in strategies {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &strategy,
            |b, strategy| {
                // Pre-create test data
                let temp_dir = TempDir::new().unwrap();
                let storage_dir = TempDir::new().unwrap();
                
                // Create mixed files once
                let mut rng = StdRng::seed_from_u64(42);
                
                // Reduced file count for faster benchmarks
                // Text files (compressible)
                for i in 0..20 {
                    let path = temp_dir.path().join(format!("text_{}.txt", i));
                    let content = "Hello World! ".repeat(rng.random_range(50..200));
                    fs::write(path, content).unwrap();
                }
                
                // Binary files (less compressible)
                for i in 0..20 {
                    let path = temp_dir.path().join(format!("binary_{}.bin", i));
                    let size = rng.random_range(500..2000);
                    let content: Vec<u8> = (0..size).map(|_| rng.random()).collect();
                    fs::write(path, content).unwrap();
                }
                
                // Benchmark
                b.iter(|| {
                    let mut titor = TitorBuilder::new()
                        .compression_strategy(strategy.clone())
                        .build(
                            temp_dir.path().to_path_buf(),
                            storage_dir.path().to_path_buf(),
                        )
                        .unwrap();
                    
                    let checkpoint = titor.checkpoint(Some("Compression benchmark".to_string())).unwrap();
                    black_box(checkpoint);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark incremental changes
fn bench_incremental_changes(c: &mut Criterion) {
    let mut group = c.benchmark_group("incremental_changes");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    
    for change_percentage in [1, 5, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}%", change_percentage)),
            change_percentage,
            |b, &change_percentage| {
                // Setup
                let temp_dir = TempDir::new().unwrap();
                let storage_dir = TempDir::new().unwrap();
                
                let mut titor = TitorBuilder::new()
                    .compression_strategy(CompressionStrategy::Fast)
                    .build(
                        temp_dir.path().to_path_buf(),
                        storage_dir.path().to_path_buf(),
                    )
                    .unwrap();
                
                // Create initial files - reduced count
                let file_count = 200;
                let mut rng = StdRng::seed_from_u64(42);
                
                for i in 0..file_count {
                    let path = temp_dir.path().join(format!("file_{}.txt", i));
                    let size = rng.random_range(100..1000);
                    let content: Vec<u8> = (0..size).map(|_| rng.random()).collect();
                    fs::write(path, content).unwrap();
                }
                
                // Initial checkpoint
                titor.checkpoint(Some("Initial".to_string())).unwrap();
                
                // Benchmark incremental changes
                b.iter_batched(
                    || {},
                    |_| {
                        // Change percentage of files
                        let files_to_change = (file_count * change_percentage) / 100;
                        for i in 0..files_to_change {
                            let path = temp_dir.path().join(format!("file_{}.txt", i));
                            let new_content = format!("Modified content {}", rng.random::<u32>());
                            fs::write(path, new_content).unwrap();
                        }
                        
                        let checkpoint = titor.checkpoint(Some("Incremental".to_string())).unwrap();
                        black_box(checkpoint);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }
    
    group.finish();
}

/// Benchmark merkle tree computation
fn bench_merkle_tree(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree");
    group.measurement_time(Duration::from_secs(2));
    
    // Reduced file counts for reasonable benchmark times
    for file_count in [50, 100, 500, 1000].iter() {
        // Adjust sample size based on file count
        let sample_size = if *file_count > 500 { 10 } else { 20 };
        group.sample_size(sample_size);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(file_count),
            file_count,
            |b, &file_count| {
                // Setup
                let temp_dir = TempDir::new().unwrap();
                let storage_dir = TempDir::new().unwrap();
                
                // Create files once
                for i in 0..file_count {
                    let path = temp_dir.path().join(format!("file_{}.txt", i));
                    let content = format!("File content {}", i);
                    fs::write(path, content).unwrap();
                }
                
                let titor = TitorBuilder::new()
                    .build(
                        temp_dir.path().to_path_buf(),
                        storage_dir.path().to_path_buf(),
                    )
                    .unwrap();
                
                // Benchmark merkle tree computation
                b.iter(|| {
                    let merkle_root = titor.compute_current_merkle_root().unwrap();
                    black_box(merkle_root);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark verification operations
fn bench_verification(c: &mut Criterion) {
    let mut group = c.benchmark_group("verification");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    
    // Setup test data
    let temp_dir = TempDir::new().unwrap();
    let storage_dir = TempDir::new().unwrap();
    
    let mut titor = TitorBuilder::new()
        .build(
            temp_dir.path().to_path_buf(),
            storage_dir.path().to_path_buf(),
        )
        .unwrap();
    
    // Create smaller checkpoints
    let mut checkpoint_ids = Vec::new();
    let mut rng = StdRng::seed_from_u64(42);
    
    for checkpoint_idx in 0..3 {
        // Create fewer files per checkpoint
        for i in 0..100 {
            let path = temp_dir.path().join(format!("checkpoint_{}_file_{}.txt", checkpoint_idx, i));
            let size = rng.random_range(100..1000);
            let content: Vec<u8> = (0..size).map(|_| rng.random()).collect();
            fs::write(path, content).unwrap();
        }
        
        let checkpoint = titor.checkpoint(Some(format!("Checkpoint {}", checkpoint_idx))).unwrap();
        checkpoint_ids.push(checkpoint.id);
    }
    
    // Benchmark checkpoint verification
    group.bench_function("checkpoint_verification", |b| {
        b.iter(|| {
            for checkpoint_id in &checkpoint_ids {
                let report = titor.verify_checkpoint(checkpoint_id).unwrap();
                black_box(report);
            }
        });
    });
    
    // Benchmark timeline verification
    group.bench_function("timeline_verification", |b| {
        b.iter(|| {
            let report = titor.verify_timeline().unwrap();
            black_box(report);
        });
    });
    
    group.finish();
}

/// Benchmark parallel operations
fn bench_parallel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_operations");
    group.measurement_time(Duration::from_secs(3));
    for worker_count in [1, 2, 4, 8].iter() {
        // Criterion requires at least 10 samples
        let sample_size = if *worker_count >= 8 { 10 } else { 10 };
        group.sample_size(sample_size);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_workers", worker_count)),
            worker_count,
            |b, &worker_count| {
                // Setup
                let temp_dir = TempDir::new().unwrap();
                let storage_dir = TempDir::new().unwrap();
                
                // Create fewer files for faster benchmarks
                let mut rng = StdRng::seed_from_u64(42);
                for i in 0..1000 {
                    let path = temp_dir.path().join(format!("file_{}.txt", i));
                    let size = rng.random_range(100..500);
                    let content: Vec<u8> = (0..size).map(|_| rng.random()).collect();
                    fs::write(path, content).unwrap();
                }
                
                // Benchmark with different worker counts
                b.iter(|| {
                    let mut titor = TitorBuilder::new()
                        .parallel_workers(worker_count)
                        .build(
                            temp_dir.path().to_path_buf(),
                            storage_dir.path().to_path_buf(),
                        )
                        .unwrap();
                    
                    let checkpoint = titor.checkpoint(Some("Parallel benchmark".to_string())).unwrap();
                    black_box(checkpoint);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory usage patterns
fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);
    
    // Test with different file sizes - reduced sizes
    for avg_file_size in [1_000, 10_000, 50_000, 100_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}KB_files", avg_file_size / 1000)),
            avg_file_size,
            |b, &avg_file_size| {
                // Setup
                let temp_dir = TempDir::new().unwrap();
                let storage_dir = TempDir::new().unwrap();
                
                // Create files of specified size
                let mut rng = StdRng::seed_from_u64(42);
                let file_count = 50; // Reduced from 100
                
                for i in 0..file_count {
                    let path = temp_dir.path().join(format!("file_{}.bin", i));
                    let size = rng.random_range(avg_file_size/2..avg_file_size*2);
                    let content: Vec<u8> = vec![0; size];
                    fs::write(path, content).unwrap();
                }
                
                // Benchmark memory usage during checkpoint
                b.iter(|| {
                    let mut titor = TitorBuilder::new()
                        .compression_strategy(CompressionStrategy::Fast)
                        .build(
                            temp_dir.path().to_path_buf(),
                            storage_dir.path().to_path_buf(),
                        )
                        .unwrap();
                    
                    let checkpoint = titor.checkpoint(Some("Memory benchmark".to_string())).unwrap();
                    black_box(checkpoint);
                });
            },
        );
    }
    
    group.finish();
}

fn manifest_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("manifest_serialization");
    
    // Create manifests of different sizes
    let sizes = vec![10, 100, 1000, 5000];
    
    for size in sizes {
        // Create a test manifest with the specified number of files
        let manifest = create_test_manifest(size);
        
        // Benchmark JSON serialization
        group.bench_with_input(
            BenchmarkId::new("json", size),
            &manifest,
            |b, manifest| {
                b.iter(|| {
                    let json = serde_json::to_string(manifest).unwrap();
                    black_box(json);
                });
            }
        );
        
        // Benchmark bincode serialization
        group.bench_with_input(
            BenchmarkId::new("bincode", size),
            &manifest,
            |b, manifest| {
                b.iter(|| {
                    let bytes = bincode::serde::encode_to_vec(manifest, bincode::config::standard()).unwrap();
                    black_box(bytes);
                });
            }
        );
        
        // Benchmark JSON deserialization
        let json = serde_json::to_string(&manifest).unwrap();
        group.bench_with_input(
            BenchmarkId::new("json_deserialize", size),
            &json,
            |b, json| {
                b.iter(|| {
                    let manifest: FileManifest = serde_json::from_str(json).unwrap();
                    black_box(manifest);
                });
            }
        );
        
        // Benchmark bincode deserialization
                    let bytes = bincode::serde::encode_to_vec(&manifest, bincode::config::standard()).unwrap();
        group.bench_with_input(
            BenchmarkId::new("bincode_deserialize", size),
            &bytes,
            |b, bytes| {
                b.iter(|| {
                    let (manifest, _): (FileManifest, _) = bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
                    black_box(manifest);
                });
            }
        );
        
        // Compare sizes
        println!("Manifest with {} files:", size);
        println!("  JSON size: {} bytes", json.len());
        println!("  Bincode size: {} bytes", bytes.len());
        println!("  Size ratio: {:.2}x smaller", json.len() as f64 / bytes.len() as f64);
    }
    
    group.finish();
}

fn create_test_manifest(file_count: usize) -> FileManifest {
    let mut rng = rand::rng();
    let mut files = Vec::with_capacity(file_count);
    
    for i in 0..file_count {
        let path = PathBuf::from(format!("dir{}/file{}.txt", i / 100, i));
        let content_hash = format!("{:064x}", rng.random::<u128>());
        let metadata_hash = format!("{:064x}", rng.random::<u128>());
        let combined_hash = format!("{:064x}", rng.random::<u128>());
        
        files.push(FileEntry {
            path,
            content_hash,
            size: rng.random_range(100..1000000),
            permissions: 0o644,
            modified: Utc::now(),
            is_compressed: rng.random_bool(0.7),
            metadata_hash,
            combined_hash,
            is_symlink: false,
            symlink_target: None,
            is_directory: false,
        });
    }
    
    FileManifest {
        checkpoint_id: "test-checkpoint-123".to_string(),
        files,
        total_size: rng.random_range(1000000..100000000),
        file_count,
        merkle_root: format!("{:064x}", rng.random::<u128>()),
        created_at: Utc::now(),
    }
}

// Quick benchmarks for development
#[cfg(feature = "quick-bench")]
criterion_group!(benches, 
    bench_checkpoint_creation,
    bench_compression_strategies
);

#[cfg(not(feature = "quick-bench"))]
criterion_group!(benches, 
    bench_checkpoint_creation,
    bench_restoration,
    bench_compression_strategies,
    bench_incremental_changes,
    bench_merkle_tree,
    bench_verification,
    bench_parallel_operations,
    bench_memory_usage,
    manifest_serialization
);

criterion_main!(benches); 