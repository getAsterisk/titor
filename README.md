<div align="center">
  
  <img src=".assets/titor.png" alt="Titor Logo" width="200">
  
  <h1><code>titor</code></h1>
  
  <p>
    <a href="https://crates.io/crates/titor"><img src="https://img.shields.io/crates/v/titor?style=flat-square" alt="Crates.io"></a>
    <a href="https://docs.rs/titor"><img src="https://img.shields.io/badge/docs-rs-blue?style=flat-square" alt="Documentation"></a>
  </p>
  
  <p>
    <em>A high-performance checkpointing library for Rust that enables time-travel capabilities through directory snapshots. Titor provides efficient incremental backups with cryptographic verification, content deduplication, and parallel processing.</em>
  </p>
</div>

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Performance](#performance)
- [Storage Format](#storage-format)
- [Security](#security)
- [CLI Usage](#cli-usage)
- [Advanced Features](#advanced-features)
- [Development](#development)
- [License](#license)

## Features

### Core Capabilities

- **Incremental Snapshots**: Only changed files are stored between checkpoints, minimizing storage overhead
- **Content-Addressable Storage**: Automatic deduplication across all checkpoints using SHA-256 hashing
- **LZ4 Compression**: Extreme-speed compression (500+ MB/s) with adaptive strategies
- **Parallel Processing**: Multi-threaded file scanning and hashing leveraging all available CPU cores
- **Merkle Tree Verification**: Cryptographic integrity checking for tamper detection
- **Timeline Branching**: Git-like branching model supporting multiple independent timelines
- **Atomic Operations**: All checkpoint and restore operations are atomic with rollback on failure
- **Memory Efficient**: Streaming architecture for large files without full memory loading
- **Line-Level Diffs**: Git-like unified diff output showing exactly what changed between checkpoints

### Technical Specifications

- **Compression**: LZ4 via `lz4_flex` crate with 4+ GB/s decompression speeds
- **Hashing**: SHA-256 for content identification and verification  
- **Serialization**: Bincode for efficient binary storage of metadata
- **Concurrency**: Rayon-based parallel processing with configurable worker pools
- **Storage**: Sharded object storage with reference counting for garbage collection

## Architecture

### System Overview

Titor implements a content-addressable storage system with the following components:

1. **Storage Layer**: Manages compressed object storage with automatic deduplication
2. **Checkpoint System**: Creates and manages independent snapshots with metadata
3. **Timeline Manager**: Maintains DAG structure of checkpoints with branching support
4. **Verification Engine**: Provides cryptographic integrity checking via Merkle trees
5. **Compression Engine**: Adaptive LZ4 compression with configurable strategies

### Storage Layout

```
storage_root/
├── metadata.json          # Storage configuration and version info
├── timeline.json          # Timeline DAG structure  
├── checkpoints/           # Checkpoint metadata directory
│   └── {checkpoint_id}/   
│       ├── metadata.json  # Checkpoint metadata (size, timestamps, etc.)
│       └── manifest.bin   # Binary manifest of all files (bincode format)
├── objects/               # Content-addressable object storage
│   └── {prefix}/          # Two-character sharding for performance
│       └── {hash}         # LZ4-compressed file content
└── refs/                  # Reference counting for garbage collection
    └── {object_hash}      # Reference count per object
```

### Object Storage

Files are stored using content-based addressing:
- SHA-256 hash computed for file content
- Objects sharded by first 2 hash characters for filesystem performance
- Reference counting enables safe garbage collection
- Compression applied based on configurable strategies

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
titor = "0.1.2"
```

### Dependencies

Required system dependencies:
- Rust 1.70+ (for stable async traits)
- Platform-specific filesystem capabilities for symbolic links

## Quick Start

### Basic Usage

```rust
use titor::{Titor, TitorBuilder, CompressionStrategy};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize with adaptive compression
    let mut titor = TitorBuilder::new()
        .compression_strategy(CompressionStrategy::Adaptive {
            min_size: 4096,
            skip_extensions: vec!["jpg".to_string(), "mp4".to_string()],
        })
        .parallel_workers(8)
        .build(
            PathBuf::from("/path/to/project"),
            PathBuf::from("/path/to/project/.titor")
        )?;

    // Create checkpoint
    let checkpoint = titor.checkpoint(Some("Initial state".to_string()))?;
    println!("Created checkpoint: {}", checkpoint.id);

    // Restore to checkpoint
    titor.restore(&checkpoint.id)?;

    Ok(())
}
```

### Line-Level Diff Example

```rust
use titor::types::DiffOptions;

// Get detailed diff with line-level changes
let options = DiffOptions {
    context_lines: 3,
    ignore_whitespace: false,
    show_line_numbers: true,
    max_file_size: 10 * 1024 * 1024, // 10MB
};

let detailed_diff = titor.diff_detailed(&checkpoint1.id, &checkpoint2.id, options)?;

// Display results
println!("Total lines added: {}", detailed_diff.total_lines_added);
println!("Total lines deleted: {}", detailed_diff.total_lines_deleted);

for file_diff in &detailed_diff.file_diffs {
    println!("\nFile: {:?}", file_diff.path);
    
    for hunk in &file_diff.hunks {
        println!("@@ -{},{} +{},{} @@", 
            hunk.from_line, hunk.from_count,
            hunk.to_line, hunk.to_count);
            
        for change in &hunk.changes {
            match change {
                LineChange::Added(_, line) => println!("+{}", line),
                LineChange::Deleted(_, line) => println!("-{}", line),
                LineChange::Context(_, line) => println!(" {}", line),
            }
        }
    }
}
```

### Advanced Configuration

```rust
let titor = TitorBuilder::new()
    .compression_strategy(CompressionStrategy::Custom(Arc::new(|path, size| {
        // Custom compression logic
        size > 1024 && !path.extension().map_or(false, |ext| ext == "zip")
    })))
    .ignore_patterns(vec![
        "*.tmp".to_string(),
        "node_modules/**".to_string(),
    ])
    .max_file_size(100 * 1024 * 1024) // 100MB limit
    .follow_symlinks(false)
    .parallel_workers(num_cpus::get())
    .build(root_path, storage_path)?;
```

## API Documentation

### Core Types

#### `Titor`

The main interface for checkpoint operations.

```rust
impl Titor {
    /// Create a new checkpoint capturing current directory state
    pub fn checkpoint(&mut self, description: Option<String>) -> Result<Checkpoint>;
    
    /// Restore directory to a specific checkpoint state
    pub fn restore(&mut self, checkpoint_id: &str) -> Result<RestoreResult>;
    
    /// List all checkpoints in chronological order
    pub fn list_checkpoints(&self) -> Result<Vec<Checkpoint>>;
    
    /// Get timeline DAG structure
    pub fn get_timeline(&self) -> Result<Timeline>;
    
    /// Create a new branch from existing checkpoint
    pub fn fork(&mut self, checkpoint_id: &str, description: Option<String>) -> Result<Checkpoint>;
    
    /// Compare two checkpoints (file-level)
    pub fn diff(&self, from_id: &str, to_id: &str) -> Result<CheckpointDiff>;
    
    /// Compare file with line-level differences
    pub fn diff_file(&self, from_id: &str, to_id: &str, path: &Path, options: DiffOptions) -> Result<FileDiff>;
    
    /// Get detailed diff with line-level changes for all files
    pub fn diff_detailed(&self, from_id: &str, to_id: &str, options: DiffOptions) -> Result<DetailedCheckpointDiff>;
    
    /// Garbage collect unreferenced objects
    pub fn gc(&self) -> Result<GcStats>;
    
    /// Verify checkpoint integrity
    pub fn verify_checkpoint(&self, checkpoint_id: &str) -> Result<VerificationReport>;
}
```

#### `CompressionStrategy`

Configurable compression strategies for different use cases.

```rust
pub enum CompressionStrategy {
    /// No compression - maximum speed
    None,
    
    /// LZ4 compression for all files (default)
    Fast,
    
    /// Adaptive compression based on file attributes
    Adaptive {
        min_size: usize,              // Skip files smaller than this
        skip_extensions: Vec<String>, // Skip these extensions
    },
    
    /// Custom compression predicate
    Custom(Arc<dyn Fn(&Path, usize) -> bool + Send + Sync>),
}
```

#### `Checkpoint`

Represents a point-in-time snapshot.

```rust
pub struct Checkpoint {
    pub id: String,                          // UUID v4 identifier
    pub parent_id: Option<String>,           // Parent checkpoint for timeline
    pub timestamp: DateTime<Utc>,            // Creation timestamp
    pub description: Option<String>,         // User description
    pub metadata: CheckpointMetadata,        // Size, file count, etc.
    pub state_hash: String,                  // SHA-256 of checkpoint state
    pub content_merkle_root: String,         // Merkle tree root hash
}
```

#### `DiffOptions`

Configure line-level diff generation.

```rust
pub struct DiffOptions {
    pub context_lines: usize,      // Lines of context (default: 3)
    pub ignore_whitespace: bool,   // Ignore whitespace changes
    pub show_line_numbers: bool,   // Include line numbers
    pub max_file_size: u64,       // Max file size for diffs (default: 10MB)
}
```

#### `FileDiff`

Line-level diff information for a single file.

```rust
pub struct FileDiff {
    pub path: PathBuf,
    pub is_binary: bool,
    pub hunks: Vec<DiffHunk>,     // Contiguous blocks of changes
    pub lines_added: usize,
    pub lines_deleted: usize,
}
```

### Error Handling

Titor uses a comprehensive error type hierarchy:

```rust
#[derive(Debug, thiserror::Error)]
pub enum TitorError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Checkpoint not found: {0}")]
    CheckpointNotFound(String),
    
    #[error("Storage corruption detected: {0}")]
    StorageCorruption(String),
    
    #[error("Checkpoint has children: {0}")]
    CheckpointHasChildren(String),
    
    // Additional error variants...
}
```

## Performance

### Optimization Strategies

1. **Parallel File Scanning**: Utilizes Rayon for concurrent directory traversal
2. **Streaming Compression**: Processes large files in chunks to minimize memory usage
3. **Content Deduplication**: Identical files stored only once across all checkpoints
4. **Lazy Loading**: Objects loaded from storage only when needed
5. **Sharded Storage**: Two-character prefix sharding prevents filesystem bottlenecks

## Storage Format

### Object Format

Each stored object consists of:
- 4-byte header indicating compression status
- LZ4-compressed content (if compression applied)
- Original content (if compression not beneficial)

### Manifest Format

File manifests use Bincode serialization for efficiency:
```rust
pub struct FileManifest {
    pub checkpoint_id: String,
    pub files: Vec<FileEntry>,
    pub total_size: u64,
    pub file_count: usize,
    pub merkle_root: String,
    pub created_at: DateTime<Utc>,
}
```

## Security

### Cryptographic Verification

Titor implements multiple layers of integrity checking:

1. **Content Hashing**: SHA-256 hash for each file
2. **Merkle Trees**: Cryptographic proof of entire checkpoint state
3. **State Hashing**: Combined hash of all checkpoint components
4. **Tamper Detection**: Automatic corruption detection during verification

### Verification API

```rust
let verifier = CheckpointVerifier::new(&storage);
let report = verifier.verify_complete(&checkpoint)?;

if !report.is_valid() {
    eprintln!("Verification failed: {}", report.summary());
    for error in &report.errors {
        eprintln!("  - {}", error);
    }
}
```

## CLI Usage

Titor includes a comprehensive command-line interface.

### Installation

```bash
cargo install titor-cli
```

### Commands

```bash
# Initialize repository
titor init --compression adaptive

# Create checkpoint
titor checkpoint -m "Before refactoring"

# List checkpoints
titor list --detailed

# Restore to checkpoint
titor restore <checkpoint-id>

# Show timeline tree
titor timeline

# Compare checkpoints
titor diff <from-id> <to-id>

# Compare with line-level differences (git-like)
titor diff <from-id> <to-id> --lines

# Compare with custom context lines
titor diff <from-id> <to-id> --lines --context 5

# Show only statistics
titor diff <from-id> <to-id> --stat

# Ignore whitespace changes
titor diff <from-id> <to-id> --lines --ignore-whitespace

# Verify integrity
titor verify --all

# Garbage collection
titor gc --dry-run
```

## Advanced Features

### Timeline Branching

Create independent branches for experimentation:

```rust
// Fork from existing checkpoint
let fork = titor.fork(&checkpoint_id, Some("Experimental branch".to_string()))?;

// Work on branch...

// Later, restore to original
titor.restore(&checkpoint_id)?;
```

### Auto-Checkpoint Strategies

Configure automatic checkpoint creation:

```rust
titor.set_auto_checkpoint(AutoCheckpointStrategy::Smart {
    min_files_changed: 10,
    min_size_changed: 1024 * 1024, // 1MB
    max_time_between: Duration::from_secs(3600), // 1 hour
});
```

### Checkpoint Hooks

Implement custom logic for checkpoint events:

```rust
impl CheckpointHook for MyHook {
    fn pre_checkpoint(&self, stats: &ChangeStats) -> Result<()> {
        println!("Creating checkpoint with {} changes", stats.total_operations());
        Ok(())
    }
    
    fn post_checkpoint(&self, checkpoint: &Checkpoint) -> Result<()> {
        // Custom post-checkpoint logic
        Ok(())
    }
}

titor.add_hook(Box::new(MyHook));
```

## Development

### Building from Source

```bash
git clone https://github.com/mufeedvh/titor.git
cd titor
cargo build --release
```

### Running Tests

```bash
# Unit and integration tests
cargo test

# Include ignored tests
cargo test -- --ignored

# Run with logging
RUST_LOG=debug cargo test
```

### Project Structure

```
titor/
├── src/
│   ├── lib.rs              # Public API
│   ├── titor.rs           # Core implementation
│   ├── storage.rs          # Storage backend
│   ├── checkpoint.rs       # Checkpoint types
│   ├── timeline.rs         # Timeline management
│   ├── compression.rs      # Compression engine
│   ├── verification.rs     # Integrity checking
│   └── merkle.rs          # Merkle tree implementation
├── benches/               # Performance benchmarks
├── examples/              # Example implementations
└── tests/                # Integration tests
```

## License

Licensed under the MIT License, see [LICENSE](LICENSE) for more information.