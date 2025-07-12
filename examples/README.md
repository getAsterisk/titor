# Titor CLI - Time Travel for Your Files

A comprehensive command-line interface demonstrating all features of the Titor checkpointing library. This CLI provides Git-like version control for any directory with powerful time-travel capabilities.

## Features Overview

- **Checkpoint Management**: Create, list, and restore directory snapshots
- **Timeline Navigation**: Visualize and navigate through checkpoint history
- **Branching**: Fork from any checkpoint to create alternative timelines
- **Diffing**: Compare changes between checkpoints
- **Verification**: Ensure data integrity with cryptographic verification
- **Storage Optimization**: Garbage collection and compression
- **Progress Tracking**: Visual feedback for long operations

## Installation

Build the CLI from the project root:

```bash
# Build in release mode for optimal performance
cargo build --example titor_cli --release

# Copy to your PATH (optional)
cp target/release/examples/titor_cli ~/.local/bin/titor

# Or run directly
cargo run --example titor_cli -- --help
```

## Quick Start

```bash
# Initialize Titor in current directory
titor init

# Create your first checkpoint
titor checkpoint -m "Initial project state"

# Make some changes to your files...
echo "new content" > file.txt

# Create another checkpoint
titor checkpoint -m "Added new file"

# View your timeline
titor timeline

# Compare checkpoints
titor diff <checkpoint1> <checkpoint2>

# Restore to previous state
titor restore <checkpoint-id>
```

## Command Reference

### Global Options

- `-p, --path <PATH>`: Target directory (defaults to current)
- `-s, --storage <PATH>`: Storage directory (defaults to .titor)
- `-v, --verbose`: Enable debug output
- `--help`: Show help information
- `--version`: Show version information

### `init` - Initialize Repository

Initialize Titor tracking in a directory.

```bash
titor init [OPTIONS]

Options:
  --compression <MODE>     Compression strategy [none, fast, adaptive] (default: adaptive)
  -i, --ignore <PATTERN>   File patterns to ignore (gitignore syntax)
  --force                  Force initialization even if already initialized
```

**Examples:**

```bash
# Basic initialization
titor init

# Custom compression and ignore patterns
titor init --compression fast -i "*.log" -i "node_modules/"

# Initialize with custom storage location
titor -s /backup/project-titor init
```

### `checkpoint` (alias: `cp`) - Create Checkpoint

Capture the current state of your directory.

```bash
titor checkpoint [OPTIONS]

Options:
  -m, --message <MESSAGE>  Description for the checkpoint
  --progress               Show progress bar
```

**Examples:**

```bash
# Quick checkpoint
titor cp

# With description
titor checkpoint -m "Before major refactoring"

# With progress tracking
titor checkpoint --progress -m "Release v1.0.0"
```

### `restore` (alias: `rs`) - Restore Checkpoint

Restore directory to a previous checkpoint state.

```bash
titor restore <CHECKPOINT> [OPTIONS]

Options:
  --progress  Show progress bar
```

**Examples:**

```bash
# Restore by checkpoint ID (first 8 chars sufficient)
titor restore abc12345

# Restore with progress
titor restore abc12345 --progress
```

### `list` (alias: `ls`) - List Checkpoints

Display all checkpoints with metadata.

```bash
titor list [OPTIONS]

Options:
  -d, --detailed       Show detailed information
  -l, --limit <NUM>    Limit number of results
```

**Examples:**

```bash
# Basic list
titor ls

# Detailed view with file counts
titor list --detailed

# Show only last 10 checkpoints
titor list --limit 10
```

### `timeline` (alias: `tl`) - Show Timeline Tree

Visualize checkpoint history as a tree structure.

```bash
titor timeline [OPTIONS]

Options:
  --ascii  Use ASCII characters for tree drawing
```

**Examples:**

```bash
# Beautiful Unicode tree
titor timeline

# ASCII tree for compatibility
titor timeline --ascii
```

### `fork` - Create Branch

Fork from an existing checkpoint to create a new timeline branch.

```bash
titor fork <CHECKPOINT> [OPTIONS]

Options:
  -m, --message <MESSAGE>  Description for the fork
```

**Examples:**

```bash
# Fork from checkpoint
titor fork abc12345 -m "Experimental feature branch"

# Fork from current checkpoint
titor fork HEAD -m "Alternative approach"
```

### `diff` - Compare Checkpoints

Show differences between two checkpoints.

```bash
titor diff <FROM> <TO> [OPTIONS]

Options:
  -l, --lines              Show line-level differences (like git diff)
  --context <NUM>          Number of context lines (default: 3)
  --stat                   Show only statistics
  --ignore-whitespace      Ignore whitespace changes
```

**Examples:**

```bash
# Basic file-level comparison
titor diff abc12345 def67890

# Line-level diff with git-like output
titor diff abc12345 def67890 --lines

# Custom context lines
titor diff abc12345 def67890 --lines --context 5

# Statistics only
titor diff abc12345 def67890 --stat

# Ignore whitespace changes
titor diff abc12345 def67890 --lines --ignore-whitespace
```

**Output Format:**

With `--lines`, the output shows unified diff format:
```diff
--- a/file.txt
+++ b/file.txt
@@ -1,5 +1,6 @@
 context line
-removed line
+added line
 another context line
```

### `verify` - Verify Integrity

Perform cryptographic verification of checkpoint data.

```bash
titor verify [CHECKPOINT] [OPTIONS]

Options:
  --all  Verify all checkpoints
```

**Examples:**

```bash
# Verify current checkpoint
titor verify

# Verify specific checkpoint
titor verify abc12345

# Verify entire timeline
titor verify --all
```

### `gc` - Garbage Collection

Remove unreferenced objects to free storage space.

```bash
titor gc [OPTIONS]

Options:
  --dry-run  Show what would be deleted without removing
```

**Examples:**

```bash
# Preview cleanup
titor gc --dry-run

# Perform cleanup
titor gc
```

### `status` - Show Status

Display current repository status and statistics.

```bash
titor status
```

### `info` - Checkpoint Information

Show detailed information about a specific checkpoint.

```bash
titor info <CHECKPOINT>
```

**Examples:**

```bash
# Get checkpoint details
titor info abc12345
```

## Advanced Usage

### Working with Large Directories

For directories with many files, use progress tracking:

```bash
# Show progress for checkpoint creation
titor checkpoint --progress -m "Large directory snapshot"

# Show progress for restoration
titor restore abc12345 --progress
```

### Branching Workflows

Create experimental branches without affecting main timeline:

```bash
# Create a fork for experimentation
titor fork abc12345 -m "Try new approach"

# Make changes and checkpoint
echo "experimental" > test.txt
titor cp -m "Experimental change"

# If happy, continue on this branch
# If not, restore to original branch
titor restore abc12345
```

### Automation with Scripts

The CLI is designed for scripting:

```bash
#!/bin/bash
# Auto-checkpoint script

# Create checkpoint with timestamp
titor checkpoint -m "Auto-checkpoint $(date +%Y-%m-%d_%H:%M:%S)"

# Verify integrity
if titor verify; then
    echo "Checkpoint verified successfully"
else
    echo "Checkpoint verification failed!"
    exit 1
fi

# Clean up old objects
titor gc
```

### Integration with CI/CD

```yaml
# Example GitHub Actions workflow
name: Checkpoint on Release

on:
  release:
    types: [created]

jobs:
  checkpoint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Titor
        run: cargo install --example titor_cli
      
      - name: Create release checkpoint
        run: |
          titor init
          titor checkpoint -m "Release ${{ github.event.release.tag_name }}"
      
      - name: Verify checkpoint
        run: titor verify
```

## Performance Tips

1. **Compression Strategy**:
   - Use `adaptive` for mixed content (default)
   - Use `fast` for speed priority
   - Use `none` for already-compressed files

2. **Ignore Patterns**:
   - Exclude build artifacts: `-i "target/" -i "dist/"`
   - Exclude dependencies: `-i "node_modules/" -i "vendor/"`
   - Exclude logs: `-i "*.log" -i "*.tmp"`

3. **Storage Location**:
   - Use SSD for storage directory when possible
   - Consider separate disk for large repositories
   - Network storage works but may be slower

## Troubleshooting

### Common Issues

**"Not a Titor repository"**
- Run `titor init` first
- Check you're in the correct directory
- Verify storage directory exists

**"Checkpoint not found"**
- Use `titor list` to see available checkpoints
- Check checkpoint ID spelling
- First 8 characters are usually sufficient

**"Permission denied"**
- Ensure write access to storage directory
- Check file permissions in target directory
- Run with appropriate user privileges

### Debug Mode

Enable verbose output for troubleshooting:

```bash
titor -v <command>
```

### Storage Inspection

Check storage health:

```bash
# Show storage statistics
titor status

# Verify all checkpoints
titor verify --all

# Check for orphaned objects
titor gc --dry-run
```

## Architecture Overview

The CLI demonstrates all major Titor features:

1. **Content-Addressable Storage**: Files stored by content hash
2. **Merkle Trees**: Cryptographic verification of file integrity
3. **Incremental Snapshots**: Only changed files stored
4. **Compression**: LZ4 compression for efficiency
5. **Deduplication**: Identical files stored once
6. **Atomic Operations**: All-or-nothing checkpoint/restore
7. **Line-Level Diffs**: Git-like unified diff output for text files

## Contributing

This CLI serves as a reference implementation. To extend:

1. Add new commands in the `Commands` enum
2. Implement command handler functions
3. Update help documentation
4. Add tests for new functionality

## License

Same as the Titor library - MIT