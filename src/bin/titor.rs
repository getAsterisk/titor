//! # Titor CLI - Time Travel for Your Files
//!
//! A comprehensive command-line interface for the Titor checkpointing library.
//! 
//! ## Features
//! - Create and manage checkpoints of directory states
//! - Navigate through timeline history
//! - Compare changes between checkpoints with line-level diffs
//! - Verify checkpoint integrity
//! - Optimize storage with garbage collection
//!
//! ## Usage
//! ```bash
//! # Initialize Titor in current directory
//! titor init
//! 
//! # Create a checkpoint
//! titor checkpoint -m "Initial state"
//! 
//! # List all checkpoints
//! titor list
//! 
//! # Restore to a checkpoint
//! titor restore <checkpoint-id>
//! 
//! # Compare checkpoints with line-level diff
//! titor diff <from-id> <to-id> --lines
//! ```

use clap::{Parser, Subcommand, ValueEnum};
use colored::*;
use indicatif::{ProgressBar, ProgressStyle};
use humantime::format_duration;
use std::path::{Path, PathBuf};
use std::time::Instant;
use titor::{
    CompressionStrategy, Result, Titor, TitorBuilder, TitorError, 
};

/// Titor CLI - Comprehensive checkpoint management for directories
#[derive(Parser)]
#[command(name = "titor")]
#[command(author = "Mufeed VH <mufeed@asterisk.so>")]
#[command(version = "1.0")]
#[command(about = "Time travel for your files - checkpoint and restore directory states")]
#[command(long_about = None)]
struct Cli {
    /// Path to directory (defaults to current)
    #[arg(short, long, global = true)]
    path: Option<PathBuf>,

    /// Storage directory (defaults to .titor)
    #[arg(short, long, global = true)]
    storage: Option<PathBuf>,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize Titor in a directory
    Init {
        /// Compression strategy
        #[arg(long, value_enum, default_value = "adaptive")]
        compression: CompressionMode,

        /// Ignore patterns (gitignore syntax)
        #[arg(short, long)]
        ignore: Vec<String>,

        /// Force initialization
        #[arg(long)]
        force: bool,
    },

    /// Create a checkpoint
    #[command(alias = "cp")]
    Checkpoint {
        /// Description message
        #[arg(short, long)]
        message: Option<String>,

        /// Show progress
        #[arg(long)]
        progress: bool,
    },

    /// Restore to a checkpoint
    #[command(alias = "rs")]
    Restore {
        /// Checkpoint ID
        checkpoint: String,

        /// Show progress
        #[arg(long)]
        progress: bool,
    },

    /// List checkpoints
    #[command(alias = "ls")]
    List {
        /// Show detailed info
        #[arg(short, long)]
        detailed: bool,

        /// Limit results
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Show timeline
    #[command(alias = "tl")]
    Timeline {
        /// Use ASCII characters
        #[arg(long)]
        ascii: bool,
    },

    /// Fork from checkpoint
    Fork {
        /// Checkpoint to fork from
        checkpoint: String,

        /// Fork message
        #[arg(short, long)]
        message: Option<String>,
    },

    /// Compare checkpoints
    Diff {
        /// From checkpoint
        from: String,

        /// To checkpoint
        to: String,
        
        /// Show line-level differences (like git diff)
        #[arg(short, long)]
        lines: bool,
        
        /// Number of context lines to show
        #[arg(long, default_value = "3")]
        context: usize,
        
        /// Show only statistics
        #[arg(long)]
        stat: bool,
        
        /// Ignore whitespace changes
        #[arg(long)]
        ignore_whitespace: bool,
    },

    /// Verify integrity
    Verify {
        /// Checkpoint to verify
        checkpoint: Option<String>,

        /// Verify all
        #[arg(long)]
        all: bool,
    },

    /// Garbage collect
    #[command(alias = "gc")]
    GarbageCollect {
        /// Dry run
        #[arg(long)]
        dry_run: bool,
    },

    /// Show status
    Status,

    /// Show checkpoint info
    Info {
        /// Checkpoint ID
        checkpoint: String,
    },
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum CompressionMode {
    None,
    Fast,
    Adaptive,
}

fn main() {
    let cli = Cli::parse();

    // Set up logging
    if cli.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    }

    // Disable colors if needed
    if std::env::var("NO_COLOR").is_ok() {
        colored::control::set_override(false);
    }

    // Run command
    if let Err(e) = run(cli) {
        eprintln!("{}: {}", "Error".red().bold(), e);
        std::process::exit(1);
    }
}

/// Main command runner
fn run(cli: Cli) -> Result<()> {
    let root_path = cli.path.unwrap_or_else(|| PathBuf::from("."));
    let storage_path = cli.storage.unwrap_or_else(|| root_path.join(".titor"));

    match cli.command {
        Commands::Init { compression, ignore, force } => {
            cmd_init(root_path, storage_path, compression, ignore, force)
        }
        Commands::Checkpoint { message, progress } => {
            cmd_checkpoint(root_path, storage_path, message, progress)
        }
        Commands::Restore { checkpoint, progress } => {
            cmd_restore(root_path, storage_path, checkpoint, progress)
        }
        Commands::List { detailed, limit } => {
            cmd_list(root_path, storage_path, detailed, limit)
        }
        Commands::Timeline { ascii } => {
            cmd_timeline(root_path, storage_path, ascii)
        }
        Commands::Fork { checkpoint, message } => {
            cmd_fork(root_path, storage_path, checkpoint, message)
        }
        Commands::Diff { from, to, lines, context, stat, ignore_whitespace } => {
            cmd_diff(root_path, storage_path, from, to, lines, context, stat, ignore_whitespace)
        }
        Commands::Verify { checkpoint, all } => {
            cmd_verify(root_path, storage_path, checkpoint, all)
        }
        Commands::GarbageCollect { dry_run } => {
            cmd_gc(root_path, storage_path, dry_run)
        }
        Commands::Status => {
            cmd_status(root_path, storage_path)
        }
        Commands::Info { checkpoint } => {
            cmd_info(root_path, storage_path, checkpoint)
        }
    }
}

/// Initialize Titor in a directory
///
/// This creates the storage structure needed for Titor to track checkpoints.
/// The storage directory contains:
/// - metadata.json: Configuration and version info
/// - timeline.json: Timeline structure  
/// - checkpoints/: Checkpoint metadata
/// - objects/: Content-addressable file storage
/// - refs/: Reference counting for garbage collection
fn cmd_init(
    root_path: PathBuf,
    storage_path: PathBuf,
    compression: CompressionMode,
    ignore: Vec<String>,
    force: bool,
) -> Result<()> {
    // Check if already initialized
    if storage_path.exists() && !force {
        return Err(TitorError::internal(
            "Directory already initialized. Use --force to reinitialize."
        ));
    }

    println!("{}", "Initializing Titor...".blue().bold());

    // Set up compression strategy
    let compression_strategy = match compression {
        CompressionMode::None => CompressionStrategy::None,
        CompressionMode::Fast => CompressionStrategy::Fast,
        CompressionMode::Adaptive => CompressionStrategy::Adaptive {
            min_size: 4096,
            skip_extensions: vec![
                "jpg", "jpeg", "png", "gif", "mp4", "mp3", 
                "zip", "gz", "bz2", "7z", "rar"
            ].iter().map(|s| s.to_string()).collect(),
        },
    };

    // Build Titor instance
    let _titor = TitorBuilder::new()
        .compression_strategy(compression_strategy)
        .ignore_patterns(ignore)
        .build(root_path.clone(), storage_path.clone())?;

    println!("{} Initialized Titor repository", "✓".green().bold());
    println!("  Root: {}", root_path.display().to_string().cyan());
    println!("  Storage: {}", storage_path.display().to_string().cyan());
    println!("\nNext steps:");
    println!("  - Create your first checkpoint: {}", "titor checkpoint -m \"Initial state\"".yellow());
    println!("  - View timeline: {}", "titor timeline".yellow());

    Ok(())
}

/// Create a new checkpoint
///
/// Checkpoints capture the complete state of your directory at a point in time.
/// Features:
/// - Incremental storage (only changed files stored)
/// - Content deduplication across checkpoints
/// - Cryptographic verification with merkle trees
/// - Compression for efficient storage
fn cmd_checkpoint(
    root_path: PathBuf,
    storage_path: PathBuf,
    message: Option<String>,
    show_progress: bool,
) -> Result<()> {
    let mut titor = open_titor(root_path, storage_path)?;
    
    println!("{}", "Creating checkpoint...".blue().bold());

    let start = Instant::now();
    let progress = if show_progress {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        pb.set_message("Scanning files...");
        Some(pb)
    } else {
        None
    };

    // Create checkpoint
    let checkpoint = titor.checkpoint(message.clone())?;

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    let duration = start.elapsed();
    
    println!("{} Created checkpoint {}", 
        "✓".green().bold(),
        checkpoint.id[..8].yellow().bold()
    );
    
    if let Some(msg) = &message {
        println!("  Message: {}", msg.cyan());
    }
    
    println!("  Files: {}", checkpoint.metadata.file_count.to_string().cyan());
    println!("  Size: {}", format_bytes(checkpoint.metadata.total_size).cyan());
    println!("  Time: {}", format_duration(duration).to_string().cyan());
    
    if checkpoint.metadata.files_changed > 0 {
        println!("  Changed: {} files", checkpoint.metadata.files_changed.to_string().yellow());
    }

    Ok(())
}

/// Restore to a checkpoint
///
/// This operation atomically restores your directory to the exact state
/// captured in the specified checkpoint. Files are:
/// - Restored if they existed in the checkpoint
/// - Deleted if they didn't exist in the checkpoint
/// - Updated with correct permissions and content
fn cmd_restore(
    root_path: PathBuf,
    storage_path: PathBuf,
    checkpoint_id: String,
    show_progress: bool,
) -> Result<()> {
    let mut titor = open_titor(root_path, storage_path)?;
    
    // Resolve prefix to full ID to enhance ergonomics
    let full_id = {
        let checkpoints = titor.list_checkpoints()?;
        checkpoints
            .iter()
            .find(|c| c.id.starts_with(&checkpoint_id))
            .map(|c| c.id.clone())
            .ok_or_else(|| TitorError::internal(format!("Checkpoint not found: {}", checkpoint_id)))?
    };

    println!("{} {}", 
        "Restoring to checkpoint".blue().bold(),
        full_id[..8].yellow()
    );

    let _start = Instant::now();
    let progress = if show_progress {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_spinner()
                .template("{spinner:.green} {msg}")
                .unwrap()
        );
        pb.set_message("Restoring files...");
        Some(pb)
    } else {
        None
    };

    // Restore
    let result = titor.restore(&full_id)?;

    if let Some(pb) = progress {
        pb.finish_and_clear();
    }

    println!("{} Restoration complete", "✓".green().bold());
    println!("  Files restored: {}", result.files_restored.to_string().cyan());
    println!("  Files deleted: {}", result.files_deleted.to_string().yellow());
    println!("  Bytes written: {}", format_bytes(result.bytes_written).cyan());
    println!("  Time: {}", format_duration(std::time::Duration::from_millis(result.duration_ms)).to_string().cyan());
    
    if !result.warnings.is_empty() {
        println!("\n{}", "Warnings:".yellow().bold());
        for warning in &result.warnings {
            println!("  - {}", warning.yellow());
        }
    }

    Ok(())
}

/// List all checkpoints
///
/// Shows checkpoint history with essential information.
/// Checkpoints are shown in chronological order by default.
fn cmd_list(
    root_path: PathBuf,
    storage_path: PathBuf,
    detailed: bool,
    limit: Option<usize>,
) -> Result<()> {
    let titor = open_titor(root_path, storage_path)?;
    let checkpoints = titor.list_checkpoints()?;
    
    if checkpoints.is_empty() {
        println!("{}", "No checkpoints found.".yellow());
        return Ok(());
    }

    println!("{}", "Checkpoints:".blue().bold());
    println!();

    let display_count = limit.unwrap_or(checkpoints.len()).min(checkpoints.len());
    
    for checkpoint in checkpoints.iter().take(display_count) {
        let marker = if titor.get_timeline()?.current_checkpoint_id == Some(checkpoint.id.clone()) {
            "*".green().bold()
        } else {
            " ".normal()
        };
        
        print!("{} {} ", marker, checkpoint.id[..8].yellow().bold());
        print!("{} ", checkpoint.timestamp.format("%Y-%m-%d %H:%M:%S").to_string().dimmed());
        
        if let Some(desc) = &checkpoint.description {
            print!("{}", desc.cyan());
        }
        
        println!();
        
        if detailed {
            println!("    Files: {} | Size: {} | Changed: {}", 
                checkpoint.metadata.file_count.to_string().dimmed(),
                format_bytes(checkpoint.metadata.total_size).dimmed(),
                checkpoint.metadata.files_changed.to_string().dimmed()
            );
            
            if let Some(parent) = &checkpoint.parent_id {
                println!("    Parent: {}", parent[..8].dimmed());
            }
            
            if checkpoint.metadata.titor_version != env!("CARGO_PKG_VERSION") {
                println!("    Version: {}", checkpoint.metadata.titor_version.dimmed());
            }
            
            println!();
        }
    }
    
    if limit.is_some() && display_count < checkpoints.len() {
        println!("\n{}", 
            format!("Showing {} of {} checkpoints", display_count, checkpoints.len()).dimmed()
        );
    }

    Ok(())
}

/// Display timeline as a tree
///
/// Visualizes the checkpoint history showing parent-child relationships
/// and branching structure, similar to git log --graph.
fn cmd_timeline(
    root_path: PathBuf,
    storage_path: PathBuf,
    ascii: bool,
) -> Result<()> {
    let titor = open_titor(root_path, storage_path)?;
    let timeline = titor.get_timeline()?;
    
    println!("{}", "Timeline:".blue().bold());
    println!();
    
    // Get checkpoints sorted by timestamp
    let mut checkpoints: Vec<_> = timeline.checkpoints.values().collect();
    checkpoints.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
    
    // Build tree structure
    for (idx, checkpoint) in checkpoints.iter().enumerate() {
        let is_current = timeline.current_checkpoint_id == Some(checkpoint.id.clone());
        let prefix = if is_current { "*" } else { "○" };
        
        // Draw tree lines
        if idx > 0 {
            println!("{}", if ascii { "|" } else { "│" }.dimmed());
        }
        
        print!("{} {} ", 
            prefix.green().bold(),
            checkpoint.id[..8].yellow()
        );
        
        print!("{} ", 
            checkpoint.timestamp.format("%Y-%m-%d %H:%M").to_string().dimmed()
        );
        
        if let Some(desc) = &checkpoint.description {
            print!("{}", desc.cyan());
        }
        
        if is_current {
            print!(" {}", "(current)".green().dimmed());
        }
        
        println!();
    }

    Ok(())
}

/// Fork from a checkpoint
///
/// Creates a new branch in the timeline starting from the specified checkpoint.
/// This is useful for experimenting with changes without affecting the main timeline.
fn cmd_fork(
    root_path: PathBuf,
    storage_path: PathBuf,
    checkpoint_id: String,
    message: Option<String>,
) -> Result<()> {
    let mut titor = open_titor(root_path, storage_path)?;
    
    // Resolve prefix to full ID to enhance ergonomics
    let full_id = {
        let checkpoints = titor.list_checkpoints()?;
        checkpoints
            .iter()
            .find(|c| c.id.starts_with(&checkpoint_id))
            .map(|c| c.id.clone())
            .ok_or_else(|| TitorError::internal(format!("Checkpoint not found: {}", checkpoint_id)))?
    };
    
    println!("{}{}",
        "Forking from checkpoint ".blue().bold(),
        full_id[..8].yellow()
    );

    let fork = titor.fork(&full_id, message)?;
    
    println!("{} Created fork {}", 
        "✓".green().bold(),
        fork.id[..8].yellow().bold()
    );
    
    if let Some(desc) = &fork.description {
        println!("  Message: {}", desc.cyan());
    }

    Ok(())
}

/// Compare two checkpoints
///
/// Shows differences between checkpoints including:
/// - Files added, modified, or deleted
/// - Size changes
/// - Statistics about changes
/// - Line-level differences (with --lines flag)
fn cmd_diff(
    root_path: PathBuf,
    storage_path: PathBuf,
    from_id: String,
    to_id: String,
    show_lines: bool,
    context_lines: usize,
    stat_only: bool,
    ignore_whitespace: bool,
) -> Result<()> {
    let titor = open_titor(root_path, storage_path)?;
    
    // Resolve shortened IDs (prefixes) to full checkpoint IDs for convenience
    let checkpoints = titor.list_checkpoints()?;
    let resolve_id = |prefix: &str| {
        checkpoints
            .iter()
            .find(|c| c.id.starts_with(prefix))
            .map(|c| c.id.clone())
            .ok_or_else(|| TitorError::internal(format!("Checkpoint not found: {}", prefix)))
    };

    let from_id_full = resolve_id(&from_id)?;
    let to_id_full = resolve_id(&to_id)?;

    println!("{} {} → {}", 
        "Comparing".blue().bold(),
        from_id_full[..8].yellow(),
        to_id_full[..8].yellow()
    );
    println!();

    // Get the diff based on whether we want line-level details
    if show_lines && !stat_only {
        // Get detailed diff with line-level changes
        let options = titor::types::DiffOptions {
            context_lines,
            ignore_whitespace,
            show_line_numbers: true,
            max_file_size: 10 * 1024 * 1024, // 10MB
        };
        
        let detailed_diff = titor.diff_detailed(&from_id_full, &to_id_full, options)?;
        
        // Show summary statistics first
        show_diff_stats(&detailed_diff.basic_diff, detailed_diff.total_lines_added, detailed_diff.total_lines_deleted);
        
        if !stat_only {
            // Show line-level diffs for each file
            for file_diff in &detailed_diff.file_diffs {
                println!("\n{} {}", "diff --git".dimmed(), file_diff.path.display().to_string().cyan());
                println!("{} a/{}", "---".dimmed(), file_diff.path.display());
                println!("{} b/{}", "+++".dimmed(), file_diff.path.display());
                
                if file_diff.is_binary {
                    println!("{}", "Binary files differ".yellow());
                    continue;
                }
                
                // Print hunks
                for hunk in &file_diff.hunks {
                    println!("{} @@ -{},{} +{},{} @@", 
                        "@@".cyan(),
                        hunk.from_line, hunk.from_count,
                        hunk.to_line, hunk.to_count
                    );
                    
                    for change in &hunk.changes {
                        match change {
                            titor::types::LineChange::Added(_, content) => {
                                println!("{}{}", "+".green(), content.green());
                            }
                            titor::types::LineChange::Deleted(_, content) => {
                                println!("{}{}", "-".red(), content.red());
                            }
                            titor::types::LineChange::Context(_, content) => {
                                println!(" {}", content.dimmed());
                            }
                        }
                    }
                }
            }
            
            // Show file lists for added/deleted files
            if !detailed_diff.basic_diff.added_files.is_empty() {
                println!("\n{}", "Added files:".green().bold());
                for file in &detailed_diff.basic_diff.added_files {
                    println!("  + {}", file.path.display().to_string().green());
                }
            }
            
            if !detailed_diff.basic_diff.deleted_files.is_empty() {
                println!("\n{}", "Deleted files:".red().bold());
                for file in &detailed_diff.basic_diff.deleted_files {
                    println!("  - {}", file.path.display().to_string().red());
                }
            }
        }
    } else {
        // Basic file-level diff
        let diff = titor.diff(&from_id_full, &to_id_full)?;
        
        if stat_only {
            show_diff_stats(&diff, 0, 0);
        } else {
            show_basic_diff(&diff);
        }
    }

    Ok(())
}

/// Show diff statistics
fn show_diff_stats(
    diff: &titor::types::CheckpointDiff, 
    lines_added: usize, 
    lines_deleted: usize
) {
    println!("{}", "Summary:".bold());
    
    if lines_added > 0 || lines_deleted > 0 {
        // Show line-level stats if available
        let net_lines = lines_added as i32 - lines_deleted as i32;
        let sign = if net_lines >= 0 { "+" } else { "" };
        
        println!("  {} files changed, {} insertions(+), {} deletions(-)",
            diff.stats.total_operations(),
            lines_added.to_string().green(),
            lines_deleted.to_string().red()
        );
        println!("  Net lines: {}{}", sign, net_lines);
    } else {
        // File-level stats only
        println!("  Added: {} files ({})", 
            diff.stats.files_added.to_string().green(),
            format_bytes(diff.stats.bytes_added).green()
        );
        println!("  Modified: {} files ({})", 
            diff.stats.files_modified.to_string().yellow(),
            format_bytes(diff.stats.bytes_modified).yellow()
        );
        println!("  Deleted: {} files ({})", 
            diff.stats.files_deleted.to_string().red(),
            format_bytes(diff.stats.bytes_deleted).red()
        );
    }
}

/// Show basic file-level diff
fn show_basic_diff(diff: &titor::types::CheckpointDiff) {
    // Summary
    show_diff_stats(diff, 0, 0);
    
    // File lists
    if !diff.added_files.is_empty() {
        println!("\n{}", "Added files:".green().bold());
        for file in diff.added_files.iter().take(10) {
            println!("  + {}", file.path.display().to_string().green());
        }
        if diff.added_files.len() > 10 {
            println!("  ... and {} more", diff.added_files.len() - 10);
        }
    }
    
    if !diff.modified_files.is_empty() {
        println!("\n{}", "Modified files:".yellow().bold());
        for (_, new) in diff.modified_files.iter().take(10) {
            println!("  ~ {}", new.path.display().to_string().yellow());
        }
        if diff.modified_files.len() > 10 {
            println!("  ... and {} more", diff.modified_files.len() - 10);
        }
    }
    
    if !diff.deleted_files.is_empty() {
        println!("\n{}", "Deleted files:".red().bold());
        for file in diff.deleted_files.iter().take(10) {
            println!("  - {}", file.path.display().to_string().red());
        }
        if diff.deleted_files.len() > 10 {
            println!("  ... and {} more", diff.deleted_files.len() - 10);
        }
    }
}

/// Verify checkpoint integrity
///
/// Performs cryptographic verification including:
/// - State hash verification
/// - Merkle tree validation
/// - File content verification
/// - Parent chain validation
fn cmd_verify(
    root_path: PathBuf,
    storage_path: PathBuf,
    checkpoint_id: Option<String>,
    verify_all: bool,
) -> Result<()> {
    let titor = open_titor(root_path, storage_path)?;
    
    if verify_all {
        println!("{}", "Verifying all checkpoints...".blue().bold());
        let report = titor.verify_timeline()?;
        
        println!("\n{}", "Timeline Verification Report:".bold());
        println!("  Total checkpoints: {}", report.total_checkpoints);
        println!("  Valid checkpoints: {}", report.valid_checkpoints.to_string().green());
        println!("  Invalid checkpoints: {}", report.invalid_checkpoints.to_string().red());
        
        if report.timeline_structure_valid {
            println!("  Timeline structure: {}", "✓ Valid".green());
        } else {
            println!("  Timeline structure: {}", "✗ Invalid".red());
        }
        
        if report.no_hash_conflicts {
            println!("  Hash conflicts: {}", "✓ None".green());
        } else {
            println!("  Hash conflicts: {}", "✗ Found".red());
        }
        
        println!("  Verification time: {}ms", report.verification_time_ms);
    } else {
        let id_prefix = checkpoint_id.unwrap_or_else(|| {
            titor.get_timeline().unwrap()
                .current_checkpoint_id
                .unwrap_or_else(|| "".to_string())
        });

        if id_prefix.is_empty() {
            return Err(TitorError::internal("No checkpoint specified and no current checkpoint"));
        }

        // Resolve shortened checkpoint ID to full ID
        let full_id = {
            let checkpoints = titor.list_checkpoints()?;
            checkpoints
                .iter()
                .find(|c| c.id.starts_with(&id_prefix))
                .map(|c| c.id.clone())
                .ok_or_else(|| TitorError::internal(format!("Checkpoint not found: {}", id_prefix)))?
        };

        println!("{} {}", 
            "Verifying checkpoint".blue().bold(),
            full_id[..8.min(full_id.len())].yellow()
        );

        let report = titor.verify_checkpoint(&full_id)?;
        
        println!("\n{}", "Verification Report:".bold());
        println!("  Metadata: {}", 
            if report.metadata_valid { "✓ Valid".green() } else { "✗ Invalid".red() }
        );
        println!("  State hash: {}", 
            if report.state_hash_valid { "✓ Valid".green() } else { "✗ Invalid".red() }
        );
        println!("  Merkle root: {}", 
            if report.merkle_root_valid { "✓ Valid".green() } else { "✗ Invalid".red() }
        );
        println!("  Parent chain: {}", 
            if report.parent_valid { "✓ Valid".green() } else { "✗ Invalid".red() }
        );
        
        let valid_files = report.file_checks.iter()
            .filter(|f| f.content_hash_valid && f.metadata_hash_valid)
            .count();
        println!("  Files: {}/{} valid", 
            valid_files.to_string().green(),
            report.file_checks.len()
        );
        
        if !report.orphaned_objects.is_empty() {
            println!("\n{} Found {} orphaned objects", 
                "⚠".yellow().bold(),
                report.orphaned_objects.len()
            );
        }
    }

    Ok(())
}

/// Garbage collect unreferenced objects
///
/// Safely removes objects that are no longer referenced by any checkpoint.
/// This helps reclaim disk space while maintaining checkpoint integrity.
fn cmd_gc(
    root_path: PathBuf,
    storage_path: PathBuf,
    dry_run: bool,
) -> Result<()> {
    let titor = open_titor(root_path, storage_path)?;
    let start = Instant::now();
    
    if dry_run {
        println!("{}", "Analyzing garbage collection (dry run)...".blue().bold());
        let stats = titor.gc_analyze()?;
        
        println!("\n{}", "Analysis Results:".bold());
        println!("  Objects examined: {}", stats.objects_examined);
        println!("  Unreferenced objects: {}", stats.unreferenced_objects.len().to_string().yellow());
        println!("  Space to reclaim: {}", format_bytes(stats.bytes_reclaimed).green());
        
        if !stats.unreferenced_objects.is_empty() {
            println!("\n{}", "Unreferenced objects:".yellow());
            for hash in stats.unreferenced_objects.iter().take(10) {
                println!("  - {}", hash[..16].dimmed());
            }
            if stats.unreferenced_objects.len() > 10 {
                println!("  ... and {} more", (stats.unreferenced_objects.len() - 10).to_string().dimmed());
            }
        }
        
        println!("\n{}", "No changes made (dry run)".dimmed());
    } else {
        println!("{}", "Running garbage collection...".blue().bold());
        let stats = titor.gc()?;
        
        println!("\n{} Garbage collection complete", "✓".green().bold());
        println!("  Objects deleted: {}", stats.objects_deleted.to_string().green());
        println!("  Space reclaimed: {}", format_bytes(stats.bytes_reclaimed).green());
        println!("  Time: {}", format_duration(std::time::Duration::from_millis(stats.duration_ms)).to_string().cyan());
    }
    
    let elapsed = start.elapsed();
    println!("\n{}", format!("Total time: {}", format_duration(elapsed)).dimmed());
    
    Ok(())
}

/// Show current status
///
/// Displays:
/// - Current checkpoint information
/// - Changes since last checkpoint
/// - Storage statistics
fn cmd_status(root_path: PathBuf, storage_path: PathBuf) -> Result<()> {
    let titor = open_titor(root_path.clone(), storage_path.clone())?;
    let timeline = titor.get_timeline()?;
    
    println!("{}", "Titor Status:".blue().bold());
    println!();
    
    // Current checkpoint
    if let Some(current_id) = &timeline.current_checkpoint_id {
        let checkpoint = timeline.checkpoints.get(current_id)
            .ok_or_else(|| TitorError::internal("Current checkpoint not found"))?;
        
        println!("{}", "Current checkpoint:".bold());
        println!("  ID: {}", checkpoint.id[..8].yellow());
        println!("  Created: {}", checkpoint.timestamp.format("%Y-%m-%d %H:%M:%S"));
        if let Some(desc) = &checkpoint.description {
            println!("  Message: {}", desc.cyan());
        }
        println!("  Files: {}", checkpoint.metadata.file_count);
        println!("  Size: {}", format_bytes(checkpoint.metadata.total_size));
    } else {
        println!("{}", "No current checkpoint".yellow());
    }
    
    // Timeline stats
    println!("\n{}", "Timeline:".bold());
    println!("  Total checkpoints: {}", timeline.checkpoints.len());
    
    // Storage stats
    let storage_info = get_storage_info(&storage_path)?;
    println!("\n{}", "Storage:".bold());
    println!("  Location: {}", storage_path.display());
    println!("  Size: {}", format_bytes(storage_info.total_size));
    println!("  Objects: {}", storage_info.object_count);

    Ok(())
}

/// Show detailed checkpoint information
///
/// Displays comprehensive information about a checkpoint including:
/// - Metadata and timestamps
/// - File statistics
/// - Parent/child relationships
/// - Verification status
fn cmd_info(
    root_path: PathBuf,
    storage_path: PathBuf,
    checkpoint_id: String,
) -> Result<()> {
    let titor = open_titor(root_path, storage_path)?;
    let checkpoints = titor.list_checkpoints()?;
    
    let checkpoint = checkpoints.iter()
        .find(|c| c.id.starts_with(&checkpoint_id))
        .ok_or_else(|| TitorError::internal("Checkpoint not found"))?;
    
    println!("{} {}", 
        "Checkpoint".blue().bold(),
        checkpoint.id[..8].yellow().bold()
    );
    println!();
    
    // Basic info
    println!("{}", "Basic Information:".bold());
    println!("  Full ID: {}", checkpoint.id.dimmed());
    println!("  Created: {}", checkpoint.timestamp.format("%Y-%m-%d %H:%M:%S UTC"));
    if let Some(desc) = &checkpoint.description {
        println!("  Message: {}", desc.cyan());
    }
    
    // Relationships
    println!("\n{}", "Relationships:".bold());
    if let Some(parent) = &checkpoint.parent_id {
        println!("  Parent: {}", parent[..8].yellow());
    } else {
        println!("  Parent: {}", "None (root checkpoint)".dimmed());
    }
    
    // Find children
    let children: Vec<_> = checkpoints.iter()
        .filter(|c| c.parent_id.as_ref() == Some(&checkpoint.id))
        .collect();
    
    if !children.is_empty() {
        print!("  Children: ");
        for (i, child) in children.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{}", child.id[..8].yellow());
        }
        println!();
    }
    
    // Statistics
    println!("\n{}", "Statistics:".bold());
    println!("  Files: {}", checkpoint.metadata.file_count.to_string().cyan());
    println!("  Total size: {}", format_bytes(checkpoint.metadata.total_size).cyan());
    println!("  Compressed: {}", format_bytes(checkpoint.metadata.compressed_size).cyan());
    let ratio = if checkpoint.metadata.total_size > 0 {
        (checkpoint.metadata.compressed_size as f64 / checkpoint.metadata.total_size as f64) * 100.0
    } else {
        0.0
    };
    println!("  Compression ratio: {:.1}%", ratio);
    
    if checkpoint.metadata.files_changed > 0 {
        println!("  Changed from parent: {} files", checkpoint.metadata.files_changed);
    }
    
    // Technical details
    println!("\n{}", "Technical Details:".bold());
    println!("  State hash: {}", checkpoint.state_hash[..16].dimmed());
    println!("  Merkle root: {}", checkpoint.content_merkle_root[..16].dimmed());
    println!("  Host: {}", checkpoint.metadata.host_info.hostname);
    println!("  Titor version: {}", checkpoint.metadata.titor_version);
    
    // Quick verify
    println!("\n{}", "Verification:".bold());
    if checkpoint.verify_integrity()? {
        println!("  Integrity: {}", "✓ Valid".green());
    } else {
        println!("  Integrity: {}", "✗ Invalid".red());
    }

    Ok(())
}

// Helper functions

/// Open existing Titor instance
fn open_titor(root_path: PathBuf, storage_path: PathBuf) -> Result<Titor> {
    if !storage_path.exists() {
        return Err(TitorError::internal(
            "Not a Titor repository. Run 'titor init' first."
        ));
    }
    
    Titor::open(root_path, storage_path)
}

/// Format bytes in human-readable form
fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    if unit_idx == 0 {
        format!("{} {}", size as u64, UNITS[unit_idx])
    } else {
        format!("{:.2} {}", size, UNITS[unit_idx])
    }
}

/// Storage information
struct StorageInfo {
    total_size: u64,
    object_count: usize,
}

/// Get storage information
fn get_storage_info(storage_path: &Path) -> Result<StorageInfo> {
    let mut total_size = 0u64;
    let mut object_count = 0usize;
    
    // Walk storage directory
    for entry in walkdir::WalkDir::new(storage_path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
            }
            
            if entry.path().parent()
                .and_then(|p| p.file_name())
                .map(|n| n == "objects")
                .unwrap_or(false)
            {
                object_count += 1;
            }
        }
    }
    
    Ok(StorageInfo {
        total_size,
        object_count,
    })
} 