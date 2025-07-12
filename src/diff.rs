//! Line-level diff computation for text files
//!
//! This module provides functionality to compute line-by-line differences
//! between text files, similar to git's unified diff format.
//!
//! ## Overview
//!
//! The diff algorithm uses a dynamic programming approach to find the
//! Longest Common Subsequence (LCS) between two files, then generates
//! a unified diff format with configurable context lines.
//!
//! ## Features
//!
//! - **LCS-based Algorithm**: Efficient O(mn) algorithm for finding differences
//! - **Unified Diff Format**: Compatible with standard diff tools
//! - **Binary Detection**: Automatically detects and handles binary files
//! - **Configurable Context**: Control how many unchanged lines to show
//! - **Whitespace Handling**: Optional whitespace-ignoring comparison
//! - **Memory Efficient**: Processes files line-by-line
//!
//! ## Examples
//!
//! ```rust,no_run
//! use titor::diff::{compute_line_diff, create_file_diff};
//! use titor::types::DiffOptions;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure diff options
//! let options = DiffOptions {
//!     context_lines: 3,
//!     ignore_whitespace: false,
//!     show_line_numbers: true,
//!     max_file_size: 10 * 1024 * 1024, // 10MB
//! };
//!
//! // Compute diff between two text contents
//! let old_content = b"line1\nline2\nline3";
//! let new_content = b"line1\nmodified line2\nline3\nline4";
//!
//! let hunks = compute_line_diff(old_content, new_content, &options)?;
//!
//! // Or create a complete file diff
//! let file_diff = create_file_diff(
//!     Path::new("file.txt"),
//!     "old_hash",
//!     "new_hash", 
//!     old_content,
//!     new_content,
//!     &options
//! )?;
//!
//! println!("Lines added: {}, deleted: {}", 
//!     file_diff.lines_added, 
//!     file_diff.lines_deleted
//! );
//! # Ok(())
//! # }
//! ```

use crate::error::Result;
use crate::types::{DiffHunk, FileDiff, LineChange, DiffOptions};
use std::path::Path;

/// Compute line-level diff between two text contents
///
/// This function implements a simple line-based diff algorithm that:
/// 1. Splits both texts into lines
/// 2. Finds the longest common subsequence (LCS)
/// 3. Generates hunks with context lines
///
/// # Arguments
///
/// * `old_content` - Content from the source version
/// * `new_content` - Content from the target version
/// * `options` - Options controlling diff generation
///
/// # Returns
///
/// Returns a vector of `DiffHunk` representing the changes
pub fn compute_line_diff(
    old_content: &[u8],
    new_content: &[u8],
    options: &DiffOptions,
) -> Result<Vec<DiffHunk>> {
    // Convert to strings, handling potential UTF-8 errors
    let old_text = String::from_utf8_lossy(old_content);
    let new_text = String::from_utf8_lossy(new_content);
    
    // Split into lines
    let old_lines: Vec<&str> = old_text.lines().collect();
    let new_lines: Vec<&str> = new_text.lines().collect();
    
    // Handle empty files
    if old_lines.is_empty() && new_lines.is_empty() {
        return Ok(vec![]);
    }
    
    // Compute the diff
    let changes = compute_changes(&old_lines, &new_lines, options);
    
    // Group changes into hunks
    let hunks = create_hunks(changes, &old_lines, &new_lines, options.context_lines);
    
    Ok(hunks)
}

/// Represents a change operation in the diff
#[derive(Debug, Clone)]
enum ChangeOp {
    Keep(usize, usize),    // (old_line_idx, new_line_idx)
    Delete(usize),         // old_line_idx
    Insert(usize),         // new_line_idx
}

/// Compute the sequence of change operations
fn compute_changes(
    old_lines: &[&str],
    new_lines: &[&str],
    options: &DiffOptions,
) -> Vec<ChangeOp> {
    // Special cases
    if old_lines.is_empty() {
        return new_lines.iter().enumerate()
            .map(|(i, _)| ChangeOp::Insert(i))
            .collect();
    }
    if new_lines.is_empty() {
        return old_lines.iter().enumerate()
            .map(|(i, _)| ChangeOp::Delete(i))
            .collect();
    }
    
    // Use a simple dynamic programming approach for LCS
    let lcs = compute_lcs(old_lines, new_lines, options.ignore_whitespace);
    
    // Convert LCS to change operations
    lcs_to_changes(&lcs, old_lines.len(), new_lines.len())
}

/// Compute longest common subsequence using dynamic programming
fn compute_lcs(
    old_lines: &[&str],
    new_lines: &[&str],
    ignore_whitespace: bool,
) -> Vec<(usize, usize)> {
    let m = old_lines.len();
    let n = new_lines.len();
    
    // Build DP table
    let mut dp = vec![vec![0; n + 1]; m + 1];
    
    for i in 1..=m {
        for j in 1..=n {
            if lines_equal(old_lines[i-1], new_lines[j-1], ignore_whitespace) {
                dp[i][j] = dp[i-1][j-1] + 1;
            } else {
                dp[i][j] = dp[i-1][j].max(dp[i][j-1]);
            }
        }
    }
    
    // Backtrack to find LCS
    let mut lcs = Vec::new();
    let mut i = m;
    let mut j = n;
    
    while i > 0 && j > 0 {
        if lines_equal(old_lines[i-1], new_lines[j-1], ignore_whitespace) {
            lcs.push((i-1, j-1));
            i -= 1;
            j -= 1;
        } else if dp[i-1][j] > dp[i][j-1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    
    lcs.reverse();
    lcs
}

/// Check if two lines are equal, optionally ignoring whitespace
fn lines_equal(a: &str, b: &str, ignore_whitespace: bool) -> bool {
    if ignore_whitespace {
        a.trim() == b.trim()
    } else {
        a == b
    }
}

/// Convert LCS to a sequence of change operations
fn lcs_to_changes(lcs: &[(usize, usize)], old_len: usize, new_len: usize) -> Vec<ChangeOp> {
    let mut changes = Vec::new();
    let mut old_idx = 0;
    let mut new_idx = 0;
    let mut lcs_idx = 0;
    
    while old_idx < old_len || new_idx < new_len {
        if lcs_idx < lcs.len() {
            let (lcs_old, lcs_new) = lcs[lcs_idx];
            
            // Process deletions before the next match
            while old_idx < lcs_old {
                changes.push(ChangeOp::Delete(old_idx));
                old_idx += 1;
            }
            
            // Process insertions before the next match
            while new_idx < lcs_new {
                changes.push(ChangeOp::Insert(new_idx));
                new_idx += 1;
            }
            
            // Keep the matching line
            if old_idx == lcs_old && new_idx == lcs_new {
                changes.push(ChangeOp::Keep(old_idx, new_idx));
                old_idx += 1;
                new_idx += 1;
                lcs_idx += 1;
            }
        } else {
            // No more matches, process remaining lines
            while old_idx < old_len {
                changes.push(ChangeOp::Delete(old_idx));
                old_idx += 1;
            }
            while new_idx < new_len {
                changes.push(ChangeOp::Insert(new_idx));
                new_idx += 1;
            }
        }
    }
    
    changes
}

/// Create diff hunks from change operations
fn create_hunks(
    changes: Vec<ChangeOp>,
    old_lines: &[&str],
    new_lines: &[&str],
    context_lines: usize,
) -> Vec<DiffHunk> {
    if changes.is_empty() {
        return vec![];
    }
    
    let mut hunks = Vec::new();
    let mut current_hunk: Option<HunkBuilder> = None;
    
    for (i, change) in changes.iter().enumerate() {
        match change {
            ChangeOp::Keep(old_idx, new_idx) => {
                // Check if this is within context of a change
                let near_change = is_near_change(&changes, i, context_lines);
                
                if near_change {
                    // Include as context
                    if let Some(ref mut hunk) = current_hunk {
                        hunk.add_context(*old_idx, *new_idx, old_lines[*old_idx]);
                    } else {
                        // Start new hunk with context
                        let mut hunk = HunkBuilder::new();
                        
                        // Add leading context
                        let start = i.saturating_sub(context_lines);
                        for j in start..i {
                            if let ChangeOp::Keep(o, n) = &changes[j] {
                                hunk.add_context(*o, *n, old_lines[*o]);
                            }
                        }
                        
                        hunk.add_context(*old_idx, *new_idx, old_lines[*old_idx]);
                        current_hunk = Some(hunk);
                    }
                } else if let Some(hunk) = current_hunk.take() {
                    // End current hunk
                    if let Some(built) = hunk.build() {
                        hunks.push(built);
                    }
                }
            }
            ChangeOp::Delete(old_idx) => {
                if current_hunk.is_none() {
                    let mut hunk = HunkBuilder::new();
                    
                    // Add leading context
                    let start = i.saturating_sub(context_lines);
                    for j in start..i {
                        if let ChangeOp::Keep(o, n) = &changes[j] {
                            hunk.add_context(*o, *n, old_lines[*o]);
                        }
                    }
                    
                    current_hunk = Some(hunk);
                }
                
                if let Some(ref mut hunk) = current_hunk {
                    hunk.add_deletion(*old_idx, old_lines[*old_idx]);
                }
            }
            ChangeOp::Insert(new_idx) => {
                if current_hunk.is_none() {
                    let mut hunk = HunkBuilder::new();
                    
                    // Add leading context
                    let start = i.saturating_sub(context_lines);
                    for j in start..i {
                        if let ChangeOp::Keep(o, n) = &changes[j] {
                            hunk.add_context(*o, *n, old_lines[*o]);
                        }
                    }
                    
                    current_hunk = Some(hunk);
                }
                
                if let Some(ref mut hunk) = current_hunk {
                    hunk.add_insertion(*new_idx, new_lines[*new_idx]);
                }
            }
        }
    }
    
    // Finish last hunk
    if let Some(hunk) = current_hunk {
        if let Some(built) = hunk.build() {
            hunks.push(built);
        }
    }
    
    hunks
}

/// Check if a position is near a change (within context_lines)
fn is_near_change(changes: &[ChangeOp], pos: usize, context_lines: usize) -> bool {
    let start = pos.saturating_sub(context_lines);
    let end = (pos + context_lines + 1).min(changes.len());
    
    for i in start..end {
        if i == pos {
            continue;
        }
        match &changes[i] {
            ChangeOp::Delete(_) | ChangeOp::Insert(_) => return true,
            ChangeOp::Keep(_, _) => continue,
        }
    }
    
    false
}

/// Helper for building diff hunks
struct HunkBuilder {
    from_start: Option<usize>,
    to_start: Option<usize>,
    from_count: usize,
    to_count: usize,
    changes: Vec<LineChange>,
}

impl HunkBuilder {
    fn new() -> Self {
        Self {
            from_start: None,
            to_start: None,
            from_count: 0,
            to_count: 0,
            changes: Vec::new(),
        }
    }
    
    fn add_context(&mut self, old_idx: usize, new_idx: usize, content: &str) {
        if self.from_start.is_none() {
            self.from_start = Some(old_idx + 1); // 1-based
            self.to_start = Some(new_idx + 1);
        }
        self.from_count += 1;
        self.to_count += 1;
        self.changes.push(LineChange::Context(old_idx + 1, content.to_string()));
    }
    
    fn add_deletion(&mut self, old_idx: usize, content: &str) {
        if self.from_start.is_none() {
            self.from_start = Some(old_idx + 1);
            self.to_start = Some(1); // Will be adjusted
        }
        self.from_count += 1;
        self.changes.push(LineChange::Deleted(old_idx + 1, content.to_string()));
    }
    
    fn add_insertion(&mut self, new_idx: usize, content: &str) {
        if self.to_start.is_none() {
            self.from_start = Some(1); // Will be adjusted
            self.to_start = Some(new_idx + 1);
        }
        self.to_count += 1;
        self.changes.push(LineChange::Added(new_idx + 1, content.to_string()));
    }
    
    fn build(self) -> Option<DiffHunk> {
        if self.changes.is_empty() {
            return None;
        }
        
        Some(DiffHunk {
            from_line: self.from_start.unwrap_or(1),
            from_count: self.from_count,
            to_line: self.to_start.unwrap_or(1),
            to_count: self.to_count,
            changes: self.changes,
        })
    }
}

/// Check if content appears to be binary
pub fn is_binary_content(content: &[u8]) -> bool {
    // Simple heuristic: check for null bytes in first 8KB
    let check_len = content.len().min(8192);
    content[..check_len].contains(&0)
}

/// Create a FileDiff from file contents
pub fn create_file_diff(
    path: &Path,
    from_hash: &str,
    to_hash: &str,
    old_content: &[u8],
    new_content: &[u8],
    options: &DiffOptions,
) -> Result<FileDiff> {
    // Check if binary
    let is_binary = is_binary_content(old_content) || is_binary_content(new_content);
    
    let (hunks, lines_added, lines_deleted) = if is_binary {
        (vec![], 0, 0)
    } else {
        let hunks = compute_line_diff(old_content, new_content, options)?;
        
        // Count line changes
        let mut added = 0;
        let mut deleted = 0;
        for hunk in &hunks {
            for change in &hunk.changes {
                match change {
                    LineChange::Added(_, _) => added += 1,
                    LineChange::Deleted(_, _) => deleted += 1,
                    LineChange::Context(_, _) => {}
                }
            }
        }
        
        (hunks, added, deleted)
    };
    
    Ok(FileDiff {
        path: path.to_path_buf(),
        from_hash: from_hash.to_string(),
        to_hash: to_hash.to_string(),
        is_binary,
        hunks,
        lines_added,
        lines_deleted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_diff() {
        let old = b"line1\nline2\nline3";
        let new = b"line1\nline2 modified\nline3\nline4";
        
        let options = DiffOptions::default();
        let hunks = compute_line_diff(old, new, &options).unwrap();
        
        assert_eq!(hunks.len(), 1);
        let hunk = &hunks[0];
        
        // Should have context, deletion, additions
        assert!(hunk.changes.iter().any(|c| matches!(c, LineChange::Deleted(_, _))));
        assert!(hunk.changes.iter().any(|c| matches!(c, LineChange::Added(_, _))));
    }
    
    #[test]
    fn test_empty_files() {
        let options = DiffOptions::default();
        
        // Both empty
        let hunks = compute_line_diff(b"", b"", &options).unwrap();
        assert_eq!(hunks.len(), 0);
        
        // Old empty
        let hunks = compute_line_diff(b"", b"new line", &options).unwrap();
        assert_eq!(hunks.len(), 1);
        assert!(hunks[0].changes.iter().all(|c| matches!(c, LineChange::Added(_, _))));
        
        // New empty
        let hunks = compute_line_diff(b"old line", b"", &options).unwrap();
        assert_eq!(hunks.len(), 1);
        assert!(hunks[0].changes.iter().all(|c| matches!(c, LineChange::Deleted(_, _))));
    }
    
    #[test]
    fn test_binary_detection() {
        assert!(is_binary_content(b"hello\x00world"));
        assert!(!is_binary_content(b"hello world"));
    }
    
    #[test]
    fn test_context_lines() {
        let old = b"1\n2\n3\n4\n5\n6\n7\n8\n9";
        let new = b"1\n2\n3\nMODIFIED\n5\n6\n7\n8\n9";
        
        let mut options = DiffOptions::default();
        options.context_lines = 2;
        
        let hunks = compute_line_diff(old, new, &options).unwrap();
        assert_eq!(hunks.len(), 1);
        
        // Check that we have context lines
        let context_count = hunks[0].changes.iter()
            .filter(|c| matches!(c, LineChange::Context(_, _)))
            .count();
        assert!(context_count >= 4); // At least 2 before and 2 after
    }
} 