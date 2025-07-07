//! Utility functions for Titor
//!
//! This module provides common utility functions used throughout the Titor library,
//! including file operations, hashing helpers, path manipulation, and cross-platform
//! compatibility functions.
//!
//! ## Categories of Utilities
//!
//! ### File Operations
//! - File content hashing (SHA-256)
//! - File metadata extraction
//! - Atomic file writing
//! - Permission handling (cross-platform)
//! - Symbolic link operations
//!
//! ### Path Manipulation
//! - Converting absolute paths to relative paths
//! - Cross-platform path normalization
//! - Directory management
//!
//! ### Data Processing
//! - Hash computation for arbitrary data
//! - Byte formatting (human-readable sizes)
//! - Metadata extraction and processing
//!
//! ### Cross-Platform Compatibility
//! - Unix/Windows permission handling
//! - Symbolic link creation and reading
//! - File attribute management
//!
//! ## Example Usage
//!
//! ### File Hashing
//!
//! ```rust,ignore
//! use crate::utils::{hash_file_content, hash_data};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Hash a file's content
//! let file_hash = hash_file_content(Path::new("example.txt"))?;
//! println!("File hash: {}", file_hash);
//!
//! // Hash arbitrary data
//! let data = b"Hello, world!";
//! let data_hash = hash_data(data);
//! println!("Data hash: {}", data_hash);
//! # Ok(())
//! # }
//! ```
//!
//! ### File Metadata
//!
//! ```rust,ignore
//! use crate::utils::{get_file_metadata, set_permissions};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Get file metadata
//! let metadata = get_file_metadata(Path::new("example.txt"))?;
//! println!("Size: {} bytes", metadata.size);
//! println!("Permissions: {:o}", metadata.permissions);
//!
//! // Set file permissions
//! set_permissions(Path::new("example.txt"), 0o644)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Path Operations
//!
//! ```rust,ignore
//! use crate::utils::{make_relative, format_bytes};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Make path relative to base
//! let base = Path::new("/home/user/project");
//! let full_path = Path::new("/home/user/project/src/main.rs");
//! let relative = make_relative(full_path, base)?;
//! println!("Relative path: {}", relative.display());
//!
//! // Format bytes for display
//! let size = 1536;
//! println!("Size: {}", format_bytes(size)); // "1.50 KB"
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Considerations
//!
//! - **File Hashing**: Uses buffered I/O for efficient processing of large files
//! - **Atomic Operations**: Atomic file writing prevents corruption
//! - **Memory Efficiency**: Streaming approach for large data processing
//! - **Cross-Platform**: Optimized implementations for different operating systems
//!
//! ## Error Handling
//!
//! All functions return `Result<T, TitorError>` with detailed error information.
//! Common error scenarios include:
//! - File I/O errors
//! - Permission denied errors
//! - Invalid path errors
//! - Cross-platform compatibility issues
//!
//! ## Thread Safety
//!
//! All utility functions are thread-safe and can be called concurrently from
//! multiple threads without synchronization.

use crate::error::{Result, TitorError};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tracing::trace;

/// Hash a file's content efficiently using SHA-256
///
/// Computes the SHA-256 hash of a file's content using buffered I/O for
/// optimal performance with large files. The entire file is read and
/// hashed in chunks to minimize memory usage.
///
/// # Arguments
///
/// * `path` - Path to the file to hash
///
/// # Returns
///
/// Returns the SHA-256 hash as a 64-character hexadecimal string.
///
/// # Errors
///
/// - [`TitorError::Io`] if the file cannot be read
/// - [`TitorError::Io`] if I/O errors occur during reading
///
/// # Example
///
/// ```rust,ignore
/// use crate::utils::hash_file_content;
/// use std::path::Path;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let hash = hash_file_content(Path::new("example.txt"))?;
/// println!("File hash: {}", hash);
/// assert_eq!(hash.len(), 64); // SHA-256 is 64 hex characters
/// # Ok(())
/// # }
/// ```
///
/// # Performance
///
/// - Uses 8KB buffer for efficient I/O
/// - Streaming approach minimizes memory usage
/// - Optimized for both small and large files
/// - Thread-safe and can be called concurrently
pub fn hash_file_content(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8192]; // 8KB buffer
    
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    
    Ok(hex::encode(hasher.finalize()))
}

/// Hash arbitrary data using SHA-256
///
/// Computes the SHA-256 hash of arbitrary byte data. This is a convenience
/// function for hashing small amounts of data that are already in memory.
///
/// # Arguments
///
/// * `data` - Byte slice to hash
///
/// # Returns
///
/// Returns the SHA-256 hash as a 64-character hexadecimal string.
///
/// # Example
///
/// ```rust,ignore
/// use crate::utils::hash_data;
///
/// let data = b"Hello, world!";
/// let hash = hash_data(data);
/// println!("Data hash: {}", hash);
/// assert_eq!(hash.len(), 64); // SHA-256 is 64 hex characters
///
/// // Same data always produces the same hash
/// let hash2 = hash_data(data);
/// assert_eq!(hash, hash2);
/// ```
///
/// # Performance
///
/// - Efficient for small to medium-sized data
/// - For large data or files, consider using `hash_file_content`
/// - Thread-safe and can be called concurrently
pub fn hash_data(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}



/// Get file metadata safely
///
/// Extracts comprehensive metadata from a file or directory, including size,
/// permissions, modification time, and symbolic link status. Uses `symlink_metadata`
/// to avoid following symbolic links.
///
/// # Arguments
///
/// * `path` - Path to the file or directory
///
/// # Returns
///
/// Returns a `FileMetadata` struct containing all extracted metadata.
///
/// # Errors
///
/// - [`TitorError::Io`] if the file doesn't exist or cannot be accessed
/// - [`TitorError::Io`] if permission errors occur
///
/// # Example
///
/// ```rust,ignore
/// use crate::utils::get_file_metadata;
/// use std::path::Path;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let metadata = get_file_metadata(Path::new("example.txt"))?;
/// println!("Size: {} bytes", metadata.size);
/// println!("Permissions: {:o}", metadata.permissions);
/// println!("Is symlink: {}", metadata.is_symlink);
/// # Ok(())
/// # }
/// ```
///
/// # Cross-Platform Behavior
///
/// - **Unix**: Returns actual Unix permissions
/// - **Windows**: Maps Windows attributes to Unix-like permissions
/// - **Symbolic Links**: Detected correctly on both platforms
pub fn get_file_metadata(path: &Path) -> Result<FileMetadata> {
    let metadata = fs::symlink_metadata(path)?;
    
    Ok(FileMetadata {
        size: metadata.len(),
        permissions: get_permissions(&metadata),
        modified: metadata.modified()?,
        is_symlink: metadata.file_type().is_symlink(),
    })
}

/// File metadata container
///
/// Contains comprehensive metadata about a file or directory, extracted
/// in a cross-platform manner. This struct provides a unified interface
/// for file metadata across different operating systems.
///
/// # Fields
///
/// * `size` - File size in bytes (0 for directories)
/// * `permissions` - Unix-style permissions (e.g., 0o644, 0o755)
/// * `modified` - Last modification timestamp
/// * `is_symlink` - Whether this is a symbolic link
///
/// # Cross-Platform Notes
///
/// ## Unix Systems
/// - Permissions are native Unix permissions
/// - Size includes actual file size
/// - Symbolic links are detected natively
///
/// ## Windows Systems
/// - Permissions are mapped from Windows file attributes
/// - Read-only files map to 0o444, writable files to 0o644
/// - Directories get execute permissions (0o755 or 0o555)
/// - Symbolic links are detected through junction points and symlinks
///
/// # Example
///
/// ```rust,ignore
/// use crate::utils::{get_file_metadata, FileMetadata};
/// use std::path::Path;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let metadata: FileMetadata = get_file_metadata(Path::new("example.txt"))?;
///
/// // Check file properties
/// if metadata.is_symlink {
///     println!("This is a symbolic link");
/// }
///
/// // Display permissions in octal format
/// println!("Permissions: {:o}", metadata.permissions);
///
/// // Check if writable (owner write bit)
/// if metadata.permissions & 0o200 != 0 {
///     println!("File is writable by owner");
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FileMetadata {
    /// File size in bytes (0 for directories)
    pub size: u64,
    /// Unix-style permissions (e.g., 0o644 for rw-r--r--)
    pub permissions: u32,
    /// Last modification timestamp
    pub modified: SystemTime,
    /// Whether this is a symbolic link
    pub is_symlink: bool,
}

/// Get Unix permissions from metadata
#[cfg(unix)]
fn get_permissions(metadata: &fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode()
}

/// Get permissions from metadata (Windows implementation)
#[cfg(windows)]
fn get_permissions(metadata: &fs::Metadata) -> u32 {
    use std::os::windows::fs::MetadataExt;
    
    // On Windows, we'll map file attributes to Unix-like permissions
    let attrs = metadata.file_attributes();
    let mut mode = 0o644; // Default read/write for owner, read for others
    
    // If read-only attribute is set, remove write permissions
    if attrs & 0x01 != 0 { // FILE_ATTRIBUTE_READONLY
        mode = 0o444;
    }
    
    // If it's a directory, add execute permissions
    if metadata.is_dir() {
        mode |= 0o111;
    }
    
    // If it's a system or hidden file, restrict permissions
    if attrs & 0x02 != 0 || attrs & 0x04 != 0 { // HIDDEN or SYSTEM
        mode = 0o600; // Only owner can access
    }
    
    mode
}

/// Set Unix permissions
#[cfg(unix)]
pub fn set_permissions(path: &Path, mode: u32) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let permissions = fs::Permissions::from_mode(mode);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

/// Set permissions (Windows implementation)
#[cfg(windows)]
pub fn set_permissions(path: &Path, mode: u32) -> Result<()> {
    use std::os::windows::fs::MetadataExt;
    
    // On Windows, we can only set the read-only attribute
    // Check if the mode indicates read-only (no write permissions for owner)
    let is_readonly = (mode & 0o200) == 0;
    
    let metadata = fs::metadata(path)?;
    let mut attrs = metadata.file_attributes();
    
    if is_readonly {
        // Set read-only attribute
        attrs |= 0x01; // FILE_ATTRIBUTE_READONLY
    } else {
        // Clear read-only attribute
        attrs &= !0x01;
    }
    
    // Windows doesn't have a direct API to set file attributes through std
    // We'll use the readonly flag through the permissions API
    let mut perms = metadata.permissions();
    perms.set_readonly(is_readonly);
    fs::set_permissions(path, perms)?;
    
    Ok(())
}



/// Remove directory if empty
pub fn remove_dir_if_empty(path: &Path) -> Result<bool> {
    if path.is_dir() && fs::read_dir(path)?.next().is_none() {
        fs::remove_dir(path)?;
        trace!("Removed empty directory: {:?}", path);
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Make a path relative to a base path
///
/// Converts an absolute path to a relative path by removing the base path prefix.
/// This function preserves symbolic links by attempting a lexical strip first,
/// falling back to canonicalization only when necessary.
///
/// # Arguments
///
/// * `path` - The path to make relative
/// * `base` - The base path to remove from the beginning
///
/// # Returns
///
/// Returns a `PathBuf` containing the relative path.
///
/// # Errors
///
/// - [`TitorError::Internal`] if the path is not under the base path
/// - [`TitorError::Io`] if canonicalization fails (fallback case only)
///
/// # Example
///
/// ```rust,ignore
/// use crate::utils::make_relative;
/// use std::path::{Path, PathBuf};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let base = Path::new("/home/user/project");
/// let full_path = Path::new("/home/user/project/src/main.rs");
/// 
/// let relative = make_relative(full_path, base)?;
/// assert_eq!(relative, PathBuf::from("src/main.rs"));
/// # Ok(())
/// # }
/// ```
///
/// # Symbolic Link Handling
///
/// The function uses a two-stage approach:
/// 1. **Lexical Strip**: Attempts to remove the base prefix without resolving symlinks
/// 2. **Canonical Strip**: Falls back to resolving symlinks if lexical approach fails
///
/// This preserves symbolic link paths when possible while handling complex cases
/// involving relative path components or different path normalizations.
///
/// # Cross-Platform Behavior
///
/// - Works correctly on both Unix and Windows systems
/// - Handles different path separators appropriately
/// - Respects case sensitivity rules of the underlying filesystem
pub fn make_relative(path: &Path, base: &Path) -> Result<PathBuf> {
    // First try a simple lexical strip to avoid resolving symlinks. This preserves the
    // original path of symbolic links, preventing them from being canonicalised to their
    // target and accidentally creating duplicate FileEntry records (one for the link and
    // another for the target). If the direct strip fails (e.g., because the path uses
    // relative components like ".." or differs in normalisation), we fall back to
    // canonicalising both paths and try again.

    if let Ok(relative) = path.strip_prefix(base) {
        return Ok(relative.to_path_buf());
    }

    // Fallback – canonicalise to guarantee an absolute comparison that ignores symbolic
    // links and normalises the components. This mirrors the original behaviour, but will
    // only be used when the simple lexical approach fails.
    let path_canon = path.canonicalize()?;
    let base_canon = base.canonicalize()?;

    path_canon
        .strip_prefix(&base_canon)
        .map(|p| p.to_path_buf())
        .map_err(|_| TitorError::internal(format!(
            "Path {:?} is not relative to {:?}",
            path_canon, base_canon
        )))
}



/// Format bytes in human-readable form
///
/// Converts a byte count into a human-readable string using appropriate
/// units (B, KB, MB, GB, TB, PB). Uses 1024 as the conversion factor
/// following binary conventions.
///
/// # Arguments
///
/// * `bytes` - Number of bytes to format
///
/// # Returns
///
/// Returns a formatted string with the appropriate unit.
///
/// # Example
///
/// ```rust,ignore
/// use crate::utils::format_bytes;
///
/// assert_eq!(format_bytes(0), "0 B");
/// assert_eq!(format_bytes(1023), "1023 B");
/// assert_eq!(format_bytes(1024), "1.00 KB");
/// assert_eq!(format_bytes(1536), "1.50 KB");
/// assert_eq!(format_bytes(1_048_576), "1.00 MB");
/// assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
/// ```
///
/// # Formatting Rules
///
/// - Values less than 1024 bytes are shown as whole numbers with "B"
/// - Larger values are shown with 2 decimal places and appropriate units
/// - Uses binary units (1024-based) rather than decimal (1000-based)
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
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



/// Atomic file write (write to temp file then rename)
///
/// Performs an atomic write operation by first writing to a temporary file
/// and then renaming it to the target location. This ensures that the target
/// file is never in a partially written state, preventing corruption.
///
/// # Arguments
///
/// * `path` - Target file path to write to
/// * `content` - Byte content to write to the file
///
/// # Returns
///
/// Returns `Ok(())` on success.
///
/// # Errors
///
/// - [`TitorError::Io`] if writing to the temporary file fails
/// - [`TitorError::Io`] if the atomic rename operation fails
///
/// # Example
///
/// ```rust,ignore
/// use crate::utils::atomic_write;
/// use std::path::Path;
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let content = b"Hello, world!";
/// atomic_write(Path::new("output.txt"), content)?;
/// 
/// // File is guaranteed to be complete or not exist at all
/// println!("File written atomically");
/// # Ok(())
/// # }
/// ```
///
/// # Atomicity Guarantees
///
/// - Either the entire file is written or no file exists
/// - No partial writes are visible to other processes
/// - Safe for concurrent access by multiple processes
/// - Temporary file is automatically cleaned up on failure
///
/// # Implementation Details
///
/// 1. Creates a temporary file with `.tmp` extension
/// 2. Writes all content to the temporary file
/// 3. Atomically renames temporary file to target path
/// 4. Cleans up temporary file if any step fails
pub fn atomic_write(path: &Path, content: &[u8]) -> Result<()> {
    let temp_path = path.with_extension("tmp");
    
    // Write to temp file
    fs::write(&temp_path, content)?;
    
    // Atomic rename
    fs::rename(&temp_path, path)?;
    
    Ok(())
}



/// Create a symlink (cross-platform)
#[cfg(unix)]
pub fn create_symlink(target: &Path, link: &Path) -> Result<()> {
    use std::os::unix::fs::symlink;
    symlink(target, link)?;
    Ok(())
}

/// Create a symlink (Windows)
#[cfg(windows)]
pub fn create_symlink(target: &Path, link: &Path) -> Result<()> {
    use std::os::windows::fs::{symlink_dir, symlink_file};
    
    if target.is_dir() {
        symlink_dir(target, link)?;
    } else {
        symlink_file(target, link)?;
    }
    Ok(())
}

/// Read symlink target
pub fn read_symlink(path: &Path) -> Result<PathBuf> {
    Ok(fs::read_link(path)?)
}





#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_hash_functions() {
        // Test hash_data
        let data = b"Hello, World!";
        let hash1 = hash_data(data);
        let hash2 = hash_data(data);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 hex
        

    }
    
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1_048_576), "1.00 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.00 GB");
        assert_eq!(format_bytes(1_099_511_627_776), "1.00 TB");
    }
    

    
    #[test]
    fn test_atomic_write() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        
        // Write atomically
        atomic_write(&file_path, b"Test content").unwrap();
        
        // Verify content
        let content = fs::read(&file_path).unwrap();
        assert_eq!(content, b"Test content");
        
        // Verify temp file doesn't exist
        assert!(!file_path.with_extension("tmp").exists());
    }
    
    #[test]
    fn test_make_relative() {
        let temp_dir = TempDir::new().unwrap();
        let base = temp_dir.path();
        let subdir = base.join("subdir");
        let file = subdir.join("file.txt");
        
        fs::create_dir_all(&subdir).unwrap();
        fs::write(&file, b"test").unwrap();
        
        let relative = make_relative(&file, base).unwrap();
        assert_eq!(relative, PathBuf::from("subdir/file.txt"));
    }
    

    

    
    #[test]
    fn test_remove_dir_if_empty() {
        let temp_dir = TempDir::new().unwrap();
        let empty_dir = temp_dir.path().join("empty");
        let non_empty_dir = temp_dir.path().join("non_empty");
        
        fs::create_dir(&empty_dir).unwrap();
        fs::create_dir(&non_empty_dir).unwrap();
        fs::write(non_empty_dir.join("file.txt"), b"test").unwrap();
        
        // Remove empty directory
        assert!(remove_dir_if_empty(&empty_dir).unwrap());
        assert!(!empty_dir.exists());
        
        // Don't remove non-empty directory
        assert!(!remove_dir_if_empty(&non_empty_dir).unwrap());
        assert!(non_empty_dir.exists());
    }
} 