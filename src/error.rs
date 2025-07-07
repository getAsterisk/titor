//! Error types for the Titor library
//!
//! This module defines all error types that can occur during Titor operations.
//! Errors are designed to be informative and actionable, providing clear context
//! about what went wrong and potential remediation steps.

use std::path::PathBuf;
use thiserror::Error;

/// Type alias for Results in the Titor library
pub type Result<T> = std::result::Result<T, TitorError>;

/// Main error type for all Titor operations
#[derive(Debug, Error)]
pub enum TitorError {
    /// I/O errors during file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Errors during JSON serialization/deserialization
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    
    /// Errors during bincode serialization/deserialization
    #[error("Bincode error: {0}")]
    Bincode(String),
    
    /// Checkpoint not found in storage
    #[error("Checkpoint not found: {0}")]
    CheckpointNotFound(String),
    
    /// Object not found in content-addressable storage
    #[error("Object not found: {0}")]
    ObjectNotFound(String),
    
    /// Corruption detected during verification
    #[error("Corruption detected: {0}")]
    CorruptionDetected(String),
    
    /// Hash mismatch during verification
    #[error("Hash mismatch - expected: {expected}, actual: {actual}")]
    HashMismatch {
        /// Expected hash value
        expected: String,
        /// Actual computed hash value
        actual: String,
    },
    
    /// Storage-related errors
    #[error("Storage error: {0}")]
    Storage(String),
    
    /// Compression errors
    #[error("Compression error: {0}")]
    Compression(String),
    
    /// Decompression errors
    #[error("Decompression error: {0}")]
    Decompression(String),
    
    /// Invalid checkpoint structure
    #[error("Invalid checkpoint: {0}")]
    InvalidCheckpoint(String),
    
    /// Invalid timeline structure
    #[error("Invalid timeline: {0}")]
    InvalidTimeline(String),
    
    /// File too large for configured limits
    #[error("File too large: {path:?} ({size} bytes exceeds limit of {limit} bytes)")]
    FileTooLarge {
        /// Path to the file
        path: PathBuf,
        /// Actual file size
        size: u64,
        /// Configured size limit
        limit: u64,
    },
    
    /// Permission denied for file operation
    #[error("Permission denied: {path:?}")]
    PermissionDenied {
        /// Path where permission was denied
        path: PathBuf,
    },
    
    /// Unsupported file type
    #[error("Unsupported file type: {path:?}")]
    UnsupportedFileType {
        /// Path to the unsupported file
        path: PathBuf,
    },
    
    /// Circular dependency in timeline
    #[error("Circular dependency detected in timeline")]
    CircularDependency,
    
    /// Checkpoint has children and cannot be deleted
    #[error("Cannot delete checkpoint {0}: it has child checkpoints")]
    CheckpointHasChildren(String),
    
    /// Storage is not initialized
    #[error("Storage not initialized at path: {0:?}")]
    StorageNotInitialized(PathBuf),
    
    /// Storage already exists
    #[error("Storage already exists at path: {0:?}")]
    StorageAlreadyExists(PathBuf),
    
    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    /// Concurrent modification detected
    #[error("Concurrent modification detected: {0}")]
    ConcurrentModification(String),
    
    /// Garbage collection error
    #[error("Garbage collection error: {0}")]
    GarbageCollection(String),
    
    /// Verification failed
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
    
    /// Restore operation failed
    #[error("Restore failed: {0}")]
    RestoreFailed(String),
    
    /// Fork operation failed
    #[error("Fork failed: {0}")]
    ForkFailed(String),
    
    /// Diff operation failed
    #[error("Diff failed: {0}")]
    DiffFailed(String),
    
    /// Pattern parsing error
    #[error("Invalid ignore pattern: {0}")]
    InvalidPattern(String),
    
    /// Walk directory error from walkdir crate
    #[error("Walk directory error")]
    WalkDir(#[from] walkdir::Error),
    
    /// UUID generation error
    #[error("UUID error")]
    Uuid(#[from] uuid::Error),
    
    /// UTF-8 conversion error
    #[error("UTF-8 error: {0}")]
    Utf8(#[from] std::str::Utf8Error),
    
    /// Path conversion error
    #[error("Path conversion error: {0:?}")]
    PathConversion(std::ffi::OsString),
    
    /// Lock acquisition timeout
    #[error("Lock acquisition timeout")]
    LockTimeout,
    
    /// Memory map error
    #[error("Memory map error: {0}")]
    MemoryMap(String),
    
    /// Thread pool error
    #[error("Thread pool error: {0}")]
    ThreadPool(String),
    
    /// Auto-checkpoint strategy error
    #[error("Auto-checkpoint error: {0}")]
    AutoCheckpoint(String),
    
    /// Hook execution error
    #[error("Hook execution error: {0}")]
    HookExecution(String),
    
    /// Generic error for unexpected conditions
    #[error("Internal error: {0}")]
    Internal(String),
    
    /// Custom error type for extensions
    #[error("{0}")]
    Custom(String),
}

// Implement conversions for bincode 2.0 error types
impl From<bincode::error::DecodeError> for TitorError {
    fn from(err: bincode::error::DecodeError) -> Self {
        TitorError::Bincode(err.to_string())
    }
}

impl From<bincode::error::EncodeError> for TitorError {
    fn from(err: bincode::error::EncodeError) -> Self {
        TitorError::Bincode(err.to_string())
    }
}

impl TitorError {
    /// Create a storage error with a custom message
    pub fn storage(msg: impl Into<String>) -> Self {
        TitorError::Storage(msg.into())
    }
    
    /// Create a compression error with a custom message
    pub fn compression(msg: impl Into<String>) -> Self {
        TitorError::Compression(msg.into())
    }
    
    /// Create a decompression error with a custom message
    pub fn decompression(msg: impl Into<String>) -> Self {
        TitorError::Decompression(msg.into())
    }
    
    /// Create an internal error with a custom message
    pub fn internal(msg: impl Into<String>) -> Self {
        TitorError::Internal(msg.into())
    }
    
    /// Create a custom error with a custom message
    pub fn custom(msg: impl Into<String>) -> Self {
        TitorError::Custom(msg.into())
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            TitorError::LockTimeout
                | TitorError::ConcurrentModification(_)
                | TitorError::PermissionDenied { .. }
        )
    }
    
    /// Check if this error indicates corruption
    pub fn is_corruption(&self) -> bool {
        matches!(
            self,
            TitorError::CorruptionDetected(_)
                | TitorError::HashMismatch { .. }
                | TitorError::InvalidCheckpoint(_)
                | TitorError::InvalidTimeline(_)
        )
    }
    
    /// Get a user-friendly error message with suggestions
    pub fn user_message(&self) -> String {
        match self {
            TitorError::CheckpointNotFound(id) => {
                format!("Checkpoint '{}' not found. Use 'list_checkpoints()' to see available checkpoints.", id)
            }
            TitorError::PermissionDenied { path } => {
                format!("Permission denied for {:?}. Check file permissions or run with appropriate privileges.", path)
            }
            TitorError::FileTooLarge { path, size, limit } => {
                format!(
                    "File {:?} is too large ({} bytes). Maximum allowed size is {} bytes. \
                     Consider increasing the limit or excluding large files.",
                    path, size, limit
                )
            }
            TitorError::StorageNotInitialized(path) => {
                format!("Storage not initialized at {:?}. Run 'Titor::init()' first.", path)
            }
            TitorError::LockTimeout => {
                "Operation timed out waiting for lock. Another operation may be in progress. Try again later.".to_string()
            }
            _ => self.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_display() {
        let err = TitorError::CheckpointNotFound("abc123".to_string());
        assert_eq!(err.to_string(), "Checkpoint not found: abc123");
    }
    
    #[test]
    fn test_error_recoverable() {
        assert!(TitorError::LockTimeout.is_recoverable());
        assert!(!TitorError::CorruptionDetected("test".to_string()).is_recoverable());
    }
    
    #[test]
    fn test_error_corruption() {
        assert!(TitorError::HashMismatch {
            expected: "abc".to_string(),
            actual: "def".to_string(),
        }.is_corruption());
        assert!(!TitorError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "test"
        )).is_corruption());
    }
} 