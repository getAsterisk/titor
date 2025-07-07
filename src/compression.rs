//! High-performance LZ4 compression engine for Titor
//!
//! This module provides extreme-speed compression using LZ4, optimized for
//! interactive use with minimal CPU overhead. It supports adaptive compression
//! strategies and streaming for large files.
//!
//! ## Overview
//!
//! The compression engine uses LZ4, a fast compression algorithm that provides:
//! - Compression speeds > 500 MB/s per core
//! - Decompression speeds > 2 GB/s per core
//! - Reasonable compression ratios (typically 2:1 to 5:1 for source code)
//!
//! ## Compression Strategies
//!
//! Multiple strategies are available to balance speed and storage efficiency:
//!
//! - **None**: No compression, maximum speed
//! - **Fast**: LZ4 compression for all files (default)
//! - **Adaptive**: Smart compression based on file size and type
//! - **Custom**: User-defined compression logic
//!
//! ## Format
//!
//! Compressed data is stored with a 4-byte magic header:
//! - `LZ4T` (0x4C5A3454): LZ4 compressed data follows
//! - `\0\0\0\0`: Uncompressed data follows
//!
//! This allows seamless handling of both compressed and uncompressed content.
//!
//! ## Examples
//!
//! ```rust
//! use titor::compression::{CompressionEngine, CompressionStrategy};
//! use std::path::Path;
//!
//! // Create engine with fast compression
//! let mut engine = CompressionEngine::new(CompressionStrategy::Fast);
//!
//! // Compress some data
//! let data = b"Hello, world! This is some text to compress.";
//! let compressed = engine.compress(Path::new("test.txt"), data).unwrap();
//!
//! // Decompress it back
//! let decompressed = engine.decompress(&compressed).unwrap();
//! assert_eq!(decompressed, data);
//! ```
//!
//! ## Streaming Large Files
//!
//! For files too large to fit in memory, use the streaming API:
//!
//! ```rust,no_run
//! use titor::compression::CompressedFileStream;
//! use std::fs::File;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let file = File::open("large_file.bin")?;
//! let mut stream = CompressedFileStream::new(file, 1024 * 1024); // 1MB chunks
//!
//! let stats = stream.process_chunks(|compressed_chunk| {
//!     // Process each compressed chunk
//!     println!("Got chunk of {} bytes", compressed_chunk.len());
//!     Ok(())
//! })?;
//!
//! println!("Compressed {} files, saved {} bytes", 
//!          stats.files_compressed, stats.bytes_saved);
//! # Ok(())
//! # }
//! ```

use crate::error::{Result, TitorError};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, trace};

/// Compression strategies optimized for performance
///
/// Determines when and how files are compressed. Different strategies
/// can be used to optimize for speed, storage efficiency, or specific
/// file types.
///
/// # Examples
///
/// ```rust
/// use titor::compression::CompressionStrategy;
///
/// // No compression - fastest option
/// let none = CompressionStrategy::None;
///
/// // Default fast compression
/// let fast = CompressionStrategy::Fast;
///
/// // Adaptive compression with custom settings
/// let adaptive = CompressionStrategy::Adaptive {
///     min_size: 4096,  // Skip files < 4KB
///     skip_extensions: vec![
///         "jpg".to_string(),
///         "png".to_string(),
///         "zip".to_string(),
///     ],
/// };
/// ```
#[derive(Clone)]
pub enum CompressionStrategy {
    /// No compression - maximum speed for small projects
    ///
    /// Use this when:
    /// - Working with small files or projects
    /// - Storage space is not a concern
    /// - Maximum speed is required
    None,
    
    /// LZ4 compression for all eligible files (default)
    ///
    /// Use this when:
    /// - You want good compression with minimal CPU impact
    /// - Working with typical source code or text files
    /// - Storage efficiency matters
    ///
    /// Files smaller than 1KB are stored uncompressed for efficiency.
    Fast,
    
    /// Adaptive compression based on file characteristics
    ///
    /// Use this when:
    /// - Working with mixed file types
    /// - Want to skip already-compressed formats
    /// - Need fine-grained control over compression
    Adaptive {
        /// Skip compression for files smaller than this (default: 4KB)
        min_size: usize,
        /// Skip these file extensions (already compressed)
        skip_extensions: Vec<String>,
    },
    
    /// Custom strategy function
    ///
    /// Provides complete control over compression decisions.
    /// The function receives the file path and size, and returns
    /// whether compression should be applied.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::compression::CompressionStrategy;
    /// use std::sync::Arc;
    ///
    /// // Only compress files in src/ directory
    /// let strategy = CompressionStrategy::Custom(Arc::new(|path, size| {
    ///     path.starts_with("src/") && size > 1024
    /// }));
    /// ```
    Custom(Arc<dyn Fn(&Path, usize) -> bool + Send + Sync>),
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        CompressionStrategy::Fast
    }
}

impl std::fmt::Debug for CompressionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Fast => write!(f, "Fast"),
            Self::Adaptive { min_size, skip_extensions } => f
                .debug_struct("Adaptive")
                .field("min_size", min_size)
                .field("skip_extensions", skip_extensions)
                .finish(),
            Self::Custom(_) => write!(f, "Custom(Fn)"),
        }
    }
}

/// Compression statistics for monitoring
///
/// Tracks compression performance and effectiveness. Useful for
/// understanding the impact of compression on your repository.
#[derive(Debug, Default, Clone)]
pub struct CompressionStats {
    /// Number of files compressed
    pub files_compressed: usize,
    /// Number of files stored raw
    pub files_stored_raw: usize,
    /// Total bytes saved by compression
    pub bytes_saved: usize,
    /// Total compression time in milliseconds
    pub compression_time_ms: u64,
    /// Total decompression time in milliseconds
    pub decompression_time_ms: u64,
}

impl CompressionStats {
    /// Get compression ratio (0.0 to 1.0, where 1.0 is 100% compression)
    ///
    /// # Returns
    ///
    /// The ratio of compressed files to total files processed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::compression::CompressionStats;
    /// # let stats = CompressionStats { files_compressed: 75, files_stored_raw: 25, ..Default::default() };
    /// let ratio = stats.compression_ratio();
    /// println!("{}% of files were compressed", ratio * 100.0);
    /// ```
    pub fn compression_ratio(&self) -> f64 {
        if self.files_compressed == 0 {
            return 0.0;
        }
        
        let total_files = self.files_compressed + self.files_stored_raw;
        self.files_compressed as f64 / total_files as f64
    }
    
    /// Get average bytes saved per compressed file
    ///
    /// # Returns
    ///
    /// Average space savings per compressed file in bytes.
    pub fn avg_bytes_saved_per_file(&self) -> usize {
        if self.files_compressed == 0 {
            return 0;
        }
        self.bytes_saved / self.files_compressed
    }
}

/// Compression engine optimized for speed
///
/// The main compression engine that handles file compression and decompression
/// using LZ4. It tracks statistics and applies compression strategies.
///
/// # Thread Safety
///
/// The engine is not thread-safe. For concurrent compression, create
/// multiple engine instances.
///
/// # Examples
///
/// ```rust
/// use titor::compression::{CompressionEngine, CompressionStrategy};
/// use std::path::Path;
///
/// let mut engine = CompressionEngine::new(CompressionStrategy::Fast);
///
/// // Compress a file
/// let content = b"Source code content here...";
/// let compressed = engine.compress(Path::new("main.rs"), content)?;
///
/// // Check statistics
/// let stats = engine.stats();
/// println!("Compressed {} files, saved {} bytes",
///          stats.files_compressed, stats.bytes_saved);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct CompressionEngine {
    strategy: CompressionStrategy,
    stats: CompressionStats,
}

// Magic bytes to identify LZ4 compressed data
const LZ4_MAGIC: &[u8] = b"LZ4T"; // LZ4 Titor

impl CompressionEngine {
    /// Create a new compression engine with the specified strategy
    ///
    /// # Arguments
    ///
    /// * `strategy` - The compression strategy to use
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::compression::{CompressionEngine, CompressionStrategy};
    ///
    /// // Fast compression (default)
    /// let engine = CompressionEngine::new(CompressionStrategy::Fast);
    ///
    /// // No compression
    /// let engine = CompressionEngine::new(CompressionStrategy::None);
    /// ```
    pub fn new(strategy: CompressionStrategy) -> Self {
        Self {
            strategy,
            stats: CompressionStats::default(),
        }
    }
    
    /// Get current compression statistics
    ///
    /// Returns statistics about compression performance since creation
    /// or last reset.
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }
    
    /// Reset statistics
    ///
    /// Clears all compression statistics, useful for measuring
    /// performance over specific operations.
    pub fn reset_stats(&mut self) {
        self.stats = CompressionStats::default();
    }
    
    /// Compress file content based on strategy
    ///
    /// Applies compression according to the configured strategy.
    /// Small files and already-compressed formats may be stored
    /// uncompressed for efficiency.
    ///
    /// # Arguments
    ///
    /// * `path` - Path of the file being compressed (for strategy decisions)
    /// * `content` - File content to compress
    ///
    /// # Returns
    ///
    /// Compressed data with appropriate header:
    /// - LZ4T header + compressed data if compression was beneficial
    /// - Null header + original data if compression was not beneficial
    ///
    /// # Errors
    ///
    /// Returns an error if compression fails (rare with LZ4).
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::compression::{CompressionEngine, CompressionStrategy};
    /// # use std::path::Path;
    /// # let mut engine = CompressionEngine::new(CompressionStrategy::Fast);
    /// let content = b"print('Hello, World!')\n" .repeat(100);
    /// let compressed = engine.compress(Path::new("hello.py"), &content)?;
    /// 
    /// // Compressed data will be smaller than original
    /// assert!(compressed.len() < content.len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn compress(&mut self, path: &Path, content: &[u8]) -> Result<Vec<u8>> {
        let start = Instant::now();
        
        // Check if compression should be applied
        if !self.should_compress(path, content.len()) {
            trace!("Skipping compression for {:?} (strategy)", path);
            self.stats.files_stored_raw += 1;
            // Return with uncompressed marker (empty magic + original content)
            let mut result = Vec::with_capacity(4 + content.len());
            result.extend_from_slice(&[0, 0, 0, 0]); // Empty magic = uncompressed
            result.extend_from_slice(content);
            return Ok(result);
        }
        
        // For very small files, skip compression attempt
        if content.len() < 64 {
            trace!("File too small to benefit from compression: {:?}", path);
            self.stats.files_stored_raw += 1;
            let mut result = Vec::with_capacity(4 + content.len());
            result.extend_from_slice(&[0, 0, 0, 0]);
            result.extend_from_slice(content);
            return Ok(result);
        }
        
        // Use LZ4 with prepended size for safety
        let compressed = compress_prepend_size(content);
        
        if compressed.len() < content.len() {
            // Compression was beneficial
            let saved = content.len() - compressed.len();
            self.stats.bytes_saved += saved;
            self.stats.files_compressed += 1;
            self.stats.compression_time_ms += start.elapsed().as_millis() as u64;
            
            debug!(
                "Compressed {:?}: {} -> {} bytes (saved {} bytes, {:.1}%)",
                path,
                content.len(),
                compressed.len(),
                saved,
                (saved as f64 / content.len() as f64) * 100.0
            );
            
            // Prepend magic bytes
            let mut result = Vec::with_capacity(LZ4_MAGIC.len() + compressed.len());
            result.extend_from_slice(LZ4_MAGIC);
            result.extend_from_slice(&compressed);
            Ok(result)
        } else {
            // Compression didn't help, store raw
            trace!("Compression not beneficial for {:?}, storing raw", path);
            self.stats.files_stored_raw += 1;
            // Return with uncompressed marker
            let mut result = Vec::with_capacity(4 + content.len());
            result.extend_from_slice(&[0, 0, 0, 0]); // Empty magic = uncompressed
            result.extend_from_slice(content);
            Ok(result)
        }
    }
    
    /// Decompress content
    ///
    /// Decompresses data that was previously compressed by this engine.
    /// Automatically detects the format (compressed or uncompressed) based
    /// on the header bytes.
    ///
    /// # Arguments
    ///
    /// * `content` - Compressed or uncompressed data with header
    ///
    /// # Returns
    ///
    /// Original uncompressed data.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Data is too short (less than 4 bytes)
    /// - Decompression fails due to corruption
    /// - Unknown format header
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::compression::{CompressionEngine, CompressionStrategy};
    /// # use std::path::Path;
    /// # let mut engine = CompressionEngine::new(CompressionStrategy::Fast);
    /// let original = b"Some data to compress";
    /// let compressed = engine.compress(Path::new("data.txt"), original)?;
    /// let decompressed = engine.decompress(&compressed)?;
    /// assert_eq!(decompressed, original);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decompress(&mut self, content: &[u8]) -> Result<Vec<u8>> {
        let start = Instant::now();
        
        if content.len() < 4 {
            return Err(TitorError::decompression("Content too short"));
        }
        
        // Check magic bytes
        if content.starts_with(LZ4_MAGIC) {
            // LZ4 compressed data
            let compressed_data = &content[LZ4_MAGIC.len()..];
            match decompress_size_prepended(compressed_data) {
                Ok(decompressed) => {
                    self.stats.decompression_time_ms += start.elapsed().as_millis() as u64;
                    trace!("Decompressed {} bytes to {} bytes", content.len(), decompressed.len());
                    Ok(decompressed)
                }
                Err(e) => Err(TitorError::decompression(format!(
                    "LZ4 decompression failed: {}",
                    e
                ))),
            }
        } else if content.starts_with(&[0, 0, 0, 0]) {
            // Uncompressed data marker
            trace!("Content not compressed, returning as-is");
            Ok(content[4..].to_vec())
        } else {
            // Unknown format - for backward compatibility, try as raw LZ4
            if self.is_legacy_lz4_compressed(content) {
                match decompress_size_prepended(content) {
                    Ok(decompressed) => {
                        self.stats.decompression_time_ms += start.elapsed().as_millis() as u64;
                        trace!("Decompressed legacy LZ4: {} bytes to {} bytes", content.len(), decompressed.len());
                        Ok(decompressed)
                    }
                    Err(_) => {
                        // Not actually compressed, return as-is for backward compatibility
                        trace!("Legacy format not compressed, returning as-is");
                        Ok(content.to_vec())
                    }
                }
            } else {
                // Assume uncompressed for backward compatibility
                trace!("Unknown format, returning as-is");
                Ok(content.to_vec())
            }
        }
    }
    
    /// Determine if compression should be applied
    fn should_compress(&self, path: &Path, size: usize) -> bool {
        match &self.strategy {
            CompressionStrategy::None => false,
            CompressionStrategy::Fast => size >= 1024, // Skip very small files
            CompressionStrategy::Adaptive { min_size, skip_extensions } => {
                if size < *min_size {
                    return false;
                }
                
                // Check file extension
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    let should_skip = skip_extensions
                        .iter()
                        .any(|skip| skip.eq_ignore_ascii_case(ext));
                    !should_skip
                } else {
                    true
                }
            }
            CompressionStrategy::Custom(func) => func(path, size),
        }
    }
    
    /// Check if content has legacy LZ4 compression header (for backward compatibility)
    fn is_legacy_lz4_compressed(&self, content: &[u8]) -> bool {
        // LZ4 with prepended size has 4-byte size header
        if content.len() < 8 {  // Need at least size header + some data
            return false;
        }
        
        let size = u32::from_le_bytes([content[0], content[1], content[2], content[3]]) as usize;
        
        // Sanity checks
        size > 0 && size < 100_000_000 && content.len() > 4
    }
    
    /// Check if content has LZ4 compression header
    #[cfg(test)]
    fn is_lz4_compressed(&self, content: &[u8]) -> bool {
        content.starts_with(LZ4_MAGIC)
    }
}

/// Stream large files with compression
///
/// Provides streaming compression for files too large to fit in memory.
/// Files are processed in chunks, with each chunk compressed independently.
///
/// # Type Parameters
///
/// * `R` - Any type that implements `Read`
///
/// # Examples
///
/// ```rust,no_run
/// use titor::compression::CompressedFileStream;
/// use std::fs::File;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let file = File::open("large_video.mp4")?;
/// let mut stream = CompressedFileStream::new(file, 4 * 1024 * 1024); // 4MB chunks
///
/// stream.process_chunks(|chunk| {
///     // Store or transmit compressed chunk
///     Ok(())
/// })?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct CompressedFileStream<R: Read> {
    reader: BufReader<R>,
    chunk_size: usize,
    _marker: std::marker::PhantomData<R>,
}

impl<R: Read> CompressedFileStream<R> {
    /// Create a new compressed file stream
    ///
    /// # Arguments
    ///
    /// * `reader` - Source of data to compress
    /// * `chunk_size` - Size of chunks to process at once
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size` is 0.
    pub fn new(reader: R, chunk_size: usize) -> Self {
        Self {
            reader: BufReader::with_capacity(chunk_size, reader),
            chunk_size,
            _marker: std::marker::PhantomData,
        }
    }
    
    /// Process file in chunks to avoid memory issues
    pub fn process_chunks<F>(&mut self, mut processor: F) -> Result<CompressionStats>
    where
        F: FnMut(&[u8]) -> Result<()>,
    {
        let mut buffer = vec![0u8; self.chunk_size];
        let mut stats = CompressionStats::default();
        let mut total_uncompressed = 0u64;
        let mut total_compressed = 0u64;
        let start = Instant::now();
        
        loop {
            match self.reader.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(bytes_read) => {
                    let chunk = &buffer[..bytes_read];
                    total_uncompressed += bytes_read as u64;
                    
                    // Try to compress chunk
                    let compressed = compress_prepend_size(chunk);
                    
                    // Only use compressed if it's smaller
                    if compressed.len() < chunk.len() {
                        processor(&compressed)?;
                        total_compressed += compressed.len() as u64;
                        stats.files_compressed += 1;
                        stats.bytes_saved += chunk.len() - compressed.len();
                    } else {
                        // Prepend marker for uncompressed data
                        let mut uncompressed = vec![0u8; 4 + chunk.len()];
                        uncompressed[0..4].copy_from_slice(&[0, 0, 0, 0]); // Zero size = uncompressed
                        uncompressed[4..].copy_from_slice(chunk);
                        processor(&uncompressed)?;
                        total_compressed += uncompressed.len() as u64;
                        stats.files_stored_raw += 1;
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        
        stats.compression_time_ms = start.elapsed().as_millis() as u64;
        trace!(
            "Streamed compression: {} bytes -> {} bytes (saved {} bytes)",
            total_uncompressed,
            total_compressed,
            total_uncompressed.saturating_sub(total_compressed)
        );
        
        Ok(stats)
    }
    
    /// Process file in chunks asynchronously
    // #[cfg(feature = "async")]
    #[allow(unused)]
    pub async fn process_chunks_async<F, Fut>(&mut self, mut processor: F) -> Result<CompressionStats>
    where
        F: FnMut(Vec<u8>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        // use tokio::io::{AsyncReadExt, AsyncBufReadExt};
        
        let mut buffer = vec![0u8; self.chunk_size];
        let mut stats = CompressionStats::default();
        let mut total_uncompressed = 0u64;
        let mut total_compressed = 0u64;
        let start = Instant::now();
        
        loop {
            // Simulate async read (in real implementation, use AsyncRead trait)
            let bytes_read = match self.reader.read(&mut buffer) {
                Ok(n) => n,
                Err(e) => return Err(e.into()),
            };
            
            if bytes_read == 0 {
                break; // EOF
            }
            
            let chunk = buffer[..bytes_read].to_vec();
            total_uncompressed += bytes_read as u64;
            
            // Compress in background task
            let compressed = tokio::task::spawn_blocking(move || {
                compress_prepend_size(&chunk)
            }).await.map_err(|e| TitorError::internal(format!("Compression task failed: {}", e)))?;
            
            // Process compressed data
            if compressed.len() < bytes_read {
                processor(compressed.clone()).await?;
                total_compressed += compressed.len() as u64;
                stats.files_compressed += 1;
                stats.bytes_saved += bytes_read - compressed.len();
            } else {
                // Prepend marker for uncompressed data
                let mut uncompressed = vec![0u8; 4 + bytes_read];
                uncompressed[0..4].copy_from_slice(&[0, 0, 0, 0]);
                uncompressed[4..].copy_from_slice(&buffer[..bytes_read]);
                processor(uncompressed.clone()).await?;
                total_compressed += uncompressed.len() as u64;
                stats.files_stored_raw += 1;
            }
        }
        
        stats.compression_time_ms = start.elapsed().as_millis() as u64;
        Ok(stats)
    }
}

/// Writer that compresses data in chunks
#[derive(Debug)]
pub struct CompressedWriter<W: Write> {
    writer: Option<W>,
    buffer: Vec<u8>,
    chunk_size: usize,
    stats: CompressionStats,
    compression_start: Option<Instant>,
}

impl<W: Write> CompressedWriter<W> {
    /// Create a new compressed writer
    pub fn new(writer: W, chunk_size: usize) -> Self {
        Self {
            writer: Some(writer),
            buffer: Vec::with_capacity(chunk_size),
            chunk_size,
            stats: CompressionStats::default(),
            compression_start: Some(Instant::now()),
        }
    }
    
    /// Get compression statistics
    pub fn stats(&self) -> &CompressionStats {
        &self.stats
    }
    
    /// Flush any buffered data
    pub fn flush_buffer(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            let uncompressed_size = self.buffer.len();
            let compressed = compress_prepend_size(&self.buffer);
            
            // Only use compressed if beneficial
            if let Some(writer) = self.writer.as_mut() {
                if compressed.len() < uncompressed_size {
                    writer.write_all(&compressed)?;
                    self.stats.files_compressed += 1;
                    self.stats.bytes_saved += uncompressed_size - compressed.len();
                    trace!(
                        "Compressed chunk: {} -> {} bytes (saved {} bytes)",
                        uncompressed_size,
                        compressed.len(),
                        uncompressed_size - compressed.len()
                    );
                } else {
                    // Write uncompressed with marker
                    writer.write_all(&[0, 0, 0, 0])?; // Zero size = uncompressed
                    writer.write_all(&self.buffer)?;
                    self.stats.files_stored_raw += 1;
                    trace!("Stored chunk uncompressed: {} bytes", uncompressed_size);
                }
            }
            
            self.buffer.clear();
        }
        Ok(())
    }
    
    /// Finish writing and return the underlying writer and stats
    pub fn finish(mut self) -> Result<(W, CompressionStats)> {
        self.flush_buffer()?;
        
        // Take the writer
        let mut writer = self.writer.take()
            .ok_or_else(|| TitorError::internal("Writer already consumed"))?;
        writer.flush()?;
        
        // Update compression time
        if let Some(start) = self.compression_start.take() {
            self.stats.compression_time_ms = start.elapsed().as_millis() as u64;
        }
        
        Ok((writer, self.stats.clone()))
    }
}

impl<W: Write> Write for CompressedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.writer.is_none() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Writer already consumed"
            ));
        }
        
        let written = buf.len();
        
        // If buffer + new data exceeds chunk size, process in parts
        let mut offset = 0;
        while offset < buf.len() {
            let remaining_in_buffer = self.chunk_size - self.buffer.len();
            let to_copy = remaining_in_buffer.min(buf.len() - offset);
            
            self.buffer.extend_from_slice(&buf[offset..offset + to_copy]);
            offset += to_copy;
            
            // Flush if buffer is full
            if self.buffer.len() >= self.chunk_size {
                self.flush_buffer()
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            }
        }
        
        Ok(written)
    }
    
    fn flush(&mut self) -> std::io::Result<()> {
        self.flush_buffer()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        if let Some(writer) = self.writer.as_mut() {
            writer.flush()
        } else {
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Writer already consumed"
            ))
        }
    }
}

impl<W: Write> Drop for CompressedWriter<W> {
    fn drop(&mut self) {
        // Best effort flush on drop
        let _ = self.flush_buffer();
    }
}

/// Get default skip extensions for adaptive compression
///
/// Provides a default list of file extensions that should not be
/// compressed because they are already in compressed formats.
/// 
/// # Returns
///
/// Vector of lowercase file extensions (without dots) that are
/// already compressed and should be stored as-is.
///
/// # Examples
///
/// ```rust
/// use titor::compression::{default_skip_extensions, CompressionStrategy};
///
/// let skip_exts = default_skip_extensions();
/// assert!(skip_exts.contains(&"jpg".to_string()));
/// assert!(skip_exts.contains(&"zip".to_string()));
///
/// // Use with adaptive strategy
/// let strategy = CompressionStrategy::Adaptive {
///     min_size: 4096,
///     skip_extensions: skip_exts,
/// };
/// ```
pub fn default_skip_extensions() -> Vec<String> {
    vec![
        // Images
        "jpg", "jpeg", "png", "gif", "webp", "ico", "bmp", "svg",
        // Video
        "mp4", "avi", "mkv", "mov", "wmv", "flv", "webm", "m4v", "mpg", "mpeg",
        // Audio
        "mp3", "wav", "flac", "aac", "ogg", "wma", "m4a", "opus",
        // Archives
        "zip", "rar", "7z", "tar", "gz", "bz2", "xz", "zst",
        // Already compressed
        "lz4", "lzo", "lzma", "br",
        // Other binary formats
        "pdf", "epub", "mobi",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_compression_fast_strategy() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Fast);
        let path = PathBuf::from("test.txt");
        
        // Small file should not be compressed
        let small_content = b"hello";
        let result = engine.compress(&path, small_content).unwrap();
        // Should have uncompressed marker + content
        assert_eq!(&result[0..4], &[0, 0, 0, 0]);
        assert_eq!(&result[4..], small_content);
        assert_eq!(engine.stats().files_stored_raw, 1);
        
        // Can decompress uncompressed data
        let decompressed = engine.decompress(&result).unwrap();
        assert_eq!(decompressed, small_content);
        
        // Large file should be compressed
        let large_content = "a".repeat(10000).into_bytes();
        let compressed = engine.compress(&path, &large_content).unwrap();
        assert!(compressed.len() < large_content.len() + LZ4_MAGIC.len());
        assert_eq!(engine.stats().files_compressed, 1);
        
        // Can decompress
        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed, large_content);
    }
    
    #[test]
    fn test_compression_adaptive_strategy() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Adaptive {
            min_size: 100,
            skip_extensions: vec!["jpg".to_string()],
        });
        
        // File below min_size
        let small_path = PathBuf::from("small.txt");
        let small_content = b"tiny";
        let result = engine.compress(&small_path, small_content).unwrap();
        assert_eq!(&result[0..4], &[0, 0, 0, 0]);
        assert_eq!(&result[4..], small_content);
        
        // Skip extension
        let jpg_path = PathBuf::from("image.jpg");
        let jpg_content = vec![0xFF; 1000]; // Fake JPEG data
        let result = engine.compress(&jpg_path, &jpg_content).unwrap();
        assert_eq!(&result[0..4], &[0, 0, 0, 0]);
        assert_eq!(&result[4..], &jpg_content[..]);
        
        // Should compress
        let txt_path = PathBuf::from("large.txt");
        let txt_content = "x".repeat(1000).into_bytes();
        let compressed = engine.compress(&txt_path, &txt_content).unwrap();
        assert!(compressed.starts_with(LZ4_MAGIC));
        assert!(compressed.len() < txt_content.len() + LZ4_MAGIC.len());
    }
    
    #[test]
    fn test_is_lz4_compressed() {
        let engine = CompressionEngine::new(CompressionStrategy::Fast);
        
        // Valid LZ4 header with magic
        let mut compressed = Vec::new();
        compressed.extend_from_slice(LZ4_MAGIC);
        compressed.extend_from_slice(&compress_prepend_size(b"test data"));
        assert!(engine.is_lz4_compressed(&compressed));
        
        // Uncompressed marker
        assert!(!engine.is_lz4_compressed(&[0, 0, 0, 0, 1, 2, 3]));
        
        // Not compressed (no magic)
        assert!(!engine.is_lz4_compressed(b"raw data"));
        
        // Too short
        assert!(!engine.is_lz4_compressed(b"abc"));
    }
    
    #[test]
    fn test_compression_with_edge_case_content() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Fast);
        let path = PathBuf::from("test.bin");
        
        // Content that looks like it could have a size prefix
        let content = vec![57, 228, 255, 0, 1, 2, 3, 4, 5];
        let compressed = engine.compress(&path, &content).unwrap();
        
        // Should have proper marker
        assert!(compressed.starts_with(&[0, 0, 0, 0]) || compressed.starts_with(LZ4_MAGIC));
        
        // Should decompress correctly
        let decompressed = engine.decompress(&compressed).unwrap();
        assert_eq!(decompressed, content);
    }
    
    #[test]
    fn test_backward_compatibility() {
        let mut engine = CompressionEngine::new(CompressionStrategy::Fast);
        
        // Test legacy compressed data (raw LZ4 with size prefix)
        let content = b"Hello, World!";
        let legacy_compressed = compress_prepend_size(content);
        
        // Should be able to decompress legacy format
        let decompressed = engine.decompress(&legacy_compressed).unwrap();
        assert_eq!(decompressed, content);
        
        // Test raw uncompressed data (no marker)
        let raw_data = b"raw content without any marker";
        let decompressed = engine.decompress(raw_data).unwrap();
        assert_eq!(decompressed, raw_data);
    }
    
    #[test]
    fn test_compression_stats() {
        let mut stats = CompressionStats::default();
        stats.files_compressed = 8;
        stats.files_stored_raw = 2;
        stats.bytes_saved = 1000;
        
        assert_eq!(stats.compression_ratio(), 0.8);
        assert_eq!(stats.avg_bytes_saved_per_file(), 125);
    }
} 