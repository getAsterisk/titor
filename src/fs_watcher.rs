//! File system watcher for tracking changes in real-time
//!
//! This module provides the `FsWatcher` struct which uses OS-specific file system
//! event APIs to track changes as they happen. This enables O(#changes) checkpoint
//! creation instead of O(total files).
//!
//! The watcher persists events to a sled database for durability and handles
//! missed events by falling back to full scans when necessary.

use crate::error::{Result, TitorError};
use crate::types::FsEvent;
use notify::{Config, Event, EventKind, RecursiveMode, Watcher};
use parking_lot::{Mutex, RwLock};
use sled::Db;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::time::Duration;
use tracing::{debug, error, info, trace};

/// File system watcher with persistent event journal
///
/// Tracks file system changes in real-time and persists them to ensure
/// no events are lost across restarts. Automatically detects missed events
/// and triggers full scans when necessary.
pub struct FsWatcher {
    /// The notify watcher instance
    watcher: Arc<Mutex<notify::RecommendedWatcher>>,
    /// Persistent event journal
    journal: Arc<Db>,
    /// Root directory being watched
    root_path: PathBuf,
    /// Current sequence number for events
    last_seq: Arc<AtomicU64>,
    /// Set of paths with pending changes
    dirty_set: Arc<RwLock<HashSet<PathBuf>>>,
    /// Whether we've missed events and need a full scan
    missed_events: Arc<AtomicBool>,
    /// Whether the watcher is running
    running: Arc<AtomicBool>,
}

impl FsWatcher {
    /// Create a new file system watcher
    ///
    /// # Arguments
    ///
    /// * `root` - The root directory to watch
    ///
    /// # Returns
    ///
    /// A new `FsWatcher` instance, or an error if initialization fails
    pub fn new(root: &Path) -> Result<Self> {
        info!("Initializing file system watcher for: {}", root.display());
        
        // Create events database directory
        let events_dir = root.join(".titor").join("events.db");
        std::fs::create_dir_all(events_dir.parent().unwrap())?;
        
        // Open or create the journal database
        let journal = sled::open(&events_dir)
            .map_err(|e| TitorError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Failed to open events journal: {}", e)
            )))?;
        
        // Get the last sequence number from the journal
        let last_seq = journal
            .get(b"last_seq")?
            .and_then(|v| v.as_ref().try_into().ok())
            .map(u64::from_be_bytes)
            .unwrap_or(0);
        
        debug!("Last event sequence number: {}", last_seq);
        
        let journal = Arc::new(journal);
        let last_seq = Arc::new(AtomicU64::new(last_seq));
        let dirty_set = Arc::new(RwLock::new(HashSet::new()));
        let missed_events = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(false));
        
        // Create the notify watcher
        let journal_clone = journal.clone();
        let last_seq_clone = last_seq.clone();
        let dirty_set_clone = dirty_set.clone();
        let missed_events_clone = missed_events.clone();
        let root_path = root.to_path_buf();
        let root_path_clone = root_path.clone();
        
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            match res {
                Ok(event) => {
                    trace!("File system event: {:?}", event);
                    Self::handle_event(
                        event,
                        &root_path_clone,
                        &journal_clone,
                        &last_seq_clone,
                        &dirty_set_clone,
                    );
                }
                Err(e) => {
                    error!("Watch error: {}", e);
                    // On error, mark that we've missed events
                    missed_events_clone.store(true, Ordering::Relaxed);
                }
            }
        })?;
        
        // Configure the watcher
        watcher.configure(Config::default().with_poll_interval(Duration::from_secs(1)))?;
        
        Ok(FsWatcher {
            watcher: Arc::new(Mutex::new(watcher)),
            journal,
            root_path,
            last_seq,
            dirty_set,
            missed_events,
            running,
        })
    }
    
    /// Start watching for file system events
    ///
    /// # Returns
    ///
    /// Ok(()) if watching started successfully, or an error
    pub fn watch(&self) -> Result<()> {
        if self.running.swap(true, Ordering::Relaxed) {
            debug!("Watcher already running");
            return Ok(());
        }
        
        info!("Starting file system watch on: {}", self.root_path.display());
        
        self.watcher
            .lock()
            .watch(&self.root_path, RecursiveMode::Recursive)?;
        
        Ok(())
    }
    
    /// Stop watching for file system events
    pub fn stop(&self) -> Result<()> {
        if !self.running.swap(false, Ordering::Relaxed) {
            debug!("Watcher not running");
            return Ok(());
        }
        
        info!("Stopping file system watch");
        
        self.watcher
            .lock()
            .unwatch(&self.root_path)?;
        
        Ok(())
    }
    
    /// Check if the watcher has any pending events
    pub fn has_events(&self) -> bool {
        !self.dirty_set.read().is_empty()
    }
    
    /// Check if we've missed any events
    pub fn missed_events(&self) -> bool {
        self.missed_events.load(Ordering::Relaxed)
    }
    
    /// Drain all pending events
    ///
    /// Returns a vector of file system events and clears the dirty set.
    /// Also merges any persisted events that might have been missed.
    pub fn drain(&self) -> Result<Vec<FsEvent>> {
        let mut events = Vec::new();
        
        // Get events from dirty set
        let dirty_paths: Vec<PathBuf> = {
            let mut dirty = self.dirty_set.write();
            dirty.drain().collect()
        };
        
        debug!("Draining {} events from dirty set", dirty_paths.len());
        
        // Check for missed events in the journal
        let _current_seq = self.last_seq.load(Ordering::Relaxed);
        let mut seq = 0u64;
        
        // Scan journal for any events we might have missed
        for item in self.journal.iter() {
            let (key, value) = item?;
            
            // Skip metadata keys
            if key == b"last_seq" {
                continue;
            }
            
            // Parse sequence number from key
            if let Ok(seq_bytes) = key.as_ref().try_into() {
                let event_seq = u64::from_be_bytes(seq_bytes);
                
                // Only process events we haven't seen
                if event_seq > seq {
                    seq = event_seq;
                    
                    // Deserialize event
                    if let Ok((event, _)) = bincode::serde::decode_from_slice::<FsEvent, _>(&value, bincode::config::standard()) {
                        events.push(event);
                    }
                }
            }
        }
        
        // Add current dirty set events
        for path in dirty_paths {
            // Determine event type based on path existence
            let event = if !path.exists() {
                FsEvent::Delete(path)
            } else if path.metadata().map(|m| m.modified()).is_ok() {
                FsEvent::Modify(path)
            } else {
                FsEvent::Add(path)
            };
            
            events.push(event);
        }
        
        // Clear missed events flag
        self.missed_events.store(false, Ordering::Relaxed);
        
        // Clear old events from journal (keep last 1000)
        if seq > 1000 {
            let cutoff = seq - 1000;
            for i in 0..cutoff {
                let key = i.to_be_bytes();
                let _ = self.journal.remove(&key);
            }
        }
        
        info!("Drained {} total events", events.len());
        
        Ok(events)
    }
    
    /// Handle a file system event
    fn handle_event(
        event: Event,
        root_path: &Path,
        journal: &Arc<Db>,
        last_seq: &Arc<AtomicU64>,
        dirty_set: &Arc<RwLock<HashSet<PathBuf>>>,
    ) {
        // Convert notify event to our FsEvent type
        let fs_events: Vec<FsEvent> = match event.kind {
            EventKind::Create(_) => {
                event.paths.into_iter()
                    .filter_map(|p| Self::normalize_path(&p, root_path).map(FsEvent::Add))
                    .collect()
            }
            EventKind::Modify(_) => {
                event.paths.into_iter()
                    .filter_map(|p| Self::normalize_path(&p, root_path).map(FsEvent::Modify))
                    .collect()
            }
            EventKind::Remove(_) => {
                event.paths.into_iter()
                    .filter_map(|p| Self::normalize_path(&p, root_path).map(FsEvent::Delete))
                    .collect()
            }
            _ => vec![],
        };
        
        // Process each event
        for fs_event in fs_events {
            // Get path from event
            let path = match &fs_event {
                FsEvent::Add(p) | FsEvent::Modify(p) | FsEvent::Delete(p) => p.clone(),
            };
            
            // Add to dirty set
            dirty_set.write().insert(path);
            
            // Persist to journal
            let seq = last_seq.fetch_add(1, Ordering::Relaxed) + 1;
            let key = seq.to_be_bytes();
            
            if let Ok(serialized) = bincode::serde::encode_to_vec(&fs_event, bincode::config::standard()) {
                if let Err(e) = journal.insert(&key, serialized) {
                    error!("Failed to persist event: {}", e);
                }
            }
            
            // Update last_seq in journal
            if let Err(e) = journal.insert(b"last_seq", &seq.to_be_bytes()) {
                error!("Failed to update last_seq: {}", e);
            }
        }
    }
    
    /// Normalize a path to be relative to the root
    fn normalize_path(path: &Path, root: &Path) -> Option<PathBuf> {
        path.strip_prefix(root).ok().map(|p| p.to_path_buf())
    }
}

impl Drop for FsWatcher {
    fn drop(&mut self) {
        // Stop watching
        let _ = self.stop();
        
        // Flush journal
        if let Err(e) = self.journal.flush() {
            error!("Failed to flush event journal: {}", e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_fs_watcher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let watcher = FsWatcher::new(temp_dir.path()).unwrap();
        
        assert!(!watcher.has_events());
        assert!(!watcher.missed_events());
    }
    
    #[test]
    fn test_fs_watcher_events() {
        let temp_dir = TempDir::new().unwrap();
        let watcher = FsWatcher::new(temp_dir.path()).unwrap();
        
        // Start watching
        watcher.watch().unwrap();
        
        // Give the watcher time to initialize
        thread::sleep(Duration::from_millis(100));
        
        // Create a file
        let test_file = temp_dir.path().join("test.txt");
        fs::write(&test_file, "test content").unwrap();
        
        // Give the watcher time to detect the change
        thread::sleep(Duration::from_secs(1));
        
        // Check that we have events
        assert!(watcher.has_events(), "No events detected after file creation");
        
        // Drain events
        let events = watcher.drain().unwrap();
        assert!(!events.is_empty(), "Drained events list is empty");
        
        // Should have no events after draining
        assert!(!watcher.has_events());
        
        // Stop watching
        watcher.stop().unwrap();
    }
    
    #[test]
    fn test_normalize_path() {
        let root = PathBuf::from("/home/user/project");
        let full_path = PathBuf::from("/home/user/project/src/main.rs");
        let normalized = FsWatcher::normalize_path(&full_path, &root);
        
        assert_eq!(normalized, Some(PathBuf::from("src/main.rs")));
    }
}