//! Timeline management for checkpoint navigation
//!
//! This module provides the timeline structure that maintains the tree
//! of checkpoints, enabling navigation between different points in time
//! and management of branching timelines.
//!
//! ## Overview
//!
//! The timeline is a directed acyclic graph (DAG) that tracks relationships
//! between checkpoints. It supports:
//!
//! - **Linear History**: Simple parent-child relationships
//! - **Branching**: Multiple checkpoints can have the same parent
//! - **Navigation**: Finding paths between checkpoints
//! - **Analysis**: Common ancestors, descendants, and statistics
//!
//! ## Structure
//!
//! ```text
//! root
//! ├── checkpoint-1
//! │   ├── checkpoint-2
//! │   └── checkpoint-3 (fork)
//! │       └── checkpoint-4
//! └── checkpoint-5 (another root)
//! ```
//!
//! ## Examples
//!
//! ```rust
//! use titor::timeline::Timeline;
//! use titor::checkpoint::Checkpoint;
//! # use titor::checkpoint::CheckpointMetadataBuilder;
//!
//! let mut timeline = Timeline::new();
//!
//! // Add checkpoints
//! # let checkpoint = Checkpoint::new(None, None, CheckpointMetadataBuilder::new().build(), "".to_string());
//! timeline.add_checkpoint(checkpoint)?;
//!
//! // Navigate timeline
//! let current = timeline.current_checkpoint();
//! let ancestors = timeline.get_ancestors("checkpoint-id");
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::checkpoint::Checkpoint;
use crate::error::{Result, TitorError};
use serde::{Deserialize, Serialize};
use crate::collections::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::collections::VecDeque;
use tracing::{debug, trace};

/// Timeline structure representing the checkpoint tree
///
/// Maintains the relationships between checkpoints and provides
/// navigation and analysis capabilities. The timeline ensures
/// consistency and prevents circular dependencies.
///
/// # Structure
///
/// The timeline is internally represented as:
/// - A map of all checkpoints by ID
/// - Parent-child relationships
/// - The current checkpoint pointer
/// - Root checkpoints (those without parents)
///
/// # Thread Safety
///
/// Timeline is not thread-safe. Use external synchronization
/// (like `RwLock`) when accessing from multiple threads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// All checkpoints indexed by ID
    pub checkpoints: HashMap<String, Checkpoint>,
    /// Current checkpoint ID
    pub current_checkpoint_id: Option<String>,
    /// Root checkpoint IDs (checkpoints with no parent)
    pub roots: Vec<String>,
    /// Children mapping (parent_id -> \[child_ids\])
    pub children: HashMap<String, Vec<String>>,
}

impl Timeline {
    /// Create a new empty timeline
    ///
    /// # Examples
    ///
    /// ```rust
    /// use titor::timeline::Timeline;
    ///
    /// let timeline = Timeline::new();
    /// assert!(timeline.checkpoints.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            checkpoints: HashMap::new(),
            current_checkpoint_id: None,
            roots: Vec::new(),
            children: HashMap::new(),
        }
    }
    
    /// Add a checkpoint to the timeline
    ///
    /// Adds a checkpoint and updates all relationships. The first
    /// checkpoint added becomes the current checkpoint automatically.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The checkpoint to add
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Adding the checkpoint would create a circular dependency
    /// - The parent checkpoint doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::timeline::Timeline;
    /// # use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
    /// let mut timeline = Timeline::new();
    /// 
    /// let checkpoint = Checkpoint::new(
    ///     None,
    ///     Some("Initial state".to_string()),
    ///     CheckpointMetadataBuilder::new().build(),
    ///     "merkle".to_string()
    /// );
    /// 
    /// timeline.add_checkpoint(checkpoint)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn add_checkpoint(&mut self, checkpoint: Checkpoint) -> Result<()> {
        let checkpoint_id = checkpoint.id.clone();
        let parent_id = checkpoint.parent_id.clone();
        
        // Check for circular dependency
        if let Some(parent_id) = &parent_id {
            if self.would_create_cycle(&checkpoint_id, parent_id) {
                return Err(TitorError::CircularDependency);
            }
        }
        
        // Add to checkpoints map
        self.checkpoints.insert(checkpoint_id.clone(), checkpoint);
        
        // Update parent-child relationships
        if let Some(parent_id) = parent_id {
            self.children
                .entry(parent_id)
                .or_insert_with(Vec::new)
                .push(checkpoint_id.clone());
        } else {
            // This is a root checkpoint
            self.roots.push(checkpoint_id.clone());
        }
        
        // If this is the first checkpoint, make it current
        if self.current_checkpoint_id.is_none() {
            self.current_checkpoint_id = Some(checkpoint_id.clone());
        }
        
        debug!("Added checkpoint {} to timeline", &checkpoint_id[..8]);
        Ok(())
    }
    
    /// Remove a checkpoint from the timeline
    ///
    /// Removes a checkpoint if it has no children. The current checkpoint
    /// is updated to the parent if the removed checkpoint was current.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_id` - ID of the checkpoint to remove
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The checkpoint has children (cannot create orphans)
    /// - The checkpoint doesn't exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use titor::timeline::Timeline;
    /// # use titor::checkpoint::{Checkpoint, CheckpointMetadataBuilder};
    /// # let mut timeline = Timeline::new();
    /// # let cp = Checkpoint::new(None, None, CheckpointMetadataBuilder::new().build(), "".to_string());
    /// # let id = cp.id.clone();
    /// # timeline.add_checkpoint(cp)?;
    /// // Remove a leaf checkpoint
    /// timeline.remove_checkpoint(&id)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn remove_checkpoint(&mut self, checkpoint_id: &str) -> Result<()> {
        // Check if checkpoint has children
        if self.children.contains_key(checkpoint_id) {
            return Err(TitorError::CheckpointHasChildren(checkpoint_id.to_string()));
        }
        
        // Get checkpoint to find parent
        let checkpoint = self.checkpoints
            .get(checkpoint_id)
            .ok_or_else(|| TitorError::CheckpointNotFound(checkpoint_id.to_string()))?
            .clone();
        
        // Remove from checkpoints map
        self.checkpoints.remove(checkpoint_id);
        
        // Update parent's children list
        if let Some(parent_id) = &checkpoint.parent_id {
            if let Some(children) = self.children.get_mut(parent_id) {
                children.retain(|id| id != checkpoint_id);
                if children.is_empty() {
                    self.children.remove(parent_id);
                }
            }
        } else {
            // Remove from roots
            self.roots.retain(|id| id != checkpoint_id);
        }
        
        // Update current checkpoint if necessary
        if self.current_checkpoint_id.as_deref() == Some(checkpoint_id) {
            self.current_checkpoint_id = checkpoint.parent_id;
        }
        
        debug!("Removed checkpoint {} from timeline", &checkpoint_id[..8]);
        Ok(())
    }
    
    /// Set the current checkpoint
    pub fn set_current(&mut self, checkpoint_id: &str) -> Result<()> {
        if !self.checkpoints.contains_key(checkpoint_id) {
            return Err(TitorError::CheckpointNotFound(checkpoint_id.to_string()));
        }
        
        self.current_checkpoint_id = Some(checkpoint_id.to_string());
        trace!("Set current checkpoint to {}", &checkpoint_id[..8]);
        Ok(())
    }
    
    /// Get the current checkpoint
    pub fn current_checkpoint(&self) -> Option<&Checkpoint> {
        self.current_checkpoint_id
            .as_ref()
            .and_then(|id| self.checkpoints.get(id))
    }
    
    /// Get a checkpoint by ID
    pub fn get_checkpoint(&self, checkpoint_id: &str) -> Option<&Checkpoint> {
        self.checkpoints.get(checkpoint_id)
    }
    
    /// Get all direct children of a checkpoint
    pub fn get_children(&self, checkpoint_id: &str) -> Vec<&Checkpoint> {
        self.children
            .get(checkpoint_id)
            .map(|child_ids| {
                child_ids
                    .iter()
                    .filter_map(|id| self.checkpoints.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Get all descendants of a checkpoint
    pub fn get_descendants(&self, checkpoint_id: &str) -> Vec<&Checkpoint> {
        let mut descendants = Vec::new();
        let mut queue = VecDeque::new();
        
        // Start with direct children
        if let Some(children) = self.children.get(checkpoint_id) {
            queue.extend(children.iter());
        }
        
        // BFS to find all descendants
        while let Some(id) = queue.pop_front() {
            if let Some(checkpoint) = self.checkpoints.get(id) {
                descendants.push(checkpoint);
                
                // Add this checkpoint's children to queue
                if let Some(children) = self.children.get(id) {
                    queue.extend(children.iter());
                }
            }
        }
        
        descendants
    }
    
    /// Get all ancestors of a checkpoint (from root to parent)
    pub fn get_ancestors(&self, checkpoint_id: &str) -> Vec<&Checkpoint> {
        let mut ancestors = Vec::new();
        let mut current_id = checkpoint_id;
        
        while let Some(checkpoint) = self.checkpoints.get(current_id) {
            if let Some(parent_id) = &checkpoint.parent_id {
                if let Some(parent) = self.checkpoints.get(parent_id) {
                    ancestors.push(parent);
                    current_id = parent_id;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        
        ancestors.reverse();
        ancestors
    }
    
    /// Find common ancestor of two checkpoints
    pub fn find_common_ancestor(&self, id1: &str, id2: &str) -> Option<&Checkpoint> {
        let ancestors1: HashSet<_> = self.get_ancestors(id1)
            .into_iter()
            .map(|c| &c.id)
            .collect();
        
        // Walk up from id2 until we find a common ancestor
        let mut current_id = id2;
        
        loop {
            if ancestors1.contains(&current_id.to_string()) {
                return self.checkpoints.get(current_id);
            }
            
            let checkpoint = self.checkpoints.get(current_id)?;
            match &checkpoint.parent_id {
                Some(parent_id) => current_id = parent_id,
                None => return None,
            }
        }
    }
    
    /// Get the path between two checkpoints
    ///
    /// Finds the sequence of checkpoints that connect two points in the
    /// timeline. This is useful for understanding the changes needed to
    /// move from one state to another.
    ///
    /// # Arguments
    ///
    /// * `from_id` - Starting checkpoint ID
    /// * `to_id` - Target checkpoint ID
    ///
    /// # Returns
    ///
    /// A vector of checkpoints representing the path, including both
    /// endpoints. Empty vector if from and to are the same.
    ///
    /// # Errors
    ///
    /// Returns an error if no path exists between the checkpoints.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use titor::timeline::Timeline;
    /// # let timeline = Timeline::new();
    /// // Find path between two checkpoints
    /// let path = timeline.get_path("old-checkpoint", "new-checkpoint")?;
    /// 
    /// for checkpoint in path {
    ///     println!("Step through: {}", checkpoint.short_id());
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_path(&self, from_id: &str, to_id: &str) -> Result<Vec<&Checkpoint>> {
        if from_id == to_id {
            return Ok(vec![]);
        }
        
        // Check if direct ancestor/descendant relationship
        let from_ancestors = self.get_ancestors(from_id);
        let to_ancestors = self.get_ancestors(to_id);
        
        // If to is ancestor of from
        if from_ancestors.iter().any(|c| c.id == to_id) {
            return Ok(from_ancestors
                .into_iter()
                .take_while(|c| c.id != to_id)
                .chain(std::iter::once(self.checkpoints.get(to_id).unwrap()))
                .collect());
        }
        
        // If from is ancestor of to
        if to_ancestors.iter().any(|c| c.id == from_id) {
            return Ok(to_ancestors
                .into_iter()
                .skip_while(|c| c.id != from_id)
                .skip(1)
                .chain(std::iter::once(self.checkpoints.get(to_id).unwrap()))
                .collect());
        }
        
        // Find common ancestor and build path
        if let Some(common) = self.find_common_ancestor(from_id, to_id) {
            let mut path = Vec::new();
            
            // Path from 'from' to common ancestor (reverse order)
            let mut current_id = from_id;
            while current_id != &common.id {
                let checkpoint = self.checkpoints.get(current_id).unwrap();
                path.push(checkpoint);
                current_id = checkpoint.parent_id.as_ref().unwrap();
            }
            
            // Add common ancestor
            path.push(common);
            
            // Path from common ancestor to 'to'
            let to_ancestors_from_common: Vec<_> = self.get_ancestors(to_id)
                .into_iter()
                .skip_while(|c| c.id != common.id)
                .skip(1)
                .collect();
            
            path.extend(to_ancestors_from_common);
            path.push(self.checkpoints.get(to_id).unwrap());
            
            Ok(path)
        } else {
            Err(TitorError::internal(format!(
                "No path found between {} and {}",
                from_id, to_id
            )))
        }
    }
    
    /// Check if adding a parent relationship would create a cycle
    fn would_create_cycle(&self, child_id: &str, parent_id: &str) -> bool {
        if child_id == parent_id {
            return true;
        }
        
        // Check if parent is a descendant of child
        let mut current_id = parent_id;
        let mut visited = HashSet::new();
        
        while let Some(checkpoint) = self.checkpoints.get(current_id) {
            if !visited.insert(current_id) {
                // Already visited, cycle detected
                return true;
            }
            
            if current_id == child_id {
                return true;
            }
            
            match &checkpoint.parent_id {
                Some(id) => current_id = id,
                None => break,
            }
        }
        
        false
    }
    
    /// Get timeline statistics
    pub fn stats(&self) -> TimelineStats {
        let total_checkpoints = self.checkpoints.len();
        let root_checkpoints = self.roots.len();
        let leaf_checkpoints = self.checkpoints
            .keys()
            .filter(|id| !self.children.contains_key(*id))
            .count();
        
        let max_depth = self.roots
            .iter()
            .map(|root| self.calculate_depth(root))
            .max()
            .unwrap_or(0);
        
        let total_branches = self.children
            .values()
            .filter(|children| children.len() > 1)
            .count();
        
        TimelineStats {
            total_checkpoints,
            root_checkpoints,
            leaf_checkpoints,
            max_depth,
            total_branches,
        }
    }
    
    /// Calculate the maximum depth from a given checkpoint
    fn calculate_depth(&self, checkpoint_id: &str) -> usize {
        if let Some(children) = self.children.get(checkpoint_id) {
            1 + children
                .iter()
                .map(|child| self.calculate_depth(child))
                .max()
                .unwrap_or(0)
        } else {
            1
        }
    }
    
    /// Convert timeline to tree nodes for visualization
    pub fn to_tree_nodes(&self) -> Vec<TimelineNode> {
        self.roots
            .iter()
            .map(|root_id| self.build_tree_node(root_id))
            .collect()
    }
    
    /// Build a tree node recursively
    fn build_tree_node(&self, checkpoint_id: &str) -> TimelineNode {
        let checkpoint = self.checkpoints.get(checkpoint_id).unwrap();
        let children = self.children
            .get(checkpoint_id)
            .map(|child_ids| {
                child_ids
                    .iter()
                    .map(|id| self.build_tree_node(id))
                    .collect()
            })
            .unwrap_or_default();
        
        TimelineNode {
            checkpoint: checkpoint.clone(),
            children,
            is_current: self.current_checkpoint_id.as_deref() == Some(checkpoint_id),
        }
    }
}

/// Node in the timeline tree for visualization
#[derive(Debug, Clone)]
pub struct TimelineNode {
    /// The checkpoint at this node
    pub checkpoint: Checkpoint,
    /// Child nodes
    pub children: Vec<TimelineNode>,
    /// Whether this is the current checkpoint
    pub is_current: bool,
}

impl TimelineNode {
    /// Format the tree for display
    pub fn format_tree(&self, prefix: &str, is_last: bool) -> String {
        let mut result = String::new();
        
        // Current node
        let connector = if is_last { "└── " } else { "├── " };
        let marker = if self.is_current { "* " } else { "" };
        
        result.push_str(prefix);
        result.push_str(connector);
        result.push_str(marker);
        result.push_str(&self.checkpoint.display_format());
        result.push('\n');
        
        // Children
        let extension = if is_last { "    " } else { "│   " };
        let child_prefix = format!("{}{}", prefix, extension);
        
        for (i, child) in self.children.iter().enumerate() {
            let is_last_child = i == self.children.len() - 1;
            result.push_str(&child.format_tree(&child_prefix, is_last_child));
        }
        
        result
    }
}

/// Timeline statistics
#[derive(Debug)]
pub struct TimelineStats {
    /// Total number of checkpoints
    pub total_checkpoints: usize,
    /// Number of root checkpoints
    pub root_checkpoints: usize,
    /// Number of leaf checkpoints (no children)
    pub leaf_checkpoints: usize,
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Number of branch points (checkpoints with multiple children)
    pub total_branches: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::CheckpointMetadataBuilder;
    
    fn create_test_checkpoint(id: &str, parent_id: Option<String>) -> Checkpoint {
        let mut checkpoint = Checkpoint::new(
            parent_id,
            Some(format!("Checkpoint {}", id)),
            CheckpointMetadataBuilder::new().build(),
            "merkle_root".to_string(),
        );
        checkpoint.id = id.to_string(); // Override for predictable IDs in tests
        checkpoint
    }
    
    #[test]
    fn test_timeline_basic_operations() {
        let mut timeline = Timeline::new();
        
        // Add root checkpoint
        let root = create_test_checkpoint("root", None);
        timeline.add_checkpoint(root).unwrap();
        
        assert_eq!(timeline.checkpoints.len(), 1);
        assert_eq!(timeline.roots.len(), 1);
        assert_eq!(timeline.current_checkpoint_id, Some("root".to_string()));
        
        // Add child checkpoint
        let child = create_test_checkpoint("child", Some("root".to_string()));
        timeline.add_checkpoint(child).unwrap();
        
        assert_eq!(timeline.checkpoints.len(), 2);
        assert_eq!(timeline.get_children("root").len(), 1);
    }
    
    #[test]
    fn test_timeline_circular_dependency() {
        let mut timeline = Timeline::new();
        
        // Add checkpoints
        let c1 = create_test_checkpoint("c1", None);
        let c2 = create_test_checkpoint("c2", Some("c1".to_string()));
        timeline.add_checkpoint(c1).unwrap();
        timeline.add_checkpoint(c2).unwrap();
        
        // Try to create circular dependency
        let circular = create_test_checkpoint("c1", Some("c2".to_string()));
        assert!(matches!(
            timeline.add_checkpoint(circular),
            Err(TitorError::CircularDependency)
        ));
    }
    
    #[test]
    fn test_timeline_remove_checkpoint() {
        let mut timeline = Timeline::new();
        
        // Create tree: root -> child1, child2
        timeline.add_checkpoint(create_test_checkpoint("root", None)).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("child1", Some("root".to_string()))).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("child2", Some("root".to_string()))).unwrap();
        
        // Cannot remove root (has children)
        assert!(matches!(
            timeline.remove_checkpoint("root"),
            Err(TitorError::CheckpointHasChildren(_))
        ));
        
        // Can remove leaf
        timeline.remove_checkpoint("child1").unwrap();
        assert_eq!(timeline.checkpoints.len(), 2);
        assert_eq!(timeline.get_children("root").len(), 1);
    }
    
    #[test]
    fn test_timeline_navigation() {
        let mut timeline = Timeline::new();
        
        // Create tree:
        //   root
        //   ├── branch1
        //   │   └── leaf1
        //   └── branch2
        timeline.add_checkpoint(create_test_checkpoint("root", None)).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("branch1", Some("root".to_string()))).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("leaf1", Some("branch1".to_string()))).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("branch2", Some("root".to_string()))).unwrap();
        
        // Test ancestors
        let ancestors = timeline.get_ancestors("leaf1");
        assert_eq!(ancestors.len(), 2);
        assert_eq!(ancestors[0].id, "root");
        assert_eq!(ancestors[1].id, "branch1");
        
        // Test descendants
        let descendants = timeline.get_descendants("root");
        assert_eq!(descendants.len(), 3);
        
        // Test common ancestor
        let common = timeline.find_common_ancestor("leaf1", "branch2").unwrap();
        assert_eq!(common.id, "root");
        
        // Test path
        let path = timeline.get_path("leaf1", "branch2").unwrap();
        assert_eq!(path.len(), 4); // leaf1 -> branch1 -> root -> branch2
    }
    
    #[test]
    fn test_timeline_stats() {
        let mut timeline = Timeline::new();
        
        // Create a more complex tree
        timeline.add_checkpoint(create_test_checkpoint("root", None)).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("b1", Some("root".to_string()))).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("b2", Some("root".to_string()))).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("l1", Some("b1".to_string()))).unwrap();
        timeline.add_checkpoint(create_test_checkpoint("l2", Some("b1".to_string()))).unwrap();
        
        let stats = timeline.stats();
        assert_eq!(stats.total_checkpoints, 5);
        assert_eq!(stats.root_checkpoints, 1);
        assert_eq!(stats.leaf_checkpoints, 3); // b2, l1, l2
        assert_eq!(stats.max_depth, 3); // root -> b1 -> l1/l2
        assert_eq!(stats.total_branches, 2); // root and b1 have multiple children
    }
} 