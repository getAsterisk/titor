//! # Titor MCP Server - Time Travel for Your Files via Model Context Protocol
//!
//! An MCP server that exposes Titor checkpoint functionality to LLM agents.
//! This allows AI assistants to manage file versioning and time-travel capabilities.
//!
//! ## Features
//! - Initialize Titor repositories
//! - Create and manage checkpoints
//! - Navigate timeline history
//! - Compare changes between checkpoints
//! - Verify checkpoint integrity
//! - Optimize storage with garbage collection
//!
//! ## Usage with Claude Desktop
//! ```json
//! {
//!   "mcpServers": {
//!     "titor": {
//!       "command": "path/to/titor_mcp_server",
//!       "args": []
//!     }
//!   }
//! }
//! ```

use anyhow::Result;
use rmcp::{
    Error as McpError, RoleServer, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, tool::Parameters},
    model::*,
    schemars,
    service::RequestContext,
    tool, tool_handler, tool_router,
    transport::stdio,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use titor::{
    CompressionStrategy, Titor, TitorBuilder, TitorError,
};
use tokio::sync::RwLock;
use tracing_subscriber::{self, EnvFilter};

/// Thread-safe storage for Titor instances
/// Maps workspace paths to Titor instances
type TitorStore = Arc<RwLock<HashMap<String, Arc<RwLock<Titor>>>>>;

/// Titor MCP Server
#[derive(Clone)]
pub struct TitorMcpServer {
    store: TitorStore,
    tool_router: ToolRouter<Self>,
}

// Parameter structures for tools

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct InitParams {
    #[schemars(description = "Root directory path to track")]
    pub root_path: String,
    #[schemars(description = "Storage directory path (defaults to .titor)")]
    pub storage_path: Option<String>,
    #[schemars(description = "Compression strategy: none, fast, or adaptive")]
    pub compression: Option<String>,
    #[schemars(description = "Patterns to ignore (gitignore syntax)")]
    pub ignore_patterns: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct CheckpointParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "Optional checkpoint description")]
    pub message: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RestoreParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "Checkpoint ID to restore to")]
    pub checkpoint_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ListParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "Maximum number of checkpoints to return")]
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TimelineParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ForkParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "Checkpoint ID to fork from")]
    pub checkpoint_id: String,
    #[schemars(description = "Optional fork description")]
    pub message: Option<String>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct DiffParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "From checkpoint ID")]
    pub from_id: String,
    #[schemars(description = "To checkpoint ID")]
    pub to_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct VerifyParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "Checkpoint ID to verify (optional, defaults to current)")]
    pub checkpoint_id: Option<String>,
    #[schemars(description = "Verify all checkpoints")]
    pub verify_all: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GcParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "Perform dry run without deleting")]
    pub dry_run: Option<bool>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct StatusParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct InfoParams {
    #[schemars(description = "Root directory path")]
    pub root_path: String,
    #[schemars(description = "Checkpoint ID")]
    pub checkpoint_id: String,
}

// Response structures

#[derive(Debug, Serialize)]
pub struct InitResponse {
    pub success: bool,
    pub root_path: String,
    pub storage_path: String,
}

#[derive(Debug, Serialize)]
pub struct CheckpointResponse {
    pub id: String,
    pub message: Option<String>,
    pub file_count: usize,
    pub total_size: u64,
    pub files_changed: usize,
}

#[derive(Debug, Serialize)]
pub struct RestoreResponse {
    pub success: bool,
    pub files_restored: usize,
    pub files_deleted: usize,
    pub bytes_written: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct CheckpointInfo {
    pub id: String,
    pub timestamp: String,
    pub description: Option<String>,
    pub file_count: usize,
    pub total_size: u64,
    pub parent_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct TimelineResponse {
    pub checkpoints: Vec<CheckpointInfo>,
    pub current_checkpoint_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct DiffResponse {
    pub files_added: usize,
    pub files_modified: usize,
    pub files_deleted: usize,
    pub bytes_added: u64,
    pub bytes_modified: u64,
    pub bytes_deleted: u64,
}

#[derive(Debug, Serialize)]
pub struct VerifyResponse {
    pub valid: bool,
    pub metadata_valid: bool,
    pub state_hash_valid: bool,
    pub merkle_root_valid: bool,
    pub parent_valid: bool,
    pub errors: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct GcResponse {
    pub objects_deleted: usize,
    pub bytes_reclaimed: u64,
    pub unreferenced_objects: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub initialized: bool,
    pub current_checkpoint: Option<CheckpointInfo>,
    pub total_checkpoints: usize,
    pub storage_size: u64,
}

// Implementation continues in next part...

#[tool_router]
impl TitorMcpServer {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            tool_router: Self::tool_router(),
        }
    }

    /// Get or create a Titor instance for a workspace
    async fn get_titor(&self, root_path: &str) -> Result<Arc<RwLock<Titor>>, McpError> {
        let root_path_buf = PathBuf::from(root_path);
        let storage_path = root_path_buf.join(".titor");
        
        let mut store = self.store.write().await;
        
        if let Some(titor) = store.get(root_path) {
            Ok(titor.clone())
        } else {
            // Try to open existing Titor instance
            match Titor::open(root_path_buf.clone(), storage_path) {
                Ok(titor) => {
                    let titor_arc = Arc::new(RwLock::new(titor));
                    store.insert(root_path.to_string(), titor_arc.clone());
                    Ok(titor_arc)
                }
                Err(_) => {
                    Err(McpError::invalid_params(
                        "Titor not initialized in this directory. Run titor_init first.",
                        None
                    ))
                }
            }
        }
    }

    /// Initialize Titor in a directory
    #[tool(description = "Initialize Titor checkpointing in a directory")]
    async fn titor_init(
        &self,
        Parameters(params): Parameters<InitParams>,
    ) -> Result<CallToolResult, McpError> {
        let root_path = PathBuf::from(&params.root_path);
        let storage_path = params.storage_path
            .map(PathBuf::from)
            .unwrap_or_else(|| root_path.join(".titor"));

        // Parse compression strategy
        let compression = match params.compression.as_deref() {
            Some("none") => CompressionStrategy::None,
            Some("fast") => CompressionStrategy::Fast,
            Some("adaptive") | None => CompressionStrategy::Adaptive {
                min_size: 4096,
                skip_extensions: vec![
                    "jpg", "jpeg", "png", "gif", "mp4", "mp3",
                    "zip", "gz", "bz2", "7z", "rar"
                ].iter().map(|s| s.to_string()).collect(),
            },
            _ => return Err(McpError::invalid_params(
                "Invalid compression strategy. Use 'none', 'fast', or 'adaptive'.",
                None
            )),
        };

        // Create Titor instance
        let titor = TitorBuilder::new()
            .compression_strategy(compression)
            .ignore_patterns(params.ignore_patterns.unwrap_or_default())
            .build(root_path.clone(), storage_path.clone())
            .map_err(|e| McpError::internal_error(
                format!("Failed to initialize Titor: {}", e),
                None
            ))?;

        // Store the instance
        let mut store = self.store.write().await;
        store.insert(params.root_path.clone(), Arc::new(RwLock::new(titor)));

        let response = InitResponse {
            success: true,
            root_path: params.root_path,
            storage_path: storage_path.to_string_lossy().to_string(),
        };

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&response).unwrap())
        ]))
    }

    /// Create a checkpoint
    #[tool(description = "Create a checkpoint of the current directory state")]
    async fn titor_checkpoint(
        &self,
        Parameters(params): Parameters<CheckpointParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let mut titor = titor_lock.write().await;
        
        let checkpoint = titor.checkpoint(params.message.clone())
            .map_err(|e| McpError::internal_error(
                format!("Failed to create checkpoint: {}", e),
                None
            ))?;

        let response = CheckpointResponse {
            id: checkpoint.id.clone(),
            message: params.message,
            file_count: checkpoint.metadata.file_count,
            total_size: checkpoint.metadata.total_size,
            files_changed: checkpoint.metadata.files_changed,
        };

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&response).unwrap())
        ]))
    }

    /// Restore to a checkpoint
    #[tool(description = "Restore directory to a previous checkpoint state")]
    async fn titor_restore(
        &self,
        Parameters(params): Parameters<RestoreParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let mut titor = titor_lock.write().await;
        
        // Find full checkpoint ID from prefix
        let checkpoints = titor.list_checkpoints()
            .map_err(|e| McpError::internal_error(format!("Failed to list checkpoints: {}", e), None))?;
        
        let full_id = checkpoints
            .iter()
            .find(|c| c.id.starts_with(&params.checkpoint_id))
            .map(|c| c.id.clone())
            .ok_or_else(|| McpError::invalid_params(
                format!("Checkpoint not found: {}", params.checkpoint_id),
                None
            ))?;

        let result = titor.restore(&full_id)
            .map_err(|e| McpError::internal_error(
                format!("Failed to restore checkpoint: {}", e),
                None
            ))?;

        let response = RestoreResponse {
            success: true,
            files_restored: result.files_restored,
            files_deleted: result.files_deleted,
            bytes_written: result.bytes_written,
            warnings: result.warnings,
        };

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&response).unwrap())
        ]))
    }

    /// List checkpoints
    #[tool(description = "List all checkpoints in the repository")]
    async fn titor_list(
        &self,
        Parameters(params): Parameters<ListParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let titor = titor_lock.read().await;
        
        let checkpoints = titor.list_checkpoints()
            .map_err(|e| McpError::internal_error(format!("Failed to list checkpoints: {}", e), None))?;

        let limit = params.limit.unwrap_or(checkpoints.len());
        let checkpoint_infos: Vec<CheckpointInfo> = checkpoints
            .iter()
            .take(limit)
            .map(|c| CheckpointInfo {
                id: c.id.clone(),
                timestamp: c.timestamp.to_rfc3339(),
                description: c.description.clone(),
                file_count: c.metadata.file_count,
                total_size: c.metadata.total_size,
                parent_id: c.parent_id.clone(),
            })
            .collect();

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&checkpoint_infos).unwrap())
        ]))
    }

    /// Show timeline
    #[tool(description = "Show the checkpoint timeline as a tree structure")]
    async fn titor_timeline(
        &self,
        Parameters(params): Parameters<TimelineParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let titor = titor_lock.read().await;
        
        let timeline = titor.get_timeline()
            .map_err(|e| McpError::internal_error(format!("Failed to get timeline: {}", e), None))?;

        let checkpoints: Vec<CheckpointInfo> = timeline.checkpoints
            .values()
            .map(|c| CheckpointInfo {
                id: c.id.clone(),
                timestamp: c.timestamp.to_rfc3339(),
                description: c.description.clone(),
                file_count: c.metadata.file_count,
                total_size: c.metadata.total_size,
                parent_id: c.parent_id.clone(),
            })
            .collect();

        let response = TimelineResponse {
            checkpoints,
            current_checkpoint_id: timeline.current_checkpoint_id,
        };

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&response).unwrap())
        ]))
    }

    /// Fork from a checkpoint
    #[tool(description = "Create a new branch from an existing checkpoint")]
    async fn titor_fork(
        &self,
        Parameters(params): Parameters<ForkParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let mut titor = titor_lock.write().await;
        
        // Find full checkpoint ID from prefix
        let checkpoints = titor.list_checkpoints()
            .map_err(|e| McpError::internal_error(format!("Failed to list checkpoints: {}", e), None))?;
        
        let full_id = checkpoints
            .iter()
            .find(|c| c.id.starts_with(&params.checkpoint_id))
            .map(|c| c.id.clone())
            .ok_or_else(|| McpError::invalid_params(
                format!("Checkpoint not found: {}", params.checkpoint_id),
                None
            ))?;

        let fork = titor.fork(&full_id, params.message.clone())
            .map_err(|e| McpError::internal_error(
                format!("Failed to fork checkpoint: {}", e),
                None
            ))?;

        let response = CheckpointResponse {
            id: fork.id.clone(),
            message: params.message,
            file_count: fork.metadata.file_count,
            total_size: fork.metadata.total_size,
            files_changed: fork.metadata.files_changed,
        };

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&response).unwrap())
        ]))
    }

    // Additional tools will be implemented in the next part...

    /// Compare two checkpoints
    #[tool(description = "Show differences between two checkpoints")]
    async fn titor_diff(
        &self,
        Parameters(params): Parameters<DiffParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let titor = titor_lock.read().await;
        
        // Resolve shortened IDs to full checkpoint IDs
        let checkpoints = titor.list_checkpoints()
            .map_err(|e| McpError::internal_error(format!("Failed to list checkpoints: {}", e), None))?;
        
        let resolve_id = |prefix: &str| {
            checkpoints
                .iter()
                .find(|c| c.id.starts_with(prefix))
                .map(|c| c.id.clone())
                .ok_or_else(|| McpError::invalid_params(
                    format!("Checkpoint not found: {}", prefix),
                    None
                ))
        };

        let from_id = resolve_id(&params.from_id)?;
        let to_id = resolve_id(&params.to_id)?;

        let diff = titor.diff(&from_id, &to_id)
            .map_err(|e| McpError::internal_error(
                format!("Failed to compare checkpoints: {}", e),
                None
            ))?;

        let response = DiffResponse {
            files_added: diff.added_files.len(),
            files_modified: diff.modified_files.len(),
            files_deleted: diff.deleted_files.len(),
            bytes_added: diff.stats.bytes_added,
            bytes_modified: diff.stats.bytes_modified,
            bytes_deleted: diff.stats.bytes_deleted,
        };

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&response).unwrap())
        ]))
    }

    /// Verify checkpoint integrity
    #[tool(description = "Verify the integrity of checkpoints")]
    async fn titor_verify(
        &self,
        Parameters(params): Parameters<VerifyParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let titor = titor_lock.read().await;
        
        if params.verify_all.unwrap_or(false) {
            let report = titor.verify_timeline()
                .map_err(|e| McpError::internal_error(
                    format!("Failed to verify timeline: {}", e),
                    None
                ))?;

            let response = json!({
                "timeline_valid": report.timeline_structure_valid,
                "total_checkpoints": report.total_checkpoints,
                "valid_checkpoints": report.valid_checkpoints,
                "invalid_checkpoints": report.invalid_checkpoints,
                "no_hash_conflicts": report.no_hash_conflicts,
                "verification_time_ms": report.verification_time_ms,
            });

            Ok(CallToolResult::success(vec![
                Content::text(serde_json::to_string_pretty(&response).unwrap())
            ]))
        } else {
            let checkpoint_id = match params.checkpoint_id {
                Some(id) => {
                    // Resolve shortened ID
                    let checkpoints = titor.list_checkpoints()
                        .map_err(|e| McpError::internal_error(format!("Failed to list checkpoints: {}", e), None))?;
                    
                    checkpoints
                        .iter()
                        .find(|c| c.id.starts_with(&id))
                        .map(|c| c.id.clone())
                        .ok_or_else(|| McpError::invalid_params(
                            format!("Checkpoint not found: {}", id),
                            None
                        ))?
                }
                None => {
                    titor.get_timeline()
                        .map_err(|e| McpError::internal_error(format!("Failed to get timeline: {}", e), None))?
                        .current_checkpoint_id
                        .ok_or_else(|| McpError::invalid_params("No current checkpoint", None))?
                }
            };

            let report = titor.verify_checkpoint(&checkpoint_id)
                .map_err(|e| McpError::internal_error(
                    format!("Failed to verify checkpoint: {}", e),
                    None
                ))?;

            let response = VerifyResponse {
                valid: report.metadata_valid && report.state_hash_valid && report.merkle_root_valid,
                metadata_valid: report.metadata_valid,
                state_hash_valid: report.state_hash_valid,
                merkle_root_valid: report.merkle_root_valid,
                parent_valid: report.parent_valid,
                errors: vec![], // Could be populated from report if needed
            };

            Ok(CallToolResult::success(vec![
                Content::text(serde_json::to_string_pretty(&response).unwrap())
            ]))
        }
    }

    /// Run garbage collection
    #[tool(description = "Remove unreferenced objects to free storage space")]
    async fn titor_gc(
        &self,
        Parameters(params): Parameters<GcParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let titor = titor_lock.read().await;
        
        let dry_run = params.dry_run.unwrap_or(false);
        
        if dry_run {
            let stats = titor.gc_analyze()
                .map_err(|e| McpError::internal_error(
                    format!("Failed to analyze garbage collection: {}", e),
                    None
                ))?;

            let response = GcResponse {
                objects_deleted: 0,
                bytes_reclaimed: stats.bytes_reclaimed,
                unreferenced_objects: Some(stats.unreferenced_objects),
            };

            Ok(CallToolResult::success(vec![
                Content::text(serde_json::to_string_pretty(&response).unwrap())
            ]))
        } else {
            let stats = titor.gc()
                .map_err(|e| McpError::internal_error(
                    format!("Failed to run garbage collection: {}", e),
                    None
                ))?;

            let response = GcResponse {
                objects_deleted: stats.objects_deleted,
                bytes_reclaimed: stats.bytes_reclaimed,
                unreferenced_objects: None,
            };

            Ok(CallToolResult::success(vec![
                Content::text(serde_json::to_string_pretty(&response).unwrap())
            ]))
        }
    }

    /// Get repository status
    #[tool(description = "Get the current status of the Titor repository")]
    async fn titor_status(
        &self,
        Parameters(params): Parameters<StatusParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = match self.get_titor(&params.root_path).await {
            Ok(lock) => lock,
            Err(_) => {
                // Repository not initialized
                let response = StatusResponse {
                    initialized: false,
                    current_checkpoint: None,
                    total_checkpoints: 0,
                    storage_size: 0,
                };
                
                return Ok(CallToolResult::success(vec![
                    Content::text(serde_json::to_string_pretty(&response).unwrap())
                ]));
            }
        };
        
        let titor = titor_lock.read().await;
        let timeline = titor.get_timeline()
            .map_err(|e| McpError::internal_error(format!("Failed to get timeline: {}", e), None))?;

        let current_checkpoint = if let Some(id) = &timeline.current_checkpoint_id {
            timeline.checkpoints.get(id).map(|c| CheckpointInfo {
                id: c.id.clone(),
                timestamp: c.timestamp.to_rfc3339(),
                description: c.description.clone(),
                file_count: c.metadata.file_count,
                total_size: c.metadata.total_size,
                parent_id: c.parent_id.clone(),
            })
        } else {
            None
        };

        // TODO: Calculate actual storage size
        let storage_size = 0u64;

        let response = StatusResponse {
            initialized: true,
            current_checkpoint,
            total_checkpoints: timeline.checkpoints.len(),
            storage_size,
        };

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&response).unwrap())
        ]))
    }

    /// Get detailed checkpoint information
    #[tool(description = "Get detailed information about a specific checkpoint")]
    async fn titor_info(
        &self,
        Parameters(params): Parameters<InfoParams>,
    ) -> Result<CallToolResult, McpError> {
        let titor_lock = self.get_titor(&params.root_path).await?;
        let titor = titor_lock.read().await;
        
        let checkpoints = titor.list_checkpoints()
            .map_err(|e| McpError::internal_error(format!("Failed to list checkpoints: {}", e), None))?;
        
        let checkpoint = checkpoints
            .iter()
            .find(|c| c.id.starts_with(&params.checkpoint_id))
            .ok_or_else(|| McpError::invalid_params(
                format!("Checkpoint not found: {}", params.checkpoint_id),
                None
            ))?;

        let info = json!({
            "id": checkpoint.id,
            "timestamp": checkpoint.timestamp.to_rfc3339(),
            "description": checkpoint.description,
            "parent_id": checkpoint.parent_id,
            "metadata": {
                "file_count": checkpoint.metadata.file_count,
                "total_size": checkpoint.metadata.total_size,
                "compressed_size": checkpoint.metadata.compressed_size,
                "files_changed": checkpoint.metadata.files_changed,
                "host_info": {
                    "hostname": checkpoint.metadata.host_info.hostname,
                    "username": checkpoint.metadata.host_info.username,
                },
                "titor_version": checkpoint.metadata.titor_version,
            },
            "state_hash": checkpoint.state_hash,
            "content_merkle_root": checkpoint.content_merkle_root,
        });

        Ok(CallToolResult::success(vec![
            Content::text(serde_json::to_string_pretty(&info).unwrap())
        ]))
    }
}

// ServerHandler implementation
#[tool_handler]
impl ServerHandler for TitorMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
            server_info: Implementation {
                name: "titor-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(
                "Titor MCP Server - Time travel for your files via Model Context Protocol. \
                This server provides checkpointing and version control capabilities for directories. \
                Use titor_init to initialize a repository, titor_checkpoint to create snapshots, \
                and titor_restore to time travel to previous states.".to_string()
            ),
        }
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParam>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        // List all tracked repositories
        let store = self.store.read().await;
        let resources: Vec<Resource> = store.keys()
            .map(|path| {
                RawResource::new(
                    &format!("titor://{}", path),
                    format!("Titor repository at {}", path)
                ).no_annotation()
            })
            .collect();

        Ok(ListResourcesResult {
            resources,
            next_cursor: None,
        })
    }

    async fn read_resource(
        &self,
        ReadResourceRequestParam { uri }: ReadResourceRequestParam,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        if let Some(path) = uri.strip_prefix("titor://") {
            match self.get_titor(path).await {
                Ok(titor_lock) => {
                    let titor = titor_lock.read().await;
                    let timeline = titor.get_timeline()
                        .map_err(|e| McpError::internal_error(format!("Failed to get timeline: {}", e), None))?;
                    
                    let content = json!({
                        "path": path,
                        "total_checkpoints": timeline.checkpoints.len(),
                        "current_checkpoint_id": timeline.current_checkpoint_id,
                    });

                    Ok(ReadResourceResult {
                        contents: vec![ResourceContents::text(
                            serde_json::to_string_pretty(&content).unwrap(),
                            uri
                        )],
                    })
                }
                Err(_) => {
                    Err(McpError::resource_not_found(
                        "Titor repository not found",
                        Some(json!({"uri": uri}))
                    ))
                }
            }
        } else {
            Err(McpError::invalid_params(
                "Invalid resource URI. Expected titor://path format",
                None
            ))
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to stderr
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    tracing::info!("Starting Titor MCP Server");

    // Create server instance
    let server = TitorMcpServer::new();
    
    // Serve on stdio
    let service = server.serve(stdio()).await?;
    
    // Wait for shutdown
    service.waiting().await?;
    
    Ok(())
}