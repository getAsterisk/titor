# Titor MCP Server

A Model Context Protocol (MCP) server that exposes Titor checkpoint functionality to LLM agents. This allows AI assistants like Claude Desktop, Claude Code, Claudia, Cursor, Windsurf, etc. to manage file versioning and time-travel capabilities for your projects.

## Features

- **Initialize Titor repositories** - Set up checkpointing in any directory
- **Create checkpoints** - Capture directory states with optional descriptions
- **Restore checkpoints** - Time travel to previous states
- **Navigate timeline** - View checkpoint history and relationships
- **Compare changes** - Diff between checkpoints
- **Verify integrity** - Ensure checkpoint data integrity
- **Optimize storage** - Run garbage collection to free space
- **Resource access** - Query repository status via MCP resources

## Installation

### Building from Source

From the Titor project root:

```bash
# Build the MCP server
cargo build --example titor_mcp_server --features rmcp,schemars --release

# The binary will be at:
# target/release/examples/titor_mcp_server
```

## Configuration

### Claude Desktop

Add this to your Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "titor": {
      "command": "/path/to/titor_mcp_server",
      "args": []
    }
  }
}
```

Replace `/path/to/titor_mcp_server` with the actual path to your built binary.

### MCP Inspector

Test the server using the MCP Inspector:

```bash
npx @modelcontextprotocol/inspector /path/to/titor_mcp_server
```

## Usage

Once configured, you can interact with Titor through your MCP client. Here are the available tools:

### titor_init

Initialize Titor in a directory.

```
Parameters:
- root_path: Directory path to track
- storage_path: Optional storage directory (defaults to .titor)
- compression: Optional compression strategy (none, fast, adaptive)
- ignore_patterns: Optional list of patterns to ignore (gitignore syntax)
```

### titor_checkpoint

Create a checkpoint of the current directory state.

```
Parameters:
- root_path: Directory path
- message: Optional checkpoint description
```

### titor_restore

Restore directory to a previous checkpoint state.

```
Parameters:
- root_path: Directory path
- checkpoint_id: Checkpoint ID to restore to (can use prefix)
```

### titor_list

List all checkpoints in the repository.

```
Parameters:
- root_path: Directory path
- limit: Optional maximum number of checkpoints to return
```

### titor_timeline

Show the checkpoint timeline as a tree structure.

```
Parameters:
- root_path: Directory path
```

### titor_fork

Create a new branch from an existing checkpoint.

```
Parameters:
- root_path: Directory path
- checkpoint_id: Checkpoint ID to fork from (can use prefix)
- message: Optional fork description
```

### titor_diff

Show differences between two checkpoints.

```
Parameters:
- root_path: Directory path
- from_id: From checkpoint ID (can use prefix)
- to_id: To checkpoint ID (can use prefix)
```

### titor_verify

Verify the integrity of checkpoints.

```
Parameters:
- root_path: Directory path
- checkpoint_id: Optional checkpoint ID to verify (defaults to current)
- verify_all: Optional flag to verify all checkpoints
```

### titor_gc

Remove unreferenced objects to free storage space.

```
Parameters:
- root_path: Directory path
- dry_run: Optional flag to perform dry run without deleting
```

### titor_status

Get the current status of the Titor repository.

```
Parameters:
- root_path: Directory path
```

### titor_info

Get detailed information about a specific checkpoint.

```
Parameters:
- root_path: Directory path
- checkpoint_id: Checkpoint ID (can use prefix)
```

## Example Conversation with Claude

```
User: Initialize Titor in my project directory /home/user/myproject

Claude: I'll initialize Titor checkpointing in your project directory.

[Uses titor_init with root_path="/home/user/myproject"]

User: Create a checkpoint before I make major changes

Claude: I'll create a checkpoint to save the current state of your project.

[Uses titor_checkpoint with message="Before major changes"]

User: I made a mistake, can you restore to the previous checkpoint?

Claude: I'll restore your project to the previous checkpoint.

[Uses titor_list to find the checkpoint, then titor_restore]

User: Show me what changed between the last two checkpoints

Claude: I'll show you the differences between the last two checkpoints.

[Uses titor_diff to compare the checkpoints]
```

## Resources

The server also exposes Titor repositories as MCP resources:

- **URI Format**: `titor://path/to/repository`
- **Content**: JSON with repository status including total checkpoints and current checkpoint ID

## Security Considerations

- The server has full read/write access to directories where Titor is initialized
- Checkpoint data is stored in the `.titor` directory within each repository
- No authentication is built into the MCP protocol - access control depends on your system

## Troubleshooting

### "Titor not initialized" Error

Run `titor_init` first to set up checkpointing in the directory.

### Checkpoint ID Not Found

Use `titor_list` to see available checkpoints. You can use ID prefixes (first 8 characters usually sufficient).

### Permission Errors

Ensure the MCP server has read/write permissions for both the target directory and storage location.

## Architecture

The Titor MCP server:
- Maintains a thread-safe store of active Titor instances
- Automatically opens existing repositories when accessed
- Provides comprehensive error handling and user-friendly messages
- Uses JSON for all tool responses for easy parsing

## Contributing

The Titor MCP server is part of the Titor project. See the main project README for contribution guidelines. 