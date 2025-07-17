# Protocol Buffer WebSocket Server

This server uses Protocol Buffers for type-safe message serialisation between Python clients and the Rust WebSocket server.

## Setup

1. **Install dependencies**:
   ```bash
   # For Python client
   pip install websockets protobuf
   
   # For Rust server (dependencies are in Cargo.toml)
   cargo build
   ```

2. **Generate protobuf files**:
   ```bash
   # Generate Python protobuf files
   ./generate_protobuf.sh
   
   # Rust protobuf files are generated automatically during cargo build
   ```

## Protobuf Generation Options

### Option 1: Runtime Generation (Recommended for Development)
The Python code can automatically generate protobuf files when needed:

```python
# In your Python script
from protobuf_generator import import_protobuf

# This will generate protobuf files if they don't exist or are outdated
comms_pb2 = import_protobuf()
if comms_pb2 is None:
    print("Failed to generate/import protobuf files")
    sys.exit(1)
```

### Option 2: Manual Generation
Generate files manually before running:

```bash
# Generate Python protobuf files
python protobuf_generator.py

# Or use the shell script
./generate_protobuf.sh
```

### Option 3: CI/CD Generation
For production deployments, generate protobuf files in your CI/CD pipeline:

```yaml
# Example GitHub Actions step
- name: Generate Protobuf Files
  run: |
    python protobuf_generator.py
    cargo build  # Generates Rust protobuf files
```

## Running the Server

```bash
cargo run
```

The server will:
- Start HTTP server on `127.0.0.1:9001`
- Start WebSocket server on `127.0.0.1:8080`
- Accept binary protobuf messages

## Testing

Run the Python test client:
```bash
python src/test_websocket_protobuf.py
```

## Message Types

The server handles these protobuf message types:
- `PlayerConnection` - When a player joins
- `TagEvent` - When a player tags another
- `TagImage` - Image data from a tag
- `GameStatus` - Current game state

All messages are wrapped in a `GameMessage` with a timestamp and oneof payload.
