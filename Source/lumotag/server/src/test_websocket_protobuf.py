import asyncio
import sys
import os
import subprocess

# Check for websockets dependency
try:
    import websockets
except ImportError:
    print("Error: websockets not installed. "
          "Install with: pip install websockets")
    sys.exit(1)

# Add the parent directory to the path to import the generated protobuf
sys.path.append(os.path.join(os.path.dirname(__file__), '..',
                             'generated_protobuffs'))

def generate_protobuf_if_needed():
    """Generate protobuf files if they don't exist or are outdated."""
    proto_file = os.path.join(os.path.dirname(__file__), '..', 'common',
                              'messages.proto')
    output_dir = os.path.join(os.path.dirname(__file__), '..',
                              'generated_protobuffs')
    output_file = os.path.join(output_dir, 'comms_pb2.py')
    
    # Check if output file exists and is newer than proto file
    if os.path.exists(output_file):
        proto_mtime = os.path.getmtime(proto_file)
        output_mtime = os.path.getmtime(output_file)
        if output_mtime > proto_mtime:
            print("Protobuf files are up to date")
            return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating protobuf files...")
    try:
        subprocess.run([
            'protoc',
            f'--python_out={output_dir}',
            proto_file
        ], capture_output=True, text=True, check=True)
        print("Protobuf generation successful")
    except subprocess.CalledProcessError as e:
        print(f"Error generating protobuf files: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: protoc not found. Please install protobuf compiler:")
        print("  macOS: brew install protobuf")
        print("  Ubuntu: sudo apt-get install protobuf-compiler")
        print("  Or install via pip: pip install grpcio-tools")
        sys.exit(1)


# Generate protobuf files if needed
generate_protobuf_if_needed()

try:
    import comms_pb2
except ImportError:
    print("Error: Could not import comms_pb2. "
          "Make sure protobuf is generated.")
    print("Run: protoc --python_out=../generated_protobuffs "
          "../common/messages.proto")
    sys.exit(1)


async def test_protobuf_websocket():
    uri = "ws://127.0.0.1:8080"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to {uri}")
            
            # Create a player connection message
            connection_msg = comms_pb2.GameMessage()
            connection_msg.timestamp = 1234567890
            connection_msg.connection.player_id = "player123"
            connection_msg.connection.player_name = "TestPlayer"
            connection_msg.connection.model = comms_pb2.DEFAULT
            connection_msg.connection.team = comms_pb2.RED
            
            # Create a tag event message
            tag_msg = comms_pb2.GameMessage()
            tag_msg.timestamp = 1234567891
            tag_msg.tag.tagger_id = "player123"
            tag_msg.tag.tagged_id = "player456"
            tag_msg.tag.image_id = "tag_image_001"
            
            # Create a tag image message
            image_msg = comms_pb2.GameMessage()
            image_msg.timestamp = 1234567892
            image_msg.tag_image.image_id = "tag_image_001"
            image_msg.tag_image.player_id = "player456"
            image_msg.tag_image.image_data = b"fake_image_data_here"
            
            # Create a game status message
            status_msg = comms_pb2.GameMessage()
            status_msg.timestamp = 1234567893
            status_msg.game_status.red_team_score = 10
            status_msg.game_status.blue_team_score = 5
            status_msg.game_status.time_remaining = 300  # 5 minutes
            
            # Add a player status
            player_status = status_msg.game_status.players.add()
            player_status.player_id = "player123"
            player_status.hitpoints = 100
            player_status.active_power_up = comms_pb2.NONE
            player_status.team = comms_pb2.RED
            player_status.is_alive = True
            player_status.score = 10
            
            # Send connection message
            print("Sending player connection message...")
            await websocket.send(connection_msg.SerializeToString())
            await asyncio.sleep(1)
            
            # Send tag event message
            print("Sending tag event message...")
            await websocket.send(tag_msg.SerializeToString())
            await asyncio.sleep(1)
            
            # Send tag image message
            print("Sending tag image message...")
            await websocket.send(image_msg.SerializeToString())
            await asyncio.sleep(1)
            
            # Send game status message
            print("Sending game status message...")
            await websocket.send(status_msg.SerializeToString())
            
            # Listen for any responses
            print("Listening for responses...")
            try:
                async for message in websocket:
                    print(f"Received: {len(message)} bytes")
                    if isinstance(message, bytes):
                        try:
                            # Try to decode as protobuf
                            decoded = comms_pb2.GameMessage()
                            decoded.ParseFromString(message)
                            print(f"Decoded protobuf: {decoded}")
                        except Exception as e:
                            print(f"Could not decode as protobuf: {e}")
                    else:
                        print(f"Text message: {message}")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server")
                
    except Exception as e:
        print(f"Error connecting to WebSocket server: {e}")


if __name__ == "__main__":
    asyncio.run(test_protobuf_websocket()) 