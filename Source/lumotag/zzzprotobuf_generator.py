#!/usr/bin/env python3
"""
Protobuf generator module for lumotag project.
Automatically generates Python protobuf files when needed.
"""

import subprocess
import sys
from pathlib import Path


def ensure_protobuf_generated():
    """
    Ensure protobuf files are generated and up to date.
    Returns True if generation was successful, False otherwise.
    """
    # Get the project root directory
    project_root = Path(__file__).parent
    proto_file = project_root / "common" / "messages.proto"
    output_dir = project_root / "generated_protobuffs"
    output_file = output_dir / "comms_pb2.py"
    
    # Check if output file exists and is newer than proto file
    if output_file.exists():
        proto_mtime = proto_file.stat().st_mtime
        output_mtime = output_file.stat().st_mtime
        if output_mtime > proto_mtime:
            print("Protobuf files are up to date")
            return True
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    print("Generating protobuf files...")
    try:
        subprocess.run([
            'protoc',
            f'--python_out={output_dir}',
            str(proto_file)
        ], capture_output=True, text=True, check=True)
        print("Protobuf generation successful")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating protobuf files: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: protoc not found. Please install protobuf compiler:")
        print("  macOS: brew install protobuf")
        print("  Ubuntu: sudo apt-get install protobuf-compiler")
        print("  Or install via pip: pip install grpcio-tools")
        return False


def import_protobuf():
    """
    Import the generated protobuf module.
    Returns the module if successful, None otherwise.
    """
    if not ensure_protobuf_generated():
        return None
    
    # Add generated_protobuffs to Python path
    project_root = Path(__file__).parent
    generated_dir = project_root / "generated_protobuffs"
    sys.path.insert(0, str(generated_dir))
    
    try:
        import comms_pb2
        return comms_pb2
    except ImportError as e:
        print(f"Error importing protobuf module: {e}")
        return None


if __name__ == "__main__":
    # When run directly, just generate the files
    success = ensure_protobuf_generated()
    sys.exit(0 if success else 1) 