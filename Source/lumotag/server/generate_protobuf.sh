#!/bin/bash

# Generate Rust protobuf code (this happens automatically during cargo build)
echo "Rust protobuf code will be generated during 'cargo build'"

# Generate Python protobuf code
echo "Generating Python protobuf code..."
protoc --python_out=../generated_protobuffs ../protobuffers/messages.proto

echo "Protobuf generation complete!" 