fn main() {
    // Tell Cargo to rerun this build script if the proto file changes
    println!("cargo:rerun-if-changed=../protobuffers/messages.proto");
    
    // Generate Rust code from the protobuf definition
    // Since there's no package declaration, the output will be in OUT_DIR/messages.rs
    prost_build::compile_protos(&["../protobuffers/messages.proto"], &["../protobuffers"])
        .unwrap();
} 