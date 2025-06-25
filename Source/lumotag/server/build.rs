fn main() {
    // Tell Cargo to rerun this build script if the proto file changes
    println!("cargo:rerun-if-changed=../common/messages.proto");
    
    // Generate Rust code from the protobuf definition
    // Since there's no package declaration, the output will be in OUT_DIR/messages.rs
    prost_build::compile_protos(&["../common/messages.proto"], &["../common"])
        .unwrap();
} 