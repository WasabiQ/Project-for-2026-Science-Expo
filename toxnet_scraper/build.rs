fn main() {
    // This automates the binary serialization logic
    prost_build::compile_protos(&["src/skynet.proto"], &["src/"]).unwrap();
}