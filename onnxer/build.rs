use protobuf_codegen::Codegen;

fn main() {
    pyo3_build_config::add_extension_module_link_args();
    Codegen::new()
        .pure()
        .cargo_out_dir("generated_with_pure")
        .input("src/protos/onnx.proto3")
        .include("src/protos")
        .run_from_script();
}
