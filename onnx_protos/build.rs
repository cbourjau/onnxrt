use protobuf_codegen::Codegen;

fn main() {
    Codegen::new()
        .pure()
        .cargo_out_dir("generated_with_pure")
        .input("../onnx/onnx/onnx.proto3")
        .include("../onnx/onnx/")
        .run_from_script();
}
