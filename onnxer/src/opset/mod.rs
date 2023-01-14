use std::collections::HashMap;

use crate::session::SetupFnSignature;

mod ai_onnx;
mod ai_onnx_ml;

pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut wire_dict: HashMap<_, SetupFnSignature> = HashMap::new();
    wire_dict.extend(ai_onnx::get_opset().into_iter());
    wire_dict.extend(ai_onnx_ml::get_opset().into_iter());
    wire_dict
}
