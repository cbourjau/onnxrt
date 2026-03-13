use std::collections::HashMap;

use crate::session::SetupFnSignature;

pub mod ai;
mod utils;

pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut setup_dict: HashMap<_, SetupFnSignature> = HashMap::new();
    setup_dict.extend(ai::onnx::v18::get_opset());
    setup_dict.extend(ai::onnx::ml::v3::get_opset());
    setup_dict
}
