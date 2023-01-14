mod binarizer;
mod imputer;
mod label_encoder_and_category_mapper;
mod other_operators;
mod tree_ensemble;

use std::collections::HashMap;

use crate::session::SetupFnSignature;
use label_encoder_and_category_mapper::{wire_category_mapper, wire_label_encoder};
use tree_ensemble::wire_tree_ensemble_regressor;

pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut wire_dict: HashMap<_, SetupFnSignature> = HashMap::new();
    wire_dict.insert(
        "ArrayFeatureExtractor",
        Box::new(other_operators::wire_array_feature_extractor),
    );
    wire_dict.insert("Binarizer", Box::new(binarizer::wire_binarizer));
    wire_dict.insert("CategoryMapper", Box::new(wire_category_mapper));
    wire_dict.insert("Imputer", Box::new(imputer::wire_imputer));
    wire_dict.insert("LabelEncoder", Box::new(wire_label_encoder));
    wire_dict.insert(
        "TreeEnsembleRegressor",
        Box::new(wire_tree_ensemble_regressor),
    );
    wire_dict
}
