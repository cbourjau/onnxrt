mod binarizer;
mod imputer;
mod label_encoder_and_category_mapper;
mod other_operators;
mod tree_ensemble;

use std::collections::HashMap;

use crate::session::SetupFnSignature;
use label_encoder_and_category_mapper::{setup_category_mapper, setup_label_encoder};
use tree_ensemble::setup_tree_ensemble_regressor;

pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut setup_dict: HashMap<_, SetupFnSignature> = HashMap::new();
    setup_dict.insert(
        "ArrayFeatureExtractor",
        Box::new(other_operators::setup_array_feature_extractor),
    );
    setup_dict.insert("Binarizer", Box::new(binarizer::setup_binarizer));
    setup_dict.insert("CategoryMapper", Box::new(setup_category_mapper));
    setup_dict.insert("Imputer", Box::new(imputer::setup_imputer));
    setup_dict.insert("LabelEncoder", Box::new(setup_label_encoder));
    setup_dict.insert(
        "TreeEnsembleRegressor",
        Box::new(setup_tree_ensemble_regressor),
    );
    setup_dict
}
