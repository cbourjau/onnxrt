use num_traits::Float;

/// Data types optimized for inference
use super::data_structure::{Node, Tree as OnnxTree};

#[derive(Debug, Clone)]
pub struct Split<T> {
    threshold: T,
    feature_idx: usize,
    nan_follows_true: bool,
    comparison: fn(&T, &T) -> bool,
    true_node: Box<InferenceTree<T>>,
    false_node: Box<InferenceTree<T>>,
}

#[derive(Debug, Clone)]
pub enum InferenceTree<T> {
    Split(Split<T>),
    Leaf(T),
}

impl<T> InferenceTree<T> {
    pub fn from_onnx_tree(onnx_tree: &OnnxTree<T>, start_node: usize) -> Self
    where
        T: Float,
    {
        match &onnx_tree.nodes[start_node] {
            Node::Split(split_node) => {
                // let left = onnx_tree.nodes[split_node.true_false_node_idx.0];
                // let right = onnx_tree.nodes[split_node.true_false_node_idx.1];
                let left = split_node.true_false_node_idx.0;
                let right = split_node.true_false_node_idx.1;
                Self::Split(Split {
                    threshold: split_node.threshold,
                    feature_idx: split_node.feature_idx,
                    nan_follows_true: split_node.nan_follows_true,
                    comparison: split_node.comparison,
                    true_node: Box::new(Self::from_onnx_tree(onnx_tree, left)),
                    false_node: Box::new(Self::from_onnx_tree(onnx_tree, right)),
                })
            }
            Node::Leaf(val) => Self::Leaf(*val),
        }
    }

    pub fn predict(&self, features: &ndarray::ArrayView1<T>) -> T
    where
        T: Float,
    {
        match self {
            Self::Split(split) => {
                let mut current_node = split;
                loop {
                    let feature_idx = split.feature_idx;
                    let feature = features[feature_idx];
                    let next_node = if feature.is_nan() {
                        if current_node.nan_follows_true {
                            &current_node.true_node
                        } else {
                            &current_node.false_node
                        }
                    } else if (current_node.comparison)(&feature, &current_node.threshold) {
                        &current_node.true_node
                    } else {
                        &current_node.false_node
                    };
                    match next_node.as_ref() {
                        Self::Split(split) => {
                            current_node = split;
                        }
                        Self::Leaf(val) => break *val,
                    }
                }
            }
            Self::Leaf(val) => *val,
        }
    }
}
