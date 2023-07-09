use std::collections::{HashMap, HashSet};
use std::str::FromStr;

use anyhow::{anyhow, Error, Result};
use itertools::izip;
use ndarray::{Array2, ArrayD, Axis};

use super::inference::InferenceTree;
use crate::{ArcArrayD, Attributes};

#[derive(Debug)]
pub struct TreeEnsemble {
    /// Vector of trees per class. The outer vector is the class bin.
    trees_per_class: Vec<Vec<InferenceTree>>,
}

#[derive(Debug)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

#[derive(Debug)]
pub enum Node {
    Split(SplitNode),
    Leaf(f64),
}

#[derive(Debug)]
pub struct SplitNode {
    pub threshold: f64,
    pub feature_idx: usize,
    pub nan_follows_true: bool,
    pub comparison: fn(f64, f64) -> bool,
    pub true_false_node_idx: (usize, usize),
}

#[derive(Debug)]
enum NodeType {
    BranchLEq,
    BranchLT,
    BranchGTe,
    BranchGT,
    BranchEq,
    BranchNEq,
    Leaf,
}

#[derive(PartialEq, Debug)]
enum PostTransform {
    None,
    Softmax,
    Logistic,
    SoftmaxZero,
    Probit,
}

#[derive(PartialEq, Debug)]
enum AggregateFunction {
    Average,
    Sum,
    Min,
    Max,
}

impl FromStr for NodeType {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "BRANCH_LEQ" => NodeType::BranchLEq,
            "BRANCH_LT" => NodeType::BranchLT,
            "BRANCH_GTE" => NodeType::BranchGTe,
            "BRANCH_GT" => NodeType::BranchGT,
            "BRANCH_EQ" => NodeType::BranchEq,
            "BRANCH_NEQ" => NodeType::BranchNEq,
            "LEAF" => NodeType::Leaf,
            s => Err(anyhow!("Illegal node type: `{}`", s))?,
        })
    }
}

impl FromStr for PostTransform {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s.to_lowercase().as_str() {
            "none" => Self::None,
            "softmax" => Self::Softmax,
            "logistic" => Self::Logistic,
            "softmax_zero" => Self::SoftmaxZero,
            "probit" => Self::Probit,
            _ => Err(anyhow!("Illegal post transform `{}`.", s))?,
        })
    }
}

impl FromStr for AggregateFunction {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "AVERAGE" => Self::Average,
            "SUM" => Self::Sum,
            "MIN" => Self::Min,
            "MAX" => Self::Max,
            s => Err(anyhow!("Illegal aggregate function `{}`.", s))?,
        })
    }
}

#[derive(Debug)]
struct TreeEnsembleAttributes {
    // n_targets: i64,
    aggregate_function: AggregateFunction,
    base_values_as_tensor: Vec<f64>,
    nodes_falsenodeids: Vec<i64>,
    nodes_featureids: Vec<i64>,
    nodes_missing_value_tracks_true: Vec<i64>,
    nodes_modes: Vec<String>,
    nodes_nodeids: Vec<i64>,
    nodes_treeids: Vec<i64>,
    nodes_truenodeids: Vec<i64>,
    nodes_values_as_tensor: Vec<f64>,
    post_transform: PostTransform,
    target_ids: Vec<i64>,
    target_nodeids: Vec<i64>,
    target_treeids: Vec<i64>,
    target_weights_as_tensor: Vec<f64>,
}

impl TreeEnsembleAttributes {
    pub fn from_attributes(attrs: &Attributes) -> Result<Self> {
        let out = Self {
            aggregate_function: attrs
                .get_with_default("aggregate_function", "SUM".to_owned())?
                .parse()?,
            base_values_as_tensor: attrs.get_with_default("base_values_as_tensor", vec![])?,
            nodes_falsenodeids: attrs.get("nodes_falsenodeids")?.try_into()?,
            nodes_featureids: attrs.get("nodes_featureids")?.try_into()?,
            nodes_missing_value_tracks_true: attrs
                .get("nodes_missing_value_tracks_true")?
                .try_into()?,
            nodes_modes: attrs.get("nodes_modes")?.try_into()?,
            nodes_nodeids: attrs.get("nodes_nodeids")?.try_into()?,
            nodes_treeids: attrs.get("nodes_treeids")?.try_into()?,
            nodes_truenodeids: attrs.get("nodes_truenodeids")?.try_into()?,
            // TODO: Actually not mandatory!
            nodes_values_as_tensor: attrs.get("nodes_values_as_tensor")?.try_into()?,
            post_transform: attrs
                .get_with_default("post_transform", "None".to_owned())?
                .parse()?,
            target_ids: attrs.get("target_ids")?.try_into()?,
            target_nodeids: attrs.get("target_nodeids")?.try_into()?,
            target_treeids: attrs.get("target_treeids")?.try_into()?,
            // TODO: actually not mandatory!
            target_weights_as_tensor: attrs.get("target_weights_as_tensor")?.try_into()?,
        };
        if out.post_transform != PostTransform::None {
            unimplemented!("Post transform {:?} is not implemented", out.post_transform);
        }
        if out.aggregate_function != AggregateFunction::Sum {
            unimplemented!(
                "Aggregate function {:?} is not implemented",
                out.aggregate_function
            );
        }
        if out.base_values_as_tensor.iter().to_owned().sum::<f64>() != 0.0 {
            unimplemented!("Base values are not implemented")
        }
        Ok(out)
    }
}

impl TreeEnsemble {
    pub fn from_attributes(attrs: &Attributes) -> Result<Self> {
        let attrs = TreeEnsembleAttributes::from_attributes(attrs)?;
        // The leaf weights are can be looked up by nodeid & tree id
        let target_lookup: HashMap<_, _> = attrs
            .target_nodeids
            .iter()
            .zip(attrs.target_treeids.iter().map(|el| *el as usize))
            .zip(
                attrs
                    .target_weights_as_tensor
                    .iter()
                    .copied()
                    .zip(attrs.target_ids.iter().copied()),
            )
            .collect();

        let mut trees = vec![];
        let mut target_id_of_tree = vec![
            HashSet::new();
            attrs
                .target_treeids
                .iter()
                .map(|el| *el as usize)
                .max()
                .unwrap()
                + 1
        ];
        let data = izip!(
            &attrs.nodes_nodeids,
            &attrs.nodes_treeids,
            &attrs.nodes_truenodeids,
            &attrs.nodes_falsenodeids,
            attrs
                .nodes_modes
                .iter()
                .map(|s| s.parse::<NodeType>().unwrap()),
            &attrs.nodes_values_as_tensor,
            attrs
                .nodes_missing_value_tracks_true
                .iter()
                .map(|el| *el != 0),
            &attrs.nodes_featureids
        );
        for (
            node_id,
            tree_id,
            true_node,
            false_node,
            mode,
            threshold,
            nan_follows_true,
            feature_idx,
        ) in data
        {
            let tree_id = *tree_id as usize;
            if trees.len() < tree_id + 1 {
                trees.push(Tree { nodes: vec![] });
            }
            let tree = &mut trees[tree_id];
            let node = match mode {
                NodeType::Leaf => {
                    let (value, target_id) = target_lookup.get(&(node_id, tree_id)).unwrap();
                    target_id_of_tree[tree_id].insert(target_id);
                    Node::Leaf(*value)
                }
                NodeType::BranchLEq => Node::Split(SplitNode {
                    threshold: *threshold,
                    feature_idx: *feature_idx as usize,
                    comparison: comparisons::leq,
                    nan_follows_true,
                    true_false_node_idx: (*true_node as usize, *false_node as usize),
                }),
                NodeType::BranchEq => Node::Split(SplitNode {
                    threshold: *threshold,
                    feature_idx: *feature_idx as usize,
                    comparison: comparisons::eq,
                    nan_follows_true,
                    true_false_node_idx: (*true_node as usize, *false_node as usize),
                }),
                ty => {
                    unimplemented!("Node type {:?} is unimplemented.", ty)
                }
            };
            tree.nodes.push(node);
        }

        // Structure the trees in order of their class (aka target)
        for observed_ids_per_tree in target_id_of_tree.iter() {
            if observed_ids_per_tree.len() > 1 {
                panic!("All leafs of a tree must have the same target id.");
            }
        }

        let target_id_of_trees: Vec<_> = target_id_of_tree
            .into_iter()
            .map(|set| *set.into_iter().next().unwrap() as usize)
            .collect();
        let n_classes = *target_id_of_trees.iter().max().unwrap() + 1;
        let mut trees_per_class = vec![vec![]; n_classes];
        for (i, onnx_tree) in trees.iter().enumerate() {
            let i_class = target_id_of_trees[i];
            trees_per_class[i_class].push(InferenceTree::from_onnx_tree(onnx_tree, 0));
        }
        Ok(Self { trees_per_class })
    }

    pub fn predict(&self, input: &ArcArrayD<f64>) -> ArrayD<f32> {
        let n_classes = self.n_classes();
        let mut output = Array2::<f64>::zeros((input.shape()[0], n_classes));

        let input = input.clone().into_dimensionality::<ndarray::Ix2>().unwrap();

        // Assumption: It is more expensive to fetch the tree than the data
        // FIXME: It turns out to be slower to have the trees in
        // separate vectors rather than one vector. Its probably still
        // a good idea to sort by class, though!
        for (mut out, trees) in output
            .axis_iter_mut(Axis(1))
            .zip(self.trees_per_class.iter())
        {
            for tree in trees {
                // The compiler seems smart enough to optimize out
                // allocating in each loop.
                out += &input.map_axis(Axis(1), |row| tree.predict(&row));
            }
        }
        output.mapv(|el| el as f32).into_dyn()
    }

    fn n_classes(&self) -> usize {
        self.trees_per_class.len()
    }
}

mod comparisons {
    pub fn leq(a: f64, b: f64) -> bool {
        a <= b
    }
    pub fn eq(a: f64, b: f64) -> bool {
        a == b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_tree() -> InferenceTree {
        let nodes = vec![
            Node::Split(SplitNode {
                threshold: 1.0,
                feature_idx: 1,
                nan_follows_true: false,
                true_false_node_idx: (1, 2),
                comparison: comparisons::leq,
            }),
            Node::Leaf(-1.0),
            Node::Leaf(42.0),
        ];
        InferenceTree::from_onnx_tree(&Tree { nodes }, 0)
    }

    #[test]
    fn test_tree() {
        let tree = make_tree();
        assert_eq!(tree.predict(&array![0.0, 0.0].view()), -1.0);
        assert_eq!(tree.predict(&array![0.0, 2.0].view()), 42.0);
    }
    #[test]
    fn test_ensemble() {
        let ensemble = TreeEnsemble {
            trees_per_class: vec![vec![make_tree()]],
        };
        let input = array![[0.0, 0.0], [0.0, 2.0]].into_dyn().into();
        assert_eq!(ensemble.predict(&input), array![[-1.0], [42.]].into_dyn());
    }
}
