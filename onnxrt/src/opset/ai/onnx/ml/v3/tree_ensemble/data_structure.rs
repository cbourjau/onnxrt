use std::collections::HashMap;
use std::iter::Sum;
use std::ops::{AddAssign, Rem};
use std::unimplemented;

use anyhow::Result;
use itertools::izip;
use ndarray::{Array2, ArrayD, Axis};
use num_traits::{AsPrimitive, Float};

use super::parsing::{NodeType, TreeEnsembleAttributes};
use crate::attribute::Attribute;
use crate::{ArcArrayD, Attributes};

#[derive(Debug)]
pub struct TreeEnsemble<T> {
    class_trees_nodes: Vec<Vec<Vec<Node<T>>>>,
}

type ClassIdx = i64;
type NodeIdx = i64;
type TreeIdx = i64;

#[derive(Debug)]
pub struct Tree<T> {
    pub nodes: Vec<Node<T>>,
}

#[derive(Debug, Clone, Copy)]
pub enum Node<T> {
    Split(SplitNode<T>),
    Leaf(T),
}

#[derive(Debug, Clone, Copy)]
pub struct SplitNode<T> {
    pub threshold: T,
    pub feature_idx: usize,
    pub nan_follows_true: bool,
    pub comparison: fn(&T, &T) -> bool,
    pub true_idx: NodeIdx,
    pub false_idx: NodeIdx,
}

impl<T> Node<T>
where
    T: Clone + Copy + Float + Sum<T> + 'static + std::fmt::Debug,
{
    fn new_split(
        threshold: T,
        feature_idx: i64,
        nan_follows_true: bool,
        mode: &NodeType,
        true_node: NodeIdx,
        false_node: NodeIdx,
    ) -> Self {
        match mode {
            &NodeType::BranchLEq => Node::Split(SplitNode {
                threshold,
                feature_idx: feature_idx as usize,
                comparison: comparisons::leq,
                nan_follows_true,
                true_idx: true_node,
                false_idx: false_node,
            }),
            &NodeType::BranchEq => Node::Split(SplitNode {
                threshold,
                feature_idx: feature_idx as usize,
                comparison: comparisons::eq,
                nan_follows_true,
                true_idx: true_node,
                false_idx: false_node,
            }),
            ty => {
                dbg!(feature_idx, nan_follows_true, mode, true_node, false_node);

                unimplemented!("Node type {:?} is unimplemented.", ty)
            }
        }
    }
    fn new_leaf(value: T) -> Self {
        Node::Leaf(value)
    }
}

impl<T> TreeEnsemble<T>
where
    T: Clone + Copy + Float + Sum<T> + 'static + std::fmt::Debug,
{
    pub fn from_attributes<'s>(attrs: &'s Attributes) -> Result<Self>
    where
        Vec<T>: TryFrom<&'s Attribute, Error = anyhow::Error>,
    {
        let attrs = TreeEnsembleAttributes::<T>::from_attributes(attrs)?;
        let n_classes = attrs.n_targets;

        let mut nodes: HashMap<ClassIdx, HashMap<TreeIdx, HashMap<NodeIdx, _>>> = HashMap::new();
        for ((node_id, tree_id), (weight, _target_id)) in attrs
            .target_nodeids
            .into_iter()
            .zip(attrs.target_treeids.into_iter())
            .zip(
                attrs
                    .target_weights_as_tensor
                    .into_iter()
                    .zip(attrs.target_ids.into_iter()),
            )
        {
            let node = Node::new_leaf(weight);
            let class_id = tree_id.rem(n_classes);
            nodes
                .entry(class_id)
                .or_insert_with(HashMap::new)
                .entry(tree_id)
                .or_insert_with(HashMap::new)
                .insert(node_id, node);
        }

        let data = izip!(
            &attrs.nodes_nodeids,
            &attrs.nodes_treeids,
            &attrs.nodes_truenodeids,
            &attrs.nodes_falsenodeids,
            &attrs.nodes_modes,
            &attrs.nodes_values_as_tensor,
            attrs.nodes_missing_value_tracks_true,
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
            if let NodeType::Leaf = mode {
                continue;
            }
            let node = Node::new_split(
                *threshold,
                *feature_idx,
                nan_follows_true,
                mode,
                *true_node,
                *false_node,
            );
            let class_id = tree_id.rem(n_classes);
            nodes
                .get_mut(&class_id)
                .unwrap()
                .get_mut(tree_id)
                .unwrap()
                .insert(*node_id, node);
        }

        let mut nodes: Vec<(ClassIdx, _)> = nodes
            .into_iter()
            .map(|(cls_idx, trees)| {
                let mut trees: Vec<_> = trees.into_iter().collect();
                trees.sort_unstable_by_key(|el| el.0);
                (cls_idx, trees.into_iter().map(|el| dfs(el.1)).collect())
            })
            .collect();
        nodes.sort_unstable_by_key(|el| el.0);
        Ok(Self {
            class_trees_nodes: nodes.into_iter().map(|(_k, v)| v).collect(),
        })
    }

    pub fn predict(&self, input: &ArcArrayD<T>) -> ArrayD<f32>
    where
        T: 'static + Float + AddAssign<T> + AsPrimitive<f32>,
    {
        let mut output = Array2::<T>::zeros((input.shape()[0], self.class_trees_nodes.len()));

        let input = input
            .as_standard_layout()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();

        for (mut out, trees_of_class) in output.axis_iter_mut(Axis(1)).zip(&self.class_trees_nodes)
        {
            for tree in trees_of_class {
                for (row, out) in input.axis_iter(Axis(0)).zip(out.iter_mut()) {
                    *out += predict(tree, row.as_slice().unwrap());
                }
            }
        }
        output.mapv(|el| el.as_()).into_dyn()
    }
}

fn dfs<T>(tree: HashMap<i64, Node<T>>) -> Vec<Node<T>>
where
    T: Copy,
{
    let mut out = Vec::with_capacity(tree.len());
    let mut old2new_idx = HashMap::with_capacity(tree.len());

    let mut stack = vec![0];

    loop {
        if let Some(old_idx) = stack.pop() {
            let node = tree.get(&old_idx).unwrap();
            out.push(*node);
            old2new_idx.insert(old_idx, out.len() - 1);

            if let Node::Split(split) = node {
                for idx in &[split.true_idx, split.false_idx] {
                    if !old2new_idx.contains_key(idx) {
                        stack.push(*idx)
                    }
                }
            }
        } else {
            break;
        }
    }

    // Update indices
    for node in out.iter_mut() {
        if let Node::Split(split) = node {
            split.true_idx = *old2new_idx.get(&split.true_idx).unwrap() as i64;
            split.false_idx = *old2new_idx.get(&split.false_idx).unwrap() as i64;
        }
    }
    out
}

mod comparisons {
    use num_traits::Float;

    pub fn leq<T: Float>(a: &T, b: &T) -> bool {
        a <= b
    }
    pub fn eq<T: Float>(a: &T, b: &T) -> bool {
        a == b
    }
}

fn predict<T>(tree: &[Node<T>], features: &[T]) -> T
where
    T: Float + std::fmt::Debug,
{
    // Prediction starts at index 0
    match tree.get(0).unwrap() {
        Node::Split(split) => {
            let mut current_node = split;
            loop {
                let feature_idx = split.feature_idx;
                let feature = features[feature_idx];
                let next_node_idx = if feature.is_nan() {
                    if current_node.nan_follows_true {
                        &current_node.true_idx
                    } else {
                        &current_node.false_idx
                    }
                } else if (current_node.comparison)(&feature, &current_node.threshold) {
                    &current_node.true_idx
                } else {
                    &current_node.false_idx
                };

                match tree.get(*next_node_idx as usize).unwrap() {
                    Node::Split(ref split) => {
                        current_node = split;
                    }
                    Node::Leaf(val) => break *val,
                }
            }
        }
        Node::Leaf(val) => *val,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_tree() -> Vec<Node<f64>> {
        vec![
            Node::Split(SplitNode {
                threshold: 1.0,
                feature_idx: 1,
                nan_follows_true: false,
                true_idx: 1,
                false_idx: 2,
                comparison: comparisons::leq,
            }),
            Node::Leaf(-1.0),
            Node::Leaf(42.0),
        ]
    }

    #[test]
    fn test_tree() {
        let tree = make_tree();
        assert_eq!(predict(&tree, &[0.0, 0.0]), -1.0);
        assert_eq!(predict(&tree, &[0.0, 2.0]), 42.0);
    }

    #[test]
    fn test_ensemble() {
        let ensemble = TreeEnsemble {
            class_trees_nodes: vec![vec![make_tree()]],
        };
        let input = array![[0.0, 0.0], [0.0, 2.0]].into_dyn().into();
        assert_eq!(ensemble.predict(&input), array![[-1.0], [42.]].into_dyn());
    }
}
