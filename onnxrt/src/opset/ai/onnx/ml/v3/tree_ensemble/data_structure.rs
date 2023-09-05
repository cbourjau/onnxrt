use std::collections::HashMap;
use std::iter::Sum;
use std::ops::AddAssign;

use anyhow::Result;
use ndarray::{Array2, ArrayD, Axis};
use num_traits::{AsPrimitive, Float};

use super::ir;
use super::parsing::NodeType;
use crate::attribute::Attribute;
use crate::{ArcArrayD, Attributes};

pub struct TreeEnsemble<T> {
    class_trees_nodes: Vec<Vec<Vec<Node<T>>>>,
}

#[derive(Debug)]
struct Node<T> {
    threshold: T,
    feature_idx: usize,
    mode: Comparison,
    true_branch: SplitResult<T>,
    false_branch: SplitResult<T>,
}

#[derive(Debug)]
enum Comparison {
    Eq,
    EqNanTrue,
    LeqNanTrue,
}

#[derive(Debug)]
enum SplitResult<T> {
    Value(T),
    Idx(usize),
}

impl Comparison {
    fn new<T: Copy>(node: &ir::SplitNode<T>) -> Self {
        match (node.comparison, node.nan_follows_true) {
            (NodeType::BranchEq, false) => Self::Eq,
            (NodeType::BranchEq, true) => Self::EqNanTrue,
            (NodeType::BranchLEq, true) => Self::LeqNanTrue,
            nt => todo!("{:?}", nt),
        }
    }

    fn cmp<T>(&self, feature: &T, threshold: &T) -> bool
    where
        T: PartialEq<T> + Float + PartialOrd<T>,
    {
        match self {
            Self::Eq => feature == threshold,
            Self::EqNanTrue => feature.is_nan() || feature == threshold,
            Self::LeqNanTrue => feature.is_nan() || feature <= threshold,
        }
    }
}

impl<T> SplitResult<T> {
    fn new(
        result_idx: ir::NodeIdx,
        tree: &HashMap<ir::NodeIdx, ir::Node<T>>,
        old2new_idx: &HashMap<ir::NodeIdx, usize>,
    ) -> Self
    where
        T: Copy,
    {
        match tree.get(&result_idx).unwrap() {
            ir::Node::Split(_) => {
                let new_idx = old2new_idx[&result_idx];
                Self::Idx(new_idx)
            }
            ir::Node::Leaf(val) => Self::Value(*val),
        }
    }
}

impl<T> Node<T> {
    // Create a new split node. This still uses the onnx-indices (cast to usize)
    fn new(
        node: &ir::Node<T>,
        tree: &HashMap<ir::NodeIdx, ir::Node<T>>,
        old2new_idx: &HashMap<ir::NodeIdx, usize>,
    ) -> Self
    where
        T: Copy,
    {
        match node {
            ir::Node::Split(split) => Self {
                mode: Comparison::new(split),
                threshold: split.threshold,
                feature_idx: split.feature_idx,
                true_branch: SplitResult::new(split.true_idx, tree, old2new_idx),
                false_branch: SplitResult::new(split.false_idx, tree, old2new_idx),
            },
            _ => panic!(),
        }
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
        let ir_ensemble = ir::TreeEnsemble::from_attributes(attrs)?;

        let mut nodes: Vec<(ir::ClassIdx, _)> = ir_ensemble
            .0
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
                    *out += predict(tree.as_slice(), row.as_slice().unwrap());
                }
            }
        }
        output.mapv(|el| el.as_()).into_dyn()
    }
}

fn dfs<T>(tree: HashMap<i64, ir::Node<T>>) -> Vec<Node<T>>
where
    T: Copy + std::fmt::Debug,
{
    let mut out = Vec::with_capacity(tree.len());
    let mut old2new_idx = HashMap::with_capacity(tree.len());

    let mut stack = vec![0];

    while let Some(old_idx) = stack.pop() {
        let node = tree.get(&old_idx).unwrap();

        if let ir::Node::Split(split) = node {
            out.push(*node);
            old2new_idx.insert(old_idx, out.len() - 1);

            for idx in &[split.true_idx, split.false_idx] {
                if !old2new_idx.contains_key(idx) {
                    stack.push(*idx)
                }
            }
        }
    }

    out.iter()
        .map(|n| Node::new(n, &tree, &old2new_idx))
        .collect()
}

fn predict<T>(tree: &[Node<T>], features: &[T]) -> T
where
    T: Float + std::fmt::Debug,
{
    let mut node = &tree[0];
    loop {
        let feature_idx = node.feature_idx;
        let feature = features[feature_idx];
        let res = if node.mode.cmp(&feature, &node.threshold) {
            &node.true_branch
        } else {
            &node.false_branch
        };
        match res {
            SplitResult::Idx(idx) => {
                node = &tree[*idx];
            }
            SplitResult::Value(val) => {
                return *val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_tree() -> Vec<Node<f64>> {
        vec![Node {
            threshold: 1.0,
            feature_idx: 1,
            mode: Comparison::LeqNanTrue,
            true_branch: SplitResult::Value(-1.0),
            false_branch: SplitResult::Value(42.0),
        }]
    }

    #[test]
    fn test_tree() {
        let tree = make_tree();
        assert_eq!(predict(tree.as_slice(), &[0.0, 0.0]), -1.0);
        assert_eq!(predict(tree.as_slice(), &[0.0, 2.0]), 42.0);
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
