use std::collections::HashMap;
use std::iter::Sum;
use std::ops::Rem;
use std::unimplemented;

use anyhow::Result;
use itertools::izip;
use num_traits::Float;

use super::parsing::{NodeType, TreeEnsembleAttributes};
use crate::attribute::Attribute;
use crate::Attributes;

#[derive(Debug)]
pub struct TreeEnsemble<T>(pub HashMap<ClassIdx, HashMap<TreeIdx, HashMap<NodeIdx, Node<T>>>>);

pub type ClassIdx = i64;
pub type NodeIdx = i64;
type TreeIdx = i64;

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
    pub(crate) comparison: NodeType,
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
            &NodeType::Leaf => unimplemented!("Expected split node, found leaf node"),
            mode => Node::Split(SplitNode {
                threshold,
                feature_idx: feature_idx as usize,
                comparison: *mode,
                nan_follows_true,
                true_idx: true_node,
                false_idx: false_node,
            }),
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

        let mut nodes = HashMap::new();
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

        Ok(Self(nodes))
    }
}
