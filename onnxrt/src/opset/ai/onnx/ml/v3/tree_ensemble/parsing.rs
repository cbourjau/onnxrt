use std::iter::Sum;
use std::str::FromStr;

use anyhow::{anyhow, Error, Result};
use num_traits::{Float, Zero};

use crate::attribute::Attribute;
use crate::Attributes;

#[derive(Debug)]
pub struct TreeEnsembleAttributes<T> {
    // n_targets: i64,
    pub(crate) aggregate_function: AggregateFunction,
    pub(crate) base_values_as_tensor: Vec<T>,
    pub(crate) nodes_falsenodeids: Vec<i64>,
    pub(crate) nodes_featureids: Vec<i64>,
    pub(crate) nodes_missing_value_tracks_true: Vec<i64>,
    pub(crate) nodes_modes: Vec<NodeType>,
    pub(crate) nodes_nodeids: Vec<i64>,
    pub(crate) nodes_treeids: Vec<i64>,
    pub(crate) nodes_truenodeids: Vec<i64>,
    pub(crate) nodes_values_as_tensor: Vec<T>,
    pub(crate) post_transform: PostTransform,
    pub(crate) target_ids: Vec<i64>,
    pub(crate) target_nodeids: Vec<i64>,
    pub(crate) target_treeids: Vec<i64>,
    pub(crate) target_weights_as_tensor: Vec<T>,
}

#[derive(Debug)]
pub(crate) enum NodeType {
    BranchLEq,
    BranchLT,
    BranchGTe,
    BranchGT,
    BranchEq,
    BranchNEq,
    Leaf,
}

#[derive(PartialEq, Debug)]
pub(crate) enum PostTransform {
    None,
    Softmax,
    Logistic,
    SoftmaxZero,
    Probit,
}

#[derive(PartialEq, Debug)]
pub(crate) enum AggregateFunction {
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

impl<T> TreeEnsembleAttributes<T> {
    pub fn from_attributes<'s>(attrs: &'s Attributes) -> Result<Self>
    where
        Vec<T>: TryFrom<&'s Attribute, Error = anyhow::Error>,
        T: Float + Zero + Sum<T> + 'static,
    {
        let nodes_modes: Vec<String> = attrs.get("nodes_modes")?.try_into()?;
        let nodes_modes = nodes_modes
            .into_iter()
            .map(|el| el.parse())
            .collect::<Result<Vec<_>>>()?;
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
            nodes_modes,
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
        if out.base_values_as_tensor.iter().copied().sum::<T>() != T::zero() {
            unimplemented!("Base values are not implemented")
        }
        Ok(out)
    }
}
