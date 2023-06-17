use anyhow::{bail, Result};
use itertools::Itertools;
use ndarray::Axis;

use crate::session::{ExecutionStep, OpContext};
use crate::{ArcArrayD, TaggedReceiver};

use crate::setup_utils::setup;

pub fn setup_array_feature_extractor(ctx: OpContext) -> Result<ExecutionStep> {
    let (data_ch, indices_ch) = ctx.input_channels().into_iter().collect_tuple().unwrap();
    let indices_ch = match indices_ch {
        TaggedReceiver::I64(ch) => ch,
        _ => bail!("Input `Y` of ArrayFeatureExtractor must be of type `tensor(int64)`"),
    };

    match data_ch {
        TaggedReceiver::I32(ch) => setup((ch, indices_ch), array_feature_extractor),
        TaggedReceiver::I64(ch) => setup((ch, indices_ch), array_feature_extractor),
        TaggedReceiver::U32(ch) => setup((ch, indices_ch), array_feature_extractor),
        TaggedReceiver::U64(ch) => setup((ch, indices_ch), array_feature_extractor),
        TaggedReceiver::F32(ch) => setup((ch, indices_ch), array_feature_extractor),
        TaggedReceiver::F64(ch) => setup((ch, indices_ch), array_feature_extractor),
        TaggedReceiver::String(ch) => setup((ch, indices_ch), array_feature_extractor),
        TaggedReceiver::Bool(ch) => setup((ch, indices_ch), array_feature_extractor),
    }
}

fn array_feature_extractor<D>(
    (data, indices): (ArcArrayD<D>, ArcArrayD<i64>),
) -> Result<ArcArrayD<D>>
where
    D: Clone,
{
    let last_axis = Axis(data.ndim() - 1);
    let outs: Vec<_> = indices
        .into_iter()
        .map(|idx| {
            data.index_axis(last_axis, idx as usize)
                .insert_axis(last_axis)
        })
        .collect();
    Ok(ndarray::concatenate(Axis(data.ndim() - 1), outs.as_slice())?.into())
}
