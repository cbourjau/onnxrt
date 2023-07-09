use anyhow::Result;
use ndarray::Axis;

use crate::opset::utils::*;

pub fn setup_array_feature_extractor(ctx: OpContext) -> Result<ExecutionStep> {
    let (data_ch, indices_ch): (_, Receiver<ArcArrayD<i64>>) =
        ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedReceiver,
        data_ch,
        data_ch,
        setup((data_ch, indices_ch), array_feature_extractor)
    )
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
