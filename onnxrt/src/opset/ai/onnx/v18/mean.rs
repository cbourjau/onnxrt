use std::ops::{Add, Div};

use num_traits::AsPrimitive;

use crate::opset::utils::*;

fn mean<T>(arrays: Vec<ArcArrayD<T>>) -> Result<ArrayD<T>>
where
    T: Add<Output = T> + Div<Output = T> + Clone + Copy + 'static + ndarray::ScalarOperand,
    usize: AsPrimitive<T>,
{
    let n = arrays.len();
    let mut iter = arrays.into_iter();
    let mut out = iter
        .next()
        .ok_or_else(|| anyhow!("Mean: At least one input is required"))?
        .to_owned();
    for arr in iter {
        out = zip_map_with_co_broadcast(out.view(), arr.view(), |&a, &b| a + b)?;
    }
    Ok(out / n.as_())
}

pub fn setup_mean(ctx: OpContext) -> Result<ExecutionStep> {
    let data = ctx.subscribe_to_inputs().narrow()?;
    setup_all!(
        HomogeneousNumeric,
        data,
        data,
        setup(data, move |data| mean(data))
    )
}
