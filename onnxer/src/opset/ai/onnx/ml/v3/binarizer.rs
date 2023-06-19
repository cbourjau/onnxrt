use anyhow::{bail, Result};
use itertools::Itertools;
use num::cast::AsPrimitive;

use crate::opset::utils::*;

fn binarize<T>(a: ArcArrayD<T>, threshold: T) -> Result<ArcArrayD<T>>
where
    T: PartialOrd + Clone + Copy + 'static,
    u8: AsPrimitive<T>,
{
    Ok(a.mapv(|el| ((el > threshold) as u8).as_()).into())
}

pub fn setup_binarizer(ctx: OpContext) -> Result<ExecutionStep> {
    let (a_ch,) = ctx.input_channels().into_iter().collect_tuple().unwrap();
    let threshold: f32 = ctx.attrs.get("threshold")?.try_into()?;

    match a_ch {
        TaggedReceiver::F32(rec) => setup((rec,), move |(a,)| binarize(a, threshold)),
        TaggedReceiver::F64(rec) => setup((rec,), move |(a,)| binarize(a, threshold as f64)),
        // `floor` first for int cast in order to ensure that we don't
        // round up (for negative numbers).
        TaggedReceiver::I32(rec) => {
            setup((rec,), move |(a,)| binarize(a, threshold.floor() as i32))
        }
        TaggedReceiver::I64(rec) => {
            setup((rec,), move |(a,)| binarize(a, threshold.floor() as i64))
        }
        _ => bail!("Binarizer is only defined for f32, f64, i32 and i64 inputs"),
    }
}
