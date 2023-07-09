use anyhow::Result;
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
    let threshold: f32 = ctx.attrs.get("threshold")?.try_into()?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;
    match inputs {
        F32([rec]) => setup((rec,), move |(a,)| binarize(a, threshold)),
        F64([rec]) => setup((rec,), move |(a,)| binarize(a, threshold as f64)),
        // `floor` first for int cast in order to ensure that we don't
        // round up (for negative numbers).
        I32([rec]) => setup((rec,), move |(a,)| binarize(a, threshold.floor() as i32)),
        I64([rec]) => setup((rec,), move |(a,)| binarize(a, threshold.floor() as i64)),
        // The standard only supports the above four types
        _ => bail!("Binarizer: Unsupported input types {:?}", inputs),
    }
}
