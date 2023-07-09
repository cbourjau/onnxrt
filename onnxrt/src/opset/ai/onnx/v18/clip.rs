use std::sync::mpsc::Receiver;

use crate::opset::utils::*;
use crate::Dispatchable;

pub fn setup_clip(ctx: OpContext) -> Result<ExecutionStep> {
    let mut inputs = ctx.input_channels_with_optional().into_iter();
    let input = inputs
        .next()
        .flatten()
        .ok_or_else(|| anyhow!("Mandatory input 'input' is missing"))?;
    let min = inputs.next().flatten();
    let max = inputs.next().flatten();

    use TaggedReceiver::*;
    match (input, min, max) {
        // F32
        (F32(input), Some(F32(min)), Some(F32(max))) => match_arm(input, Some(min), Some(max)),
        (F32(input), None, Some(F32(max))) => match_arm(input, None, Some(max)),
        (F32(input), Some(F32(min)), None) => match_arm(input, Some(min), None),
        (F32(input), None, None) => match_arm(input, None, None),

        // F64
        (F64(input), Some(F64(min)), Some(F64(max))) => match_arm(input, Some(min), Some(max)),
        (F64(input), None, Some(F64(max))) => match_arm(input, None, Some(max)),
        (F64(input), Some(F64(min)), None) => match_arm(input, Some(min), None),
        (F64(input), None, None) => match_arm(input, None, None),

        // I32
        (I32(input), Some(I32(min)), Some(I32(max))) => match_arm(input, Some(min), Some(max)),
        (I32(input), None, Some(I32(max))) => match_arm(input, None, Some(max)),
        (I32(input), Some(I32(min)), None) => match_arm(input, Some(min), None),
        (I32(input), None, None) => match_arm(input, None, None),

        // I64
        (I64(input), Some(I64(min)), Some(I64(max))) => match_arm(input, Some(min), Some(max)),
        (I64(input), None, Some(I64(max))) => match_arm(input, None, Some(max)),
        (I64(input), Some(I64(min)), None) => match_arm(input, Some(min), None),
        (I64(input), None, None) => match_arm(input, None, None),
        inputs => bail!("Unexpected input for operator 'Clip': {:?}", inputs),
    }
}

fn match_arm<T>(
    input: Receiver<ArcArrayD<T>>,
    min: Option<Receiver<ArcArrayD<T>>>,
    max: Option<Receiver<ArcArrayD<T>>>,
) -> Result<ExecutionStep>
where
    ArcArrayD<T>: Dispatchable,
    T: Sync + Send + 'static + Copy + PartialOrd,
{
    setup((input, min, max), move |(input, min, max)| {
        clip(input, min, max)
    })
}

fn clip<T>(
    input: ArcArrayD<T>,
    min: Option<ArcArrayD<T>>,
    max: Option<ArcArrayD<T>>,
) -> Result<ArcArrayD<T>>
where
    T: Copy + PartialOrd,
{
    let output = if let Some(min) = min {
        let min = min.to_owned().into_shape([])?.into_scalar();
        input.mapv(|el| if el < min { min } else { el }).into()
    } else {
        input
    };
    let output = if let Some(max) = max {
        let max = max.to_owned().into_shape([])?.into_scalar();
        output.mapv(|el| if el > max { max } else { el }).into()
    } else {
        output
    };
    Ok(output)
}
