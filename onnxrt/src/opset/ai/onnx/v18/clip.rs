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

    macro_rules! match_vars {
        ($($var:tt),*) => {{
            use TaggedReceiver::*;
            match (input, min, max) {
                $(
                    ($var(input), Some($var(min)), Some($var(max))) => {
                        match_arm(input, Some(min), Some(max))
                    }
                    ($var(input), None, Some($var(max))) => match_arm(input, None, Some(max)),
                    ($var(input), Some($var(min)), None) => match_arm(input, Some(min), None),
                    ($var(input), None, None) => match_arm(input, None, None),
                )*
                    inputs => bail!("Clip: Unexpected input {:?}", inputs),
            }
        }};
    }
    match_vars!(F32, F64, I8, I16, I32, I64, U8, U16, U32, U64)
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
