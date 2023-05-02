use super::*;

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

pub fn match_arm<T>(
    input: Receiver<ArcArrayD<T>>,
    min: Option<Receiver<ArcArrayD<T>>>,
    max: Option<Receiver<ArcArrayD<T>>>,
) -> Result<ExecutionStep>
where
    ArcArrayD<T>: Dispatchable,
    T: Sync + Send + 'static + Copy + PartialOrd,
{
    let (sdr, rec) = channel();
    let fun: Box<dyn Fn() -> Result<()> + Send> = match (min, max) {
        (Some(min), Some(max)) => Box::new(move || {
            let input = input.recv()?;
            let min = min.recv()?;
            let max = max.recv()?;
            sdr.send(clip(input, Some(min), Some(max))?)?;
            Ok(())
        }),
        (None, Some(max)) => Box::new(move || {
            let input = input.recv()?;
            let max = max.recv()?;
            sdr.send(clip(input, None, Some(max))?)?;
            Ok(())
        }),
        (Some(min), None) => Box::new(move || {
            let input = input.recv()?;
            let min = min.recv()?;
            sdr.send(clip(input, Some(min), None)?)?;
            Ok(())
        }),
        (None, None) => Box::new(move || {
            let input = input.recv()?;
            sdr.send(clip(input, None, None)?)?;
            Ok(())
        }),
    };
    Ok(ExecutionStep {
        fun,
        outputs: vec![rec],
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
