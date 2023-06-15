use super::*;

pub fn setup_reshape(ctx: OpContext) -> Result<ExecutionStep> {
    let [data, shape]: [_; 2] = ctx.input_channels().try_into().unwrap();
    let allow_zero = ctx.attrs.get_with_default("allowzero", 0)? == 1;

    let shape = {
        if let TaggedReceiver::I64(ch) = shape {
            ch
        } else {
            bail!(
                "Input 'shape' must be of type int64. Found: {:?}",
                shape.dtype()
            )
        }
    };

    Ok(match data {
        TaggedReceiver::Bool(data) => match_arm(data, shape, allow_zero)?,
        TaggedReceiver::F32(data) => match_arm(data, shape, allow_zero)?,
        TaggedReceiver::F64(data) => match_arm(data, shape, allow_zero)?,
        TaggedReceiver::I32(data) => match_arm(data, shape, allow_zero)?,
        TaggedReceiver::I64(data) => match_arm(data, shape, allow_zero)?,
        TaggedReceiver::String(data) => match_arm(data, shape, allow_zero)?,
    })
}

fn match_arm<Data>(
    data: Receiver<ArcArrayD<Data>>,
    shape: Receiver<ArcArrayD<i64>>,
    allow_zero: bool,
) -> Result<ExecutionStep>
where
    ArcArrayD<Data>: Dispatchable,
    Data: 'static + Clone + std::fmt::Debug + Sync + Send + std::fmt::Debug,
{
    let (sdr, rec) = channel();
    let fun = Box::new(move || {
        let data = data.recv().map_err(|e| anyhow!("{}", e))?;
        let shape = shape.recv().map_err(|e| anyhow!("{}", e))?;
        let out = reshape(data, shape, allow_zero)?;
        sdr.send(out)?;
        Ok(())
    });
    Ok(ExecutionStep {
        fun,
        outputs: vec![rec],
    })
}

fn reshape<D>(data: ArcArrayD<D>, shape: ArcArrayD<i64>, allow_zero: bool) -> Result<ArcArrayD<D>>
where
    D: Clone + std::fmt::Debug,
{
    let n = data.len() as i64;

    // replace 0s (may still contain zeros depending on `allow_zero`)
    let fixed_zeros: Vec<i64> = data
        .shape()
        .iter()
        .zip(shape.iter())
        .map(|(old, new)| {
            if *new == 0 && !allow_zero {
                *old as i64
            } else {
                *new
            }
        })
        .collect();

    let n_without_minus_one: i64 = fixed_zeros
        .iter()
        .chain(shape.iter().skip(fixed_zeros.len()))
        .filter_map(|el| {
            if *el == -1 || *el == 0 {
                None
            } else {
                Some(*el)
            }
        })
        .product();
    let replacement_minus_one = (n / n_without_minus_one) as usize;
    let fixed: Vec<_> = fixed_zeros
        .iter()
        .chain(shape.iter().skip(fixed_zeros.len()))
        .map(|el| {
            if *el == -1 {
                replacement_minus_one
            } else {
                *el as usize
            }
        })
        .collect();

    if fixed.iter().map(|el| *el as i64).product::<i64>() != n {
        bail!("Reshaped tensor must have same number of elements as input tensor")
    }

    Ok(data.reshape(fixed).into())
}
