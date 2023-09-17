use crate::opset::utils::*;

pub fn setup_reshape(ctx: OpContext) -> Result<ExecutionStep> {
    let allow_zero = ctx.attrs.get_with_default("allowzero", 0)? == 1;

    let (data, shape): (_, Receiver<ArcArrayD<i64>>) = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedReceiver,
        data,
        data,
        setup((data, shape), move |(data, shape)| {
            reshape(
                data.into_owned(),
                shape.as_standard_layout().as_slice().unwrap(),
                allow_zero,
            )
        })
    )
}

pub(crate) fn reshape<D>(data: ArrayD<D>, shape: &[i64], allow_zero: bool) -> Result<ArrayD<D>>
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

    Ok(data.into_shape(fixed)?)
}
