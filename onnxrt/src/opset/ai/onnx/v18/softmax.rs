use num::Float;

use crate::opset::utils::*;

fn softmax<T>(mut input: ArrayD<T>, axis: i64) -> Result<ArrayD<T>>
where
    T: Float,
{
    let axis = axis.for_len(input.ndim());
    let exp = {
        input.mapv_inplace(|el| el.exp());
        input
    };
    let denom = {
        let mut sum = exp.sum_axis(Axis(axis));
        sum.insert_axis_inplace(Axis(axis));
        sum
    };
    Ok(exp / &denom)
}

fn log_softmax<T>(input: ArrayD<T>, axis: i64) -> Result<ArrayD<T>>
where
    T: Float,
{
    let mut sm = softmax(input, axis)?;
    sm.mapv_inplace(|el| el.ln());
    Ok(sm)
}

struct SoftMax {
    axis: i64,
}

impl SoftMax {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        Ok(Self {
            axis: attrs.get_with_default("axis", -1)?,
        })
    }
}

pub fn setup_softmax(ctx: OpContext) -> Result<ExecutionStep> {
    let state = SoftMax::from_attrs(&ctx.attrs)?;

    let input = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        FloatingPoint,
        input,
        input,
        setup((input,), move |(input,)| {
            Ok::<ArcArrayD<_>, _>(softmax(input.to_owned(), state.axis)?.into())
        })
    )
}

pub fn setup_log_softmax(ctx: OpContext) -> Result<ExecutionStep> {
    let state = SoftMax::from_attrs(&ctx.attrs)?;

    let input = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        FloatingPoint,
        input,
        input,
        setup((input,), move |(input,)| {
            Ok::<ArcArrayD<_>, _>(log_softmax(input.to_owned(), state.axis)?.into())
        })
    )
}
