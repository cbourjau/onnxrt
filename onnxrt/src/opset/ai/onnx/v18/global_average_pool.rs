use num::{Float, Zero};
use num_traits::AsPrimitive;

use crate::opset::utils::*;

fn global_average_pool<T>(a: ArcArrayD<T>) -> Result<ArcArrayD<T>>
where
    T: Copy + Zero + 'static + Float,
    usize: AsPrimitive<T>,
{
    let out_shape: Vec<_> = a
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &len)| if i < 2 { len } else { 1 })
        .collect();

    let mut out = ArrayD::zeros(out_shape);

    for (a, mut out) in a.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
        for (a, mut out) in a.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            let sum = a.sum();
            let avg = sum / a.len().as_();
            out.fill(avg);
        }
    }

    Ok(out.into_dyn().into())
}

pub fn setup_gap(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;

    match inputs {
        F32(a) => setup(a, global_average_pool),
        F64(a) => setup(a, global_average_pool),
    }
}
