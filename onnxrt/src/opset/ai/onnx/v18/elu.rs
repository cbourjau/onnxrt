use num::{Float, One};
use num_traits::AsPrimitive;

use crate::opset::utils::*;

fn elu<T>(mut x: ArrayD<T>, alpha: f32) -> ArrayD<T>
where
    T: Float + One + 'static + PartialOrd,
    f32: AsPrimitive<T>,
{
    let alpha = alpha.as_();
    x.mapv_inplace(|el| {
        if el < T::zero() {
            alpha * (el.exp() - T::one())
        } else {
            el
        }
    });
    x
}

pub fn setup_elu(ctx: OpContext) -> Result<ExecutionStep> {
    let alpha = ctx.attrs.get_with_default("alpha", 1.0)?;
    let x = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        FloatingPoint,
        x,
        x,
        setup((x,), move |(x,)| {
            let x = x.to_owned();
            Ok(elu(x, alpha))
        })
    )
}
