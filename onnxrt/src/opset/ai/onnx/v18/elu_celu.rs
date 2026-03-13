use num::{Float, One, Zero};
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

fn celu<T>(mut x: ArrayD<T>, alpha: f32) -> ArrayD<T>
where
    T: Float + One + 'static + PartialOrd + Zero,
    f32: AsPrimitive<T>,
{
    // max(0,x) + min(0,alpha*(exp(x/alpha)-1))
    let alpha = alpha.as_();
    x.mapv_inplace(|el| {
        el.max(T::zero()) + (alpha * ((el / alpha).exp() - T::one())).min(T::zero())
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
        setup(x, move |x| { Ok(elu(x.to_owned(), alpha)) })
    )
}

pub fn setup_celu(ctx: OpContext) -> Result<ExecutionStep> {
    let alpha = ctx.attrs.get_with_default("alpha", 1.0)?;
    let x = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        FloatingPoint,
        x,
        x,
        setup(x, move |x| { Ok(celu(x.to_owned(), alpha)) })
    )
}
