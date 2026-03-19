use anyhow::bail;
use num::{Float, One, Zero};
use num_traits::AsPrimitive;

use crate::{opset::utils::*, session::OpContext};

pub fn setup_celu(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let alpha = ctx.attrs.get_with_default("alpha", 1.0)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;
    // Standard only specifies Float
    match inputs {
        (F32([a])) => setup_unary(a, move |a| Ok(a.mapv(|el| celu(el, alpha)))),
        // (F16([a])) => setup_unary(a, move |a| Ok(a.mapv(|el| celu(el, alpha)))),
        // (F64([a])) => setup_unary(a, move |a| Ok(a.mapv(|el| celu(el, alpha)))),
        _ => bail!("Celu: expected floating point input got `{:?}`", inputs),
    }
}

pub fn setup_elu(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let alpha = ctx.attrs.get_with_default("alpha", 1.0)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;
    // Standard only specifies Float
    match inputs {
        (F16([a])) => setup_unary(a, move |a| Ok(a.mapv(|el| elu(el, alpha)))),
        (F32([a])) => setup_unary(a, move |a| Ok(a.mapv(|el| elu(el, alpha)))),
        (F64([a])) => setup_unary(a, move |a| Ok(a.mapv(|el| elu(el, alpha)))),
        _ => bail!("Elu: expected floating point input got `{:?}`", inputs),
    }
}

fn elu<T>(mut x: T, alpha: f32) -> T
where
    T: Float + One + 'static + PartialOrd,
    f32: AsPrimitive<T>,
{
    let alpha = alpha.as_();
    if x < T::zero() {
        alpha * (x.exp() - T::one())
    } else {
        x
    }
}

fn celu<T>(mut x: T, alpha: f32) -> T
where
    T: Float + One + 'static + PartialOrd + Zero,
    f32: AsPrimitive<T>,
{
    // max(0,x) + min(0,alpha*(exp(x/alpha)-1))
    let alpha = alpha.as_();
    if x.is_nan() {
        x
    } else {
        x.max(T::zero()) + (alpha * ((x / alpha).exp() - T::one())).min(T::zero())
    }
}
