use num::{Float, Zero};
use num_traits::AsPrimitive;

use crate::opset::utils::*;

struct AttrRelu {
    alpha: f32,
}

impl AttrRelu {
    fn from_attrs(attrs: &Attributes, default: f32) -> Result<Self> {
        let alpha = attrs.get_with_default("alpha", default)?;

        Ok(Self { alpha })
    }

    fn leaky<T>(&self, a: ArcArrayD<T>) -> Result<ArcArrayD<T>>
    where
        T: Copy + 'static + Zero + Float,
        f32: AsPrimitive<T>,
    {
        let alpha: T = self.alpha.as_();
        let out = a.mapv(|el| if el < T::zero() { alpha * el } else { el });
        Ok(out.into())
    }

    fn thresholded<T>(&self, a: ArcArrayD<T>) -> Result<ArcArrayD<T>>
    where
        T: Copy + 'static + Zero + Float,
        f32: AsPrimitive<T>,
    {
        let alpha: T = self.alpha.as_();
        let out = a.mapv(|el| if el > alpha { el } else { T::zero() });
        Ok(out.into())
    }
}

pub fn setup_leaky_relu(ctx: OpContext) -> Result<ExecutionStep> {
    let state = AttrRelu::from_attrs(&ctx.attrs, 0.01)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;

    match inputs {
        F32(a) => setup(a, move |a| state.leaky(a)),
        F64(a) => setup(a, move |a| state.leaky(a)),
    }
}

pub fn setup_thresholded_relu(ctx: OpContext) -> Result<ExecutionStep> {
    let state = AttrRelu::from_attrs(&ctx.attrs, 1.0)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;

    match inputs {
        F32(a) => setup(a, move |a| state.thresholded(a)),
        F64(a) => setup(a, move |a| state.thresholded(a)),
    }
}
