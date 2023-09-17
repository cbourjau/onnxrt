use num::{Float, Zero};

use crate::opset::utils::*;

struct IsInf {
    detect_negative: bool,
    detect_positive: bool,
}

impl IsInf {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let detect_negative = attrs.get_with_default("detect_negative", 1i64)? != 0;
        let detect_positive = attrs.get_with_default("detect_positive", 1i64)? != 0;

        Ok(Self {
            detect_negative,
            detect_positive,
        })
    }

    fn compute<T>(&self, a: ArcArrayD<T>) -> Result<ArcArrayD<bool>>
    where
        T: Float + Zero,
    {
        let out = match (self.detect_negative, self.detect_positive) {
            (true, true) => a.mapv(|el| el.is_infinite()),
            (true, false) => a.mapv(|el| el.is_infinite() && (el < T::zero())),
            (false, true) => a.mapv(|el| el.is_infinite() && (el > T::zero())),
            (false, false) => a.mapv(|_| false),
        };
        Ok(out.into())
    }
}

pub fn setup_isinf(ctx: OpContext) -> Result<ExecutionStep> {
    let state = IsInf::from_attrs(&ctx.attrs)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;

    match inputs {
        F32(a) => setup(a, move |a| state.compute(a)),
        F64(a) => setup(a, move |a| state.compute(a)),
    }
}
