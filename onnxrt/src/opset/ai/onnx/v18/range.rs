use std::ops::{Add, Mul};

use num_traits::AsPrimitive;

use crate::opset::utils::*;

fn range<T>(start: T, limit: T, delta: T) -> Array1<T>
where
    T: AsPrimitive<f64> + Add<Output = T> + Mul<Output = T>,
    usize: AsPrimitive<T>,
{
    let limit = limit.as_();
    let n_elements = (((limit - start.as_()) / delta.as_()).ceil())
        .max(0.0)
        .as_();

    Array1::from_iter((0..n_elements).map(|i| start + i.as_() * delta))
}

pub fn setup_range(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs: HomogeneousNumericFixed<3> = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        HomogeneousNumericFixed,
        inputs,
        [start, limit, delta],
        setup((start, limit, delta), move |(s, l, d)| {
            let s = s.into_shape(())?.to_owned().into_scalar();
            let l = l.into_shape(())?.to_owned().into_scalar();
            let d = d.into_shape(())?.to_owned().into_scalar();

            Ok::<ArcArrayD<_>, _>(range(s, l, d).into_dyn().into())
        })
    )
}
