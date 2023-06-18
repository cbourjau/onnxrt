use std::ops::Mul;

use num::Zero;

use super::*;

use crate::{input_types::HomogeneousNumeric, setup_utils::setup};

fn min<T>((inputs,): (Vec<ArcArrayD<T>>,)) -> Result<ArcArrayD<T>>
where
    T: Clone + PartialOrd + Copy + Zero + Mul<Output = T>,
{
    let mut inputs = inputs.into_iter();
    let mut acc = inputs
        .next()
        .ok_or_else(|| anyhow!("'Min' operator expects at least one input"))?;

    for el in inputs {
        acc = zip_map_with_co_broadcast(acc, el, |a, b| {
            if a <= b {
                *a
            } else if a > b {
                *b
            } else {
                // The nan case. We don't know which one is nan but the mul will take care of it.
                *a * (*b) * T::zero()
            }
        })?;
    }
    Ok(acc)
}

pub fn setup_min(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs = HomogeneousNumeric::new_from_iter(ctx.input_channels().into_iter())?;

    use HomogeneousNumeric::*;
    match inputs {
        F32(inputs) => setup((inputs,), min),
        F64(inputs) => setup((inputs,), min),
        I32(inputs) => setup((inputs,), min),
        I64(inputs) => setup((inputs,), min),
        U32(inputs) => setup((inputs,), min),
        U64(inputs) => setup((inputs,), min),
    }
}
