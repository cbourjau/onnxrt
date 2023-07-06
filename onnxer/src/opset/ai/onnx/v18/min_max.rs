use num::Float;
use ordered_float::OrderedFloat;

use crate::opset::utils::*;

fn min_float<T>(a: T, b: T) -> T
where
    T: Float,
{
    Float::min(OrderedFloat(a), OrderedFloat(b)).0
}

fn max_float<T>(a: T, b: T) -> T
where
    T: Float,
{
    Float::max(OrderedFloat(a), OrderedFloat(b)).0
}

fn reduce<T, F>((inputs,): (Vec<ArcArrayD<T>>,), fun: F) -> Result<ArcArrayD<T>>
where
    T: Clone,
    F: Fn(&T, &T) -> T + Copy + Send + Sync + 'static,
{
    let mut inputs = inputs.into_iter();
    let mut acc = inputs
        .next()
        .ok_or_else(|| anyhow!("'Min' operator expects at least one input"))?;

    for el in inputs {
        acc = zip_map_with_co_broadcast(acc, el, fun)?;
    }
    Ok(acc)
}

pub fn setup_min(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumeric::*;
    match inputs {
        F32(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| min_float(a, b))),
        F64(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| min_float(a, b))),
        I32(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::min(a, b))),
        I64(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::min(a, b))),
        U32(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::min(a, b))),
        U64(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::min(a, b))),
    }
}

pub fn setup_max(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumeric::*;
    match inputs {
        F32(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| max_float(a, b))),
        F64(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| max_float(a, b))),
        I32(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::max(a, b))),
        I64(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::max(a, b))),
        U32(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::max(a, b))),
        U64(inputs) => setup((inputs,), |inp| reduce(inp, |&a, &b| Ord::max(a, b))),
    }
}
