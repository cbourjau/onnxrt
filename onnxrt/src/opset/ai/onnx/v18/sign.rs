use std::ops::Neg;

use num_traits::{One, Zero};

use crate::opset::utils::*;

fn sign<T>(el: T) -> T
where
    T: One + Zero + PartialOrd + Ord + Neg<Output = T>,
{
    if el > T::zero() {
        T::one()
    } else if el < T::zero() {
        T::one().neg()
    } else {
        T::zero()
    }
}

fn sign_for_unsigned<T>(el: T) -> T
where
    T: One + Zero + PartialOrd + Ord,
{
    if el > T::zero() {
        T::one()
    } else {
        T::zero()
    }
}

pub fn setup_sign(ctx: OpContext) -> Result<ExecutionStep> {
    let x = ctx.subscribe_to_inputs().narrow()?;

    // Spec: ambiguous for floating points / NaN
    use Integer::*;
    match x {
        U8(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign_for_unsigned(el)).into())
        }),
        U16(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign_for_unsigned(el)).into())
        }),
        U32(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign_for_unsigned(el)).into())
        }),
        U64(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign_for_unsigned(el)).into())
        }),
        I8(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign(el)).into())
        }),
        I16(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign(el)).into())
        }),
        I32(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign(el)).into())
        }),
        I64(x) => setup((x,), move |(x,)| {
            Ok::<ArcArrayD<_>, _>(x.mapv(|el| sign(el)).into())
        }),
    }
}
