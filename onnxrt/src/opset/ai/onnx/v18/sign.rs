use std::{cmp::Ordering, ops::Neg};

use num_traits::{One, Zero};

use crate::opset::utils::*;

fn sign<T>(el: T) -> T
where
    T: One + Zero + PartialOrd + Ord + Neg<Output = T>,
{
    match el.cmp(&T::zero()) {
        Ordering::Greater => T::one(),
        Ordering::Less => T::one().neg(),
        Ordering::Equal => T::zero(),
    }
}

fn sign_for_unsigned<T>(el: T) -> T
where
    T: One + Zero + PartialOrd + Ord,
{
    if el > T::zero() { T::one() } else { T::zero() }
}

pub fn setup_sign(ctx: OpContext) -> Result<ExecutionStep> {
    let x = ctx.subscribe_to_inputs().narrow()?;

    // Spec: ambiguous for floating points / NaN
    use Integer::*;
    match x {
        U8(x) => setup(x, move |x| Ok(x.mapv(sign_for_unsigned))),
        U16(x) => setup(x, move |x| Ok(x.mapv(sign_for_unsigned))),
        U32(x) => setup(x, move |x| Ok(x.mapv(sign_for_unsigned))),
        U64(x) => setup(x, move |x| Ok(x.mapv(sign_for_unsigned))),
        I8(x) => setup(x, move |x| Ok(x.mapv(sign))),
        I16(x) => setup(x, move |x| Ok(x.mapv(sign))),
        I32(x) => setup(x, move |x| Ok(x.mapv(sign))),
        I64(x) => setup(x, move |x| Ok(x.mapv(sign))),
    }
}
