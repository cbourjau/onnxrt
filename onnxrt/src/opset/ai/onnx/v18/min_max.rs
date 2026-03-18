use anyhow::bail;
use num::Float;
use ordered_float::OrderedFloat;

use crate::{opset::utils::*, session::OpContext};

/// NaN-aware version of min
pub fn min_nan<T>(a: T, b: T) -> T
where
    T: Float,
{
    if a.is_nan() | b.is_nan() {
        T::nan()
    } else {
        a.min(b)
    }
}

/// NaN-aware version of max
pub fn max_nan<T>(a: T, b: T) -> T
where
    T: Float,
{
    if a.is_nan() | b.is_nan() {
        T::nan()
    } else {
        a.max(b)
    }
}

pub fn setup_max(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumeric::*;
    match inputs {
        (I8(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (I16(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (I32(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (I64(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (U8(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (U16(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (U32(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (U64(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.max(b)),
        (F16(rest)) => setup_variadic_zipped_reduce(rest, |a, b| max_nan(*a, *b)),
        (F32(rest)) => setup_variadic_zipped_reduce(rest, |a, b| max_nan(*a, *b)),
        (F64(rest)) => setup_variadic_zipped_reduce(rest, |a, b| max_nan(*a, *b)),
        _ => bail!("Max: unexpected input data types"),
    }
}

pub fn setup_min(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumeric::*;
    match inputs {
        (I8(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (I16(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (I32(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (I64(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (U8(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (U16(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (U32(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (U64(rest)) => setup_variadic_zipped_reduce(rest, |a, b| *a.min(b)),
        (F16(rest)) => setup_variadic_zipped_reduce(rest, |a, b| min_nan(*a, *b)),
        (F32(rest)) => setup_variadic_zipped_reduce(rest, |a, b| min_nan(*a, *b)),
        (F64(rest)) => setup_variadic_zipped_reduce(rest, |a, b| min_nan(*a, *b)),
        _ => bail!("Min: unexpected input data types"),
    }
}
