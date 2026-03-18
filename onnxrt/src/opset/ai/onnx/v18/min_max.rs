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
) -> Result<(
    Box<dyn crate::Executable + Send + Sync>,
    Vec<crate::TaggedReceiver>,
)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;
    match inputs {
        (I8([a]), HomogeneousNumeric::I8(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (I16([a]), HomogeneousNumeric::I16(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (I32([a]), HomogeneousNumeric::I32(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (I64([a]), HomogeneousNumeric::I64(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (U8([a]), HomogeneousNumeric::U8(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (U16([a]), HomogeneousNumeric::U16(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (U32([a]), HomogeneousNumeric::U32(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (U64([a]), HomogeneousNumeric::U64(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.max(b))
        }
        (F16([a]), HomogeneousNumeric::F16(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| max_nan(*a, *b))
        }
        (F32([a]), HomogeneousNumeric::F32(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| max_nan(*a, *b))
        }
        (F64([a]), HomogeneousNumeric::F64(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| max_nan(*a, *b))
        }
        _ => bail!("Min: unexpected input data types"),
    }
}

pub fn setup_min(
    ctx: OpContext,
) -> Result<(
    Box<dyn crate::Executable + Send + Sync>,
    Vec<crate::TaggedReceiver>,
)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;
    match inputs {
        (I8([a]), HomogeneousNumeric::I8(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (I16([a]), HomogeneousNumeric::I16(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (I32([a]), HomogeneousNumeric::I32(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (I64([a]), HomogeneousNumeric::I64(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (U8([a]), HomogeneousNumeric::U8(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (U16([a]), HomogeneousNumeric::U16(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (U32([a]), HomogeneousNumeric::U32(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (U64([a]), HomogeneousNumeric::U64(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| *a.min(b))
        }
        (F16([a]), HomogeneousNumeric::F16(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| min_nan(*a, *b))
        }
        (F32([a]), HomogeneousNumeric::F32(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| min_nan(*a, *b))
        }
        (F64([a]), HomogeneousNumeric::F64(rest)) => {
            setup_variadic_zipped_reduce(a, rest, |a, b| min_nan(*a, *b))
        }
        _ => bail!("Min: unexpected input data types"),
    }
}
