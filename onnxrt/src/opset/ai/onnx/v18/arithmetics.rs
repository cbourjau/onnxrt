use anyhow::bail;

use crate::{opset::utils::*, session::OpContext};

pub fn setup_div(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;

    match inputs {
        (F16([a, b])) => setup_binary_zipped(a, b, |a, b| a / b),
        (F32([a, b])) => setup_binary_zipped(a, b, |a, b| a / b),
        (F64([a, b])) => setup_binary_zipped(a, b, |a, b| a / b),
        (U8([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
        (U16([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
        (U32([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
        (U64([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
        (I8([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
        (I16([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
        (I32([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
        (I64([a, b])) => setup_binary_zipped(a, b, |a, b| num::integer::div_floor(*a, *b)),
    }
}

pub fn setup_add(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;

    match inputs {
        (F16([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (F32([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (F64([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (U8([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (U16([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (U32([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (U64([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (I8([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (I16([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (I32([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
        (I64([a, b])) => setup_binary_zipped(a, b, |a, b| a + b),
    }
}

pub fn setup_mul(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;

    match inputs {
        (F16([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (F32([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (F64([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (U8([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (U16([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (U32([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (U64([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (I8([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (I16([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (I32([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
        (I64([a, b])) => setup_binary_zipped(a, b, |a, b| a * b),
    }
}

pub fn setup_sub(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;

    match inputs {
        (F16([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (F32([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (F64([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (U8([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (U16([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (U32([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (U64([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (I8([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (I16([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (I32([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
        (I64([a, b])) => setup_binary_zipped(a, b, |a, b| a - b),
    }
}
