use anyhow::bail;

use crate::{TaggedReceiver, opset::utils::*, session::OpContext};

pub fn setup_equal(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousFixed::*;

    match inputs {
        (F16([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (F32([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (F64([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),

        (I8([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (I16([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (I32([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (I64([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),

        (U8([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (U16([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (U32([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (U64([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),

        (Bool([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
        (String([a, b])) => setup_binary_zipped(a, b, |a, b| a == b),
    }
}

pub fn setup_greater(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let (a, b) = ctx.subscribe_to_inputs().narrow()?;

    setup_greater_inner(a, b)
}

pub fn setup_greater_equal(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let (a, b) = ctx.subscribe_to_inputs().narrow()?;

    setup_greater_equal_inner(a, b)
}

pub fn setup_less(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let (a, b) = ctx.subscribe_to_inputs().narrow()?;

    setup_greater_inner(b, a)
}

pub fn setup_less_equal(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let (a, b) = ctx.subscribe_to_inputs().narrow()?;

    setup_greater_equal_inner(b, a)
}

fn setup_greater_inner(
    a: TaggedReceiver,
    b: TaggedReceiver,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = vec![a, b].narrow()?;
    use HomogeneousNumericFixed::*;
    match inputs {
        (F16([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (F32([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (F64([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),

        (I8([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (I16([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (I32([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (I64([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),

        (U8([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (U16([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (U32([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
        (U64([a, b])) => setup_binary_zipped(a, b, |a, b| a > b),
    }
}

fn setup_greater_equal_inner(
    a: TaggedReceiver,
    b: TaggedReceiver,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let inputs = vec![a, b].narrow()?;
    use HomogeneousNumericFixed::*;
    match inputs {
        (F16([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (F32([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (F64([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),

        (I8([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (I16([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (I32([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (I64([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),

        (U8([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (U16([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (U32([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
        (U64([a, b])) => setup_binary_zipped(a, b, |a, b| a >= b),
    }
}
