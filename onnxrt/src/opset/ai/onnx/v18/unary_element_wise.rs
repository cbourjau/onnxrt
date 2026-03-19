use std::ops::{Add, Mul, Neg};

use anyhow::bail;
use half::f16;
use libm;
use num::{One, Zero};
use num_traits::Float;

use crate::{opset::utils::*, session::OpContext};

pub fn setup_abs(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use Numeric::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.abs())),
        F32(a) => setup_unary(a, |a| Ok(a.abs())),
        F64(a) => setup_unary(a, |a| Ok(a.abs())),

        I8(a) => setup_unary(a, |a| Ok(a.mapv(|el| el.abs()))),
        I16(a) => setup_unary(a, |a| Ok(a.mapv(|el| el.abs()))),
        I32(a) => setup_unary(a, |a| Ok(a.mapv(|el| el.abs()))),
        I64(a) => setup_unary(a, |a| Ok(a.mapv(|el| el.abs()))),
        U8(a) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        U16(a) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        U32(a) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        U64(a) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
    }
}
pub fn setup_acos(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.acos())),
        F32(a) => setup_unary(a, |a| Ok(a.acos())),
        F64(a) => setup_unary(a, |a| Ok(a.acos())),
    }
}

pub fn setup_acosh(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.acosh())),
        F32(a) => setup_unary(a, |a| Ok(a.acosh())),
        F64(a) => setup_unary(a, |a| Ok(a.acosh())),
    }
}

pub fn setup_asin(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.asin())),
        F32(a) => setup_unary(a, |a| Ok(a.asin())),
        F64(a) => setup_unary(a, |a| Ok(a.asin())),
    }
}

pub fn setup_asinh(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.asinh())),
        F32(a) => setup_unary(a, |a| Ok(a.asinh())),
        F64(a) => setup_unary(a, |a| Ok(a.asinh())),
    }
}

pub fn setup_atan(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.atan())),
        F32(a) => setup_unary(a, |a| Ok(a.atan())),
        F64(a) => setup_unary(a, |a| Ok(a.atan())),
    }
}

pub fn setup_atanh(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.atanh())),
        F32(a) => setup_unary(a, |a| Ok(a.atanh())),
        F64(a) => setup_unary(a, |a| Ok(a.atanh())),
    }
}

pub fn setup_bitwise_not(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use Integer::*;
    match a {
        I8(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
        I16(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
        I32(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
        I64(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
        U8(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
        U16(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
        U32(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
        U64(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(|el| !el))),
    }
}

pub fn setup_ceil(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.ceil())),
        F32(a) => setup_unary(a, |a| Ok(a.ceil())),
        F64(a) => setup_unary(a, |a| Ok(a.ceil())),
    }
}

pub fn setup_cos(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.cos())),
        F32(a) => setup_unary(a, |a| Ok(a.cos())),
        F64(a) => setup_unary(a, |a| Ok(a.cos())),
    }
}

pub fn setup_cosh(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.cosh())),
        F32(a) => setup_unary(a, |a| Ok(a.cosh())),
        F64(a) => setup_unary(a, |a| Ok(a.cosh())),
    }
}

pub fn setup_erf(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| {
            Ok(a.to_owned()
                .mapv(|el| f16::from_f32(libm::erff(f32::from(el)))))
        }),
        F32(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(libm::erff))),
        F64(a) => setup_unary(a, |a| Ok(a.to_owned().mapv(libm::erf))),
    }
}

pub fn setup_exp(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.exp())),
        F32(a) => setup_unary(a, |a| Ok(a.exp())),
        F64(a) => setup_unary(a, |a| Ok(a.exp())),
    }
}

pub fn setup_floor(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.floor())),
        F32(a) => setup_unary(a, |a| Ok(a.floor())),
        F64(a) => setup_unary(a, |a| Ok(a.floor())),
    }
}

pub fn setup_isinf(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.is_infinite())),
        F32(a) => setup_unary(a, |a| Ok(a.is_infinite())),
        F64(a) => setup_unary(a, |a| Ok(a.is_infinite())),
    }
}
pub fn setup_isnan(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.is_nan())),
        F32(a) => setup_unary(a, |a| Ok(a.is_nan())),
        F64(a) => setup_unary(a, |a| Ok(a.is_nan())),
    }
}

pub fn setup_log(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.ln())),
        F32(a) => setup_unary(a, |a| Ok(a.ln())),
        F64(a) => setup_unary(a, |a| Ok(a.ln())),
    }
}

pub fn setup_neg(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use Numeric::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.to_owned().neg())),
        F32(a) => setup_unary(a, |a| Ok(a.to_owned().neg())),
        F64(a) => setup_unary(a, |a| Ok(a.to_owned().neg())),

        I8(a) => setup_unary(a, |a| Ok(a.to_owned().neg())),
        I16(a) => setup_unary(a, |a| Ok(a.to_owned().neg())),
        I32(a) => setup_unary(a, |a| Ok(a.to_owned().neg())),
        I64(a) => setup_unary(a, |a| Ok(a.to_owned().neg())),

        _ => bail!("Neg: expected signed data type"),
    }
}

pub fn setup_not(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;
    use TaggedTensorReceiver::*;
    match a {
        Bool(a) => setup_unary(a, |a| Ok(!a.to_owned())),
        _ => bail!("Not: expected boolean data type"),
    }
}

pub fn setup_reciprocal(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.recip())),
        F32(a) => setup_unary(a, |a| Ok(a.recip())),
        F64(a) => setup_unary(a, |a| Ok(a.recip())),
    }
}

pub fn setup_round(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| {
            Ok(a.mapv(|el| f16::from_f32(f32::from(el).round_ties_even())))
        }),
        F32(a) => setup_unary(a, |a| Ok(a.mapv(|el| el.round_ties_even()))),
        F64(a) => setup_unary(a, |a| Ok(a.mapv(|el| el.round_ties_even()))),
    }
}

fn sigmoid<T>(x: T) -> T
where
    T: Float,
{
    if x > T::zero() {
        T::one() / (T::one() + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (T::one() + exp_x)
    }
}

pub fn setup_sigmoid(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.mapv(sigmoid))),
        F32(a) => setup_unary(a, |a| Ok(a.mapv(sigmoid))),
        F64(a) => setup_unary(a, |a| Ok(a.mapv(sigmoid))),
    }
}

fn sign<T>(x: T) -> T
where
    T: Zero + One + Neg<Output = T> + PartialOrd + Mul<Output = T>,
{
    if x > T::zero() {
        T::one()
    } else if x < T::zero() {
        -T::one()
    } else {
        // Propagate nan like the array-api NumPy. The standard does not define this!
        // TODO: Upstream clarification
        T::zero() * x
    }
}

pub fn setup_sign(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use Numeric::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.mapv(sign))),
        F32(a) => setup_unary(a, |a| Ok(a.mapv(sign))),
        F64(a) => setup_unary(a, |a| Ok(a.mapv(sign))),
        I8(a) => setup_unary(a, |a| Ok(a.mapv(sign))),
        I16(a) => setup_unary(a, |a| Ok(a.mapv(sign))),
        I32(a) => setup_unary(a, |a| Ok(a.mapv(sign))),
        I64(a) => setup_unary(a, |a| Ok(a.mapv(sign))),
        U8(a) => setup_unary(a, |a| Ok(a.mapv(|x| (x > 0) as u8))),
        U16(a) => setup_unary(a, |a| Ok(a.mapv(|x| (x > 0) as u16))),
        U32(a) => setup_unary(a, |a| Ok(a.mapv(|x| (x > 0) as u32))),
        U64(a) => setup_unary(a, |a| Ok(a.mapv(|x| (x > 0) as u64))),
    }
}

pub fn setup_sin(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.sin())),
        F32(a) => setup_unary(a, |a| Ok(a.sin())),
        F64(a) => setup_unary(a, |a| Ok(a.sin())),
    }
}

pub fn setup_sinh(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.sinh())),
        F32(a) => setup_unary(a, |a| Ok(a.sinh())),
        F64(a) => setup_unary(a, |a| Ok(a.sinh())),
    }
}

pub fn setup_sqrt(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.sqrt())),
        F32(a) => setup_unary(a, |a| Ok(a.sqrt())),
        F64(a) => setup_unary(a, |a| Ok(a.sqrt())),
    }
}

pub fn setup_tan(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.tan())),
        F32(a) => setup_unary(a, |a| Ok(a.tan())),
        F64(a) => setup_unary(a, |a| Ok(a.tan())),
    }
}

pub fn setup_tanh(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, |a| Ok(a.tanh())),
        F32(a) => setup_unary(a, |a| Ok(a.tanh())),
        F64(a) => setup_unary(a, |a| Ok(a.tanh())),
    }
}

pub fn setup_mish(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(
            a,
            |a| Ok(a.mapv(|x| x * (f16::one() + x.exp()).ln().tanh())),
        ),
        F32(a) => setup_unary(a, |a| Ok(a.mapv(|x| x * (1.0 + x.exp()).ln().tanh()))),
        F64(a) => setup_unary(a, |a| Ok(a.mapv(|x| x * (1.0 + x.exp()).ln().tanh()))),
    }
}

fn leaky_relu<T>(x: T, alpha: T) -> T
where
    T: Float,
{
    if x < T::zero() { alpha * x } else { x }
}
pub fn setup_leaky_relu(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let alpha = ctx.attrs.get_with_default("alpha", 0.01f32)?;
    let a = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a {
        F16(a) => setup_unary(a, move |a| {
            Ok(a.mapv(|x| leaky_relu(x, f16::from_f32(alpha))))
        }),
        F32(a) => setup_unary(a, move |a| Ok(a.mapv(|x| leaky_relu(x, alpha as _)))),
        F64(a) => setup_unary(a, move |a| Ok(a.mapv(|x| leaky_relu(x, alpha as _)))),
    }
}
