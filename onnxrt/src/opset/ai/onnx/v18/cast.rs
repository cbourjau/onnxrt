use anyhow::bail;
use half::f16;
use num::{FromPrimitive, One, ToPrimitive, Zero};
use num_traits::Float;

use crate::{DataType, opset::utils::*, session::OpContext};

pub fn setup_cast(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let to_type = DataType::from_i32(i64::try_from(ctx.attrs.get("to")?)? as i32)?;
    let a = ctx.subscribe_to_inputs().narrow()?;
    cast(a, to_type)
}

pub fn setup_cast_like(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    // The tricky part here is to avoid actually subscribing to the
    // target input. We know the dtype without it and that allows us
    // to reuse the actual `Cast` logic.
    let to_type = ctx.input_dtypes()[1];
    let (a,) = ctx.subscribe_to_inputs_by_index(&[0]).narrow()?;
    cast(a, to_type)
}

fn cast(
    a: TaggedTensorReceiver,
    to_type: DataType,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    use TaggedTensorReceiver::*;
    // Standard: "Conversion from a numerical type to any numerical type is always allowed"
    // Conversions from values that are out of range are undefined
    match (a, to_type) {
        (F16(a), DataType::F16) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (F16(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_f32()))),
        (F16(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_f64()))),
        (F16(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (F16(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (F16(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (F16(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (F16(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (F16(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (F16(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (F16(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (F16(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != f16::zero()))),
        (F16(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (F32(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(f16::from_f32))),
        (F32(a), DataType::F32) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (F32(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (F32(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (F32(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (F32(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (F32(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (F32(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (F32(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (F32(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (F32(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (F32(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0.0))),
        (F32(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (F64(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(f16::from_f64))),
        (F64(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (F64(a), DataType::F64) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (F64(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (F64(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (F64(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (F64(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (F64(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (F64(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (F64(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (F64(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (F64(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0.0))),
        (F64(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (U8(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(f16::from))),
        (U8(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (U8(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (U8(a), DataType::U8) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (U8(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (U8(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (U8(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (U8(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (U8(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (U8(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (U8(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (U8(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (U8(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (U16(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(|x| f16::from_f32(x as f32)))),
        (U16(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (U16(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (U16(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (U16(a), DataType::U16) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (U16(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (U16(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (U16(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (U16(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (U16(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (U16(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (U16(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (U16(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (U32(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(|x| f16::from_f32(x as f32)))),
        (U32(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (U32(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (U32(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (U32(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (U32(a), DataType::U32) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (U32(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (U32(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (U32(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (U32(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (U32(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (U32(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (U32(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (U64(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(|x| f16::from_f32(x as f32)))),
        (U64(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (U64(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (U64(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (U64(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (U64(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (U64(a), DataType::U64) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (U64(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (U64(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (U64(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (U64(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (U64(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (U64(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (I8(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(f16::from))),
        (I8(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (I8(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (I8(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (I8(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (I8(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (I8(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (I8(a), DataType::I8) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (I8(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (I8(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (I8(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (I8(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (I8(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (I16(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(|x| f16::from_f32(x as f32)))),
        (I16(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (I16(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (I16(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (I16(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (I16(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (I16(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (I16(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (I16(a), DataType::I16) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (I16(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (I16(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (I16(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (I16(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (I32(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(|x| f16::from_f32(x as f32)))),
        (I32(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (I32(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (I32(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (I32(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (I32(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (I32(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (I32(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (I32(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (I32(a), DataType::I32) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (I32(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i64().unwrap_or(0)))),
        (I32(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (I32(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (I64(a), DataType::F16) => setup_unary(a, |a| Ok(a.mapv(|x| f16::from_f32(x as f32)))),
        (I64(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| x as f32))),
        (I64(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| x as f64))),
        (I64(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u8().unwrap_or(0)))),
        (I64(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u16().unwrap_or(0)))),
        (I64(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u32().unwrap_or(0)))),
        (I64(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_u64().unwrap_or(0)))),
        (I64(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i8().unwrap_or(0)))),
        (I64(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i16().unwrap_or(0)))),
        (I64(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| x.to_i32().unwrap_or(0)))),
        (I64(a), DataType::I64) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (I64(a), DataType::Bool) => setup_unary(a, |a| Ok(a.mapv(|x| x != 0))),
        (I64(a), DataType::String) => setup_unary(a, |a| Ok(a.mapv(|x| format!("{}", x)))),

        (Bool(a), DataType::F16) => setup_unary(a, |a| {
            Ok(a.mapv(|x| if x { f16::one() } else { f16::zero() }))
        }),
        (Bool(a), DataType::F32) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1f32 } else { 0.0 }))),
        (Bool(a), DataType::F64) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1f64 } else { 0.0 }))),
        (Bool(a), DataType::U8) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1u8 } else { 0 }))),
        (Bool(a), DataType::U16) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1u16 } else { 0 }))),
        (Bool(a), DataType::U32) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1u32 } else { 0 }))),
        (Bool(a), DataType::U64) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1u64 } else { 0 }))),
        (Bool(a), DataType::I8) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1i8 } else { 0 }))),
        (Bool(a), DataType::I16) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1i16 } else { 0 }))),
        (Bool(a), DataType::I32) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1i32 } else { 0 }))),
        (Bool(a), DataType::I64) => setup_unary(a, |a| Ok(a.mapv(|x| if x { 1i64 } else { 0 }))),
        (Bool(a), DataType::Bool) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
        (Bool(a), DataType::String) => {
            bail!("Cast: string representation of boolean values is undefined by the standard")
        }

        (String(a), DataType::F16) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<f16>())),
        (String(a), DataType::F32) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<f32>())),
        (String(a), DataType::F64) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<f64>())),
        (String(a), DataType::U8) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<u8>())),
        (String(a), DataType::U16) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<u16>())),
        (String(a), DataType::U32) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<u32>())),
        (String(a), DataType::U64) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<u64>())),
        (String(a), DataType::I8) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<i8>())),
        (String(a), DataType::I16) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<i16>())),
        (String(a), DataType::I32) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<i32>())),
        (String(a), DataType::I64) => setup_unary(a, |a| try_map(a.view(), |x| x.parse::<i64>())),
        // Standard is unclear about boolean string representation!
        (String(a), DataType::Bool) => {
            bail!("Cast: string representation of boolean values is undefined by the standard")
        }
        (String(a), DataType::String) => Ok((Box::new(|| Ok(())) as _, vec![a.into()])),
    }
}

fn try_map<T1, T2, F, E>(a: ArrayViewD<T1>, f: F) -> Result<ArrayD<T2>>
where
    F: Fn(&T1) -> std::result::Result<T2, E>,
    T2: Default + Clone,
    E: std::error::Error + Send + Sync,
{
    let mut out = ndarray::ArrayD::from_elem(a.shape(), T2::default());
    for (x1, x2) in a.iter().zip(out.iter_mut()) {
        *x2 = f(x1).map_err(|e| anyhow::anyhow!("{:?}", e))?;
    }
    Ok(out)
}
