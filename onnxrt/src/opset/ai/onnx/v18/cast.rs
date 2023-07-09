use crate::opset::utils::*;
use crate::DataType;

pub fn setup_cast(ctx: OpContext) -> Result<ExecutionStep> {
    let to_type = DataType::from_i32(i64::try_from(ctx.attrs.get("to")?)? as i32)?;

    let a_ch = ctx.subscribe_to_inputs().narrow()?;

    setup_cast_(a_ch, to_type)
}

pub fn setup_cast_like(ctx: OpContext) -> Result<ExecutionStep> {
    // The tricky part here is to avoid actually subscribing to the
    // target input. We know the dtype without it and that allows us
    // to reuse the actual `Cast` logic.
    let to_type = ctx.input_dtypes()[1];
    let (inp,) = ctx.subscribe_to_inputs_by_index(&[0]).narrow()?;
    setup_cast_(inp, to_type)
}

fn setup_cast_(input: TaggedReceiver, to: DataType) -> Result<ExecutionStep> {
    macro_rules! make_output_fallible {
        ($inp:expr, $cast:expr) => {{
            wrap_1_input($inp, |a| {
                let str_vec = a.iter().map($cast).collect::<Result<_, _>>()?;
                let out = Array::from_vec(str_vec).into_shape(a.shape())?;
                Ok(out.into())
            })
        }};
    }

    match input {
        TaggedReceiver::Bool(a_ch) => match to {
            DataType::Bool => wrap_1_input(a_ch, Ok),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::F32(a_ch) => match to {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, Ok),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::F64(a_ch) => match to {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, Ok),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::I32(a_ch) => match to {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, Ok),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::I64(a_ch) => match to {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, Ok),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::U32(a_ch) => match to {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, Ok),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::U64(a_ch) => match to {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::String(a_ch) => match to {
            DataType::Bool => make_output_fallible!(a_ch, |el| el.parse::<bool>()),
            DataType::F32 => make_output_fallible!(a_ch, |el| el.parse::<f32>()),
            DataType::F64 => make_output_fallible!(a_ch, |el| el.parse::<f64>()),
            DataType::I32 => make_output_fallible!(a_ch, |el| el.parse::<i32>()),
            DataType::I64 => make_output_fallible!(a_ch, |el| el.parse::<i64>()),
            DataType::U32 => make_output_fallible!(a_ch, |el| el.parse::<u32>()),
            DataType::U64 => make_output_fallible!(a_ch, |el| el.parse::<u64>()),
            DataType::String => wrap_1_input(a_ch, Ok),
        },
    }
}
