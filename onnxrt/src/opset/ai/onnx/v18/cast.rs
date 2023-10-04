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

fn setup_cast_(input: TaggedTensorReceiver, to: DataType) -> Result<ExecutionStep> {
    macro_rules! make_output_fallible {
        ($inp:expr, $cast:expr) => {{
            wrap_1_input($inp, |a| {
                let str_vec = a.iter().map($cast).collect::<Result<_, _>>()?;
                let out = Array::from_vec(str_vec).into_shape(a.shape())?;
                Ok(out.into())
            })
        }};
    }

    // Match arm for casts from numbers to number
    macro_rules! num_to_num {
        ($a_ch:tt) => {{
            use num::zero;
            use DataType::*;
            match to {
                Bool => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el != zero()).into())),
                F32 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
                F64 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as f64).into())),

                I8 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as i8).into())),
                I16 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as i16).into())),
                I32 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
                I64 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as i64).into())),

                U8 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as u8).into())),
                U16 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as u16).into())),
                U32 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
                U64 => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el as u64).into())),

                String => wrap_1_input($a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
            }
        }};
    }

    match input {
        TaggedTensorReceiver::F32(a_ch) => num_to_num!(a_ch),
        TaggedTensorReceiver::F64(a_ch) => num_to_num!(a_ch),

        TaggedTensorReceiver::I8(a_ch) => num_to_num!(a_ch),
        TaggedTensorReceiver::I16(a_ch) => num_to_num!(a_ch),
        TaggedTensorReceiver::I32(a_ch) => num_to_num!(a_ch),
        TaggedTensorReceiver::I64(a_ch) => num_to_num!(a_ch),

        TaggedTensorReceiver::U8(a_ch) => num_to_num!(a_ch),
        TaggedTensorReceiver::U16(a_ch) => num_to_num!(a_ch),
        TaggedTensorReceiver::U32(a_ch) => num_to_num!(a_ch),
        TaggedTensorReceiver::U64(a_ch) => num_to_num!(a_ch),

        TaggedTensorReceiver::Bool(a_ch) => match to {
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f64).into())),

            DataType::I8 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i8).into())),
            DataType::I16 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i16).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),

            DataType::U8 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8).into())),
            DataType::U16 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u16).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),

            DataType::Bool => wrap_1_input(a_ch, Ok),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedTensorReceiver::String(a_ch) => match to {
            DataType::F32 => make_output_fallible!(a_ch, |el| el.parse::<f32>()),
            DataType::F64 => make_output_fallible!(a_ch, |el| el.parse::<f64>()),

            DataType::I8 => make_output_fallible!(a_ch, |el| el.parse::<i8>()),
            DataType::I16 => make_output_fallible!(a_ch, |el| el.parse::<i16>()),
            DataType::I32 => make_output_fallible!(a_ch, |el| el.parse::<i32>()),
            DataType::I64 => make_output_fallible!(a_ch, |el| el.parse::<i64>()),

            DataType::U8 => make_output_fallible!(a_ch, |el| el.parse::<u8>()),
            DataType::U16 => make_output_fallible!(a_ch, |el| el.parse::<u16>()),
            DataType::U32 => make_output_fallible!(a_ch, |el| el.parse::<u32>()),
            DataType::U64 => make_output_fallible!(a_ch, |el| el.parse::<u64>()),

            DataType::Bool => make_output_fallible!(a_ch, |el| el.parse::<bool>()),
            DataType::String => wrap_1_input(a_ch, Ok),
        },
    }
}
