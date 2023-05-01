use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Receiver;

use anyhow::{anyhow, bail, Result};
use itertools::sorted;
use ndarray::{Array, Array2, Axis, IntoDimension, Ix1, IxDyn, Zip};
use onnx_protos::onnx_proto3::GraphProto;

use crate::attribute::Attribute;
use crate::session::{ExecutionStep, OpContext, SetupFnSignature};
use crate::utils::{
    co_broadcast, convert_negative_index, wrap_1_input, wrap_binary_input,
    zip_map_with_co_broadcast,
};
use crate::{
    channel, value_channel, ArcArrayD, DataType, Dispatchable, EnumReceiver,
    HomogeneousNumericFixed, Session,
};

macro_rules! element_wise_numeric {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let chs = ctx.input_channels().try_into()?;
            match chs {
                HomogeneousNumericFixed::F32([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
                HomogeneousNumericFixed::F64([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
                HomogeneousNumericFixed::I32([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
                HomogeneousNumericFixed::I64([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
            }
        })
    };
}

/// Make setup function for element wise operations with floating point input
macro_rules! element_wise_float {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
            match a_ch {
                EnumReceiver::F32(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                EnumReceiver::F64(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                // Internal error. This should never happen.
                rec => bail!("Expected floating point input, found: '{:?}'", rec),
            }
        })
    };
}

/// Setup functions with two numeric inputs
macro_rules! binary_numeric {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let channels = ctx.input_channels().try_into()?;
            match channels {
                HomogeneousNumericFixed::F32([a_ch, b_ch]) => wrap_binary_input(a_ch, b_ch, $fun),
                HomogeneousNumericFixed::F64([a_ch, b_ch]) => wrap_binary_input(a_ch, b_ch, $fun),
                HomogeneousNumericFixed::I32([a_ch, b_ch]) => wrap_binary_input(a_ch, b_ch, $fun),
                HomogeneousNumericFixed::I64([a_ch, b_ch]) => wrap_binary_input(a_ch, b_ch, $fun),
            }
        })
    };
}

fn make_equal() -> SetupFnSignature {
    binary_numeric!(|a, b| zip_map_with_co_broadcast(a, b, PartialEq::eq))
}

fn make_less_equal() -> SetupFnSignature {
    binary_numeric!(|a, b| zip_map_with_co_broadcast(a, b, PartialOrd::le))
}

fn make_less() -> SetupFnSignature {
    binary_numeric!(|a, b| zip_map_with_co_broadcast(a, b, PartialOrd::lt))
}

fn make_greater_equal() -> SetupFnSignature {
    binary_numeric!(|a, b| zip_map_with_co_broadcast(a, b, PartialOrd::ge))
}

fn make_greater() -> SetupFnSignature {
    binary_numeric!(|a, b| zip_map_with_co_broadcast(a, b, PartialOrd::gt))
}

/// HashMap of setup function of the ai.onnx domain version 18(?).
#[rustfmt::skip]
pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut wire_dict: HashMap<_, SetupFnSignature> = HashMap::new();
    wire_dict.insert("Abs", element_wise_numeric!(|el| el.abs()));
    wire_dict.insert("Acos", element_wise_float!(|el| el.acos()));
    wire_dict.insert("Acos", element_wise_float!(|el| el.acos()));
    wire_dict.insert("Acosh", element_wise_float!(|el| el.acosh()));
    wire_dict.insert("Acosh", element_wise_float!(|el| el.acosh()));
    wire_dict.insert("Add", binary_numeric!(|a, b| Ok(a + b)));
    wire_dict.insert("And", make_binary_boolean_fn(|a, b| Ok(a & b)));
    wire_dict.insert("ArgMax", Box::new(argminmax::wire_argmax));
    wire_dict.insert("ArgMin", Box::new(argminmax::wire_argmin));
    wire_dict.insert("Asin", element_wise_float!(|el| el.asin()));
    wire_dict.insert("Asinh", element_wise_float!(|el| el.asinh()));
    wire_dict.insert("Atan", element_wise_float!(|el| el.atan()));
    wire_dict.insert("Atanh", element_wise_float!(|el| el.atanh()));
    wire_dict.insert("Cast", Box::new(wire_cast));
    wire_dict.insert("Ceil", element_wise_float!(|el| el.ceil()));
    wire_dict.insert("Clip", Box::new(wire_clip));
    wire_dict.insert("Concat", Box::new(wire_concat));
    wire_dict.insert("Constant", Box::new(wire_constant));
    wire_dict.insert("Cos", element_wise_float!(|el| el.cos()));
    wire_dict.insert("Cosh", element_wise_float!(|el| el.cosh()));
    wire_dict.insert("Div", binary_numeric!(|a, b| Ok(a / b)));
    wire_dict.insert("Equal", make_equal());
    wire_dict.insert("Exp", element_wise_float!(|el| el.exp()));
    wire_dict.insert("Greater", make_greater());    
    wire_dict.insert("GreaterOrEqual", make_greater_equal());
    wire_dict.insert("Floor", element_wise_float!(|el| el.floor()));
    wire_dict.insert("Identity", Box::new(wire_identity));
    wire_dict.insert("If", Box::new(wire_if));
    wire_dict.insert("IsInf", element_wise_float!(|el| dbg!(el).is_infinite())); // FIXME
    wire_dict.insert("IsNaN", element_wise_float!(|el| el.is_nan()));
    wire_dict.insert("Less", make_less());    
    wire_dict.insert("LessOrEqual", make_less_equal());
    wire_dict.insert("Log", element_wise_float!(|el| el.ln()));
    wire_dict.insert("Mul", binary_numeric!(|a, b| Ok(a * b)));
    wire_dict.insert("Neg", element_wise_numeric!(|a| -a)); 
    wire_dict.insert("Not", Box::new(wire_not));
    wire_dict.insert("Or", make_binary_boolean_fn(|a, b| Ok(a | b)));
    wire_dict.insert("OneHot", Box::new(wire_one_hot));

    wire_dict.insert("ReduceMax", Box::new(reduce::wire_reducemax));
    wire_dict.insert("ReduceMean", Box::new(reduce::wire_reducemean));
    wire_dict.insert("ReduceMin", Box::new(reduce::wire_reducemin));
    wire_dict.insert("ReduceProd", Box::new(reduce::wire_reduceprod));
    wire_dict.insert("ReduceSum", Box::new(reduce::wire_reducesum));

    // TODO: Rust's `round` rounds towards 0.0 for half-way
    // values. ONNX standard rounds to nearest even!
    wire_dict.insert("Round",element_wise_float!(|el| el.round()));
        
    wire_dict.insert("Sigmoid",element_wise_float!(|el| 1. / (1. + (-el).exp())));
    wire_dict.insert("Sin", element_wise_float!(|el| el.sin()));
    wire_dict.insert("Sinh", element_wise_float!(|el| el.sinh()));
    wire_dict.insert("Split", Box::new(wire_split));
    wire_dict.insert("Sqrt",element_wise_float!(|el| el.sqrt()));
    wire_dict.insert("Squeeze", Box::new(wire_squeeze));
    wire_dict.insert("Sub", binary_numeric!(|a, b| Ok(a - b)));
    wire_dict.insert("Tan", element_wise_float!(|el| el.tan()));
    wire_dict.insert("Tanh", element_wise_float!(|el| el.tanh()));
    wire_dict.insert("Unsqueeze", Box::new(wire_unsqueeze));
    wire_dict.insert("Where", Box::new(wire_where));
    wire_dict.insert("Xor", make_binary_boolean_fn(|a, b| Ok(a ^ b)));
    wire_dict
}

/// Create setup function with two boolean inputs
fn make_binary_boolean_fn<Op>(op: Op) -> SetupFnSignature
where
    Op: Fn(ArcArrayD<bool>, ArcArrayD<bool>) -> Result<ArcArrayD<bool>>
        + Send
        + Sync
        + 'static
        + Copy,
{
    Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
        let [a_ch, b_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();
        match (a_ch, b_ch) {
            (EnumReceiver::Bool(a_ch), EnumReceiver::Bool(b_ch)) => {
                wrap_binary_input(a_ch, b_ch, op)
            }
            // Internal error. This should never happen!
            _ => panic!(),
        }
    })
}

fn wire_identity(ctx: OpContext) -> Result<ExecutionStep> {
    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    match a_ch {
        EnumReceiver::Bool(a_ch) => wrap_1_input(a_ch, |a| Ok(a.into())),
        EnumReceiver::F32(a_ch) => wrap_1_input(a_ch, |a| Ok(a.into())),
        EnumReceiver::F64(a_ch) => wrap_1_input(a_ch, |a| Ok(a.into())),
        EnumReceiver::I32(a_ch) => wrap_1_input(a_ch, |a| Ok(a.into())),
        EnumReceiver::I64(a_ch) => wrap_1_input(a_ch, |a| Ok(a.into())),
        EnumReceiver::String(a_ch) => wrap_1_input(a_ch, |a| Ok(a.into())),
    }
}

fn wire_not(ctx: OpContext) -> Result<ExecutionStep> {
    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    match a_ch {
        EnumReceiver::Bool(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| !el).into())),
        _ => bail!("`Not` input must be of type bool."),
    }
}

fn wire_clip(ctx: OpContext) -> Result<ExecutionStep> {
    pub fn match_arm<T>(
        input: Receiver<ArcArrayD<T>>,
        min: Option<Receiver<ArcArrayD<T>>>,
        max: Option<Receiver<ArcArrayD<T>>>,
    ) -> Result<ExecutionStep>
    where
        ArcArrayD<T>: Dispatchable,
        T: Sync + Send + 'static + Copy + PartialOrd,
    {
        let (sdr, rec) = channel();
        let fun: Box<dyn Fn() -> Result<()> + Send> = match (min, max) {
            (Some(min), Some(max)) => Box::new(move || {
                let input = input.recv()?;
                let min = min.recv()?;
                let max = max.recv()?;
                sdr.send(clip(input, Some(min), Some(max))?)?;
                Ok(())
            }),
            (None, Some(max)) => Box::new(move || {
                let input = input.recv()?;
                let max = max.recv()?;
                sdr.send(clip(input, None, Some(max))?)?;
                Ok(())
            }),
            (Some(min), None) => Box::new(move || {
                let input = input.recv()?;
                let min = min.recv()?;
                sdr.send(clip(input, Some(min), None)?)?;
                Ok(())
            }),
            (None, None) => Box::new(move || {
                let input = input.recv()?;
                sdr.send(clip(input, None, None)?)?;
                Ok(())
            }),
        };
        Ok(ExecutionStep {
            fun,
            outputs: vec![rec],
        })
    }

    let mut inputs = ctx.input_channels_with_optional().into_iter();
    let input = inputs
        .next()
        .flatten()
        .ok_or_else(|| anyhow!("Mandatory input 'input' is missing"))?;
    let min = inputs.next().flatten();
    let max = inputs.next().flatten();
    use EnumReceiver::*;
    match (input, min, max) {
        (F32(input), Some(F32(min)), Some(F32(max))) => match_arm(input, Some(min), Some(max)),
        (F32(input), None, Some(F32(max))) => match_arm(input, None, Some(max)),
        (F32(input), Some(F32(min)), None) => match_arm(input, Some(min), None),
        (F32(input), None, None) => match_arm(input, None, None),
        inputs => bail!("Unexpected input for operator 'Clip': {:?}", inputs),
    }
}

fn clip<T>(
    input: ArcArrayD<T>,
    min: Option<ArcArrayD<T>>,
    max: Option<ArcArrayD<T>>,
) -> Result<ArcArrayD<T>>
where
    T: Copy + PartialOrd,
{
    let output = if let Some(min) = min {
        let min = min.to_owned().into_shape([])?.into_scalar();
        input.mapv(|el| if el < min { min } else { el }).into()
    } else {
        input
    };
    let output = if let Some(max) = max {
        let max = max.to_owned().into_shape([])?.into_scalar();
        output.mapv(|el| if el > max { max } else { el }).into()
    } else {
        output
    };
    Ok(output)
}

fn wire_cast(ctx: OpContext) -> Result<ExecutionStep> {
    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    macro_rules! make_output_fallible {
        ($inp:expr, $cast:expr) => {{
            wrap_1_input($inp, |a| {
                let str_vec = a.iter().map($cast).collect::<Result<_, _>>()?;
                let out = Array::from_vec(str_vec).into_shape(a.shape())?;
                Ok(out.into())
            })
        }};
    }
    let to_type = DataType::from_i32(i64::try_from(ctx.attrs.get("to")?)? as i32)?;
    match a_ch {
        EnumReceiver::Bool(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, Ok),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        EnumReceiver::F32(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, Ok),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        EnumReceiver::F64(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, Ok),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        EnumReceiver::I32(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, Ok),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        EnumReceiver::I64(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, Ok),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        EnumReceiver::String(a_ch) => match to_type {
            DataType::Bool => make_output_fallible!(a_ch, |el| el.parse::<bool>()),
            DataType::F32 => make_output_fallible!(a_ch, |el| el.parse::<f32>()),
            DataType::F64 => make_output_fallible!(a_ch, |el| el.parse::<f64>()),
            DataType::I32 => make_output_fallible!(a_ch, |el| el.parse::<i32>()),
            DataType::I64 => make_output_fallible!(a_ch, |el| el.parse::<i64>()),
            DataType::String => wrap_1_input(a_ch, Ok),
        },
    }
}

fn wire_constant(ctx: OpContext) -> Result<ExecutionStep> {
    if ctx.attrs.get("sparse_value").is_ok() {
        unimplemented!("Sparse tensors are not supported.")
    }
    macro_rules! scalar {
        ($val:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();
            let val = ndarray::arr0($val).into_dyn();
            let fun = move || {
                sdr.send(val.clone().into())?;
                Ok(())
            };
            Ok(ExecutionStep {
                fun: Box::new(fun),
                outputs: vec![rec.into()],
            })
        }};
    }
    macro_rules! tensor_attr {
        ($val:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();
            let val = $val.clone();
            let fun = move || {
                sdr.send(val.clone())?;
                Ok(())
            };
            Ok(ExecutionStep {
                fun: Box::new(fun),
                outputs: vec![rec.into()],
            })
        }};
    }

    // Value could be any of the others
    if let Ok(attr) = ctx.attrs.get("value") {
        return match attr {
            Attribute::Bools(val) => tensor_attr!(val),
            Attribute::F32s(val) => tensor_attr!(val),
            Attribute::F64s(val) => tensor_attr!(val),
            Attribute::I32s(val) => tensor_attr!(val),
            Attribute::I64s(val) => tensor_attr!(val),
            Attribute::Strings(val) => tensor_attr!(val),
            Attribute::ScalarF32(val) => scalar!(*val),
            Attribute::ScalarI64(val) => scalar!(*val),
            Attribute::ScalarString(val) => {
                let val = val.to_owned();
                scalar!(val)
            }
            // TODO: Support const bools! Needs refactor of attrs
            _ => todo!("{:?}", attr),
        };
    };

    if let Ok(attr) = ctx.attrs.get("value_float") {
        return scalar!(f32::try_from(attr)?);
    };
    if let Ok(attr) = ctx.attrs.get("value_int") {
        return scalar!(i64::try_from(attr)?);
    };
    if let Ok(attr) = ctx.attrs.get("value_string") {
        return scalar!(String::try_from(attr)?);
    };
    if let Ok(attr) = ctx.attrs.get("value_floats") {
        return tensor_attr!(ArcArrayD::<f32>::try_from(attr)?);
    };
    if let Ok(attr) = ctx.attrs.get("value_ints") {
        return tensor_attr!(ArcArrayD::<i64>::try_from(attr)?);
    };
    if let Ok(attr) = ctx.attrs.get("value_strings") {
        return tensor_attr!(ArcArrayD::<String>::try_from(attr)?);
    };
    Err(anyhow!("`Constant` node did not have expected attributes"))
}

fn wire_split(ctx: OpContext) -> Result<ExecutionStep> {
    let n_outputs = ctx.n_outputs();
    let axis = {
        let axis = ctx.attrs.get_with_default("axis", 0i64)?;
        if axis < 0 {
            todo!("Negative `axis` in split")
        }
        axis
    };

    // `input_channels` creates channels and must only be called once!
    let input_channels = ctx.input_channels();
    if input_channels.len() > 1 {
        bail!("`split` input of `Split` operator is not supported");
    }

    let [a_ch]: [_; 1] = input_channels.try_into().unwrap();
    // Get from attrs
    macro_rules! match_arm {
        ($ty:ty, $inp:expr) => {{
            let (sdrs, recs): (Vec<_>, Vec<_>) = (0..n_outputs)
                .map(|_| ArcArrayD::channel())
                .map(|(sdr, rec)| (sdr, rec.into()))
                .unzip();
            let fun = Box::new(move || {
                let a = $inp.try_recv()?;
                let outs = split(&a, axis, n_outputs)?;
                for (out, sdr) in outs.zip(&sdrs) {
                    sdr.send(out)?;
                }
                Ok(())
            });
            Ok(ExecutionStep { fun, outputs: recs })
        }};
    }

    match a_ch {
        EnumReceiver::F32(a_ch) => match_arm!(f32, a_ch),
        EnumReceiver::F64(a_ch) => match_arm!(f64, a_ch),
        EnumReceiver::I32(a_ch) => match_arm!(i32, a_ch),
        EnumReceiver::I64(a_ch) => match_arm!(i64, a_ch),
        EnumReceiver::String(a_ch) => match_arm!(String, a_ch),
        EnumReceiver::Bool(a_ch) => match_arm!(bool, a_ch),
    }
}

fn split<T>(
    a: &ArcArrayD<T>,
    axis: i64,
    n_outputs: usize,
) -> Result<impl Iterator<Item = ArcArrayD<T>> + '_>
where
    T: Clone,
{
    let axis = axis as usize;
    let len_axis = a.shape()[axis];
    let chunk_size = len_axis / n_outputs;
    Ok(a.axis_chunks_iter(Axis(axis), chunk_size)
        .map(|arr| arr.to_owned().into()))
}

fn wire_unsqueeze(ctx: OpContext) -> Result<ExecutionStep> {
    let [data_ch, axes_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();
    // Function to compute the output shape
    let compute_expanded = |data_shape: &[usize], axes: &[i64]| {
        let mut shape_expanded = vec![0; data_shape.len() + axes.len()];
        let n_out = shape_expanded.len();
        let mut data_iter = data_shape.iter();
        // Convert negative values to positive
        let axes: HashSet<_> = axes
            .iter()
            .map(|el| convert_negative_index(*el, n_out))
            .collect();
        for (i, exp_dim) in shape_expanded.iter_mut().enumerate() {
            if axes.contains(&i) {
                *exp_dim = 1;
            } else {
                *exp_dim = *data_iter.next().unwrap();
            }
        }
        shape_expanded
    };

    macro_rules! match_arm {
        ($data_ch:expr, $axes_ch:expr) => {{
            let (sdr, rec) = ArcArrayD::<_>::channel();
            let fun = Box::new(move || {
                let data = $data_ch.try_recv()?;
                let axes = $axes_ch.try_recv()?;
                // Check that axes is 1D
                let axes = axes.into_dimensionality::<Ix1>()?.to_vec();
                let new_shape = compute_expanded(data.shape(), &axes);
                let out = data.into_shape(new_shape)?;
                sdr.send(out.into_dyn().into())?;
                Ok(())
            });
            Ok(ExecutionStep {
                fun,
                outputs: vec![rec.into()],
            })
        }};
    }
    if let EnumReceiver::I64(axes_ch) = axes_ch {
        match data_ch {
            EnumReceiver::String(data_ch) => match_arm!(data_ch, axes_ch),
            EnumReceiver::I32(data_ch) => match_arm!(data_ch, axes_ch),
            EnumReceiver::I64(data_ch) => match_arm!(data_ch, axes_ch),
            EnumReceiver::F32(data_ch) => match_arm!(data_ch, axes_ch),
            EnumReceiver::F64(data_ch) => match_arm!(data_ch, axes_ch),
            EnumReceiver::Bool(data_ch) => match_arm!(data_ch, axes_ch),
        }
    } else {
        panic!()
    }
}

fn wire_concat(ctx: OpContext) -> Result<ExecutionStep> {
    let axis: i64 = ctx.attrs.get("axis")?.try_into()?;

    macro_rules! match_arm {
        ($input_channels:expr, $variant:tt) => {{
            let input_channels = $input_channels
                .into_iter()
                .map(|ch| match ch {
                    EnumReceiver::$variant(ch) => Ok(ch),
                    _ => Err(anyhow!("All inputs must be of the same type")),
                })
                .collect::<Result<Vec<_>>>()?;

            let (sdr, rec) = ArcArrayD::<_>::channel();
            let fun = Box::new(move || {
                let arrays: Vec<_> = input_channels
                    .iter()
                    .map(|ch| ch.try_recv())
                    .collect::<Result<_, _>>()?;
                let array_views: Vec<_> = arrays.iter().map(|a| a.into()).collect();
                let n_dims = arrays[0].ndim();
                let axis = convert_negative_index(axis, n_dims);
                let out = ndarray::concatenate(Axis(axis), array_views.as_slice())?;
                sdr.send(out.into())?;
                Ok(())
            });
            Ok(ExecutionStep {
                fun,
                outputs: vec![rec.into()],
            })
        }};
    }

    let inputs = ctx.input_channels();
    match &inputs[0] {
        EnumReceiver::F32(_) => match_arm!(inputs, F32),
        EnumReceiver::F64(_) => match_arm!(inputs, F64),
        EnumReceiver::I32(_) => match_arm!(inputs, I32),
        EnumReceiver::I64(_) => match_arm!(inputs, I64),
        EnumReceiver::String(_) => match_arm!(inputs, String),
        EnumReceiver::Bool(_) => match_arm!(inputs, Bool),
    }
}

fn wire_squeeze(ctx: OpContext) -> Result<ExecutionStep> {
    let [data_ch, axes_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();
    let axes_ch = match axes_ch {
        EnumReceiver::I64(ch) => ch,
        _ => bail!("Input `axes` of Squeeze must be of type `tensor(int64)`"),
    };

    macro_rules! match_arm {
        ($ch:expr, $axes_ch:expr) => {{
            let (sdr, rec) = ArcArrayD::<_>::channel();
            let fun = Box::new(move || {
                let data = $ch.try_recv()?;
                let indices = $axes_ch.try_recv()?.into_dimensionality::<Ix1>()?.to_vec();
                // Convert negative indices to positive ones and ensure they are sorted
                let mut indices = indices
                    .iter()
                    .map(|&idx| convert_negative_index(idx, data.ndim()))
                    .map(Axis)
                    .collect::<Vec<_>>();
                indices.sort();

                // Remove axis one after the other starting from the back of `data`.
                let out = indices
                    .iter()
                    .rev()
                    .fold(data, |arr, &axis| arr.remove_axis(axis));
                sdr.send(out)?;
                Ok(())
            });
            Ok(ExecutionStep {
                fun,
                outputs: vec![rec.into()],
            })
        }};
    }
    match data_ch {
        EnumReceiver::I32(ch) => match_arm!(ch, axes_ch),
        EnumReceiver::F32(ch) => match_arm!(ch, axes_ch),
        EnumReceiver::I64(ch) => match_arm!(ch, axes_ch),
        EnumReceiver::F64(ch) => match_arm!(ch, axes_ch),
        EnumReceiver::Bool(ch) => match_arm!(ch, axes_ch),
        EnumReceiver::String(ch) => match_arm!(ch, axes_ch),
    }
}

fn where_<T>(cond: ArcArrayD<bool>, x: ArcArrayD<T>, y: ArcArrayD<T>) -> Result<ArcArrayD<T>>
where
    T: Clone,
{
    let common_shape: IxDyn = {
        let com1: IxDyn =
            co_broadcast(&cond.shape().into_dimension(), &x.shape().into_dimension())?;
        co_broadcast(&com1, &y.shape().into_dimension())?
    };
    Ok(Zip::from(cond.view().broadcast(common_shape).unwrap())
        .and_broadcast(x.view())
        .and_broadcast(y.view())
        .map_collect(|&cond, x, y| if cond { x.clone() } else { y.clone() })
        .into())
}

fn wire_where(ctx: OpContext) -> Result<ExecutionStep> {
    let [cond_ch, x_ch, y_ch]: [_; 3] = ctx.input_channels().try_into().unwrap();

    pub fn match_arm<T>(
        cond_ch: Receiver<ArcArrayD<bool>>,
        x_ch: Receiver<ArcArrayD<T>>,
        y_ch: Receiver<ArcArrayD<T>>,
    ) -> Result<ExecutionStep>
    where
        ArcArrayD<T>: Dispatchable,
        T: Sync + Send + 'static + Clone,
    {
        let (sdr, rec) = channel();
        let fun = Box::new(move || {
            let (cond, x, y) = (cond_ch.recv()?, x_ch.recv()?, y_ch.recv()?);
            sdr.send(where_(cond, x, y)?)?;
            Ok(())
        });
        Ok(ExecutionStep {
            fun,
            outputs: vec![rec],
        })
    }

    // Condition must be of type bool
    let cond_ch = if let EnumReceiver::Bool(cond_ch) = cond_ch {
        cond_ch
    } else {
        bail!("Conditional tensor must be of type `bool`.")
    };

    match (x_ch, y_ch) {
        (EnumReceiver::F32(x_ch), EnumReceiver::F32(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (EnumReceiver::F64(x_ch), EnumReceiver::F64(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (EnumReceiver::I32(x_ch), EnumReceiver::I32(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (EnumReceiver::I64(x_ch), EnumReceiver::I64(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (EnumReceiver::Bool(x_ch), EnumReceiver::Bool(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (EnumReceiver::String(x_ch), EnumReceiver::String(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        _ => bail!("`X` and `Y` must be of identical type."),
    }
}

fn wire_if(ctx: OpContext) -> Result<ExecutionStep> {
    // Explicit node input
    let [cond_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    // Condition must be of type bool
    let cond_ch = if let EnumReceiver::Bool(cond_ch) = cond_ch {
        cond_ch
    } else {
        bail!("Conditional tensor must be of type `bool`.")
    };

    // Create the sessions for the `then` and `else` branch.
    let make_attr_session = |attr_name| {
        let g: GraphProto = ctx.attrs.get(attr_name)?.try_into()?;
        Session::from_nested_graph(&g, ctx.opset, ctx.scope)
    };
    let then_session = make_attr_session("then_branch")?;
    let else_session = make_attr_session("else_branch")?;

    // Get input channels for implicit inputs. We always pull all channels!
    let cumulated_input_names = HashSet::<&str>::from_iter(
        then_session
            .input_names()
            .into_iter()
            .chain(else_session.input_names().into_iter()),
    );

    let graph_input_channels: HashMap<_, _> = cumulated_input_names
        .iter()
        .map(|name| (name.to_string(), ctx.subscribe(name)))
        .collect();

    // Create output channels by "selecting" the outputs of either of
    // the two branches.
    assert_eq!(then_session.outputs.len(), else_session.outputs.len());
    assert!(then_session
        .output_dtypes()
        .iter()
        .zip(&else_session.output_dtypes())
        .all(|((_, dtype_left), (_, dtype_right))| dtype_left == dtype_right));

    let mut out_dispatchers = vec![];
    let mut out_senders = vec![];
    // Only using then outputs types since they should be identical
    for (_out_name, out_dtype) in &then_session.output_dtypes() {
        let (out_sdr, out_disp) = value_channel(out_dtype);
        out_senders.push(out_sdr);
        out_dispatchers.push(out_disp);
    }
    let fun = move || {
        let cond = {
            let conds = cond_ch.recv().unwrap();
            if conds.len() != 1 {
                return Err(anyhow!("`Cond` input of `If` must be a scalar."));
            }
            conds.to_owned().into_shape([])?.into_scalar()
        };

        let graph_inputs = graph_input_channels
            .iter()
            .map(|(k, ch)| (k.clone(), ch.recv().unwrap()))
            .collect();
        // If-graphs do not have explicit inputs; only the implicit ones
        if cond {
            // The outputs are defined by order, not name
            let mut result = then_session.run(&graph_inputs)?;
            for ((name, _), sender) in then_session.outputs.iter().zip(&out_senders) {
                sender.send(result.remove(name).unwrap()).unwrap();
            }
        } else {
            // The outputs are defined by order, not name
            let mut result = else_session.run(&graph_inputs)?;
            for ((name, _), sender) in else_session.outputs.iter().zip(&out_senders) {
                sender.send(result.remove(name).unwrap()).unwrap();
            }
        };
        Ok(())
    };
    Ok(ExecutionStep {
        fun: Box::new(fun),
        outputs: out_dispatchers,
    })
}

mod argminmax {
    use super::*;

    macro_rules! match_arm_argminmax {
        ($fn:expr, $ch:expr, $axis:expr, $keepdims:expr, $use_last:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();
            let fun = move || {
                let data = $ch.recv()?;
                let out = $fn(&data, $axis, $keepdims, $use_last)?;
                sdr.send(out)?;
                Ok(())
            };
            Ok(ExecutionStep {
                fun: Box::new(fun),
                outputs: vec![rec],
            })
        }};
    }

    pub fn wire_argmax(ctx: OpContext) -> Result<ExecutionStep> {
        let axis = ctx.attrs.get_with_default("axis", 0i64)?;
        let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
        let use_last = ctx.attrs.get_with_default("select_last_index", 0i64)? != 0;

        match ctx.input_channels().try_into()? {
            HomogeneousNumericFixed::F32([ch]) => {
                match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
            }
            HomogeneousNumericFixed::F64([ch]) => {
                match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
            }
            HomogeneousNumericFixed::I32([ch]) => {
                match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
            }
            HomogeneousNumericFixed::I64([ch]) => {
                match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
            }
        }
    }

    pub fn wire_argmin(ctx: OpContext) -> Result<ExecutionStep> {
        let axis = ctx.attrs.get_with_default("axis", 0i64)?;
        let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
        let use_last = ctx.attrs.get_with_default("select_last_index", 0i64)? != 0;

        match ctx.input_channels().try_into()? {
            HomogeneousNumericFixed::F32([ch]) => {
                match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
            }
            HomogeneousNumericFixed::F64([ch]) => {
                match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
            }
            HomogeneousNumericFixed::I32([ch]) => {
                match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
            }
            HomogeneousNumericFixed::I64([ch]) => {
                match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
            }
        }
    }

    fn argmax<T>(
        data: &ArcArrayD<T>,
        axis: i64,
        keep_dim: bool,
        use_last: bool,
    ) -> Result<ArcArrayD<i64>>
    where
        T: Copy + num_traits::Bounded + ::std::cmp::PartialOrd,
    {
        let start_acc = (0, T::min_value());
        let fold_fn = |acc: (usize, T), (idx, v)| {
            if v > acc.1 || (use_last && acc.1 == v) {
                (idx, v)
            } else {
                acc
            }
        };
        Ok(fold_axis(data, axis, keep_dim, start_acc, &fold_fn)?
            .mapv(|(idx, _v)| idx as i64)
            .into())
    }

    fn argmin<T>(
        data: &ArcArrayD<T>,
        axis: i64,
        keep_dim: bool,
        use_last: bool,
    ) -> Result<ArcArrayD<i64>>
    where
        T: Copy + num_traits::Bounded + ::std::cmp::PartialOrd,
    {
        let start_acc = (0, T::max_value());
        let fold_fn = |acc: (usize, T), (idx, v)| {
            if v < acc.1 || (use_last && acc.1 == v) {
                (idx, v)
            } else {
                acc
            }
        };
        Ok(fold_axis(data, axis, keep_dim, start_acc, &fold_fn)?
            .mapv(|(idx, _v)| idx as i64)
            .into())
    }

    fn fold_axis<T, Acc, Fold>(
        data: &ArcArrayD<T>,
        axis: i64,
        keep_dim: bool,
        init_fold: Acc,
        fold_fn: &Fold,
    ) -> Result<ndarray::ArrayD<Acc>>
    where
        T: Copy + num_traits::Bounded + ::std::cmp::PartialOrd,
        Acc: Copy,
        Fold: Fn(Acc, (usize, T)) -> Acc,
    {
        let axis = convert_negative_index(axis, data.shape().len());
        let rank = data.shape().len();
        if axis >= rank {
            bail!(
                "Axis `{}` is out of bounds for input of rank {}",
                axis,
                rank
            );
        }
        let mut out = data.map_axis(Axis(axis), |lane| {
            lane.iter().copied().enumerate().fold(init_fold, fold_fn)
        });
        if keep_dim {
            out.insert_axis_inplace(Axis(axis));
        }
        Ok(out)
    }
}

mod reduce {
    use num_traits::Bounded;
    use std::cmp::PartialOrd;
    use std::ops::{Add, Mul};

    use super::*;

    struct CommonArgs {
        keepdims: bool,
        noop_empty: bool,
        axes_ch: Option<Receiver<ArcArrayD<i64>>>,
    }

    fn setup(ctx: OpContext) -> Result<(EnumReceiver, CommonArgs)> {
        let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
        let noop_empty = ctx.attrs.get_with_default("noop_with_empty_axes", 0i64)? != 0;

        let mut inputs = ctx.input_channels().into_iter();
        let data_ch = inputs
            .next()
            .ok_or_else(|| anyhow!("Input `data` is missing"))?;
        let axes_ch = match inputs.next() {
            Some(EnumReceiver::I64(ch)) => Some(ch),
            None => None,
            Some(_) => bail!("Input `axes` must be of type `int64`"),
        };
        if inputs.len() != 0 {
            bail!("Unexpected inputs");
        }

        Ok((
            data_ch,
            CommonArgs {
                keepdims,
                noop_empty,
                axes_ch,
            },
        ))
    }

    macro_rules! match_arm {
        ($ty:ty, $data_ch:expr, $common_args:expr, $acc:expr, $fold_fn:expr) => {
            match_arm!($ty, $data_ch, $common_args, $acc, $fold_fn, false)
        };
        ($ty:ty, $data_ch:expr, $common_args:expr, $acc:expr, $fold_fn:expr, $norm:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();
            let CommonArgs {
                keepdims,
                noop_empty,
                axes_ch,
            } = $common_args;
            let fun = Box::new(move || {
                let data = $data_ch.recv()?;
                let axes = match &axes_ch {
                    Some(ch) => {
                        let arr = ch.recv()?;
                        if arr.shape().len() != 1 {
                            bail!("Input `axes` must be of rank 1");
                        }
                        let len = arr.len();
                        arr.into_shape((len,))?.to_vec()
                    }
                    None => vec![],
                };
                let n_initial = data.len();
                let reduced = reduce(data, &axes, keepdims, noop_empty, $acc, $fold_fn)?;
                // Normalized each bin by the number of bins that was reduced into it.
                if $norm {
                    let n_reduced = reduced.len();
                    let scaling = n_initial / n_reduced;
                    return sdr.send(reduced.mapv(|v| v / scaling as $ty).into());
                }

                sdr.send(reduced)?;
                Ok(())
            });
            ExecutionStep {
                fun,
                outputs: vec![rec],
            }
        }};
    }

    pub fn wire_reducemax(ctx: OpContext) -> Result<ExecutionStep> {
        let (data_ch, common_args) = setup(ctx)?;
        let data_ch = HomogeneousNumericFixed::try_from(vec![data_ch])?;
        Ok(match data_ch {
            HomogeneousNumericFixed::F32([data_ch]) => {
                match_arm!(f32, data_ch, common_args, -f32::INFINITY, fold_max)
            }
            HomogeneousNumericFixed::F64([data_ch]) => {
                match_arm!(f64, data_ch, common_args, -f64::INFINITY, fold_max)
            }
            HomogeneousNumericFixed::I64([data_ch]) => {
                match_arm!(i64, data_ch, common_args, i64::MIN, fold_max)
            }
            HomogeneousNumericFixed::I32([data_ch]) => {
                match_arm!(i32, data_ch, common_args, i32::MIN, fold_max)
            }
        })
    }

    pub fn wire_reducemean(ctx: OpContext) -> Result<ExecutionStep> {
        let (data_ch, common_args) = setup(ctx)?;
        let data_ch = HomogeneousNumericFixed::try_from(vec![data_ch])?;
        Ok(match data_ch {
            HomogeneousNumericFixed::F32([data_ch]) => {
                match_arm!(f32, data_ch, common_args, 0.0, fold_sum, true)
            }
            HomogeneousNumericFixed::F64([data_ch]) => {
                match_arm!(f64, data_ch, common_args, 0.0, fold_sum, true)
            }
            HomogeneousNumericFixed::I32([data_ch]) => {
                match_arm!(i32, data_ch, common_args, 0, fold_sum, true)
            }
            HomogeneousNumericFixed::I64([data_ch]) => {
                match_arm!(i64, data_ch, common_args, 0, fold_sum, true)
            }
        })
    }

    pub fn wire_reducemin(ctx: OpContext) -> Result<ExecutionStep> {
        let (data_ch, common_args) = setup(ctx)?;
        let data_ch = HomogeneousNumericFixed::try_from(vec![data_ch])?;
        Ok(match data_ch {
            HomogeneousNumericFixed::F32([data_ch]) => {
                match_arm!(f32, data_ch, common_args, f32::INFINITY, fold_min)
            }
            HomogeneousNumericFixed::F64([data_ch]) => {
                match_arm!(f64, data_ch, common_args, f64::INFINITY, fold_min)
            }
            HomogeneousNumericFixed::I64([data_ch]) => {
                match_arm!(i64, data_ch, common_args, i64::MAX, fold_min)
            }
            HomogeneousNumericFixed::I32([data_ch]) => {
                match_arm!(i32, data_ch, common_args, i32::MAX, fold_min)
            }
        })
    }

    pub fn wire_reduceprod(ctx: OpContext) -> Result<ExecutionStep> {
        let (data_ch, common_args) = setup(ctx)?;
        let data_ch = HomogeneousNumericFixed::try_from(vec![data_ch])?;
        Ok(match data_ch {
            HomogeneousNumericFixed::F32([data_ch]) => {
                match_arm!(f32, data_ch, common_args, 1.0, fold_mul)
            }
            HomogeneousNumericFixed::F64([data_ch]) => {
                match_arm!(f64, data_ch, common_args, 1.0, fold_mul)
            }
            HomogeneousNumericFixed::I32([data_ch]) => {
                match_arm!(i32, data_ch, common_args, 1, fold_mul)
            }
            HomogeneousNumericFixed::I64([data_ch]) => {
                match_arm!(i64, data_ch, common_args, 1, fold_mul)
            }
        })
    }

    pub fn wire_reducesum(ctx: OpContext) -> Result<ExecutionStep> {
        let (data_ch, common_args) = setup(ctx)?;
        let data_ch = HomogeneousNumericFixed::try_from(vec![data_ch])?;
        Ok(match data_ch {
            HomogeneousNumericFixed::F32([data_ch]) => {
                match_arm!(f32, data_ch, common_args, 0.0, fold_sum)
            }
            HomogeneousNumericFixed::F64([data_ch]) => {
                match_arm!(f64, data_ch, common_args, 0.0, fold_sum)
            }
            HomogeneousNumericFixed::I32([data_ch]) => {
                match_arm!(i32, data_ch, common_args, 0, fold_sum)
            }
            HomogeneousNumericFixed::I64([data_ch]) => {
                match_arm!(i64, data_ch, common_args, 0, fold_sum)
            }
        })
    }

    fn fold_max<T: PartialOrd + Copy>(acc: &T, el: &T) -> T {
        if el > acc {
            *el
        } else {
            *acc
        }
    }

    fn fold_min<T: PartialOrd + Copy>(acc: &T, el: &T) -> T {
        if el < acc {
            *el
        } else {
            *acc
        }
    }

    fn fold_mul<T: Copy + Mul<Output = T>>(acc: &T, el: &T) -> T {
        *acc * *el
    }

    fn fold_sum<T: Copy + Add<Output = T>>(acc: &T, el: &T) -> T {
        *acc + *el
    }

    fn reduce<T, F>(
        arr: ArcArrayD<T>,
        axes: &[i64],
        keepdims: bool,
        empty_is_noop: bool,
        fold_init: T,
        fold_fn: F,
    ) -> Result<ArcArrayD<T>>
    where
        T: Copy + Bounded + PartialOrd,
        F: Fn(&T, &T) -> T,
    {
        let initial_shape = arr.shape().to_vec();
        let rank = initial_shape.len();

        // Deal with special case around empty `arr`
        let axes = if axes.is_empty() {
            if empty_is_noop {
                return Ok(arr);
            } else {
                (0..arr.shape().len()).collect()
            }
        } else {
            axes.iter()
                .map(|idx| convert_negative_index(*idx, rank))
                .collect::<HashSet<_>>()
        };
        if let Some(max_axis_input) = axes.iter().max() {
            if *max_axis_input >= rank {
                bail!(
                    "Input `axes={}` is incompatible tensor of `rank={}`",
                    *max_axis_input,
                    rank
                );
            }
        }
        // Reorder the axes to put the once that are being reduced at the end
        let regrouped_axes: Vec<_> = (0..rank)
            .filter(|axis| !axes.contains(axis))
            .chain(sorted(axes.clone()))
            .collect();

        let reduce_mask: Vec<_> = (0..rank).map(|axis| axes.contains(&axis)).collect();

        // Number of elements that are being reduced
        let len_reduce_axis = initial_shape
            .iter()
            .zip(&reduce_mask)
            .filter(|(_, &red_this)| red_this)
            .fold(1, |acc, el| acc * el.0);

        let new_shape: Vec<_> = initial_shape
            .iter()
            .zip(&reduce_mask)
            .filter_map(|(dim_len, &do_red)| if do_red { None } else { Some(dim_len) })
            .chain(&[len_reduce_axis])
            .copied()
            .collect();
        let last_axis = Axis(new_shape.len() - 1);
        let reduced = arr
            .permuted_axes(regrouped_axes)
            .as_standard_layout()
            .into_shape(new_shape)
            .expect("Invalid shape.")
            .fold_axis(last_axis, fold_init, fold_fn);

        let reduced = if keepdims {
            let out_shape: Vec<_> = initial_shape
                .iter()
                .enumerate()
                .map(|(i, &el)| if axes.contains(&i) { 1 } else { el })
                .collect();
            reduced.into_shape(out_shape)?
        } else {
            reduced
        };
        Ok(reduced.into())
    }
}

fn wire_one_hot(ctx: OpContext) -> Result<ExecutionStep> {
    let [indices, depth, values]: [_; 3] = ctx.input_channels().try_into().unwrap();
    let axis = ctx.attrs.get_optional("axis")?;

    // Cast other types to i64 (as stated in the standard...)
    let map_err = |e| format!("Failed to receive value: {:?}", e);
    let recv_depth: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send> = match depth {
        EnumReceiver::F32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        EnumReceiver::I32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        EnumReceiver::F64(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        EnumReceiver::I64(ch) => Box::new(move || Ok(ch.recv().map_err(map_err)?)),
        _ => bail!("Unsupported type for 'depth' input."),
    };
    let recv_indices: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send> = match indices {
        EnumReceiver::F32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        EnumReceiver::I32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        EnumReceiver::F64(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        EnumReceiver::I64(ch) => Box::new(move || Ok(ch.recv().map_err(map_err)?)),
        _ => bail!("Unsupported type for 'depth' input."),
    };

    pub fn match_arm<Vals>(
        recv_indices: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send>,
        recv_depth: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send>,
        values: Receiver<ArcArrayD<Vals>>,
        axis: Option<i64>,
    ) -> Result<ExecutionStep>
    where
        ArcArrayD<Vals>: Dispatchable,
        Vals: 'static + Copy + std::fmt::Debug + Sync + Send,
    {
        let (sdr, rec) = channel();
        let fun = Box::new(move || {
            let indices = recv_indices().map_err(|e| anyhow!("{}", e))?;
            let depth = recv_depth().map_err(|e| anyhow!("{}", e))?;
            let values = values.recv().map_err(|e| anyhow!("{}", e))?;
            let [off, on]: [Vals; 2] = values.into_iter().collect::<Vec<_>>().try_into().unwrap();
            let [depth]: [_; 1] = depth.into_iter().collect::<Vec<_>>().try_into().unwrap();
            let res: ArcArrayD<Vals> = one_hot(indices, depth.try_into()?, off, on, axis)?;
            sdr.send(res)?;
            Ok(())
        });
        Ok(ExecutionStep {
            fun,
            outputs: vec![rec],
        })
    }

    match values {
        EnumReceiver::F32(values) => match_arm(recv_indices, recv_depth, values, axis),
        EnumReceiver::F64(values) => match_arm(recv_indices, recv_depth, values, axis),
        EnumReceiver::I32(values) => match_arm(recv_indices, recv_depth, values, axis),
        EnumReceiver::I64(values) => match_arm(recv_indices, recv_depth, values, axis),
        _ => bail!("Values data type is not implemented: '{:?}'", values),
    }
}

pub fn one_hot<Vals>(
    indices: ArcArrayD<i64>,
    depth: usize,
    off_value: Vals,
    on_value: Vals,
    axis: Option<i64>,
) -> Result<ArcArrayD<Vals>>
where
    Vals: Copy + std::fmt::Debug,
{
    // Compute output shape
    let input_rank = indices.ndim();
    let output_rank = input_rank + 1;
    let axis = axis.unwrap_or_else(|| -1);
    let axis_pos = convert_negative_index(axis, output_rank);
    if axis_pos > input_rank {
        bail!(
            "Axis must be compatible with input rank ('{}'). Found '{}'.",
            input_rank,
            axis
        );
    }
    let dims = indices.shape().iter();

    // Put new axis at the end for now
    let out_shape: Vec<_> = dims
        // .take(axis_pos)
        .chain([depth as usize].iter())
        // .chain(dims)
        .cloned()
        .collect();

    let flat_indices = indices.iter();
    let mut flatish_out = Array2::from_elem((indices.len(), depth), off_value);
    for (idx, mut out_slice) in flat_indices.zip(flatish_out.axis_iter_mut(Axis(0))) {
        let val = convert_negative_index(*idx, depth);
        if let Some(el) = out_slice.get_mut([val]) {
            *el = on_value;
        }
    }

    // Reshape and reanrange axes if necessary
    if axis != -1 {
        let output_rank = input_rank + 1;
        let mut tmp = flatish_out
            .into_shape(out_shape)?
            .insert_axis(Axis(axis_pos));
        tmp.swap_axes(axis_pos, output_rank);
        let tmp = tmp.remove_axis(Axis(output_rank));
        Ok(tmp.into_dyn().into())
    } else {
        Ok(flatish_out.into_shape(out_shape)?.into_dyn().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_where() {
        let cond = array![[true, false], [true, true]].into_dyn().into();
        let x = array![[1, 2], [3, 4]].into_dyn().into();
        let y = array![[9, 8], [7, 6]].into_dyn().into();
        let expected: ArcArrayD<_> = array![[1, 8], [3, 4]].into_dyn().into();
        assert_eq!(where_(cond, x, y).unwrap(), expected);
    }
}
