use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Receiver;

use anyhow::{anyhow, bail, Result};
use ndarray::{Array, Axis, IntoDimension, Ix1, IxDyn, Zip};
use onnx_protos::onnx_proto3::GraphProto;

use crate::attribute::Attribute;
use crate::session::{OpContext, SetupFnSignature, SetupResult};
use crate::utils::{
    co_broadcast, convert_negative_index, wrap_1_input, wrap_binary_input,
    zip_map_with_co_broadcast,
};
use crate::{channel, value_channel, ArcArrayD, DataType, Dispatchable, EnumReceiver, Session};

macro_rules! element_wise_numeric {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> SetupResult {
            let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
            match a_ch {
                EnumReceiver::F32(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                EnumReceiver::F64(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                EnumReceiver::I32(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                EnumReceiver::I64(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                // Internal error. This should never happen.
                _ => panic!(),
            }
        })
    };
}

/// Make setup function for element wise operations with floating point input
macro_rules! element_wise_float {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> SetupResult {
            let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
            match a_ch {
                EnumReceiver::F32(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                EnumReceiver::F64(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                // Internal error. This should never happen.
                _ => panic!(),
            }
        })
    };
}

/// Setup functions with two numeric inputs
macro_rules! binary_numeric {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> SetupResult {
            let [a_ch, b_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();
            match (a_ch, b_ch) {
                (EnumReceiver::I32(a_ch), EnumReceiver::I32(b_ch)) => {
                    wrap_binary_input(a_ch, b_ch, $fun)
                }
                (EnumReceiver::F32(a_ch), EnumReceiver::F32(b_ch)) => {
                    wrap_binary_input(a_ch, b_ch, $fun)
                }
                (EnumReceiver::I64(a_ch), EnumReceiver::I64(b_ch)) => {
                    wrap_binary_input(a_ch, b_ch, $fun)
                }
                (EnumReceiver::F64(a_ch), EnumReceiver::F64(b_ch)) => {
                    wrap_binary_input(a_ch, b_ch, $fun)
                }
                // Internal error. This should never happen!
                _ => panic!(),
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
    wire_dict.insert("ArgMax", Box::new(wire_argmax));
    wire_dict.insert("ArgMin", Box::new(wire_argmin));
    wire_dict.insert("Asin", element_wise_float!(|el| el.asin()));
    wire_dict.insert("Asinh", element_wise_float!(|el| el.asinh()));
    wire_dict.insert("Atan", element_wise_float!(|el| el.atan()));
    wire_dict.insert("Atanh", element_wise_float!(|el| el.atanh()));
    wire_dict.insert("Cast", Box::new(wire_cast));
    wire_dict.insert("Ceil", element_wise_float!(|el| el.ceil()));
    wire_dict.insert("Concat", Box::new(wire_concat));
    wire_dict.insert("Constant", Box::new(wire_constant));
    wire_dict.insert("Cos", element_wise_float!(|el| el.cos()));
    wire_dict.insert("Cosh", element_wise_float!(|el| el.cosh()));
    wire_dict.insert("Div", binary_numeric!(|a, b| Ok(a / b)));
    wire_dict.insert("Equal", make_equal());
    wire_dict.insert("Exp", element_wise_float!(|el| el.exp()));
    wire_dict.insert("Floor", element_wise_float!(|el| el.floor()));
    wire_dict.insert("Identity", element_wise_float!(|el| el));
    wire_dict.insert("If", Box::new(wire_if));
    wire_dict.insert("IsInf", element_wise_float!(|el| dbg!(el).is_infinite())); // FIXME
    wire_dict.insert("IsNaN", element_wise_float!(|el| el.is_nan()));
    wire_dict.insert("LessOrEqual", make_less_equal());
    wire_dict.insert("Log", element_wise_float!(|el| el.ln()));
    wire_dict.insert("Mul", binary_numeric!(|a, b| Ok(a * b)));
    wire_dict.insert("Not", Box::new(wire_not));
    wire_dict.insert("Or", make_binary_boolean_fn(|a, b| Ok(a | b)));
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
    Box::new(move |ctx: OpContext| -> SetupResult {
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

fn wire_not(ctx: OpContext) -> SetupResult {
    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    match a_ch {
        EnumReceiver::Bool(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| !el).into())),
        _ => bail!("`Not` input must be of type bool."),
    }
}

fn wire_cast(ctx: OpContext) -> SetupResult {
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

fn wire_constant(ctx: OpContext) -> SetupResult {
    if ctx.attrs.get("sparse_value").is_ok() {
        unimplemented!("Sparse tensors are not supported.")
    }
    macro_rules! scalar {
        ($val:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();
            let val = ndarray::arr0($val).into_dyn();
            Ok((
                Box::new(move || {
                    sdr.send(val.clone().into())?;
                    Ok(())
                }),
                vec![rec.into()],
            ))
        }};
    }
    macro_rules! tensor_attr {
        ($val:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();
            let val = $val.clone();
            Ok((
                Box::new(move || {
                    sdr.send(val.clone())?;
                    Ok(())
                }),
                vec![rec.into()],
            ))
        }};
    }

    // Value could be any of the others
    if let Ok(attr) = ctx.attrs.get("value") {
        return match attr {
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

fn wire_split(ctx: OpContext) -> SetupResult {
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
            Ok((fun, recs))
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

fn wire_unsqueeze(ctx: OpContext) -> SetupResult {
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
            Ok((fun, vec![rec.into()]))
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

fn wire_concat(ctx: OpContext) -> SetupResult {
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
            Ok((fun, vec![rec.into()]))
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

fn wire_squeeze(ctx: OpContext) -> SetupResult {
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
            Ok((fun, vec![rec.into()]))
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

fn wire_where(ctx: OpContext) -> SetupResult {
    let [cond_ch, x_ch, y_ch]: [_; 3] = ctx.input_channels().try_into().unwrap();

    pub fn match_arm<T>(
        cond_ch: Receiver<ArcArrayD<bool>>,
        x_ch: Receiver<ArcArrayD<T>>,
        y_ch: Receiver<ArcArrayD<T>>,
    ) -> SetupResult
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
        Ok((fun, vec![rec]))
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

fn wire_if(ctx: OpContext) -> SetupResult {
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

    Ok((
        Box::new(move || {
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
        }),
        out_dispatchers,
    ))
}

fn wire_argmax(ctx: OpContext) -> SetupResult {
    let axis = ctx.attrs.get_with_default("axis", 0i64)?;
    let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
    let select_last_index = ctx.attrs.get_with_default("select_last_index", 0i64)? != 0;

    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();

    macro_rules! match_arm {
        ($ch:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();

            Ok((
                Box::new(move || {
                    let data = $ch.recv()?;
                    let out = argmax(&data, axis, keepdims, select_last_index)?;
                    sdr.send(out)?;
                    Ok(())
                }),
                vec![rec],
            ))
        }};
    }

    match a_ch {
        EnumReceiver::F32(ch) => match_arm!(ch),
        EnumReceiver::F64(ch) => match_arm!(ch),
        EnumReceiver::I32(ch) => match_arm!(ch),
        EnumReceiver::I64(ch) => match_arm!(ch),
        _ => bail!("Unimplemented type for op `ArgMax`."),
    }
}

fn wire_argmin(ctx: OpContext) -> SetupResult {
    let axis = ctx.attrs.get_with_default("axis", 0i64)?;
    let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
    let select_last_index = ctx.attrs.get_with_default("select_last_index", 0i64)? != 0;

    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();

    macro_rules! match_arm {
        ($ch:expr) => {{
            let (sdr, rec) = ArcArrayD::channel();

            Ok((
                Box::new(move || {
                    let data = $ch.recv()?;
                    let out = argmin(&data, axis, keepdims, select_last_index)?;
                    sdr.send(out)?;
                    Ok(())
                }),
                vec![rec],
            ))
        }};
    }

    match a_ch {
        EnumReceiver::F32(ch) => match_arm!(ch),
        EnumReceiver::F64(ch) => match_arm!(ch),
        EnumReceiver::I32(ch) => match_arm!(ch),
        EnumReceiver::I64(ch) => match_arm!(ch),
        _ => bail!("Unimplemented type for op `ArgMax`."),
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
