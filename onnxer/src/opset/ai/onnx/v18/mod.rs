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
    channel, value_channel, ArcArrayD, DataType, Dispatchable, HomogeneousNumericFixed, Session,
    TaggedReceiver,
};

mod arg_min_max;
mod clip;
mod one_hot;
mod reduce;
mod reshape;
mod simple_element_wise;
mod where_;

/// Setup functions with two numeric inputs
macro_rules! bin_numeric {
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

macro_rules! bin_numeric_with_co_broadcast {
    ($fun:expr) => {
        bin_numeric!(|a, b| zip_map_with_co_broadcast(a, b, $fun))
    };
}

/// HashMap of setup function of the ai.onnx domain version 18(?).
#[rustfmt::skip]
pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut d = simple_element_wise::get_setup_dict();
    d.insert("Add", bin_numeric!(|a, b| Ok(a + b)));
    d.insert("And", binary_boolean_factory(|a, b| Ok(a & b)));
    d.insert("ArgMax", Box::new(arg_min_max::setup_argmax));
    d.insert("ArgMin", Box::new(arg_min_max::setup_argmin));
    d.insert("Cast", Box::new(setup_cast));
    d.insert("Clip", Box::new(clip::setup_clip));
    d.insert("Concat", Box::new(setup_concat));
    d.insert("Constant", Box::new(setup_constant));
    d.insert("Div", bin_numeric!(|a, b| Ok(a / b)));
    d.insert("Equal", bin_numeric_with_co_broadcast!(PartialEq::eq));
    d.insert("Greater", bin_numeric_with_co_broadcast!(PartialOrd::gt));
    d.insert("GreaterOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::ge));
    d.insert("If", Box::new(setup_if));
    d.insert("Less", bin_numeric_with_co_broadcast!(PartialOrd::lt));
    d.insert("LessOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::le));
    d.insert("Mul", bin_numeric!(|a, b| Ok(a * b)));
    d.insert("OneHot", Box::new(one_hot::setup_one_hot));
    d.insert("Or", binary_boolean_factory(|a, b| Ok(a | b)));
    d.insert("ReduceMax", Box::new(reduce::setup_reducemax));
    d.insert("ReduceMean", Box::new(reduce::setup_reducemean));
    d.insert("ReduceMin", Box::new(reduce::setup_reducemin));
    d.insert("ReduceProd", Box::new(reduce::setup_reduceprod));
    d.insert("ReduceSum", Box::new(reduce::setup_reducesum));
    d.insert("Reshape", Box::new(reshape::setup_reshape));
    d.insert("Split", Box::new(setup_split));
    d.insert("Squeeze", Box::new(setup_squeeze));
    d.insert("Sub", bin_numeric!(|a, b| Ok(a - b)));
    d.insert("Unsqueeze", Box::new(setup_unsqueeze));
    d.insert("Where", Box::new(where_::setup_where));
    d.insert("Xor", binary_boolean_factory(|a, b| Ok(a ^ b)));
    d
}

/// Create setup function with two boolean inputs
fn binary_boolean_factory<Op>(op: Op) -> SetupFnSignature
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
            (TaggedReceiver::Bool(a_ch), TaggedReceiver::Bool(b_ch)) => {
                wrap_binary_input(a_ch, b_ch, op)
            }
            // Internal error. This should never happen!
            _ => panic!(),
        }
    })
}

fn setup_cast(ctx: OpContext) -> Result<ExecutionStep> {
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
        TaggedReceiver::Bool(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, Ok),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u8 as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::F32(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, Ok),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::F64(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, Ok),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::I32(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, Ok),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::I64(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, Ok),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::String(a_ch) => match to_type {
            DataType::Bool => make_output_fallible!(a_ch, |el| el.parse::<bool>()),
            DataType::F32 => make_output_fallible!(a_ch, |el| el.parse::<f32>()),
            DataType::F64 => make_output_fallible!(a_ch, |el| el.parse::<f64>()),
            DataType::I32 => make_output_fallible!(a_ch, |el| el.parse::<i32>()),
            DataType::I64 => make_output_fallible!(a_ch, |el| el.parse::<i64>()),
            DataType::String => wrap_1_input(a_ch, Ok),
        },
    }
}

fn setup_constant(ctx: OpContext) -> Result<ExecutionStep> {
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

fn setup_split(ctx: OpContext) -> Result<ExecutionStep> {
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
        TaggedReceiver::F32(a_ch) => match_arm!(f32, a_ch),
        TaggedReceiver::F64(a_ch) => match_arm!(f64, a_ch),
        TaggedReceiver::I32(a_ch) => match_arm!(i32, a_ch),
        TaggedReceiver::I64(a_ch) => match_arm!(i64, a_ch),
        TaggedReceiver::String(a_ch) => match_arm!(String, a_ch),
        TaggedReceiver::Bool(a_ch) => match_arm!(bool, a_ch),
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

fn setup_unsqueeze(ctx: OpContext) -> Result<ExecutionStep> {
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
    if let TaggedReceiver::I64(axes_ch) = axes_ch {
        match data_ch {
            TaggedReceiver::String(data_ch) => match_arm!(data_ch, axes_ch),
            TaggedReceiver::I32(data_ch) => match_arm!(data_ch, axes_ch),
            TaggedReceiver::I64(data_ch) => match_arm!(data_ch, axes_ch),
            TaggedReceiver::F32(data_ch) => match_arm!(data_ch, axes_ch),
            TaggedReceiver::F64(data_ch) => match_arm!(data_ch, axes_ch),
            TaggedReceiver::Bool(data_ch) => match_arm!(data_ch, axes_ch),
        }
    } else {
        panic!()
    }
}

fn setup_concat(ctx: OpContext) -> Result<ExecutionStep> {
    let axis: i64 = ctx.attrs.get("axis")?.try_into()?;

    macro_rules! match_arm {
        ($input_channels:expr, $variant:tt) => {{
            let input_channels = $input_channels
                .into_iter()
                .map(|ch| match ch {
                    TaggedReceiver::$variant(ch) => Ok(ch),
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
        TaggedReceiver::F32(_) => match_arm!(inputs, F32),
        TaggedReceiver::F64(_) => match_arm!(inputs, F64),
        TaggedReceiver::I32(_) => match_arm!(inputs, I32),
        TaggedReceiver::I64(_) => match_arm!(inputs, I64),
        TaggedReceiver::String(_) => match_arm!(inputs, String),
        TaggedReceiver::Bool(_) => match_arm!(inputs, Bool),
    }
}

fn setup_squeeze(ctx: OpContext) -> Result<ExecutionStep> {
    let [data_ch, axes_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();
    let axes_ch = match axes_ch {
        TaggedReceiver::I64(ch) => ch,
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
        TaggedReceiver::I32(ch) => match_arm!(ch, axes_ch),
        TaggedReceiver::F32(ch) => match_arm!(ch, axes_ch),
        TaggedReceiver::I64(ch) => match_arm!(ch, axes_ch),
        TaggedReceiver::F64(ch) => match_arm!(ch, axes_ch),
        TaggedReceiver::Bool(ch) => match_arm!(ch, axes_ch),
        TaggedReceiver::String(ch) => match_arm!(ch, axes_ch),
    }
}

fn setup_if(ctx: OpContext) -> Result<ExecutionStep> {
    // Explicit node input
    let [cond_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    // Condition must be of type bool
    let cond_ch = if let TaggedReceiver::Bool(cond_ch) = cond_ch {
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
