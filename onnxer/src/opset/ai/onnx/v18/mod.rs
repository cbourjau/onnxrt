use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Receiver;

use anyhow::{anyhow, bail, Result};
use itertools::sorted;
use ndarray::{Array, Array2, Axis, IntoDimension, Ix1, IxDyn, Zip};
use onnx_protos::onnx_proto3::GraphProto;

use crate::attribute::{Attribute, Attributes};
use crate::input_types::Homogeneous;
use crate::session::{ExecutionStep, OpContext, SetupFnSignature};
use crate::utils::{co_broadcast, convert_negative_index, wrap_1_input, zip_map_with_co_broadcast};
use crate::{
    channel, value_channel, ArcArrayD, DataType, Dispatchable, HomogeneousNumericFixed, Session,
    TaggedReceiver,
};

mod arg_min_max;
mod clip;
mod cumsum;
mod min;
mod mod_;
mod one_hot;
mod reduce;
mod reshape;
mod simple_element_wise;
mod where_;

use crate::setup_utils::{setup, setup_variadic_out};

macro_rules! binary_op {
    (int, $fun:expr) => {
        binary_op!(
            $fun,
            I32, I64, U32, U64 |
            rec => bail!("Expected integer input, found: '{:?}'", rec)
        )
    };
    (numeric, $fun:expr) => {
        binary_op!(
            $fun,
            F32, F64, I32, I64, U32, U64 |
        )
    };
    (float, $fun:expr) => {
        binary_op!(
            $fun,
            F32, F64 |
            rec => bail!("Expected floating point input, found: '{:?}'", rec)
        )
    };
    (signed, $fun:expr) => {
        binary_op!(
            $fun,
            I32, I64, F32, F64 |
            rec => bail!("Expected signed input, found: '{:?}'", rec)
        )
    };
    ($fun:expr, $($variant:tt),* | $($rec:tt => $bail:expr)?) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let inputs: HomogeneousNumericFixed<2> = ctx.input_channels().try_into().unwrap();
            use HomogeneousNumericFixed::*;
            match inputs {
                $($variant([a_ch, b_ch]) => {
                    setup((a_ch, b_ch), move |(a, b)| $fun(a, b))
                })*,
                $($rec => $bail)?
            }
        })
    };
}

macro_rules! bin_numeric_with_co_broadcast {
    ($fun:expr) => {
        binary_op!(numeric, |a, b| zip_map_with_co_broadcast(a, b, $fun))
    };
}

macro_rules! setup_all {
    ($tagged_rec:expr, $rec_name:ident, $setup:expr) => {
        match $tagged_rec {
            TaggedReceiver::F32($rec_name) => $setup,
            TaggedReceiver::F64($rec_name) => $setup,
            TaggedReceiver::I32($rec_name) => $setup,
            TaggedReceiver::I64($rec_name) => $setup,
            TaggedReceiver::U32($rec_name) => $setup,
            TaggedReceiver::U64($rec_name) => $setup,
            TaggedReceiver::String($rec_name) => $setup,
            TaggedReceiver::Bool($rec_name) => $setup,
        }
    };
}

/// HashMap of setup function of the ai.onnx domain version 18(?).
#[rustfmt::skip]
pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut d = simple_element_wise::get_setup_dict();
    d.insert("Add", binary_op!(numeric, |a, b| Ok(a + b)));
    d.insert("And", binary_boolean_factory(|(a, b)| Ok(a & b)));
    d.insert("ArgMax", Box::new(arg_min_max::setup_argmax));
    d.insert("ArgMin", Box::new(arg_min_max::setup_argmin));

    d.insert("BitwiseAnd", binary_op!(int, |a, b| Ok(a & b)));
    d.insert("BitwiseOr", binary_op!(int, |a, b| Ok(a | b)));
    d.insert("BitwiseXor", binary_op!(int, |a, b| Ok(a ^ b)));
    
    d.insert("Cast", Box::new(setup_cast));
    d.insert("Clip", Box::new(clip::setup_clip));
    d.insert("CumSum", Box::new(cumsum::setup_cumsum));
    d.insert("Concat", Box::new(setup_concat));
    d.insert("Constant", Box::new(setup_constant));
    d.insert("Div", binary_op!(numeric, |a, b| Ok(a / b)));
    d.insert("Equal", bin_numeric_with_co_broadcast!(PartialEq::eq));
    d.insert("Greater", bin_numeric_with_co_broadcast!(PartialOrd::gt));
    d.insert("GreaterOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::ge));
    d.insert("If", Box::new(setup_if));
    d.insert("Less", bin_numeric_with_co_broadcast!(PartialOrd::lt));
    d.insert("LessOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::le));
    d.insert("Mod", Box::new(mod_::setup_mod));
    d.insert("Min", Box::new(min::setup_min));
    d.insert("Mul", binary_op!(numeric, |a, b| Ok(a * b)));
    d.insert("OneHot", Box::new(one_hot::setup_one_hot));
    d.insert("Or", binary_boolean_factory(|(a, b)| Ok(a | b)));
    d.insert("ReduceMax", Box::new(reduce::setup_reducemax));
    d.insert("ReduceMean", Box::new(reduce::setup_reducemean));
    d.insert("ReduceMin", Box::new(reduce::setup_reducemin));
    d.insert("ReduceProd", Box::new(reduce::setup_reduceprod));
    d.insert("ReduceL1", Box::new(reduce::setup_reducel1));
    d.insert("ReduceL2", Box::new(reduce::setup_reducel2));
    d.insert("ReduceSumSquare", Box::new(reduce::setup_reducesumsquare));
    d.insert("ReduceSum", Box::new(reduce::setup_reducesum));
    d.insert("ReduceLogSum", Box::new(reduce::setup_reducelogsum));
    d.insert("ReduceLogSumExp", Box::new(reduce::setup_reducelogsumexp));

    d.insert("Reshape", Box::new(reshape::setup_reshape));
    d.insert("Split", Box::new(setup_split));
    d.insert("Squeeze", Box::new(setup_squeeze));
    d.insert("Sub", binary_op!(numeric, |a, b| Ok(a - b)));
    d.insert("Unsqueeze", Box::new(setup_unsqueeze));
    d.insert("Where", Box::new(where_::setup_where));
    d.insert("Xor", binary_boolean_factory(|(a, b)| Ok(a ^ b)));
    d
}

/// Create setup function with two boolean inputs
fn binary_boolean_factory<Op>(op: Op) -> SetupFnSignature
where
    Op: Fn((ArcArrayD<bool>, ArcArrayD<bool>)) -> Result<ArcArrayD<bool>>
        + Send
        + Sync
        + 'static
        + Copy,
{
    Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
        let [a_ch, b_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();
        use TaggedReceiver::*;
        match (a_ch, b_ch) {
            (Bool(a_ch), Bool(b_ch)) => setup((a_ch, b_ch), op),
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
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::F32(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, Ok),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::F64(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0.0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, Ok),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::I32(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, Ok),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::I64(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, Ok),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::U32(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, Ok),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::U64(a_ch) => match to_type {
            DataType::Bool => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el != 0).into())),
            DataType::F32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f32).into())),
            DataType::F64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as f64).into())),
            DataType::I32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i32).into())),
            DataType::I64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as i64).into())),
            DataType::U32 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u32).into())),
            DataType::U64 => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el as u64).into())),
            DataType::String => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| el.to_string()).into())),
        },
        TaggedReceiver::String(a_ch) => match to_type {
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

fn setup_constant(ctx: OpContext) -> Result<ExecutionStep> {
    if ctx.attrs.get("sparse_value").is_ok() {
        unimplemented!("Sparse tensors are not supported.")
    }
    macro_rules! scalar {
        ($val:expr) => {{
            let val: ArcArrayD<_> = ndarray::arr0($val).into_dyn().into();
            setup((), move |()| Ok(val.clone()))
        }};
    }

    macro_rules! tensor_attr {
        ($val:expr) => {{
            let val = $val.clone();
            setup((), move |()| Ok(val.clone()))
        }};
    }

    // Value could be any of the others
    if let Ok(attr) = ctx.attrs.get("value").as_ref() {
        return match attr.clone() {
            Attribute::Bools(val) => tensor_attr!(val),
            Attribute::F32s(val) => tensor_attr!(val),
            Attribute::F64s(val) => tensor_attr!(val),
            Attribute::I32s(val) => tensor_attr!(val),
            Attribute::I64s(val) => tensor_attr!(val),
            Attribute::Strings(val) => tensor_attr!(val),
            Attribute::ScalarF32(val) => scalar!(*val),
            Attribute::ScalarI64(val) => scalar!(*val),
            Attribute::ScalarString(val) => {
                scalar!(val.to_owned())
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

    let state = Split::new(&ctx)?;

    // `input_channels` creates channels and must only be called once!
    let input_channels = ctx.input_channels();
    if input_channels.len() > 1 {
        bail!("`split` input of `Split` operator is not supported");
    }

    let [a_ch]: [_; 1] = input_channels.try_into().unwrap();

    setup_all!(
        a_ch,
        a_ch,
        setup_variadic_out((a_ch,), move |(a,)| state.split(a), n_outputs)
    )
}

struct Split {
    axis: i64,
    num_outputs: Option<usize>,
}

impl Split {
    fn new(ctx: &OpContext) -> Result<Self> {
        // Terrible naming in the standard! This is actually the chunk size!
        let num_outputs: Option<i64> = ctx.attrs.get_optional("num_outputs")?;

        let num_outputs = {
            if let Some(num_outputs) = num_outputs {
                let num_outputs: usize = num_outputs.try_into()?;
                if ctx.n_outputs() != num_outputs {
                    panic!()
                }
                Some(num_outputs)
            } else {
                Some(ctx.n_outputs())
            }
        };
        let axis = ctx.attrs.get_with_default("axis", 0i64)?;
        Ok(Self { axis, num_outputs })
    }

    fn split<T>(&self, a: ArcArrayD<T>) -> Result<Vec<ArcArrayD<T>>>
    where
        T: Clone,
    {
        let axis = convert_negative_index(self.axis, a.shape().len());

        let len_axis = a.shape()[axis];

        let chunk_size = match self.num_outputs {
            Some(num_outputs) => {
                if len_axis < num_outputs {
                    bail!(
                        "Split: Cannot split dimension with '{}' elements into '{}' chunks.",
                        len_axis,
                        num_outputs,
                    )
                }
                if len_axis % num_outputs == 0 {
                    len_axis / num_outputs
                } else {
                    len_axis / num_outputs + 1
                }
            }
            None => {
                // Requires 'split' input; not implemented
                panic!()
            }
        };
        Ok(a.axis_chunks_iter(Axis(axis), chunk_size)
            .map(|arr| arr.to_owned().into())
            .collect())
    }
}

fn setup_unsqueeze(ctx: OpContext) -> Result<ExecutionStep> {
    let [data_ch, axes_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();

    if let TaggedReceiver::I64(axes_ch) = axes_ch {
        setup_all!(data_ch, data_ch, setup((data_ch, axes_ch), unsqueeze))
    } else {
        panic!()
    }
}

fn unsqueeze<D>((data, axes): (ArcArrayD<D>, ArcArrayD<i64>)) -> Result<ArcArrayD<D>> {
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

    let axes = axes.into_dimensionality::<Ix1>()?.to_vec();
    let new_shape = compute_expanded(data.shape(), &axes);
    Ok(data.into_shape(new_shape)?)
}

fn setup_concat(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Concat::new(&ctx.attrs)?;

    let inputs = Homogeneous::new_from_iter(ctx.input_channels().into_iter())?;

    use Homogeneous::*;
    match inputs {
        F32(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
        F64(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
        I32(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
        I64(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
        U32(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
        U64(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
        String(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
        Bool(rec) => setup((rec,), move |(inputs,)| state.concat((inputs,))),
    }
}

#[derive(Copy, Clone)]
struct Concat {
    axis: i64,
}
impl Concat {
    fn new(attrs: &Attributes) -> Result<Self> {
        let axis: i64 = attrs.get("axis")?.try_into()?;
        Ok(Self { axis })
    }

    fn concat<T>(self, (arrays,): (Vec<ArcArrayD<T>>,)) -> Result<ArcArrayD<T>>
    where
        T: Clone,
    {
        let array_views: Vec<_> = arrays.iter().map(|a| a.into()).collect();
        let n_dims = arrays[0].ndim();
        let axis = convert_negative_index(self.axis, n_dims);
        let out = ndarray::concatenate(Axis(axis), array_views.as_slice())?;
        Ok(out.into())
    }
}

fn setup_squeeze(ctx: OpContext) -> Result<ExecutionStep> {
    let [data_ch, axes_ch]: [_; 2] = ctx.input_channels().try_into().unwrap();
    let axes_ch = match axes_ch {
        TaggedReceiver::I64(ch) => ch,
        _ => bail!("Input `axes` of Squeeze must be of type `tensor(int64)`"),
    };

    setup_all!(data_ch, data_ch, setup((data_ch, axes_ch), squeeze))
}

fn squeeze<D>((data, indices): (ArcArrayD<D>, ArcArrayD<i64>)) -> Result<ArcArrayD<D>> {
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

    Ok(out)
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
