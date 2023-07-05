use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Receiver;

use itertools::sorted;
use onnx_protos::onnx_proto3::GraphProto;

use crate::opset::utils::*;
use crate::session::SetupFnSignature;
use crate::{channel, value_channel, Dispatchable, Session};

mod arg_min_max;
mod cast;
mod clip;
mod constant;
mod cumsum;
mod gather;
mod gather_elements;
mod min;
mod mod_;
mod one_hot;
mod reduce;
mod reshape;
mod shape;
mod simple_element_wise;
mod split;
mod transpose;
mod unique;
mod where_;

/// HashMap of setup function of the ai.onnx domain version 18(?).
#[rustfmt::skip]
pub fn get_opset() -> HashMap<&'static str, SetupFnSignature> {
    let mut d = simple_element_wise::get_setup_dict();
    d.insert("Add", binary_op_factory!(numeric, |a, b| Ok(a + b)));
    d.insert("And", binary_op_factory!(boolean, |a, b| Ok(a & b)));
    d.insert("ArgMax", Box::new(arg_min_max::setup_argmax));
    d.insert("ArgMin", Box::new(arg_min_max::setup_argmin));

    d.insert("BitwiseAnd", binary_op_factory!(int, |a, b| Ok(a & b)));
    d.insert("BitwiseOr", binary_op_factory!(int, |a, b| Ok(a | b)));
    d.insert("BitwiseXor", binary_op_factory!(int, |a, b| Ok(a ^ b)));
    
    d.insert("Cast", Box::new(cast::setup_cast));
    d.insert("CastLike", Box::new(cast::setup_cast_like));
    d.insert("Clip", Box::new(clip::setup_clip));
    d.insert("CumSum", Box::new(cumsum::setup_cumsum));
    d.insert("Concat", Box::new(setup_concat));
    d.insert("Constant", Box::new(constant::setup_constant));
    d.insert("Div", binary_op_factory!(numeric, |a, b| Ok(a / b)));
    d.insert("Equal", bin_numeric_with_co_broadcast!(PartialEq::eq));
    d.insert("Greater", bin_numeric_with_co_broadcast!(PartialOrd::gt));
    d.insert("Gather", Box::new(gather::setup_gather));
    d.insert("GatherElements", Box::new(gather_elements::setup_gather_elements));

    d.insert("GreaterOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::ge));
    d.insert("If", Box::new(setup_if));
    d.insert("Less", bin_numeric_with_co_broadcast!(PartialOrd::lt));
    d.insert("LessOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::le));
    d.insert("Mod", Box::new(mod_::setup_mod));
    d.insert("Min", Box::new(min::setup_min));
    d.insert("Mul", binary_op_factory!(numeric, |a, b| Ok(a * b)));
    d.insert("OneHot", Box::new(one_hot::setup_one_hot));
    d.insert("Or", binary_op_factory!(boolean, |a, b| Ok(a | b)));
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
    d.insert("Split", Box::new(split::setup_split));
    d.insert("Squeeze", Box::new(setup_squeeze));
    d.insert("Sub", binary_op_factory!(numeric, |a, b| Ok(a - b)));
    d.insert("Shape", Box::new(shape::setup_shape));
    d.insert("Transpose", Box::new(transpose::setup_transpose));
    d.insert("Unsqueeze", Box::new(setup_unsqueeze));
    d.insert("Unique", Box::new(unique::setup_unique));
    d.insert("Where", Box::new(where_::setup_where));
    d.insert("Xor", binary_op_factory!(boolean, |a, b| Ok(a ^ b)));
    d
}

fn setup_unsqueeze(ctx: OpContext) -> Result<ExecutionStep> {
    let (data_ch, axes_ch): (_, Receiver<ArcArrayD<i64>>) = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedReceiver,
        data_ch,
        data_ch,
        setup((data_ch, axes_ch), unsqueeze)
    )
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

    let inputs = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        Homogeneous,
        inputs,
        rec,
        setup((rec,), move |(inputs,)| state.concat((inputs,)))
    )
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
    let (data_ch, axes_ch): (_, Receiver<ArcArrayD<i64>>) = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedReceiver,
        data_ch,
        data_ch,
        setup((data_ch, axes_ch), squeeze)
    )
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

    // Explicit node input
    let cond_ch: Receiver<ArcArrayD<bool>> = ctx.subscribe_to_inputs().narrow()?;

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
