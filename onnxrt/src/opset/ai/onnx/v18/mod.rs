use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Receiver;

use crate::opset::utils::*;
use crate::session::SetupFnSignature;
use crate::{channel, Sendable};
use itertools::sorted;

mod arg_min_max;
mod batch_normalization;
mod bitshift;
mod cast;
mod clip;
mod compress;
mod constant;
mod constant_of_shape;
mod conv;
mod cumsum;
mod elu;
mod equal;
mod expand;
mod eye_like;
mod flatten;
mod gather;
mod gather_elements;
mod gather_nd;
mod global_average_pool;
mod if_;
mod isinf;
mod layer_normalization;
mod loop_;
mod matmul;
mod min_max;
mod mod_;
mod one_hot;
mod pow;
mod random_uniform_like;
mod range;
mod reduce;
mod relu_with_attr;
mod reshape;
mod scatter;
mod shape;
mod simple_element_wise;
mod size;
mod slice;
mod softmax;
mod split;
mod sum;
mod topk;
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
    d.insert("BatchNormalization", Box::new(batch_normalization::setup_bn));
    d.insert("BitShift", Box::new(bitshift::setup_bishift));
    d.insert("BitwiseAnd", binary_op_factory!(int, |a, b| Ok(a & b)));
    d.insert("BitwiseOr", binary_op_factory!(int, |a, b| Ok(a | b)));
    d.insert("BitwiseXor", binary_op_factory!(int, |a, b| Ok(a ^ b)));
    d.insert("Cast", Box::new(cast::setup_cast));
    d.insert("CastLike", Box::new(cast::setup_cast_like));
    d.insert("Clip", Box::new(clip::setup_clip));
    d.insert("Compress", Box::new(compress::setup_compress));
    d.insert("Concat", Box::new(setup_concat));
    d.insert("Constant", Box::new(constant::setup_constant));
    d.insert("ConstantOfShape", Box::new(constant_of_shape::setup_constant_of_shape));
    d.insert("Conv", Box::new(conv::setup_conv));
    d.insert("CumSum", Box::new(cumsum::setup_cumsum));
    d.insert("Div", binary_op_factory!(numeric, |a, b| Ok(a / b)));
    d.insert("Elu", Box::new(elu::setup_elu));
    d.insert("Equal", Box::new(equal::setup_equal));
    d.insert("Expand", Box::new(expand::setup_expand));
    d.insert("EyeLike", Box::new(eye_like::setup_eyelike));
    d.insert("Flatten", Box::new(flatten::setup_flatten));
    d.insert("Gather", Box::new(gather::setup_gather));
    d.insert("GatherElements", Box::new(gather_elements::setup_gather_elements));
    d.insert("GatherND", Box::new(gather_nd::setup_gather_nd));
    d.insert("GlobalAveragePool", Box::new(global_average_pool::setup_gap));
    d.insert("Greater", bin_numeric_with_co_broadcast!(PartialOrd::gt));
    d.insert("GreaterOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::ge));
    d.insert("If", Box::new(if_::setup_if));
    d.insert("IsInf", Box::new(isinf::setup_isinf));
    d.insert("LayerNormalization", Box::new(layer_normalization::setup_ln));
    d.insert("Loop", Box::new(loop_::setup_loop));
    d.insert("LogSoftmax", Box::new(softmax::setup_log_softmax));
    d.insert("LeakyRelu", Box::new(relu_with_attr::setup_leaky_relu));
    d.insert("Less", bin_numeric_with_co_broadcast!(PartialOrd::lt));
    d.insert("LessOrEqual", bin_numeric_with_co_broadcast!(PartialOrd::le));
    d.insert("MatMul", Box::new(matmul::setup_matmul));
    d.insert("Max", Box::new(min_max::setup_max));
    d.insert("Min", Box::new(min_max::setup_min));
    d.insert("Mod", Box::new(mod_::setup_mod));
    d.insert("Mul", binary_op_factory!(numeric, |a, b| Ok(a * b)));
    d.insert("OneHot", Box::new(one_hot::setup_one_hot));
    d.insert("Or", binary_op_factory!(boolean, |a, b| Ok(a | b)));
    d.insert("Pow", Box::new(pow::setup_pow));
    d.insert("Range", Box::new(range::setup_range));
    d.insert("RandomUniformLike", Box::new(random_uniform_like::setup_random_uniform_like));
    d.insert("ReduceL1", Box::new(reduce::setup_reducel1));
    d.insert("ReduceL2", Box::new(reduce::setup_reducel2));
    d.insert("ReduceLogSum", Box::new(reduce::setup_reducelogsum));
    d.insert("ReduceLogSumExp", Box::new(reduce::setup_reducelogsumexp));
    d.insert("ReduceMax", Box::new(reduce::setup_reducemax));
    d.insert("ReduceMean", Box::new(reduce::setup_reducemean));
    d.insert("ReduceMin", Box::new(reduce::setup_reducemin));
    d.insert("ReduceProd", Box::new(reduce::setup_reduceprod));
    d.insert("ReduceSum", Box::new(reduce::setup_reducesum));
    d.insert("ReduceSumSquare", Box::new(reduce::setup_reducesumsquare));
    d.insert("Reshape", Box::new(reshape::setup_reshape));
    d.insert("ScatterElements", Box::new(scatter::elements::setup_scatter_elements));
    d.insert("ScatterND", Box::new(scatter::nd::setup_scatter_nd));
    d.insert("Shape", Box::new(shape::setup_shape));
    d.insert("Size", Box::new(size::setup_size));
    d.insert("Slice", Box::new(slice::setup_slice));
    d.insert("Softmax", Box::new(softmax::setup_softmax));    
    d.insert("Split", Box::new(split::setup_split));
    d.insert("Squeeze", Box::new(setup_squeeze));
    d.insert("Sub", binary_op_factory!(numeric, |a, b| Ok(a - b)));
    d.insert("Sum", Box::new(sum::setup_sum));
    d.insert("ThresholdedRelu", Box::new(relu_with_attr::setup_thresholded_relu));
    d.insert("TopK", Box::new(topk::setup_topk));
    d.insert("Transpose", Box::new(transpose::setup_transpose));
    d.insert("Unique", Box::new(unique::setup_unique));
    d.insert("Unsqueeze", Box::new(setup_unsqueeze));
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

fn unsqueeze<D: std::fmt::Debug>(
    (data, axes): (ArcArrayD<D>, ArcArrayD<i64>),
) -> Result<ArcArrayD<D>> {
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
    let axes = axes.as_standard_layout().to_owned();
    // Safe to unwrap because we ensure standard layout above.
    let axes = axes.as_slice().unwrap();

    let new_shape = compute_expanded(data.shape(), axes);
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
