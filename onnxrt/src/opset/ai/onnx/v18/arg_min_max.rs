use anyhow::Result;
use ndarray::Axis;

use crate::{Attributes, opset::utils::*, session::OpContext};

pub fn setup_argmax(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let State {
        axis,
        keep_dims,
        use_last,
    } = State::new(&ctx.attrs)?;
    let a = ctx.subscribe_to_inputs().narrow()?;

    use Numeric::*;
    match a {
        F16(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        F32(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        F64(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),

        I8(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        I16(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        I32(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        I64(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),

        U8(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        U16(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        U32(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
        U64(a) => setup_unary(a, move |a| arg_max(a.view(), axis, keep_dims, use_last)),
    }
}

pub fn setup_argmin(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let State {
        axis,
        keep_dims,
        use_last,
    } = State::new(&ctx.attrs)?;
    let a = ctx.subscribe_to_inputs().narrow()?;

    use Numeric::*;
    match a {
        F16(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        F32(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        F64(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),

        I8(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        I16(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        I32(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        I64(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),

        U8(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        U16(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        U32(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
        U64(a) => setup_unary(a, move |a| arg_min(a.view(), axis, keep_dims, use_last)),
    }
}

pub fn arg_max<T>(
    data: ArrayViewD<T>,
    axis: i64,
    keep_dims: bool,
    use_last: bool,
) -> Result<ArrayD<i64>>
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
    Ok(fold_axis(data, axis, keep_dims, start_acc, &fold_fn)?.mapv(|(idx, _v)| idx as i64))
}

pub fn arg_min<T>(
    data: ArrayViewD<T>,
    axis: i64,
    keep_dims: bool,
    use_last: bool,
) -> Result<ArrayD<i64>>
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
    Ok(fold_axis(data, axis, keep_dims, start_acc, &fold_fn)?.mapv(|(idx, _v)| idx as i64))
}

/// State of the ArgMin and ArgMax operator
struct State {
    axis: i64,
    keep_dims: bool,
    use_last: bool,
}

impl State {
    fn new(attrs: &Attributes) -> Result<Self> {
        let axis = attrs.get_with_default("axis", 0i64)?;
        let keep_dims = attrs.get_with_default("keepdims", 1i64)? == 1;
        let use_last = attrs.get_with_default("select_last_index", 0i64)? != 0;

        Ok(Self {
            axis,
            keep_dims,
            use_last,
        })
    }
}

fn fold_axis<T, Acc, Fold>(
    data: ArrayViewD<T>,
    axis: i64,
    keep_dim: bool,
    init_fold: Acc,
    fold_fn: &Fold,
) -> Result<ArrayD<Acc>>
where
    T: Copy + num_traits::Bounded + ::std::cmp::PartialOrd,
    Acc: Copy,
    Fold: Fn(Acc, (usize, T)) -> Acc,
{
    let axis = convert_negative_index(axis, data.shape().len());
    let rank = data.shape().len();
    if axis >= rank {
        anyhow::bail!(
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
