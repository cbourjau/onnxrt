use super::*;

use crate::{attribute::Attributes, setup_utils::setup};

pub fn setup_argmax(ctx: OpContext) -> Result<ExecutionStep> {
    let state = State::new(&ctx.attrs)?;

    match ctx.input_channels().try_into()? {
        HomogeneousNumericFixed::F32([ch]) => setup((ch,), move |(data,)| state.arg_max(&data)),
        HomogeneousNumericFixed::F64([ch]) => setup((ch,), move |(data,)| state.arg_max(&data)),
        HomogeneousNumericFixed::I32([ch]) => setup((ch,), move |(data,)| state.arg_max(&data)),
        HomogeneousNumericFixed::I64([ch]) => setup((ch,), move |(data,)| state.arg_max(&data)),
    }
}

pub fn setup_argmin(ctx: OpContext) -> Result<ExecutionStep> {
    let state = State::new(&ctx.attrs)?;

    match ctx.input_channels().try_into()? {
        HomogeneousNumericFixed::F32([ch]) => setup((ch,), move |(data,)| state.arg_min(&data)),
        HomogeneousNumericFixed::F64([ch]) => setup((ch,), move |(data,)| state.arg_min(&data)),
        HomogeneousNumericFixed::I32([ch]) => setup((ch,), move |(data,)| state.arg_min(&data)),
        HomogeneousNumericFixed::I64([ch]) => setup((ch,), move |(data,)| state.arg_min(&data)),
    }
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

    fn arg_max<T>(&self, data: &ArcArrayD<T>) -> Result<ArcArrayD<i64>>
    where
        T: Copy + num_traits::Bounded + ::std::cmp::PartialOrd,
    {
        let start_acc = (0, T::min_value());
        let fold_fn = |acc: (usize, T), (idx, v)| {
            if v > acc.1 || (self.use_last && acc.1 == v) {
                (idx, v)
            } else {
                acc
            }
        };
        Ok(
            fold_axis(data, self.axis, self.keep_dims, start_acc, &fold_fn)?
                .mapv(|(idx, _v)| idx as i64)
                .into(),
        )
    }

    fn arg_min<T>(&self, data: &ArcArrayD<T>) -> Result<ArcArrayD<i64>>
    where
        T: Copy + num_traits::Bounded + ::std::cmp::PartialOrd,
    {
        let start_acc = (0, T::max_value());
        let fold_fn = |acc: (usize, T), (idx, v)| {
            if v < acc.1 || (self.use_last && acc.1 == v) {
                (idx, v)
            } else {
                acc
            }
        };
        Ok(
            fold_axis(data, self.axis, self.keep_dims, start_acc, &fold_fn)?
                .mapv(|(idx, _v)| idx as i64)
                .into(),
        )
    }
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
