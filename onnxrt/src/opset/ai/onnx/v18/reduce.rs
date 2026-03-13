use num_traits::{AsPrimitive, Bounded, Float, One, Signed, Zero};
use std::cmp::PartialOrd;
use std::ops::{Add, Mul};

use crate::attribute::Attributes;

use super::*;

fn reduce_max<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd,
{
    reduce(data, axes, D::min_value(), fold_max, keep_dims, noop_empty)
}

fn reduce_min<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd,
{
    reduce(data, axes, D::max_value(), fold_min, keep_dims, noop_empty)
}

fn reduce_sum<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + Zero,
{
    reduce(data, axes, D::zero(), fold_sum, keep_dims, noop_empty)
}

fn reduce_sum_square<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + Zero + Mul<Output = D>,
{
    let sq_data = data.mapv(|el| el * el);
    reduce_sum(sq_data.view(), axes, keep_dims, noop_empty)
}

fn reduce_l1<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + Zero + Signed,
{
    let abs_data = data.mapv(|el| el.abs());
    reduce_sum(abs_data.view(), axes, keep_dims, noop_empty)
}

fn reduce_l2<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + Zero + Float,
{
    Ok(reduce_sum_square(data, axes, keep_dims, noop_empty)?.mapv(|el| el.sqrt()))
}

fn reduce_log_sum<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + Zero + Float,
{
    let sum = reduce_sum(data, axes, keep_dims, noop_empty)?;
    let log_data = sum.mapv(|el| el.ln());
    Ok(log_data)
}

fn reduce_log_sum_exp<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + Zero + Float,
{
    let exp = data.mapv(|el| el.exp());
    let sum = reduce_sum(exp.view(), axes, keep_dims, noop_empty)?;
    let log_data = sum.mapv(|el| el.ln());
    Ok(log_data)
}

fn reduce_prod<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + One,
{
    reduce(data, axes, D::one(), fold_mul, keep_dims, noop_empty)
}

pub fn reduce_mean<D>(
    data: ArrayViewD<D>,
    axes: &[i64],
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<D>>
where
    D: Bounded + Copy + PartialOrd + Zero + std::ops::Div<Output = D> + 'static,
    usize: AsPrimitive<D>,
{
    let n_initial: D = data.len().as_();

    let reduced = reduce(data, axes, D::zero(), fold_sum, keep_dims, noop_empty)?;
    let n_reduced: D = reduced.len().as_();
    let scaling = n_initial / n_reduced;
    Ok(reduced.mapv(|v| v / scaling))
}

fn reduce<T, F>(
    arr: ArrayViewD<T>,
    axes: &[i64],
    fold_init: T,
    fold_fn: F,
    keep_dims: bool,
    noop_empty: bool,
) -> Result<ArrayD<T>>
where
    T: Copy + Bounded + PartialOrd,
    F: Fn(&T, &T) -> T,
{
    let initial_shape = arr.shape().to_vec();
    let rank = initial_shape.len();

    // Deal with special case around empty `arr`
    let axes = if axes.is_empty() {
        if noop_empty {
            return Ok(arr.to_owned());
        } else {
            (0..arr.shape().len()).collect()
        }
    } else {
        axes.iter()
            .map(|idx| convert_negative_index(*idx, rank))
            .collect::<HashSet<_>>()
    };
    if let Some(max_axis_input) = axes.iter().max()
        && *max_axis_input >= rank
    {
        bail!(
            "Input `axes={}` is incompatible tensor of `rank={}`",
            *max_axis_input,
            rank
        );
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
        .filter(|&(_, &red_this)| red_this)
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

    let reduced = if keep_dims {
        let out_shape: Vec<_> = initial_shape
            .iter()
            .enumerate()
            .map(|(i, &el)| if axes.contains(&i) { 1 } else { el })
            .collect();
        reduced.into_shape(out_shape)?
    } else {
        reduced
    };
    Ok(reduced)
}

struct State {
    keep_dims: bool,
    noop_empty: bool,
}

impl State {
    fn new(attrs: &Attributes) -> Result<Self> {
        let keep_dims = attrs.get_with_default("keepdims", 1i64)? == 1;
        let noop_empty = attrs.get_with_default("noop_with_empty_axes", 0i64)? != 0;
        Ok(Self {
            keep_dims,
            noop_empty,
        })
    }
}

fn fold_max<T: PartialOrd + Copy>(acc: &T, el: &T) -> T {
    if el > acc { *el } else { *acc }
}

fn fold_min<T: PartialOrd + Copy>(acc: &T, el: &T) -> T {
    if el < acc { *el } else { *acc }
}

fn fold_mul<T: Copy + Mul<Output = T>>(acc: &T, el: &T) -> T {
    *acc * *el
}

fn fold_sum<T: Copy + Add<Output = T>>(acc: &T, el: &T) -> T {
    *acc + *el
}

fn validate_inputs(
    ctx: OpContext,
) -> Result<(TaggedTensorReceiver, Option<Receiver<ArcArrayD<i64>>>)> {
    let (data, axes): (Option<_>, _) = ctx.input_channels_with_optional().narrow()?;
    let data = data.ok_or_else(|| anyhow!("Input `data` is missing"))?;
    Ok((data, axes))
}

macro_rules! create_setup_function {
    ($setup_name:tt, $fun:tt, $($variant:tt),*) => {
        pub fn $setup_name(ctx: OpContext) -> Result<ExecutionStep> {
            let state = State::new(&ctx.attrs)?;
            let (data_ch, axes_ch) = validate_inputs(ctx)?;

            use TaggedTensorReceiver::*;
            match data_ch {
                $($variant(data_ch) => setup((data_ch, axes_ch), move |(data, axes)| {
                    let axes = axes.map(|axes| axes.as_standard_layout().to_owned());
                    let axes = {
                        if let Some(axes) = axes.as_ref() {
                            axes.as_slice().unwrap()
                        } else {
                            &[][..]
                        }
                    };
                    let out = $fun(data.view(), axes, state.keep_dims, state.noop_empty)?;
                    Ok(out)
                }),)*
                _ => bail!("Type support of reduce operations is limited."),
            }
        }
    };
}

create_setup_function!(setup_reducemax, reduce_max, I32, I64, F32, F64, U32, U64);
create_setup_function!(setup_reducemin, reduce_min, I32, I64, F32, F64, U32, U64);
create_setup_function!(setup_reducesum, reduce_sum, I32, I64, F32, F64, U32, U64);
create_setup_function!(setup_reducemean, reduce_mean, I32, I64, F32, F64, U32, U64);
create_setup_function!(setup_reduceprod, reduce_prod, I32, I64, F32, F64, U32, U64);
create_setup_function!(
    setup_reducesumsquare,
    reduce_sum_square,
    I32,
    I64,
    F32,
    F64,
    U32,
    U64
);
create_setup_function!(setup_reducel1, reduce_l1, I32, I64, F32, F64); // missing abs for unsigned
create_setup_function!(setup_reducel2, reduce_l2, F32, F64); // missing sqrt for integers
create_setup_function!(setup_reducelogsum, reduce_log_sum, F32, F64); // missing sqrt for integers
create_setup_function!(setup_reducelogsumexp, reduce_log_sum_exp, F32, F64); // missing sqrt for integers
