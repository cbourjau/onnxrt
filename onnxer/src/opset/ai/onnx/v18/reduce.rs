use num_traits::{AsPrimitive, Bounded, Float, One, Signed, Zero};
use std::cmp::PartialOrd;
use std::ops::{Add, Mul};

use crate::attribute::Attributes;

use super::*;

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

    fn reduce_max<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd,
    {
        self.reduce(data, axes, D::min_value(), fold_max)
    }

    fn reduce_min<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd,
    {
        self.reduce(data, axes, D::max_value(), fold_min)
    }

    fn reduce_sum<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + Zero,
    {
        self.reduce(data, axes, D::zero(), fold_sum)
    }

    fn reduce_sum_square<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + Zero + Mul<Output = D>,
    {
        let sq_data: ArcArrayD<_> = data.mapv(|el| el * el).into();
        self.reduce_sum(sq_data, axes)
    }

    fn reduce_l1<D>(&self, data: ArcArrayD<D>, axes: Option<ArcArrayD<i64>>) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + Zero + Signed,
    {
        let abs_data: ArcArrayD<_> = data.mapv(|el| el.abs()).into();
        self.reduce_sum(abs_data, axes)
    }

    fn reduce_l2<D>(&self, data: ArcArrayD<D>, axes: Option<ArcArrayD<i64>>) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + Zero + Float,
    {
        Ok(self
            .reduce_sum_square(data, axes)?
            .mapv(|el| el.sqrt())
            .into())
    }

    fn reduce_log_sum<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + Zero + Float,
    {
        let sum = self.reduce_sum(data, axes)?;
        let log_data: ArcArrayD<_> = sum.mapv(|el| el.ln()).into();
        Ok(log_data)
    }

    fn reduce_log_sum_exp<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + Zero + Float,
    {
        let exp: ArcArrayD<_> = data.mapv(|el| el.exp()).into();
        let sum = self.reduce_sum(exp, axes)?;
        let log_data: ArcArrayD<_> = sum.mapv(|el| el.ln()).into();
        Ok(log_data)
    }

    fn reduce_prod<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + One,
    {
        self.reduce(data, axes, D::one(), fold_mul)
    }

    fn reduce_mean<D>(
        &self,
        data: ArcArrayD<D>,
        axes: Option<ArcArrayD<i64>>,
    ) -> Result<ArcArrayD<D>>
    where
        D: Bounded + Copy + PartialOrd + Zero + std::ops::Div<Output = D> + 'static,
        usize: AsPrimitive<D>,
    {
        let n_initial: D = data.len().as_();

        let reduced = self.reduce(data, axes, D::zero(), fold_sum)?;
        let n_reduced: D = reduced.len().as_();
        let scaling = n_initial / n_reduced;
        Ok(reduced.mapv(|v| v / scaling).into())
    }

    fn reduce<T, F>(
        &self,
        arr: ArcArrayD<T>,
        axes: Option<ArcArrayD<i64>>,
        fold_init: T,
        fold_fn: F,
    ) -> Result<ArcArrayD<T>>
    where
        T: Copy + Bounded + PartialOrd,
        F: Fn(&T, &T) -> T,
    {
        let axes = axes.unwrap_or_else(|| ndarray::array!([]).into_dyn().into());
        let initial_shape = arr.shape().to_vec();
        let rank = initial_shape.len();

        // Deal with special case around empty `arr`
        let axes = if axes.is_empty() {
            if self.noop_empty {
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

        let reduced = if self.keep_dims {
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

fn validate_inputs(ctx: OpContext) -> Result<(TaggedReceiver, Option<Receiver<ArcArrayD<i64>>>)> {
    let mut inputs = ctx.input_channels().into_iter();
    let data_ch = inputs
        .next()
        .ok_or_else(|| anyhow!("Input `data` is missing"))?;
    let axes_ch = match inputs.next() {
        Some(TaggedReceiver::I64(ch)) => Some(ch),
        None => None,
        Some(_) => bail!("Input `axes` must be of type `int64`"),
    };
    if inputs.len() != 0 {
        bail!("Unexpected inputs");
    }
    Ok((data_ch, axes_ch))
}

macro_rules! create_setup_function {
    ($setup_name:tt, $member_fun:tt, $($variant:tt),*) => {
        pub fn $setup_name(ctx: OpContext) -> Result<ExecutionStep> {
            let state = State::new(&ctx.attrs)?;
            let (data_ch, axes_ch) = validate_inputs(ctx)?;

            use TaggedReceiver::*;
            match data_ch {
                $($variant(data_ch) => setup((data_ch, axes_ch), move |(data, axes)| {
                    state.$member_fun(data, axes)
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
