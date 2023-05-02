use num_traits::Bounded;
use std::cmp::PartialOrd;
use std::ops::{Add, Mul};

use super::*;

struct CommonArgs {
    keepdims: bool,
    noop_empty: bool,
    axes_ch: Option<Receiver<ArcArrayD<i64>>>,
}

fn setup(ctx: OpContext) -> Result<(TaggedReceiver, CommonArgs)> {
    let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
    let noop_empty = ctx.attrs.get_with_default("noop_with_empty_axes", 0i64)? != 0;

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

pub fn setup_reducemax(ctx: OpContext) -> Result<ExecutionStep> {
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

pub fn setup_reducemean(ctx: OpContext) -> Result<ExecutionStep> {
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

pub fn setup_reducemin(ctx: OpContext) -> Result<ExecutionStep> {
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

pub fn setup_reduceprod(ctx: OpContext) -> Result<ExecutionStep> {
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

pub fn setup_reducesum(ctx: OpContext) -> Result<ExecutionStep> {
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
