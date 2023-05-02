use super::*;

macro_rules! match_arm_argminmax {
    ($fn:expr, $ch:expr, $axis:expr, $keepdims:expr, $use_last:expr) => {{
        let (sdr, rec) = ArcArrayD::channel();
        let fun = move || {
            let data = $ch.recv()?;
            let out = $fn(&data, $axis, $keepdims, $use_last)?;
            sdr.send(out)?;
            Ok(())
        };
        Ok(ExecutionStep {
            fun: Box::new(fun),
            outputs: vec![rec],
        })
    }};
}

pub fn setup_argmax(ctx: OpContext) -> Result<ExecutionStep> {
    let axis = ctx.attrs.get_with_default("axis", 0i64)?;
    let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
    let use_last = ctx.attrs.get_with_default("select_last_index", 0i64)? != 0;

    match ctx.input_channels().try_into()? {
        HomogeneousNumericFixed::F32([ch]) => {
            match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
        }
        HomogeneousNumericFixed::F64([ch]) => {
            match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
        }
        HomogeneousNumericFixed::I32([ch]) => {
            match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
        }
        HomogeneousNumericFixed::I64([ch]) => {
            match_arm_argminmax!(argmax, ch, axis, keepdims, use_last)
        }
    }
}

pub fn setup_argmin(ctx: OpContext) -> Result<ExecutionStep> {
    let axis = ctx.attrs.get_with_default("axis", 0i64)?;
    let keepdims = ctx.attrs.get_with_default("keepdims", 1i64)? == 1;
    let use_last = ctx.attrs.get_with_default("select_last_index", 0i64)? != 0;

    match ctx.input_channels().try_into()? {
        HomogeneousNumericFixed::F32([ch]) => {
            match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
        }
        HomogeneousNumericFixed::F64([ch]) => {
            match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
        }
        HomogeneousNumericFixed::I32([ch]) => {
            match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
        }
        HomogeneousNumericFixed::I64([ch]) => {
            match_arm_argminmax!(argmin, ch, axis, keepdims, use_last)
        }
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
