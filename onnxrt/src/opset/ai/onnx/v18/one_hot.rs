use super::*;

pub fn setup_one_hot(ctx: OpContext) -> Result<ExecutionStep> {
    let axis = ctx.attrs.get_with_default("axis", -1)?;

    // Spec: Standard is insanely lose around `indices` and
    // `depth`. Technically, floats may be used as indices, but that
    // is just a terrible. Thus, this is only implemented
    // for integers.
    let (indices, depth, values): (Integer, Integer, _) = ctx.subscribe_to_inputs().narrow()?;
    let indices = indices.into_i64();
    let depth = depth.into_i64();

    setup_all!(
        TaggedReceiver,
        values,
        values,
        setup((indices, depth, values), move |(indices, depth, values)| {
            let depth: usize = depth.into_shape([])?.into_scalar().try_into()?;
            if values.len() != 2 {
                bail!("OneHot: Input 'values' must have two elements.")
            }
            let mut values_it = values.into_iter();
            let off_value = values_it.next().unwrap();
            let on_value = values_it.next().unwrap();
            one_hot(indices, depth, &off_value, &on_value, axis)
        })
    )
}

fn one_hot<Vals>(
    indices: ArrayD<i64>,
    depth: usize,
    off_value: &Vals,
    on_value: &Vals,
    axis: i64,
) -> Result<ArcArrayD<Vals>>
where
    Vals: Clone,
{
    // Compute output shape
    let input_rank = indices.ndim();
    let output_rank = input_rank + 1;
    let axis_pos = convert_negative_index(axis, output_rank);
    if axis_pos > input_rank {
        bail!(
            "Axis must be compatible with input rank ('{}'). Found '{}'.",
            input_rank,
            axis
        );
    }

    let dims = indices.shape().iter();

    // Put new axis at the end for now
    let out_shape: Vec<_> = dims.chain([depth].iter()).cloned().collect();

    let flat_indices = indices.iter();
    let mut out_2d = Array2::from_elem((indices.len(), depth), off_value.clone());
    for (idx, mut out_slice) in flat_indices.zip(out_2d.axis_iter_mut(Axis(0))) {
        let val = convert_negative_index(*idx, depth);
        if let Some(el) = out_slice.get_mut([val]) {
            *el = on_value.clone();
        }
    }

    // Reshape and rearrange axes if necessary
    if axis != -1 {
        let output_rank = input_rank + 1;
        let mut tmp = out_2d.into_shape(out_shape)?.insert_axis(Axis(axis_pos));
        tmp.swap_axes(axis_pos, output_rank);
        let tmp = tmp.remove_axis(Axis(output_rank));
        Ok(tmp.into_dyn().into())
    } else {
        Ok(out_2d.into_shape(out_shape)?.into_dyn().into())
    }
}
