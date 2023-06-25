use super::*;

pub fn setup_one_hot(ctx: OpContext) -> Result<ExecutionStep> {
    let axis = ctx.attrs.get_optional("axis")?;

    let [indices, depth, values]: [_; 3] = ctx.subscribe_to_inputs().try_into().unwrap();

    // Cast other types to i64 (as stated in the standard...)
    let map_err = |e| format!("Failed to receive value: {:?}", e);
    let recv_depth: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send> = match depth {
        TaggedReceiver::F32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        TaggedReceiver::I32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        TaggedReceiver::F64(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        TaggedReceiver::I64(ch) => Box::new(move || ch.recv().map_err(map_err)),
        _ => bail!("Unsupported type for 'depth' input."),
    };
    let recv_indices: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send> = match indices {
        TaggedReceiver::F32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        TaggedReceiver::I32(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        TaggedReceiver::F64(ch) => {
            Box::new(move || Ok(ch.recv().map_err(map_err)?.mapv(|el| el as i64).into()))
        }
        TaggedReceiver::I64(ch) => Box::new(move || ch.recv().map_err(map_err)),
        _ => bail!("Unsupported type for 'depth' input."),
    };

    match values {
        TaggedReceiver::F32(values) => match_arm(recv_indices, recv_depth, values, axis),
        TaggedReceiver::F64(values) => match_arm(recv_indices, recv_depth, values, axis),
        TaggedReceiver::I32(values) => match_arm(recv_indices, recv_depth, values, axis),
        TaggedReceiver::I64(values) => match_arm(recv_indices, recv_depth, values, axis),
        _ => bail!("Values data type is not implemented: '{:?}'", values),
    }
}

fn match_arm<Vals>(
    recv_indices: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send>,
    recv_depth: Box<dyn Fn() -> Result<ArcArrayD<i64>, String> + Send>,
    values: Receiver<ArcArrayD<Vals>>,
    axis: Option<i64>,
) -> Result<ExecutionStep>
where
    ArcArrayD<Vals>: Dispatchable,
    Vals: 'static + Copy + std::fmt::Debug + Sync + Send,
{
    let (sdr, rec) = channel();
    let fun = Box::new(move || {
        let indices = recv_indices().map_err(|e| anyhow!("{}", e))?;
        let depth = recv_depth().map_err(|e| anyhow!("{}", e))?;
        let values = values.recv().map_err(|e| anyhow!("{}", e))?;
        let [off, on]: [Vals; 2] = values.into_iter().collect::<Vec<_>>().try_into().unwrap();
        let [depth]: [_; 1] = depth.into_iter().collect::<Vec<_>>().try_into().unwrap();
        let res: ArcArrayD<Vals> = one_hot(indices, depth.try_into()?, off, on, axis)?;
        sdr.send(res)?;
        Ok(())
    });
    Ok(ExecutionStep {
        fun,
        outputs: vec![rec],
    })
}

fn one_hot<Vals>(
    indices: ArcArrayD<i64>,
    depth: usize,
    off_value: Vals,
    on_value: Vals,
    axis: Option<i64>,
) -> Result<ArcArrayD<Vals>>
where
    Vals: Copy + std::fmt::Debug,
{
    // Compute output shape
    let input_rank = indices.ndim();
    let output_rank = input_rank + 1;
    let axis = axis.unwrap_or(-1);
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
    let mut out_2d = Array2::from_elem((indices.len(), depth), off_value);
    for (idx, mut out_slice) in flat_indices.zip(out_2d.axis_iter_mut(Axis(0))) {
        let val = convert_negative_index(*idx, depth);
        if let Some(el) = out_slice.get_mut([val]) {
            *el = on_value;
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
