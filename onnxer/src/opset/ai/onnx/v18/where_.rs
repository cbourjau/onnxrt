use super::*;

pub fn setup_where(ctx: OpContext) -> Result<ExecutionStep> {
    let [cond_ch, x_ch, y_ch]: [_; 3] = ctx.input_channels().try_into().unwrap();

    // Condition must be of type bool
    let cond_ch = if let TaggedReceiver::Bool(cond_ch) = cond_ch {
        cond_ch
    } else {
        bail!("Conditional tensor must be of type `bool`.")
    };

    match (x_ch, y_ch) {
        (TaggedReceiver::F32(x_ch), TaggedReceiver::F32(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (TaggedReceiver::F64(x_ch), TaggedReceiver::F64(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (TaggedReceiver::I32(x_ch), TaggedReceiver::I32(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (TaggedReceiver::I64(x_ch), TaggedReceiver::I64(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (TaggedReceiver::Bool(x_ch), TaggedReceiver::Bool(y_ch)) => match_arm(cond_ch, x_ch, y_ch),
        (TaggedReceiver::String(x_ch), TaggedReceiver::String(y_ch)) => {
            match_arm(cond_ch, x_ch, y_ch)
        }
        _ => bail!("`X` and `Y` must be of identical type."),
    }
}

fn match_arm<T>(
    cond_ch: Receiver<ArcArrayD<bool>>,
    x_ch: Receiver<ArcArrayD<T>>,
    y_ch: Receiver<ArcArrayD<T>>,
) -> Result<ExecutionStep>
where
    ArcArrayD<T>: Dispatchable,
    T: Sync + Send + 'static + Clone,
{
    let (sdr, rec) = channel();
    let fun = Box::new(move || {
        let (cond, x, y) = (cond_ch.recv()?, x_ch.recv()?, y_ch.recv()?);
        sdr.send(where_(cond, x, y)?)?;
        Ok(())
    });
    Ok(ExecutionStep {
        fun,
        outputs: vec![rec],
    })
}

fn where_<T>(cond: ArcArrayD<bool>, x: ArcArrayD<T>, y: ArcArrayD<T>) -> Result<ArcArrayD<T>>
where
    T: Clone,
{
    let common_shape: IxDyn = {
        let com1: IxDyn =
            co_broadcast(&cond.shape().into_dimension(), &x.shape().into_dimension())?;
        co_broadcast(&com1, &y.shape().into_dimension())?
    };
    Ok(Zip::from(cond.view().broadcast(common_shape).unwrap())
        .and_broadcast(x.view())
        .and_broadcast(y.view())
        .map_collect(|&cond, x, y| if cond { x.clone() } else { y.clone() })
        .into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_where() {
        let cond = array![[true, false], [true, true]].into_dyn().into();
        let x = array![[1, 2], [3, 4]].into_dyn().into();
        let y = array![[9, 8], [7, 6]].into_dyn().into();
        let expected: ArcArrayD<_> = array![[1, 8], [3, 4]].into_dyn().into();
        assert_eq!(where_(cond, x, y).unwrap(), expected);
    }
}
