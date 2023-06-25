use ndarray::{IntoDimension, Zip};

use crate::opset::utils::*;

pub fn setup_where(ctx: OpContext) -> Result<ExecutionStep> {
    let (cond_ch, data_chs): (Receiver<ArcArrayD<bool>>, _) = ctx.input_channels().narrow()?;

    setup_all!(
        HomogeneousFixed,
        data_chs,
        [x_ch, y_ch],
        setup((cond_ch, x_ch, y_ch), where_)
    )
}

fn where_<T>((cond, x, y): (ArcArrayD<bool>, ArcArrayD<T>, ArcArrayD<T>)) -> Result<ArcArrayD<T>>
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
        assert_eq!(where_((cond, x, y)).unwrap(), expected);
    }
}
