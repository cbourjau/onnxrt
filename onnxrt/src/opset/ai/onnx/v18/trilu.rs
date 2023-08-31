use num::Zero;

use crate::opset::utils::*;

fn trilu<T>(x: ArrayD<T>, k: isize, retain_upper: bool) -> Result<ArrayD<T>>
where
    T: Clone + Zero + Copy + std::fmt::Debug,
{
    let input_shape = x.shape().to_vec();
    let ndim = x.ndim();
    let first_dim = input_shape.iter().take(ndim - 2).product();

    let (max_i, max_j) = (input_shape[ndim - 2], input_shape[ndim - 1]);
    let shape_3d = [first_dim, max_i, max_j];

    let mut x = x.into_shape(shape_3d)?;

    if retain_upper {
        // Fill everything below the diagonal
        for mut view in x.axis_iter_mut(Axis(0)) {
            for (i, mut lane) in view.lanes_mut(Axis(1)).into_iter().enumerate() {
                for elem in lane.iter_mut().take((i as isize + k).max(0) as _) {
                    *elem = T::zero();
                }
            }
        }
    } else {
        // Fill everything above the diagonal
        for mut view in x.axis_iter_mut(Axis(0)) {
            for (i, mut lane) in view.lanes_mut(Axis(1)).into_iter().enumerate() {
                for elem in lane.iter_mut().skip((i as isize + k + 1).max(0) as usize) {
                    *elem = T::zero();
                }
            }
        }
    }
    Ok(x.into_shape(input_shape)?)
}

#[derive(Debug)]
struct Trilu {
    upper: bool,
}

impl Trilu {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let upper = attrs.get_with_default::<i64>("upper", 1)? != 0;
        Ok(Self { upper })
    }
}

pub fn setup_trilu(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Trilu::from_attrs(&ctx.attrs)?;

    // SPEC: Standard claims that this should work for non-numeric
    // types, too. But does not say how.
    let (x, k): (Numeric, Option<Receiver<ArcArrayD<i64>>>) =
        ctx.input_channels_with_optional().narrow()?;

    match k {
        Some(k) => {
            setup_all!(
                Numeric,
                x,
                x,
                setup((x, k), move |(x, k)| {
                    let k = k.into_shape(())?.to_owned().into_scalar() as isize;
                    Ok::<ArcArrayD<_>, _>(trilu(x.into_owned(), k, state.upper)?.into())
                })
            )
        }
        None => setup_all!(
            Numeric,
            x,
            x,
            setup((x,), move |(x,)| {
                let k = 0;
                Ok::<ArcArrayD<_>, _>(trilu(x.into_owned(), k, state.upper)?.into())
            })
        ),
    }
}

#[cfg(test)]
mod test {
    use std::assert_eq;

    use super::*;

    macro_rules! test_trilu {
        ($name:ident, $k:expr, $upper:expr, $exp:expr) => {
            #[test]
            fn $name() {
                #[rustfmt::skip]
                let arr = array![
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ].into_dyn();
                #[rustfmt::skip]

                assert_eq!($exp.into_dyn(), trilu(arr, $k, $upper).unwrap());
            }
        };
    }

    #[rustfmt::skip]
    test_trilu!(
        k_zero_retain_lower,
        0,
        false,
        array![
            [1, 0, 0],
            [4, 5, 0],
            [7, 8, 9],
        ]
    );

    #[rustfmt::skip]
    test_trilu!(
        k_one_retain_lower,
        1,
        false,
        array![
            [1, 2, 0],
            [4, 5, 6],
            [7, 8, 9],
        ]
    );

    #[rustfmt::skip]
    test_trilu!(
        k_minus_one_retain_lower,
        -1,
        false,
        array![
            [0, 0, 0],
            [4, 0, 0],
            [7, 8, 0],
        ]
    );

    #[rustfmt::skip]
    test_trilu!(
        k_zero_retain_upper,
        0,
        true,
        array![
            [1, 2, 3],
            [0, 5, 6],
            [0, 0, 9],
        ]
    );

    #[rustfmt::skip]
    test_trilu!(
        k_one_retain_upper,
        1,
        true,
        array![
            [0, 2, 3],
            [0, 0, 6],
            [0, 0, 0],
        ]
    );

    #[rustfmt::skip]
    test_trilu!(
        k_minus_one_retain_upper,
        -1,
        true,
        array![
            [1, 2, 3],
            [4, 5, 6],
            [0, 8, 9],
        ]
    );
}
