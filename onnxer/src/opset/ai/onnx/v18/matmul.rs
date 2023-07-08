use crate::opset::utils::*;
use ndarray::ArrayViewD;

use num::{Float, One, Zero};

pub fn setup_matmul(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs: HomogeneousNumericFixed<2> = ctx.subscribe_to_inputs().narrow()?;

    // TODO: broadcast check
    use HomogeneousNumericFixed::*;
    match inputs {
        F32([a, b]) => setup((a, b), |(a, b)| {
            if a.ndim() != b.ndim() {
                bail!("MatMul: Inputs must have equal rank")
            }
            if a.ndim() > 2 {
                Ok::<ArcArrayD<_>, _>(matmul_nd(a.view(), b.view())?.into())
            } else {
                Ok::<ArcArrayD<_>, _>(matmul_2d(a.view(), b.view())?.into())
            }
        }),
        F64([a, b]) => setup((a, b), |(a, b)| {
            if a.ndim() != b.ndim() {
                bail!("MatMul: Inputs must have equal rank")
            }
            if a.ndim() > 2 {
                Ok::<ArcArrayD<_>, _>(matmul_nd(a.view(), b.view())?.into())
            } else {
                Ok::<ArcArrayD<_>, _>(matmul_2d(a.view(), b.view())?.into())
            }
        }),
        _ => bail!("MatMul: Not implemented for integer types."),
    }
}

fn matmul_2d<'a, 'b, T>(a: ArrayViewD<'a, T>, b: ArrayViewD<'b, T>) -> Result<ArrayD<T>>
where
    T: Clone + Float + One + Zero + 'static,
{
    let a: ArrayView2<_> = a.into_dimensionality()?;
    let b: ArrayView2<_> = b.into_dimensionality()?;

    let c_shape = [a.shape()[0], b.shape()[1]];
    let mut c = Array2::zeros(c_shape);

    ndarray::linalg::general_mat_mul(T::one(), &a, &b, T::zero(), &mut c);

    Ok(c.into_dyn())
}

fn matmul_nd<'a, 'b, T>(a: ArrayViewD<'a, T>, b: ArrayViewD<'b, T>) -> Result<ArrayD<T>>
where
    T: Clone + Float + One + Zero + 'static,
{
    let out_shape: Vec<_> = a.shape()[..a.ndim() - 1]
        .iter()
        .chain(&[b.shape()[b.ndim() - 1]])
        .cloned()
        .collect();
    let a = to_3d(a)?;
    let b = to_3d(b)?;

    if a.shape()[0] != b.shape()[0] {
        bail!("MatMul: Broadcasting is not supported, yet");
    }
    let c_shape = [a.shape()[0], a.shape()[1], b.shape()[2]];
    let mut c = ndarray::Array3::zeros(c_shape);

    for ((a, b), mut c) in a
        .axis_iter(Axis(0))
        .zip(b.axis_iter(Axis(0)))
        .zip(c.axis_iter_mut(Axis(0)))
    {
        ndarray::linalg::general_mat_mul(T::one(), &a, &b, T::zero(), &mut c);
    }

    Ok(c.into_shape(out_shape.as_slice())?)
}

fn to_3d<'inp, T>(a: ArrayViewD<'inp, T>) -> Result<Array3<T>>
where
    T: Clone,
{
    let ndim = a.ndim();
    let [m, n]: [usize; 2] = a.shape()[(ndim - 2)..].try_into()?;
    let l = a.len() / (m * n);
    let a: ndarray::Array3<_> = a
        .into_shape(&[l, m, n][..])?
        .to_owned()
        .into_dimensionality()?;
    Ok(a)
}
