use crate::opset::utils::*;
use ndarray::ArrayViewD;

use num::{Float, One, Zero};

pub fn setup_matmul(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs: HomogeneousNumericFixed<2> = ctx.subscribe_to_inputs().narrow()?;

    // TODO: broadcast check
    use HomogeneousNumericFixed::*;
    match inputs {
        F32([a, b]) => setup((a, b), |(a, b)| {
            if a.ndim() == 2 && b.ndim() == 1 {
                matmul_2d_1d(a.view(), b.view())
            } else if a.ndim() == 2 && b.ndim() == 2 {
                matmul_2d(a.view(), b.view())
            } else if a.ndim() != b.ndim() {
                bail!(
                    "MatMul: Inputs must have equal rank. Found shapes a: '{:?}', b: '{:?}'",
                    a.shape(),
                    b.shape()
                )
            } else {
                matmul_nd(a.view(), b.view())
            }
        }),
        F64([a, b]) => setup((a, b), |(a, b)| {
            if a.ndim() == 2 && b.ndim() == 1 {
                matmul_2d_1d(a.view(), b.view())
            } else if a.ndim() == 2 && b.ndim() == 2 {
                matmul_2d(a.view(), b.view())
            } else if a.ndim() != b.ndim() {
                bail!(
                    "MatMul: Inputs must have equal rank. Found shapes a: '{:?}', b: '{:?}'",
                    a.shape(),
                    b.shape()
                )
            } else {
                matmul_nd(a.view(), b.view())
            }
        }),
        _ => bail!("MatMul: Not implemented for integer types."),
    }
}

fn matmul_2d_1d<T>(a: ArrayViewD<'_, T>, b: ArrayViewD<'_, T>) -> Result<ArrayD<T>>
where
    T: Clone + Float + One + Zero + 'static,
{
    let a: ArrayView2<_> = a.into_dimensionality()?;
    let b: ArrayView1<_> = b.into_dimensionality()?;

    let c_shape = [a.shape()[0]];
    let mut c = Array1::zeros(c_shape);

    ndarray::linalg::general_mat_vec_mul(T::one(), &a, &b, T::zero(), &mut c);

    Ok(c.into_dyn())
}

fn matmul_2d<T>(a: ArrayViewD<'_, T>, b: ArrayViewD<'_, T>) -> Result<ArrayD<T>>
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

fn matmul_nd<T>(a: ArrayViewD<'_, T>, b: ArrayViewD<'_, T>) -> Result<ArrayD<T>>
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

fn to_3d<T>(a: ArrayViewD<'_, T>) -> Result<Array3<T>>
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
