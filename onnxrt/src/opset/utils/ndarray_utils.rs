//! Utils to help with the tensor operation of an operation
//! This module is not concerned with the monomophization ("setup")
//! step.
use anyhow::{bail, Result};
use itertools::izip;
use ndarray::{ArrayD, ArrayViewD, Dimension, IntoDimension, IxDyn, Zip};

/// Index types as used in the ONNX standard. Namely i32 and
/// i64. Negative values are counted from the end of a dimension's
/// length.
pub trait Index {
    fn for_len(&self, len: usize) -> usize;
}

impl Index for i32 {
    fn for_len(&self, len: usize) -> usize {
        (if *self < 0 {
            len as Self + *self
        } else {
            *self
        }) as _
    }
}

impl Index for i64 {
    fn for_len(&self, len: usize) -> usize {
        (if *self < 0 {
            len as Self + *self
        } else {
            *self
        }) as _
    }
}

/// Convert a negative index to a positive one under the assumption
/// that it applies to a dimension of length `len`.
pub fn convert_negative_index(idx: i64, len: usize) -> usize {
    let len = len as i64;
    (if idx < 0 { len + idx } else { idx } as usize)
}

/// Calculate the common shape for a pair of array shapes, that they can be broadcasted
/// to. Return an error if the shapes are not compatible.
///
/// Uses the [NumPy broadcasting rules]
//  (https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).
pub fn co_broadcast<D1, D2, Output>(shape1: &D1, shape2: &D2) -> Result<Output>
where
    D1: Dimension,
    D2: Dimension,
    Output: Dimension,
{
    let (k, overflow) = shape1.ndim().overflowing_sub(shape2.ndim());
    // Swap the order if d2 is longer.
    if overflow {
        return co_broadcast::<D2, D1, Output>(shape2, shape1);
    }
    // The output should be the same length as shape1.
    let mut out = Output::zeros(shape1.ndim());
    for (out, s) in izip!(out.slice_mut(), shape1.slice()) {
        *out = *s;
    }
    for (out, s2) in izip!(&mut out.slice_mut()[k..], shape2.slice()) {
        if *out != *s2 {
            if *out == 1 {
                *out = *s2
            } else if *s2 != 1 {
                bail!("Incompatible shapes")
            }
        }
    }
    Ok(out)
}

pub fn zip_map_with_co_broadcast<A, B, Out, Op>(
    a: ArrayViewD<A>,
    b: ArrayViewD<B>,
    op: Op,
) -> Result<ArrayD<Out>>
where
    A: Clone,
    B: Clone,
    Op: Fn(&A, &B) -> Out + Send + Sync + 'static + Copy,
{
    let common_shape: IxDyn =
        co_broadcast(&a.shape().into_dimension(), &b.shape().into_dimension())?;
    // Unwrap is fine below since we just computed the common shape above.
    Ok(Zip::from(a.broadcast(common_shape).unwrap())
        .and_broadcast(b)
        .map_collect(op))
}

#[cfg(tests)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn co_broadcast() {
        let a = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]].into_dyn();
        let b = array![1, 1, 1].into_dyn();
        let res = zip_map_with_co_broadcast(a.into(), b.into(), |el_a, el_b| *el_a - *el_b);
        let exp = array![[0, 0, 0], [0, 0, 0], [0, 0, 0]].into_dyn();
        assert_eq!(res, exp)
    }
}
