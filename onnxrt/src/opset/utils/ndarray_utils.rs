//! Utils to help with the tensor operation of an operation
//! This module is not concerned with the monomorphization ("setup")
//! step.
use anyhow::{Result, bail};
use itertools::{EitherOrBoth, Itertools};
use ndarray::{ArrayD, ArrayViewD, Dimension, IntoDimension, Zip};

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
pub fn co_broadcast<D1, D2>(shape1: &D1, shape2: &D2) -> Result<Vec<usize>>
where
    D1: Dimension,
    D2: Dimension,
{
    let mut out = vec![0; shape1.ndim().max(shape2.ndim())];
    for (item, out) in shape1
        .as_array_view()
        .iter()
        .rev()
        .zip_longest(shape2.as_array_view().iter().rev())
        .zip(out.iter_mut().rev())
    {
        *out = match item {
            EitherOrBoth::Both(l, r) => {
                if (l == r) | (*l == 1) {
                    *r
                } else if *r == 1 {
                    *l
                } else {
                    bail!(
                        "cannot broadcast shapes `{}` and `{}`",
                        shape1.as_array_view(),
                        shape1.as_array_view()
                    )
                }
            }
            EitherOrBoth::Left(v) | EitherOrBoth::Right(v) => *v,
        };
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
    let common_shape = co_broadcast(&a.shape().into_dimension(), &b.shape().into_dimension())?;
    // Unwrap is fine below since we just computed the common shape above.
    Ok(Zip::from(a.broadcast(common_shape).unwrap())
        .and_broadcast(b)
        .map_collect(op))
}

pub fn zip_map_with_co_broadcast3<A, B, C, Out, Op>(
    a: ArrayViewD<A>,
    b: ArrayViewD<B>,
    c: ArrayViewD<C>,
    op: Op,
) -> Result<ArrayD<Out>>
where
    A: Clone,
    B: Clone,
    C: Clone,
    Op: Fn(&A, &B, &C) -> Out + Send + Sync + 'static + Copy,
{
    let common_shape_a_b = co_broadcast(&a.shape().into_dimension(), &b.shape().into_dimension())?;
    let common_shape = co_broadcast(
        &common_shape_a_b.into_dimension(),
        &c.shape().into_dimension(),
    )?;
    // Unwrap is fine below since we just computed the common shape above.
    Ok(Zip::from(a.broadcast(common_shape).unwrap())
        .and_broadcast(b)
        .and_broadcast(c)
        .map_collect(op))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn co_broadcast() {
        let a = array![[1, 1, 1], [1, 1, 1], [1, 1, 1]].into_dyn();
        let b = array![1, 1, 1].into_dyn();
        let res = zip_map_with_co_broadcast((&a).into(), (&b).into(), |el_a, el_b| *el_a - *el_b)
            .unwrap();
        let exp = array![[0, 0, 0], [0, 0, 0], [0, 0, 0]].into_dyn();
        assert_eq!(res, exp)
    }
}
