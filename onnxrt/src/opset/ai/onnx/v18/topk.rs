use std::{cmp::Reverse, collections::BinaryHeap};

use ordered_float::OrderedFloat;

use crate::opset::utils::*;

fn top_k<T, Idx>(
    x: ArrayViewD<T>,
    k: usize,
    axis: Idx,
    largest: bool,
    sorted: bool,
) -> Result<(ArrayD<T>, ArrayD<usize>)>
where
    T: Clone + Default + Ord + Copy,
    Idx: Index,
{
    let axis = axis.for_len(x.ndim());
    let k = x.shape()[axis].min(k);
    let x_shape = x.shape();
    let out_shape: Vec<_> = x_shape[..axis]
        .iter()
        .chain(&[k])
        .chain(&x_shape[(axis + 1)..])
        .copied()
        .collect();
    let mut result_values = ArrayD::default(out_shape.as_slice());
    let mut result_indices = ArrayD::default(out_shape.as_slice());

    let axis = Axis(axis);
    for ((x, res_vals), res_idxs) in x
        .lanes(axis)
        .into_iter()
        .zip(result_values.lanes_mut(axis))
        .zip(result_indices.lanes_mut(axis))
    {
        if largest {
            for (((val, idx), res_val), res_idx) in top_k_iter(x.iter().copied(), k, sorted)
                .zip(res_vals)
                .zip(res_idxs)
            {
                *res_val = val;
                *res_idx = idx;
            }
        } else {
            for (((val, idx), res_val), res_idx) in
                top_k_iter(x.iter().copied().map(Reverse), k, sorted)
                    .zip(res_vals)
                    .zip(res_idxs)
            {
                *res_val = val.0;
                *res_idx = idx;
            }
        }
    }
    Ok((result_values, result_indices))
}

/// Find top `k` values in the given slice.
fn top_k_iter<I, T>(x: I, k: usize, sorted: bool) -> impl Iterator<Item = (T, usize)>
where
    I: Iterator<Item = T>,
    T: Copy + Ord,
{
    let mut heap = BinaryHeap::new();
    heap.reserve_exact(k);
    for (idx, x) in x.enumerate() {
        if heap.len() < k {
            heap.push((Reverse(x), idx));
        } else {
            let mut elem = heap.peek_mut().unwrap();
            // Find the smallest element currently in the heap. It is
            // wrapped in a `Reverse`. If the current element `x` is
            // larger than the *unwrapped* element we replace it.
            if x > elem.0 .0 {
                *elem = (Reverse(x), idx);
            }
        }
    }

    let v = if sorted {
        heap.into_sorted_vec()
    } else {
        heap.into_vec()
    };
    v.into_iter().map(|(val, idx)| (val.0, idx))
}

#[derive(Debug)]
struct TopK {
    axis: i64,
    largest: bool,
    sorted: bool,
}

trait WrapForOrd {
    type Return;

    fn wrap(self) -> Self::Return;

    fn unwrap_ord(wrapped: Self::Return) -> Self;
}

macro_rules! noop_wraps {
    ($ty:ty) => {
        impl WrapForOrd for $ty {
            type Return = Self;
            fn wrap(self) -> Self::Return {
                self
            }
            fn unwrap_ord(wrapped: Self::Return) -> Self {
                wrapped
            }
        }
    };
}

noop_wraps!(i8);
noop_wraps!(i16);
noop_wraps!(i32);
noop_wraps!(i64);
noop_wraps!(u8);
noop_wraps!(u16);
noop_wraps!(u32);
noop_wraps!(u64);

impl WrapForOrd for f32 {
    type Return = OrderedFloat<Self>;
    fn wrap(self) -> Self::Return {
        OrderedFloat(self)
    }
    fn unwrap_ord(wrapped: Self::Return) -> Self {
        wrapped.0
    }
}

impl WrapForOrd for f64 {
    type Return = OrderedFloat<Self>;
    fn wrap(self) -> Self::Return {
        OrderedFloat(self)
    }
    fn unwrap_ord(wrapped: Self::Return) -> Self {
        wrapped.0
    }
}

impl TopK {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let axis = attrs.get_with_default("axis", -1)?;
        let largest = attrs.get_with_default("largest", 1)? != 0;
        let sorted = attrs.get_with_default("sorted", 1)? != 0;

        Ok(Self {
            axis,
            largest,
            sorted,
        })
    }

    fn compute<T>(
        &self,
        x: ArcArrayD<T>,
        k: ArrayViewD<i64>,
    ) -> Result<(ArcArrayD<T>, ArcArrayD<i64>)>
    where
        T: WrapForOrd + Clone,
        <T as WrapForOrd>::Return: Ord + Copy + Default,
    {
        let x = x.mapv(|el| el.wrap());
        let k = k.into_shape([])?.into_scalar();
        let k = (*k).try_into()?; // converter to usize
        let (vals, idxs) = top_k(x.view(), k, self.axis, self.largest, self.sorted)?;

        Ok((
            vals.mapv(|el| T::unwrap_ord(el)).into(),
            idxs.mapv(|el| el as i64).into(),
        ))
    }
}

pub fn setup_topk(ctx: OpContext) -> Result<ExecutionStep> {
    let state = TopK::from_attrs(&ctx.attrs)?;

    let (x, k): (_, Receiver<ArcArrayD<i64>>) = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        Numeric,
        x,
        x,
        setup((x, k), move |(x, k)| state.compute(x, k.view()))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_topk_iter() {
        assert_eq!(
            top_k_iter([3, 1, 2].into_iter(), 2, true).collect::<Vec<(i64, usize)>>(),
            vec![(3, 0), (2, 2)]
        )
    }
    #[test]
    fn simple_topk_largest() {
        let (vals, idxs) = top_k(array![3, 1, 2].into_dyn().view(), 2, 0, true, true).unwrap();
        assert_eq!(vals, array![3, 2].into_dyn());
        assert_eq!(idxs, array![0, 2].into_dyn());
    }
    #[test]
    fn simple_topk_smallest() {
        let (vals, idxs) = top_k(array![3, 1, 2].into_dyn().view(), 2, 0, false, true).unwrap();
        assert_eq!(vals, array![1, 2].into_dyn());
        assert_eq!(idxs, array![1, 2].into_dyn());
    }
}
