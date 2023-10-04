use std::{cmp::Ordering, collections::BTreeMap, hash::Hash};

use ordered_float::OrderedFloat;

use crate::opset::utils::*;

#[derive(Debug)]
struct Unique {
    axis: Option<i64>,
    sorted: bool,
}

impl Unique {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let axis = attrs.get_optional("axis")?;
        let sorted = attrs.get_with_default("sorted", 1i64)? == 1;
        Ok(Self { axis, sorted })
    }

    #[allow(clippy::type_complexity)]
    fn compute<T>(
        &self,
        (x,): (ArcArrayD<T>,),
    ) -> Result<(ArcArrayD<T>, ArcArrayD<i64>, ArcArrayD<i64>, ArcArrayD<i64>)>
    where
        T: Hash + Eq + Clone + std::fmt::Debug + Ord,
    {
        let (axis, data) = if let Some(axis) = self.axis {
            let ndim = x.shape().len();
            let axis = convert_negative_index(axis, ndim);
            (Axis(axis), x)
        } else {
            let x_n_elems = x.len();
            (Axis(0), x.into_shape((x_n_elems,))?.into_dyn())
        };

        // Map to occurrences (indices) in X
        let mut map: BTreeMap<_, Vec<usize>> = BTreeMap::new();

        for (idx, view) in data.axis_iter(axis).enumerate() {
            let wrapped = OrderedArray(view);
            if let Some(occurences_in_x) = map.get_mut(&wrapped) {
                occurences_in_x.push(idx);
                continue;
            }

            map.insert(wrapped.clone(), vec![idx]);
        }
        let (keys, values) = {
            if self.sorted {
                // The BTreeMap already has the sorting we want
                let (keys, values): (Vec<_>, Vec<_>) = map.into_iter().unzip();
                (keys, values)
            } else {
                // Sort by the occurrence in input X
                let mut kvs: Vec<_> = map.into_iter().collect();
                kvs.sort_unstable_by_key(|(_, occus_in_x)| occus_in_x[0]);
                kvs.into_iter().unzip()
            }
        };

        let (indices, counts): (Vec<_>, Vec<_>) = values
            .iter()
            .map(|vec| (vec[0] as i64, vec.len() as i64))
            .unzip();

        let inverse_indices = {
            let mut inverse_indices = vec![0; data.shape()[axis.0]];
            for (i, inverse_indices_this_el) in values.iter().enumerate() {
                for inverse_idx in inverse_indices_this_el {
                    inverse_indices[*inverse_idx] = i as i64;
                }
            }
            inverse_indices
        };

        let unique_values = ndarray::stack(
            axis,
            keys.iter()
                .map(|wrapped| wrapped.0.clone())
                .collect::<Vec<_>>()
                .as_slice(),
        )?
        .into_dyn()
        .into();

        let indices = ndarray::Array::from_vec(indices).into_dyn().into();
        let inverse_indices = ndarray::Array::from_vec(inverse_indices).into_dyn().into();
        let counts = ndarray::Array::from_vec(counts).into_dyn().into();
        Ok((unique_values, indices, inverse_indices, counts))
    }
}

#[derive(Eq, PartialEq, Clone, Debug)]
struct OrderedArray<'a, T>(ndarray::ArrayViewD<'a, T>);

impl<'a, T> PartialOrd for OrderedArray<'a, T>
where
    T: Eq + Ord,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<'a, T> Ord for OrderedArray<'a, T>
where
    T: Eq + Ord,
{
    fn cmp(&self, other: &Self) -> Ordering {
        for (ela, elb) in self.0.iter().zip(other.0.iter()) {
            let res = ela.cmp(elb);
            if res != Ordering::Equal {
                return res;
            }
        }
        Ordering::Equal
    }
}

pub fn setup_unique(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Unique::from_attrs(&ctx.attrs)?;
    let (x,) = ctx.subscribe_to_inputs().narrow()?;

    use TaggedTensorReceiver::*;
    match x {
        F32(rec) => setup((rec,), move |(x,)| {
            let (a, b, c, d) = state.compute((x.mapv(OrderedFloat).into(),))?;
            let a: ArcArrayD<_> = a.mapv(|el| el.0).into();
            Ok((a, b, c, d))
        }),
        F64(rec) => setup((rec,), move |(x,)| {
            let (a, b, c, d) = state.compute((x.mapv(OrderedFloat).into(),))?;
            let a: ArcArrayD<_> = a.mapv(|el| el.0).into();
            Ok((a, b, c, d))
        }),

        I8(rec) => setup((rec,), move |x| state.compute(x)),
        I16(rec) => setup((rec,), move |x| state.compute(x)),
        I32(rec) => setup((rec,), move |x| state.compute(x)),
        I64(rec) => setup((rec,), move |x| state.compute(x)),

        U8(rec) => setup((rec,), move |x| state.compute(x)),
        U16(rec) => setup((rec,), move |x| state.compute(x)),
        U32(rec) => setup((rec,), move |x| state.compute(x)),
        U64(rec) => setup((rec,), move |x| state.compute(x)),

        Bool(rec) => setup((rec,), move |x| state.compute(x)),
        String(rec) => setup((rec,), move |x| state.compute(x)),
    }
}
