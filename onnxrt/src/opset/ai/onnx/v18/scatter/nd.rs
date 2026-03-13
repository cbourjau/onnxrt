use ndarray::{SliceInfo, SliceInfoElem};

use super::reduction::Reduction;
use crate::opset::utils::*;

fn scatter_nd<T, Idx, F>(
    mut data: ArrayD<T>,
    indices: ArrayViewD<Idx>,
    updates: ArrayViewD<T>,
    reduction: F,
) -> Result<ArrayD<T>>
where
    T: Clone + Default,
    Idx: Index + Clone,
    F: Fn(&mut T, T),
{
    validate(&data, &indices, &updates)?;

    // Flatten indices into a 2D matrix of shape (-1, k)
    let k = *indices.shape().last().unwrap();
    let flat_shape_idxs = [indices.shape()[..indices.ndim() - 1].iter().product(), k];
    let indices = indices.as_standard_layout().into_shape(flat_shape_idxs)?;

    // Flatten the leading dimensions of `updates` accordingly
    let q = indices.ndim();
    let flat_shape_updates: Vec<_> = [flat_shape_idxs[0]]
        .iter()
        .chain(updates.shape()[(q - 1)..].iter())
        .copied()
        .collect();
    let updates = updates
        .as_standard_layout()
        .into_shape(flat_shape_updates.as_slice())?;

    let leftover_ndims = updates.shape()[(q - 1)..].len();

    for (idx_lane, update_view) in indices.axis_iter(Axis(0)).zip(updates.axis_iter(Axis(0))) {
        let slices: Vec<_> = idx_lane
            .iter()
            .zip(data.shape())
            .map(|(idx, dim_len)| idx.for_len(*dim_len))
            .map(|idx| SliceInfoElem::Index(idx as isize))
            .chain(std::iter::repeat_n((0..).into(), leftover_ndims))
            .collect();
        let slice_info: SliceInfo<_, IxDyn, IxDyn> = slices.try_into()?;
        data.slice_mut(slice_info)
            .zip_mut_with(&update_view, |a, b| reduction(a, b.clone()));
    }

    Ok(data)
}

fn validate<T, Idx>(
    data: &ArrayD<T>,
    indices: &ArrayViewD<Idx>,
    updates: &ArrayViewD<T>,
) -> Result<()> {
    // Validate ranks
    let r = data.ndim();
    if r < 1 {
        bail!("ScatterND: 'data' must be of rank >= 1")
    }
    let q = indices.ndim();
    if q < 1 {
        bail!("ScatterND: 'indices' must be of rank >= 1")
    }
    let k = *indices.shape().last().unwrap();
    let u = updates.ndim();

    if q + r - k - 1 != u {
        bail!("ScatterND: 'update' has invalid rank {}", u)
    }

    if updates.shape()[..q - 1] != indices.shape()[..q - 1] {
        bail!("ScatterND: 'updates' and 'indices' have incompatible shapes.")
    }
    if updates.shape()[(q - 1)..] != data.shape()[k..] {
        bail!("ScatterND: 'updates' and 'data' have incompatible shapes.")
    }
    Ok(())
}

#[derive(Debug)]
struct ScatterNd {
    reduction: Reduction,
}

impl ScatterNd {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let reduction = attrs
            .get_with_default("reduction", "none".to_string())?
            .parse()?;

        Ok(Self { reduction })
    }
}

pub fn setup_scatter_nd(ctx: OpContext) -> Result<ExecutionStep> {
    let state = ScatterNd::from_attrs(&ctx.attrs)?;
    let (data, indices, updates) = ctx.subscribe_to_inputs().narrow()?;

    let data_and_updates: HomogeneousFixed<2> = vec![data, updates].narrow()?;

    if state.reduction != Reduction::None
        && let HomogeneousFixed::String(_) | HomogeneousFixed::Bool(_) = data_and_updates
    {
        bail!("ScatterND: 'reduction' is only supported for numeric types")
    }

    macro_rules! match_arm {
        ($data:tt, $indices:tt, $updates:tt, $reduction:expr) => {
            setup(
                ($data, $indices, $updates),
                move |(data, indices, updates)| {
                    scatter_nd(data.to_owned(), indices.view(), updates.view(), $reduction)
                },
            )
        };
    }

    use HomogeneousFixed::*;
    // The bool and string case are special cases which prohibit us from using setup_all...
    match indices {
        TaggedTensorReceiver::I32(indices) => match data_and_updates {
            U8([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            U16([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            U32([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            U64([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I8([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I16([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I32([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I64([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            F32([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            F64([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            String([data, updates]) => {
                match_arm!(data, indices, updates, |a, b| {
                    *a = b;
                })
            }
            Bool([data, updates]) => {
                match_arm!(data, indices, updates, |a, b| {
                    *a = b;
                })
            }
        },
        TaggedTensorReceiver::I64(indices) => match data_and_updates {
            U8([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            U16([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            U32([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            U64([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I8([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I16([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I32([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            I64([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            F32([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            F64([data, updates]) => {
                match_arm!(data, indices, updates, state.reduction.as_function())
            }
            String([data, updates]) => {
                match_arm!(data, indices, updates, |a, b| {
                    *a = b;
                })
            }
            Bool([data, updates]) => {
                match_arm!(data, indices, updates, |a, b| {
                    *a = b;
                })
            }
        },
        _ => bail!("ScatterND: 'indices' must be of type tensor(int32) or tensor(int64)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example1() {
        let data = array![1, 2, 3, 4, 5, 6, 7, 8].into_dyn();
        let indices = array![[4], [3], [1], [7]].into_dyn();
        let updates = array![9, 10, 11, 12].into_dyn();
        let exp = array![1, 11, 3, 10, 9, 6, 7, 12].into_dyn();
        let obs = scatter_nd(data, indices.view(), updates.view(), |a, b| *a = b).unwrap();
        assert_eq!(exp, obs);
    }

    #[test]
    fn example2() {
        let data = array![
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]
        ]
        .into_dyn();
        let indices = array![[0], [2]].into_dyn();
        let updates = array![
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
        ]
        .into_dyn();
        let exp = array![
            [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        ]
        .into_dyn();
        let obs = scatter_nd(data, indices.view(), updates.view(), |a, b| *a = b).unwrap();
        assert_eq!(exp, obs);
    }
}
