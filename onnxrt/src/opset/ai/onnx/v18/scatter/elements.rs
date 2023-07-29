use super::reduction::Reduction;
use crate::opset::utils::*;

/// Scatter elements from `updates` into `data` along a specified
/// `axis` based on the given `indices`.
/// The `reduction` function is used to combine the elements in `data`
/// with the corresponding elements in `updates`.
///
/// # Arguments
///
/// * `data` - A mutable n-dimensional array (`ArrayD`) containing
///   elements to be updated.
/// * `indices` - An n-dimensional view (`ArrayViewD`) of indices
///   specifying the positions where elements from `updates` should be
///   scattered into `data`.
/// * `updates` - An n-dimensional view (`ArrayViewD`) containing
///   elements to be scattered into `data`.
/// * `axis` - An `i64` value indicating the axis along which the
///   scatter operation should be performed.
/// * `reduction` - Closure for updating values in `data` with values
///   from `update`.
fn scatter_elements<T, Idx, F>(
    mut data: ArrayD<T>,
    indices: ArrayViewD<Idx>,
    updates: ArrayViewD<T>,
    axis: i64,
    reduction: F,
) -> Result<ArrayD<T>>
where
    T: Clone + std::fmt::Debug + Default,
    Idx: Index,
    F: Fn(&mut T, T),
{
    let axis = Axis(axis.for_len(data.ndim()));
    // let axis = Axis(convert_negative_index(axis, data.ndim()));
    let axis_data_len = data.shape()[axis.0];
    for ((mut data_lane, idx_lane), updates_lane) in data
        .lanes_mut(axis)
        .into_iter()
        .zip(indices.lanes(axis))
        .zip(updates.lanes(axis))
    {
        for (idx, update) in idx_lane.into_iter().zip(updates_lane.into_iter()) {
            let idx = idx.for_len(axis_data_len);
            let data_elem = data_lane.get_mut(idx).unwrap();
            reduction(data_elem, update.clone());
        }
    }
    Ok(data)
}

#[derive(Debug)]
struct ScatterElements {
    axis: i64,
    reduction: Reduction,
}

impl ScatterElements {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let axis = attrs.get_with_default("axis", 0)?;
        let reduction = attrs
            .get_with_default("reduction", "none".to_string())?
            .parse()?;

        Ok(Self { axis, reduction })
    }
}

pub fn setup_scatter_elements(ctx: OpContext) -> Result<ExecutionStep> {
    let state = ScatterElements::from_attrs(&ctx.attrs)?;
    let [data, indices, updates]: [TaggedReceiver; 3] = ctx
        .subscribe_to_inputs()
        .try_into()
        .map_err(|e| anyhow!("ScatterElements: Unexpected number of inputs: '{:?}'", e))?;

    let data_and_updates: HomogeneousFixed<2> = vec![data, updates].narrow()?;

    if state.reduction != Reduction::None {
        if let HomogeneousFixed::String(_) = data_and_updates {
            bail!("ScatterElements: 'reduction' is only supported for numeric types")
        }
    }

    macro_rules! match_arm {
        ($data:tt, $indices:tt, $updates:tt, $reduction:expr) => {
            setup(($data, $updates), move |(data, updates)| {
                let indices = $indices.recv()?;
                let out = scatter_elements(
                    data.to_owned(),
                    indices.view(),
                    updates.view(),
                    state.axis,
                    $reduction,
                )?;
                Ok::<ArcArrayD<_>, _>(out.into())
            })
        };
    }

    use HomogeneousFixed::*;
    match indices {
        TaggedReceiver::I32(indices) => match data_and_updates {
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
        TaggedReceiver::I64(indices) => match data_and_updates {
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
        _ => bail!("ScatterElements: 'indices' must be of type tensor(int32) or tensor(int64)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example1() {
        let data = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]].into_dyn();
        let indices = array![[1, 0, 2], [0, 2, 1]].into_dyn();
        let updates = array![[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]].into_dyn();
        let exp = array![[2.0, 1.1, 0.0], [1.0, 0.0, 2.2], [0.0, 2.1, 1.2]].into_dyn();

        let obs = scatter_elements(data, indices.view(), updates.view(), 0, |a, b| *a = b).unwrap();
        assert_eq!(exp, obs);
    }

    #[test]
    fn example2() {
        let data = array![[1.0, 2.0, 3.0, 4.0, 5.0]].into_dyn();
        let indices = array![[1, 3]].into_dyn();
        let updates = array![[1.1, 2.1]].into_dyn();
        let axis = 1;
        let exp = array![[1.0, 1.1, 3.0, 2.1, 5.0]].into_dyn();
        let obs =
            scatter_elements(data, indices.view(), updates.view(), axis, |a, b| *a = b).unwrap();
        assert_eq!(exp, obs);
    }
}
