use crate::opset::utils::*;
use ndarray::{SliceInfo, SliceInfoElem};

struct GatherNd {
    batch_dims: usize,
}

impl FromAttributes for GatherNd {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let batch_dims = attrs.get_with_default("batch_dims", 0i64)?.try_into()?;
        Ok(Self { batch_dims })
    }
}

// Helper to create a `SliceInfoElem::Slice` that takes the entire dimension
fn full_slice() -> SliceInfoElem {
    (0..).into()
}

fn validation<T>(data: &ArrayViewD<T>, indices: &ArrayViewD<i64>, batch_dims: usize) -> Result<()> {
    let tuple_len = *indices.shape().last().unwrap();

    if tuple_len > (data.ndim() - batch_dims) {
        bail!("GatherND: Invalid input shapes")
    }

    if batch_dims > 1 {
        bail!("GatherND: 'batch_dims > 1' are not yet suppported. Please file an issue.")
    }

    if data.shape()[..batch_dims] != indices.shape()[..batch_dims] {
        bail!("GatherND: 'data' shape and 'indices' shape must match for all 'batch_dims'")
    }
    Ok(())
}

/// Output shape that always has exactly one batch dimension (possibly length 1).
fn normalized_output_shape<T>(
    data: &ArrayViewD<T>,
    indices: &ArrayViewD<i64>,
    batch_dims: usize,
) -> Vec<usize> {
    let tuple_len = *indices.shape().last().unwrap();

    // Shape of indices (i.e. more than 2D) are preserved
    [normalized_batch_dim(data, batch_dims)]
        .iter()
        .chain(indices.shape()[batch_dims..indices.ndim() - 1].iter())
        .copied()
        // Indices may point to views or scalars
        .chain(data.shape().iter().copied().skip(batch_dims + tuple_len))
        .collect()
}

/// Normalize the data shape to always have exactly one batch dimension
fn normalize_data_shape<T>(data: ArrayViewD<T>, batch_dims: usize) -> Result<ArrayViewD<T>>
where
    T: Clone + std::fmt::Debug,
{
    let data_shape_flat_batch: Vec<_> = [normalized_batch_dim(&data, batch_dims)]
        .iter()
        .chain(data.shape()[batch_dims..].iter())
        .copied()
        .collect();

    Ok(data.into_shape(data_shape_flat_batch.as_slice())?)
}

/// A single, normalized batch dim.
fn normalized_batch_dim<T>(data: &ArrayViewD<T>, batch_dims: usize) -> usize {
    data.shape()[..batch_dims].iter().product()
}

fn gather_nd<T>(
    data: ArrayViewD<T>,
    indices: ArrayViewD<i64>,
    batch_dims: usize,
) -> Result<ArrayD<T>>
where
    T: Clone + std::fmt::Debug + Default,
{
    validation(&data, &indices, batch_dims)?;
    let tuple_len = *indices.shape().last().unwrap();

    let gathered_element_n_dim = (data.ndim() - batch_dims) - tuple_len;

    let mut slice_info_elems: Vec<SliceInfo<_, IxDyn, IxDyn>> = vec![];

    for tuple in indices.lanes(Axis(indices.ndim() - 1)) {
        let mut slice_info_one_tuple = Vec::with_capacity(tuple.len());
        for (idx, data_dim) in tuple.into_iter().zip(data.shape()[batch_dims..].iter()) {
            let data_dim = *data_dim as isize;
            let idx = *idx as isize;
            if ((-data_dim)..data_dim).contains(&idx) {
                slice_info_one_tuple.push(SliceInfoElem::Index(idx));
            } else {
                bail!(
                    "GatherND: Index '{}' is out of range for axis of length '{}'",
                    idx,
                    data_dim
                )
            }
        }

        for _ in 0..gathered_element_n_dim {
            slice_info_one_tuple.push(full_slice());
        }

        slice_info_elems.push(slice_info_one_tuple.try_into()?);
    }

    if batch_dims > 0 {
        let mut out = ArrayD::<T>::default(normalized_output_shape(&data, &indices, batch_dims));

        // Normalize data such that the first dim is a single "batch dim"
        let data = normalize_data_shape(data, batch_dims)?;

        for ((mut out, data), slice_info) in out
            .axis_iter_mut(Axis(0))
            .zip(data.axis_iter(Axis(0)))
            .zip(slice_info_elems.iter())
        {
            out.assign(&data.slice(slice_info));
        }

        Ok(out)
    } else {
        let stack: Vec<_> = slice_info_elems.iter().map(|s| data.slice(s)).collect();

        let stacked = ndarray::stack(Axis(batch_dims), &stack)?.to_owned();
        let out_shape: Vec<_> = indices.shape()[..indices.ndim() - 1]
            .iter()
            .copied()
            // Indices may point to views or scalars
            .chain(data.shape().iter().copied().skip(batch_dims + tuple_len))
            .collect();

        Ok(stacked
            .as_standard_layout()
            .to_owned()
            .into_shape(out_shape)?)
    }
}

pub fn setup_gather_nd(ctx: OpContext) -> Result<ExecutionStep> {
    let state = GatherNd::from_attrs(&ctx.attrs)?;
    let (data, indices): (_, Receiver<ArcArrayD<i64>>) = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedReceiver,
        data,
        inp,
        setup((inp, indices), move |(inp, indices)| {
            gather_nd(inp.view(), indices.view(), state.batch_dims)
        })
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example1() {
        let batch_dims = 0;
        let data = array![[0, 1], [2, 3]].into_dyn();
        let indices = array![[0, 0], [1, 1]].into_dyn();
        let exp = array![0, 3].into_dyn();

        let observed = gather_nd(data.view(), indices.view(), batch_dims).unwrap();
        assert_eq!(exp, observed);
    }

    #[test]
    fn example2() {
        let batch_dims = 0;
        let data = array![[0, 1], [2, 3]].into_dyn();
        let indices = array![[1], [0]].into_dyn();
        let exp = array![[2, 3], [0, 1]].into_dyn();
        let observed = gather_nd(data.view(), indices.view(), batch_dims).unwrap();
        assert_eq!(exp, observed);
    }
    #[test]
    fn example3() {
        let batch_dims = 0;
        let data = array![[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into_dyn();
        let indices = array![[0, 1], [1, 0]].into_dyn();
        let exp = array![[2, 3], [4, 5]].into_dyn();
        let observed = gather_nd(data.view(), indices.view(), batch_dims).unwrap();
        assert_eq!(exp, observed);
    }

    #[test]
    fn example4() {
        let batch_dims = 0;
        let data = array![[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into_dyn();
        let indices = array![[[0, 1]], [[1, 0]]].into_dyn();
        let exp = array![[[2, 3]], [[4, 5]]].into_dyn();

        let observed = gather_nd(data.view(), indices.view(), batch_dims).unwrap();
        assert_eq!(exp, observed);
    }

    #[test]
    fn example5() {
        let batch_dims = 1;
        let data = array![[[0, 1], [2, 3]], [[4, 5], [6, 7]]].into_dyn();
        let indices = array![[1], [0]].into_dyn();
        let exp = array![[2, 3], [4, 5]].into_dyn();

        let observed = gather_nd(data.view(), indices.view(), batch_dims).unwrap();
        assert_eq!(exp, observed);
    }

    #[test]
    fn tf_batch_example() {
        let batch_dims = 1;
        let indices = array![[1], [0], [4], [2], [1]].into_dyn();
        let data = Array3::<i64>::zeros([5, 7, 3]).into_dyn();
        let exp_shape = [5, 3];

        let observed = gather_nd(data.view(), indices.view(), batch_dims).unwrap();
        assert_eq!(exp_shape, observed.shape());
    }

    #[test]
    fn tf_batch_example_written_out() {
        let batch_dims = 0;
        let indices = array![[0, 1], [1, 0], [2, 4], [3, 2], [4, 1]].into_dyn();
        let data = Array3::<i64>::zeros([5, 7, 3]).into_dyn();
        let exp_shape = [5, 3];

        let observed = gather_nd(data.view(), indices.view(), batch_dims).unwrap();
        assert_eq!(exp_shape, observed.shape());
    }
}
