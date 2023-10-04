use num::Integer;
use num_traits::AsPrimitive;

use crate::opset::utils::*;

#[derive(Debug)]
struct Gather {
    axis: i64,
}

impl Gather {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let axis = attrs.get_with_default("axis", 0)?;

        Ok(Self { axis })
    }

    fn compute<T, Idx>(&self, data: ArcArrayD<T>, indices: ArrayD<Idx>) -> Result<ArcArrayD<T>>
    where
        T: Clone + std::fmt::Debug,
        Idx: Integer + AsPrimitive<i64> + Copy + 'static + std::fmt::Debug,
        i64: AsPrimitive<Idx>,
    {
        let axis = convert_negative_index(self.axis, data.ndim());

        let out: Vec<_> = indices
            .iter()
            .map(|idx| {
                let idx = convert_negative_index(idx.as_(), data.shape()[axis]);

                data.index_axis(Axis(axis), idx)
            })
            .collect();

        let stacked = ndarray::stack(Axis(0), &out)?;

        // (indices.len, *data.shape[..axis], *data.shape[(axis+1)..])
        let current_shape = stacked.shape();

        // (*current_shape[..axis], *indices.shape, *data.shape[(axis+1)..])
        let out_shape: Vec<_> = current_shape[1..=axis]
            .iter()
            .chain(indices.shape())
            .chain(&current_shape[(axis + 1)..])
            .cloned()
            .collect();

        // Move axes around before reshaping
        let permuted_shape: Vec<_> = (1..=axis)
            .chain([0].into_iter())
            .chain((axis + 1)..current_shape.len())
            .collect();

        let stacked = stacked.permuted_axes(permuted_shape.as_slice());
        let stacked = stacked.as_standard_layout().to_owned();
        Ok(stacked.into_shape(out_shape)?.into())
    }
}

pub fn setup_gather(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Gather::from_attrs(&ctx.attrs)?;
    let (data, indices): (_, OnnxIndex) = ctx.subscribe_to_inputs().narrow()?;
    let indices = indices.into_i64();
    setup_all!(
        TaggedTensorReceiver,
        data,
        data,
        setup((data, indices), move |(data, indices)| {
            state.compute(data, indices)
        })
    )
}
