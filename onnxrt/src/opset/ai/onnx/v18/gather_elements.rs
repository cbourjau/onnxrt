use num::Integer;
use num_traits::AsPrimitive;

use crate::opset::utils::*;

#[derive(Debug)]
struct GatherElements {
    axis: i64,
}

impl GatherElements {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let axis = attrs.get_with_default("axis", 0)?;

        Ok(Self { axis })
    }

    fn compute<T, Idx>(&self, data: ArcArrayD<T>, indices: ArrayD<Idx>) -> Result<ArcArrayD<T>>
    where
        T: Clone + Default,
        Idx: Integer + AsPrimitive<i64> + Copy + 'static,
        i64: AsPrimitive<Idx>,
    {
        let axis = convert_negative_index(self.axis, data.ndim());

        let len_axis = data.shape().get(axis).unwrap();
        let mut out = ndarray::ArrayD::<T>::default(indices.shape());
        for ((data1d, indices1d), mut out1d) in data
            .lanes(Axis(axis))
            .into_iter()
            .zip(indices.lanes(Axis(axis)).into_iter())
            .zip(out.lanes_mut(Axis(axis)))
        {
            out1d.assign(&indices1d.mapv(|idx| {
                let idx = convert_negative_index(idx.as_(), *len_axis);
                data1d.get(idx).unwrap().clone()
            }))
        }

        Ok(out.into())
    }
}

pub fn setup_gather_elements(ctx: OpContext) -> Result<ExecutionStep> {
    let state = GatherElements::from_attrs(&ctx.attrs)?;

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
