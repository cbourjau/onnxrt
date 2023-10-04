use crate::opset::utils::*;

struct Compress {
    axis: Option<i64>,
}

impl FromAttributes for Compress {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let axis: Option<i64> = attrs.get_optional("axis")?;
        Ok(Self { axis })
    }
}

fn compress<T>(input: ArrayViewD<T>, cond: ArrayViewD<bool>, axis: Option<i64>) -> Result<ArrayD<T>>
where
    T: Clone,
{
    if let Some(axis) = axis {
        let axis = convert_negative_index(axis, input.ndim());
        let views: Vec<_> = input
            .axis_iter(Axis(axis))
            .zip(cond.iter())
            .filter_map(|(view, &cond)| if cond { Some(view) } else { None })
            .collect();
        Ok(ndarray::stack(Axis(axis), &views)?.into_dyn())
    } else {
        let iter = input
            .into_iter()
            .zip(cond.iter())
            .filter_map(|(el, &cond)| if cond { Some(el.clone()) } else { None });
        Ok(Array1::from_iter(iter).into_dyn())
    }
}

pub fn setup_compress(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Compress::from_attrs(&ctx.attrs)?;
    let (input, cond): (_, Receiver<ArcArrayD<bool>>) = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedTensorReceiver,
        input,
        inp,
        setup((inp, cond), move |(inp, cond)| {
            compress(inp.view(), cond.view(), state.axis)
        })
    )
}
