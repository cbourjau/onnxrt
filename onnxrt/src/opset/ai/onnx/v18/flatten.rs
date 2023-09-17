use crate::opset::utils::*;

fn flatten_2d<T>(input: ArrayViewD<T>, axis: i64) -> Result<Array2<T>>
where
    T: Clone,
{
    let axis = convert_negative_index(axis, input.ndim());
    let input_shape = input.shape();

    let x1 = input_shape[..axis].iter().product();
    let x2 = input_shape[axis..].iter().product();

    Ok(input.into_shape([x1, x2])?.to_owned())
}

struct Flatten {
    axis: i64,
}

impl Flatten {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        Ok(Self {
            axis: attrs.get_with_default("axis", 1)?,
        })
    }
}

pub fn setup_flatten(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Flatten::from_attrs(&ctx.attrs)?;

    let input = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedReceiver,
        input,
        input,
        setup(input, move |input| {
            Ok(flatten_2d(input.view(), state.axis)?.into_dyn())
        })
    )
}
