use std::sync::mpsc::Receiver;

use crate::opset::utils::*;

fn expand<T>(input: ArrayViewD<T>, shape: &[usize]) -> Result<ArrayD<T>>
where
    T: Clone,
{
    let input_shape = input.shape();
    Ok(input
        .broadcast(shape)
        .ok_or_else(|| {
            anyhow!(
                "Expand: Cannot broadcast input of shape '{:?}' to shape '{:?}'.",
                input_shape,
                shape
            )
        })?
        .to_owned())
}

pub fn setup_expand(ctx: OpContext) -> Result<ExecutionStep> {
    let (input, shape): (_, Receiver<ArcArrayD<i64>>) = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        TaggedReceiver,
        input,
        input,
        setup((input, shape), |(input, shape)| {
            // SPECS UNCLEAR: What happens with negative indices?
            let shape = shape.as_standard_layout().mapv(|el| el as usize);
            let shape = shape.as_slice().unwrap();
            expand(input.view(), shape)
        })
    )
}
