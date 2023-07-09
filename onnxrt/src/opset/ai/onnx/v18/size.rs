use crate::opset::utils::*;

pub fn setup_size(ctx: OpContext) -> Result<ExecutionStep> {
    let data = ctx.subscribe_to_inputs().narrow()?;
    setup_all!(
        TaggedReceiver,
        data,
        data,
        setup((data,), move |(data,)| Ok::<ArcArrayD<_>, _>(
            arr0(data.len() as i64).into_dyn().into()
        ))
    )
}
