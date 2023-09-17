use crate::opset::utils::*;

pub fn setup_equal(ctx: OpContext) -> Result<ExecutionStep> {
    let data = ctx.subscribe_to_inputs().narrow()?;

    setup_all!(
        HomogeneousFixed,
        data,
        [a, b],
        setup((a, b), move |(a, b)| zip_map_with_co_broadcast(
            a.view(),
            b.view(),
            |a, b| a == b
        ))
    )
}
