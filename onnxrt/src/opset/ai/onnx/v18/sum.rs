use crate::opset::utils::*;

pub fn setup_sum(ctx: OpContext) -> Result<ExecutionStep> {
    let data = ctx.subscribe_to_inputs().narrow()?;
    setup_all!(
        HomogeneousNumeric,
        data,
        data,
        setup((data,), move |(data,)| {
            let mut iter = data.into_iter();
            let mut out = iter
                .next()
                .ok_or_else(|| anyhow!("Sum: At least one input is required"))?
                .to_owned();
            for arr in iter {
                out = zip_map_with_co_broadcast(out.view(), arr.view(), |a, b| a + b)?;
            }
            Ok(out)
        })
    )
}
