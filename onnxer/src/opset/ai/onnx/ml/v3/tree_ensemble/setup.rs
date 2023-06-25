use anyhow::Result;

use super::data_structure::TreeEnsemble;
use crate::opset::utils::*;
use crate::session::{ExecutionStep, OpContext};
use crate::{ArcArrayD, TaggedReceiver};

pub fn setup_tree_ensemble_regressor(ctx: OpContext) -> Result<ExecutionStep> {
    let a_ch = ctx.input_channels().narrow()?;

    let ensemble = TreeEnsemble::from_attributes(&ctx.attrs).unwrap();
    match a_ch {
        TaggedReceiver::F64(a_ch) => setup((a_ch,), move |(a,)| {
            Ok::<ArcArrayD<_>, _>(ensemble.predict(&a).into())
        }),
        _ => bail!("TreeEnsembleRegressor: Unexpected input type"),
    }
}
