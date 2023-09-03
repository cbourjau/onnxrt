use anyhow::Result;

use super::data_structure::TreeEnsemble;
use crate::opset::utils::*;
use crate::session::{ExecutionStep, OpContext};

pub fn setup_tree_ensemble_regressor(ctx: OpContext) -> Result<ExecutionStep> {
    // We parse the attributes after matching the inputs since the
    // generic parameter of `TreeEnsemble` depends on the input type
    let attrs = ctx.attrs.clone();
    let a_ch = ctx.subscribe_to_inputs().narrow()?;

    use FloatingPoint::*;
    match a_ch {
        F64(a_ch) => {
            let ensemble = TreeEnsemble::from_attributes(&attrs).unwrap();
            setup((a_ch,), move |(a,)| Ok(ensemble.predict(&a)).into_arc())
        }
        F32(a_ch) => {
            let ensemble = TreeEnsemble::from_attributes(&attrs).unwrap();
            setup((a_ch,), move |(a,)| Ok(ensemble.predict(&a)).into_arc())
        }
    }
}
