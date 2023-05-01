use anyhow::Result;

use super::data_structure::TreeEnsemble;
use crate::session::{ExecutionStep, OpContext};
use crate::{ArcArrayD, Dispatchable, EnumReceiver};

pub fn setup_tree_ensemble_regressor(ctx: OpContext) -> Result<ExecutionStep> {
    let mut input_channels = ctx.input_channels().into_iter();
    let a_ch = input_channels.next().unwrap();
    let ensemble = TreeEnsemble::from_attributes(&ctx.attrs).unwrap();
    let (sdr, rec) = ArcArrayD::<_>::channel();
    match a_ch {
        EnumReceiver::F64(a_ch) => {
            let fun = Box::new(move || {
                let a = a_ch.recv().unwrap();
                let out = ensemble.predict(&a);
                sdr.send(out.into_dyn().into())?;
                Ok(())
            });
            Ok(ExecutionStep {
                fun,
                outputs: vec![rec],
            })
        }
        _ => panic!(),
    }
}
