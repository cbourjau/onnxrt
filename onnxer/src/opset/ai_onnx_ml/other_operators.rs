use anyhow::{bail, Result};
use itertools::Itertools;
use ndarray::{Axis, Ix1};

use crate::session::{ExecutionStep, OpContext};
use crate::{ArcArrayD, Dispatchable, EnumReceiver};

pub fn wire_array_feature_extractor(ctx: OpContext) -> Result<ExecutionStep> {
    let (data_ch, indices_ch) = ctx.input_channels().into_iter().collect_tuple().unwrap();
    let indices_ch = match indices_ch {
        EnumReceiver::I64(ch) => ch,
        _ => bail!("Input `Y` of ArrayFeatureExtractor must be of type `tensor(int64)`"),
    };

    macro_rules! match_arm {
        ($ch:expr, $indices_ch:expr) => {{
            let (sdr, rec) = ArcArrayD::<_>::channel();
            let fun = Box::new(move || {
                let data = $ch.try_recv()?;
                let indices = $indices_ch
                    .try_recv()?
                    .into_dimensionality::<Ix1>()?
                    .to_vec();
                let last_axis = Axis(data.ndim() - 1);
                let outs: Vec<_> = indices
                    .into_iter()
                    .map(|idx| {
                        data.index_axis(last_axis, idx as usize)
                            .insert_axis(last_axis)
                    })
                    .collect();
                let out = ndarray::concatenate(Axis(data.ndim() - 1), outs.as_slice())?;
                sdr.send(out.to_owned().into())?;
                Ok(())
            });
            Ok(ExecutionStep {
                fun,
                outputs: vec![rec.into()],
            })
        }};
    }
    match data_ch {
        EnumReceiver::I32(ch) => match_arm!(ch, indices_ch),
        EnumReceiver::I64(ch) => match_arm!(ch, indices_ch),
        EnumReceiver::F32(ch) => match_arm!(ch, indices_ch),
        EnumReceiver::F64(ch) => match_arm!(ch, indices_ch),
        EnumReceiver::String(ch) => match_arm!(ch, indices_ch),
        EnumReceiver::Bool(ch) => match_arm!(ch, indices_ch),
    }
}
