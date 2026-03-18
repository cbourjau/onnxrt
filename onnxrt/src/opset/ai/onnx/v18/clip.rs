use anyhow::{anyhow, bail};
use flume::Receiver;
use half::f16;
use num::Float;
use num_traits::FloatConst;

use crate::{Executable, TaggedReceiver, channels::unbounded, opset::utils::*, session::OpContext};

use super::min_max::{max_nan, min_nan};

pub fn setup_clip(
    ctx: OpContext,
) -> Result<(
    Box<dyn crate::Executable + Send + Sync>,
    Vec<crate::TaggedReceiver>,
)> {
    let mut inputs = ctx.input_channels_with_optional();
    // Optional inputs may be omitted; we normalize by filling
    if inputs.len() > 3 {
        bail!("expected 3 or less inputs, got `{}`", inputs.len())
    }
    inputs.resize(3, None);
    let inputs: [_; 3] = inputs
        .try_into()
        .map_err(|e| anyhow!("unexpected number of inputs: {:?}", e))?;

    use TaggedReceiver::*;
    match inputs {
        [Some(a), Some(b), Some(c)] => {
            use HomogeneousNumericFixed::*;
            let inputs = vec![a, b, c].narrow()?;
            match inputs {
                U8([a, b, c]) => setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min))),
                U16([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min)))
                }
                U32([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min)))
                }
                U64([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min)))
                }
                I8([a, b, c]) => setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min))),
                I16([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min)))
                }
                I32([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min)))
                }
                I64([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| *max.min(x.max(min)))
                }
                F16([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| max_nan(min_nan(*x, *max), *min))
                }
                F32([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| max_nan(min_nan(*x, *max), *min))
                }
                F64([a, b, c]) => {
                    setup_tertiary_zipped(a, b, c, |x, min, max| max_nan(min_nan(*x, *max), *min))
                }
            }
        }
        [Some(a), None, Some(c)] => {
            use HomogeneousNumericFixed::*;
            let inputs = vec![a, c].narrow()?;
            match inputs {
                U8([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                U16([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                U32([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                U64([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                I8([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                I16([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                I32([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                I64([a, c]) => setup_binary_zipped(a, c, |x, max| *max.min(x)),
                F16([a, c]) => setup_binary_zipped(a, c, |x, max| min_nan(*x, *max)),
                F32([a, c]) => setup_binary_zipped(a, c, |x, max| min_nan(*x, *max)),
                F64([a, c]) => setup_binary_zipped(a, c, |x, max| min_nan(*x, *max)),
            }
        }
        [Some(a), Some(b), None] => {
            use HomogeneousNumericFixed::*;
            let inputs = vec![a, b].narrow()?;
            match inputs {
                U8([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                U16([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                U32([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                U64([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                I8([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                I16([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                I32([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                I64([a, b]) => setup_binary_zipped(a, b, |x, min| *x.max(min)),
                F16([a, b]) => setup_binary_zipped(a, b, |x, min| max_nan(*x, *min)),
                F32([a, b]) => setup_binary_zipped(a, b, |x, min| max_nan(*x, *min)),
                F64([a, b]) => setup_binary_zipped(a, b, |x, min| max_nan(*x, *min)),
            }
        }
        [Some(a), None, None] => Ok((Box::new(|| Ok(())) as _, vec![a])),
        _ => bail!("Clip: input 'input' is not optional"),
    }
}
