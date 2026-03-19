use anyhow::bail;
use num::Integer;

use crate::{TaggedReceiver, opset::utils::*, session::OpContext};

pub fn setup_mod(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let fmod: i64 = ctx.attrs.get_with_default("fmod", 0)?;
    let inputs = ctx.subscribe_to_inputs().narrow()?;

    use HomogeneousNumericFixed::*;
    if fmod == 0 {
        match inputs {
            U8([a, b]) => setup_binary_zipped(a, b, |a, b| a % b),
            U16([a, b]) => setup_binary_zipped(a, b, |a, b| a % b),
            U32([a, b]) => setup_binary_zipped(a, b, |a, b| a % b),
            U64([a, b]) => setup_binary_zipped(a, b, |a, b| a % b),
            I8([a, b]) => setup_binary_zipped(a, b, |a, b| a.mod_floor(b)),
            I16([a, b]) => setup_binary_zipped(a, b, |a, b| a.mod_floor(b)),
            I32([a, b]) => setup_binary_zipped(a, b, |a, b| a.mod_floor(b)),
            I64([a, b]) => setup_binary_zipped(a, b, |a, b| a.mod_floor(b)),
            _ => bail!("Mod: expected integer data types due to 'fmod=0'"),
        }
    } else {
        match inputs {
            F16([a, b]) => setup_binary_zipped(a, b, |a, b| a % b),
            F32([a, b]) => setup_binary_zipped(a, b, |a, b| a % b),
            F64([a, b]) => setup_binary_zipped(a, b, |a, b| a % b),

            _ => bail!("Mod: expected floating data types due to 'fmod=1'"),
        }
    }
}
