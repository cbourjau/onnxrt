use anyhow::{bail, Result};
use itertools::Itertools;
use num::cast::AsPrimitive;

use crate::session::{ExecFnSignature, OpContext};
use crate::utils::wrap_1_input;
use crate::{ArcArrayD, Dispatcher, EnumReceiver};

fn binarize<T>(a: ArcArrayD<T>, threshold: T) -> Result<ArcArrayD<T>>
where
    T: PartialOrd + Clone + Copy + 'static,
    u8: AsPrimitive<T>,
{
    Ok(a.mapv(|el| ((el > threshold) as u8).as_()).into())
}

pub fn wire_binarizer(ctx: OpContext) -> Result<(ExecFnSignature, Vec<Dispatcher>)> {
    let (a_ch,) = ctx.input_channels().into_iter().collect_tuple().unwrap();
    let threshold: f32 = ctx.attrs.get("threshold")?.try_into()?;

    match a_ch {
        EnumReceiver::F32(rec) => wrap_1_input(rec, move |a| binarize(a, threshold)),
        EnumReceiver::F64(rec) => wrap_1_input(rec, move |a| binarize(a, threshold as f64)),
        // `floor` first for int cast in order to ensure that we don't
        // round up (for negative numbers).
        EnumReceiver::I32(rec) => wrap_1_input(rec, move |a| binarize(a, threshold.floor() as i32)),
        EnumReceiver::I64(rec) => wrap_1_input(rec, move |a| binarize(a, threshold.floor() as i64)),
        _ => bail!("Binarizer is only defined for f32, f64, i32 and i64 inputs"),
    }
}
