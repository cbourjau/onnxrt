use anyhow::{bail, Result};
use itertools::Itertools;
use ndarray::Axis;

use crate::session::{ExecFnSignature, OpContext};
use crate::utils::wrap_1_input;
use crate::{ArcArrayD, Dispatcher, EnumReceiver};

fn impute<T>(a: ArcArrayD<T>, keys: &[T], value: T) -> Result<ArcArrayD<T>>
where
    T: PartialEq + Clone + Copy + std::fmt::Debug,
{
    if keys.len() == 1 {
        Ok(a.mapv(|el| if keys.contains(&el) { value } else { el })
            .into())
    } else if keys.len() == a.shape()[a.ndim() - 1] {
        let last_axis = a.ndim() - 1;
        let mut a = a;
        for (mut view, k) in a.axis_iter_mut(Axis(last_axis)).zip(keys) {
            view.mapv_inplace(|el| if el == *k { value } else { el });
        }
        Ok(a)
    } else {
        bail!(
            "Last input dimension match attribute length of `{}` or be `1`",
            keys.len()
        )
    }
}

pub fn wire_imputer(ctx: OpContext) -> Result<(ExecFnSignature, Vec<Dispatcher>)> {
    // Should use ndarrays instead of vec?
    let (a_ch,) = ctx.input_channels().into_iter().collect_tuple().unwrap();
    use crate::Attribute;
    match a_ch {
        EnumReceiver::F32(a_ch) => {
            if let Ok(Attribute::F32s(_arr)) = ctx.attrs.get("imputed_value_floats") {
                unimplemented!();
            };
            let keys: Vec<f32> = ctx.attrs.get("imputed_value_floats")?.try_into()?;
            let value = ctx.attrs.get_with_default("replaced_value_float", 0.0)?;
            wrap_1_input(a_ch, move |a| impute(a, &keys, value))
        }
        EnumReceiver::F64(a_ch) => {
            let keys = Vec::<f32>::try_from(ctx.attrs.get("imputed_value_floats")?)?
                .into_iter()
                .map(|el| el as f64)
                .collect::<Vec<_>>();
            let value = ctx.attrs.get_with_default("replaced_value_float", 0.0)? as f64;
            wrap_1_input(a_ch, move |a| impute(a, &keys, value))
        }
        EnumReceiver::I64(a_ch) => {
            let keys: Vec<i64> = ctx.attrs.get("imputed_value_int64s")?.try_into()?;
            let value = ctx.attrs.get_with_default("replaced_value_int64", 0)?;
            wrap_1_input(a_ch, move |a| impute(a, &keys, value))
        }
        EnumReceiver::I32(a_ch) => {
            let keys: Vec<i64> = ctx.attrs.get("imputed_value_int64s")?.try_into()?;
            let value = ctx.attrs.get_with_default("replaced_value_int64", 0)?;
            wrap_1_input(a_ch, move |a| {
                impute(a.mapv(|el| el as i64).into(), &keys, value)
            })
        }
        _ => bail!(
            "Input of Imputer must be `tensor(f32)`, `tensor(f64)`, `tensor(i32)`, `tensor(i64)`."
        ),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn broadcasting() {
        let input: ArcArrayD<_> = array![[0, 1], [2, 3]].into_dyn().into();
        let expected: ArcArrayD<_> = array![[42, 1], [2, 3]].into_dyn().into();
        assert_eq!(impute(input.clone(), &[0], 42).unwrap(), expected);

        let expected: ArcArrayD<_> = array![[42, 1], [2, 42]].into_dyn().into();
        assert_eq!(impute(input, &[0, 3], 42).unwrap(), expected);
    }
}
