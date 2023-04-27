use std::collections::HashMap;

use anyhow::{bail, Result};
use itertools::Itertools;
use ordered_float::OrderedFloat;

use crate::session::{ExecutionStep, OpContext};
use crate::utils::wrap_1_input;
use crate::{ArcArrayD, Attributes, EnumReceiver};

enum LabelEncoder {
    StringToString {
        dict: HashMap<String, String>,
        default: String,
    },
    StringToI64 {
        dict: HashMap<String, i64>,
        default: i64,
    },
    StringToF32 {
        dict: HashMap<String, f32>,
        default: f32,
    },
    I64ToI64 {
        dict: HashMap<i64, i64>,
        default: i64,
    },
    I64ToF32 {
        dict: HashMap<i64, f32>,
        default: f32,
    },
    I64ToString {
        dict: HashMap<i64, String>,
        default: String,
    },
    F32ToString {
        dict: HashMap<OrderedFloat<f32>, String>,
        default: String,
    },
    F32ToF32 {
        dict: HashMap<OrderedFloat<f32>, f32>,
        default: f32,
    },
    F32ToI64 {
        dict: HashMap<OrderedFloat<f32>, i64>,
        default: i64,
    },
}

fn replace<Input, Key, Out>(
    a: ArcArrayD<Input>,
    lookup: &HashMap<Key, Out>,
    default: &Out,
) -> Result<ArcArrayD<Out>>
where
    Input: Clone,
    Key: std::hash::Hash + Eq + From<Input>,
    Out: Clone,
{
    Ok(
        a.mapv(|el| lookup.get(&el.into()).unwrap_or(default).to_owned())
            .into(),
    )
}

fn make_lookup<Input, Key, Value>(keys: &[Input], values: &[Value]) -> HashMap<Key, Value>
where
    Input: Clone,
    Key: Clone + Eq + std::hash::Hash + From<Input>,
    Value: Clone,
{
    keys.iter()
        .cloned()
        .zip(values)
        .map(|(k, v)| (k.into(), v.to_owned()))
        .collect()
}

impl LabelEncoder {
    fn from_attributes(attrs: &Attributes) -> Result<Self> {
        let keys_strings: Option<Vec<String>> = attrs.get_optional("keys_strings")?;
        let keys_i64s: Option<Vec<i64>> = attrs.get_optional("keys_int64s")?;
        let keys_f32s: Option<Vec<f32>> = attrs.get_optional("keys_floats")?;

        let values_strings: Option<Vec<String>> = attrs.get_optional("values_strings")?;
        let values_i64s: Option<Vec<i64>> = attrs.get_optional("values_int64s")?;
        let values_f32s: Option<Vec<f32>> = attrs.get_optional("values_floats")?;

        let default_string = attrs.get_with_default("default_string", "_Unused".to_string())?;
        let default_int64 = attrs.get_with_default("default_int64", -1)?;
        let default_float = attrs.get_with_default("default_float", -0.0)?;

        match (keys_strings, keys_i64s, keys_f32s) {
            (Some(keys), None, None) => match (values_strings, values_i64s, values_f32s) {
                (Some(values), None, None) => Ok(Self::StringToString {
                    dict: make_lookup(&keys, &values),
                    default: default_string,
                }),
                (None, Some(values), None) => Ok(Self::StringToI64 {
                    dict: make_lookup(&keys, &values),
                    default: default_int64,
                }),
                (None, None, Some(values)) => Ok(Self::StringToF32 {
                    dict: make_lookup(&keys, &values),
                    default: default_float,
                }),
                _ => panic!(),
            },
            (None, Some(keys), None) => match (values_strings, values_i64s, values_f32s) {
                (None, Some(values), None) => Ok(Self::I64ToI64 {
                    dict: make_lookup(&keys, &values),
                    default: default_int64,
                }),
                (Some(values), None, None) => Ok(Self::I64ToString {
                    dict: make_lookup(&keys, &values),
                    default: default_string,
                }),
                (None, None, Some(values)) => Ok(Self::I64ToF32 {
                    dict: make_lookup(&keys, &values),
                    default: default_float,
                }),
                _ => panic!(),
            },
            (None, None, Some(keys)) => match (values_strings, values_i64s, values_f32s) {
                (Some(values), None, None) => Ok(Self::F32ToString {
                    dict: make_lookup(&keys, &values),
                    default: default_string,
                }),
                (None, Some(values), None) => Ok(Self::F32ToI64 {
                    dict: make_lookup(&keys, &values),
                    default: default_int64,
                }),
                (None, None, Some(values)) => Ok(Self::F32ToF32 {
                    dict: make_lookup(&keys, &values),
                    default: default_float,
                }),
                _ => panic!(),
            },
            keys => panic!("Invalid key combination: {:?}", keys),
        }
    }
}

pub fn wire_label_encoder(ctx: OpContext) -> Result<ExecutionStep> {
    let (a_ch,) = ctx.input_channels().into_iter().collect_tuple().unwrap();
    let label_encoder = LabelEncoder::from_attributes(&ctx.attrs)?;

    match (a_ch, label_encoder) {
        (EnumReceiver::String(a_ch), LabelEncoder::StringToString { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::String(a_ch), LabelEncoder::StringToI64 { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::String(a_ch), LabelEncoder::StringToF32 { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::I64(a_ch), LabelEncoder::I64ToI64 { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::I64(a_ch), LabelEncoder::I64ToString { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::I64(a_ch), LabelEncoder::I64ToF32 { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::F32(a_ch), LabelEncoder::F32ToString { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::F32(a_ch), LabelEncoder::F32ToI64 { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        (EnumReceiver::F32(a_ch), LabelEncoder::F32ToF32 { dict, default }) => {
            wrap_1_input(a_ch, move |a| replace(a, &dict, &default))
        }
        _ => panic!(),
    }
}

pub fn wire_category_mapper(ctx: OpContext) -> Result<ExecutionStep> {
    let (a_ch,) = ctx.input_channels().into_iter().collect_tuple().unwrap();
    let cats_int64s: Vec<i64> = ctx.attrs.get("cats_int64s")?.try_into()?;
    let cats_strings: Vec<String> = ctx.attrs.get("cats_strings")?.try_into()?;
    let default_int64 = ctx.attrs.get_with_default("default_int64", -1)?;
    let default_string = ctx
        .attrs
        .get_with_default("default_string", "_Unused".to_owned())?;

    let i64_to_string = HashMap::from_iter(
        cats_int64s
            .iter()
            .cloned()
            .zip(cats_strings.iter().cloned()),
    );
    let string_to_int = HashMap::from_iter(
        cats_strings
            .iter()
            .cloned()
            .zip(cats_int64s.iter().cloned()),
    );
    match a_ch {
        EnumReceiver::I64(a_ch) => {
            wrap_1_input(a_ch, move |a| replace(a, &i64_to_string, &default_string))
        }
        EnumReceiver::String(a_ch) => {
            wrap_1_input(a_ch, move |a| replace(a, &string_to_int, &default_int64))
        }
        _ => bail!("Illegal input type"),
    }
}
#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn float2float() {
        let subnormal = 1.0e-38f32;
        let dict = HashMap::from_iter([
            (OrderedFloat(f32::NAN), "f32::NAN".to_owned()),
            (OrderedFloat(0.0), "0.0".to_owned()),
            (OrderedFloat(subnormal), "subnormal".to_owned()),
        ]);
        replace(
            array![f32::NAN, -0.0, subnormal].into_dyn().into(),
            &dict,
            &"42.0".to_owned(),
        )
        .unwrap();
    }
}
