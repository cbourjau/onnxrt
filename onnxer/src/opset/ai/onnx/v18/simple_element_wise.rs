//! Setup functions for element-wise operators without attributes.

use std::collections::HashMap;

use anyhow::{bail, Result};

use crate::session::{ExecutionStep, OpContext, SetupFnSignature};
use crate::utils::wrap_1_input;
use crate::{HomogeneousNumericFixed, TaggedReceiver};

macro_rules! element_wise_numeric {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let chs = ctx.input_channels().try_into()?;
            match chs {
                HomogeneousNumericFixed::F32([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
                HomogeneousNumericFixed::F64([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
                HomogeneousNumericFixed::I32([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
                HomogeneousNumericFixed::I64([a_ch]) => {
                    wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into()))
                }
            }
        })
    };
}

/// Make setup function for element wise operations with floating point input
macro_rules! element_wise_float {
    ($fun:expr) => {
        Box::new(move |ctx: OpContext| -> Result<ExecutionStep> {
            let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();

            match a_ch {
                TaggedReceiver::F32(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                TaggedReceiver::F64(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv($fun).into())),
                // Internal error. This should never happen.
                rec => bail!("Expected floating point input, found: '{:?}'", rec),
            }
        })
    };
}

/// HashMap of setup function of the ai.onnx domain version 18(?).
pub fn get_setup_dict() -> HashMap<&'static str, SetupFnSignature> {
    let mut setup_dict: HashMap<_, SetupFnSignature> = HashMap::new();
    setup_dict.insert("Abs", element_wise_numeric!(|el| el.abs()));
    setup_dict.insert("Acos", element_wise_float!(|el| el.acos()));
    setup_dict.insert("Acos", element_wise_float!(|el| el.acos()));
    setup_dict.insert("Acosh", element_wise_float!(|el| el.acosh()));
    setup_dict.insert("Acosh", element_wise_float!(|el| el.acosh()));
    setup_dict.insert("Asin", element_wise_float!(|el| el.asin()));
    setup_dict.insert("Asinh", element_wise_float!(|el| el.asinh()));
    setup_dict.insert("Atan", element_wise_float!(|el| el.atan()));
    setup_dict.insert("Atanh", element_wise_float!(|el| el.atanh()));
    setup_dict.insert("Ceil", element_wise_float!(|el| el.ceil()));
    setup_dict.insert("Cos", element_wise_float!(|el| el.cos()));
    setup_dict.insert("Cosh", element_wise_float!(|el| el.cosh()));
    setup_dict.insert("Exp", element_wise_float!(|el| el.exp()));
    setup_dict.insert("Floor", element_wise_float!(|el| el.floor()));
    // setup_dict.insert("IsInf", element_wise_float!(|el| dbg!(el).is_infinite())); // FIXME
    setup_dict.insert("Identity", Box::new(setup_identity));
    setup_dict.insert("IsNaN", element_wise_float!(|el| el.is_nan()));
    setup_dict.insert("Log", element_wise_float!(|el| el.ln()));
    setup_dict.insert("Neg", element_wise_numeric!(|a| -a));
    setup_dict.insert("Not", Box::new(setup_not));

    // TODO: Rust's `round` rounds towards 0.0 for half-way
    // values. ONNX standard rounds to nearest even!
    setup_dict.insert("Round", element_wise_float!(|el| el.round()));
    setup_dict.insert("Sigmoid", element_wise_float!(|el| 1. / (1. + (-el).exp())));
    setup_dict.insert("Sin", element_wise_float!(|el| el.sin()));
    setup_dict.insert("Sinh", element_wise_float!(|el| el.sinh()));
    setup_dict.insert("Sqrt", element_wise_float!(|el| el.sqrt()));
    setup_dict.insert("Tan", element_wise_float!(|el| el.tan()));
    setup_dict.insert("Tanh", element_wise_float!(|el| el.tanh()));
    setup_dict
}

fn setup_identity(ctx: OpContext) -> Result<ExecutionStep> {
    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    match a_ch {
        TaggedReceiver::Bool(a_ch) => wrap_1_input(a_ch, Ok),
        TaggedReceiver::F32(a_ch) => wrap_1_input(a_ch, Ok),
        TaggedReceiver::F64(a_ch) => wrap_1_input(a_ch, Ok),
        TaggedReceiver::I32(a_ch) => wrap_1_input(a_ch, Ok),
        TaggedReceiver::I64(a_ch) => wrap_1_input(a_ch, Ok),
        TaggedReceiver::String(a_ch) => wrap_1_input(a_ch, Ok),
    }
}

fn setup_not(ctx: OpContext) -> Result<ExecutionStep> {
    let [a_ch]: [_; 1] = ctx.input_channels().try_into().unwrap();
    match a_ch {
        TaggedReceiver::Bool(a_ch) => wrap_1_input(a_ch, |a| Ok(a.mapv(|el| !el).into())),
        _ => bail!("`Not` input must be of type bool."),
    }
}
