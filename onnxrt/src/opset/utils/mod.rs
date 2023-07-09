mod macros;
mod narrowed_arguments;
mod ndarray_utils;
mod setup;

// Prelude for op implementations
pub use std::sync::mpsc::Receiver;

pub use anyhow::{anyhow, bail, Result};
pub use ndarray::prelude::*;

pub use self::narrowed_arguments::{
    Homogeneous, HomogeneousFixed, HomogeneousNumeric, HomogeneousNumericFixed, NarrowFromVec,
    Numeric,
};
pub use self::ndarray_utils::{co_broadcast, convert_negative_index, zip_map_with_co_broadcast};
pub use self::setup::{setup, setup_variadic_out, wrap_1_input};
pub use crate::edges::TaggedReceiver;
pub use crate::session::ExecutionStep;
pub use crate::session::OpContext;
pub use crate::ArcArrayD;
pub use crate::{bin_numeric_with_co_broadcast, binary_op_factory, setup_all};
pub(crate) use crate::{Attribute, Attributes};
