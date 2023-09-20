mod macros;
mod narrowed_arguments;
mod ndarray_utils;
mod setup;

// Prelude for op implementations
pub use std::sync::mpsc::Receiver;

pub use anyhow::{anyhow, bail, Result};
pub use ndarray::prelude::*;

pub use self::narrowed_arguments::{
    FloatingPoint, Homogeneous, HomogeneousFixed, HomogeneousNumeric, HomogeneousNumericFixed,
    Integer, Narrow, Numeric, OnnxIndex,
};
pub use self::ndarray_utils::{
    co_broadcast, convert_negative_index, zip_map_with_co_broadcast, Index,
};
pub use self::setup::{
    setup, setup_variadic_hetero_output, setup_variadic_out, wrap_1_input, FromAttributes,
};
pub use crate::channels::TaggedReceiver;
pub use crate::session::{get_implicit_inputs, ExecutionStep, OpContext};
pub use crate::ArcArrayD;
pub(crate) use crate::Attributes;
pub use crate::{bin_numeric_with_co_broadcast, binary_op_factory, setup_all};

pub trait IntoArc {
    type Output;
    fn into_arc(self) -> Result<Self::Output>;
}

// // Deprecated
// impl<T> IntoArc for Result<ArrayD<T>> {
//     type Output = ArcArrayD<T>;
//     fn into_arc(self) -> Result<Self::Output> {
//         self.map(Into::into)
//     }
// }

impl<T> IntoArc for ArcArrayD<T> {
    type Output = ArcArrayD<T>;
    fn into_arc(self) -> Result<Self::Output> {
        Ok(self)
    }
}

impl<T> IntoArc for ArrayD<T> {
    type Output = ArcArrayD<T>;
    fn into_arc(self) -> Result<Self::Output> {
        Ok(self.into())
    }
}
