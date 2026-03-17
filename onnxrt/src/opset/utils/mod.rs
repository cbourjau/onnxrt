// mod macros;
mod narrowed_arguments;
mod ndarray_utils;
// mod setup;
mod binary_utils;
mod unary_utils;

pub use anyhow::Result;
pub use ndarray::prelude::*;

pub use self::narrowed_arguments::{HomogeneousFixed, HomogeneousNumericFixed, Narrow};
pub use crate::ArcArrayD;
pub use crate::channels::TaggedTensorReceiver;
pub(crate) use binary_utils::{
    BinaryKernel, BinaryWrapper, make_binary_boolean, make_binary_numeric, match_binary,
};
pub(crate) use unary_utils::*;

// pub use crate::{bin_numeric_with_co_broadcast, binary_op_factory, setup_all};

pub trait IntoArc {
    type Output;
    fn into_arc(self) -> Result<Self::Output>;
}

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
