use anyhow::Result;
use half::f16;
use ndarray::{ArcArray, IxDyn};

mod attribute;
mod channels;
mod data_types;
mod opset;
mod python_api;
mod session;

pub(crate) use attribute::Attributes;
pub use channels::{TaggedReceiver, TaggedTensorReceiver, value_channel};
pub use data_types::DataType;

pub use opset::get_opset;
pub use session::Session;

pub type ArcArrayD<T> = ArcArray<T, IxDyn>;

/// Enum capable of representing any possible value. Primarily used in `Session::run`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(ArcArrayD<bool>),
    F16(ArcArrayD<f16>),
    F32(ArcArrayD<f32>),
    F64(ArcArrayD<f64>),
    I16(ArcArrayD<i16>),
    I32(ArcArrayD<i32>),
    I64(ArcArrayD<i64>),
    I8(ArcArrayD<i8>),
    String(ArcArrayD<String>),
    U16(ArcArrayD<u16>),
    U32(ArcArrayD<u32>),
    U64(ArcArrayD<u64>),
    U8(ArcArrayD<u8>),
}

/// Trait that executes a single node and sends its outputs further.
pub trait Executable {
    fn execute(&self) -> Result<()>;
}

pub trait Setup {
    fn setup(&self) -> Result<(Box<dyn Executable + Send>, Vec<TaggedReceiver>)>;
}
