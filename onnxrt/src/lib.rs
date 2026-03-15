use ndarray::{ArcArray, IxDyn};

mod attribute;
mod channels;
mod data_types;
mod opset;
mod python_api;
mod session;

pub(crate) use attribute::Attributes;
pub use channels::{Sendable, TaggedReceiver, TaggedTensorReceiver, ValueSender, value_channel};
pub use data_types::DataType;

pub use opset::get_opset;
pub use session::Session;

pub type ArcArrayD<T> = ArcArray<T, IxDyn>;

/// Enum capable of representing any possible value. Primarily used in `Session::run`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(ArcArrayD<bool>),
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
