use anyhow::{Result, bail};
use half::f16;

use crate::{
    ArcArrayD, DataType, Value,
    data_types::{TensorType, ValueType},
};
use flume::{Receiver, SendError, Sender};

/// Wrapper around a `Sender` which sends one clone per existing receiver.
#[derive(Debug)]
pub struct Broadcaster<T> {
    sender: Sender<T>,
}

#[derive(Debug)]
pub enum TaggedBroadcaster {
    Tensor(TaggedTensorBroadcaster),
}

#[derive(Debug)]
pub enum TaggedTensorBroadcaster {
    F16(Broadcaster<ArcArrayD<f16>>),
    F32(Broadcaster<ArcArrayD<f32>>),
    F64(Broadcaster<ArcArrayD<f64>>),
    I8(Broadcaster<ArcArrayD<i8>>),
    I16(Broadcaster<ArcArrayD<i16>>),
    I32(Broadcaster<ArcArrayD<i32>>),
    I64(Broadcaster<ArcArrayD<i64>>),
    U8(Broadcaster<ArcArrayD<u8>>),
    U16(Broadcaster<ArcArrayD<u16>>),
    U32(Broadcaster<ArcArrayD<u32>>),
    U64(Broadcaster<ArcArrayD<u64>>),
    Bool(Broadcaster<ArcArrayD<bool>>),
    String(Broadcaster<ArcArrayD<String>>),
}

#[derive(Debug, Clone)]
pub enum TaggedReceiver {
    Tensor(TaggedTensorReceiver),
    Sequence(()),
    Map,
}

// NOTE: The `Broadcaster` sends one value for each existing `Receiver`. I.e., these must not hang around idling.
#[derive(Debug, Clone)]
pub enum TaggedTensorReceiver {
    F16(Receiver<ArcArrayD<f16>>),
    F32(Receiver<ArcArrayD<f32>>),
    F64(Receiver<ArcArrayD<f64>>),
    I8(Receiver<ArcArrayD<i8>>),
    I16(Receiver<ArcArrayD<i16>>),
    I32(Receiver<ArcArrayD<i32>>),
    I64(Receiver<ArcArrayD<i64>>),
    U8(Receiver<ArcArrayD<u8>>),
    U16(Receiver<ArcArrayD<u16>>),
    U32(Receiver<ArcArrayD<u32>>),
    U64(Receiver<ArcArrayD<u64>>),
    Bool(Receiver<ArcArrayD<bool>>),
    String(Receiver<ArcArrayD<String>>),
}

pub fn value_channel(value_type: &ValueType) -> (TaggedBroadcaster, TaggedReceiver) {
    match value_type {
        ValueType::Tensor(TensorType(t)) => match t {
            DataType::Bool => {
                let (b, rec) = unbounded::<ArcArrayD<bool>>();
                (b.into(), rec)
            }
            DataType::String => {
                let (b, rec) = unbounded::<ArcArrayD<String>>();
                (b.into(), rec)
            }
            DataType::U8 => {
                let (b, rec) = unbounded::<ArcArrayD<u8>>();
                (b.into(), rec)
            }
            DataType::U16 => {
                let (b, rec) = unbounded::<ArcArrayD<u16>>();
                (b.into(), rec)
            }
            DataType::U32 => {
                let (b, rec) = unbounded::<ArcArrayD<u32>>();
                (b.into(), rec)
            }
            DataType::U64 => {
                let (b, rec) = unbounded::<ArcArrayD<u64>>();
                (b.into(), rec)
            }
            DataType::I8 => {
                let (b, rec) = unbounded::<ArcArrayD<i8>>();
                (b.into(), rec)
            }
            DataType::I16 => {
                let (b, rec) = unbounded::<ArcArrayD<i16>>();
                (b.into(), rec)
            }
            DataType::I32 => {
                let (b, rec) = unbounded::<ArcArrayD<i32>>();
                (b.into(), rec)
            }
            DataType::I64 => {
                let (b, rec) = unbounded::<ArcArrayD<i64>>();
                (b.into(), rec)
            }
            DataType::F16 => {
                let (b, rec) = unbounded::<ArcArrayD<f16>>();
                (b.into(), rec)
            }
            DataType::F32 => {
                let (b, rec) = unbounded::<ArcArrayD<f32>>();
                (b.into(), rec)
            }
            DataType::F64 => {
                let (b, rec) = unbounded::<ArcArrayD<f64>>();
                (b.into(), rec)
            }
        },
        _ => panic!(),
    }
}

pub fn unbounded<T>() -> (Broadcaster<T>, TaggedReceiver)
where
    Receiver<T>: Into<TaggedReceiver>,
{
    let (sender, receiver) = flume::unbounded::<T>();
    (Broadcaster { sender }, receiver.into())
}

impl<T: Clone> Broadcaster<T> {
    /// Send one clone of `T` per existing receiver.
    pub fn send(&self, value: T) -> Result<(), SendError<T>> {
        for _ in 0..self.sender.receiver_count() {
            self.sender.send(value.clone())?
        }
        Ok(())
    }
}

impl TaggedBroadcaster {
    pub fn send(&self, value: Value) -> Result<()> {
        match self {
            TaggedBroadcaster::Tensor(t) => match (t, value) {
                (TaggedTensorBroadcaster::F16(brcstr), Value::F16(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::F32(brcstr), Value::F32(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::F64(brcstr), Value::F64(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::I8(brcstr), Value::I8(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::I16(brcstr), Value::I16(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::I32(brcstr), Value::I32(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::I64(brcstr), Value::I64(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::U8(brcstr), Value::U8(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::U16(brcstr), Value::U16(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::U32(brcstr), Value::U32(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::U64(brcstr), Value::U64(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::Bool(brcstr), Value::Bool(a)) => brcstr.send(a)?,
                (TaggedTensorBroadcaster::String(brcstr), Value::String(a)) => brcstr.send(a)?,
                _ => anyhow::bail!("expected {:?}, got TODO", self.value_type()),
            },
        };
        Ok(())
    }

    pub fn value_type(&self) -> ValueType {
        todo!()
    }
}

impl TaggedReceiver {
    pub fn tensor_dtype(&self) -> Result<DataType> {
        todo!()
    }

    pub fn recv(&self) -> Result<Value> {
        match self {
            Self::Tensor(t) => t.recv(),
            _ => todo!(),
        }
    }

    pub fn unpack_tensor_receiver(self) -> Result<TaggedTensorReceiver> {
        match self {
            Self::Tensor(t) => Ok(t),
            Self::Sequence(_) => bail!("expected 'tensor' found 'sequence'"),
            Self::Map => bail!("expected 'tensor' found 'map'"),
        }
    }
}

impl TaggedTensorReceiver {
    pub fn recv(&self) -> Result<Value> {
        Ok(match self {
            Self::F16(rec) => Value::F16(rec.recv()?),
            Self::F32(rec) => Value::F32(rec.recv()?),
            Self::F64(rec) => Value::F64(rec.recv()?),
            Self::I8(rec) => Value::I8(rec.recv()?),
            Self::I16(rec) => Value::I16(rec.recv()?),
            Self::I32(rec) => Value::I32(rec.recv()?),
            Self::I64(rec) => Value::I64(rec.recv()?),
            Self::U8(rec) => Value::U8(rec.recv()?),
            Self::U16(rec) => Value::U16(rec.recv()?),
            Self::U32(rec) => Value::U32(rec.recv()?),
            Self::U64(rec) => Value::U64(rec.recv()?),
            Self::Bool(rec) => Value::Bool(rec.recv()?),
            Self::String(rec) => Value::String(rec.recv()?),
        })
    }

    pub fn dtype(&self) -> DataType {
        todo!()
    }
}

macro_rules! impl_from_receiver_array_for_tagged_receiver {
    ($ty:ty, $variant:ident) => {
        impl From<Receiver<ArcArrayD<$ty>>> for TaggedReceiver {
            fn from(value: Receiver<ArcArrayD<$ty>>) -> Self {
                TaggedReceiver::Tensor(TaggedTensorReceiver::$variant(value))
            }
        }
    };
}
impl_from_receiver_array_for_tagged_receiver!(f16, F16);
impl_from_receiver_array_for_tagged_receiver!(f32, F32);
impl_from_receiver_array_for_tagged_receiver!(f64, F64);
impl_from_receiver_array_for_tagged_receiver!(i8, I8);
impl_from_receiver_array_for_tagged_receiver!(i16, I16);
impl_from_receiver_array_for_tagged_receiver!(i32, I32);
impl_from_receiver_array_for_tagged_receiver!(i64, I64);
impl_from_receiver_array_for_tagged_receiver!(u8, U8);
impl_from_receiver_array_for_tagged_receiver!(u16, U16);
impl_from_receiver_array_for_tagged_receiver!(u32, U32);
impl_from_receiver_array_for_tagged_receiver!(u64, U64);
impl_from_receiver_array_for_tagged_receiver!(bool, Bool);
impl_from_receiver_array_for_tagged_receiver!(String, String);

macro_rules! impl_from_broadcaster_array_for_tagged_broadcaster {
    ($ty:ty, $variant:ident) => {
        impl From<Broadcaster<ArcArrayD<$ty>>> for TaggedBroadcaster {
            fn from(value: Broadcaster<ArcArrayD<$ty>>) -> Self {
                TaggedBroadcaster::Tensor(TaggedTensorBroadcaster::$variant(value))
            }
        }
    };
}

impl_from_broadcaster_array_for_tagged_broadcaster!(f16, F16);
impl_from_broadcaster_array_for_tagged_broadcaster!(f32, F32);
impl_from_broadcaster_array_for_tagged_broadcaster!(f64, F64);
impl_from_broadcaster_array_for_tagged_broadcaster!(i8, I8);
impl_from_broadcaster_array_for_tagged_broadcaster!(i16, I16);
impl_from_broadcaster_array_for_tagged_broadcaster!(i32, I32);
impl_from_broadcaster_array_for_tagged_broadcaster!(i64, I64);
impl_from_broadcaster_array_for_tagged_broadcaster!(u8, U8);
impl_from_broadcaster_array_for_tagged_broadcaster!(u16, U16);
impl_from_broadcaster_array_for_tagged_broadcaster!(u32, U32);
impl_from_broadcaster_array_for_tagged_broadcaster!(u64, U64);
impl_from_broadcaster_array_for_tagged_broadcaster!(bool, Bool);
impl_from_broadcaster_array_for_tagged_broadcaster!(String, String);
