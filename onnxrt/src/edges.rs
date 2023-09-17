use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use onnx_protos::onnx_proto3::{tensor_proto, type_proto, ValueInfoProto};
use protobuf::Enum;

use crate::ArcArrayD;

/// Data types which may be exchange between nodes.
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum DataType {
    Bool,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    String,
}

/// Enum capable of representing any possible value. Primarily used in `Session::run`.
/// A concrete tensor
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

/// Sender Consuming a `Value` and forwarding it to the `Sender` of the correct type.
#[derive(Debug)]
pub enum ValueSender {
    Bool(Sender<ArcArrayD<bool>>),
    F32(Sender<ArcArrayD<f32>>),
    F64(Sender<ArcArrayD<f64>>),

    I8(Sender<ArcArrayD<i8>>),
    I16(Sender<ArcArrayD<i16>>),
    I32(Sender<ArcArrayD<i32>>),
    I64(Sender<ArcArrayD<i64>>),

    U8(Sender<ArcArrayD<u8>>),
    U16(Sender<ArcArrayD<u16>>),
    U32(Sender<ArcArrayD<u32>>),
    U64(Sender<ArcArrayD<u64>>),
    String(Sender<ArcArrayD<String>>),
}

/// Typed sender. Used to send values of known types out of "wire" functions.
#[derive(Default, Debug)]
pub struct Sender<T> {
    sub_senders: Arc<Mutex<Vec<SyncSender<T>>>>,
}

/// Object returned by "wire" functions. Only used during the setup
/// phase. Downstream nodes can subscribe to this object to receive
/// outputs of previous nodes. It serves as a one to many channel
/// where all consumers receive a send object.
#[derive(Debug)]
pub enum Dispatcher {
    Bool(Arc<Mutex<Vec<SyncSender<ArcArrayD<bool>>>>>),
    F32(Arc<Mutex<Vec<SyncSender<ArcArrayD<f32>>>>>),
    F64(Arc<Mutex<Vec<SyncSender<ArcArrayD<f64>>>>>),

    I8(Arc<Mutex<Vec<SyncSender<ArcArrayD<i8>>>>>),
    I16(Arc<Mutex<Vec<SyncSender<ArcArrayD<i16>>>>>),
    I32(Arc<Mutex<Vec<SyncSender<ArcArrayD<i32>>>>>),
    I64(Arc<Mutex<Vec<SyncSender<ArcArrayD<i64>>>>>),

    U8(Arc<Mutex<Vec<SyncSender<ArcArrayD<u8>>>>>),
    U16(Arc<Mutex<Vec<SyncSender<ArcArrayD<u16>>>>>),
    U32(Arc<Mutex<Vec<SyncSender<ArcArrayD<u32>>>>>),
    U64(Arc<Mutex<Vec<SyncSender<ArcArrayD<u64>>>>>),

    String(Arc<Mutex<Vec<SyncSender<ArcArrayD<String>>>>>),
}

/// Object obtained by subscribing to a `Dispatcher`. Used inside
/// "wire" functions and to obtain the final values of a session. The
/// values of each variant are typed std receivers.
#[derive(Debug)]
pub enum TaggedReceiver {
    Bool(Receiver<ArcArrayD<bool>>),
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

    String(Receiver<ArcArrayD<String>>),
}

/// Create a value-channel used as an entry point in a session
pub fn value_channel(dtype: &DataType) -> (ValueSender, Dispatcher) {
    macro_rules! map_vars {
        ($(($var:tt, $ty:ty)),*) => {
            match dtype {
                $(DataType::$var => {
                    let (sdr, disp) = ArcArrayD::<$ty>::channel();
                    (ValueSender::$var(sdr), disp)
                })*
            }
        }
    }
    map_vars!(
        (Bool, bool),
        (F32, f32),
        (F64, f64),
        (I8, i8),
        (I16, i16),
        (I32, i32),
        (I64, i64),
        (U8, u8),
        (U16, u16),
        (U32, u32),
        (U64, u64),
        (String, String)
    )
}

impl ValueSender {
    pub fn send(&self, value: Value) -> Result<()> {
        macro_rules! map_vars {
            ($($var:tt),*) => {
                match (self, value) {
                    $(
                        (Self::$var(sdr), Value::$var(val)) => {
                            sdr.send(val)
                        },
                    )*
                        (sdr, val) => Err(anyhow!("Expected {:?}, found {:?}", sdr.dtype(), val)),
                }
            }
        }

        map_vars!(Bool, F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, String)
    }

    pub fn dtype(&self) -> DataType {
        match self {
            Self::Bool(_) => DataType::Bool,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,

            Self::I8(_) => DataType::I8,
            Self::I16(_) => DataType::I16,
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,

            Self::U8(_) => DataType::U8,
            Self::U16(_) => DataType::U16,
            Self::U32(_) => DataType::U32,
            Self::U64(_) => DataType::U64,

            Self::String(_) => DataType::String,
        }
    }
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<()>
    where
        T: Clone,
    {
        for sdr in self.sub_senders.lock().unwrap().iter() {
            // TODO: We could define a helpful error here with
            // information about which nodes were involved.
            sdr.send(value.clone()).unwrap();
        }
        Ok(())
    }
}

impl Dispatcher {
    /// Subscribe to an upstream slot. Only called during the setup phase.
    pub fn subscribe(&self) -> TaggedReceiver {
        macro_rules! impl_all {
            ($($variant:tt ),*) => {{
                match self {
                    $(Self::$variant(senders) => {
                        let (sdr, rec) = sync_channel(1);
                        senders.lock().unwrap().push(sdr);
                        TaggedReceiver::$variant(rec)
                    })*
                }
            }};
        }
        impl_all!(Bool, F32, F64, I8, I16, I32, I64, String, U8, U16, U32, U64)
    }

    /// Data type of this object.
    pub fn dtype(&self) -> DataType {
        // Name it dtype everywhere
        match self {
            Self::Bool(_) => DataType::Bool,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,

            Self::I8(_) => DataType::I8,
            Self::I16(_) => DataType::I16,
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,

            Self::U8(_) => DataType::U8,
            Self::U16(_) => DataType::U16,
            Self::U32(_) => DataType::U32,
            Self::U64(_) => DataType::U64,

            Self::String(_) => DataType::String,
        }
    }
}

impl TaggedReceiver {
    /// Receive a value. Used when gathering the outputs of a
    /// session. Wire functions match on the variants before
    /// receiving.
    pub fn recv(&self) -> Result<Value> {
        match self {
            Self::Bool(rec) => Ok(Value::Bool(rec.recv()?)),
            Self::F32(rec) => Ok(Value::F32(rec.recv()?)),
            Self::F64(rec) => Ok(Value::F64(rec.recv()?)),

            Self::I8(rec) => Ok(Value::I8(rec.recv()?)),
            Self::I16(rec) => Ok(Value::I16(rec.recv()?)),
            Self::I32(rec) => Ok(Value::I32(rec.recv()?)),
            Self::I64(rec) => Ok(Value::I64(rec.recv()?)),

            Self::U8(rec) => Ok(Value::U8(rec.recv()?)),
            Self::U16(rec) => Ok(Value::U16(rec.recv()?)),
            Self::U32(rec) => Ok(Value::U32(rec.recv()?)),
            Self::U64(rec) => Ok(Value::U64(rec.recv()?)),

            Self::String(rec) => Ok(Value::String(rec.recv()?)),
        }
    }

    /// Data type of this object.
    pub fn dtype(&self) -> DataType {
        // Name it dtype everywhere
        match self {
            Self::Bool(_) => DataType::Bool,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,

            Self::I8(_) => DataType::I8,
            Self::I16(_) => DataType::I16,
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,

            Self::U8(_) => DataType::U8,
            Self::U16(_) => DataType::U16,
            Self::U32(_) => DataType::U32,
            Self::U64(_) => DataType::U64,

            Self::String(_) => DataType::String,
        }
    }
}

pub trait Sendable {
    /// Type which is send over the channel
    // type ArcType;

    fn channel() -> (Sender<Self>, Dispatcher)
    where
        Self: Sized;
}

macro_rules! impl_dispatchable {
    ($ty:ty, $dispatcher:path) => {
        impl Sendable for ArcArrayD<$ty> {
            fn channel() -> (Sender<Self>, Dispatcher)
            where
                Self: Sized,
            {
                let sender = Sender::<Self>::default();
                let disp = $dispatcher(sender.sub_senders.clone());
                (sender, disp)
            }
        }
    };
}

impl_dispatchable!(i8, Dispatcher::I8);
impl_dispatchable!(i16, Dispatcher::I16);
impl_dispatchable!(i32, Dispatcher::I32);
impl_dispatchable!(i64, Dispatcher::I64);

impl_dispatchable!(u8, Dispatcher::U8);
impl_dispatchable!(u16, Dispatcher::U16);
impl_dispatchable!(u32, Dispatcher::U32);
impl_dispatchable!(u64, Dispatcher::U64);

impl_dispatchable!(f32, Dispatcher::F32);
impl_dispatchable!(f64, Dispatcher::F64);
impl_dispatchable!(String, Dispatcher::String);
impl_dispatchable!(bool, Dispatcher::Bool);

impl DataType {
    pub fn from_i32(type_number: i32) -> Result<Self> {
        use tensor_proto::DataType::*;
        let dtype_proto = tensor_proto::DataType::from_i32(type_number)
            .ok_or_else(|| anyhow!("Invalid type number: {}", type_number))?;
        Ok(match dtype_proto {
            // Defined in tensor_proto::DataType (cf onnx.proto)
            BOOL => DataType::Bool,
            DOUBLE => DataType::F64,
            FLOAT => DataType::F32,
            INT8 => DataType::I8,
            INT16 => DataType::I16,
            INT32 => DataType::I32,
            INT64 => DataType::I64,
            STRING => DataType::String,
            UINT8 => DataType::U8,
            UINT16 => DataType::U16,
            UINT32 => DataType::U32,
            UINT64 => DataType::U64,
            val => bail!("Unsupported type `{:?}`", val),
        })
    }
    pub fn from_value_info(value_info: &ValueInfoProto) -> Result<DataType> {
        let err_fn = || {
            anyhow!(
                "Type information is missing from `ValueInfo`: [information hidden for brevity]",
            )
        };
        if let type_proto::Value::TensorType(type_proto::Tensor { elem_type, .. }) = value_info
            .type_
            .as_ref()
            .ok_or_else(err_fn)?
            .value
            .as_ref()
            .ok_or_else(err_fn)?
        {
            Self::from_i32(*elem_type)
        } else {
            Err(err_fn())
        }
    }

    pub fn standard_name(&self) -> &str {
        match self {
            DataType::Bool => "tensor(bool)",

            DataType::I8 => "tensor(int8)",
            DataType::I16 => "tensor(int16)",
            DataType::I32 => "tensor(int32)",
            DataType::I64 => "tensor(int64)",

            DataType::U8 => "tensor(uint8)",
            DataType::U16 => "tensor(uint16)",
            DataType::U32 => "tensor(uint32)",
            DataType::U64 => "tensor(uint64)",

            DataType::F32 => "tensor(float)",
            DataType::F64 => "tensor(double)",

            DataType::String => "tensor(string)",
        }
    }
}
