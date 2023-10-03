use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};

use crate::data_types::{MapType, SequenceType, TensorType, ValueType};
use crate::{ArcArrayD, DataType, Value};

#[derive(Default, Debug, Clone)]
struct Channel<T> {
    /// Senders feeding into subscribers
    senders: Arc<Mutex<Vec<SyncSender<T>>>>,
}

/// Typed sender. Used to send values of known types *out* of "setup" functions.
#[derive(Default, Debug, Clone)]
pub struct Sender<T>(Channel<T>);

/// Sender Consuming a `Value` and forwarding it to the `Sender` of the correct type.
#[derive(Debug)]
pub struct ValueSender(TaggedChannel);

/// Object returned by "setup" functions. Only used during the setup
/// phase. It connects one output to N downstream inputs. The
/// `Dispatcher` is not itself a receiver; it is only a proxy
/// object. Downstream nodes can subscribe to this object to receive
/// outputs of previous nodes. It serves as a T-channel where all
/// consumers receive a clone of the sent object.
pub struct Dispatcher(TaggedChannel);

#[derive(Debug)]
enum TaggedChannel {
    Bool(Channel<ArcArrayD<bool>>),
    F32(Channel<ArcArrayD<f32>>),
    F64(Channel<ArcArrayD<f64>>),

    I8(Channel<ArcArrayD<i8>>),
    I16(Channel<ArcArrayD<i16>>),
    I32(Channel<ArcArrayD<i32>>),
    I64(Channel<ArcArrayD<i64>>),

    U8(Channel<ArcArrayD<u8>>),
    U16(Channel<ArcArrayD<u16>>),
    U32(Channel<ArcArrayD<u32>>),
    U64(Channel<ArcArrayD<u64>>),

    String(Channel<ArcArrayD<String>>),
}

#[derive(Debug)]
pub enum TaggedReceiver {
    Tensor(TaggedTensorReceiver),
    Sequence(TaggedSequenceReceiver),
    Map,
}

/// Object obtained by subscribing to a `Dispatcher`. Used inside
/// "wire" functions and to obtain the final values of a session. The
/// values of each variant are typed std receivers.
#[derive(Debug)]
pub enum TaggedTensorReceiver {
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

#[derive(Debug)]
pub enum TaggedSequenceReceiver {
    Bool(Receiver<Vec<ArcArrayD<bool>>>),
    F32(Receiver<Vec<ArcArrayD<f32>>>),
    F64(Receiver<Vec<ArcArrayD<f64>>>),

    I8(Receiver<Vec<ArcArrayD<i8>>>),
    I16(Receiver<Vec<ArcArrayD<i16>>>),
    I32(Receiver<Vec<ArcArrayD<i32>>>),
    I64(Receiver<Vec<ArcArrayD<i64>>>),

    U8(Receiver<Vec<ArcArrayD<u8>>>),
    U16(Receiver<Vec<ArcArrayD<u16>>>),
    U32(Receiver<Vec<ArcArrayD<u32>>>),
    U64(Receiver<Vec<ArcArrayD<u64>>>),

    String(Receiver<Vec<ArcArrayD<String>>>),
}

/// Create a value-channel used as an entry point in a session
pub fn value_channel(ty: &ValueType) -> (ValueSender, Dispatcher) {
    match ty {
        ValueType::Tensor(TensorType(dtype)) => match dtype {
            DataType::Bool => ArcArrayD::<bool>::value_channel(),
            DataType::I8 => ArcArrayD::<i8>::value_channel(),
            DataType::I16 => ArcArrayD::<i16>::value_channel(),
            DataType::I32 => ArcArrayD::<i32>::value_channel(),
            DataType::I64 => ArcArrayD::<i64>::value_channel(),
            DataType::U8 => ArcArrayD::<u8>::value_channel(),
            DataType::U16 => ArcArrayD::<u16>::value_channel(),
            DataType::U32 => ArcArrayD::<u32>::value_channel(),
            DataType::U64 => ArcArrayD::<u64>::value_channel(),
            DataType::F32 => ArcArrayD::<f32>::value_channel(),
            DataType::F64 => ArcArrayD::<f64>::value_channel(),
            DataType::String => ArcArrayD::<String>::value_channel(),
        },
        ValueType::Sequence(SequenceType::Tensor(_)) => {
            unimplemented!("'Sequence's are not yet supported")
        }
        ValueType::Map(MapType { .. }) => {
            unimplemented!("'Map's are not yet supported")
        }
    }
}

impl ValueSender {
    pub fn send(&self, value: Value) -> Result<()> {
        macro_rules! map_vars {
            ($($var:tt),*) => {
                match (self, value) {
                    $(
                        (Self(TaggedChannel::$var(sdr)), Value::$var(val)) => {
                            sdr.send(val)
                        },
                    )*
                        (sdr, val) => Err(anyhow!("Expected {:?}, found {:?}", sdr, val)),
                }
            }
        }

        map_vars!(Bool, F32, F64, I8, I16, I32, I64, U8, U16, U32, U64, String)
    }

    pub fn tensor_dtype(&self) -> Result<DataType> {
        self.0.tensor_dtype()
    }

    pub fn value_type(&self) -> Result<ValueType> {
        unimplemented!()
    }
}

impl<T> Sender<T> {
    pub fn send(&self, value: T) -> Result<()>
    where
        T: Clone,
    {
        self.0.send(value)
    }
}

impl<T> Channel<T> {
    pub fn send(&self, value: T) -> Result<()>
    where
        T: Clone,
    {
        for sdr in self.senders.lock().unwrap().iter() {
            sdr.send(value.clone()).unwrap();
        }
        Ok(())
    }
}

impl TaggedChannel {
    /// Subscribe to an upstream slot. Only called during the setup phase.
    pub fn subscribe(&self) -> TaggedReceiver {
        macro_rules! impl_all {
            ($($variant:tt ),*) => {{
                match self {
                    $(Self::$variant(Channel{senders}) => {
                        let (sdr, rec) = sync_channel(1);
                        senders.lock().unwrap().push(sdr);
                        TaggedReceiver::Tensor(TaggedTensorReceiver::$variant(rec))
                    })*
                }
            }};
        }
        impl_all!(Bool, F32, F64, I8, I16, I32, I64, String, U8, U16, U32, U64)
    }

    /// Data type of this object.
    pub fn tensor_dtype(&self) -> Result<DataType> {
        // Name it dtype everywhere
        Ok(match self {
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
        })
    }
}

impl Dispatcher {
    /// Subscribe to an upstream slot. Only called during the setup phase.
    pub fn subscribe(&self) -> TaggedReceiver {
        self.0.subscribe()
    }

    /// Data type of this object.
    pub fn tensor_dtype(&self) -> Result<DataType> {
        self.0.tensor_dtype()
    }
}

impl TaggedReceiver {
    /// Receive a value. Used when gathering the outputs of a
    /// session. Wire functions match on the variants before
    /// receiving.
    pub fn recv(&self) -> Result<Value> {
        match self {
            Self::Tensor(tensor) => {
                use TaggedTensorReceiver::*;
                match tensor {
                    Bool(tensor) => Ok(Value::Bool(tensor.recv()?)),
                    F32(tensor) => Ok(Value::F32(tensor.recv()?)),
                    F64(tensor) => Ok(Value::F64(tensor.recv()?)),

                    I8(tensor) => Ok(Value::I8(tensor.recv()?)),
                    I16(tensor) => Ok(Value::I16(tensor.recv()?)),
                    I32(tensor) => Ok(Value::I32(tensor.recv()?)),
                    I64(tensor) => Ok(Value::I64(tensor.recv()?)),

                    U8(tensor) => Ok(Value::U8(tensor.recv()?)),
                    U16(tensor) => Ok(Value::U16(tensor.recv()?)),
                    U32(tensor) => Ok(Value::U32(tensor.recv()?)),
                    U64(tensor) => Ok(Value::U64(tensor.recv()?)),

                    String(tensor) => Ok(Value::String(tensor.recv()?)),
                }
            }
            Self::Sequence(_) => unimplemented!(),
            Self::Map => unimplemented!(),
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
    /// Create a new empty `TaggedDispatcher` (second element) fed by
    /// the `Dispatcher` returned as the first element.
    fn channel() -> (Sender<Self>, Dispatcher)
    where
        Self: Sized;

    /// Create a `ValueSender` for objects of type `Self`.
    fn value_channel() -> (ValueSender, Dispatcher);
}

macro_rules! impl_sendable {
    ($ty:ty, $dispatcher:path) => {
        impl Sendable for ArcArrayD<$ty> {
            fn channel() -> (Sender<Self>, Dispatcher)
            where
                Self: Sized,
            {
                let sender = Sender::<Self>::default();
                let disp = Dispatcher($dispatcher(sender.0.clone()));
                (sender, disp)
            }
            fn value_channel() -> (ValueSender, Dispatcher) {
                let channel = Channel::<Self>::default();
                let sdr = ValueSender($dispatcher(channel.clone()));
                let disp = Dispatcher($dispatcher(channel));
                (sdr, disp)
            }
        }
    };
}

impl_sendable!(i8, TaggedChannel::I8);
impl_sendable!(i16, TaggedChannel::I16);
impl_sendable!(i32, TaggedChannel::I32);
impl_sendable!(i64, TaggedChannel::I64);

impl_sendable!(u8, TaggedChannel::U8);
impl_sendable!(u16, TaggedChannel::U16);
impl_sendable!(u32, TaggedChannel::U32);
impl_sendable!(u64, TaggedChannel::U64);

impl_sendable!(f32, TaggedChannel::F32);
impl_sendable!(f64, TaggedChannel::F64);
impl_sendable!(String, TaggedChannel::String);
impl_sendable!(bool, TaggedChannel::Bool);
