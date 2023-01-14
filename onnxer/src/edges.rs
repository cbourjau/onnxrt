use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};

use crate::protos::onnx_proto3::{type_proto, ValueInfoProto};
use crate::ArcArrayD;

/// Data types which may be exchange between nodes.
#[derive(Debug, PartialEq)]
pub enum DataType {
    Bool,
    F32,
    F64,
    I32,
    I64,
    String,
}

/// Enum capable of representing any possible value. Primarily used in `Session::run`.
/// A concrete tensor
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(ArcArrayD<bool>),
    F32(ArcArrayD<f32>),
    F64(ArcArrayD<f64>),
    I32(ArcArrayD<i32>),
    I64(ArcArrayD<i64>),
    String(ArcArrayD<String>),
}

/// Sender Consuming a `Value` and forwarding it to the `Sender` of the correct type.
#[derive(Debug)]
pub enum ValueSender {
    Bool(Sender<ArcArrayD<bool>>),
    F32(Sender<ArcArrayD<f32>>),
    F64(Sender<ArcArrayD<f64>>),
    I32(Sender<ArcArrayD<i32>>),
    I64(Sender<ArcArrayD<i64>>),
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
    I32(Arc<Mutex<Vec<SyncSender<ArcArrayD<i32>>>>>),
    I64(Arc<Mutex<Vec<SyncSender<ArcArrayD<i64>>>>>),
    String(Arc<Mutex<Vec<SyncSender<ArcArrayD<String>>>>>),
}

/// Object obtained by subscribing to a `Dispatcher`. Used inside
/// "wire" functions and to obtain the final values of a session. The
/// values of each variant are typed std receivers.
#[derive(Debug)]
pub enum EnumReceiver {
    Bool(Receiver<ArcArrayD<bool>>),
    F32(Receiver<ArcArrayD<f32>>),
    F64(Receiver<ArcArrayD<f64>>),
    I32(Receiver<ArcArrayD<i32>>),
    I64(Receiver<ArcArrayD<i64>>),
    String(Receiver<ArcArrayD<String>>),
}

/// Create a typed sender and dispatcher. Used inside "wire" functions.
pub fn channel<T: Dispatchable>() -> (Sender<T>, Dispatcher) {
    T::channel()
}

/// Create a value-channel used as an entry point in a session
pub fn value_channel(dtype: &DataType) -> (ValueSender, Dispatcher) {
    match dtype {
        DataType::Bool => {
            let (sdr, disp) = channel();
            (ValueSender::Bool(sdr), disp)
        }
        DataType::F32 => {
            let (sdr, disp) = channel();
            (ValueSender::F32(sdr), disp)
        }
        DataType::F64 => {
            let (sdr, disp) = channel();
            (ValueSender::F64(sdr), disp)
        }
        DataType::I32 => {
            let (sdr, disp) = channel();
            (ValueSender::I32(sdr), disp)
        }
        DataType::I64 => {
            let (sdr, disp) = channel();
            (ValueSender::I64(sdr), disp)
        }
        DataType::String => {
            let (sdr, disp) = channel();
            (ValueSender::String(sdr), disp)
        }
    }
}

impl ValueSender {
    pub fn send(&self, value: Value) -> Result<()> {
        match (self, value) {
            (Self::Bool(sdr), Value::Bool(val)) => sdr.send(val),
            (Self::F32(sdr), Value::F32(val)) => sdr.send(val),
            (Self::F64(sdr), Value::F64(val)) => sdr.send(val),
            (Self::I32(sdr), Value::I32(val)) => sdr.send(val),
            (Self::I64(sdr), Value::I64(val)) => sdr.send(val),
            (Self::String(sdr), Value::String(val)) => sdr.send(val),
            _ => Err(anyhow!("Unexpected data type")),
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
    pub fn subscribe(&self) -> EnumReceiver {
        match self {
            Self::Bool(senders) => {
                let (sdr, rec) = sync_channel(1);
                senders.lock().unwrap().push(sdr);
                EnumReceiver::Bool(rec)
            }
            Self::F32(senders) => {
                let (sdr, rec) = sync_channel(1);
                senders.lock().unwrap().push(sdr);
                EnumReceiver::F32(rec)
            }
            Self::F64(senders) => {
                let (sdr, rec) = sync_channel(1);
                senders.lock().unwrap().push(sdr);
                EnumReceiver::F64(rec)
            }
            Self::I32(senders) => {
                let (sdr, rec) = sync_channel(1);
                senders.lock().unwrap().push(sdr);
                EnumReceiver::I32(rec)
            }
            Self::I64(senders) => {
                let (sdr, rec) = sync_channel(1);
                senders.lock().unwrap().push(sdr);
                EnumReceiver::I64(rec)
            }
            Self::String(senders) => {
                let (sdr, rec) = sync_channel(1);
                senders.lock().unwrap().push(sdr);
                EnumReceiver::String(rec)
            }
        }
    }

    /// Data type of this object.
    pub fn dtype(&self) -> DataType {
        // Name it dtype everywhere
        match self {
            Self::Bool(_) => DataType::Bool,
            Self::F32(_) => DataType::F32,
            Self::F64(_) => DataType::F64,
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,
            Self::String(_) => DataType::String,
        }
    }
}

impl EnumReceiver {
    /// Receive a value. Used when gathering the outputs of a
    /// session. Wire functions match on the variants before
    /// receiving.
    pub fn recv(&self) -> Result<Value> {
        match self {
            Self::Bool(rec) => Ok(Value::Bool(rec.recv()?)),
            Self::F32(rec) => Ok(Value::F32(rec.recv()?)),
            Self::F64(rec) => Ok(Value::F64(rec.recv()?)),
            Self::I32(rec) => Ok(Value::I32(rec.recv()?)),
            Self::I64(rec) => Ok(Value::I64(rec.recv()?)),
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
            Self::I32(_) => DataType::I32,
            Self::I64(_) => DataType::I64,
            Self::String(_) => DataType::String,
        }
    }
}

pub trait Dispatchable {
    fn channel() -> (Sender<Self>, Dispatcher)
    where
        Self: Sized;
}

macro_rules! impl_dispatchable {
    ($ty:ty, $dispatcher:path) => {
        impl Dispatchable for ArcArrayD<$ty> {
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
impl_dispatchable!(i32, Dispatcher::I32);
impl_dispatchable!(i64, Dispatcher::I64);
impl_dispatchable!(f32, Dispatcher::F32);
impl_dispatchable!(f64, Dispatcher::F64);
impl_dispatchable!(String, Dispatcher::String);
impl_dispatchable!(bool, Dispatcher::Bool);

impl DataType {
    pub fn from_i32(type_number: i32) -> Result<Self> {
        Ok(match type_number {
            // Defined in tensor_proto::DataType
            1 => DataType::F32,
            11 => DataType::F64,
            6 => DataType::I32,
            7 => DataType::I64,
            8 => DataType::String,
            9 => DataType::Bool,
            val => bail!("Type number `{}`", val),
        })
    }
    pub fn from_value_info(value_info: &ValueInfoProto) -> Result<DataType> {
        let err_fn = || {
            anyhow!(
                "Type information is missing from `ValueInfo`: {:#?}",
                value_info
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
}
