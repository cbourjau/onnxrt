use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::session::DataType;
use crate::{ArcArrayD, RcReceiver, Value};

/// The `Dispatcher` is responsible for distributing the single output
/// of an operation to its `N` consumers.
#[derive(Debug, Clone)]
pub struct Dispatcher {
    senders: ArcSenders,
}

/// Non-generic version to send input values from session
#[derive(Debug, Clone)]
pub struct ValueDispatcher {
    senders: ArcSenders,
}

#[derive(Debug, Clone)]
enum ArcSenders {
    I32(ArcSendersGen<ArcArrayD<i32>>),
    I64(ArcSendersGen<ArcArrayD<i64>>),
    F32(ArcSendersGen<ArcArrayD<f32>>),
    F64(ArcSendersGen<ArcArrayD<f64>>),
    String(ArcSendersGen<ArcArrayD<String>>),
    Bool(ArcSendersGen<ArcArrayD<bool>>),
}

type ArcSendersGen<T> = Arc<Mutex<Vec<SyncSender<T>>>>;

#[derive(Debug, Clone)]
pub struct DispatcherSender<T> {
    senders: ArcSendersGen<T>,
}

pub trait Dispatchable {
    fn channel() -> (DispatcherSender<Self>, Dispatcher)
    where
        Self: Sized;
}

macro_rules! impl_dispatchable {
    ($ty:ty, $dtype:path) => {
        impl Dispatchable for ArcArrayD<$ty> {
            fn channel() -> (DispatcherSender<Self>, Dispatcher)
            where
                Self: Sized,
            {
                let senders_gen = ArcSendersGen::<Self>::default();
                let senders = $dtype(senders_gen.clone());
                let dispatcher = Dispatcher { senders };
                (
                    DispatcherSender {
                        senders: senders_gen,
                    },
                    dispatcher,
                )
            }
        }
    };
}

impl_dispatchable!(f32, ArcSenders::F32);
impl_dispatchable!(i32, ArcSenders::I32);
impl_dispatchable!(f64, ArcSenders::F64);
impl_dispatchable!(i64, ArcSenders::I64);
impl_dispatchable!(bool, ArcSenders::Bool);
impl_dispatchable!(String, ArcSenders::String);

impl<T> DispatcherSender<T> {
    pub fn send(&self, arr: T)
    where
        T: Clone,
    {
        for sender in self.senders.lock().unwrap().iter() {
            sender.send(arr.clone()).unwrap();
        }
    }
}

impl Dispatcher {
    /// Create a new dispatcher which receives values through
    /// `receiver` and forwards them to all subscribers. The `name` is
    /// used for error messages only.
    pub fn new(receiver: DataType) -> Self {
        let senders = match receiver {
            DataType::I32 => ArcSenders::I32(Default::default()),
            DataType::I64 => ArcSenders::I64(Default::default()),
            DataType::F32 => ArcSenders::F32(Default::default()),
            DataType::F64 => ArcSenders::F64(Default::default()),
            DataType::String => ArcSenders::String(Default::default()),
            DataType::Bool => ArcSenders::Bool(Default::default()),
        };
        Self { senders }
    }

    pub fn new_like(reciever: &RcReceiver) -> Self {
        match reciever {
            RcReceiver::I32(_) => Self::new(DataType::I32),
            RcReceiver::I64(_) => Self::new(DataType::I64),
            RcReceiver::F32(_) => Self::new(DataType::F32),
            RcReceiver::F64(_) => Self::new(DataType::F64),
            RcReceiver::String(_) => Self::new(DataType::String),
            RcReceiver::Bool(_) => Self::new(DataType::Bool),
        }
    }

    /// Subscribe to this dispatcher.
    ///
    /// The returned `Receiver` will be filled when the `runner` of
    /// this object is called.
    pub fn subscribe(&self) -> RcReceiver {
        match &self.senders {
            ArcSenders::I32(ss) => {
                let (sender, rec) = sync_channel(1);
                ss.lock().unwrap().push(sender);
                RcReceiver::I32(rec)
            }
            ArcSenders::I64(ss) => {
                let (sender, rec) = sync_channel(1);
                ss.lock().unwrap().push(sender);
                RcReceiver::I64(rec)
            }
            ArcSenders::F32(ss) => {
                let (sender, rec) = sync_channel(1);
                ss.lock().unwrap().push(sender);
                RcReceiver::F32(rec)
            }
            ArcSenders::F64(ss) => {
                let (sender, rec) = sync_channel(1);
                ss.lock().unwrap().push(sender);
                RcReceiver::F64(rec)
            }
            ArcSenders::Bool(ss) => {
                let (sender, rec) = sync_channel(1);
                ss.lock().unwrap().push(sender);
                RcReceiver::Bool(rec)
            }
            ArcSenders::String(ss) => {
                let (sender, rec) = sync_channel(1);
                ss.lock().unwrap().push(sender);
                RcReceiver::String(rec)
            }
        }
    }

    pub fn into_value_dispatcher(self) -> ValueDispatcher {
        ValueDispatcher {
            senders: self.senders,
        }
    }

    pub fn data_type(&self) -> DataType {
        match self.senders {
            ArcSenders::I32(_) => DataType::I32,
            ArcSenders::I64(_) => DataType::I64,
            ArcSenders::F32(_) => DataType::F32,
            ArcSenders::F64(_) => DataType::F64,
            ArcSenders::String(_) => DataType::String,
            ArcSenders::Bool(_) => DataType::Bool,
        }
    }
}

impl ValueDispatcher {
    pub fn send(&self, value: Value) -> Result<()> {
        match (&self.senders, value) {
            (ArcSenders::I32(senders), Value::I32(val)) => {
                for sender in senders.lock().unwrap().iter() {
                    sender.send(val.clone())?;
                }
            }
            (ArcSenders::I64(senders), Value::I64(val)) => {
                for sender in senders.lock().unwrap().iter() {
                    sender.send(val.clone())?;
                }
            }
            (ArcSenders::F32(senders), Value::F32(val)) => {
                for sender in senders.lock().unwrap().iter() {
                    sender.send(val.clone())?;
                }
            }
            (ArcSenders::F64(senders), Value::F64(val)) => {
                for sender in senders.lock().unwrap().iter() {
                    sender.send(val.clone())?;
                }
            }
            (ArcSenders::Bool(senders), Value::Bool(val)) => {
                for sender in senders.lock().unwrap().iter() {
                    sender.send(val.clone())?;
                }
            }
            (ArcSenders::String(senders), Value::String(val)) => {
                for sender in senders.lock().unwrap().iter() {
                    sender.send(val.clone())?;
                }
            }
            // The types of the receiver and sender must always match
            _ => panic!("Impossible combination of receiver and sender types"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_value_dispatcher() {
        let dispatcher = Dispatcher::new(DataType::I32);
        let value_dispatcher = dispatcher.clone().into_value_dispatcher();
        let value = Value::I32(arr1(&[]).into_dyn().into());

        // No receiver
        value_dispatcher.send(value.clone()).unwrap();

        let rx1 = dispatcher.subscribe();
        let rx2 = dispatcher.subscribe();

        value_dispatcher.send(value).unwrap();
        rx1.recv().unwrap();
        rx2.recv().unwrap();
    }

    #[test]
    fn test_typed_dispatcher() {
        let (sender, dispatcher) = ArcArrayD::<i32>::channel();
        let value: ArcArrayD<i32> = arr1(&[]).into_dyn().into();

        // No receiver
        sender.send(value.clone());

        let rx1 = dispatcher.subscribe();
        let rx2 = dispatcher.subscribe();
        if let ArcSenders::I32(senders) = dispatcher.senders {
            assert_eq!(senders.lock().unwrap().len(), 2);
        } else {
            panic!()
        }
        sender.send(value);
        rx1.recv().unwrap();
        rx2.recv().unwrap();
    }
}
