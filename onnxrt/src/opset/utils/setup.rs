//! Module with traits and functions to simplify implementation work

use crate::channels::value_channel;
use crate::data_types::ValueType;
use crate::opset::utils::*;
use crate::{
    Sendable, Value,
    channels::{Dispatcher, Sender},
};
use std::collections::{HashMap, HashSet};
use std::sync::mpsc::Receiver;

pub trait FromAttributes {
    fn from_attrs(attrs: &Attributes) -> Result<Self>
    where
        Self: Sized;
}

// Deprecated. Use 'setup' directly
pub fn wrap_1_input<In, Out, Op>(input: Receiver<ArcArrayD<In>>, f: Op) -> Result<ExecutionStep>
where
    ArcArrayD<Out>: Sendable,
    In: Sync + Send + 'static,
    Out: Sync + Send + 'static,
    Op: Fn(ArcArrayD<In>) -> Result<ArcArrayD<Out>> + Send + Sync + 'static,
{
    setup((input,), move |(input,)| f(input))
}

pub fn setup<F, InCh, In, Out>(in_channels: InCh, fun: F) -> Result<ExecutionStep>
where
    InCh: InChannels<Items = In> + Send + 'static,
    F: Fn(In) -> Result<Out> + Send + 'static,
    // We must be able to create channels for the output of 'F'
    Out: Clone + SendableOutputs + Send + 'static,
    // The Senders produced for 'Out' must be able to consume 'Out'
    <Out as SendableOutputs>::Senders: Send + 'static,
{
    let (sdrs, recs) = Out::channels();
    let fun = Box::new(move || {
        let inputs = in_channels.recv().unwrap();
        fun(inputs)?.send_via(&sdrs)?;
        Ok(())
    });
    Ok(ExecutionStep { fun, outputs: recs })
}

pub fn setup_variadic_out<F, InCh, In, Out>(
    in_channels: InCh,
    fun: F,
    n_outputs: usize,
) -> Result<ExecutionStep>
where
    InCh: InChannels<Items = In> + Send + 'static,
    F: Fn(In) -> Result<Out> + Send + 'static,
    // We must be able to create channels for the output of 'F'
    Out: Clone + VariadicDispatchable + Send + 'static,
    <Out as VariadicDispatchable>::Senders: Send + 'static,
{
    let (sdrs, recs) = Out::channels(n_outputs);
    let fun = Box::new(move || {
        let inputs = in_channels.recv().unwrap();
        fun(inputs)?.send_via(&sdrs)?;
        Ok(())
    });
    Ok(ExecutionStep { fun, outputs: recs })
}

pub fn setup_variadic_hetero_output<F, InCh, In>(
    in_channels: InCh,
    fun: F,
    dtypes: Vec<(String, ValueType)>,
) -> Result<ExecutionStep>
where
    InCh: InChannels<Items = In> + Send + 'static,
    F: Fn(In) -> Result<Vec<Value>> + Send + 'static,
{
    let mut out_dispatchers = vec![];
    let mut out_senders = vec![];

    for (_name, out_dtype) in &dtypes {
        let (out_sdr, out_disp) = value_channel(out_dtype);
        out_senders.push(out_sdr);
        out_dispatchers.push(out_disp);
    }

    // Validate that we have unique names
    let unique_outputs: HashSet<_> = dtypes.iter().map(|(name, _)| name).collect();
    if unique_outputs.len() != dtypes.len() {
        bail!("Output names must be unique.")
    }

    let fun = Box::new(move || {
        let inputs = in_channels.recv().unwrap();
        for (val, sdr) in fun(inputs)?.into_iter().zip(&out_senders) {
            sdr.send(val)?;
        }
        Ok(())
    });
    Ok(ExecutionStep {
        fun,
        outputs: out_dispatchers,
    })
}

pub trait VariadicDispatchable {
    type Senders;

    fn channels(n: usize) -> (Self::Senders, Vec<Dispatcher>);

    /// Send the outputs through the created channels. This could be
    /// it's own trait on `Self::Senders` but it might be easier to
    /// have it all in one place.
    fn send_via(self, _senders: &Self::Senders) -> Result<()>
    where
        Self: Sized;
}

impl<T> VariadicDispatchable for Vec<T>
where
    T: IntoArc,
    <T as IntoArc>::Output: Sendable + Clone,
{
    type Senders = Vec<Sender<<T as IntoArc>::Output>>;

    fn channels(n: usize) -> (Self::Senders, Vec<Dispatcher>) {
        let mut sdrs = vec![];
        let mut recs = vec![];

        for _ in 0..n {
            let (sdr, rec) = <T as IntoArc>::Output::channel();
            sdrs.push(sdr);
            recs.push(rec);
        }
        (sdrs, recs)
    }

    fn send_via(self, senders: &Self::Senders) -> Result<()> {
        for (item, sdr) in self.into_iter().zip(senders.iter()) {
            sdr.send(item.into_arc()?)?;
        }
        Ok(())
    }
}

/// Output types of a compute function which can be processed by the `setup` function
pub trait SendableOutputs {
    type Senders;

    /// Create `Sender`s and `Receiver`s (`Dispatcher`s) for the implementing output type.
    fn channels() -> (Self::Senders, Vec<Dispatcher>);

    /// Send the outputs through the created channels. This could be
    /// it's own trait on `Self::Senders` but it might be easier to
    /// have it all in one place.
    fn send_via(self, _senders: &Self::Senders) -> Result<()>
    where
        Self: Sized;
}

impl<A> SendableOutputs for A
where
    A: IntoArc,
    <A as IntoArc>::Output: Sendable + Clone,
{
    type Senders = Sender<<A as IntoArc>::Output>;

    fn channels() -> (Self::Senders, Vec<Dispatcher>) {
        let (sdr, disp) = <A as IntoArc>::Output::channel();
        (sdr, vec![disp])
    }

    fn send_via(self, sender: &Self::Senders) -> Result<()> {
        sender.send(self.into_arc()?)?;
        Ok(())
    }
}

macro_rules! impl_SendableOutputs_for_tuples {
    ($($ty:tt),* | $($name:ident),* | $($idx:tt),*) => {
        impl<$($ty),*> SendableOutputs for ($($ty,)*)
        where
            $($ty: IntoArc, <$ty as IntoArc>::Output: Sendable + Clone),*
        {
            type Senders = ($(Sender<<$ty as IntoArc>::Output>,)*);

            fn channels() -> (Self::Senders, Vec<Dispatcher>) {
                $(
                    let $name = <$ty as IntoArc>::Output::channel();
                )*
                (($($name.0,)*), vec![$($name.1),*])
            }

            fn send_via(self, sender: &Self::Senders) -> Result<()> {
                $(sender.$idx.send(self.$idx.into_arc()?)?;)*
                Ok(())
            }
        }
    };
}

// Implement for up to 6 outputs
impl_SendableOutputs_for_tuples!(A | a | 0);
impl_SendableOutputs_for_tuples!(A, B | a, b | 0, 1);
impl_SendableOutputs_for_tuples!(A, B, C | a, b, c | 0, 1, 2);
impl_SendableOutputs_for_tuples!(A, B, C, D | a, b, c, d | 0, 1, 2, 3);
impl_SendableOutputs_for_tuples!(A, B, C, D, E | a, b, c, d, e | 0, 1, 2, 3, 4);
impl_SendableOutputs_for_tuples!(A, B, C, D, E, F | a, b, c, d, e, f | 0, 1, 2, 3, 4, 5);

pub trait InChannel {
    type Item;
    fn recv(&self) -> Result<Self::Item>;
}

impl<T> InChannel for Receiver<T> {
    type Item = T;

    fn recv(&self) -> Result<Self::Item> {
        Ok(self.recv()?)
    }
}

impl InChannel for TaggedReceiver {
    type Item = Value;

    fn recv(&self) -> Result<Self::Item> {
        self.recv()
    }
}

impl<T> InChannel for Option<T>
where
    T: InChannel,
{
    type Item = Option<T::Item>;

    fn recv(&self) -> Result<Self::Item> {
        match self {
            Some(rec) => Ok(Some(rec.recv()?)),
            None => Ok(None),
        }
    }
}

impl<T> InChannel for Vec<T>
where
    T: InChannel,
{
    type Item = Vec<T::Item>;

    fn recv(&self) -> Result<Self::Item> {
        let mut out = Vec::with_capacity(self.len());
        for el in self.iter() {
            out.push(el.recv()?);
        }
        Ok(out)
    }
}

impl<T> InChannel for HashMap<String, T>
where
    T: InChannel,
{
    type Item = HashMap<String, T::Item>;

    fn recv(&self) -> Result<Self::Item> {
        let mut res = Self::Item::new();
        for (name, item) in self.iter() {
            res.insert(name.to_owned(), item.recv()?);
        }
        Ok(res)
    }
}

pub trait InChannels {
    type Items;
    fn recv(&self) -> Result<Self::Items>;
}

// Implementation for scalars (rather than tuples)
impl<T> InChannels for T
where
    T: InChannel,
{
    type Items = T::Item;

    fn recv(&self) -> Result<Self::Items> {
        self.recv()
    }
}

/// Operators such as 'Constant' do not have inputs.
impl InChannels for () {
    type Items = ();

    fn recv(&self) -> Result<Self::Items> {
        Ok(())
    }
}

macro_rules! impl_InChannels {
    ($($Ts:tt),* | Z | $($idx:tt),* ) => {
        impl<$( $Ts ,)* Z> InChannels for ($($Ts,)* Z)
        where
            $($Ts: InChannel,)*
            Z: InChannel,
        {
            type Items = ($(<$Ts>::Item,)* Z::Item);

            fn recv(&self) -> Result<Self::Items> {
                Ok(($(self.$idx.recv()?,)* ))
            }
        }
    };
}

impl<A> InChannels for (A,)
where
    A: InChannel,
{
    type Items = (A::Item,);

    fn recv(&self) -> Result<Self::Items> {
        Ok((self.0.recv()?,))
    }
}

impl_InChannels!(T1 | Z | 0, 1);
impl_InChannels!(T1, T2 | Z | 0, 1, 2);
impl_InChannels!(T1, T2, T3 | Z | 0, 1, 2, 3);
impl_InChannels!(T1, T2, T3, T4 | Z | 0, 1, 2, 3, 4);
impl_InChannels!(T1, T2, T3, T4, T5 | Z | 0, 1, 2, 3, 4, 5);

pub struct MappedIntoI64<T> {
    inner: T,
}

impl Integer {
    pub fn into_i64(self) -> MappedIntoI64<Self> {
        MappedIntoI64 { inner: self }
    }
}

impl OnnxIndex {
    pub fn into_i64(self) -> MappedIntoI64<Self> {
        MappedIntoI64 { inner: self }
    }
}

impl InChannel for MappedIntoI64<Integer> {
    type Item = ArrayD<i64>;

    fn recv(&self) -> Result<Self::Item> {
        use Integer::*;
        match &self.inner {
            I8(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
            I16(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
            I32(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
            I64(rec) => Ok(rec.recv()?.to_owned()),
            U8(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
            U16(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
            U32(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
            U64(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
        }
    }
}

impl InChannel for MappedIntoI64<OnnxIndex> {
    type Item = ArrayD<i64>;

    fn recv(&self) -> Result<Self::Item> {
        use OnnxIndex::*;
        match &self.inner {
            I32(rec) => Ok(rec.recv()?.mapv(|e| e as i64)),
            I64(rec) => Ok(rec.recv()?.to_owned()),
        }
    }
}
