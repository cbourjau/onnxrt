//! Module with traits and functions to simplify implementation work

use std::sync::mpsc::Receiver;

use crate::opset::utils::*;
use crate::{
    edges::{Dispatcher, Sender},
    Sendable,
};

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
    T: Sendable + Clone,
{
    type Senders = Vec<Sender<T>>;

    fn channels(n: usize) -> (Self::Senders, Vec<Dispatcher>) {
        let mut sdrs = vec![];
        let mut recs = vec![];

        for _ in 0..n {
            let (sdr, rec) = T::channel();
            sdrs.push(sdr);
            recs.push(rec);
        }
        (sdrs, recs)
    }

    fn send_via(self, senders: &Self::Senders) -> Result<()> {
        for (item, sdr) in self.into_iter().zip(senders.iter()) {
            sdr.send(item)?;
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
    A: Sendable + Clone,
{
    type Senders = Sender<A>;

    fn channels() -> (Self::Senders, Vec<Dispatcher>) {
        let (sdr, disp) = A::channel();
        (sdr, vec![disp])
    }

    fn send_via(self, sender: &Self::Senders) -> Result<()> {
        sender.send(self)?;
        Ok(())
    }
}

macro_rules! impl_SendableOutputs_for_tuples {
    ($($ty:tt),* | $($name:ident),* | $($idx:tt),*) => {
        impl<$($ty),*> SendableOutputs for ($($ty,)*)
        where
            $($ty: Sendable + Clone),*
        {
            type Senders = ($(Sender<$ty>,)*);

            fn channels() -> (Self::Senders, Vec<Dispatcher>) {
                $(
                    let $name = $ty::channel();
                )*
                (($($name.0,)*), vec![$($name.1),*])
            }

            fn send_via(self, sender: &Self::Senders) -> Result<()> {
                $(sender.$idx.send(self.$idx)?;)*
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

pub trait LastInChannel {
    type Item;
    fn recv(&self) -> Result<Self::Item>;
}

impl<T> InChannel for Receiver<T> {
    type Item = T;

    fn recv(&self) -> Result<Self::Item> {
        Ok(self.recv()?)
    }
}

impl<T> InChannel for Option<Receiver<T>> {
    type Item = Option<T>;

    fn recv(&self) -> Result<Self::Item> {
        match self {
            Some(rec) => Ok(Some(rec.recv()?)),
            None => Ok(None),
        }
    }
}

impl<T> LastInChannel for T
where
    T: InChannel,
{
    type Item = T::Item;

    fn recv(&self) -> Result<Self::Item> {
        self.recv()
    }
}

// impl Input for ArcArrayD<f32> {}
impl<T> LastInChannel for Vec<T>
where
    T: InChannel,
{
    type Item = Vec<T::Item>;

    fn recv(&self) -> Result<Self::Item> {
        self.iter().map(|el| el.recv()).collect()
    }
}

pub trait InChannels {
    type Items;
    fn recv(&self) -> Result<Self::Items>;
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
            Z: LastInChannel,
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
    A: LastInChannel,
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
