//! Module with traits and functions to simplify implementation work

use std::sync::mpsc::Receiver;

use crate::opset::utils::*;
use crate::{
    edges::{Dispatcher, Sender},
    Dispatchable,
};

pub trait FromAttributes {
    fn from_attrs(attrs: &Attributes) -> Result<Self>
    where
        Self: Sized;
}

// Deprecated. Use 'setup' directly
pub fn wrap_1_input<In, Out, Op>(input: Receiver<ArcArrayD<In>>, f: Op) -> Result<ExecutionStep>
where
    ArcArrayD<Out>: Dispatchable,
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
    Out: Clone + Dispatchables + Send + 'static,
    // The Senders produced for 'Out' must be able to consume 'Out'
    <Out as Dispatchables>::Senders: MultiOutputSender<Items = Out> + Send + 'static,
{
    let (sdrs, recs) = Out::channels();
    let fun = Box::new(move || {
        let inputs = in_channels.recv().unwrap();
        let out = fun(inputs)?;
        sdrs.send(out)?;
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
    <Out as VariadicDispatchable>::Senders: MultiOutputSender<Items = Out> + Send + 'static,
{
    let (sdrs, recs) = Out::channels(n_outputs);
    let fun = Box::new(move || {
        let inputs = in_channels.recv().unwrap();
        let out = fun(inputs)?;
        sdrs.send(out)?;
        Ok(())
    });
    Ok(ExecutionStep { fun, outputs: recs })
}

pub trait VariadicDispatchable {
    type Senders;

    fn channels(n: usize) -> (Self::Senders, Vec<Dispatcher>);
}

impl<T> VariadicDispatchable for Vec<T>
where
    T: Dispatchable + Clone,
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
}

pub trait MultiOutputSender {
    type Items;

    fn send(&self, items: Self::Items) -> Result<()>;
}

impl<A> MultiOutputSender for Sender<A>
where
    A: Clone + Dispatchable,
{
    type Items = A;

    fn send(&self, items: Self::Items) -> Result<()> {
        self.send(items)?;
        Ok(())
    }
}

impl<A> MultiOutputSender for (Sender<A>,)
where
    A: Clone + Dispatchable,
{
    type Items = (A,);

    fn send(&self, items: Self::Items) -> Result<()> {
        self.0.send(items.0)?;
        Ok(())
    }
}

impl<A, B> MultiOutputSender for (Sender<A>, Sender<B>)
where
    A: Clone,
    B: Clone,
{
    type Items = (A, B);

    fn send(&self, items: Self::Items) -> Result<()> {
        self.0.send(items.0)?;
        self.1.send(items.1)?;
        Ok(())
    }
}

impl<A, B, C, D> MultiOutputSender for (Sender<A>, Sender<B>, Sender<C>, Sender<D>)
where
    A: Clone,
    B: Clone,
    C: Clone,
    D: Clone,
{
    type Items = (A, B, C, D);

    fn send(&self, items: Self::Items) -> Result<()> {
        self.0.send(items.0)?;
        self.1.send(items.1)?;
        self.2.send(items.2)?;
        self.3.send(items.3)?;
        Ok(())
    }
}

impl<A> MultiOutputSender for Vec<Sender<A>>
where
    A: Clone,
{
    type Items = Vec<A>;

    fn send(&self, items: Self::Items) -> Result<()> {
        for (item, sdr) in items.into_iter().zip(self.iter()) {
            sdr.send(item)?;
        }
        Ok(())
    }
}

pub trait Dispatchables {
    type Senders;

    fn channels() -> (Self::Senders, Vec<Dispatcher>);
}

impl<A> Dispatchables for A
where
    A: Dispatchable + Clone,
{
    type Senders = Sender<A>;

    fn channels() -> (Self::Senders, Vec<Dispatcher>) {
        let (sdr, disp) = A::channel();
        (sdr, vec![disp])
    }
}

impl<A> Dispatchables for (A,)
where
    A: Dispatchable + Clone,
{
    type Senders = (Sender<A>,);

    fn channels() -> (Self::Senders, Vec<Dispatcher>) {
        let (sdr, disp) = A::channel();
        ((sdr,), vec![disp])
    }
}

impl<A, B> Dispatchables for (A, B)
where
    A: Dispatchable + Clone,
    B: Dispatchable + Clone,
{
    type Senders = (Sender<A>, Sender<B>);

    fn channels() -> (Self::Senders, Vec<Dispatcher>) {
        let pair_a = A::channel();
        let pair_b = B::channel();
        ((pair_a.0, pair_b.0), vec![pair_a.1, pair_b.1])
    }
}

impl<A, B, C, D> Dispatchables for (A, B, C, D)
where
    A: Dispatchable + Clone,
    B: Dispatchable + Clone,
    C: Dispatchable + Clone,
    D: Dispatchable + Clone,
{
    type Senders = (Sender<A>, Sender<B>, Sender<C>, Sender<D>);

    fn channels() -> (Self::Senders, Vec<Dispatcher>) {
        let pair_a = A::channel();
        let pair_b = B::channel();
        let pair_c = C::channel();
        let pair_d = D::channel();
        (
            (pair_a.0, pair_b.0, pair_c.0, pair_d.0),
            vec![pair_a.1, pair_b.1, pair_c.1, pair_d.1],
        )
    }
}

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
