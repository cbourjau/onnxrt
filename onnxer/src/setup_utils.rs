//! Module with traits and functions to simplify implementation work

use std::sync::mpsc::Receiver;

use crate::{
    edges::{Dispatcher, Sender},
    session::ExecutionStep,
    Dispatchable,
};
use anyhow::Result;

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
        Ok(self.recv()?)
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

impl<A> InChannels for (A,)
where
    A: LastInChannel,
{
    type Items = (A::Item,);

    fn recv(&self) -> Result<Self::Items> {
        Ok((self.0.recv()?,))
    }
}

impl<A, B> InChannels for (A, B)
where
    A: InChannel,
    B: LastInChannel,
{
    type Items = (A::Item, B::Item);

    fn recv(&self) -> Result<Self::Items> {
        Ok((self.0.recv()?, self.1.recv()?))
    }
}

impl<A, B, C> InChannels for (A, B, C)
where
    A: InChannel,
    B: InChannel,
    C: LastInChannel,
{
    type Items = (A::Item, B::Item, C::Item);

    fn recv(&self) -> Result<Self::Items> {
        Ok((self.0.recv()?, self.1.recv()?, self.2.recv()?))
    }
}
