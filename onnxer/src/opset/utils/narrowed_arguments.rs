// Some helper types for validating inputs.
use std::sync::mpsc::Receiver;

use anyhow::{bail, Result};

use crate::{ArcArrayD, TaggedReceiver};

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum Homogeneous {
    F32(Vec<Receiver<ArcArrayD<f32>>>),
    F64(Vec<Receiver<ArcArrayD<f64>>>),
    I32(Vec<Receiver<ArcArrayD<i32>>>),
    I64(Vec<Receiver<ArcArrayD<i64>>>),
    U32(Vec<Receiver<ArcArrayD<u32>>>),
    U64(Vec<Receiver<ArcArrayD<u64>>>),
    String(Vec<Receiver<ArcArrayD<String>>>),
    Bool(Vec<Receiver<ArcArrayD<bool>>>),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum HomogeneousNumeric {
    F32(Vec<Receiver<ArcArrayD<f32>>>),
    F64(Vec<Receiver<ArcArrayD<f64>>>),
    I32(Vec<Receiver<ArcArrayD<i32>>>),
    I64(Vec<Receiver<ArcArrayD<i64>>>),
    U32(Vec<Receiver<ArcArrayD<u32>>>),
    U64(Vec<Receiver<ArcArrayD<u64>>>),
}

/// Homogeneous variadic inputs
#[derive(Debug)]
pub enum HomogeneousFixed<const N: usize> {
    F32([Receiver<ArcArrayD<f32>>; N]),
    F64([Receiver<ArcArrayD<f64>>; N]),
    I32([Receiver<ArcArrayD<i32>>; N]),
    I64([Receiver<ArcArrayD<i64>>; N]),
    U32([Receiver<ArcArrayD<u32>>; N]),
    U64([Receiver<ArcArrayD<u64>>; N]),
    Bool([Receiver<ArcArrayD<bool>>; N]),
    String([Receiver<ArcArrayD<String>>; N]),
}

/// Homogeneous, numeric variadic inputs
#[derive(Debug)]
pub enum HomogeneousNumericFixed<const N: usize> {
    F32([Receiver<ArcArrayD<f32>>; N]),
    F64([Receiver<ArcArrayD<f64>>; N]),
    I32([Receiver<ArcArrayD<i32>>; N]),
    I64([Receiver<ArcArrayD<i64>>; N]),
    U32([Receiver<ArcArrayD<u32>>; N]),
    U64([Receiver<ArcArrayD<u64>>; N]),
}

pub trait NarrowFromVec<T> {
    fn narrow(self) -> Result<T>
    where
        Self: Sized;
}

trait NarrowFromIter {
    fn parse<I>(objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver> + ExactSizeIterator,
        Self: Sized;
}

trait NarrowFromIterLast {
    fn parse<I>(objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver> + ExactSizeIterator,
        Self: Sized;
}

impl NarrowFromIter for TaggedReceiver {
    fn parse<I>(mut objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver>,
    {
        Ok((
            match objs.next() {
                Some(trec) => trec,
                None => bail!("Insufficient number of inputs"),
            },
            objs,
        ))
    }
}

macro_rules! impl_NarrowFromIter {
    ($ty:ty, $var:tt) => {
        impl NarrowFromIter for Receiver<ArcArrayD<$ty>> {
            fn parse<I>(mut objs: I) -> Result<(Self, I)>
            where
                I: Iterator<Item = TaggedReceiver>,
            {
                Ok((
                    match objs.next() {
                        Some(TaggedReceiver::$var(rec)) => rec,
                        Some(trec) => bail!("Expected 'tensor(bool)' found {:?}", trec),
                        None => bail!("Insufficient number of inputs"),
                    },
                    objs,
                ))
            }
        }
    };
}
impl_NarrowFromIter!(bool, Bool);
impl_NarrowFromIter!(i32, I32);
impl_NarrowFromIter!(i64, I64);
impl_NarrowFromIter!(f32, F32);
impl_NarrowFromIter!(f64, F64);
impl_NarrowFromIter!(u32, U32);
impl_NarrowFromIter!(String, String);

impl<T> NarrowFromIterLast for T
where
    T: NarrowFromIter,
{
    fn parse<I>(objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver> + ExactSizeIterator,
    {
        T::parse(objs)
    }
}

impl<const N: usize> NarrowFromIterLast for HomogeneousFixed<N> {
    fn parse<I>(mut objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver> + ExactSizeIterator,
    {
        let first = objs.next().unwrap();

        macro_rules! monomorphize {
            ($($variant:tt),*) => {
                match first {
                    $(TaggedReceiver::$variant(rec) => {
                        let mut out = vec![rec];
                        for _ in 1..N {
                            match objs.next() {
                                Some(TaggedReceiver::$variant(next)) => {
                                    out.push(next);
                                }
                                Some(_) => panic!(),
                                None => panic!(),
                            }
                        }
                        Self::$variant(out.try_into().unwrap())
                    },)*
                }
            };
        }

        Ok((
            monomorphize!(F32, F64, I32, I64, U32, U64, Bool, String),
            objs,
        ))
    }
}

impl<const N: usize> NarrowFromIterLast for HomogeneousNumericFixed<N> {
    fn parse<I>(objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver> + ExactSizeIterator,
    {
        let (vals, objs) = HomogeneousFixed::<N>::parse(objs)?;
        macro_rules! do_match {
            ($($variant:tt),*) => {
                match vals {
                    $(HomogeneousFixed::$variant(arr) => {
                        Ok((HomogeneousNumericFixed::$variant(arr), objs))
                    },)*
                    HomogeneousFixed::Bool(_) | HomogeneousFixed::String(_) => {
                        bail!("Expected numeric input")
                    }
                }
            };
        }
        do_match!(F32, F64, I32, I64, U32, U64)
    }
}

impl NarrowFromIterLast for Homogeneous {
    fn parse<I>(mut objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver> + ExactSizeIterator,
    {
        let len = objs.len();

        let first = objs.next().unwrap();

        macro_rules! monomorphize {
            ($($variant:tt),*) => {
                match first {
                    $(TaggedReceiver::$variant(rec) => {
                        let mut out = vec![rec];
                        for i in 1..len {
                            match objs.next() {
                                Some(TaggedReceiver::$variant(next)) => {
                                    out.push(next);
                                }
                                Some(rec) => bail!(
                                    "Expected homogeneous input of type 'TaggedReceiver::{}'. Found '{:?}'",
                                    stringify!($variant), rec
                                ),
                                None => bail!(
                                    "Expected {} homogeneous inputs. Found {}",
                                    len,
                                    i - 1
                                ),
                            }
                        }
                        Self::$variant(out.try_into().unwrap())
                    },)*
                }
            };
        }

        Ok((
            monomorphize!(F32, F64, I32, I64, U32, U64, Bool, String),
            objs,
        ))
    }
}

impl NarrowFromIterLast for HomogeneousNumeric {
    fn parse<I>(objs: I) -> Result<(Self, I)>
    where
        I: Iterator<Item = TaggedReceiver> + ExactSizeIterator,
    {
        let (vals, objs) = Homogeneous::parse(objs)?;

        macro_rules! do_match {
            ($($variant:tt),*) => {
                match vals {
                    $(Homogeneous::$variant(arr) => {
                        Ok((HomogeneousNumeric::$variant(arr), objs))
                    },)*
                    Homogeneous::Bool(_) | Homogeneous::String(_) => {
                        bail!("Expected numeric input")
                    }
                }
            };
        }
        do_match!(F32, F64, I32, I64, U32, U64)
    }
}

impl<T1> NarrowFromVec<T1> for Vec<TaggedReceiver>
where
    T1: NarrowFromIterLast,
{
    fn narrow(self) -> Result<T1> {
        let iter = self.into_iter();
        let (out1, iter) = T1::parse(iter)?;

        if iter.count() != 0 {
            bail!("Unexpected number of inputs.")
        }
        Ok(out1)
    }
}

impl<T1> NarrowFromVec<(T1,)> for Vec<TaggedReceiver>
where
    T1: NarrowFromIterLast,
{
    fn narrow(self) -> Result<(T1,)> {
        let iter = self.into_iter();
        let (out1, iter) = T1::parse(iter)?;

        if iter.count() != 0 {
            bail!("Unexpected number of inputs.")
        }
        Ok((out1,))
    }
}

impl<T1, T2> NarrowFromVec<(T1, T2)> for Vec<TaggedReceiver>
where
    T1: NarrowFromIter,
    T2: NarrowFromIterLast,
{
    fn narrow(self) -> Result<(T1, T2)> {
        let iter = self.into_iter();
        let (out1, iter) = T1::parse(iter)?;
        let (out2, iter) = T2::parse(iter)?;

        if iter.count() != 0 {
            bail!("Unexpected number of inputs.")
        }

        Ok((out1, out2))
    }
}
