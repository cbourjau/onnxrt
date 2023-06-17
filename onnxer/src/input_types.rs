// Some helper types for validating inputs.
use std::sync::mpsc::Receiver;

use anyhow::{anyhow, bail, Error, Result};

use crate::{ArcArrayD, TaggedReceiver};

/// Homogeneous, numeric variadic inputs
pub enum HomogeneousNumericFixed<const N: usize> {
    F32([Receiver<ArcArrayD<f32>>; N]),
    F64([Receiver<ArcArrayD<f64>>; N]),
    I32([Receiver<ArcArrayD<i32>>; N]),
    I64([Receiver<ArcArrayD<i64>>; N]),
}

impl<const N: usize> TryFrom<Vec<TaggedReceiver>> for HomogeneousNumericFixed<N> {
    type Error = Error;

    fn try_from(vec: Vec<TaggedReceiver>) -> Result<Self, Self::Error> {
        let len = vec.len();
        macro_rules! make_err {
            () => {
                |_| anyhow!("Unexpected {} inputs found {}", N, len)
            };
        }
        Ok(match HomogeneousNumeric::new_from_iter(vec.into_iter())? {
            HomogeneousNumeric::F32(vec) => Self::F32(vec.try_into().map_err(make_err!())?),
            HomogeneousNumeric::F64(vec) => Self::F64(vec.try_into().map_err(make_err!())?),
            HomogeneousNumeric::I32(vec) => Self::I32(vec.try_into().map_err(make_err!())?),
            HomogeneousNumeric::I64(vec) => Self::I64(vec.try_into().map_err(make_err!())?),
        })
    }
}

/// Homogeneous, numeric variadic inputs
pub enum Homogeneous {
    F32(Vec<Receiver<ArcArrayD<f32>>>),
    F64(Vec<Receiver<ArcArrayD<f64>>>),
    I32(Vec<Receiver<ArcArrayD<i32>>>),
    I64(Vec<Receiver<ArcArrayD<i64>>>),
    String(Vec<Receiver<ArcArrayD<String>>>),
    Bool(Vec<Receiver<ArcArrayD<bool>>>),
}

impl Homogeneous {
    pub fn new_from_iter<I>(inputs: I) -> Result<Self>
    where
        I: IntoIterator<Item = TaggedReceiver> + ExactSizeIterator,
    {
        if inputs.len() == 0 {
            bail!("Construction `HomogeneousChannels` requires at least one input.")
        }

        let mut inputs = inputs.into_iter().peekable();
        let first: &TaggedReceiver = inputs.peek().unwrap();

        macro_rules! match_arm {
            ($variant:tt) => {
                Homogeneous::$variant({
                    inputs
                        .into_iter()
                        .map(|el| {
                            if let TaggedReceiver::$variant(el) = el {
                                Ok(el)
                            } else {
                                bail!("Inputs are not homogeneous.")
                            }
                        })
                        .collect::<Result<_>>()?
                })
            };
        }

        Ok(match first {
            TaggedReceiver::F32(_) => match_arm!(F32),
            TaggedReceiver::F64(_) => match_arm!(F64),
            TaggedReceiver::I32(_) => match_arm!(I32),
            TaggedReceiver::I64(_) => match_arm!(I64),
            TaggedReceiver::Bool(_) => match_arm!(Bool),
            TaggedReceiver::String(_) => match_arm!(String),
        })
    }
}

/// Homogeneous, numeric variadic inputs
pub enum HomogeneousNumeric {
    F32(Vec<Receiver<ArcArrayD<f32>>>),
    F64(Vec<Receiver<ArcArrayD<f64>>>),
    I32(Vec<Receiver<ArcArrayD<i32>>>),
    I64(Vec<Receiver<ArcArrayD<i64>>>),
}

impl HomogeneousNumeric {
    pub fn new_from_iter<I>(inputs: I) -> Result<Self>
    where
        I: IntoIterator<Item = TaggedReceiver> + ExactSizeIterator,
    {
        let inputs = Homogeneous::new_from_iter(inputs)?;

        Ok(match inputs {
            Homogeneous::F32(arrs) => HomogeneousNumeric::F32(arrs),
            Homogeneous::F64(arrs) => HomogeneousNumeric::F64(arrs),
            Homogeneous::I32(arrs) => HomogeneousNumeric::I32(arrs),
            Homogeneous::I64(arrs) => HomogeneousNumeric::I64(arrs),
            _ => bail!("Inputs are not numeric."),
        })
    }
}
