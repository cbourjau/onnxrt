// Some helper types for validating inputs.
use std::sync::mpsc::Receiver;

use anyhow::{anyhow, bail, Error, Result};

use crate::{ArcArrayD, EnumReceiver};

/// Homogeneous, numeric variadic inputs
pub enum HomogeneousNumericFixed<const N: usize> {
    F32([Receiver<ArcArrayD<f32>>; N]),
    F64([Receiver<ArcArrayD<f64>>; N]),
    I32([Receiver<ArcArrayD<i32>>; N]),
    I64([Receiver<ArcArrayD<i64>>; N]),
}

impl<const N: usize> TryFrom<Vec<EnumReceiver>> for HomogeneousNumericFixed<N> {
    type Error = Error;

    fn try_from(vec: Vec<EnumReceiver>) -> Result<Self, Self::Error> {
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
pub enum HomogeneousNumeric {
    F32(Vec<Receiver<ArcArrayD<f32>>>),
    F64(Vec<Receiver<ArcArrayD<f64>>>),
    I32(Vec<Receiver<ArcArrayD<i32>>>),
    I64(Vec<Receiver<ArcArrayD<i64>>>),
}

impl HomogeneousNumeric {
    pub fn new_from_iter<I>(inputs: I) -> Result<Self>
    where
        I: IntoIterator<Item = EnumReceiver> + ExactSizeIterator,
    {
        if inputs.len() == 0 {
            bail!("Construction `HomogeneousChannels` requires at least one input.")
        }

        let mut inputs = inputs.into_iter().peekable();
        let first: &EnumReceiver = inputs.peek().unwrap();

        macro_rules! match_arm {
            ($variant:tt) => {
                HomogeneousNumeric::$variant({
                    inputs
                        .into_iter()
                        .map(|el| {
                            if let EnumReceiver::$variant(el) = el {
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
            EnumReceiver::F32(_) => match_arm!(F32),
            EnumReceiver::F64(_) => match_arm!(F64),
            EnumReceiver::I32(_) => match_arm!(I32),
            EnumReceiver::I64(_) => match_arm!(I64),
            _ => bail!("Inputs are not numeric."),
        })
    }
}
