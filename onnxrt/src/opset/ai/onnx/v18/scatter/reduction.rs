use std::str::FromStr;

use anyhow::bail;
use ordered_float::OrderedFloat;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reduction {
    None,
    Add,
    Mul,
    Max,
    Min,
}

impl FromStr for Reduction {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "none" => Self::None,
            "add" => Self::Add,
            "mul" => Self::Mul,
            "min" => Self::Min,
            "max" => Self::Max,
            s => bail!("ScatterElements: Invalid reduction '{}'", s),
        })
    }
}

impl Reduction {
    #[allow(clippy::type_complexity)]
    pub fn as_function<T>(&self) -> Box<dyn Fn(&mut T, T)>
    where
        T: Reducible + 'static,
    {
        match self {
            Reduction::None => Box::new(T::assign),
            Reduction::Add => Box::new(T::add_assign),
            Reduction::Mul => Box::new(T::mul_assign),
            Reduction::Min => Box::new(T::min),
            Reduction::Max => Box::new(T::max),
        }
    }
}

pub trait Reducible {
    fn add_assign(&mut self, rhs: Self);
    fn mul_assign(&mut self, rhs: Self);
    fn min(&mut self, rhs: Self);
    fn max(&mut self, rhs: Self);
    fn assign(&mut self, rhs: Self)
    where
        Self: Sized,
    {
        *self = rhs;
    }
}

macro_rules! impl_reducible_ints {
    ($ty:ty) => {
        impl Reducible for $ty {
            fn add_assign(&mut self, rhs: Self) {
                *self += rhs;
            }

            fn mul_assign(&mut self, rhs: Self) {
                *self *= rhs;
            }

            fn min(&mut self, rhs: Self) {
                *self = Ord::min(*self, rhs);
            }

            fn max(&mut self, rhs: Self) {
                *self = Ord::max(*self, rhs);
            }
        }
    };
}

impl_reducible_ints!(i8);
impl_reducible_ints!(i16);
impl_reducible_ints!(i32);
impl_reducible_ints!(i64);
impl_reducible_ints!(u8);
impl_reducible_ints!(u16);
impl_reducible_ints!(u32);
impl_reducible_ints!(u64);

macro_rules! impl_reducible_floats {
    ($ty:ty) => {
        impl Reducible for $ty {
            fn add_assign(&mut self, rhs: Self) {
                *self += rhs;
            }

            fn mul_assign(&mut self, rhs: Self) {
                *self *= rhs;
            }

            fn min(&mut self, rhs: Self) {
                *self = Ord::min(OrderedFloat(*self), OrderedFloat(rhs)).0;
            }

            fn max(&mut self, rhs: Self) {
                *self = Ord::max(OrderedFloat(*self), OrderedFloat(rhs)).0;
            }
        }
    };
}

impl_reducible_floats!(f32);
impl_reducible_floats!(f64);
