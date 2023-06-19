use num::Integer;
use num_traits::{AsPrimitive, Float};

use crate::opset::utils::*;

#[derive(Debug)]
struct State {
    fmod: bool,
}

fn fmod<T>(x: T, y: T) -> T
where
    T: Float,
{
    let n = (x / y).trunc();
    let res = x - n * y;

    // Returned sign only depends on sign of x
    if x > T::zero() {
        res.abs()
    } else {
        res.abs().neg()
    }
}

fn python_mod_float<T>(x: T, y: T) -> T
where
    T: Float,
{
    let n = (x / y).floor();
    let res = x - n * y;

    // Returned sign only depends on sign of y; flip if necessary.
    if (y > T::zero()) & (res < T::zero()) {
        T::zero() - res
    } else {
        res
    }
}

fn python_mod_int<T>(x: T, y: T) -> T
where
    T: Integer,
{
    x.mod_floor(&y)
}

impl State {
    fn mod_<T>(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> Result<ArcArrayD<T>>
    where
        T: Float,
    {
        if self.fmod {
            zip_map_with_co_broadcast(a, b, |&a, &b| fmod(a, b))
        } else {
            zip_map_with_co_broadcast(a, b, |&a, &b| python_mod_float(a, b))
        }
    }

    fn integer_mod<T>(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> Result<ArcArrayD<T>>
    where
        T: Integer + AsPrimitive<f64>,
        f64: AsPrimitive<T>,
    {
        if !self.fmod {
            zip_map_with_co_broadcast(a, b, |&a, &b| python_mod_int(a, b))
        } else {
            let a = a.mapv(|el| el.as_()).into();
            let b = b.mapv(|el| el.as_()).into();
            let res = self.mod_(a, b)?;
            let out: ArcArrayD<_> = res.mapv(|el| el.as_()).into();
            Ok(out)
        }
    }
}

pub fn setup_mod(ctx: OpContext) -> Result<ExecutionStep> {
    let inputs: HomogeneousNumericFixed<2> = ctx.input_channels().try_into().unwrap();

    let fmod = ctx.attrs.get_with_default("fmod", 0)? == 1;
    let state = State { fmod };

    use HomogeneousNumericFixed::*;
    match inputs {
        F32([a, b]) => setup((a, b), move |(a, b)| state.mod_(a, b)),
        F64([a, b]) => setup((a, b), move |(a, b)| state.mod_(a, b)),
        I32([a, b]) => setup((a, b), move |(a, b)| state.integer_mod(a, b)),
        I64([a, b]) => setup((a, b), move |(a, b)| state.integer_mod(a, b)),
        U32([a, b]) => setup((a, b), move |(a, b)| state.integer_mod(a, b)),
        U64([a, b]) => setup((a, b), move |(a, b)| state.integer_mod(a, b)),
    }
}
