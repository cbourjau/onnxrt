use std::{
    iter::repeat,
    ops::{AddAssign, DivAssign, MulAssign, SubAssign},
};

use num::Float;
use num_traits::AsPrimitive;

use crate::opset::utils::*;

#[allow(dead_code)]
struct BatchNormalization {
    epsilon: f32,  // (default is 1e-05)
    momentum: f32, // (default is 0.9)
    training_mode: bool,
}

impl BatchNormalization {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let epsilon = attrs.get_with_default("epsilon", 1e-5f32)?;
        let momentum = attrs.get_with_default("momentum", 0.9f32)?;
        let training_mode = attrs.get_with_default("training_mode", 0)? != 0;

        Ok(Self {
            epsilon,
            momentum,
            training_mode,
        })
    }

    fn compute<T>(
        &self,
        mut x: ArrayD<T>,
        scale: ArrayViewD<T>,
        b: ArrayViewD<T>,
        mean: ArrayViewD<T>,
        var: ArrayViewD<T>,
    ) -> Result<ArcArrayD<T>>
    where
        T: Clone + Float + ndarray::ScalarOperand + SubAssign + AddAssign + DivAssign + MulAssign,
        f32: AsPrimitive<T>,
    {
        let new_shape: Vec<_> = [1, x.shape()[1]]
            .into_iter()
            .chain(repeat(1).take(x.ndim() - 2))
            .collect();

        let scale = scale.into_shape(new_shape.as_slice())?;
        let b = b.into_shape(new_shape.as_slice())?;
        let mean = mean.into_shape(new_shape.as_slice())?;
        let var = var.into_shape(new_shape.as_slice())?;

        x -= &mean;
        let denom = (&var + self.epsilon.as_()).mapv(|el| el.sqrt());
        x /= &(denom / scale);
        x += &b;
        Ok(x.into())
    }
}

pub fn setup_bn(ctx: OpContext) -> Result<ExecutionStep> {
    let state = BatchNormalization::from_attrs(&ctx.attrs)?;

    if state.training_mode {
        bail!("BatchNormalization: 'training_mode=true' is not supported")
    }

    let (x, scale, b, mean, var) = ctx.subscribe_to_inputs().narrow()?;

    // The standard is rather rule and allows for three different
    // input types (T, T1, and T2). Let's just do everything in f64 in most cases.
    macro_rules! cast_all_to_f64 {
        ($x:tt, $s:tt, $b:tt, $m:tt, $v:tt) => {
            setup(($x, $s, $b, $m, $v), move |(x, s, b, m, v)| {
                let x = x.mapv(|el| el as f64);
                let s = s.mapv(|el| el as f64);
                let b = b.mapv(|el| el as f64);
                let m = m.mapv(|el| el as f64);
                let v = v.mapv(|el| el as f64);
                state.compute(x, s.view(), b.view(), m.view(), v.view())
            })
        };
    }
    use FloatingPoint::*;
    match (x, scale, b, mean, var) {
        (F32(x), F32(s), F32(b), F32(m), F32(v)) => {
            setup((x, s, b, m, v), move |(x, s, b, m, v)| {
                let x = x.to_owned();
                let s = s.view();
                let b = b.view();
                let m = m.view();
                let v = v.view();
                state.compute(x, s, b, m, v)
            })
        }
        (F32(x), F64(s), F64(b), F32(m), F32(v)) => {
            cast_all_to_f64!(x, s, b, m, v)
        }
        (F32(x), F32(s), F32(b), F64(m), F64(v)) => {
            cast_all_to_f64!(x, s, b, m, v)
        }
        (F64(x), F32(s), F32(b), F32(m), F32(v)) => {
            cast_all_to_f64!(x, s, b, m, v)
        }
        (F64(x), F64(s), F64(b), F32(m), F32(v)) => {
            cast_all_to_f64!(x, s, b, m, v)
        }
        (F64(x), F32(s), F32(b), F64(m), F64(v)) => {
            cast_all_to_f64!(x, s, b, m, v)
        }
        (F64(x), F64(s), F64(b), F64(m), F64(v)) => {
            setup((x, s, b, m, v), move |(x, s, b, m, v)| {
                let x = x.to_owned();
                let s = s.view();
                let b = b.view();
                let m = m.view();
                let v = v.view();
                state.compute(x, s, b, m, v)
            })
        }
        _ => bail!("BatchNormalization: Unexpected input types"),
    }
}
