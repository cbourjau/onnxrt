// Implementation note:
// The rng will be rebuild for each invocation. If a seed is given,
// the result will be the same each time this operator is called.
use num_traits::AsPrimitive;
use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::{edges::DataType, opset::utils::*};

enum FloatDataType {
    F32,
    F64,
}

struct RandomUniformLike {
    dtype: Option<FloatDataType>,
    high: f32,
    low: f32,
    seed: Option<f32>,
}

impl FloatDataType {
    fn from_i64(type_num: i64) -> Result<Self> {
        let dtype = DataType::from_i32(type_num as i32)?;
        Ok(match dtype {
            DataType::F32 => Self::F32,
            DataType::F64 => Self::F64,
            _ => bail!("Expected floating point data type."),
        })
    }
}

fn rng_from_seed(seed: Option<f32>) -> Result<StdRng> {
    let rng = match seed {
        Some(f) => {
            let seed = u64::from_be_bytes((f as f64).to_be_bytes());
            StdRng::seed_from_u64(seed)
        }
        None => {
            let rng = rand::thread_rng();
            StdRng::from_rng(rng)?
        }
    };
    Ok(rng)
}

impl RandomUniformLike {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let dtype = {
            let dtype: Option<i64> = attrs.get_optional("dtype")?;
            dtype.map(FloatDataType::from_i64).transpose()?
        };

        let high = attrs.get_with_default("high", 1.0f32)?;
        let low = attrs.get_with_default("low", 0.0f32)?;

        let seed: Option<f32> = attrs.get_optional("seed")?;

        //
        Ok(Self {
            dtype,
            high,
            low,
            seed,
        })
    }

    fn sample<T>(&self, n: usize) -> Result<Vec<T>>
    where
        T: Copy + 'static,
        f32: AsPrimitive<T>,
    {
        let dist = Uniform::from(self.low..self.high);
        let rng = rng_from_seed(self.seed)?;

        Ok(dist.sample_iter(rng).take(n).map(|el| el.as_()).collect())
    }
}

pub fn setup_random_uniform_like(ctx: OpContext) -> Result<ExecutionStep> {
    let state = RandomUniformLike::from_attrs(&ctx.attrs)?;

    let (a,) = ctx.subscribe_to_inputs().narrow()?;
    use TaggedReceiver::*;

    macro_rules! non_float_arm {
        ($rec:tt) => {
            match state.dtype {
            Some(FloatDataType::F32) => setup($rec, move |a| {
                let one_d = state.sample::<f32>(a.len())?;
                Ok(Array1::from_vec(one_d).into_shape(a.shape())?)
            }),
            Some(FloatDataType::F64) => setup($rec, move |a| {
                let one_d = state.sample::<f64>(a.len())?;
                Ok(Array1::from_vec(one_d).into_shape(a.shape())?)
            }),
            None => bail!(
                "RandomUniformLike: Attribute 'dtype' must be specified for given input dtype"
            ),
        }
        }}

    match a {
        F32(rec) => match state.dtype {
            None | Some(FloatDataType::F32) => setup(rec, move |a| {
                let one_d = state.sample::<f32>(a.len())?;
                Ok(Array1::from_vec(one_d).into_shape(a.shape())?)
            }),
            Some(FloatDataType::F64) => setup(rec, move |a| {
                let one_d = state.sample::<f64>(a.len())?;
                Ok(Array1::from_vec(one_d).into_shape(a.shape())?)
            }),
        },
        F64(rec) => match state.dtype {
            Some(FloatDataType::F32) => setup(rec, move |a| {
                let one_d = state.sample::<f32>(a.len())?;
                Ok(Array1::from_vec(one_d).into_shape(a.shape())?)
            }),
            None | Some(FloatDataType::F64) => setup(rec, move |a| {
                let one_d = state.sample::<f64>(a.len())?;
                Ok(Array1::from_vec(one_d).into_shape(a.shape())?)
            }),
        },
        Bool(rec) => non_float_arm!(rec),
        U8(rec) => non_float_arm!(rec),
        U16(rec) => non_float_arm!(rec),
        U32(rec) => non_float_arm!(rec),
        U64(rec) => non_float_arm!(rec),

        I8(rec) => non_float_arm!(rec),
        I16(rec) => non_float_arm!(rec),
        I32(rec) => non_float_arm!(rec),
        I64(rec) => non_float_arm!(rec),
        String(rec) => non_float_arm!(rec),
    }
}
