use std::cmp::Ordering;

use num::Zero;

use crate::DataType;
use crate::opset::utils::*;

fn eye_like<T>(shape: [usize; 2], elem: T, k: isize) -> Result<Array2<T>>
where
    T: Clone + Zero,
{
    let mut a = Array2::zeros(shape);
    if k >= shape[0] as isize || k <= -(shape[1] as isize) {
        bail!("Parameter 'k={}' is out-of-bounds for shape {:?}", k, shape)
    }
    match k.cmp(&0) {
        Ordering::Equal => a.diag_mut().fill(elem),
        Ordering::Less => a.slice_mut(s![-k.., ..]).diag_mut().fill(elem),
        Ordering::Greater => a.slice_mut(s![.., k..]).diag_mut().fill(elem),
    }
    Ok(a)
}

#[derive(Debug)]
struct EyeLike {
    dtype: Option<DataType>,
    k: i64,
}

impl EyeLike {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let dtype = attrs
            .get_optional::<i64>("dtype")?
            .map(|dtype| DataType::from_i32(dtype as i32))
            .transpose()?;

        let k = attrs.get_with_default("k", 0i64)?;

        Ok(Self { dtype, k })
    }
}

pub fn setup_eyelike(ctx: OpContext) -> Result<ExecutionStep> {
    let state = EyeLike::from_attrs(&ctx.attrs)?;
    let input = ctx.subscribe_to_inputs().narrow()?;

    use Numeric::*;
    let dtype = match state.dtype {
        Some(dtype) => dtype,
        None => match input {
            I8(_) => DataType::I8,
            I16(_) => DataType::I16,
            I32(_) => DataType::I32,
            I64(_) => DataType::I64,
            U8(_) => DataType::U8,
            U16(_) => DataType::U16,
            U32(_) => DataType::U32,
            U64(_) => DataType::U64,
            F32(_) => DataType::F32,
            F64(_) => DataType::F64,
        },
    };

    let k = state.k as isize;
    setup_all!(Numeric, input, rec, {
        match dtype {
            DataType::U8 => setup(rec, move |input| {
                Ok::<ArrayD<u8>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::U16 => setup(rec, move |input| {
                Ok::<ArrayD<u16>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::U32 => setup(rec, move |input| {
                Ok::<ArrayD<u32>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::U64 => setup(rec, move |input| {
                Ok::<ArrayD<u64>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::I8 => setup(rec, move |input| {
                Ok::<ArrayD<i8>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::I16 => setup(rec, move |input| {
                Ok::<ArrayD<i16>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::I32 => setup(rec, move |input| {
                Ok::<ArrayD<i32>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::I64 => setup(rec, move |input| {
                Ok::<ArrayD<i64>, _>(eye_like(input.shape().try_into()?, 1, k)?.into_dyn())
            }),
            DataType::F32 => setup(rec, move |input| {
                Ok::<ArrayD<f32>, _>(eye_like(input.shape().try_into()?, 1.0, k)?.into_dyn())
            }),
            DataType::F64 => setup(rec, move |input| {
                Ok::<ArrayD<f64>, _>(eye_like(input.shape().try_into()?, 1.0, k)?.into_dyn())
            }),
            _ => bail!("EyeLike: Not implemented for output type {:?}", dtype),
        }
    })
}
