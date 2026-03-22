use crate::{opset::utils::*, session::OpContext};

fn size<T>(a: &ArcArrayD<T>) -> Result<ArrayD<i64>> {
    let arr = ndarray::Array0::from_elem((), a.len() as i64);
    Ok(arr.into_dyn())
}

pub fn setup_size(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use TaggedTensorReceiver::*;
    match a {
        Bool(a) => setup_unary(a, |a| size(&a)),
        String(a) => setup_unary(a, |a| size(&a)),

        F16(a) => setup_unary(a, |a| size(&a)),
        F32(a) => setup_unary(a, |a| size(&a)),
        F64(a) => setup_unary(a, |a| size(&a)),

        I8(a) => setup_unary(a, |a| size(&a)),
        I16(a) => setup_unary(a, |a| size(&a)),
        I32(a) => setup_unary(a, |a| size(&a)),
        I64(a) => setup_unary(a, |a| size(&a)),

        U8(a) => setup_unary(a, |a| size(&a)),
        U16(a) => setup_unary(a, |a| size(&a)),
        U32(a) => setup_unary(a, |a| size(&a)),
        U64(a) => setup_unary(a, |a| size(&a)),
    }
}

fn shape<T>(a: &ArcArrayD<T>) -> Result<ArrayD<i64>> {
    let shape = a.shape();
    let arr = ndarray::Array::from_iter(shape.iter().map(|x| *x as i64)).into_dyn();
    Ok(arr)
}

pub fn setup_shape(
    ctx: OpContext,
) -> Result<(Box<dyn crate::Executable>, Vec<crate::TaggedReceiver>)> {
    let a = ctx.subscribe_to_inputs().narrow()?;

    use TaggedTensorReceiver::*;
    match a {
        Bool(a) => setup_unary(a, |a| shape(&a)),
        String(a) => setup_unary(a, |a| shape(&a)),

        F16(a) => setup_unary(a, |a| shape(&a)),
        F32(a) => setup_unary(a, |a| shape(&a)),
        F64(a) => setup_unary(a, |a| shape(&a)),

        I8(a) => setup_unary(a, |a| shape(&a)),
        I16(a) => setup_unary(a, |a| shape(&a)),
        I32(a) => setup_unary(a, |a| shape(&a)),
        I64(a) => setup_unary(a, |a| shape(&a)),

        U8(a) => setup_unary(a, |a| shape(&a)),
        U16(a) => setup_unary(a, |a| shape(&a)),
        U32(a) => setup_unary(a, |a| shape(&a)),
        U64(a) => setup_unary(a, |a| shape(&a)),
    }
}
