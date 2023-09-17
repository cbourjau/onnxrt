use crate::opset::utils::*;

pub fn setup_pow(ctx: OpContext) -> Result<ExecutionStep> {
    let (x, y) = ctx.subscribe_to_inputs().narrow()?;

    use Numeric::*;

    // Hacky: just always cat to f64...
    macro_rules! match_arm {
        ($x:tt, $ty:ty) => {
            match y {
                F32(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powf(y as f64) as $ty)
                }),
                F64(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powf(y) as $ty)
                }),

                I8(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                }),
                I16(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                }),
                I32(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y) as $ty)
                }),
                I64(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                }),

                U8(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                }),
                U16(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                }),
                U32(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                }),
                U64(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                }),
            }
        };
    }

    match x {
        F32(x) => match_arm!(x, f32),
        F64(x) => match_arm!(x, f64),
        I32(x) => match_arm!(x, i32),
        I64(x) => match_arm!(x, i64),
        U32(x) => match_arm!(x, u32),
        U64(x) => match_arm!(x, u64),
        _ => bail!("Pow: Unsupported type for input 'x': {:?}", x),
    }
}
