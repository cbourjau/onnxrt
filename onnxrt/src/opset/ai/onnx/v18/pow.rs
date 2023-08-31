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
                        .into_arc()
                }),
                F64(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powf(y) as $ty).into_arc()
                }),

                I8(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                        .into_arc()
                }),
                I16(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                        .into_arc()
                }),
                I32(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y) as $ty).into_arc()
                }),
                I64(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                        .into_arc()
                }),

                U8(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                        .into_arc()
                }),
                U16(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                        .into_arc()
                }),
                U32(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                        .into_arc()
                }),
                U64(y) => setup(($x, y), |(x, y)| {
                    let (x, y) = (x.view(), y.view());
                    zip_map_with_co_broadcast(x, y, |&x, &y| (x as f64).powi(y as i32) as $ty)
                        .into_arc()
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
