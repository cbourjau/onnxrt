use crate::opset::utils::*;

pub fn setup_identity(
    ctx: crate::session::OpContext,
) -> anyhow::Result<(
    Box<dyn crate::Executable + Send>,
    Vec<crate::TaggedReceiver>,
)> {
    struct State;

    impl<T> crate::opset::utils::UnaryKernel<T> for State
    where
        T: Clone,
    {
        fn compute(&self, a: crate::ArcArrayD<T>) -> crate::ArcArrayD<T> {
            a
        }
    }

    crate::opset::utils::match_unary!(
        (HomogeneousFixed, ctx) => {
            [I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64, String, Bool] => State,
        }
    )
}
