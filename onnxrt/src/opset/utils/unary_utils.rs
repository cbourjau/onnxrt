use anyhow::Result;
use flume::Receiver;

use crate::channels::{Broadcaster, unbounded};
use crate::{ArcArrayD, Executable, TaggedReceiver, TaggedTensorReceiver};

pub struct UnaryWrapper<T, S> {
    pub a: Receiver<ArcArrayD<T>>,
    pub b: Broadcaster<ArcArrayD<T>>,
    pub state: S,
}

pub trait UnaryKernel<T> {
    fn compute(&self, a: ArcArrayD<T>) -> ArcArrayD<T>;
}

impl<T, S> Executable for UnaryWrapper<T, S>
where
    S: UnaryKernel<T>,
{
    /// Consume inputs and produce outputs
    fn execute(&self) -> Result<()> {
        let a = self.a.recv().unwrap();
        let res = self.state.compute(a);
        self.b.send(res).unwrap();
        Ok(())
    }
}

/// Exhaustively match all variants of `$enum` which must be an enum found in `crate::opset::utils`.
///
/// The mapped-to types must implement `UnaryKernel`
macro_rules! match_unary {
    (($enum:ident, $ctx:expr) => {$([$($variant:ident),*] => $state:expr),* $(,)?}) => {{
        use crate::opset::utils::Narrow;
        let data = $ctx.subscribe_to_inputs().narrow()?;
        match data {
            $(
                $(
                    crate::opset::utils::$enum::$variant([a]) => {
                        let (b, b_rec) = crate::channels::unbounded();
                        let state = $state;
                        Ok((Box::new(crate::opset::utils::UnaryWrapper { a, b, state }), vec![b_rec]))
                    }
                ),*
            ),*
        }}
    };
}

/// Create a setup function for a basic unary operation that does not need specialization for some types.
///
/// The first parameter is the function name. The second one is a function that is called in `compute`.
macro_rules! make_unary_numeric {
    ($name:ident, $fun:expr) => {
        pub fn $name(ctx: crate::session::OpContext) -> anyhow::Result<(Box<dyn crate::Executable + Send>, Vec<crate::TaggedReceiver>)> {
            struct State;

            impl<T> crate::opset::utils::UnaryKernel<T> for State
            where
                T: num_traits::Num + Clone,
            {
                fn compute(&self, a: crate::ArcArrayD<T>) -> crate::ArcArrayD<T> {
                    $fun(a)
                }
            }

            crate::opset::utils::match_unary!(
                (HomogeneousNumericFixed, ctx) => {
                    [I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64] => State,
                }
            )
        }
    }
}

/// Create a stateless unary operation
macro_rules! make_unary_boolean {
    ($name:ident, $fun:expr) => {
        pub fn $name(
            ctx: crate::session::OpContext,
        ) -> anyhow::Result<(
            Box<dyn crate::Executable + Send>,
            Vec<crate::TaggedReceiver>,
        )> {
            use crate::opset::utils::*;

            struct State;

            impl crate::opset::utils::UnaryKernel<bool> for State {
                fn compute(&self, a: crate::ArcArrayD<bool>) -> crate::ArcArrayD<bool> {
                    $fun(a)
                }
            }

            let (a,) = ctx.subscribe_to_inputs().narrow()?;
            match (a,) {
                (TaggedTensorReceiver::Bool(a),) => {
                    let (b, b_rec) = crate::channels::unbounded();
                    let state = State;
                    Ok((
                        Box::new(crate::opset::utils::UnaryWrapper { a, b, state }),
                        vec![b_rec],
                    ))
                }
                _ => anyhow::bail!("expected boolean input"),
            }
        }
    };
}

pub(crate) use make_unary_boolean;
pub(crate) use make_unary_numeric;
pub(crate) use match_unary;
