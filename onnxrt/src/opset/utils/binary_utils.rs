// use crate::opset::utils::*;
use anyhow::Result;
use flume::Receiver;

use crate::channels::Broadcaster;
use crate::{ArcArrayD, Executable};

pub struct BinaryWrapper<T, S> {
    pub a: Receiver<ArcArrayD<T>>,
    pub b: Receiver<ArcArrayD<T>>,
    pub c: Broadcaster<ArcArrayD<T>>,
    pub state: S,
}

pub trait BinaryKernel<T> {
    fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T>;
}

impl<T, S> Executable for BinaryWrapper<T, S>
where
    S: BinaryKernel<T>,
{
    /// Consume inputs and produce outputs
    fn execute(&self) -> Result<()> {
        let a = self.a.recv().unwrap();
        let b = self.b.recv().unwrap();
        let res = self.state.compute(a, b);
        self.c.send(res).unwrap();
        Ok(())
    }
}

/// Exhaustively match all variants of `$enum` which must be an enum found in `crate::opset::utils`.
///
/// The mapped-to types must implement `BinaryKernel`
/// ## Example
///
/// ```
/// struct AddInt;
/// struct AddFloat;
///
/// impl<T> BinaryKernel<T> for AddInt
/// where
///     T: Integer + Clone,
/// {
///     fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T> {
///         a + &b
///     }
/// }
/// impl<T> BinaryKernel<T> for AddFloat
/// where
///     T: Float,
/// {
///     fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T> {
///         a + &b
///     }
/// }
/// pub fn setup_add(ctx: OpContext) -> Result<(Box<dyn Executable + Send>, Vec<TaggedReceiver>)> {
///     Ok(match_binary!(
///         (HomogeneousNumericFixed, ctx) => {
///             [I8, I16, I32, I64, U8, U16, U32, U64] => AddInt,
///             [F32, F64] => AddFloat
///         }
///     ))
/// }
/// ```
macro_rules! match_binary {
    (($enum:ident, $ctx:expr) => {$([$($variant:ident),*] => $state:expr),* $(,)?}) => {{
        use crate::opset::utils::Narrow;
        let data = $ctx.subscribe_to_inputs().narrow()?;
        match data {
            $(
                $(
                    crate::opset::utils::$enum::$variant([a, b]) => {
                        let (c, c_rec) = crate::channels::unbounded();
                        let state = $state;
                        Ok((Box::new(crate::opset::utils::BinaryWrapper { a, b, c, state }), vec![c_rec]))
                    }
                ),*
            ),*
        }}
    };
}

/// Create a setup function for a basic binary operation that does not need specialization for some types.
///
/// The first parameter is the function name. The second one is a function that is called in `compute`.
macro_rules! make_binary_numeric {
    ($name:ident, $fun:expr) => {
        pub fn $name(ctx: crate::session::OpContext) -> anyhow::Result<(Box<dyn crate::Executable + Send>, Vec<crate::TaggedReceiver>)> {
            struct State;

            impl<T> crate::opset::utils::BinaryKernel<T> for State
            where
                T: num_traits::Num + Clone,
            {
                fn compute(&self, a: crate::ArcArrayD<T>, b: crate::ArcArrayD<T>) -> crate::ArcArrayD<T> {
                    $fun(a, b)
                }
            }

            crate::opset::utils::match_binary!(
                (HomogeneousNumericFixed, ctx) => {
                    [I8, I16, I32, I64, U8, U16, U32, U64, F16, F32, F64] => State,
                }
            )
        }
    }
}

/// Create a stateless binary operation
macro_rules! make_binary_boolean {
    ($name:ident, $fun:expr) => {
        pub fn $name(
            ctx: crate::session::OpContext,
        ) -> anyhow::Result<(
            Box<dyn crate::Executable + Send>,
            Vec<crate::TaggedReceiver>,
        )> {
            use crate::opset::utils::*;

            struct State;

            impl crate::opset::utils::BinaryKernel<bool> for State {
                fn compute(
                    &self,
                    a: crate::ArcArrayD<bool>,
                    b: crate::ArcArrayD<bool>,
                ) -> crate::ArcArrayD<bool> {
                    $fun(a, b)
                }
            }

            let (a, b) = ctx.subscribe_to_inputs().narrow()?;
            match (a, b) {
                (TaggedTensorReceiver::Bool(a), TaggedTensorReceiver::Bool(b)) => {
                    let (c, c_rec) = crate::channels::unbounded();
                    let state = State;
                    Ok((
                        Box::new(crate::opset::utils::BinaryWrapper { a, b, c, state }),
                        vec![c_rec],
                    ))
                }
                _ => anyhow::bail!("expected two boolean inputs"),
            }
        }
    };
}

pub(crate) use make_binary_boolean;
pub(crate) use make_binary_numeric;
pub(crate) use match_binary;
