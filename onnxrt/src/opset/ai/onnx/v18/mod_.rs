// use crate::opset::utils::*;
use anyhow::Result;
use flume::Receiver;
use num::{Float, Integer};

use crate::channels::{Broadcaster, TaggedReceiver, unbounded};
use crate::opset::utils::Narrow;
use crate::session::OpContext;
use crate::{ArcArrayD, Executable};

struct BinaryWrapper<T, S> {
    a: Receiver<ArcArrayD<T>>,
    b: Receiver<ArcArrayD<T>>,
    c: Broadcaster<ArcArrayD<T>>,
    state: S,
}

trait BinaryKernel<T> {
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

// Assume that we want to specialize for some data types
struct AddInt;
struct AddFloat;

impl<T> BinaryKernel<T> for AddInt
where
    T: Integer + Clone,
{
    fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T> {
        a + &b
    }
}
impl<T> BinaryKernel<T> for AddFloat
where
    T: Float,
{
    fn compute(&self, a: ArcArrayD<T>, b: ArcArrayD<T>) -> ArcArrayD<T> {
        a + &b
    }
}

/// Exhaustively match all variants of `$enum` which must be an enum found in `crate::opset::utils`.
///
/// ## Example
///
/// ```
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
        let data = $ctx.subscribe_to_inputs().narrow()?;
        match data {
            $(
                $(
                    crate::opset::utils::$enum::$variant([a, b]) => {
                        let (c, c_rec) = unbounded();
                        let state = $state;
                        (Box::new(BinaryWrapper { a, b, c, state }), vec![c_rec])
                    }
                ),*
            ),*
        }}
    };
}

pub fn setup_add(ctx: OpContext) -> Result<(Box<dyn Executable + Send>, Vec<TaggedReceiver>)> {
    Ok(match_binary!(
        (HomogeneousNumericFixed, ctx) => {
            [I8, I16, I32, I64, U8, U16, U32, U64] => AddInt,
            [F32, F64] => AddFloat
        }
    ))
}
