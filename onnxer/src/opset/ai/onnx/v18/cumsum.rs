use std::ops::{AddAssign, Sub};

use ndarray::Axis;
use num_traits::{AsPrimitive, PrimInt, Zero};

use crate::opset::utils::*;

struct State {
    exclusive: bool,
    reverse: bool,
}

impl State {
    fn cumsum<D, Idx: Clone>(
        &self,
        (x, axis): (ArcArrayD<D>, ArcArrayD<Idx>),
    ) -> Result<ArcArrayD<D>>
    where
        Idx: PrimInt + AsPrimitive<i64>,
        D: Zero + Clone + Copy + AddAssign + Sub<Output = D>,
    {
        let rank = x.shape().len();

        let axis: i64 = axis.into_shape(())?.to_owned().into_scalar().as_();
        let axis = convert_negative_index(axis, rank);
        let axis = Axis(axis);

        let mut out = x.clone();

        // Flip exclusive when in reverse
        let exclusive = if self.reverse {
            !self.exclusive
        } else {
            self.exclusive
        };

        for mut lane in out.lanes_mut(axis) {
            let mut state = if self.reverse { lane.sum() } else { D::zero() };

            for el in lane.iter_mut() {
                if exclusive {
                    let acc = if self.reverse {
                        state - *el
                    } else {
                        state + *el
                    };
                    *el = state;
                    state = acc;
                } else {
                    if self.reverse {
                        state = state - *el;
                    } else {
                        state += *el;
                    }
                    *el = state;
                }
            }
        }

        Ok(out)
    }
}

pub fn setup_cumsum(ctx: OpContext) -> Result<ExecutionStep> {
    let exclusive = ctx.attrs.get_with_default("exclusive", 0)? == 1;
    let reverse = ctx.attrs.get_with_default("reverse", 0)? == 1;

    let state = State { exclusive, reverse };

    let (x, axis) = ctx.subscribe_to_inputs().narrow()?;

    macro_rules! match_all {
        ($($x_type:tt),* ) => {
            match [x, axis] {
                $([$x_type(x), I32(axis)] => setup((x, axis), move |(x, axis)| state.cumsum((x, axis)))?,)*
                $([$x_type(x), I64(axis)] => setup((x, axis), move |(x, axis)| state.cumsum((x, axis)))?,)*
                [x, axis] => bail!(
                    "Unimplemented types for 'CumSum': x: {:?}, axis: {:?}",
                    x,
                    axis
                ),
            }
        }
    }
    use TaggedReceiver::*;
    Ok(match_all!(F32, F64, I32, I64, U32, U64))
}
