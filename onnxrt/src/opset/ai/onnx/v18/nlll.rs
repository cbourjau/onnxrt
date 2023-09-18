//! Implementation for NegativeLogLikelihoodLoss (nlll)
use std::ops::AddAssign;

use num::{Float, FromPrimitive};
use num_traits::{AsPrimitive, Zero};

use crate::opset::utils::*;

#[derive(Debug, Clone, Copy)]
pub(crate) enum Reduction {
    Sum,
    Mean,
}

pub(crate) fn nlll<T, Idx>(
    input: ArrayViewD<T>,
    target: ArrayViewD<Idx>,
    weights: Option<ArrayView1<T>>,
    ignore_index: Option<i64>,
    reduction: Option<Reduction>,
) -> Result<ArrayD<T>>
where
    T: Float + FromPrimitive + Zero + AddAssign + 'static,
    Idx: Index + AsPrimitive<i64> + TryInto<usize>,
    usize: AsPrimitive<T>,
{
    let mut out = ArrayD::zeros(target.shape());
    let n_channels = input.shape()[1];
    let mut total_weight = T::zero();

    let mut n_not_ignored: usize = 0;

    for (mut out, (input, target)) in out
        .axis_iter_mut(Axis(0))
        .zip(input.axis_iter(Axis(0)).zip(target.axis_iter(Axis(0))))
    {
        for ((out, input_per_ch), target) in out
            .iter_mut()
            .zip(input.lanes(Axis(0)).into_iter())
            .zip(target.iter())
        {
            if Some(target.as_()) == ignore_index {
                continue;
            }

            // The target class must be within [0, n_channels) unless
            // it is equal to ignore index.
            let c: usize = (*target)
                .try_into()
                .map_err(|_| anyhow!(
                    "NegativeLogLikelihoodLoss: 'target' values must equal 'ingore_index' or be within '[0, n_channels)'"
                ))?;

            n_not_ignored += 1;
            if let Some(weights) = weights {
                *out = -input_per_ch[c] * weights[c];
                total_weight += weights[c];
            } else {
                *out = -input_per_ch[target.for_len(n_channels)];
            }
        }
    }

    if let Some(reduction) = reduction {
        let out = match reduction {
            Reduction::Sum => out.sum(),
            Reduction::Mean => {
                if weights.is_some() {
                    out.sum() / total_weight
                } else {
                    out.sum() / n_not_ignored.as_()
                }
            }
        };
        Ok(array![out].into_shape(()).unwrap().into_dyn())
    } else {
        Ok(out)
    }
}

struct State {
    ignore_index: Option<i64>,
    reduction: Option<Reduction>,
}

impl State {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        // Fixme: Standard says it is optional, but does not say what the default is...
        let ignore_index = attrs.get_optional("ignore_index")?;
        let reduction = attrs.get_with_default("reduction", "mean".to_string())?;
        let reduction = match reduction.as_str() {
            "none" => None,
            "sum" => Some(Reduction::Sum),
            "mean" => Some(Reduction::Mean),
            s => bail!("SoftMaxCrossEntropyLoss: Unexpected reduction '{}'", s),
        };

        Ok(Self {
            ignore_index,
            reduction,
        })
    }
}

pub fn setup_nlll(ctx: OpContext) -> Result<ExecutionStep> {
    let state = State::from_attrs(&ctx.attrs)?;

    let (input, target, weights): (_, OnnxIndex, _) =
        ctx.input_channels_with_optional().narrow()?;

    let target = target.into_i64();

    use FloatingPoint::*;
    match (input, weights) {
        (F32(input), Some(F32(weights))) => {
            setup((input, target, weights), move |(input, target, weights)| {
                nlll(
                    input.view(),
                    target.view(),
                    Some(weights.view().into_dimensionality()?),
                    state.ignore_index,
                    state.reduction,
                )
            })
        }
        (F32(input), None) => setup((input, target), move |(input, target)| {
            nlll(
                input.view(),
                target.view(),
                None,
                state.ignore_index,
                state.reduction,
            )
        }),
        (F64(input), Some(F64(weights))) => {
            setup((input, target, weights), move |(input, target, weights)| {
                nlll(
                    input.view(),
                    target.view(),
                    Some(weights.view().into_dimensionality()?),
                    state.ignore_index,
                    state.reduction,
                )
            })
        }
        (F64(input), None) => setup((input, target), move |(input, target)| {
            nlll(
                input.view(),
                target.view(),
                None,
                state.ignore_index,
                state.reduction,
            )
        }),
        _ => bail!("NegativeLogLikelihoodLoss: Inconsistent types for 'input' and 'weights'"),
    }
}

#[cfg(test)]
mod tests {
    use std::assert_eq;

    use super::*;

    #[test]
    fn example1() {
        // negative log likelihood loss, "none" reduction
        let input = array![
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
        ]
        .into_dyn();
        let target = array![[2, 1], [0, 2]].into_dyn();
        let loss = nlll(input.view(), target.view(), None, None, None).unwrap();

        let expected = array![[-3., -2.], [-0., -2.]].into_dyn();

        assert_eq!(expected, loss);
    }

    #[test]
    fn example2() {
        // weighted negative log likelihood loss, sum reduction
        let input = array![
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
        ]
        .into_dyn();
        let target = array![[2, 1], [0, 2]].into_dyn();
        let weights = array![0.2, 0.3, 0.1];

        let loss = nlll(
            input.view(),
            target.view(),
            Some(weights.view()),
            None,
            Some(Reduction::Sum),
        )
        .unwrap();

        let expected = arr0(-1.1).into_dyn();

        assert_eq!(expected, loss);
    }

    #[test]
    fn example3() {
        // weighted negative log likelihood loss, mean reduction
        let input = array![
            [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]],
            [[0.0, 1.0], [2.0, 2.0], [1.0, 2.0]]
        ]
        .into_dyn();
        let target = array![[2, 1], [0, 2]].into_dyn();
        let weights = array![0.2, 0.3, 0.1];

        let loss = nlll(
            input.view(),
            target.view(),
            Some(weights.view()),
            None,
            Some(Reduction::Mean),
        )
        .unwrap();

        let expected = arr0(-1.5714285714285714).into_dyn();

        assert_eq!(expected, loss);
    }
}
