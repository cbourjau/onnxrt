use super::nlll::{Reduction, nlll};
use super::softmax::softmax;
use crate::opset::utils::*;

struct State {
    ignore_index: Option<i64>,
    reduction: Option<Reduction>,
}

impl State {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        // Fixme: Standard says it is optional, but does not say what the default is...
        let ignore_index = attrs.get_optional("ignore_index")?;
        let reduction = attrs.get_with_default("reduction", "none".to_string())?;
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

pub fn setup_scme(ctx: OpContext) -> Result<ExecutionStep> {
    let state = State::from_attrs(&ctx.attrs)?;

    // The second output is optional
    let with_log_prob = ctx.n_outputs() == 2;

    let (scores, labels, weights): (_, OnnxIndex, _) =
        ctx.input_channels_with_optional().narrow()?;

    let labels = labels.into_i64();

    macro_rules! optional_output (
        ($output:expr, $log_prob:expr, false) => {
            Ok($output)
        };
        ($output:expr, $log_prob:expr, true) => {
            Ok(($output, $log_prob))
        }
    );

    macro_rules! match_arm {
        ($scores:expr, $labels:expr, None, $with_lp:ident) => {
            setup(($scores, $labels), move |(scores, labels)| {
                let log_prob = softmax(scores.to_owned(), 1)?.mapv(|el| el.ln());
                let out = nlll(
                    log_prob.view(),
                    labels.view(),
                    None,
                    state.ignore_index,
                    state.reduction,
                )?;
                optional_output!(out, log_prob, $with_lp)
            })
        };
        ($scores:expr, $labels:expr, $weights:expr, $with_lp:ident) => {
            setup(
                ($scores, $labels, $weights),
                move |(scores, labels, weights)| {
                    let log_prob = softmax(scores.to_owned(), 1)?.mapv(|el| el.ln());
                    let out = nlll(
                        log_prob.view(),
                        labels.view(),
                        Some(weights.view().into_dimensionality()?),
                        state.ignore_index,
                        state.reduction,
                    )?;
                    optional_output!(out, log_prob, $with_lp)
                },
            )
        };
    }

    use FloatingPoint::*;

    if with_log_prob {
        match (scores, weights) {
            (F32(scores), Some(F32(weights))) => match_arm!(scores, labels, weights, true),
            (F32(scores), None) => match_arm!(scores, labels, None, true),
            (F64(scores), Some(F64(weights))) => match_arm!(scores, labels, weights, true),
            (F64(scores), None) => match_arm!(scores, labels, None, true),
            _ => {
                bail!("SoftMaxCrossEntropyLoss: Inconsistent types for 'scores' and 'weights'")
            }
        }
    } else {
        match (scores, weights) {
            (F32(scores), Some(F32(weights))) => match_arm!(scores, labels, weights, false),
            (F32(scores), None) => match_arm!(scores, labels, None, false),
            (F64(scores), Some(F64(weights))) => match_arm!(scores, labels, weights, false),
            (F64(scores), None) => match_arm!(scores, labels, None, false),
            _ => {
                bail!("SoftMaxCrossEntropyLoss: Inconsistent types for 'scores' and 'weights'")
            }
        }
    }
}
