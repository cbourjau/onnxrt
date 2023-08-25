use std::ops::Sub;

use ndarray::ScalarOperand;
use num::Float;
use num_traits::{AsPrimitive, Bounded, FromPrimitive, One};

use super::reduce::reduce_mean;
use crate::opset::utils::*;

#[allow(dead_code)]
struct LayerNormalization {
    axis: i64,
    epsilon: f32,
    // SPECS: The integer mapping to output types is undefined(?)
    stash_type: i64,
}

/// Perform layer normalization. Types of optional outputs are not
/// properly defined by the standard and therefor not supported.
fn layer_normalization<T>(
    axis: i64,
    x: ArrayD<T>,
    scale: ArrayViewD<T>,
    bias: Option<ArrayViewD<T>>,
    epsilon: T,
) -> (ArcArrayD<T>, ArcArrayD<T>, ArcArrayD<T>)
where
    T: Float + FromPrimitive + One + ScalarOperand + Sub<Output = T> + std::fmt::Debug + Bounded,
    usize: AsPrimitive<T>,
{
    let axis = Axis(convert_negative_index(axis, x.ndim()));
    let reduce_axes: Vec<_> = (axis.0..x.ndim()).map(|el| el as i64).collect();
    let mean = reduce_mean(x.view(), reduce_axes.as_slice(), true, false).unwrap();
    let d = x - &mean;
    let dd = d.mapv(|el| el * el);
    let var = reduce_mean(dd.view(), reduce_axes.as_slice(), true, false).unwrap();
    let var_eps = var + epsilon;
    let std_dev = var_eps.mapv(|el| el.sqrt());
    let inv_std_dev = std_dev.mapv(|el| T::one() / el);
    let normalized = d * &inv_std_dev;

    let normalized_scaled = normalized * scale;
    let y = {
        if let Some(bias) = bias {
            normalized_scaled + bias
        } else {
            normalized_scaled
        }
    };
    (y.into(), mean.into(), inv_std_dev.into())
}

impl LayerNormalization {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let axis = attrs.get_with_default("axis", -1)?;
        let epsilon = attrs.get_with_default("epsilon", 1e-5f32)?;
        let stash_type = attrs.get_with_default("stash_type", 1)?;

        if stash_type != 1 {
            bail!("LayerNormalization: Only 'stash_type=1' is currently supported")
        }

        Ok(Self {
            axis,
            epsilon,
            stash_type,
        })
    }
}

pub fn setup_ln(ctx: OpContext) -> Result<ExecutionStep> {
    let state = LayerNormalization::from_attrs(&ctx.attrs)?;

    let (x, scale, bias): (_, _, Option<_>) = ctx.input_channels_with_optional().narrow()?;

    use FloatingPoint::*;
    match (x, scale, bias) {
        (F32(x), F32(scale), Some(F32(bias))) => {
            setup((x, scale, bias), move |(x, scale, bias)| {
                Ok(layer_normalization(
                    state.axis,
                    x.to_owned(),
                    scale.view(),
                    Some(bias.view()),
                    state.epsilon,
                ))
            })
        }
        // (F32(x), F32(scale), None) => setup((x, scale), move |(x, scale)| {
        //     Ok::<ArcArrayD<_>, _>(
        //         layer_normalization(state.axis, x.to_owned(), scale.view(), None, state.epsilon)
        //             .into(),
        //     )
        // }),
        // (F64(x), F64(scale), Some(F64(bias))) => {
        //     setup((x, scale, bias), move |(x, scale, bias)| {
        //         Ok::<ArcArrayD<_>, _>(
        //             layer_normalization(
        //                 state.axis,
        //                 x.to_owned(),
        //                 scale.view(),
        //                 Some(bias.view()),
        //                 state.epsilon.into(),
        //             )
        //             .into(),
        //         )
        //     })
        // }
        // (F64(x), F64(scale), None) => setup((x, scale), move |(x, scale)| {
        //     Ok::<ArcArrayD<_>, _>(
        //         layer_normalization(
        //             state.axis,
        //             x.to_owned(),
        //             scale.view(),
        //             None,
        //             state.epsilon.into(),
        //         )
        //         .into(),
        //     )
        // }),
        _ => bail!("LayerNormalization: All inputs must be of the identical type."),
    }
}
