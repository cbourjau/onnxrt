use std::collections::HashMap;

use ndarray::SliceInfo;

use crate::opset::utils::*;

struct Pad {
    mode: Mode,
}

enum Mode {
    Constant,
    Reflect,
    Edge,
    Wrap,
}

fn pad<T>(
    x: ArrayViewD<T>,
    mode: &Mode,
    pads: &[i64],
    fill_value: Option<T>,
    axes: Option<&[i64]>,
) -> Result<ArrayD<T>>
where
    T: Clone,
{
    let pads = Padding::from_pads(pads)?;

    // fill in zero-pads according to `axes`
    let pads: Vec<_> = match axes {
        None => pads,
        Some(axes) => {
            let axes = axes.iter().map(|&ax| convert_negative_index(ax, x.ndim()));

            let axes_to_pad: HashMap<_, _> = HashMap::from_iter(axes.zip(pads.into_iter()));

            (0..x.ndim())
                .map(|ax| match axes_to_pad.get(&ax) {
                    Some(pad) => *pad,
                    None => Padding { front: 0, back: 0 },
                })
                .collect()
        }
    };

    match (mode, fill_value) {
        (Mode::Constant, Some(fill_value)) => pad_constant(x, pads, fill_value),
        _ => bail!("Pad: Only mode: 'constant' is supported"),
    }
}

fn pad_constant<T>(x: ArrayViewD<T>, pads: Vec<Padding>, fill_value: T) -> Result<ArrayD<T>>
where
    T: Clone,
{
    let paddded_shape: Vec<_> = x
        .shape()
        .iter()
        .zip(&pads)
        .map(|(len, pad)| len + pad.front + pad.back)
        .collect();

    let mut arr = ndarray::Array::from_shape_simple_fn(paddded_shape, || fill_value.clone());

    let sinfoelems: Vec<_> = pads
        .iter()
        .map(|pad| ndarray::SliceInfoElem::Slice {
            start: pad.front as _,
            end: if pad.back == 0 {
                None
            } else {
                Some(-(pad.back as isize))
            },
            step: 1,
        })
        .collect();

    let sinfo: SliceInfo<_, IxDyn, IxDyn> = sinfoelems.try_into()?;
    arr.slice_mut(sinfo).assign(&x);
    Ok(arr)
}

#[derive(Copy, Clone, Debug)]
struct Padding {
    front: usize,
    back: usize,
}

impl Padding {
    /// Padding information is encoded as `[x1_begin, x2_begin, ..., x1_end, x2_end,...]`
    fn from_pads(pads: &[i64]) -> Result<Vec<Self>> {
        if pads.len() % 2 != 0 {
            bail!("Pad: Length of 'pads' input must be divisible by two")
        }

        let n_axes = pads.len() / 2;
        let beginnings = &pads[..n_axes];
        let ends = &pads[n_axes..];
        Ok(beginnings
            .iter()
            .zip(ends)
            .map(|(front, back)| Self {
                front: *front as _,
                back: *back as _,
            })
            .collect())
    }
}

impl Pad {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let mode: String = attrs.get_with_default("mode", "constant".to_string())?;
        let mode = match mode.as_str() {
            "constant" => Mode::Constant,
            "reflect" => Mode::Reflect,
            "edge" => Mode::Edge,
            "wrap" => Mode::Wrap,
            dir => bail!("Pad: Invalid value for 'mode': '{}'", dir),
        };

        Ok(Self { mode })
    }
}

pub fn setup_pad(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Pad::from_attrs(&ctx.attrs)?;
    let (data, pads, fill_value, axes): (
        _,
        Receiver<ArcArrayD<i64>>,
        Option<TaggedReceiver>,
        Option<OnnxIndex>,
    ) = ctx.input_channels_with_optional().narrow()?;

    let axes = axes.map(|axes| axes.into_i64());
    let mode = state.mode;

    match fill_value {
        Some(fill_value) => {
            let data_and_fill = vec![data, fill_value].narrow()?;
            setup_all!(
                HomogeneousFixed,
                data_and_fill,
                [data, fill_value],
                setup(
                    (data, pads, fill_value, axes),
                    move |(data, pads, fill_value, axes)| {
                        let pads = pads.as_standard_layout();
                        let pads = pads.as_slice().unwrap();
                        let fill_value = fill_value.into_shape([])?.into_owned().into_scalar();

                        match axes {
                            Some(axes) => {
                                let axes = axes.as_standard_layout();
                                let axes = axes.as_slice().unwrap();
                                pad(data.view(), &mode, pads, Some(fill_value), Some(axes))
                            }
                            None => pad(data.view(), &mode, pads, Some(fill_value), None),
                        }
                    },
                )
            )
        }
        None => {
            let data = vec![data].narrow()?;
            setup_all!(
                TaggedTensorReceiver,
                data,
                data,
                setup((data, pads, axes), move |(data, pads, axes)| {
                    let pads = pads.as_standard_layout();
                    let pads = pads.as_slice().unwrap();

                    match axes {
                        Some(axes) => {
                            let axes = axes.as_standard_layout();
                            let axes = axes.as_slice().unwrap();
                            pad(data.view(), &mode, pads, None, Some(axes))
                        }
                        None => pad(data.view(), &mode, pads, None, None),
                    }
                })
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use std::assert_eq;

    use super::*;

    #[test]
    fn example1() {
        let data = array![[1.0, 1.2], [2.3, 3.4], [4.5, 5.7],].into_dyn();

        let pads = &[0, 2, 0, 0][..];
        let mode = Mode::Constant;

        let constant_value = 0.0;

        let expected_output = array![
            [0.0, 0.0, 1.0, 1.2],
            [0.0, 0.0, 2.3, 3.4],
            [0.0, 0.0, 4.5, 5.7],
        ]
        .into_dyn();

        let candidate = pad(data.view(), &mode, pads, Some(constant_value), None).unwrap();
        assert_eq!(expected_output, candidate);
    }
}
