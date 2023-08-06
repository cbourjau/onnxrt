use std::{iter::Sum, ops::AddAssign};

use ndarray::{SliceInfo, SliceInfoElem};
use num::Float;

use crate::opset::utils::*;

#[derive(Debug)]
struct Conv {
    padding: PaddingType,
    // dilation value along each spatial axis of the filter. If not
    // present, the dilation defaults is 1 along each spatial axis.
    dilations: Option<Vec<i64>>,
    // number of groups input channels and output channels are divided into.
    group: usize, // (default is 1)
    // Stride along each spatial axis. If not present, the stride
    // defaults is 1 along each spatial axis.
    strides: Option<Vec<usize>>,
}

#[derive(Debug)]
enum PaddingType {
    Explicit((Pads, Pads)), // 'NOTSET' is the name in the standard
    SameUpper,
    SameLower,
    Valid,
}

/// Pads for a single axis
#[derive(Debug, Copy, Clone)]
struct Pads {
    low: usize,
    high: usize,
}

impl PaddingType {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let auto_pad = attrs.get_with_default("auto_pad", "NOTSET".to_string())?;

        let pads: Option<Vec<Pads>> = attrs.get_optional::<Vec<i64>>("pads")?.map(|vec| {
            let (lows, highs) = vec.as_slice().split_at(vec.len() / 2);
            lows.iter()
                .zip(highs)
                .map(|(&low, &high)| Pads {
                    low: low as usize,
                    high: high as usize,
                })
                .collect()
        });

        Ok(match (auto_pad.as_str(), pads) {
            ("NOTSET", Some(pads)) => Self::Explicit((pads[0], pads[1])),
            ("SAME_UPPER", None) => Self::SameUpper,
            ("SAME_LOWER", None) => Self::SameLower,
            ("VALID", None) => Self::Valid,
            ("NOTSET", None) => {
                bail!("Conv: Either 'auto_pad' or 'pads' must be set.")
            }
            (s, _) => bail!("Conv: Invalid argument for 'auto_pad': '{}'", s),
        })
    }
}

impl Conv {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let dilations: Option<Vec<i64>> = attrs.get_optional("dilations")?;
        let group = attrs.get_with_default("group", 1i64)? as usize;

        let strides: Option<Vec<usize>> = attrs
            .get_optional::<Vec<i64>>("strides")?
            .map(|vec| vec.into_iter().map(|el| el as usize).collect());

        let padding = PaddingType::from_attrs(attrs)?;
        match padding {
            PaddingType::SameUpper | PaddingType::SameLower => {
                if let Some(strides) = strides.as_ref() {
                    if strides.iter().any(|&el| el > 1) {
                        bail!(
                            "Conv: Strides != 1 may only be used with explicit padding or 'VALID' 'auto_pad'"
                        );
                    }
                };
            }
            _ => {}
        };

        let out = Self {
            padding,
            dilations,
            group,
            // kernel_shape: attrs.get_optional("kernel_shape")?,
            strides,
        };

        if (out.dilation_axis(0) != 1) || (out.dilation_axis(1) != 1) {
            bail!("Conv: 'dilations' != 1 are not supported.");
        }
        Ok(out)
    }

    fn stride_axis(&self, axis: usize) -> usize {
        self.strides.as_ref().map(|v| v[axis]).unwrap_or(1)
    }
    fn dilation_axis(&self, axis: usize) -> usize {
        self.dilations.as_ref().map(|v| v[axis]).unwrap_or(1) as usize
    }

    fn validation<T>(
        &self,
        x: ArrayView4<T>,
        w: ArrayView4<T>,
        b: Option<ArrayView1<T>>,
    ) -> Result<()> {
        let in_channels = x.shape()[1];
        let in_ch_per_group = w.shape()[1];

        if in_channels / self.group != in_ch_per_group {
            bail!("Conv: Inconsistent input shapes. The second dimension of 'X' times the number of groups given in the 'group' attribute must match the second dimension of 'W'. ")
        };

        let out_channels = w.shape()[0];

        if let Some(b) = b {
            if out_channels != b.shape()[0] {
                bail!("Conv: Inconsistent input shapes. The length of the bias tensor must match the first dimension of the weight tensor.")
            }
        };

        Ok(())
    }

    fn output_shape<T>(
        &self,
        x: ArrayView4<T>,
        w: ArrayView4<T>,
    ) -> Result<(usize, usize, usize, usize)> {
        let batch_size = x.shape()[0];
        let out_chs = w.shape()[0];

        let spatial_in = &x.shape()[2..];
        let k_shape = &w.shape()[2..];
        let dilated_k_shape = (
            self.dilation_axis(0) * k_shape[0],
            self.dilation_axis(1) * k_shape[1],
        );
        let pads = self.padding(x, w)?;

        let max_idx_0 =
            (spatial_in[0] + pads.0.low + pads.0.high - dilated_k_shape.0) / self.stride_axis(0);

        let max_idx_1 =
            (spatial_in[1] + pads.1.low + pads.1.high - dilated_k_shape.1) / self.stride_axis(1);

        // +1 to go from index to length
        Ok((batch_size, out_chs, max_idx_0 + 1, max_idx_1 + 1))
    }

    fn padding<T>(&self, x: ArrayView4<T>, w: ArrayView4<T>) -> Result<(Pads, Pads)> {
        let x_shape = &x.shape()[2..];
        let k_shape = &w.shape()[2..];

        let dilated_k_shape = (
            self.dilation_axis(0) * k_shape[0],
            self.dilation_axis(1) * k_shape[1],
        );
        // Input shape is output shape; we look for the padding to make it so!
        let out_max_idx = (x_shape[0] - 1, x_shape[1] - 1);

        // P = ((S-1)*W-S+F)/2
        let total_pad0 = out_max_idx.0 * self.stride_axis(0) + dilated_k_shape.0 - x_shape[0];
        let total_pad1 = out_max_idx.1 * self.stride_axis(1) + dilated_k_shape.1 - x_shape[1];

        Ok(match self.padding {
            PaddingType::Explicit(pads) => pads,
            PaddingType::Valid => (Pads { low: 0, high: 0 }, Pads { low: 0, high: 0 }),
            PaddingType::SameUpper => (
                split_pad(total_pad0).same_upper(),
                split_pad(total_pad1).same_upper(),
            ),
            PaddingType::SameLower => (
                split_pad(total_pad0).same_lower(),
                split_pad(total_pad1).same_lower(),
            ),
        })
    }
    /// X-shape: (N, C_in​,  x0, x1)
    /// w-shape: (C_out, C_in / groups, w0, w1)
    /// b-shape: (C_out,)
    /// Return shape: (N, C_out​, out0, out1)
    fn compute<T>(
        &self,
        x: ArrayViewD<T>,
        w: ArrayViewD<T>,
        b: Option<ArrayViewD<T>>,
    ) -> Result<ArrayD<T>>
    where
        T: Float + Sum + AddAssign,
    {
        let x = x.into_dimensionality()?;
        let w = w.into_dimensionality()?;
        let b = b.map(|b| b.into_dimensionality()).transpose()?;

        self.validation(x, w, b)?;

        let kernel_size = &w.shape()[2..];
        let (pads0, pads1) = self.padding(x, w)?;

        let len0 = x.shape()[2];
        let len1 = x.shape()[3];
        let strides = &[self.stride_axis(0), self.stride_axis(1)];

        let mut out = Array4::<T>::zeros(self.output_shape(x, w)?);

        // Batch iteration
        for (x, mut out) in x.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            // Iterate through input/output channels in group chunks
            for ((x, w), mut out) in x
                .axis_chunks_iter(Axis(0), x.shape()[0] / self.group)
                .zip(w.axis_chunks_iter(Axis(0), w.shape()[0] / self.group))
                .zip(out.axis_chunks_iter_mut(Axis(0), out.shape()[0] / self.group))
            {
                for (i, mut out) in out.axis_iter_mut(Axis(1)).enumerate() {
                    for (j, mut out) in out.axis_iter_mut(Axis(1)).enumerate() {
                        let s_x =
                            slice_info_x(i, len0, j, len1, kernel_size, pads0, pads1, strides);
                        let s_w =
                            slice_info_w(i, len0, j, len1, kernel_size, pads0, pads1, strides);
                        let tmp = x.slice(s_x).to_owned() * w.slice(s_w);
                        out.assign(&tmp.sum_axis(Axis(1)).sum_axis(Axis(1)).sum_axis(Axis(1)));
                    }
                }
            }
        }
        if let Some(b) = b {
            let b_len = b.len();
            let b = b.into_shape((1, b_len, 1, 1))?;
            out += &b;
        }

        Ok(out.into_dyn())
    }
}

#[allow(clippy::too_many_arguments)]
fn slice_info_x(
    i: usize,
    max_i: usize,
    j: usize,
    max_j: usize,
    k_size: &[usize],
    padding_i: Pads,
    padding_j: Pads,
    stride: &[usize],
) -> SliceInfo<[SliceInfoElem; 4], ndarray::Dim<[usize; 3]>, ndarray::Dim<[usize; 4]>> {
    // Compute slices to be applied to the input x
    let compute_1d_x_slice = |stride, pad, idx, max, k_size| {
        let start = (stride * idx) as i64 - pad as i64;
        let end = ((start + k_size as i64) as usize).min(max);
        (start.max(0) as usize)..end
    };

    ndarray::s![
        NewAxis,
        ..,
        compute_1d_x_slice(stride[0], padding_i.low, i, max_i, k_size[0]),
        compute_1d_x_slice(stride[1], padding_j.low, j, max_j, k_size[1])
    ]
}

#[allow(clippy::too_many_arguments)]
fn slice_info_w(
    i: usize,
    max_i: usize,
    j: usize,
    max_j: usize,
    k_size: &[usize],
    padding_i: Pads,
    padding_j: Pads,
    stride: &[usize],
) -> SliceInfo<[SliceInfoElem; 4], ndarray::Dim<[usize; 4]>, ndarray::Dim<[usize; 4]>> {
    // Compute slices of the kernel
    let compute_1d_w_slice = |stride, pad, idx, max, k_size| {
        let start = (stride * idx) as i64 - pad as i64;
        let end = (max as i64 - start).min(k_size as i64) as usize;

        (start.min(0).unsigned_abs() as usize)..end
    };

    ndarray::s![
        ..,
        ..,
        compute_1d_w_slice(stride[0], padding_i.low, i, max_i, k_size[0]),
        compute_1d_w_slice(stride[1], padding_j.low, j, max_j, k_size[1])
    ]
}

struct SplitPad {
    smaller: usize,
    larger: usize,
}
impl SplitPad {
    fn same_upper(&self) -> Pads {
        Pads {
            low: self.smaller,
            high: self.larger,
        }
    }
    fn same_lower(&self) -> Pads {
        Pads {
            low: self.larger,
            high: self.smaller,
        }
    }
}
fn split_pad(total_pad: usize) -> SplitPad {
    if total_pad % 2 == 0 {
        // both are equal
        SplitPad {
            smaller: total_pad / 2,
            larger: total_pad / 2,
        }
    } else {
        SplitPad {
            smaller: total_pad / 2,
            larger: total_pad / 2 + 1,
        }
    }
}

pub fn setup_conv(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Conv::from_attrs(&ctx.attrs)?;

    // The `narrow` traits are not implemented for optional inputs
    let (x, w, b): (Option<_>, Option<_>, Option<TaggedReceiver>) =
        ctx.input_channels_with_optional().narrow()?;

    let x = x.ok_or_else(|| anyhow!("Conv: Mandatory input 'x' is missing"))?;
    let w = w.ok_or_else(|| anyhow!("Conv: Mandatory input 'w' is missing"))?;

    use TaggedReceiver::*;
    match (x, w, b) {
        (F32(x), F32(w), None) => setup((x, w), move |(x, w)| {
            Ok::<ArcArrayD<_>, _>(state.compute(x.view(), w.view(), None)?.into())
        }),
        (F64(x), F64(w), None) => setup((x, w), move |(x, w)| {
            Ok::<ArcArrayD<_>, _>(state.compute(x.view(), w.view(), None)?.into())
        }),
        // With bias
        (F32(x), F32(w), Some(F32(b))) => setup((x, w, b), move |(x, w, b)| {
            Ok::<ArcArrayD<_>, _>(state.compute(x.view(), w.view(), Some(b.view()))?.into())
        }),
        (F64(x), F64(w), Some(F64(b))) => setup((x, w, b), move |(x, w, b)| {
            Ok::<ArcArrayD<_>, _>(state.compute(x.view(), w.view(), Some(b.view()))?.into())
        }),
        (x, w, b) => bail!(
            "Conv: Unimplemented input types: x: {:?}, w: {:?} and b:{:?}",
            &x,
            &w,
            &b
        ),
    }
}
