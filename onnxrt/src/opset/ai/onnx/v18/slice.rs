use ndarray::SliceInfoElem;
use num_traits::AsPrimitive;

use crate::opset::utils::*;

fn slice<T, Idx>(
    data: ArrayViewD<T>,
    starts: ArrayView1<Idx>,
    ends: ArrayView1<Idx>,
    axes: Option<ArrayView1<Idx>>,
    steps: Option<ArrayView1<Idx>>,
) -> Result<ArrayD<T>>
where
    T: Clone,
    Idx: Index + AsPrimitive<i64>,
{
    let input_rank = data.ndim();
    let shape = data.shape();
    let axes: Vec<_> = match axes {
        Some(axes) => axes.iter().map(|idx| idx.for_len(input_rank)).collect(),
        None => (0..input_rank).collect(),
    };
    let steps: Vec<_> = match steps {
        Some(steps) => steps.iter().map(|idx| idx.as_() as isize).collect(),
        None => vec![1; starts.len()],
    };

    if steps.iter().any(|&el| el < 0) {
        bail!("Slice: Negative steps are not supported due to an ambiguity in the standard.")
    }

    if axes.len() != steps.len() {
        bail!(
            "Slice: Inconsistent number of elements in 'axes' ({:?}) and 'steps' ({:?})",
            axes.len(),
            steps.len()
        )
    }

    let starts = starts
        .iter()
        .zip(&axes)
        .map(|(start, &ax)| start.for_len(shape[ax]).min(shape[ax]) as isize);

    let ends = ends.iter().zip(&axes).map(|(end, &ax)| {
        let idx = end.for_len(shape[ax]);
        idx.min(shape[ax]) as isize
    });

    let mut slice_infos: Vec<SliceInfoElem> = vec![(..).into(); input_rank];

    for (((start, end), ax), step) in starts.zip(ends).zip(&axes).zip(&steps) {
        slice_infos[*ax] = SliceInfoElem::Slice {
            start: start.min(end),
            end: Some(start.max(end)),
            step: *step,
        };
    }

    Ok(data.slice(slice_infos.as_slice()).to_owned())
}

pub fn setup_slice(ctx: OpContext) -> Result<ExecutionStep> {
    let (data, starts, ends, axes, steps): (
        _,
        OnnxIndex,
        OnnxIndex,
        Option<OnnxIndex>,
        Option<OnnxIndex>,
    ) = ctx.input_channels_with_optional().narrow()?;

    let starts = starts.into_i64();
    let ends = ends.into_i64();
    let axes = axes.map(|axes| axes.into_i64());
    let steps = steps.map(|steps| steps.into_i64());

    match (axes, steps) {
        (Some(axes), Some(steps)) => {
            setup_all!(
                TaggedTensorReceiver,
                data,
                data,
                setup(
                    (data, starts, ends, axes, steps),
                    move |(data, starts, ends, axes, steps)| {
                        slice(
                            data.view(),
                            starts.into_dimensionality()?.view(),
                            ends.into_dimensionality()?.view(),
                            Some(axes.into_dimensionality()?.view()),
                            Some(steps.into_dimensionality()?.view()),
                        )
                    }
                )
            )
        }
        (None, Some(steps)) => {
            setup_all!(
                TaggedTensorReceiver,
                data,
                data,
                setup(
                    (data, starts, ends, steps),
                    move |(data, starts, ends, steps)| {
                        slice(
                            data.view(),
                            starts.into_dimensionality()?.view(),
                            ends.into_dimensionality()?.view(),
                            None,
                            Some(steps.into_dimensionality()?.view()),
                        )
                    }
                )
            )
        }
        (Some(axes), None) => {
            setup_all!(
                TaggedTensorReceiver,
                data,
                data,
                setup(
                    (data, starts, ends, axes),
                    move |(data, starts, ends, axes)| {
                        slice(
                            data.view(),
                            starts.into_dimensionality()?.view(),
                            ends.into_dimensionality()?.view(),
                            Some(axes.into_dimensionality()?.view()),
                            None,
                        )
                    }
                )
            )
        }
        (None, None) => {
            setup_all!(
                TaggedTensorReceiver,
                data,
                data,
                setup((data, starts, ends), move |(data, starts, ends)| {
                    slice(
                        data.view(),
                        starts.into_dimensionality()?.view(),
                        ends.into_dimensionality()?.view(),
                        None,
                        None,
                    )
                })
            )
        }
    }
}
