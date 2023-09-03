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
    T: Clone + std::fmt::Debug,
    Idx: Index + std::fmt::Debug + AsPrimitive<i64>,
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
    let (data, starts, ends, axes, steps): (Option<_>, Option<_>, Option<_>, _, _) =
        ctx.input_channels_with_optional().narrow()?;

    let data = data.ok_or_else(|| anyhow!("Slice: Required input 'data' is missing"))?;
    let starts = starts.ok_or_else(|| anyhow!("Slice: Required input 'starts' is missing"))?;
    let ends = ends.ok_or_else(|| anyhow!("Slice: Required input 'ends' is missing"))?;

    use TaggedReceiver::*;
    macro_rules! match_index_variants {
        ($data:expr, $($variant:ident),*) => {
            match (starts, ends, axes, steps) {
                $(($variant(starts), $variant(ends), Some($variant(axes)), Some($variant(steps))) => setup(
                    ($data, starts, ends, axes, steps),
                    |(data, starts, ends, axes, steps)| {
                        Ok::<ArcArrayD<_>, _>(
                            slice(
                                data.view(),
                                starts.into_dimensionality()?.view(),
                                ends.into_dimensionality()?.view(),
                                Some(axes.into_dimensionality()?.view()),
                                Some(steps.into_dimensionality()?.view()),
                            )?
                                .into_dyn()
                                .into(),
                        )
                    },
                ),
                  ($variant(starts), $variant(ends), Some($variant(axes)), None) => {
                      setup(($data, starts, ends, axes), |(data, starts, ends, axes)| {
                          Ok::<ArcArrayD<_>, _>(
                              slice(
                                  data.view(),
                                  starts.into_dimensionality()?.view(),
                                  ends.into_dimensionality()?.view(),
                                  Some(axes.into_dimensionality()?.view()),
                                  None,
                              )?
                                  .into_dyn()
                                  .into(),
                          )
                      })
                  }
                  ($variant(starts), $variant(ends), None, Some($variant(steps))) => setup(
                      ($data, starts, ends, steps),
                      |(data, starts, ends, steps)| {
                          Ok::<ArcArrayD<_>, _>(
                              slice(
                                  data.view(),
                                  starts.into_dimensionality()?.view(),
                                  ends.into_dimensionality()?.view(),
                                  None,
                                  Some(steps.into_dimensionality()?.view()),
                              )?
                                  .into_dyn()
                                  .into(),
                          )
                      },
                  ),
                  ($variant(starts), $variant(ends), None, None) =>
                  setup(($data, starts, ends), |(data, starts, ends)| {
                      Ok::<ArcArrayD<_>, _>(
                          slice(data.view(), starts.into_dimensionality()?.view(), ends.into_dimensionality()?.view(), None, None)?
                              .into_dyn()
                              .into(),
                      )
                  }),
                )*
                    _ => bail!("Slice: All index-typed inputs must have identical types"),
            }
        };
    }
    match data {
        U8(data) => match_index_variants!(data, I32, I64),
        U16(data) => match_index_variants!(data, I32, I64),
        U32(data) => match_index_variants!(data, I32, I64),
        U64(data) => match_index_variants!(data, I32, I64),

        I8(data) => match_index_variants!(data, I32, I64),
        I16(data) => match_index_variants!(data, I32, I64),
        I32(data) => match_index_variants!(data, I32, I64),
        I64(data) => match_index_variants!(data, I32, I64),

        F32(data) => match_index_variants!(data, I32, I64),
        F64(data) => match_index_variants!(data, I32, I64),

        Bool(data) => match_index_variants!(data, I32, I64),
        String(data) => match_index_variants!(data, I32, I64),
    }
}
