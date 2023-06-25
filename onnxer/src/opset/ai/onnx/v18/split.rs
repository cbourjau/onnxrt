use crate::opset::utils::*;

pub fn setup_split(ctx: OpContext) -> Result<ExecutionStep> {
    let n_outputs = ctx.n_outputs();

    let state = Split::new(&ctx)?;

    // `input_channels` creates channels and must only be called once!
    let input_channels = ctx.input_channels();
    if input_channels.len() > 2 {
        bail!("Split: Unexpected number of inputs");
    }

    let mut input_channels = input_channels.into_iter();
    let a_ch = input_channels
        .next()
        .ok_or_else(|| anyhow!("Split: Missing required input 'input'"))?;

    // Optional input
    let split = if let Some(split) = input_channels.next() {
        if let TaggedReceiver::I64(rec) = split {
            Some(rec)
        } else {
            bail!("Split: 'split' input must be of type 'tensor(i64)'");
        }
    } else {
        None
    };
    match split {
        Some(split) => {
            setup_all!(
                TaggedReceiver,
                a_ch,
                a_ch,
                setup_variadic_out(
                    (a_ch, split),
                    move |(a, split)| state.split_dyn(a, split),
                    n_outputs
                )
            )
        }
        None => {
            setup_all!(
                TaggedReceiver,
                a_ch,
                a_ch,
                setup_variadic_out((a_ch,), move |(a,)| state.split(a), n_outputs)
            )
        }
    }
}

struct Split {
    axis: i64,
    num_outputs: usize,
}

impl Split {
    fn new(ctx: &OpContext) -> Result<Self> {
        let num_outputs: Option<i64> = ctx.attrs.get_optional("num_outputs")?;

        let num_outputs = num_outputs
            .map(|el| el as usize)
            .unwrap_or_else(|| ctx.n_outputs());

        let axis = ctx.attrs.get_with_default("axis", 0i64)?;
        Ok(Self { axis, num_outputs })
    }

    fn split<T>(&self, a: ArcArrayD<T>) -> Result<Vec<ArcArrayD<T>>>
    where
        T: Clone,
    {
        let axis = convert_negative_index(self.axis, a.shape().len());

        let len_axis = a.shape()[axis];

        let chunk_size = {
            let num_outputs = self.num_outputs;
            if len_axis < num_outputs {
                bail!(
                    "Split: Cannot split dimension with '{}' elements into '{}' chunks.",
                    len_axis,
                    num_outputs,
                )
            }
            if len_axis % num_outputs == 0 {
                len_axis / num_outputs
            } else {
                len_axis / num_outputs + 1
            }
        };
        Ok(a.axis_chunks_iter(Axis(axis), chunk_size)
            .map(|arr| arr.to_owned().into())
            .collect())
    }

    fn split_dyn<T>(&self, a: ArcArrayD<T>, split: ArcArrayD<i64>) -> Result<Vec<ArcArrayD<T>>>
    where
        T: Clone + std::fmt::Debug,
    {
        let axis = convert_negative_index(self.axis, a.shape().len());

        let split = split.into_dimensionality::<Ix1>()?.mapv(|el| el as usize);
        if split.sum() != a.shape()[axis] {
            bail!("Split: 'input' and 'split' inputs are incompatible.");
        }
        if split.len() != self.num_outputs {
            bail!("Split: 'split' input does not match expected number of outputs.");
        }

        let axis = Axis(axis);
        let mut out = vec![];
        let a = a.to_owned();

        let mut rest = a.view();
        for el in &split {
            let this;
            (this, rest) = rest.split_at(axis, *el);
            out.push(this.into_owned().into());
        }
        Ok(out)
    }
}
