use crate::opset::utils::*;

#[derive(Debug)]
struct Transpose {
    perm: Vec<i64>,
}

impl Transpose {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let perm = attrs.get_with_default("perm", vec![])?;

        Ok(Self { perm })
    }

    fn compute<T>(&self, data: ArcArrayD<T>) -> Result<ArcArrayD<T>> {
        let rank = data.ndim();

        if self.perm.is_empty() {
            Ok(data.reversed_axes())
        } else {
            if *self.perm.iter().reduce(Ord::max).unwrap() as usize > rank {
                bail!("Transpose: 'axes' index is out of range")
            }
            let new_axes: Vec<_> = self.perm.iter().map(|el| *el as usize).collect();
            Ok(data.permuted_axes(new_axes))
        }
    }
}

pub fn setup_transpose(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Transpose::from_attrs(&ctx.attrs)?;
    let data = ctx.subscribe_to_inputs().narrow()?;
    setup_all!(
        TaggedReceiver,
        data,
        data,
        setup((data,), move |(data,)| state.compute(data))
    )
}
