use crate::opset::utils::*;

#[derive(Debug)]
struct Shape {
    start: i64,
    end: Option<i64>,
}

impl Shape {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let start = attrs.get_with_default("start", 0)?;
        let end = attrs.get_optional("end")?;

        Ok(Self { start, end })
    }

    fn compute<T>(&self, data: ArcArrayD<T>) -> Result<ArcArrayD<i64>> {
        let ndim = data.ndim();

        // Specs is specific about the clamping; hence a special
        // function here.
        let start = convert_negative_index_clamp(self.start, ndim);

        let end = self
            .end
            .map(|end| convert_negative_index_clamp(end, ndim))
            .unwrap_or(ndim);

        let shape = &data
            .shape()
            .get(start..end)
            .ok_or_else(|| anyhow!("Shape: 'start' or 'end' index is out of range"))?;

        Ok(ArcArray::from_iter(shape.iter().map(|el| *el as i64)).into_dyn())
    }
}

pub fn setup_shape(ctx: OpContext) -> Result<ExecutionStep> {
    let state = Shape::from_attrs(&ctx.attrs)?;
    let data = ctx.subscribe_to_inputs().narrow()?;
    setup_all!(
        TaggedReceiver,
        data,
        data,
        setup(data, move |data| state.compute(data))
    )
}

fn convert_negative_index_clamp(idx: i64, len: usize) -> usize {
    let len = len as i64;
    if idx < 0 { len + idx } else { idx }.max(0).min(len) as usize
}
