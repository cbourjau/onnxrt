use crate::opset::utils::*;
use crate::Value;

struct ConstantOfShape {
    // rank-0 tensor
    value: Value,
}

impl ConstantOfShape {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        let value = attrs
            .get_optional("value")?
            .unwrap_or_else(|| Value::F32(arr0(0.0f32).into_dyn().into()));

        // Validate that it is rank-0 tensor
        use Value::*;
        let value = match value {
            Bool(t) => Bool(t.into_shape((1,))?.into_dyn()),
            F32(t) => F32(t.into_shape((1,))?.into_dyn()),
            F64(t) => F64(t.into_shape((1,))?.into_dyn()),

            I8(t) => I8(t.into_shape((1,))?.into_dyn()),
            I16(t) => I16(t.into_shape((1,))?.into_dyn()),
            I32(t) => I32(t.into_shape((1,))?.into_dyn()),
            I64(t) => I64(t.into_shape((1,))?.into_dyn()),

            U8(t) => U8(t.into_shape((1,))?.into_dyn()),
            U16(t) => U16(t.into_shape((1,))?.into_dyn()),
            U32(t) => U32(t.into_shape((1,))?.into_dyn()),
            U64(t) => U64(t.into_shape((1,))?.into_dyn()),

            String(t) => String(t.into_shape((1,))?.into_dyn()),
        };

        Ok(Self { value })
    }
}

pub fn setup_constant_of_shape(ctx: OpContext) -> Result<ExecutionStep> {
    let ConstantOfShape { value } = ConstantOfShape::from_attrs(&ctx.attrs)?;

    let shape: Receiver<ArcArrayD<i64>> = ctx.subscribe_to_inputs().narrow()?;

    macro_rules! match_arm {
        ($v:tt) => {
            setup((shape,), move |(shape,)| {
                let shape: Vec<usize> = shape
                    .into_iter()
                    .map(|el| Ok(usize::try_from(el)?))
                    .collect::<Result<_>>()?;
                Ok($v.broadcast(shape.as_slice()).unwrap().to_owned())
            })
        };
    }
    match value {
        Value::F32(v) => match_arm!(v),
        Value::F64(v) => match_arm!(v),

        Value::I8(v) => match_arm!(v),
        Value::I16(v) => match_arm!(v),
        Value::I32(v) => match_arm!(v),
        Value::I64(v) => match_arm!(v),

        Value::U8(v) => match_arm!(v),
        Value::U16(v) => match_arm!(v),
        Value::U32(v) => match_arm!(v),
        Value::U64(v) => match_arm!(v),

        Value::Bool(v) => match_arm!(v),
        Value::String(v) => match_arm!(v),
    }
}
