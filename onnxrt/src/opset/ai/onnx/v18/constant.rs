use crate::Value;
use crate::opset::utils::*;

struct Constant {
    value: Value,
}

impl Constant {
    fn from_attrs(attrs: &Attributes) -> Result<Self> {
        if attrs.get("sparse_value").is_ok() {
            bail!("Sparse tensors are not supported.")
        }
        let floats = {
            attrs
                .get_optional("value_float")?
                .map(|el| ndarray::arr0(el).into_dyn().into())
                .or(attrs.get_optional("value_floats")?)
                .map(Value::F32)
        };
        let ints = {
            attrs
                .get_optional("value_int")?
                .map(|el| ndarray::arr0(el).into_dyn().into())
                .or(attrs.get_optional("value_ints")?)
                .map(Value::I64)
        };
        let strings = {
            attrs
                .get_optional("value_string")?
                .map(|el| ndarray::arr0(el).into_dyn().into())
                .or(attrs.get_optional("value_string")?)
                .map(Value::String)
        };

        let value = attrs
            .get_optional("value")?
            .or(floats)
            .or(ints)
            .or(strings)
            .ok_or_else(|| anyhow!("Constant: Invalid attributes"))?;

        Ok(Self { value })
    }
}

pub fn setup_constant(ctx: OpContext) -> Result<ExecutionStep> {
    let Constant { value } = Constant::from_attrs(&ctx.attrs)?;
    match value {
        Value::Bool(v) => setup((), move |()| Ok(v.clone())),
        Value::F32(v) => setup((), move |()| Ok(v.clone())),
        Value::F64(v) => setup((), move |()| Ok(v.clone())),

        Value::I8(v) => setup((), move |()| Ok(v.clone())),
        Value::I16(v) => setup((), move |()| Ok(v.clone())),
        Value::I32(v) => setup((), move |()| Ok(v.clone())),
        Value::I64(v) => setup((), move |()| Ok(v.clone())),

        Value::U8(v) => setup((), move |()| Ok(v.clone())),
        Value::U16(v) => setup((), move |()| Ok(v.clone())),
        Value::U32(v) => setup((), move |()| Ok(v.clone())),
        Value::U64(v) => setup((), move |()| Ok(v.clone())),

        Value::String(v) => setup((), move |()| Ok(v.clone())),
    }
}
