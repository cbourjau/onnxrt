use std::collections::HashMap;

use onnx_protos::onnx_proto3::GraphProto;

use crate::{
    channels::value_channel, opset::utils::*, session::get_implicit_inputs, Session, Value,
};

struct Loop {
    body: Session,
}
impl Loop {
    fn from_context(ctx: &OpContext) -> Result<(Self, HashMap<String, TaggedReceiver>)> {
        let g: GraphProto = ctx.attrs.get("body")?.try_into()?;
        let body = Session::from_nested_graph(&g, ctx.opset, ctx.scope)?;

        let captured_inputs: HashMap<_, _> = get_implicit_inputs(&g)
            .iter()
            .map(|name| (name.to_string(), ctx.subscribe(name)))
            .collect();

        Ok((Self { body }, captured_inputs))
    }

    fn run(
        &self,
        init_m: Option<ArcArrayD<i64>>,
        init_cond: Option<ArcArrayD<bool>>,
        carried: Vec<Value>,
        captured_inputs: HashMap<String, Value>,
    ) -> Result<Vec<Value>> {
        let max_n = {
            if let Some(arr) = init_m {
                Some(arr.into_shape(())?.to_owned().into_scalar())
            } else {
                None
            }
        };
        let mut cond = Value::Bool(init_cond.unwrap_or_else(|| array![false].into_dyn().into()));

        let mut carried = carried;
        let mut iter_count = Value::I64(array![0].into_dyn().into());

        // -1 because of the break condition
        let n_scan_outputs = self.body.outputs.len() - carried.len() - 1;
        let mut scan_outputs: Vec<Vec<Value>> = (0..n_scan_outputs).map(|_| vec![]).collect();
        while should_continue(max_n, &iter_count, &cond)? {
            let values = [iter_count.clone(), cond]
                .into_iter()
                .chain(carried.iter().cloned());
            let inputs = self
                .body
                .get_explicit_input_names()
                .iter()
                .map(|s| s.to_string())
                .zip(values)
                .chain(captured_inputs.clone().into_iter())
                .collect();

            let outputs = self.body.run_indexed(&inputs)?;

            let (updated_cond, updated_carried, scan_output) = {
                let (cond, carried_and_scan) = outputs.split_first().unwrap();
                let (carried, scan) = carried_and_scan.split_at(carried.len());
                (cond, carried, scan)
            };

            // Update condition and loop carried variables
            cond = updated_cond.clone();
            carried.clear();
            carried.extend_from_slice(updated_carried);

            // Collect scan outputs into their own stacks
            for (val, stack) in scan_output.iter().zip(scan_outputs.iter_mut()) {
                stack.push(val.clone());
            }

            // Advance counter
            if let Value::I64(ref mut arr) = iter_count {
                arr.mapv_inplace(|el| el + 1);
            }
        }

        let mut final_output = carried;

        // Concatenate scan outputs
        for stack in &scan_outputs {
            final_output.push(concat_values(stack)?);
        }
        Ok(final_output)
    }
}

/// Concatenate a non-empty stack of values along a newly added `0` axis.
/// All values must be of the same type and the input must not be empty.
fn concat_values(values: &[Value]) -> Result<Value> {
    macro_rules! match_arm {
        ($variant:ident, $values:expr) => {{
            let mut unwrapped_values = Vec::with_capacity($values.len());
            for v in values {
                if let Value::$variant(v) = v {
                    unwrapped_values.push(v.view());
                } else {
                    bail!("Loop: All iterations must produce outputs of identical types.")
                }
            }
            Ok(Value::$variant(
                ndarray::stack(Axis(0), unwrapped_values.as_slice())?.into(),
            ))
        }};
    }

    match values.get(0).unwrap() {
        Value::F32(_) => match_arm!(F32, values),
        Value::F64(_) => match_arm!(F64, values),
        Value::I16(_) => match_arm!(I16, values),
        Value::I32(_) => match_arm!(I32, values),
        Value::I64(_) => match_arm!(I64, values),
        Value::I8(_) => match_arm!(I8, values),
        Value::U16(_) => match_arm!(U16, values),
        Value::U32(_) => match_arm!(U32, values),
        Value::U64(_) => match_arm!(U64, values),
        Value::U8(_) => match_arm!(U8, values),

        Value::Bool(_) => match_arm!(Bool, values),
        Value::String(_) => match_arm!(String, values),
    }
}

fn should_continue(max_n: Option<i64>, n: &Value, cond: &Value) -> Result<bool> {
    // Check max iteration break condition
    if let Some(max_n) = max_n {
        if let Value::I64(n) = n {
            if *(n.first().unwrap()) < max_n {
                return Ok(true);
            } else {
                return Ok(false);
            }
        } else {
            bail!("Loop: Counter has invalid data type")
        }
    }

    // Check explicit break condition
    if let Value::Bool(ref arr) = cond {
        Ok(!arr.to_owned().into_shape(())?.into_scalar())
    } else {
        bail!("Loop: 'cond' must be of type 'bool'")
    }
}

pub fn setup_loop(ctx: OpContext) -> Result<ExecutionStep> {
    let (state, captured_inputs) = Loop::from_context(&ctx)?;

    let mut out_dispatchers = vec![];
    let mut out_senders = vec![];

    for (_out_name, out_dtype) in &state.body.output_dtypes() {
        let (out_sdr, out_disp) = value_channel(out_dtype);
        out_senders.push(out_sdr);
        out_dispatchers.push(out_disp);
    }

    // Explicit input of the 'Loop' node (i.e. not the subgraph)
    #[allow(clippy::type_complexity)]
    let (max_n_iter, cond, v_init): (
        Option<Receiver<ArcArrayD<i64>>>,
        Option<Receiver<ArcArrayD<bool>>>,
        Vec<TaggedReceiver>,
    ) = ctx.input_channels_with_optional().narrow()?;

    let dtypes = state
        .body
        .output_dtypes()
        .iter()
        .skip(1) // the first output is 'cond' which is not part of the final outputs
        .map(|&(k, dtype)| (k.to_string(), dtype))
        .collect();

    match (max_n_iter, cond) {
        (Some(mtc), Some(cond)) => setup_variadic_hetero_output(
            (mtc, cond, v_init, captured_inputs),
            move |(mtc, cond, v_initial, graph_inputs)| {
                state.run(Some(mtc), Some(cond), v_initial, graph_inputs)
            },
            dtypes,
        ),
        (None, Some(cond)) => setup_variadic_hetero_output(
            (cond, v_init, captured_inputs),
            move |(cond, v_initial, graph_inputs)| {
                state.run(None, Some(cond), v_initial, graph_inputs)
            },
            dtypes,
        ),
        (Some(mtc), None) => setup_variadic_hetero_output(
            (mtc, v_init, captured_inputs),
            move |(mtc, v_initial, graph_inputs)| {
                state.run(Some(mtc), None, v_initial, graph_inputs)
            },
            dtypes,
        ),
        (None, None) => setup_variadic_hetero_output(
            (v_init, captured_inputs),
            move |(v_initial, graph_inputs)| state.run(None, None, v_initial, graph_inputs),
            dtypes,
        ),
    }
}
