use std::collections::{HashMap, HashSet};

use onnx_protos::onnx_proto3::GraphProto;

use crate::{Session, channels::value_channel, opset::utils::*};

#[derive(Debug)]
struct If {
    then: Session,
    else_: Session,
}
impl If {
    fn from_context(ctx: &OpContext) -> Result<Self> {
        let make_attr_session = |attr_name| {
            let g: GraphProto = ctx.attrs.get(attr_name)?.try_into()?;
            Session::from_nested_graph(&g, ctx.opset, ctx.scope)
        };
        let then = make_attr_session("then_branch")?;
        let else_ = make_attr_session("else_branch")?;

        Ok(Self { then, else_ })
    }
}

pub fn setup_if(ctx: OpContext) -> Result<ExecutionStep> {
    let state = If::from_context(&ctx)?;

    // Get input channels for implicit inputs. We always pull all channels!
    let cumulated_input_names = HashSet::<&str>::from_iter(
        state
            .then
            .input_names()
            .into_iter()
            .chain(state.else_.input_names().into_iter()),
    );

    let graph_input_channels: HashMap<_, _> = cumulated_input_names
        .iter()
        .map(|name| (name.to_string(), ctx.subscribe(name)))
        .collect();

    // Create output channels by "selecting" the outputs of either of
    // the two branches.
    assert_eq!(state.then.outputs.len(), state.else_.outputs.len());
    assert!(
        state
            .then
            .output_dtypes()
            .iter()
            .zip(&state.else_.output_dtypes())
            .all(|((_, dtype_left), (_, dtype_right))| dtype_left == dtype_right)
    );

    let mut out_dispatchers = vec![];
    let mut out_senders = vec![];
    // Only using 'then' output types since they should be identical
    for (_out_name, out_dtype) in &state.then.output_dtypes() {
        let (out_sdr, out_disp) = value_channel(out_dtype);
        out_senders.push(out_sdr);
        out_dispatchers.push(out_disp);
    }

    // Explicit node input
    let cond_ch: Receiver<ArcArrayD<bool>> = ctx.subscribe_to_inputs().narrow()?;

    let fun = move || {
        let cond = {
            let conds = cond_ch.recv().unwrap();
            if conds.len() != 1 {
                return Err(anyhow!("`Cond` input of `If` must be a scalar."));
            }
            conds.to_owned().into_shape([])?.into_scalar()
        };

        let graph_inputs = graph_input_channels
            .iter()
            .map(|(k, ch)| (k.clone(), ch.recv().unwrap()))
            .collect();
        // If-graphs do not have explicit inputs; only the implicit ones
        if cond {
            // The outputs are defined by order, not name
            let mut result = state.then.run_named(&graph_inputs)?;
            for ((name, _), sender) in state.then.outputs.iter().zip(&out_senders) {
                sender.send(result.remove(name).unwrap()).unwrap();
            }
        } else {
            // The outputs are defined by order, not name
            let mut result = state.else_.run_named(&graph_inputs)?;
            for ((name, _), sender) in state.else_.outputs.iter().zip(&out_senders) {
                sender.send(result.remove(name).unwrap()).unwrap();
            }
        };
        Ok(())
    };
    Ok(ExecutionStep {
        fun: Box::new(fun),
        outputs: out_dispatchers,
    })
}
