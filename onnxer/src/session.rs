use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use ndarray::Array;
use onnx_protos::onnx_proto3::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};
use protobuf::Message;
use pyo3::prelude::*;

use crate::{value_channel, Attributes, DataType, Dispatcher, EnumReceiver, Value, ValueSender};

#[pyclass]
pub struct Session {
    /// Computational steps as callable functions
    steps: Vec<ExecFnSignature>,
    /// Mandatory inputs for this session
    inputs: HashMap<String, ValueSender>,
    /// Implicit inputs (nested graphs only)
    implicit_inputs: HashMap<String, ValueSender>,
    /// Constants (overriding is not supported)
    initializers: HashMap<String, (ValueSender, Value)>,
    /// Output channels which are filled after all `steps` where called.
    pub(crate) outputs: Vec<(String, EnumReceiver)>,
}

pub struct ExecutionStep {
    pub fun: Box<dyn Fn() -> Result<()> + Send>,
    pub outputs: Vec<Dispatcher>,
}

/// Signature of functions executed at runtime.
pub type ExecFnSignature = Box<dyn Fn() -> Result<()> + Send>;

/// Signature of "wire" functions executed in the setup phase.
pub type SetupFnSignature = Box<dyn Fn(OpContext) -> Result<ExecutionStep> + Sync>;

pub struct OpContext<'a> {
    /// Union of all implicit inputs of a node. Only used if a node
    /// has graphs as attributes.
    // pub implicit_inputs: HashMap<String, EnumReceiver>,
    pub attrs: Attributes,
    pub opset: &'a HashMap<&'static str, SetupFnSignature>,
    pub node: &'a NodeProto,
    pub scope: &'a HashMap<String, Dispatcher>,
}

impl<'a> OpContext<'a> {
    fn new(
        node: &'a NodeProto,
        scope: &'a HashMap<String, Dispatcher>,
        opset: &'a HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        Ok(Self {
            attrs: Attributes::from_node(node)?,
            node,
            scope,
            opset,
        })
    }

    /// Subscribe to the input channels for this node
    pub fn input_channels(&self) -> Vec<EnumReceiver> {
        let mut channels = vec![];
        for input_name in &self.node.input {
            channels.push(self.subscribe(input_name));
        }
        channels
    }

    /// Get inputs of operators which allow for "optional" inputs.
    ///
    /// Some operators (e.g. 'Clip') may have inputs named '""' which
    /// don't have to be defined in the current scope.
    ///
    /// Note: If the last input is optional and missing it may be
    /// absent completely in the input list of a node!
    pub fn input_channels_with_optional(&self) -> Vec<Option<EnumReceiver>> {
        let mut channels = vec![];
        for input_name in &self.node.input {
            if input_name == "" {
                channels.push(None);
            } else {
                channels.push(Some(self.subscribe(input_name)));
            }
        }
        channels
    }

    pub fn subscribe(&self, name: &str) -> EnumReceiver {
        // TODO: Should be fallible to avoid crash on bad user input
        // data (i.e. model)
        self.scope
            .get(name)
            .unwrap_or_else(|| {
                panic!(
                    "Requested input '{}' not in scope. Variables in scope: {:?}",
                    name,
                    self.scope.keys()
                )
            })
            .subscribe()
    }

    pub fn n_outputs(&self) -> usize {
        self.node.output.len()
    }
}

impl Session {
    pub fn from_file<P: AsRef<Path>>(
        path: P,
        wire_dict: &HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        let bytes = std::fs::read(path).unwrap();
        let model_proto: ModelProto = Message::parse_from_bytes(&bytes[..]).unwrap();
        Self::from_model_proto(&model_proto, wire_dict)
    }

    pub fn from_model_proto(
        model: &ModelProto,
        wire_dict: &HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        let graph = model.graph.as_ref().unwrap();
        Self::from_graph_proto(graph, wire_dict)
    }

    /// Create a session from a graph proto
    ///
    /// If the `GraphProto` is nested in an attribute, it may have
    /// "implicit inputs" (i.e. inputs not enumerated in the
    /// `GraphProto`'s `ValueInfo` section. If implicit inputs are
    /// provided to this function, they will become mandatory inputs
    /// of the session.
    pub fn from_graph_proto(
        graph: &GraphProto,
        wire_dict: &HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        // External inputs and initializers
        let (input_senders, scope) = prepare_inputs(&graph.input)?;
        let (initializers, scope) = prepare_initializer(&graph.initializer, scope)?;

        let output_names: Vec<_> = graph.output.iter().map(|el| el.name.clone()).collect();

        let (steps, outputs) = setup_steps(scope, &graph.node, &output_names, wire_dict)?;

        Ok(Self {
            steps,
            inputs: input_senders,
            implicit_inputs: Default::default(),
            outputs,
            initializers,
        })
    }

    pub(crate) fn from_nested_graph(
        graph: &GraphProto,
        wire_dict: &HashMap<&'static str, SetupFnSignature>,
        outer_scope: &HashMap<String, Dispatcher>,
    ) -> anyhow::Result<Self> {
        // External inputs and initializers
        let (input_senders, scope) = prepare_inputs(&graph.input)?;
        let (initializers, scope) = prepare_initializer(&graph.initializer, scope)?;
        let (implicit_input_senders, scope) = prepare_implicit_inputs(outer_scope, scope, graph);

        let output_names: Vec<_> = graph.output.iter().map(|el| el.name.clone()).collect();

        let (steps, outputs) = setup_steps(scope, &graph.node, &output_names, wire_dict)?;

        Ok(Self {
            steps,
            inputs: input_senders,
            implicit_inputs: implicit_input_senders,
            outputs,
            initializers,
        })
    }

    pub fn run(&self, input_values: &HashMap<String, Value>) -> Result<HashMap<String, Value>> {
        if input_values.len() != (self.inputs.len() + self.implicit_inputs.len()) {
            bail!("Unexpected number of inputs.")
        }

        for (name, input_dispatcher) in &self.inputs {
            let value = input_values
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("Input `{}` missing", name))?;
            input_dispatcher.send(value.clone())?;
        }

        for (name, input_dispatcher) in &self.implicit_inputs {
            let value = input_values
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("Input `{}` missing", name))?;
            input_dispatcher.send(value.clone())?;
        }

        for (_name, (value_dispatcher, value)) in self.initializers.iter() {
            value_dispatcher.send(value.clone()).unwrap();
        }

        for step in &self.steps {
            step().unwrap();
        }

        self.outputs
            .iter()
            .map(|(k, output_ch)| output_ch.recv().map(|v| (k.clone(), v)))
            .collect()
    }

    pub fn output_dtypes(&self) -> Vec<(&str, DataType)> {
        self.outputs
            .iter()
            .map(|(name, output)| {
                let dtype = output.dtype();
                (name.as_str(), dtype)
            })
            .collect()
    }

    pub fn input_dtypes(&self) -> HashMap<&str, DataType> {
        self.inputs
            .iter()
            .map(|(name, input)| {
                let dtype = input.dtype();
                (name.as_str(), dtype)
            })
            .collect()
    }

    /// Input names (including implict ones).
    pub(crate) fn input_names(&self) -> Vec<&str> {
        self.inputs
            .keys()
            .chain(self.implicit_inputs.keys())
            .map(|s| s.as_str())
            .collect()
    }
}

/// Setup execution steps which send inputs and outputs through channels to each other.
///
/// The `outer_scope` gives access to values defined in the outer
/// scope by the time this function is called. It is up to the caller
/// side to ensure that the execution steps for the variables in the
/// outer scope are called.
fn setup_steps<'a>(
    mut scope: HashMap<String, Dispatcher>,
    nodes: &'a [NodeProto],
    output_names: &[String],
    wire_dict: &HashMap<&'static str, SetupFnSignature>,
) -> Result<(Vec<ExecFnSignature>, Vec<(String, EnumReceiver)>)> {
    let mut steps = vec![];

    for node in nodes {
        let ctx = OpContext::new(node, &scope, wire_dict)?;
        let wire_fn = wire_dict
            .get(&node.op_type.as_str())
            .ok_or_else(|| anyhow!("Unimplemented operator: {}", node.op_type))?;

        let ExecutionStep { fun: op, outputs } = wire_fn(ctx).map_err(|e| {
            anyhow!(
                "During setup of '{}' the following error occurred: {:?}",
                node.op_type,
                e
            )
        })?;

        steps.push(op);

        if node.output.len() != outputs.len() {
            bail!(
                "Node `{}` produced an unexpected number of outputs. Expected {} but found {}",
                node.name,
                node.output.len(),
                outputs.len()
            );
        }

        for (output_name, dispatcher) in node.output.iter().zip(outputs.into_iter()) {
            scope.insert(output_name.clone(), dispatcher);
        }
    }

    // Subscribe to the outputs we would like to keep. We return those
    // output channels
    let outputs = output_names
        .iter()
        .map(|name| {
            (
                name.to_string(),
                scope.get(name.as_str()).unwrap().subscribe(),
            )
        })
        .collect();
    Ok((steps, outputs))
}

/// Create input `ValueDispatcher` and scope hashmap from graph.
fn prepare_inputs(
    inputs: &[ValueInfoProto],
) -> Result<(HashMap<String, ValueSender>, HashMap<String, Dispatcher>)> {
    let mut input_senders = HashMap::new();
    let mut scope = HashMap::new();
    for input in inputs {
        let input_name = &input.name;
        let data_type = DataType::from_value_info(input)?;
        let (sdr, disp) = value_channel(&data_type);
        input_senders.insert(input_name.to_string(), sdr);
        scope.insert(input_name.to_string(), disp);
    }
    // Add initializers to scope. Overriding initializers is not supported
    Ok((input_senders, scope))
}

/// Process initializers returning dispatchers and their input
/// channels and default values.
fn prepare_initializer(
    initializers: &[TensorProto],
    mut scope: HashMap<String, Dispatcher>,
) -> anyhow::Result<(
    HashMap<String, (ValueSender, Value)>,
    HashMap<String, Dispatcher>,
)> {
    let mut value_dispatchers = HashMap::new();
    for initializer in initializers {
        if scope.contains_key(&initializer.name) {
            bail!(
                "Overriding initializers (`{}`) is not supported.",
                initializer.name
            );
        }

        let shape: Vec<_> = initializer.dims.iter().map(|el| *el as usize).collect();
        let ty = DataType::from_i32(initializer.data_type)?;

        let value = match ty {
            DataType::F32 => {
                let arr = Array::from_vec(initializer.float_data.clone())
                    .into_shape(shape)?
                    .into_dyn()
                    .into();
                Value::F32(arr)
            }
            DataType::F64 => {
                let arr = Array::from_vec(initializer.double_data.clone())
                    .into_shape(shape)?
                    .into_dyn()
                    .into();
                Value::F64(arr)
            }
            DataType::I64 => {
                let arr = Array::from_vec(initializer.int64_data.clone())
                    .into_shape(shape)?
                    .into_dyn()
                    .into();
                Value::I64(arr)
            }
            ty => unimplemented!("Unimplemented initializer type: {:?}", ty),
        };

        // Setup sender/receiver dispatchers
        let (sdr, disp) = value_channel(&ty);
        scope.insert(initializer.name.clone(), disp);
        value_dispatchers.insert(initializer.name.clone(), (sdr, value));
    }
    Ok((value_dispatchers, scope))
}

/// Prepare dispatchers for implicit inputs of a nested graph.
///
/// The `outer_scope` is needed for the types. The implicit inputs are added to `scope`.
fn prepare_implicit_inputs(
    outer_scope: &HashMap<String, Dispatcher>,
    mut scope: HashMap<String, Dispatcher>,
    _graph: &GraphProto,
) -> (HashMap<String, ValueSender>, HashMap<String, Dispatcher>) {
    let mut dtypes = HashMap::new();

    for (k, v) in outer_scope.iter() {
        dtypes.insert(k.clone(), v.dtype());
    }

    let mut val_sdrs = HashMap::new();
    for (name, dtype) in dtypes.into_iter() {
        let (sdr, disp) = value_channel(&dtype);
        val_sdrs.insert(name.to_string(), sdr);
        scope.insert(name.to_string(), disp);
    }

    (val_sdrs, scope)
}

// /// Find all inputs which are not defined in the explicit inputs of
// /// the graph or defined by other nodes in scope. Recurses into subgraphs.
// pub(crate) fn get_implicit_inputs(g: &GraphProto) -> Result<HashSet<&str>> {
//     let mut explicit = HashSet::<_>::from_iter(g.input.iter().map(|v_info| v_info.name.clone()));
//     let mut implicit = HashSet::new();
//     for node in &g.node {
//         for input in &node.input {
//             if !explicit.contains(input) {
//                 implicit.insert(input.as_str());
//             }
//         }
//         for attr_proto in &node.attribute {
//             if let AttributeProto {
//                 g: MessageField(Some(g)),
//                 ..
//             } = attr_proto
//             {
//                 implicit.extend(get_implicit_inputs(&g)?.iter());
//             }
//         }
//         for output in &node.output {
//             explicit.insert(output.clone());
//         }
//     }
//     Ok(implicit)
// }
