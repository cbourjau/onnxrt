use anyhow::{Context, Result, anyhow, bail};
use onnx_protos::onnx_proto3::{
    AttributeProto, GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto,
};
use protobuf::{Message, MessageField};
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::channels::{TaggedBroadcaster, TaggedReceiver, value_channel};
use crate::data_types::{TensorType, ValueType};
use crate::{Attributes, DataType, Executable, Value};

#[pyclass(subclass)]
pub struct Session {
    /// Computational steps as callable functions
    steps: Vec<Box<dyn Executable + Send + Sync>>,
    /// Explicit inputs of the top-level graph.
    explicit_input_names: Vec<String>,
    /// Mandatory inputs for this session
    pub(crate) inputs: Inputs,
    /// Output channels which are filled after all `steps` where called.
    pub(crate) outputs: Vec<(String, TaggedReceiver)>,
}

/// Collection of input `ValueSender`s and their optional default values.
#[derive(Debug)]
pub struct Inputs(pub HashMap<String, (TaggedBroadcaster, Option<Value>)>);

pub struct OpContext<'a> {
    /// Union of all implicit inputs of a node. Only used if a node
    /// has graphs as attributes.
    // pub implicit_inputs: HashMap<String, EnumReceiver>,
    pub attrs: Attributes,
    pub opset: &'a HashMap<&'static str, SetupFnSignature>,
    pub node: &'a NodeProto,
    pub scope: &'a Scope,
}

/// Signature of "wire" functions executed in the setup phase.
pub type SetupFnSignature =
    Box<dyn Fn(OpContext) -> Result<(Box<dyn Executable + Send + Sync>, Vec<TaggedReceiver>)>>;

type Scope = HashMap<String, TaggedReceiver>;

impl<'a> OpContext<'a> {
    fn new(
        node: &'a NodeProto,
        scope: &'a Scope,
        opset: &'a HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        Ok(Self {
            attrs: Attributes::from_node(node)?,
            node,
            scope,
            opset,
        })
    }

    pub fn input_dtypes(&self) -> Vec<DataType> {
        let mut out = vec![];
        for name in &self.node.input {
            out.push(self.scope.get(name).unwrap().tensor_dtype().unwrap());
        }
        out
    }

    /// Subscribe to the input channels for this node.
    pub fn subscribe_to_inputs(self) -> Vec<TaggedReceiver> {
        let mut channels = vec![];
        for input_name in &self.node.input {
            channels.push(self.subscribe(input_name));
        }
        channels
    }

    /// Subscribe to the input channels for this node specified by `indices`.
    ///
    /// Always prefer using `subscribe_to_inputs` unless you really want to skip some inputs.
    pub fn subscribe_to_inputs_by_index(self, indices: &[usize]) -> Vec<TaggedReceiver> {
        let mut channels = vec![];
        for idx in indices {
            let name = self.node.input.get(*idx).unwrap();
            channels.push(self.subscribe(name));
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
    pub fn input_channels_with_optional(self) -> Vec<Option<TaggedReceiver>> {
        let mut channels = vec![];
        for input_name in &self.node.input {
            if input_name.is_empty() {
                channels.push(None);
            } else {
                channels.push(Some(self.subscribe(input_name)));
            }
        }
        channels
    }

    pub fn subscribe(&self, name: &str) -> TaggedReceiver {
        // TODO: Should be fallible to avoid crash on bad user input
        // data (i.e. model)
        self.scope
            .get(name)
            .unwrap_or_else(|| {
                panic!(
                    "requested input '{}' not in scope. Variables in scope: `{:?}`",
                    name,
                    self.scope.keys()
                )
            })
            .clone()
    }

    pub fn n_outputs(&self) -> usize {
        self.node.output.len()
    }
}

impl Session {
    pub fn from_bytes(
        bytes: &[u8],
        setup_dict: &HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        let model_proto = Message::parse_from_bytes(bytes).unwrap();
        Self::from_model_proto(&model_proto, setup_dict)
    }

    pub fn from_file<P: AsRef<Path>>(
        path: P,
        setup_dict: &HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, setup_dict)
    }

    pub fn from_model_proto(
        model: &ModelProto,
        setup_dict: &HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        let graph = model.graph.as_ref().unwrap();
        Self::from_graph_proto(graph, setup_dict)
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
        setup_dict: &HashMap<&'static str, SetupFnSignature>,
    ) -> Result<Self> {
        let explicit_input_names = graph.input.iter().map(|item| item.name.clone()).collect();

        // External inputs and initializers
        let (inputs, scope) = prepare_inputs(&graph.input)?;
        let (inputs, scope) = prepare_initializer(&graph.initializer, inputs, scope)?;

        let output_names: Vec<_> = graph.output.iter().map(|el| el.name.clone()).collect();

        let (steps, outputs) = setup_steps(scope, &graph.node, &output_names, setup_dict)?;

        Ok(Self {
            steps,
            inputs,
            outputs,
            explicit_input_names,
        })
    }

    pub(crate) fn from_nested_graph(
        graph: &GraphProto,
        setup_dict: &HashMap<&'static str, SetupFnSignature>,
        outer_scope: &Scope,
    ) -> anyhow::Result<Self> {
        let explicit_input_names = graph.input.iter().map(|item| item.name.clone()).collect();

        // External inputs and initializers
        let (inputs, scope) = prepare_inputs(&graph.input)?;
        let (inputs, scope) = prepare_initializer(&graph.initializer, inputs, scope)?;
        let (inputs, scope) = prepare_implicit_inputs(inputs, outer_scope, scope, graph);

        let output_names: Vec<_> = graph.output.iter().map(|el| el.name.clone()).collect();

        let (steps, outputs) = setup_steps(scope, &graph.node, &output_names, setup_dict)?;

        Ok(Self {
            steps,
            inputs,
            outputs,
            explicit_input_names,
        })
    }

    pub fn run_named(
        &self,
        input_values: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Value>> {
        for (name, (value_dispatcher, default)) in &self.inputs.0 {
            let provided_value = input_values.get(name.as_str());

            match (provided_value, default) {
                (None, None) => bail!("Required input '{}' was not provided", name),
                (Some(val), _) => value_dispatcher.send(val.clone()).unwrap(),
                (None, Some(val)) => value_dispatcher.send(val.clone()).unwrap(),
            }
        }

        for step in &self.steps {
            step.execute().unwrap();
        }

        self.outputs
            .iter()
            .map(|(k, output_ch)| output_ch.recv().map(|v| (k.clone(), v)))
            .collect()
    }

    pub fn run_indexed(&self, input_values: &HashMap<String, Value>) -> Result<Vec<Value>> {
        for (name, (value_dispatcher, default)) in &self.inputs.0 {
            let provided_value = input_values.get(name.as_str());

            match (provided_value, default) {
                (None, None) => bail!("Required input '{}' was not provided", name),
                (Some(val), _) => value_dispatcher.send(val.clone()).unwrap(),
                (None, Some(val)) => value_dispatcher.send(val.clone()).unwrap(),
            }
        }

        for step in &self.steps {
            step.execute()?;
        }

        self.outputs
            .iter()
            .map(|(_, output_ch)| output_ch.recv())
            .collect()
    }

    pub fn output_dtypes(&self) -> Vec<(&str, ValueType)> {
        self.outputs
            .iter()
            .map(|(name, output)| {
                let dtype = if let TaggedReceiver::Tensor(t) = output {
                    ValueType::Tensor(TensorType(t.dtype()))
                } else {
                    todo!()
                };
                (name.as_str(), dtype)
            })
            .collect()
    }

    /// Get data types of the required inputs
    pub fn input_dtypes(&self) -> HashMap<&str, ValueType> {
        self.inputs
            .0
            .iter()
            .filter(|(_, (_, default))| default.is_none())
            .map(|(name, (input, _default))| {
                let dtype = input.value_type();
                (name.as_str(), dtype)
            })
            .collect()
    }

    pub(crate) fn get_explicit_input_names(&self) -> Vec<&str> {
        self.explicit_input_names
            .iter()
            .map(|item| item.as_str())
            .collect()
    }

    /// Get names of required input names (including implicit ones).
    /// The order is not deterministic!
    pub(crate) fn input_names(&self) -> Vec<&str> {
        self.inputs
            .0
            .iter()
            .filter(|(_, (_, default))| default.is_none())
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("steps", &format!("[{} steps]", self.steps.len()))
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish()
    }
}

/// Setup execution steps which send inputs and outputs through channels to each other.
///
/// The `outer_scope` gives access to values defined in the outer
/// scope by the time this function is called. It is up to the caller
/// side to ensure that the execution steps for the variables in the
/// outer scope are called.
#[allow(clippy::type_complexity)]
fn setup_steps(
    mut scope: Scope,
    nodes: &[NodeProto],
    output_names: &[String],
    setup_dict: &HashMap<&'static str, SetupFnSignature>,
) -> Result<(
    Vec<Box<dyn Executable + Send + Sync>>,
    Vec<(String, TaggedReceiver)>,
)> {
    let mut steps = vec![];

    for node in nodes {
        let ctx = OpContext::new(node, &scope, setup_dict)?;
        let setup_fn = setup_dict
            .get(&node.op_type.as_str())
            .ok_or_else(|| anyhow!("Unimplemented operator: {}", node.op_type))?;

        let (kernel, outputs) = setup_fn(ctx).context(format!(
            "An error occurred during the setup of a '{}' node.",
            node.op_type
        ))?;

        steps.push(kernel);

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
        .map(|name| (name.to_string(), scope.get(name.as_str()).unwrap().clone()))
        .collect();
    Ok((steps, outputs))
}

/// Create input `ValueDispatcher` and scope hashmap from graph.
fn prepare_inputs(inputs: &[ValueInfoProto]) -> Result<(Inputs, Scope)> {
    let mut input_senders = HashMap::new();
    let mut scope = HashMap::new();
    for input in inputs {
        let input_name = &input.name;
        let data_type = ValueType::from_value_info(input)?;
        let (sdr, disp) = value_channel(&data_type);
        input_senders.insert(input_name.to_string(), (sdr, None));
        scope.insert(input_name.to_string(), disp);
    }
    // Add initializers to scope. Overriding initializers is not supported
    Ok((Inputs(input_senders), scope))
}

/// Process initializers returning dispatchers and their input
/// channels and default values.
///
/// Initializers as default inputs are currently not supported
#[allow(clippy::type_complexity)]
fn prepare_initializer(
    initializers: &[TensorProto],
    inputs: Inputs,
    mut scope: Scope,
) -> anyhow::Result<(Inputs, Scope)> {
    let mut value_dispatchers = inputs.0;
    for initializer in initializers {
        // FIXME: Initializers for non-tensors?
        let ty = DataType::from_i32(initializer.data_type)?;
        let ty = ValueType::Tensor(TensorType(ty));

        let value: Value = initializer.try_into()?;

        // Setup sender/receiver dispatchers
        let (sdr, disp) = value_channel(&ty);

        // Override existing (optional) inputs.
        scope.insert(initializer.name.clone(), disp);
        value_dispatchers.insert(initializer.name.clone(), (sdr, Some(value)));
    }
    Ok((Inputs(value_dispatchers), scope))
}

/// Prepare dispatchers for implicit inputs of a nested graph.
///
/// The `outer_scope` is needed for the types. The implicit inputs are added to `scope`.
fn prepare_implicit_inputs(
    inputs: Inputs,
    outer_scope: &Scope,
    mut scope: Scope,
    graph: &GraphProto,
) -> (Inputs, Scope) {
    let mut dtypes = HashMap::new();
    let mut value_dispatchers = inputs.0;

    let implicit_names = get_implicit_inputs(graph);
    for name in implicit_names.iter() {
        let dtype = outer_scope.get(*name).unwrap().tensor_dtype().unwrap();
        dtypes.insert(name, dtype);
    }

    for (name, dtype) in dtypes.into_iter() {
        // FIXME: implicit non-tensor inputs?
        let dtype = ValueType::Tensor(TensorType(dtype));
        let (sdr, disp) = value_channel(&dtype);
        value_dispatchers.insert(name.to_string(), (sdr, None));
        scope.insert(name.to_string(), disp);
    }

    (Inputs(value_dispatchers), scope)
}

/// Find all inputs which are not defined in the explicit inputs of
/// the graph or defined by other nodes in scope. Recurses into subgraphs.
pub fn get_implicit_inputs(g: &GraphProto) -> HashSet<&str> {
    let mut explicit = HashSet::<_>::from_iter(g.input.iter().map(|v_info| v_info.name.clone()));
    let mut implicit = HashSet::new();
    for node in &g.node {
        for input in &node.input {
            if !explicit.contains(input) {
                implicit.insert(input.as_str());
            }
        }
        for attr_proto in &node.attribute {
            if let AttributeProto {
                g: MessageField(Some(g)),
                ..
            } = attr_proto
            {
                implicit.extend(get_implicit_inputs(g).iter());
            }
        }
        for output in &node.output {
            explicit.insert(output.clone());
        }
    }
    implicit
}
