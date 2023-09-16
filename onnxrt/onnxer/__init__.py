from onnxrt import onnxrt


class NodeInformation:
    name: str

    def __init__(self, name):
        self.name = name


class Session(onnxrt.Session):
    def get_inputs(self):
        return [NodeInformation(name) for name in super().input_names]

    def get_outputs(self):
        return [NodeInformation(name) for name in super().output_names]

    def run(self, output_names, inputs):
        casted = {}
        for k, v in inputs.items():
            if v.dtype.kind == "U":
                casted[k] = v.astype(object)
                continue
            casted[k] = v
        ret = super().run(casted)

        if output_names is None:
            output_names = ret.keys()

        return [ret[name] for name in output_names]