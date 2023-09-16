from onnxrt import onnxrt


class NodeInformation:
    name: str

    def __init__(self, name):
        self.name = name


class InferenceSession:
    s: onnxrt.Session

    @classmethod
    def from_file(cls, path):
        ret = cls()
        ret.s = onnxrt.Session(path)
        return ret

    @classmethod
    def from_bytes(cls, bytes):
        ret = cls()
        ret.s = onnxrt.Session.py_from_bytes(bytes)
        return ret

    def __init__(self, model_path = None):
        if model_path is None:
            return
        self.s = onnxrt.Session(model_path)

    def get_inputs(self):
        return [NodeInformation(name) for name in self.s.input_names]

    def get_outputs(self):
        return [NodeInformation(name) for name in self.s.output_names]

    def run(self, inputs):
        casted = {}
        for k, v in inputs.items():
            if v.dtype.kind == "U":
                casted[k] = v.astype(object)
                continue
            casted[k] = v
        return self.s.run(casted)


# Make a compatibility layer for the onnxruntime.InferenceSession
class CompatInferenceSession(InferenceSession):
    def __init__(self, model_bytes):
        self.s = onnxrt.Session.py_from_bytes(model_bytes)

    def run(self, output_names, inputs):
        casted = {}
        for k, v in inputs.items():
            if v.dtype.kind == "U":
                casted[k] = v.astype(object)
                continue
            casted[k] = v
        ret = self.s.run(casted)

        if output_names is None:
            output_names = ret.keys()

        return [ret[name] for name in output_names]