from onnxer import onnxer


class Session(onnxer.Session):
    def run(self, inputs):
        casted = {}
        for k, v in inputs.items():
            if v.dtype.kind == "U":
                casted[k] = v.astype(object)
                continue
            casted[k] = v
        return super().run(casted)
