from pathlib import Path
import urllib.request
from time import perf_counter

import numpy as np

import onnxrt as ortrs
import onnxruntime as ort


MOBILENET_v2_7 = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx"


def get_cache_dir():
    cachedir = Path(__file__).parent / ".model_cache"
    cachedir.mkdir(exist_ok=True)
    return cachedir


def get_model_path():
    model_path = (get_cache_dir() / "mobilenetv2-7.onnx")
    if model_path.exists():
        return model_path

    with urllib.request.urlopen(MOBILENET_v2_7) as model_bytes:
        buf = model_bytes.read()
        with open(model_path, "wb") as f:
            f.write(buf)

        return model_path


def create_ort_session(fname):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    return ort.InferenceSession(fname, opts)


def test_foo():
    model = get_model_path()
    inp = {"data": np.ones(shape=(1,3,224,224), dtype=np.float32)}

    sess_cpp = create_ort_session(model)
    sess = ortrs.InferenceSession(str(model))

    t0 = perf_counter()
    out_cpp = sess_cpp.run(None, inp)
    t1 = perf_counter()
    delta_cpp = t1 - t0
    print("CPP: ", delta_cpp)

    t0 = perf_counter()
    out_rs = sess.run(inp)
    t1 = perf_counter()
    delta_rust = t1 - t0
    print("Rust: ", delta_rust)

    print("cpp / rust", delta_cpp / delta_rust)
