from spox import Tensor, build, argument
import spox.opset.ai.onnx.v18 as op
from time import perf_counter

import numpy as np
import pytest
import onnxrt as ortrs
import onnxruntime as ort


def predict_and_compare(fn, kwargs):
    print()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    sess_cpp = ort.InferenceSession(fn, opts)
    t0 = perf_counter()
    out_cpp = sess_cpp.run(None, kwargs)
    t1 = perf_counter()
    delta_cpp = t1 - t0

    print("CPP: ", delta_cpp)

    out_cpp = {outname.name: v for outname, v in zip(sess_cpp.get_outputs(), out_cpp)}
    print({k: arr.shape for k, arr in out_cpp.items()})

    sess_rs = ortrs.InferenceSession(fn)

    t0 = perf_counter()
    out_rs = sess_rs.run(kwargs)
    t1 = perf_counter()
    delta_rust = t1 - t0

    print("Rust: ", delta_rust)

    print("cpp / rust", delta_cpp / delta_rust)

    for k, v in out_rs.items():
        print(k)
        print(v.shape, v)
        print(out_cpp[k].shape, out_cpp[k])
        np.testing.assert_array_almost_equal(v, out_cpp[k])


def make_model(tmp_path, x_np, w_np, b_np, auto_pad, pads, strides):
    n_groups = int(x_np.shape[1] / w_np.shape[1])
    # Create Spox inputs from numpy arrays
    x = argument(Tensor(np.float32, x_np.shape))
    w = argument(Tensor(np.float32, w_np.shape))
    b = argument(Tensor(np.float32, b_np.shape))
    y = op.conv(x, w, b, auto_pad=auto_pad, pads=pads, group=n_groups, strides=strides)

    model = build({"x": x, "w": w, "b": b}, {"y": y})

    fn = tmp_path / "test.onnx"
    with open(fn, "wb") as f:
        f.write(model.SerializeToString())

    return str(fn)


@pytest.mark.skip
def test_error_msg():
    # TODO: File upstream issue about bad error
    a = argument(Tensor(np.float32, ("N",)))
    b = argument(Tensor(np.float32, ("N",)))

    out = op.add(a, b)

    build({"a": a}, {"out": out})


@pytest.mark.parametrize(
    "auto_pad,pads",
    [
        ("VALID", None),
        ("SAME_LOWER", None),
        ("SAME_UPPER", None),
        ("NOTSET", (0, 0, 0, 0)),
        ("NOTSET", (1, 1, 1, 1)),
    ],
)
@pytest.mark.parametrize("len_x, len_w", [(1, 1), (2, 1), (2, 2)])  # x >= 2
@pytest.mark.parametrize("strides", [1, 2])
@pytest.mark.parametrize(
    "in_channels, out_channels, n_groups",
    [
        (1, 1, 1),
        (1, 2, 1),
        (2, 1, 1),
        (2, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
        (1, 1, 1),
        (1, 1, 1),
        (4, 2, 2),
    ],
)
def test_conv_with_groups(
    tmp_path, len_x, len_w, in_channels, out_channels, n_groups, auto_pad, strides, pads
):
    # Unclear at the moment how the interplay between strides and
    # padding should be...
    if auto_pad not in ["VALID", "NOTSET"] and strides != 1:
        pytest.skip()

    BATCH = 1
    NX = BATCH * in_channels * len_x * len_x
    ch_per_gr = in_channels // n_groups
    NW = out_channels * ch_per_gr * len_w * len_w
    x = np.arange(0, NX, dtype=np.float32).reshape(BATCH, in_channels, len_x, len_x)
    w = np.arange(0, NW, dtype=np.float32).reshape(
        out_channels, ch_per_gr, len_w, len_w
    )
    b = np.zeros((out_channels,), dtype=np.float32)

    model = make_model(tmp_path, x, w, b, auto_pad, pads, [strides] * 2)

    predict_and_compare(model, {"x": x, "w": w, "b": b})
