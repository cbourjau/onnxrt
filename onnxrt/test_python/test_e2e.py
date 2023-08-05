from time import perf_counter
from pathlib import Path

import numpy as np
import pytest

import onnxrt as ortrs
import onnxruntime as ort


_N_LOOPS = 1000


@pytest.fixture(params=np.unique(np.logspace(0, 5, 50).astype(int)))
def n_rows(request):
    return request.param


@pytest.fixture
def inputs(n_rows):
    return {
        "X": np.array(
            [
                [0.0, 1.0, 1.7999999523162842, 3.0],
                [0.0, 1.0, 1.7999999523, 3.0],
                [0.0, 1.0, 1.79999995232, 3.0],
            ]
            * n_rows
        )
    }


@pytest.fixture
def expected_output(n_rows):
    return np.array(
        [[0.7284910678863525], [0.7284910678863525], [0.9134243130683899]] * n_rows
    )


@pytest.fixture
def onnx_file():
    root = Path(__file__).parent.parent.parent
    return root / "onnxruntime/onnxruntime/test/testdata/tree_ensemble_as_tensor.onnx"


def test_inference_onnxrs(inputs, expected_output, onnx_file):
    sess_rs = ortrs.Session(str(onnx_file))

    t0 = perf_counter()
    for _ in range(_N_LOOPS):
        outputs_rs = sess_rs.run(inputs)

    t1 = perf_counter()
    print("Rust: ", t1 - t0)

    assert outputs_rs.keys() == {"variable"}
    outputs_rs, *_ = outputs_rs.values()

    np.testing.assert_equal(
        outputs_rs,
        expected_output,
    )


def test_inference_onnxruntime(inputs, expected_output, onnx_file):
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_sess = ort.InferenceSession(str(onnx_file), sess_options=sess_options)

    t0 = perf_counter()
    for _ in range(_N_LOOPS):
        ort_sess.run(None, inputs)
    t1 = perf_counter()
    print("onnxruntime: ", t1 - t0)

