from time import perf_counter
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def n_rows():
    return 100_000


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
    import onnxrt as ortrs

    sess_rs = ortrs.Session(str(onnx_file))

    t0 = perf_counter()
    outputs_rs = sess_rs.run(inputs)
    t1 = perf_counter()
    print("Rust: ", t1 - t0)

    assert outputs_rs.keys() == {"variable"}
    outputs_rs, *_ = outputs_rs.values()

    np.testing.assert_equal(
        outputs_rs,
        expected_output,
    )


@pytest.mark.skip
def test_inference_onnxruntime(inputs, expected_output, onnx_file):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_sess = ort.InferenceSession(str(onnx_file), sess_options=sess_options)

    t0 = perf_counter()
    ort_sess.run(None, inputs)
    t1 = perf_counter()
    print("onnxruntime: ", t1 - t0)
