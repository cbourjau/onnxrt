from spox import Tensor, build, argument
import spox.opset.ai.onnx.v18 as op

import numpy as np
import pytest
import onnxrt as ortrs


def _as_file(model, fn):
    with open(fn, "wb") as f:
        f.write(model.SerializeToString())

    return str(fn)


@pytest.fixture
def trip_count_model(tmp_path):
    a, b = (argument(Tensor(np.float64, (None,))), argument(Tensor(np.int64, ("N",))))
    max_count = op.const(2)
    x, y, zs = op.loop(
        M=max_count,
        v_initial=[a, b],
        body=lambda i, c, a, b: [op.const(False), a, op.add(i, b), i],
    )

    model = build({"a": a, "b": b}, {"x": x, "y": y, "zs": zs})
    fn = tmp_path / "test.onnx"

    return _as_file(model, fn)


@pytest.fixture
def break_cond_model(tmp_path):
    a, b = (argument(Tensor(np.float64, (None,))), argument(Tensor(np.int64, ("N",))))
    max_count = op.const(2)
    x, y, zs = op.loop(
        M=max_count,
        v_initial=[a, b],
        body=lambda i, c, a, b: [
            op.greater_or_equal(i, op.const(2)),
            a,
            op.add(i, b),
            i,
        ],
    )

    model = build({"a": a, "b": b}, {"x": x, "y": y, "zs": zs})

    fn = tmp_path / "test.onnx"
    return _as_file(model, fn)


def test_loop_terminate_trip_count(trip_count_model):
    """Break by reaching the maximum trip count."""
    session = ortrs.InferenceSession(trip_count_model)

    inputs = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4, 5, 6])}
    (out_a, out_sum, out_idx) = session.run(inputs).values()

    np.testing.assert_array_equal([1.0, 2.0, 3.0], out_a)
    np.testing.assert_array_equal([5, 6, 7], out_sum)
    np.testing.assert_array_equal([[0], [1]], out_idx)


def test_loop_terminate_break_condition(break_cond_model):
    """Break by having the 'body' return a truthy break condition."""
    session = ortrs.InferenceSession(break_cond_model)

    inputs = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4, 5, 6])}
    (out_a, out_sum, out_idx) = session.run(inputs).values()

    np.testing.assert_array_equal([1.0, 2.0, 3.0], out_a)
    np.testing.assert_array_equal([5, 6, 7], out_sum)
    np.testing.assert_array_equal([[0], [1]], out_idx)
