from time import perf_counter

import numpy as np
import onnxruntime as ort
import onnxer as ortrs


def test_inference_simple():
    onnx_file = "/Users/c.bourjau/repos/onnxruntime/onnxruntime/test/testdata/tree_ensemble_as_tensor.onnx"
    sess_rs = ortrs.Session(onnx_file)

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    ort_sess = ort.InferenceSession(onnx_file, sess_options=sess_options)

    n_rows = 100_000
    inputs = {
        "X": np.array(
            [
                [0.0, 1.0, 1.7999999523162842, 3.0],
                [0.0, 1.0, 1.7999999523, 3.0],
                [0.0, 1.0, 1.79999995232, 3.0],
            ]
            * n_rows
        )
    }
    for _ in range(2):
        t0 = perf_counter()
        outputs_rs = sess_rs.run(inputs)
        t1 = perf_counter()
        print("Rust: ", t1 - t0)

        t0 = perf_counter()
        outputs_rs = sess_rs.run(inputs)
        t1 = perf_counter()
        print("Rust: ", t1 - t0)

        assert outputs_rs.keys() == {"variable"}
        outputs_rs, *_ = outputs_rs.values()

        np.testing.assert_equal(
            outputs_rs,
            np.array(
                [[0.7284910678863525], [0.7284910678863525], [0.9134243130683899]]
                * n_rows
            ),
        )

        output_names = [obj.name for obj in ort_sess.get_outputs()]

        t0 = perf_counter()
        ort_sess.run(output_names, inputs)
        t1 = perf_counter()
        print("onnxruntime: ", t1 - t0)

        t0 = perf_counter()
        outputs_ort = ort_sess.run(output_names, inputs)[0]
        t1 = perf_counter()
        print("onnxruntime: ", t1 - t0)

        np.testing.assert_array_equal(outputs_rs, outputs_ort)
