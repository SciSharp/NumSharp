"""
Generate the tiny ONNX models the NumSharp.Interop.OnnxRuntime test project runs.

The models are the ORT half of the interop gate: NumSharp.Tests.Interop.OnnxRuntime feeds NDArrays
through the package into a REAL ONNX Runtime session and reads the outputs back, so every model here
is a real .onnx file ONNX Runtime executes -- not a mock. They are deliberately minimal (one or two
nodes, ~100-400 bytes each) and are COMMITTED, so the C# suite needs no Python at test time (the
same offline-corpus philosophy as test/oracle/gen_oracle.py). Re-run this script only when a model
changes; it self-checks every file by executing it through onnxruntime.

    python test/oracle/gen_onnx_models.py            # writes test/NumSharp.Tests.Interop.OnnxRuntime/Models/*.onnx

Requires: onnx, onnxruntime, numpy (pip install onnx onnxruntime numpy).

Models
------
identity_<dtype>.onnx   X -> Identity -> Y, rank-agnostic (no shape declared), one per ORT tensor
                        element type NumSharp maps: float32 float64 int8 uint8 int16 uint16 int32
                        uint32 int64 uint64 bool float16. Exercises every dtype cell of the round
                        trip, 0-d / empty / N-d included.
add_f32.onnx            A, B (float32, rank-agnostic) -> Add -> C. Two inputs + ORT broadcasting.
two_outputs_f32.onnx    X -> Identity -> Y1 ; X -> Neg -> Y2. A dictionary Run with two outputs.
fixed_shape_f32.onnx    data [1,3,4,4] float32 -> Relu -> relu. Fixed dims for the Tier-2 shape
                        validation (rank + extent mismatch errors).
int64_input.onnx        ids [1,'seq'] int64 -> Identity -> out. The BERT case: Tier-2 auto-coercion
                        of an int32 NDArray into an int64 input.
matmul_f32.onnx         A ['M','K'], B ['K','N'] float32 -> MatMul -> C. Real arithmetic to compare
                        against np.matmul.
postprocess_f32.onnx    X ['N','C'] float32 -> Softmax(axis=-1) -> probs ; ArgMax(axis=-1,
                        keepdims=0) -> argmax (int64). The ORACLE for Postprocess.Softmax/Argmax.
topk_f32.onnx           X ['N','C'] float32, K [1] int64 -> TopK(axis=-1) -> Values, Indices.
                        The oracle for Postprocess.TopK.
sequence_out.onnx       X float32 -> SequenceConstruct -> seq (a sequence<tensor(float)>). Proves
                        the NDArray verbs decline a non-tensor OrtValue with a clear message.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "NumSharp.Tests.Interop.OnnxRuntime", "Models")

# opset 17 / IR 8: old enough for every ORT >= 1.16 (the package floor), new enough for every op used.
OPSET = 17
IR_VERSION = 8

DTYPES = {
    "float32": TensorProto.FLOAT,
    "float64": TensorProto.DOUBLE,
    "int8": TensorProto.INT8,
    "uint8": TensorProto.UINT8,
    "int16": TensorProto.INT16,
    "uint16": TensorProto.UINT16,
    "int32": TensorProto.INT32,
    "uint32": TensorProto.UINT32,
    "int64": TensorProto.INT64,
    "uint64": TensorProto.UINT64,
    "bool": TensorProto.BOOL,
    "float16": TensorProto.FLOAT16,
}


def _model(name, nodes, inputs, outputs, value_info=(), check=True):
    graph = helper.make_graph(nodes, name, inputs, outputs, value_info=list(value_info))
    model = helper.make_model(graph, producer_name="NumSharp.Interop.OnnxRuntime tests",
                              opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    # onnx.checker insists on a `shape` field for graph inputs/outputs, so the RANK-AGNOSTIC models
    # (tensor type with no shape at all -- legal ONNX, "unknown rank", and accepted by ORT) skip the
    # static checker; self_check() executes every model through onnxruntime anyway.
    if check:
        onnx.checker.check_model(model)
    return model


def _tensor(name, elem_type, shape=None):
    # shape=None -> rank-agnostic tensor type (accepts 0-d, empty and any rank); a list with strings
    # declares symbolic dims.
    return helper.make_tensor_value_info(name, elem_type, shape)


def build_all():
    models = {}

    for dt_name, dt in DTYPES.items():
        models[f"identity_{dt_name}"] = _model(
            f"identity_{dt_name}",
            [helper.make_node("Identity", ["X"], ["Y"])],
            [_tensor("X", dt)], [_tensor("Y", dt)], check=False)

    models["add_f32"] = _model(
        "add_f32",
        [helper.make_node("Add", ["A", "B"], ["C"])],
        [_tensor("A", TensorProto.FLOAT), _tensor("B", TensorProto.FLOAT)],
        [_tensor("C", TensorProto.FLOAT)], check=False)

    models["two_outputs_f32"] = _model(
        "two_outputs_f32",
        [helper.make_node("Identity", ["X"], ["Y1"]), helper.make_node("Neg", ["X"], ["Y2"])],
        [_tensor("X", TensorProto.FLOAT)],
        [_tensor("Y1", TensorProto.FLOAT), _tensor("Y2", TensorProto.FLOAT)], check=False)

    models["fixed_shape_f32"] = _model(
        "fixed_shape_f32",
        [helper.make_node("Relu", ["data"], ["relu"])],
        [_tensor("data", TensorProto.FLOAT, [1, 3, 4, 4])],
        [_tensor("relu", TensorProto.FLOAT, [1, 3, 4, 4])])

    models["int64_input"] = _model(
        "int64_input",
        [helper.make_node("Identity", ["ids"], ["out"])],
        [_tensor("ids", TensorProto.INT64, [1, "seq"])],
        [_tensor("out", TensorProto.INT64, [1, "seq"])])

    models["matmul_f32"] = _model(
        "matmul_f32",
        [helper.make_node("MatMul", ["A", "B"], ["C"])],
        [_tensor("A", TensorProto.FLOAT, ["M", "K"]), _tensor("B", TensorProto.FLOAT, ["K", "N"])],
        [_tensor("C", TensorProto.FLOAT, ["M", "N"])])

    models["postprocess_f32"] = _model(
        "postprocess_f32",
        [helper.make_node("Softmax", ["X"], ["probs"], axis=-1),
         helper.make_node("ArgMax", ["X"], ["argmax"], axis=-1, keepdims=0)],
        [_tensor("X", TensorProto.FLOAT, ["N", "C"])],
        [_tensor("probs", TensorProto.FLOAT, ["N", "C"]), _tensor("argmax", TensorProto.INT64, ["N"])])

    models["topk_f32"] = _model(
        "topk_f32",
        [helper.make_node("TopK", ["X", "K"], ["Values", "Indices"], axis=-1, largest=1, sorted=1)],
        [_tensor("X", TensorProto.FLOAT, ["N", "C"]), _tensor("K", TensorProto.INT64, [1])],
        [_tensor("Values", TensorProto.FLOAT, ["N", "k"]), _tensor("Indices", TensorProto.INT64, ["N", "k"])])

    seq_out = helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, None)
    models["sequence_out"] = _model(
        "sequence_out",
        [helper.make_node("SequenceConstruct", ["X"], ["seq"])],
        [_tensor("X", TensorProto.FLOAT)], [seq_out], check=False)

    return models


def self_check(path: str, name: str) -> None:
    """Execute the freshly written model through onnxruntime with a representative feed."""
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    if name.startswith("identity_"):
        dt_name = name[len("identity_"):]
        np_dt = {"float16": np.float16, "bool": np.bool_}.get(dt_name, getattr(np, dt_name, None))
        x = np.arange(6).reshape(2, 3).astype(np_dt)
        y, = sess.run(None, {"X": x})
        assert y.dtype == x.dtype and y.shape == x.shape and np.array_equal(y, x), name
        y0, = sess.run(None, {"X": np.asarray(3).astype(np_dt)})   # 0-d
        assert y0.shape == () and y0.dtype == x.dtype, name
        ye, = sess.run(None, {"X": np.zeros((0, 4), dtype=np_dt)})  # empty
        assert ye.shape == (0, 4), name
    elif name == "add_f32":
        c, = sess.run(None, {"A": np.ones((2, 3), np.float32), "B": np.arange(3, dtype=np.float32)})
        assert np.array_equal(c, np.ones((2, 3), np.float32) + np.arange(3, dtype=np.float32)), name
    elif name == "two_outputs_f32":
        y1, y2 = sess.run(None, {"X": np.arange(4, dtype=np.float32)})
        assert np.array_equal(y2, -y1), name
    elif name == "fixed_shape_f32":
        r, = sess.run(None, {"data": np.full((1, 3, 4, 4), -1, np.float32)})
        assert r.shape == (1, 3, 4, 4) and not r.any(), name
    elif name == "int64_input":
        o, = sess.run(None, {"ids": np.arange(5, dtype=np.int64).reshape(1, 5)})
        assert o.dtype == np.int64 and o.shape == (1, 5), name
    elif name == "matmul_f32":
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        b = np.arange(12, dtype=np.float32).reshape(3, 4)
        c, = sess.run(None, {"A": a, "B": b})
        assert np.allclose(c, a @ b), name
    elif name == "postprocess_f32":
        x = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float32)
        p, am = sess.run(None, {"X": x})
        assert np.allclose(p.sum(axis=-1), 1) and np.array_equal(am, x.argmax(axis=-1)), name
    elif name == "topk_f32":
        x = np.random.default_rng(0).standard_normal((3, 5)).astype(np.float32)
        v, i = sess.run(None, {"X": x, "K": np.array([2], np.int64)})
        assert v.shape == (3, 2) and i.dtype == np.int64, name
    elif name == "sequence_out":
        seq, = sess.run(None, {"X": np.arange(3, dtype=np.float32)})
        assert isinstance(seq, list) and len(seq) == 1, name
    else:
        raise AssertionError(f"no self-check for {name}")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    models = build_all()
    for name, model in models.items():
        path = os.path.join(OUT_DIR, f"{name}.onnx")
        onnx.save(model, path)
        self_check(path, name)
        print(f"{name}.onnx  {os.path.getsize(path)} bytes  ok")
    print(f"{len(models)} models -> {os.path.relpath(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
