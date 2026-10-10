"""
Generate the tiny ONNX models the NumSharp.Interop.OnnxRuntime test project runs.

The models are the ORT half of the interop gate: NumSharp.Tests.Interop.OnnxRuntime feeds NDArrays
through the package into a REAL ONNX Runtime session and reads the outputs back, so every model here
is a real .onnx file ONNX Runtime executes -- not a mock. They are deliberately minimal (one or two
nodes, ~100-400 bytes each) and are COMMITTED, so the C# suite needs no Python at test time (the
same offline-corpus philosophy as test/oracle/gen_oracle.py). Re-run this script only when a model
changes; it self-checks every file by executing it through onnxruntime.

    python test/oracle/gen_onnx_models.py            # writes test/NumSharp.Tests.Interop.OnnxRuntime/Models/*.onnx
    python test/oracle/gen_onnx_models.py NAME ...   # writes only the named models (a new model is added
                                                     # without rewriting the committed ones, whose bytes
                                                     # depend on the installed onnx version)

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
                        the NDArray verbs decline a non-tensor OrtValue with a clear message, and
                        the ToNDArrays reader (sequence of tensors).
zipmap_int64.onnx       X [N,3] float32 -> ZipMap(ai.onnx.ml, classlabels_int64s) -> Z, a
                        sequence(map(int64, float)) -- the scikit-learn classifier output. Oracle for
                        the ToMap / ToMaps map readers.
string_io.onnx          X string -> Identity -> Y string. The ReadStringTensor reader (string OUTPUT)
                        and the refusal of a string model INPUT.
sequence_input.onnx     seq sequence<tensor(float)> -> SequenceLength -> len int64. A NON-tensor model
                        INPUT: Tier-2 Run must refuse feeding it an NDArray.
optional_input.onnx     X optional(tensor(float)) -> OptionalGetElement -> Y float32. An input declared
                        OPTIONAL: Tier-2 Run unwraps the declaration and feeds a plain tensor.
overridable_init.onnx   x [3] float32, w [3] float32 (an initializer that is ALSO a graph input, value
                        [10, 20, 30]) -> Add -> y. An OVERRIDABLE INITIALIZER: ORT lists w in
                        OverridableInitializerMetadata, not in InputMetadata; Tier-2 Run accepts a value
                        for it and runs with the default when none is fed.
zipmap_string.onnx      X [N,3] float32 -> ZipMap(classlabels_strings=[cat, dog, bird]) -> Z, a
                        sequence(map(string, float)): a classifier with STRING class labels. The map
                        readers return the keys as a text array.
classifier_str.onnx     X [N,3] float32 -> ArgMax -> LabelEncoder(int64 -> string) -> label (a STRING
                        tensor [N]) ; ZipMap(strings) -> probabilities. The usual skl2onnx classifier
                        outputs with string labels: a text output next to a non-tensor output.
mixed_string_float.onnx text ['N'] string, x ['N'] float32 -> Identity each -> echo, y. A TEXT input next
                        to a numeric one in one run, and a text output next to a numeric one.
passthrough.onnx        X ['N'] float32 is a graph input AND a graph output (no node produces it) ;
                        X -> Neg -> Y. An OUTPUT THAT IS A FED INPUT: ONNX Runtime returns the fed tensor
                        itself for it and never writes a pre-allocated buffer, so the bridge must.
overridable_out.onnx    x [3] float32, w [3] float32 (an overridable initializer, value [10, 20, 30]) ->
                        Add -> y ; w is ALSO a graph output. Not fed, the output w is the default value
                        (ORT copies it); fed, it is the fed value and a pre-allocated buffer receives the
                        DEFAULT instead.
passthrough_text.onnx   text ['N'] string is a graph input AND a graph output ; x ['N'] float32 -> Neg -> y.
                        A TEXT input passed through: the value ORT returns for it is the string tensor
                        the bridge built, which is ORT's own memory, not NumSharp's.
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


def _model(name, nodes, inputs, outputs, value_info=(), check=True, opset_imports=None, initializer=()):
    graph = helper.make_graph(nodes, name, inputs, outputs, value_info=list(value_info), initializer=list(initializer))
    model = helper.make_model(graph, producer_name="NumSharp.Interop.OnnxRuntime tests",
                              opset_imports=opset_imports or [helper.make_opsetid("", OPSET)])
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

    # ZipMap (ai.onnx.ml) -> sequence(map(int64, float)): the scikit-learn classifier output shape
    # (probabilities keyed by class label). Exercises BOTH the sequence walk and the map reader.
    map_type = helper.make_map_type_proto(TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, None))
    zipmap_out = helper.make_value_info("Z", helper.make_sequence_type_proto(map_type))
    models["zipmap_int64"] = _model(
        "zipmap_int64",
        [helper.make_node("ZipMap", ["X"], ["Z"], domain="ai.onnx.ml", classlabels_int64s=[10, 20, 30])],
        [_tensor("X", TensorProto.FLOAT, ["N", 3])], [zipmap_out], check=False,
        opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid("ai.onnx.ml", 1)])

    # String Identity: a string tensor in AND out. The interop reads string OUTPUTS with a dedicated
    # helper (NumSharp has no string dtype) and refuses a string model INPUT with a clear message.
    models["string_io"] = _model(
        "string_io",
        [helper.make_node("Identity", ["X"], ["Y"])],
        [_tensor("X", TensorProto.STRING)], [_tensor("Y", TensorProto.STRING)], check=False)

    # A NON-tensor (sequence) model INPUT: SequenceLength(sequence<tensor(float)>) -> int64 scalar.
    # Tier-2 Run must refuse feeding an NDArray to it with a "not a tensor" message.
    seq_in = helper.make_tensor_sequence_value_info("seq", TensorProto.FLOAT, None)
    models["sequence_input"] = _model(
        "sequence_input",
        [helper.make_node("SequenceLength", ["seq"], ["len"])],
        [seq_in], [_tensor("len", TensorProto.INT64, [])], check=False)

    # An OPTIONAL model input: optional(tensor(float)) -> OptionalGetElement -> the tensor. ONNX Runtime
    # takes a plain tensor where an optional is declared; Tier-2 Run must unwrap the declaration.
    optional_float = onnx.TypeProto()
    optional_float.optional_type.elem_type.tensor_type.elem_type = TensorProto.FLOAT
    models["optional_input"] = _model(
        "optional_input",
        [helper.make_node("OptionalGetElement", ["X"], ["Y"])],
        [helper.make_value_info("X", optional_float)], [_tensor("Y", TensorProto.FLOAT)], check=False)

    # An OVERRIDABLE INITIALIZER: w is an initializer that is also listed as a graph input (legal from IR 4
    # on), so a caller may replace its value for one run. ORT reports it in OverridableInitializerMetadata.
    weight = helper.make_tensor("w", TensorProto.FLOAT, [3], [10.0, 20.0, 30.0])
    models["overridable_init"] = _model(
        "overridable_init",
        [helper.make_node("Add", ["x", "w"], ["y"])],
        [_tensor("x", TensorProto.FLOAT, [3]), _tensor("w", TensorProto.FLOAT, [3])],
        [_tensor("y", TensorProto.FLOAT, [3])], initializer=[weight])

    # String class labels. sequence(map(string, float)) has no helper: build the TypeProto by hand.
    def seq_of_string_map(name):
        t = onnx.TypeProto()
        t.sequence_type.elem_type.map_type.key_type = TensorProto.STRING
        t.sequence_type.elem_type.map_type.value_type.tensor_type.elem_type = TensorProto.FLOAT
        return helper.make_value_info(name, t)

    ml_opsets = [helper.make_opsetid("", OPSET), helper.make_opsetid("ai.onnx.ml", 2)]
    labels = ["cat", "dog", "bird"]
    models["zipmap_string"] = _model(
        "zipmap_string",
        [helper.make_node("ZipMap", ["X"], ["Z"], domain="ai.onnx.ml", classlabels_strings=labels)],
        [_tensor("X", TensorProto.FLOAT, ["N", 3])], [seq_of_string_map("Z")], check=False, opset_imports=ml_opsets)

    models["classifier_str"] = _model(
        "classifier_str",
        [helper.make_node("ArgMax", ["X"], ["idx"], axis=1, keepdims=0),
         helper.make_node("LabelEncoder", ["idx"], ["label"], domain="ai.onnx.ml",
                          keys_int64s=[0, 1, 2], values_strings=labels, default_string="?"),
         helper.make_node("ZipMap", ["X"], ["probabilities"], domain="ai.onnx.ml", classlabels_strings=labels)],
        [_tensor("X", TensorProto.FLOAT, ["N", 3])],
        [_tensor("label", TensorProto.STRING, ["N"]), seq_of_string_map("probabilities")],
        check=False, opset_imports=ml_opsets)

    models["mixed_string_float"] = _model(
        "mixed_string_float",
        [helper.make_node("Identity", ["x"], ["y"]), helper.make_node("Identity", ["text"], ["echo"])],
        [_tensor("text", TensorProto.STRING, ["N"]), _tensor("x", TensorProto.FLOAT, ["N"])],
        [_tensor("y", TensorProto.FLOAT, ["N"]), _tensor("echo", TensorProto.STRING, ["N"])])

    # An OUTPUT that no node produces: X is a graph input and a graph output (legal ONNX). ONNX Runtime puts the
    # fed tensor into the output slot, so the value it returns for X lies in the CALLER's memory, and a buffer the
    # caller pre-allocates for X is never written. Y = Neg(X) is an ordinary output next to it.
    models["passthrough"] = _model(
        "passthrough",
        [helper.make_node("Neg", ["X"], ["Y"])],
        [_tensor("X", TensorProto.FLOAT, ["N"])],
        [_tensor("X", TensorProto.FLOAT, ["N"]), _tensor("Y", TensorProto.FLOAT, ["N"])])

    # An output that is an OVERRIDABLE INITIALIZER: w is an initializer, a graph input and a graph output.
    weight_out = helper.make_tensor("w", TensorProto.FLOAT, [3], [10.0, 20.0, 30.0])
    models["overridable_out"] = _model(
        "overridable_out",
        [helper.make_node("Add", ["x", "w"], ["y"])],
        [_tensor("x", TensorProto.FLOAT, [3]), _tensor("w", TensorProto.FLOAT, [3])],
        [_tensor("w", TensorProto.FLOAT, [3]), _tensor("y", TensorProto.FLOAT, [3])], initializer=[weight_out])

    # A TEXT input passed through to an output. The fed string tensor is ONNX Runtime's own object (text never
    # shares NumSharp memory), so this output must take the ordinary text copy, not the fed-memory route.
    models["passthrough_text"] = _model(
        "passthrough_text",
        [helper.make_node("Neg", ["x"], ["y"])],
        [_tensor("text", TensorProto.STRING, ["N"]), _tensor("x", TensorProto.FLOAT, ["N"])],
        [_tensor("text", TensorProto.STRING, ["N"]), _tensor("y", TensorProto.FLOAT, ["N"])])

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
    elif name == "zipmap_int64":
        z, = sess.run(None, {"X": np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], np.float32)})
        assert isinstance(z, list) and len(z) == 2, name           # sequence of 2 maps
        assert isinstance(z[0], dict) and set(z[0]) == {10, 20, 30}, name
        assert abs(z[0][20] - 0.7) < 1e-6, name
    elif name == "string_io":
        y, = sess.run(None, {"X": np.array(["héllo", "world"], dtype=object)})
        assert list(y) == ["héllo", "world"], name
    elif name == "sequence_input":
        n, = sess.run(None, {"seq": [np.arange(3, dtype=np.float32), np.arange(2, dtype=np.float32)]})
        assert int(n) == 2, name
    elif name == "optional_input":
        y, = sess.run(None, {"X": np.arange(4, dtype=np.float32)})   # a plain tensor for the optional
        assert np.array_equal(y, np.arange(4, dtype=np.float32)), name
    elif name == "overridable_init":
        assert [i.name for i in sess.get_inputs()] == ["x"], name
        assert [i.name for i in sess.get_overridable_initializers()] == ["w"], name
        x = np.array([1, 2, 3], np.float32)
        y, = sess.run(None, {"x": x})                                # the default weight
        assert np.array_equal(y, np.array([11, 22, 33], np.float32)), name
        y, = sess.run(None, {"x": x, "w": np.array([1, 1, 1], np.float32)})   # the weight replaced
        assert np.array_equal(y, np.array([2, 3, 4], np.float32)), name
    elif name == "zipmap_string":
        z, = sess.run(None, {"X": np.array([[0.1, 0.7, 0.2]], np.float32)})
        assert isinstance(z, list) and set(z[0]) == {"cat", "dog", "bird"} and abs(z[0]["dog"] - 0.7) < 1e-6, name
    elif name == "classifier_str":
        label, probs = sess.run(None, {"X": np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]], np.float32)})
        assert list(label) == ["dog", "cat"], name
        assert set(probs[1]) == {"cat", "dog", "bird"}, name
    elif name == "mixed_string_float":
        y, echo = sess.run(None, {"text": np.array(["a", "bb"], dtype=object), "x": np.array([1, 2], np.float32)})
        assert list(echo) == ["a", "bb"] and np.array_equal(y, np.array([1, 2], np.float32)), name
    elif name == "passthrough":
        assert [o.name for o in sess.get_outputs()] == ["X", "Y"], name
        x = np.array([1, 2, 3], np.float32)
        x_out, y = sess.run(None, {"X": x})
        assert np.array_equal(x_out, x) and np.array_equal(y, -x), name
    elif name == "overridable_out":
        assert [i.name for i in sess.get_overridable_initializers()] == ["w"], name
        x = np.array([1, 2, 3], np.float32)
        w, y = sess.run(None, {"x": x})                              # the default weight comes out
        assert np.array_equal(w, np.array([10, 20, 30], np.float32)) and np.array_equal(y, x + w), name
        fed = np.array([100, 200, 300], np.float32)
        w, y = sess.run(None, {"x": x, "w": fed})                    # the fed weight comes out
        assert np.array_equal(w, fed) and np.array_equal(y, x + fed), name
    elif name == "passthrough_text":
        echo, y = sess.run(None, {"text": np.array(["a", "bb"], dtype=object), "x": np.array([1, 2], np.float32)})
        assert list(echo) == ["a", "bb"] and np.array_equal(y, np.array([-1, -2], np.float32)), name
    else:
        raise AssertionError(f"no self-check for {name}")


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    models = build_all()
    wanted = sys.argv[1:]
    unknown = [name for name in wanted if name not in models]
    if unknown:
        print(f"unknown model name(s): {', '.join(unknown)}; known: {', '.join(models)}")
        return 2
    if wanted:
        models = {name: models[name] for name in wanted}
    for name, model in models.items():
        path = os.path.join(OUT_DIR, f"{name}.onnx")
        onnx.save(model, path)
        self_check(path, name)
        print(f"{name}.onnx  {os.path.getsize(path)} bytes  ok")
    print(f"{len(models)} models -> {os.path.relpath(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
