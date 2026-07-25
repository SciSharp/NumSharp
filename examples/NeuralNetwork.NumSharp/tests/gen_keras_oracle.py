# Edge-case oracle generator for NeuralNetwork.NumSharp.
#
# Source of truth: Keras 3 (JAX backend, float32) for activations, losses and
# their GRADIENTS (jax.grad through the real Keras graph), Keras metric
# classes for precision/recall/binary-accuracy/top-k, scikit-learn for
# f1/r2/rmse, NumPy for argmax tie-breaking.
#
# Output: tests/corpus/keras_edge_oracle.json — committed; the C# replay
# (verify_edge_cases.cs) runs WITHOUT Python, house oracle philosophy.
#
# Regenerate:  python gen_keras_oracle.py   (from this directory)
# Requires: keras>=3.15, jax, scikit-learn, numpy (float32 defaults matter).
#
# Excused divergences (never silent): a case may carry "expected_ns" when
# NumSharp's DEFINED behavior intentionally differs from Keras; the replay
# asserts against expected_ns and prints the excuse with the Keras value.
import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import numpy as np
import jax
import jax.numpy as jnp
import keras

F32 = np.float32
INF = np.float32(np.inf)
NAN = np.float32(np.nan)

cases = []


def clean(x):
    """floats → JSON-safe (NaN/±inf as string sentinels), recursively."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.ndarray, jnp.ndarray)):
        return clean(np.asarray(x).tolist())
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if np.isnan(v):
            return "NaN"
        if np.isposinf(v):
            return "Infinity"
        if np.isneginf(v):
            return "-Infinity"
        return v
    if isinstance(x, (int, np.integer)):
        return int(x)
    return x


def add(kind, name, inputs, expected, params=None, note="", expected_ns=None, tol=None):
    c = {"kind": kind, "name": name, "inputs": clean(inputs), "expected": clean(expected)}
    if params:
        c["params"] = params
    if note:
        c["note"] = note
    if expected_ns is not None:
        c["expected_ns"] = clean(expected_ns)
    if tol:
        c["tol"] = tol
    cases.append(c)


# ======================================================================
# 1. Activations — values over the edge grid, then gradients
# ======================================================================
EDGE_X = np.array(
    [-np.inf, -1e30, -88, -20, -8.5, -1, -0.1, -1e-7, 0, 1e-7, 0.1, 1, 8.5, 20, 88, 1e30, np.inf, np.nan],
    dtype=F32,
)
# gradient grid: differentiable region + saturation tails (no nan/inf — grads
# there are backend artifacts, not contract)
GRAD_X = np.array([-88, -20, -8.5, -1, -0.1, -1e-7, 0, 1e-7, 0.1, 1, 8.5, 20, 88], dtype=F32)

ACTS = {
    "relu": lambda x: keras.activations.relu(x),
    "sigmoid": lambda x: keras.activations.sigmoid(x),
    "tanh": lambda x: keras.activations.tanh(x),
    "leaky_relu": lambda x: keras.activations.leaky_relu(x, negative_slope=0.3),
    "elu": lambda x: keras.activations.elu(x, alpha=1.0),
    "gelu": lambda x: keras.activations.gelu(x, approximate=True),
    "silu": lambda x: keras.activations.silu(x),
    "softplus": lambda x: keras.activations.softplus(x),
    "selu": lambda x: keras.activations.selu(x),
}

for name, fn in ACTS.items():
    y = np.asarray(fn(jnp.array(EDGE_X)), dtype=F32)
    add("activation", name, {"x": EDGE_X}, y)

    g = np.asarray(jax.grad(lambda x: jnp.sum(fn(x)))(jnp.array(GRAD_X)), dtype=F32)
    add("activation_grad", name, {"x": GRAD_X}, g)

# softmax rows: stability / ties / -inf lanes / single logit
SOFTMAX_ROWS = [
    [1000.0, 1000.0, 1000.0],            # huge ties -> uniform, no overflow
    [-1000.0, 0.0, 1000.0],              # extreme spread -> one-hot
    [0.0, 0.0, 0.0, 0.0],                # uniform
    ["-Infinity", 0.0, 0.0],             # -inf lane -> 0 weight
    [5.0],                               # single class -> 1
    [-1e30, -1e30, 0.0],
]
for row in SOFTMAX_ROWS:
    x = np.array([[np.float32(-np.inf) if v == "-Infinity" else v for v in row]], dtype=F32)
    y = np.asarray(keras.activations.softmax(jnp.array(x), axis=-1), dtype=F32)
    add("activation", "softmax", {"x": x}, y)

# ======================================================================
# 2. Losses — scalar values on edge inputs (Keras Loss classes = the
#    sum_over_batch_size reduction our classes implement)
# ======================================================================
def loss_value(loss_obj, yt, yp):
    return float(loss_obj(jnp.array(yt), jnp.array(yp)))


def loss_grad(loss_obj, yt, yp):
    g = jax.grad(lambda p: loss_obj(jnp.array(yt), p))(jnp.array(yp))
    return np.asarray(g, dtype=F32)


# ---- categorical crossentropy (probabilities; rows sum to 1 = the contract)
cce = keras.losses.CategoricalCrossentropy()
CCE_CASES = [
    ("identity_onehot", [[1, 0, 0]], [[1, 0, 0]]),
    ("saturated", [[0, 1, 0]], [[1e-7, 1 - 2e-7, 1e-7]]),
    ("wrong_saturated", [[1, 0, 0]], [[1e-7, 1 - 2e-7, 1e-7]]),
    ("uniform", [[1, 0, 0], [0, 0, 1]], [[1 / 3] * 3, [1 / 3] * 3]),
    ("zero_prob_true_class", [[0, 1]], [[1, 0]]),  # clip saves log(0)
]
for tag, yt, yp in CCE_CASES:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("loss", "cce", {"y_true": yt32, "y_pred": yp32}, loss_value(cce, yt32, yp32), note=tag)

# out-of-contract: non-normalized probs — Keras silently renormalizes, we
# document "expects post-softmax" and do not. Excused with both values.
yt32 = np.array([[1, 0, 0]], F32); yp32 = np.array([[0.2, 0.2, 0.2]], F32)
ns_val = float(-np.log(np.clip(0.2, 1e-7, 1 - 1e-7)))
add("loss", "cce", {"y_true": yt32, "y_pred": yp32}, loss_value(cce, yt32, yp32),
    note="non_normalized_probs — Keras renormalizes rows; NumSharp treats input as given (docstring contract)",
    expected_ns=ns_val)

# cce GRADIENT: Keras differentiates through its renormalization, adding a
# projection term even for sum=1 rows. Our backward is the exact gradient of
# OUR forward (-y/clip(p)/batch, FD-verified). Excused pair, both recorded.
yt32 = np.array([[1, 0, 0]], F32); yp32 = np.array([[0.9, 0.05, 0.05]], F32)
add("loss_grad", "cce", {"y_true": yt32, "y_pred": yp32}, loss_grad(cce, yt32, yp32),
    note="keras grad includes renormalization projection; NumSharp = textbook -y/p/batch (exact grad of our forward)",
    expected_ns=(-yt32 / np.clip(yp32, 1e-7, 1 - 1e-7) / yp32.shape[0]))

# ---- sparse categorical crossentropy (integer labels)
scce = keras.losses.SparseCategoricalCrossentropy()
SCCE_CASES = [
    ("boundary_labels", [0, 2], [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]),
    ("zero_prob", [1], [[1, 0]]),
    ("saturated", [1], [[1e-7, 1 - 2e-7, 1e-7]]),
]
for tag, yt, yp in SCCE_CASES:
    yt32 = np.array(yt, np.int32); yp32 = np.array(yp, F32)
    add("loss", "scce", {"labels": yt32.tolist(), "y_pred": yp32}, loss_value(scce, yt32, yp32), note=tag)

# ---- binary crossentropy
bce = keras.losses.BinaryCrossentropy()
BCE_CASES = [
    ("exact_01", [[1, 0]], [[1, 0]]),          # clip saves both logs
    ("exactly_wrong", [[1, 0]], [[0, 1]]),
    ("half", [[1, 0], [1, 1]], [[0.5, 0.5], [0.5, 0.5]]),
    ("near_sat", [[1]], [[1 - 1e-7]]),
]
for tag, yt, yp in BCE_CASES:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("loss", "bce", {"y_true": yt32, "y_pred": yp32}, loss_value(bce, yt32, yp32), note=tag)
    if tag == "half":
        add("loss_grad", "bce", {"y_true": yt32, "y_pred": yp32}, loss_grad(bce, yt32, yp32), note=tag)

# ---- huber: the |e| == delta boundary and both regimes; custom delta
for delta in (1.0, 0.5):
    hub = keras.losses.Huber(delta=delta)
    yt = np.array([[0.0, 0.0, 0.0, 0.0]], F32)
    yp = np.array([[0.5 * delta, delta, 2 * delta, -3 * delta]], F32)
    add("loss", "huber", {"y_true": yt, "y_pred": yp}, loss_value(hub, yt, yp), params={"delta": delta},
        note="boundary |e|==delta included")
    add("loss_grad", "huber", {"y_true": yt, "y_pred": yp}, loss_grad(hub, yt, yp), params={"delta": delta})

# ---- hinge: ±1 labels, {0,1} auto-convert, MIXED labels (no conversion), zero margin
hinge = keras.losses.Hinge()
HINGE_CASES = [
    ("pm1", [[1, -1]], [[0.8, -0.4]]),
    ("binary01", [[1, 0]], [[0.8, -0.4]]),
    ("zero_margin", [[1, -1]], [[1.0, -1.0]]),
    ("mixed_labels_no_convert", [[0.5, -0.5]], [[0.8, -0.4]]),
]
for tag, yt, yp in HINGE_CASES:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("loss", "hinge", {"y_true": yt32, "y_pred": yp32}, loss_value(hinge, yt32, yp32), note=tag)
add("loss_grad", "hinge", {"y_true": np.array([[1, -1]], F32), "y_pred": np.array([[0.8, -0.4]], F32)},
    loss_grad(hinge, np.array([[1, -1]], F32), np.array([[0.8, -0.4]], F32)))

# ---- kl divergence: exact zeros/ones in y_true, y_pred at clip floor
kld = keras.losses.KLDivergence()
KL_CASES = [
    ("true_zeros_ones", [[1, 0], [0, 1]], [[0.9, 0.1], [0.4, 0.6]]),
    ("pred_at_floor", [[0.5, 0.5]], [[1e-7, 1 - 1e-7]]),
    ("identical", [[0.3, 0.7]], [[0.3, 0.7]]),
]
for tag, yt, yp in KL_CASES:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("loss", "kl", {"y_true": yt32, "y_pred": yp32}, loss_value(kld, yt32, yp32), note=tag)
add("loss_grad", "kl", {"y_true": np.array([[0.6, 0.4]], F32), "y_pred": np.array([[0.25, 0.75]], F32)},
    loss_grad(kld, np.array([[0.6, 0.4]], F32), np.array([[0.25, 0.75]], F32)))

# ---- log cosh: overflow-safe tails, zero
lc = keras.losses.LogCosh()
LC_CASES = [
    ("zero", [[0.0, 0.0]], [[0.0, 0.0]]),
    ("large_e", [[0.0, 0.0]], [[300.0, -300.0]]),
    ("tiny_e", [[0.0]], [[1e-7]]),
]
for tag, yt, yp in LC_CASES:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("loss", "logcosh", {"y_true": yt32, "y_pred": yp32}, loss_value(lc, yt32, yp32), note=tag)
add("loss_grad", "logcosh", {"y_true": np.array([[0.0, 0.0]], F32), "y_pred": np.array([[2.0, -0.5]], F32)},
    loss_grad(lc, np.array([[0.0, 0.0]], F32), np.array([[2.0, -0.5]], F32)))

# ---- SoftmaxCrossEntropy == Keras CCE(from_logits=True): value AND dL/dlogits
cce_logits = keras.losses.CategoricalCrossentropy(from_logits=True)
SCE_CASES = [
    ("plain", [[1, 0, 0], [0, 0, 1]], [[2.0, 1.0, 0.1], [-1.0, 0.0, 1.0]]),
    ("huge_logits", [[0, 1]], [[1000.0, 1010.0]]),
    ("neg_huge", [[1, 0]], [[-1000.0, 0.0]]),
    ("ties", [[1, 0, 0]], [[3.0, 3.0, 3.0]]),
]
for tag, yt, yp in SCE_CASES:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("loss", "sce_logits", {"y_true": yt32, "logits": yp32}, loss_value(cce_logits, yt32, yp32), note=tag)
    add("loss_grad", "sce_logits", {"y_true": yt32, "logits": yp32}, loss_grad(cce_logits, yt32, yp32), note=tag)

# ======================================================================
# 3. Metrics — Keras classes + sklearn
# ======================================================================
def keras_metric(m, yt, yp):
    m.reset_state()
    m.update_state(jnp.array(yt), jnp.array(yp))
    return float(m.result())


# precision/recall: threshold-exact 0.5 (strict >), zero-denominator conventions
PRT = [
    ("threshold_exact", [1, 0, 1, 0], [0.5, 0.5, 0.7, 0.2]),   # 0.5 is NOT > 0.5
    ("no_pred_pos", [1, 1, 0], [0.1, 0.2, 0.3]),
    ("no_true_pos", [0, 0, 0], [0.9, 0.8, 0.7]),
    ("perfect", [1, 0, 1], [0.9, 0.1, 0.8]),
]
for tag, yt, yp in PRT:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("metric", "precision", {"y_true": yt32, "y_pred": yp32},
        keras_metric(keras.metrics.Precision(thresholds=0.5), yt32, yp32), note=tag)
    add("metric", "recall", {"y_true": yt32, "y_pred": yp32},
        keras_metric(keras.metrics.Recall(thresholds=0.5), yt32, yp32), note=tag)

# binary accuracy at the 0.5 boundary (Keras: pred > 0.5)
BA = [
    ("boundary", [1, 0, 1, 0], [0.5, 0.5, 0.51, 0.49]),
    ("plain", [1, 0], [0.9, 0.2]),
]
for tag, yt, yp in BA:
    yt32 = np.array(yt, F32); yp32 = np.array(yp, F32)
    add("metric", "binary_accuracy", {"y_true": yt32, "y_pred": yp32},
        keras_metric(keras.metrics.BinaryAccuracy(threshold=0.5), yt32, yp32), note=tag)

# top-k: all-tied rows, k >= classes, k=1
TOPK = [
    ("ties", 2, [3], [[0.25, 0.25, 0.25, 0.25]]),
    ("k_ge_classes", 5, [2], [[0.1, 0.2, 0.7]]),
    ("k1_is_accuracy", 1, [0, 1], [[0.9, 0.1], [0.2, 0.8]]),
    ("tie_with_true", 2, [1], [[0.4, 0.3, 0.3]]),   # 0.3 tie straddles k
]
for tag, k, lbl, yp in TOPK:
    onehot = np.eye(len(yp[0]), dtype=F32)[lbl]
    yp32 = np.array(yp, F32)
    add("metric", "top_k", {"y_true": onehot, "y_pred": yp32},
        keras_metric(keras.metrics.TopKCategoricalAccuracy(k=k), onehot, yp32),
        params={"k": k}, note=tag)

# sklearn: f1 zero-division + macro absent-class; r2 edge cases; rmse
from sklearn.metrics import f1_score, r2_score

try:
    from sklearn.metrics import root_mean_squared_error as sk_rmse
except ImportError:  # older sklearn
    from sklearn.metrics import mean_squared_error

    def sk_rmse(a, b):
        return mean_squared_error(a, b, squared=False)

F1B = [
    ("no_pred_pos", [1, 1, 0], [0.1, 0.2, 0.3]),
    ("no_positives_at_all", [0, 0, 0], [0.1, 0.2, 0.3]),
    ("perfect", [1, 0, 1], [0.9, 0.1, 0.8]),
]
for tag, yt, yp in F1B:
    pred = (np.array(yp, F32) > 0.5).astype(int)
    add("metric", "f1_binary", {"y_true": np.array(yt, F32), "y_pred": np.array(yp, F32)},
        float(f1_score(yt, pred, zero_division=0)), note=tag)

# macro F1 with a class absent from BOTH preds and labels (class 2 of 3)
mp = np.array([[.8, .1, .1], [.1, .8, .1], [.8, .1, .1]], F32)
ml_idx = [0, 1, 1]
onehot = np.eye(3, dtype=F32)[ml_idx]
add("metric", "f1_macro", {"y_true": onehot, "y_pred": mp},
    float(f1_score(ml_idx, mp.argmax(1), average="macro", zero_division=0, labels=[0, 1, 2])),
    note="class 2 absent from preds AND labels -> per-class 0 (zero_division=0)")

R2 = [
    ("constant_labels_imperfect", [2.5, 3.0], [3.0, 3.0], 0.0),
    ("constant_labels_perfect", [3.0, 3.0], [3.0, 3.0], 1.0),
    ("negative", [10.0, -10.0, 10.0], [1.0, 2.0, 3.0], None),
    ("plain", [2.5, 0.0, 2.0, 8.0], [3.0, -0.5, 2.0, 7.0], None),
]
for tag, yp, yt, forced in R2:
    yp32 = np.array(yp, F32); yt32 = np.array(yt, F32)
    if forced is None:
        val = float(r2_score(yt32, yp32))
    else:
        # sklearn WARNS and returns 0/1 by its own convention for constant y_true;
        # we pin the documented convention explicitly.
        val = forced
    add("metric", "r2", {"y_true": yt32, "y_pred": yp32}, val, note=tag)

add("metric", "rmse", {"y_true": np.array([3.0, -0.5, 2.0, 7.0], F32), "y_pred": np.array([2.5, 0.0, 2.0, 8.0], F32)},
    float(sk_rmse([3.0, -0.5, 2.0, 7.0], [2.5, 0.0, 2.0, 8.0])))

# accuracy argmax tie-breaking: numpy argmax takes the FIRST maximum
tp = np.array([[0.5, 0.5], [0.3, 0.3]], F32)
tl = np.eye(2, dtype=F32)[[0, 1]]
expected_acc = float(np.mean(tp.argmax(1) == tl.argmax(1)))
add("metric", "accuracy", {"y_true": tl, "y_pred": tp}, expected_acc,
    note="argmax tie -> first max (numpy semantics); row0 correct, row1 wrong")

# ======================================================================
# 4. Initializer std/limit targets — sampled from Keras's own initializers
#    (statistical; large shapes only, 5% tolerance in the replay)
# ======================================================================
INIT_SHAPES = {
    "he_normal": [(784, 128), (3, 3, 4, 8), (512,)],
    "glorot_uniform": [(100, 200)],
    "lecun_normal": [(64, 64)],
}
for name, shapes in INIT_SHAPES.items():
    ki = getattr(keras.initializers, {"he_normal": "HeNormal", "glorot_uniform": "GlorotUniform",
                                      "lecun_normal": "LecunNormal"}[name])
    for shape in shapes:
        stds = [float(np.std(np.asarray(ki(seed=s)(shape)))) for s in (1, 2, 3, 4, 5)]
        row = {"kind": "init_std", "name": name, "params": {"shape": list(shape)},
               "expected": float(np.mean(stds)),
               "note": f"mean of 5 seeded Keras draws; replay tolerance 5%"}
        if name == "glorot_uniform":
            row["limit"] = float(np.sqrt(6.0 / (100 + 200)))
        cases.append(row)

# ======================================================================
# 5. P4 layers — Dropout / BatchNormalization / LayerNormalization /
#    Embedding, values AND gradients, through the REAL Keras layers.
#
#    Gradients come from jax.grad over keras Layer.stateless_call, so the
#    oracle differentiates the actual Keras implementation rather than a
#    re-derivation of it. Every case carries an explicit upstream cotangent
#    so the recorded gradient is dL/d* for L = sum(out * upstream) — a
#    non-uniform upstream is what catches an axis mix-up that a vector of
#    ones would hide.
# ======================================================================
def _vars_in_order(layer, names, trainable=True):
    """Keras variables for `names`, in the order stateless_call expects."""
    pool = layer.trainable_variables if trainable else layer.non_trainable_variables
    by_name = {}
    for v in pool:
        key = v.path.split("/")[-1]
        by_name[key] = v
    missing = [n for n in names if n not in by_name]
    if missing:
        raise RuntimeError(f"{layer.__class__.__name__}: missing {missing}, have {list(by_name)}")
    ordered = [by_name[n] for n in names]
    # sanity: the pool must be exactly these, in this order
    assert [v.path for v in pool] == [v.path for v in ordered], \
        f"variable order mismatch: {[v.path for v in pool]} vs {[v.path for v in ordered]}"
    return ordered


def _layer_grads(layer, x, tv, ntv, upstream, training):
    """(y, dx, [dtv...], ntv_out) for one Keras layer via stateless_call.

    `training=` is only forwarded to layers whose call() declares it —
    LayerNormalization and Embedding do not, and passing it is a TypeError.
    """
    import inspect as _inspect
    kw = {}
    if "training" in _inspect.signature(layer.call).parameters:
        kw["training"] = training

    def fwd(x_, tv_):
        out, _ = layer.stateless_call(tv_, ntv, x_, **kw)
        return out

    y, ntv_out = layer.stateless_call(tv, ntv, jnp.array(x), **kw)
    loss = lambda x_, tv_: jnp.sum(fwd(x_, tv_) * jnp.array(upstream))

    # An integer input is not differentiable — jax.grad refuses it outright.
    # That is the same fact our Embedding encodes by leaving InputGrad null.
    if np.issubdtype(np.asarray(x).dtype, np.integer):
        dtv = jax.grad(lambda tv_: loss(jnp.array(x), tv_))(tv)
        dx = None
    else:
        dx, dtv = jax.grad(loss, argnums=(0, 1))(jnp.array(x), tv)

    return (np.asarray(y, F32),
            None if dx is None else np.asarray(dx, F32),
            [np.asarray(g, F32) for g in dtv],
            [np.asarray(v, F32) for v in ntv_out])


# ---- Dropout: the inverted-dropout CONTRACT (the mask itself is RNG-specific,
#      so what is pinned is the scale, the support {0, 1/(1-rate)} and the
#      inference passthrough).
for rate in (0.0, 0.25, 0.5, 0.9):
    ones = np.ones((200, 20), F32)
    d = keras.layers.Dropout(rate, seed=1234)
    train_out = np.asarray(d(jnp.array(ones), training=True), F32)
    eval_out = np.asarray(d(jnp.array(ones), training=False), F32)
    nonzero = np.unique(train_out[train_out != 0])
    add("dropout", "contract", {"rate": rate},
        {"scale": float(1.0 / (1.0 - rate)) if rate < 1 else 0.0,
         "nonzero_values": sorted(float(v) for v in nonzero),
         "eval_is_identity": bool(np.array_equal(eval_out, ones)),
         "kept_fraction": float((train_out != 0).mean())},
        params={"rate": rate},
        note="inverted dropout: survivors scale by 1/(1-rate); inference is the identity")

# ---- BatchNormalization: training forward+backward, running-stat update, and
#      the inference path off non-default running statistics.
BN_X = np.array([[1.0, 2.0, -3.0], [3.0, 5.0, 0.5], [7.0, 11.0, -1.0], [-2.0, 0.0, 4.0]], F32)
BN_UP = np.array([[1.0, -2.0, 0.5], [0.25, 3.0, -1.0], [-0.5, 1.0, 2.0], [2.0, -0.25, 0.75]], F32)
BN_GAMMA = np.array([1.5, -0.5, 2.0], F32)
BN_BETA = np.array([0.25, -1.0, 0.5], F32)
BN_MM = np.array([0.5, 1.0, -0.5], F32)
BN_MV = np.array([2.0, 4.0, 0.25], F32)

for training in (True, False):
    bn = keras.layers.BatchNormalization()
    bn.build(BN_X.shape)
    tvs = _vars_in_order(bn, ["gamma", "beta"])
    ntvs = _vars_in_order(bn, ["moving_mean", "moving_variance"], trainable=False)
    tv = [jnp.array(BN_GAMMA), jnp.array(BN_BETA)]
    ntv = [jnp.array(BN_MM), jnp.array(BN_MV)]

    y, dx, (dgamma, dbeta), (mm_out, mv_out) = _layer_grads(bn, BN_X, tv, ntv, BN_UP, training)
    add("batchnorm", "train" if training else "inference",
        {"x": BN_X, "gamma": BN_GAMMA, "beta": BN_BETA,
         "moving_mean": BN_MM, "moving_variance": BN_MV, "upstream": BN_UP},
        {"y": y, "dx": dx, "dgamma": dgamma, "dbeta": dbeta,
         "moving_mean_out": mm_out, "moving_variance_out": mv_out},
        params={"training": training, "momentum": float(bn.momentum), "epsilon": float(bn.epsilon)},
        note=("population(ddof=0) variance; running = momentum*old + (1-momentum)*batch"
              if training else "inference uses the running statistics and does NOT update them"))

# batch of 1 in training mode: variance is exactly 0, so epsilon alone carries
# the division — the case that separates 1e-3 from 1e-5.
bn1 = keras.layers.BatchNormalization()
bn1.build((1, 3))
x1 = np.array([[4.0, -1.0, 0.0]], F32)
up1 = np.array([[1.0, 1.0, 1.0]], F32)
y, dx, (dgamma, dbeta), (mm_out, mv_out) = _layer_grads(
    bn1, x1, [jnp.array(BN_GAMMA), jnp.array(BN_BETA)],
    [jnp.array(np.zeros(3, F32)), jnp.array(np.ones(3, F32))], up1, True)
add("batchnorm", "train", {"x": x1, "gamma": BN_GAMMA, "beta": BN_BETA,
                           "moving_mean": np.zeros(3, F32), "moving_variance": np.ones(3, F32),
                           "upstream": up1},
    {"y": y, "dx": dx, "dgamma": dgamma, "dbeta": dbeta,
     "moving_mean_out": mm_out, "moving_variance_out": mv_out},
    params={"training": True, "momentum": 0.99, "epsilon": 1e-3},
    note="batch_of_1 — zero variance, epsilon alone divides")

# ---- LayerNormalization: per-sample statistics, no running state.
LN_X = np.array([[1.0, 2.0, -3.0, 0.5], [3.0, 5.0, 0.5, -1.0], [0.0, 0.0, 0.0, 0.0]], F32)
LN_UP = np.array([[1.0, -2.0, 0.5, 0.25], [0.25, 3.0, -1.0, 1.0], [-0.5, 1.0, 2.0, -1.5]], F32)
LN_GAMMA = np.array([1.5, -0.5, 2.0, 0.75], F32)
LN_BETA = np.array([0.25, -1.0, 0.5, 0.0], F32)

ln = keras.layers.LayerNormalization()
ln.build(LN_X.shape)
_vars_in_order(ln, ["gamma", "beta"])
y, dx, (dgamma, dbeta), _ = _layer_grads(
    ln, LN_X, [jnp.array(LN_GAMMA), jnp.array(LN_BETA)], [], LN_UP, True)
add("layernorm", "default", {"x": LN_X, "gamma": LN_GAMMA, "beta": LN_BETA, "upstream": LN_UP},
    {"y": y, "dx": dx, "dgamma": dgamma, "dbeta": dbeta},
    params={"epsilon": float(ln.epsilon)},
    note="row 2 is all-zeros: variance 0, epsilon alone divides; gamma/beta grads reduce over the BATCH")

# LayerNorm must not depend on the batch: the same row scored alone.
ln1 = keras.layers.LayerNormalization()
ln1.build((1, 4))
y1, dx1, (dg1, db1), _ = _layer_grads(
    ln1, LN_X[:1], [jnp.array(LN_GAMMA), jnp.array(LN_BETA)], [], LN_UP[:1], True)
add("layernorm", "single_row", {"x": LN_X[:1], "gamma": LN_GAMMA, "beta": LN_BETA, "upstream": LN_UP[:1]},
    {"y": y1, "dx": dx1, "dgamma": dg1, "dbeta": db1},
    params={"epsilon": 1e-3},
    note="batch-independence: identical to row 0 of the batched case")

# ---- Embedding: gather + the scatter-ADD backward. Duplicate indices are the
#      point — an assignment loop keeps only the last and passes every test
#      that has no repeats.
EMB_W = np.array([[0.1, 0.2], [-0.3, 0.4], [0.5, -0.6], [0.7, 0.8], [-0.9, 1.0]], F32)
for tag, idx, up in [
    ("duplicates", np.array([0, 2, 2, 0, 4], np.int32),
     np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], F32)),
    ("no_repeats", np.array([1, 3], np.int32),
     np.array([[1.0, -1.0], [2.0, -2.0]], F32)),
    ("all_same", np.array([3, 3, 3], np.int32),
     np.array([[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]], F32)),
]:
    emb = keras.layers.Embedding(EMB_W.shape[0], EMB_W.shape[1])
    emb.build((None,))
    _vars_in_order(emb, ["embeddings"])
    y, _, (dw,), _ = _layer_grads(emb, idx, [jnp.array(EMB_W)], [], up, True)
    add("embedding", "1d", {"w": EMB_W, "indices": idx.tolist(), "upstream": up},
        {"y": y, "dw": dw}, note=tag)

# 2-D (batch, timesteps) indices -> (batch, timesteps, dim)
idx2 = np.array([[0, 1], [1, 1]], np.int32)
up2 = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], F32)
emb2 = keras.layers.Embedding(EMB_W.shape[0], EMB_W.shape[1])
emb2.build((None, None))
y, _, (dw,), _ = _layer_grads(emb2, idx2, [jnp.array(EMB_W)], [], up2, True)
add("embedding", "2d", {"w": EMB_W, "indices": idx2.tolist(), "upstream": up2},
    {"y": y, "dw": dw}, note="sequence input; index 1 repeats three times across the batch")

# ---- Regularizers: penalty AND the gradient that must match it.
REG_W = np.array([[1.0, -2.0], [3.0, -4.0], [0.0, 0.5]], F32)
for kind, kwargs, obj in [
    ("l1", {"l1": 0.1}, keras.regularizers.L1(0.1)),
    ("l2", {"l2": 0.1}, keras.regularizers.L2(0.1)),
    ("l1l2", {"l1": 0.1, "l2": 0.05}, keras.regularizers.L1L2(0.1, 0.05)),
    ("l1", {"l1": 0.01}, keras.regularizers.L1()),      # Keras default strength
    ("l2", {"l2": 0.01}, keras.regularizers.L2()),
]:
    penalty = float(obj(jnp.array(REG_W)))
    grad = np.asarray(jax.grad(lambda w: obj(w))(jnp.array(REG_W)), F32)
    add("regularizer", kind, {"w": REG_W}, {"penalty": penalty, "grad": grad},
        params=kwargs,
        note="Keras has NO 1/2 on L2, so d(l2*sum(w^2))/dw = 2*l2*w; sign(0)=0 for L1")

# ======================================================================
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus", "keras_edge_oracle.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
meta = {
    "generator": "gen_keras_oracle.py",
    "keras": keras.__version__,
    "backend": keras.backend.backend(),
    "floatx": keras.backend.floatx(),
    "numpy": np.__version__,
    "cases": len(cases),
}
with open(out, "w") as f:
    json.dump({"meta": meta, "cases": cases}, f, indent=1)
print(f"wrote {out}: {len(cases)} cases | keras {keras.__version__} ({keras.backend.backend()}, {keras.backend.floatx()})")
