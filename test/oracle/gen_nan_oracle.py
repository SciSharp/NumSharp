#!/usr/bin/env python3
"""
NaN parity oracle — "do NumSharp's functions produce NumPy's NaN?"

A dedicated, committed corpus tier (`Fuzz/corpus/nan.jsonl`) that stresses every UNARY op for
which a NaN output is reachable, over the FULL special-value grid (finite, +-0, +-inf, and BOTH
NaN signs +-NaN), and records NumPy 2.4.2's EXACT output bytes.

The C# harness (FuzzCorpusTests.RunCorpus -> CompareArray) replays it with the NaN-contract policy
already wired there:
  * complex128 unary ops in `ComplexNanContractOps` (sqrt/log/exp/.../sign/abs) are compared
    BIT-EXACT on the NaN sign (NumSharp reproduces NumPy's MSVC-UCRT NaN sign per-path);
  * every other op / dtype keeps the tokenizing compare, so the tier still asserts that NumSharp
    produces *a* NaN (the correct VALUE) exactly where NumPy does, and that the non-NaN output
    components are byte-exact — while the (non-contractual, order/algorithm/platform-dependent)
    float NaN SIGN is not false-failed.

Standalone, like gen_npy_oracle.py / gen_decimal_oracle.cs: it owns its own case numbering and
writes ONLY nan.jsonl, so it never renumbers the shared gen_oracle.py corpus.

Run (needs numpy==2.4.2):  python test/oracle/gen_nan_oracle.py
"""
import json, os, struct, sys
import numpy as np

if np.__version__ != "2.4.2":
    sys.stderr.write(f"WARNING: numpy {np.__version__}, corpus is pinned to 2.4.2\n")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "NumSharp.Tests.Oracle", "Fuzz", "corpus", "nan.jsonl")

# ---- special-value grids ---------------------------------------------------------------------
NEG_NAN = struct.unpack('<d', bytes.fromhex('000000000000f8ff'))[0]   # .NET's double.NaN sign
POS_NAN = float('nan')                                                # NumPy's np.nan (0x7ff8...)
CX_COMPONENTS = [2.5, -1.0, 0.0, -0.0, np.inf, -np.inf, POS_NAN, NEG_NAN]        # 8 -> 64 (re,im)
FLOAT_VALS    = [2.5, -1.0, 0.5, 2.0, -2.0, 0.0, -0.0, np.inf, -np.inf, POS_NAN, NEG_NAN]

# Unary ops that route through NDComplexMath and are held BIT-EXACT on the complex NaN sign
# (must stay in sync with FuzzCorpusTests.ComplexNanContractOps).
COMPLEX_OPS = ["sqrt", "log", "log2", "log10", "log1p", "exp", "exp2", "expm1", "square",
               "reciprocal", "sin", "cos", "tan", "sinh", "cosh", "tanh",
               "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh",
               "conjugate", "negative", "positive", "sign", "abs"]
# Real-valued unary ops for the value-NaN (tokenized) check at each float width.
FLOAT_OPS = ["sqrt", "log", "log2", "log10", "log1p", "exp", "exp2", "expm1", "square",
             "reciprocal", "sin", "cos", "tan", "sinh", "cosh", "tanh",
             "arcsin", "arccos", "arctan", "arcsinh", "arccosh", "arctanh",
             "negative", "positive", "sign", "abs", "cbrt", "rint", "floor", "ceil", "trunc"]

def npf(op):
    return {"abs": np.abs, "conjugate": np.conjugate}.get(op, getattr(np, op))

DTYPE_NAME = {np.dtype('complex128'): "complex128", np.dtype('float64'): "float64",
              np.dtype('float32'): "float32", np.dtype('float16'): "float16"}

def operand(a):
    a = np.ascontiguousarray(a)
    isz = a.dtype.itemsize
    return {"dtype": DTYPE_NAME[a.dtype], "shape": list(a.shape),
            "strides": [s // isz for s in a.strides], "offset": 0,
            "bufferSize": int(a.size), "buffer": a.tobytes().hex()}

def case(cid, op, inp, out):
    return {"id": cid, "op": op, "params": {}, "operands": [operand(inp)],
            "expected": {"dtype": DTYPE_NAME[np.ascontiguousarray(out).dtype],
                         "shape": list(np.ascontiguousarray(out).shape),
                         "buffer": np.ascontiguousarray(out).tobytes().hex()},
            "layout": "nan_grid", "valueclass": "nan"}

cases = []
idx = 0

# --- complex128 unary: BIT-EXACT NaN sign (the star of this oracle) ---
grid_c = np.array([complex(re, im) for re in CX_COMPONENTS for im in CX_COMPONENTS],
                  dtype=np.complex128)
for op in COMPLEX_OPS:
    try:
        with np.errstate(all='ignore'):
            out = np.asarray(npf(op)(grid_c))
    except Exception as e:
        sys.stderr.write(f"skip complex {op}: {e}\n"); continue
    cases.append(case(f"nan/complex128/{op}/{idx}", op, grid_c, out)); idx += 1

# --- float64 / float32 / float16 unary: value-NaN (tokenized) + non-NaN byte-exact ---
for dt in (np.float64, np.float32, np.float16):
    grid_f = np.array(FLOAT_VALS, dtype=dt)
    for op in FLOAT_OPS:
        try:
            with np.errstate(all='ignore'):
                out = np.asarray(npf(op)(grid_f))
        except Exception as e:
            sys.stderr.write(f"skip {DTYPE_NAME[np.dtype(dt)]} {op}: {e}\n"); continue
        # only keep the op if it actually yields a NaN somewhere OR is a pure pass-through worth
        # pinning; drop cases with no NaN and no special value to avoid duplicating unary.jsonl.
        cases.append(case(f"nan/{DTYPE_NAME[np.dtype(dt)]}/{op}/{idx}", op, grid_f, out)); idx += 1

# --- BINARY ops over the full special-pair CROSS grid (coverage plan §B3) ---
# The pairwise value tiers align A[i] with B[i] drawn from the SAME pool in the SAME order, so
# the cross-operand NaN interactions (nan+finite, ±inf against BOTH NaN signs, 0*inf, inf-inf,
# 0/0, mod-by-0, copysign-from-NaN, maximum/fmax NaN priority) never occur there; the specials
# tier forces a curated subset. This grid is EXHAUSTIVE over FLOAT_VALS x FLOAT_VALS — both NaN
# signs included — with NumPy 2.4.2's exact bytes recorded. Float NaN stays tokenized (the sign
# is non-contractual) while every non-NaN slot — the sign of a produced zero/inf included — is
# byte-exact. The op sets are per-width: float16 keeps the widen-compute-narrow / bit-level ops
# whose f16 parity NumSharp pins bit-exactly; power/arctan2/hypot/logaddexp* ride the f32/f64
# CRT-libm lanes only (their f16 loops bridge differently — unit-suite territory, not this pin).
BIN_OPS_WIDE = ["add", "subtract", "multiply", "divide", "floor_divide", "mod", "fmod",
                "maximum", "minimum", "fmax", "fmin", "copysign", "heaviside", "nextafter",
                "power", "arctan2", "hypot", "logaddexp", "logaddexp2"]
BIN_OPS_F16 = ["add", "subtract", "multiply", "divide", "floor_divide", "mod", "fmod",
               "maximum", "minimum", "fmax", "fmin", "copysign", "heaviside"]

def case2(cid, op, a, b, out):
    return {"id": cid, "op": op, "params": {}, "operands": [operand(a), operand(b)],
            "expected": {"dtype": DTYPE_NAME[np.ascontiguousarray(out).dtype],
                         "shape": list(np.ascontiguousarray(out).shape),
                         "buffer": np.ascontiguousarray(out).tobytes().hex()},
            "layout": "nan_grid_pair", "valueclass": "nan"}

for dt in (np.float64, np.float32, np.float16):
    vals = np.array(FLOAT_VALS, dtype=dt)
    A = np.repeat(vals, len(vals))          # every (a, b) combination, both operand orders
    B = np.tile(vals, len(vals))
    ops = BIN_OPS_F16 if dt is np.float16 else BIN_OPS_WIDE
    for op in ops:
        try:
            with np.errstate(all='ignore'):
                out = np.asarray(npf(op)(A, B))
        except Exception as e:
            sys.stderr.write(f"skip binary {DTYPE_NAME[np.dtype(dt)]} {op}: {e}\n"); continue
        cases.append(case2(f"nan/{DTYPE_NAME[np.dtype(dt)]}/bin/{op}/{idx}", op, A, B, out)); idx += 1

# --- complex128 binary: the 64-element re x im grid against a rolled copy ---
# divide/multiply are BYTE-EXACT ports (Smith reciprocal-multiply / fused simd_cmul); add/
# subtract are pure IEEE; power's finite ULP envelope + NaN edges ride the registry's
# complex-binary excuse. NaN stays tokenized here (complex BINARY NaN sign is non-contractual —
# only the unary NDComplexMath family carries the sign pin, see ComplexNanContractOps).
zroll = np.roll(grid_c, 7)
for op in ["add", "subtract", "multiply", "divide", "power"]:
    try:
        with np.errstate(all='ignore'):
            out = np.asarray(npf(op)(grid_c, zroll))
    except Exception as e:
        sys.stderr.write(f"skip binary complex {op}: {e}\n"); continue
    cases.append(case2(f"nan/complex128/bin/{op}/{idx}", op, grid_c, zroll, out)); idx += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="\n") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")
print(f"wrote {len(cases)} cases -> {os.path.relpath(OUT)}")
