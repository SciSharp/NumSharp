"""Python reference for the TensorFlow.NET x NumSharp text probe.

Writes the probe's inputs and what Python TensorFlow (with NumPy 2.4.2) answers for them, so the C# probe can hold
TensorFlow.NET + NumSharp to the same answers:

* texts.json      - the shared input texts (JSON, ASCII-escaped, so no source-encoding question reaches C#)
* reference.json  - per text: TF's string ops (bytes as hex) and NumPy's np.strings answers
* conversions.json - how Python TF converts NumPy text arrays (U, S, StringDType with and without NA, object) and what
                     tensor.numpy() returns for a string tensor
* u.npy / s.npy / t.npy / tna.npy - NumPy-written text arrays, to test loading them in NumSharp and TF.NET

Run with the scratch venv's interpreter (TensorFlow 2.21, NumPy 2.4.2): python -I py_reference.py <out_dir>
"""
import json
import os
import sys
import traceback

import numpy as np
import tensorflow as tf

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)
T = np.dtypes.StringDType

# The inputs: ASCII, Latin-1, case-mapping special cases (dotted capital I, final sigma, sharp s, titlecase digraph,
# ligature, n-apostrophe), CJK, an astral emoji, an embedded NUL, a long (arena / TF large) text, and the empty string.
texts = [
    "",
    "a",
    "Hello, World",
    "ÀÉÎÕÜ",
    "İstanbul",
    "ΣΟΦΟΣ ΚΑΙ ΣΟΦΟΣ",
    "straße",
    "ǅ",
    "a\x00b",
    "ﬁ ligature",
    "日本語のテキスト",
    "emoji \U0001F600 test",
    "x" * 40,
    "MiXeD 123 ÄÖÜ",
    "ŉ",
]
with open(os.path.join(out_dir, "texts.json"), "w", encoding="ascii") as f:
    json.dump(texts, f, ensure_ascii=True)


def hexes(t):
    """Every element of a string tensor as the hex of its bytes (row-major)."""
    return [b.hex() for b in np.asarray(t.numpy(), dtype=object).ravel().tolist()]


u = np.array(texts)
tt = tf.constant(u)
ref = {
    "tf_version": tf.__version__,
    "numpy_version": np.__version__,
    "tf_length_byte": tf.strings.length(tt).numpy().tolist(),
    "tf_length_utf8": tf.strings.length(tt, unit="UTF8_CHAR").numpy().tolist(),
    "tf_lower_ascii": hexes(tf.strings.lower(tt)),
    "tf_lower_utf8": hexes(tf.strings.lower(tt, encoding="utf-8")),
    "tf_upper_utf8": hexes(tf.strings.upper(tt, encoding="utf-8")),
    "tf_substr_utf8_0_2": hexes(tf.strings.substr(tt, 0, 2, unit="UTF8_CHAR")),
    "tf_substr_byte_0_1": hexes(tf.strings.substr(tt, 0, 1)),
    "tf_regex_replace": hexes(tf.strings.regex_replace(tt, "[aeiou]", "_")),
    "np_str_len": np.strings.str_len(u).tolist(),
    "np_lower": [s.encode("utf-8").hex() for s in np.strings.lower(u).tolist()],
    "np_upper": [s.encode("utf-8").hex() for s in np.strings.upper(u).tolist()],
    "np_slice_0_2": [s.encode("utf-8").hex() for s in np.strings.slice(u, 0, 2).tolist()],
}
with open(os.path.join(out_dir, "reference.json"), "w", encoding="ascii") as f:
    json.dump(ref, f, ensure_ascii=True, indent=1)


def probe(name, make):
    """Run one conversion and record its outcome: the tensor's dtype, shape and element bytes, or the exception."""
    try:
        value = make()
        if isinstance(value, tf.Tensor):
            # A 0-d string tensor's numpy() is a bare bytes object, not an array: record what it IS, then view it as one.
            raw = value.numpy()
            arr = raw if isinstance(raw, np.ndarray) else np.asarray(raw, dtype=object)
            return {"case": name, "ok": True, "dtype": value.dtype.name, "shape": list(value.shape),
                    "numpy_type": type(raw).__name__, "numpy_dtype": str(raw.dtype) if isinstance(raw, np.ndarray) else None,
                    "numpy_elem_type": type(arr.ravel()[0]).__name__ if arr.size else None,
                    "elements_hex": hexes(value)}
        return {"case": name, "ok": True, "repr": repr(value)}
    except Exception as e:  # noqa: BLE001 - the exception IS the answer being recorded
        return {"case": name, "ok": False, "error": type(e).__name__, "message": str(e).splitlines()[0][:300]}


small = ["ab", "é", ""]
conv = [
    probe("U array", lambda: tf.constant(np.array(small))),
    probe("S array", lambda: tf.constant(np.array([b"ab", b"\xff\x00", b"tail\x00"]))),
    probe("S array with an embedded NUL", lambda: tf.constant(np.array([b"a\x00b"]))),
    probe("StringDType array", lambda: tf.constant(np.array(small, dtype=T()))),
    probe("StringDType array, na_object=None, with an NA", lambda: tf.constant(np.array(["ab", None], dtype=T(na_object=None)))),
    probe("StringDType convert_to_tensor", lambda: tf.convert_to_tensor(np.array(small, dtype=T()))),
    probe("object array of str", lambda: tf.constant(np.array(small, dtype=object))),
    probe("object array of bytes", lambda: tf.constant(np.array([b"ab", b"\xff"], dtype=object))),
    probe("2-D U array", lambda: tf.constant(np.array([["a", "bb"], ["ccc", "é"]]))),
    probe("transposed U view", lambda: tf.constant(np.array([["a", "bb"], ["ccc", "é"]]).T)),
    probe("0-d U array", lambda: tf.constant(np.array("scalar"))),
    probe("empty U array", lambda: tf.constant(np.array([], dtype="U3"))),
    probe("U array with a lone surrogate", lambda: tf.constant(np.array(["\ud800"]))),
    probe("tensor.numpy() of a string tensor", lambda: repr(tf.constant(["ab", "é"]).numpy())),
    probe("tensor.numpy().astype(StringDType())", lambda: repr(tf.constant(["ab", "é"]).numpy().astype(T()))),
    probe("tensor.numpy().astype('U')", lambda: repr(tf.constant(["ab", "é"]).numpy().astype("U"))),
    probe("tensor.numpy().astype('S')", lambda: repr(tf.constant(["ab", "é"]).numpy().astype("S"))),
    probe("np.strings.decode(tensor.numpy().astype('S'))", lambda: repr(np.strings.decode(tf.constant(["ab", "é"]).numpy().astype("S")))),
    probe("invalid UTF-8 out of a BYTE substr", lambda: tf.strings.substr(tf.constant(["été"]), 0, 1)),
    probe("invalid UTF-8 to .numpy().astype(T())", lambda: repr(tf.strings.substr(tf.constant(["été"]), 0, 1).numpy().astype(T()))),
    probe("trailing NUL from TF to .astype('S')", lambda: repr(tf.constant([b"ab\x00"]).numpy().astype("S"))),
]
with open(os.path.join(out_dir, "conversions.json"), "w", encoding="ascii") as f:
    json.dump(conv, f, ensure_ascii=True, indent=1)

# NumPy-written text arrays for the .npy check: U and S are native .npy dtypes; StringDType is written through a pickle.
np.save(os.path.join(out_dir, "u.npy"), np.array([["alpha", "été"], ["日本", "z"]]))
np.save(os.path.join(out_dir, "s.npy"), np.array([b"cat", b"\xff\xfe", b"x"]))
np.save(os.path.join(out_dir, "t.npy"), np.array(["short", "a text longer than fifteen bytes", "é"], dtype=T()), allow_pickle=True)
np.save(os.path.join(out_dir, "tna.npy"), np.array(["x", None], dtype=T(na_object=None)), allow_pickle=True)

for c in conv:
    print(json.dumps(c, ensure_ascii=True)[:400])
print("wrote", out_dir)
