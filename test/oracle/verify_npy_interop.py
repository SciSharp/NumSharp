"""
verify_npy_interop.py — prove real NumPy can read what NumSharp writes.

The committed oracle (gen_npy_oracle.py + NpyOracleTests) proves the other direction: NumSharp reads
NumPy's files and reproduces NumPy's bytes exactly. That covers .npy completely — byte-equality is
the strongest claim available — but NOT .npz, whose ZIP framing (timestamps, deflate parameters,
Zip64 records) legitimately differs between Python's zipfile and .NET's ZipArchive. A NumSharp .npz
only has to be a *valid* archive of valid members, so the way to prove it is to have NumPy open it.

This runs NumSharp (via `dotnet run`), then reads every file it produced with NumPy and checks the
dtype, shape, layout and values. It needs both Python and the .NET SDK, so it is a developer/manual
gate rather than a CI test — the same role the nightly fuzz soak plays.

    python test/oracle/verify_npy_interop.py [--keep]
"""
import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WRITER = os.path.join(HERE, "interop_write.cs")

# NPTypeCode -> the dtype NumPy must report. Char rides '<U1'; Complex is always written as '<c16'.
EXPECT_DTYPE = {
    "Boolean": np.dtype("bool"), "Byte": np.dtype("uint8"), "SByte": np.dtype("int8"),
    "Int16": np.dtype("int16"), "UInt16": np.dtype("uint16"), "Int32": np.dtype("int32"),
    "UInt32": np.dtype("uint32"), "Int64": np.dtype("int64"), "UInt64": np.dtype("uint64"),
    "Half": np.dtype("float16"), "Single": np.dtype("float32"), "Double": np.dtype("float64"),
    "Complex": np.dtype("complex128"), "Char": np.dtype("<U1"),
}

failures = []
checked = 0


def check(cond, case, msg):
    global checked
    checked += 1
    if not cond:
        failures.append(f"{case}: {msg}")


def refuses_without_pickle(path):
    """True when NumPy refuses to load the file without allow_pickle, with its own ValueError text."""
    try:
        np.load(path)
    except ValueError as e:
        return "Object arrays cannot be loaded when allow_pickle=False" in str(e)
    return False


def verify_array(name, arr, meta):
    if meta.get("pickle"):
        # A StringDType array: np.dtype() cannot parse 'StringDType(na_object=None)', so compare str(), which is how
        # NumPy names the NA object and coerce (NumSharp's dtype.str is the same text).
        check(str(arr.dtype) == meta["descr"], name, f"dtype {arr.dtype} != {meta['descr']}")
    else:
        # A fixed-width string array names its own descriptor ('<U3', '|S5'); every other dtype its lane.
        want = np.dtype(meta["descr"]) if meta["dtype"] == "String" else EXPECT_DTYPE[meta["dtype"]]
        check(arr.dtype == want, name, f"dtype {arr.dtype} != {want}")
    check(list(arr.shape) == meta["shape"], name, f"shape {list(arr.shape)} != {meta['shape']}")


def main():
    keep = "--keep" in sys.argv
    out = tempfile.mkdtemp(prefix="numsharp_interop_")

    print(f"running NumSharp writer -> {out}")
    r = subprocess.run(
        ["dotnet", "run", "-c", "Release", WRITER, "--", out],
        cwd=REPO, capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"NumSharp writer failed (exit {r.returncode})")
    print("  " + r.stdout.strip().splitlines()[-1])

    manifest = json.load(open(os.path.join(out, "manifest.json")))

    for case in manifest:
        path = os.path.join(out, case["file"])
        name = case["file"]

        if case["kind"] == "npy":
            # allow_pickle stays False for every file but a StringDType array's, which NumPy itself writes as a pickle:
            # a numeric or fixed-width file that needed it would be a NumSharp bug.
            if case.get("pickle"):
                check(refuses_without_pickle(path), name, "NumPy loaded a pickled file without allow_pickle")
            arr = np.load(path, allow_pickle=bool(case.get("pickle")))
            verify_array(name, arr, case)

            # The layout claims must survive: an F-order file is F-contiguous once NumPy reads it.
            if name in ("fortran.npy", "transposed.npy"):
                check(arr.flags.f_contiguous and not arr.flags.c_contiguous, name,
                      "expected NumPy to read this back as F-contiguous")
            if name in ("2d.npy", "3d.npy", "strided.npy", "sliced_2d.npy"):
                check(arr.flags.c_contiguous, name, "expected C-contiguous")

            # Spot-check values NumSharp claimed to write.
            if name == "int64.npy":
                check(list(arr) == [np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max], name, f"values {arr}")
            elif name == "uint64.npy":
                check(list(arr) == [0, 1, np.iinfo(np.uint64).max], name, f"values {arr}")
            elif name == "float64.npy":
                check(arr[0] == 1.5 and np.signbit(arr[1]) and arr[1] == 0.0 and arr[2] == np.finfo(np.float64).max,
                      name, f"values {arr} (note -0.0 must keep its sign bit)")
            elif name == "float32.npy":
                check(arr[0] == 1.5 and np.isnan(arr[1]) and arr[2] == -np.inf, name, f"values {arr}")
            elif name == "float16.npy":
                check(arr[0] == 1.5 and arr[1] == -2.25 and arr[2] == np.inf, name, f"values {arr}")
            elif name == "complex128.npy":
                check(arr[0] == 1 + 2j and arr[1].real == -3 and np.isnan(arr[1].imag), name, f"values {arr}")
            elif name == "char.npy":
                check(list(arr) == ["a", "Z", "é"], name, f"values {list(arr)} — NumSharp Char must widen to UCS-4")
            elif name == "str_u.npy":
                check(list(arr) == ["ab", "", "dé", "😀x"], name, f"values {list(arr)}")
            elif name == "str_s.npy":
                check(list(arr) == [b"ab", b"c"], name, f"values {list(arr)}")
            elif name == "str_u_fortran.npy":
                check(arr.flags.f_contiguous and not arr.flags.c_contiguous, name, "expected F-contiguous")
                check(arr.tolist() == [["a", "bb"], ["ccc", "d"]], name, f"values {arr.tolist()}")
            elif name == "str_s_strided.npy":
                check(list(arr) == [b"a", b"c"], name, f"values {list(arr)}")
            elif name == "scalar.npy":
                check(arr.shape == () and arr[()] == 42, name, f"0-d scalar, got {arr!r}")
            elif name == "fortran.npy":
                check(np.array_equal(arr, np.arange(6).reshape(2, 3)), name, f"values {arr}")
            elif name == "transposed.npy":
                check(np.array_equal(arr, np.arange(12).reshape(3, 4).T), name, f"values {arr}")
            elif name == "strided.npy":
                check(np.array_equal(arr, np.arange(12)[::2]), name, f"values {arr}")
            elif name == "sliced_2d.npy":
                check(np.array_equal(arr, np.arange(12).reshape(3, 4)[1:, ::2]), name, f"values {arr}")
            elif name == "large.npy":
                check(np.array_equal(arr, np.arange(100_000, dtype=np.float64)), name, "100k values differ")
            elif name == "2d.npy":
                check(np.array_equal(arr, np.arange(6).reshape(2, 3)), name, f"values {arr}")
            elif name == "vstr.npy":
                check(arr.tolist() == ["cat", "a much longer text over fifteen bytes", "dé😀", ""], name,
                      f"values {arr.tolist()}")
            elif name == "vstr_na.npy":
                check(arr.tolist() == ["x", None, "y"] and arr.dtype.na_object is None, name,
                      f"values {arr.tolist()}, na_object {arr.dtype.na_object!r}")
            elif name == "vstr_na_nan.npy":
                v = arr.tolist()
                check(v[0] == "x" and isinstance(v[1], float) and np.isnan(v[1]) and np.isnan(arr.dtype.na_object),
                      name, f"values {v}, na_object {arr.dtype.na_object!r}")
            elif name == "vstr_na_str.npy":
                check(arr.tolist() == ["a", "NA"] and arr.dtype.na_object == "NA" and arr.dtype.coerce is False,
                      name, f"values {arr.tolist()}, dtype {arr.dtype!r}")
            elif name == "vstr_fortran.npy":
                check(arr.flags.f_contiguous and not arr.flags.c_contiguous, name, "expected F-contiguous")
                check(arr.tolist() == [["a", "bb long text over the fifteen"], ["ccc", "d"]], name,
                      f"values {arr.tolist()}")
            elif name == "vstr_strided.npy":
                check(arr.flags.c_contiguous and arr.tolist() == ["a", "c"], name, f"values {arr.tolist()}")
            elif name == "vstr_large.npy":
                check(arr.tolist() == [f"text number {i}" for i in range(100_000)], name, "100k texts differ")

            # The header must be exactly what NumPy's own writer would emit for this array (for a StringDType array:
            # NumPy's pickle of it, byte for byte).
            with open(path, "rb") as fh:
                raw = fh.read()
            import io
            mine = io.BytesIO()
            with warnings.catch_warnings():
                # NumPy warns that a custom dtype is saved as a pickle; the warning is not part of the bytes.
                warnings.simplefilter("ignore", UserWarning)
                np.lib.format.write_array(mine, arr)
            check(mine.getvalue() == raw, name,
                  "file is not byte-identical to NumPy's own np.save output for the same array")

        else:  # npz
            # A member that is a StringDType array is a pickle inside the archive; only then may NumPy unpickle.
            pickled = any(meta.get("pickle") for meta in case["entries"].values())
            with np.load(path, allow_pickle=pickled) as z:
                check(sorted(z.files) == sorted(case["entries"]), name,
                      f".files {sorted(z.files)} != {sorted(case['entries'])}")
                for key, meta in case["entries"].items():
                    verify_array(f"{name}[{key}]", z[key], meta)

                if name == "npz_multi.npz":
                    check(np.array_equal(z["weights"], np.arange(12).reshape(3, 4).astype(np.float32)),
                          name, "weights differ")
                    check(list(z["biases"]) == [0.5, 1.5], name, "biases differ")
                    check(bool(z["flag"]) is True, name, "flag differs")
                    check(list(z["text"]) == ["h", "i"], name, "text differs")
                    check(list(z["words"]) == ["hello", "wörld"], name, "words differ")
                    check(list(z["blobs"]) == [b"x", b"yz"], name, "blobs differ")
                elif name == "npz_fortran.npz":
                    f = z["f"]
                    check(f.flags.f_contiguous and not f.flags.c_contiguous, name,
                          "F-order must survive inside an npz")
                    check(np.array_equal(f, np.arange(6).reshape(2, 3)), name, "values differ")
                elif name == "npz_compressed.npz":
                    check(np.array_equal(z["big"], np.zeros(50_000)), name, "values differ")
                elif name == "npz_vstr.npz":
                    check(z["names"].tolist() == ["cat", None, "a much longer text over fifteen bytes"], name,
                          f"names {z['names'].tolist()}")
                    check(list(z["nums"]) == [1.5, 2.5], name, "nums differ")

    print(f"\n{len(manifest)} files, {checked} checks")
    if failures:
        print(f"\n{len(failures)} FAILURE(S):")
        for f in failures:
            print("  " + f)
        raise SystemExit(1)

    print("PASS — NumPy reads every file NumSharp wrote, and every .npy is byte-identical to NumPy's own output.")
    if keep:
        print(f"files kept at {out}")
    else:
        import shutil
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    main()
