"""
Record what PyTorch and NumPy do with every array / tensor STATE the NumSharp.Interop.TorchSharp suite crosses.

The bridge's own suite (test/NumSharp.Tests.Interop.TorchSharp, StateMatrix*Tests) rebuilds each state BY NAME with
NumSharp and TorchSharp - the same recipe as here - and holds the bridge to these answers:

  numpy_states  (NumSharp -> torch, nd.AsTensor): for each layout x dtype, the NumPy array's shape, element strides and
                flags (C, F, OWNDATA, WRITEABLE, ALIGNED) - which NumSharp's own array must reproduce - and what
                torch.from_numpy does with it: the tensor's strides and is_contiguous() and whether it shares the
                array's memory, or the error / warning it raises.
  torch_states  (torch -> NumSharp, tensor.AsNDArray): for each tensor layout x dtype, the tensor's shape, strides,
                storage_offset and is_contiguous(), and what t.numpy() returns (flags, element strides, whether it
                shares the tensor's memory) or raises, plus t.numpy(force=True) for the states numpy() refuses.

The bridge differs from torch.from_numpy / t.numpy() ON PURPOSE in a few places (a read-only, misaligned or reversed
array copies under Auto and throws under View; an expanded tensor is a read-only NumSharp view; a tensor that requires
grad is a read-only view; complex64 / bfloat16 widen); the C# suite names each divergence where it asserts it.

    <python with the pinned numpy and torch> test/oracle/gen_torch_interop_states.py
        # writes test/NumSharp.Tests.Interop.TorchSharp/Fixtures/torch_interop_states.json
        # e.g. the interop ecosystem venv test/NumSharp.Tests.Interop/python-envs/make_env.py builds (numpy + torch-cpu)

Requires: numpy - exactly the release the repository pins (numpy-version.txt, read through numpy_pin.py; the script
exits under any other) - and torch (2.3 or later, for the unsigned dtypes). Run it WITHOUT `python -I`: isolated mode
drops this folder from sys.path, so `import numpy_pin` fails. The fixture is COMMITTED, so the C# suite needs no Python
at test time; StateCatalogSyncTests.Fixture_IsThePinnedNumPy fails until the fixture is regenerated after a NumPy bump.
Values are not recorded: the C# suite compares the two sides' own copies.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import numpy_pin  # noqa: E402  (test/oracle/numpy_pin.py: the one NumPy pin)
import torch

# A fixture recorded under another NumPy would hold the bridge to that release's flags; refuse before writing a byte.
numpy_pin.require(np, "gen_torch_interop_states.py")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "NumSharp.Tests.Interop.TorchSharp", "Fixtures", "torch_interop_states.json")

# The ten dtypes NumSharp and torch share bit for bit, by NumPy name (the C# suite maps the names to NPTypeCode).
SHARED = ["bool", "uint8", "int8", "int16", "int32", "int64", "float16", "float32", "float64", "complex128"]


def ramp(shape, dt):
    """A fresh C-contiguous array of `shape` and `dt` (StateCatalog.Ramp in the C# suite builds the same layout).

    Args:
        shape: the dimensions (an empty tuple for 0-d).
        dt: a NumPy dtype name from SHARED; bool alternates 0 / 1, complex128 carries a non-zero imaginary part.

    Returns:
        A new array that owns its memory.
    """
    n = int(np.prod(shape)) if len(shape) else 1
    a = np.arange(n)
    if dt == "bool":
        a = a % 2
    elif dt == "complex128":
        a = a + 1j * (a + 0.5)
    return np.ascontiguousarray(a.reshape(shape).astype(dt))


def readonly(a):
    """Clears WRITEABLE on `a` (NumSharp's setflags(write: false)).

    Args:
        a: the array to mark read-only.

    Returns:
        The same array, now read-only.
    """
    a.flags.writeable = False
    return a


def unaligned_flag(a):
    """Clears ALIGNED on `a` although its memory is aligned (NumSharp's setflags(align: false)).

    Args:
        a: the array whose flag to clear.

    Returns:
        The same array, now reporting ALIGNED false.
    """
    a.setflags(align=False)
    return a


def misaligned(dt):
    """Four elements of `dt` starting one byte into a fresh uint8 buffer (NumPy's ALIGNED is False there).

    Args:
        dt: a NumPy dtype name wider than one byte.

    Returns:
        A writeable, contiguous view whose data address is odd.
    """
    size = np.dtype(dt).itemsize
    raw = np.zeros(8 * size + 1, dtype=np.uint8)
    return raw[1:1 + 4 * size].view(dt)


def overlapping(dt):
    """Windows of 4 over a 10-element ramp, each one element after the previous (NumPy's as_strided, writeable).

    Args:
        dt: a NumPy dtype name from SHARED.

    Returns:
        A (3, 4) view whose rows overlap (element strides (1, 1)).
    """
    b = ramp((10,), dt)
    size = b.itemsize
    return np.lib.stride_tricks.as_strided(b, shape=(3, 4), strides=(size, size))


# name -> (recipe(dtype) -> ndarray, dtypes it applies to). Mirrored by StateCatalog.NDArrayStates in the C# suite.
NUMPY_STATES = {
    "c_1d": (lambda dt: ramp((7,), dt), SHARED),
    "c_2d": (lambda dt: ramp((4, 5), dt), SHARED),
    "c_3d": (lambda dt: ramp((2, 3, 4), dt), SHARED),
    "f_2d_owner": (lambda dt: np.asfortranarray(ramp((4, 5), dt)), SHARED),
    "f_3d_view": (lambda dt: ramp((2, 3, 4), dt).transpose(2, 1, 0), SHARED),
    "transposed_2d": (lambda dt: ramp((4, 5), dt).T, SHARED),
    "transposed_3d": (lambda dt: ramp((2, 3, 4), dt).transpose(2, 0, 1), SHARED),
    "offset_1d": (lambda dt: ramp((10,), dt)[2:7], SHARED),
    "stepped_1d": (lambda dt: ramp((10,), dt)[::2], SHARED),
    "stepped_cols_2d": (lambda dt: ramp((4, 6), dt)[:, ::2], SHARED),
    "stepped_rows_2d": (lambda dt: ramp((6, 4), dt)[::2, :], SHARED),
    "stepped_offset_3d": (lambda dt: ramp((4, 5, 6), dt)[1:, 1::2, 2:], SHARED),
    "sliced_composed": (lambda dt: ramp((6, 4), dt)[1:5].T, SHARED),
    "negstride_1d": (lambda dt: ramp((8,), dt)[::-1], SHARED),
    "negstride_2d": (lambda dt: ramp((4, 5), dt)[::-1, ::-1], SHARED),
    "negstride_inner_2d": (lambda dt: ramp((4, 5), dt)[:, ::-1], SHARED),
    "negstride_len1_axis": (lambda dt: ramp((3, 1, 4), dt)[:, ::-1, :], SHARED),
    "singleton_3d": (lambda dt: ramp((3, 1, 4), dt), SHARED),
    "newaxis": (lambda dt: ramp((5,), dt)[None, :], SHARED),
    "reshape_view": (lambda dt: ramp((24,), dt).reshape(4, 6), SHARED),
    "highrank_8d": (lambda dt: ramp((2,) * 8, dt), SHARED),
    "highrank_8d_permuted": (lambda dt: ramp((2,) * 8, dt).transpose(1, 0, 3, 2, 5, 4, 7, 6), SHARED),
    "zerod_owner": (lambda dt: np.zeros((), dtype=dt), SHARED),
    "zerod_view": (lambda dt: ramp((24,), dt)[23:24].reshape(()), SHARED),
    "one_element_1d": (lambda dt: ramp((1,), dt), SHARED),
    "empty_2d": (lambda dt: np.zeros((0, 3), dtype=dt), SHARED),
    "empty_view": (lambda dt: ramp((4, 6), dt)[2:2], SHARED),
    "empty_composed": (lambda dt: np.zeros((0, 4), dtype=dt)[::2, :], SHARED),
    "broadcast_row": (lambda dt: np.broadcast_to(ramp((5,), dt), (4, 5)), SHARED),
    "broadcast_col": (lambda dt: np.broadcast_to(ramp((4,), dt)[:, None], (4, 5)), SHARED),
    "broadcast_strided_base": (lambda dt: np.broadcast_to(ramp((10,), dt)[::2], (3, 5)), SHARED),
    "broadcast_of_readonly": (lambda dt: np.broadcast_to(readonly(ramp((6,), dt)), (2, 6)), SHARED),
    "readonly_owner": (lambda dt: readonly(ramp((6,), dt)), SHARED),
    "readonly_view": (lambda dt: readonly(ramp((6,), dt))[1:], SHARED),
    "aligned_flag_cleared": (lambda dt: unaligned_flag(ramp((6,), dt)), SHARED),
    "misaligned": (lambda dt: misaligned(dt), [d for d in SHARED if np.dtype(d).itemsize > 1]),
    "overlapping_windows": (lambda dt: overlapping(dt), SHARED),
    "matrix": (lambda dt: np.asmatrix(ramp((2, 3), dt)), ["float64", "int32"]),
    "matrix_T": (lambda dt: np.asmatrix(ramp((2, 3), dt)).T, ["float64", "int32"]),
    "frombuffer": (lambda dt: np.frombuffer(bytearray(6 * np.dtype(dt).itemsize), dtype=dt), SHARED),
    # dtype-specific views
    "complex_real": (lambda dt: ramp((6,), "complex128").real, ["float64"]),
    "complex_imag": (lambda dt: ramp((6,), "complex128").imag, ["float64"]),
    "complex_8mod16": (lambda dt: ramp((10,), "float64")[1:9].view(np.complex128), ["complex128"]),
    "view_int64_of_float64": (lambda dt: ramp((6,), "float64").view(np.int64), ["int64"]),
    "view_uint8_of_float64": (lambda dt: ramp((3,), "float64").view(np.uint8), ["uint8"]),
}


def address(a):
    """The address of an array's first element.

    Args:
        a: the array.

    Returns:
        The data pointer NumPy reports (an int).
    """
    return a.__array_interface__["data"][0]


def numpy_record(a):
    """What the C# suite compares a NumSharp array with: shape, element strides and the five NumPy flags.

    Args:
        a: the NumPy array.

    Returns:
        A JSON-ready dict.
    """
    size = a.itemsize
    return {
        "shape": list(a.shape),
        "strides": [s // size for s in a.strides],
        "flags": {"C": bool(a.flags.c_contiguous), "F": bool(a.flags.f_contiguous), "O": bool(a.flags.owndata),
                  "W": bool(a.flags.writeable), "A": bool(a.flags.aligned)},
    }


def from_numpy_record(a):
    """What torch.from_numpy does with `a`: the tensor's dtype, shape, element strides, contiguity and sharing, or the
    error it raises. Its "not writable" warning is not recorded: PyTorch emits it once per process.

    Args:
        a: the NumPy array.

    Returns:
        A JSON-ready dict with "ok" False and the error, or the tensor's layout.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            t = torch.from_numpy(a)
        except Exception as ex:  # noqa: BLE001 - every refusal is an answer to record
            return {"ok": False, "error": type(ex).__name__, "message": str(ex).splitlines()[0]}
    return {
        "ok": True,
        "dtype": str(t.dtype).replace("torch.", ""),
        "shape": list(t.shape),
        "strides": list(t.stride()),
        "is_contiguous": bool(t.is_contiguous()),
        "shares": bool(a.size > 0 and t.data_ptr() == address(a)),
    }


def tramp(shape, dt):
    """A fresh contiguous tensor of `shape` and `dt` (StateCatalog.TorchRamp in the C# suite builds the same layout).

    Args:
        shape: the dimensions (an empty tuple for 0-d).
        dt: a torch dtype; bool alternates 0 / 1, complex carries a non-zero imaginary part.

    Returns:
        A new tensor that owns its storage.
    """
    n = 1
    for d in shape:
        n *= d
    t = torch.arange(n, dtype=torch.float64)
    if dt == torch.bool:
        t = t.remainder(2)
    if dt in (torch.complex64, torch.complex128):
        t = torch.complex(t, t + 0.5)
    return t.reshape(shape).to(dt).contiguous()


T_SHARED = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32,
            torch.float64, torch.complex128]
T_FLOAT = [torch.float32, torch.float64]
T_COMPLEX = [torch.complex64, torch.complex128]
T_EXTRA = [torch.bfloat16, torch.complex64, torch.uint16, torch.uint32, torch.uint64]


def requires_grad(t):
    """Turns gradient tracking on for a leaf tensor.

    Args:
        t: a floating-point tensor.

    Returns:
        The same tensor, now requiring grad.
    """
    return t.requires_grad_()


# name -> (recipe(dtype) -> tensor, dtypes). Mirrored by StateCatalog.TensorStates in the C# suite.
TORCH_STATES = {
    "contiguous_3d": (lambda dt: tramp((2, 3, 4), dt), T_SHARED + T_EXTRA),
    "transposed_2d": (lambda dt: tramp((4, 5), dt).t(), T_SHARED),
    "permuted_3d": (lambda dt: tramp((2, 3, 4), dt).permute(2, 0, 1), T_SHARED),
    "narrowed": (lambda dt: tramp((4, 6), dt).narrow(1, 1, 3), T_SHARED),
    "stepped": (lambda dt: tramp((4, 6), dt)[:, ::2], T_SHARED),
    "stepped_offset_3d": (lambda dt: tramp((4, 5, 6), dt)[1:, 1::2, 2:], T_SHARED),
    "select": (lambda dt: tramp((3, 4), dt)[1], T_SHARED),
    "zerod_select": (lambda dt: tramp((3, 4), dt)[1, 2], T_SHARED),
    "zerod_tensor": (lambda dt: tramp((), dt), T_SHARED),
    "one_element": (lambda dt: tramp((1,), dt), T_SHARED),
    "unsqueezed": (lambda dt: tramp((5,), dt).unsqueeze(0), T_SHARED),
    "expanded_row": (lambda dt: tramp((5,), dt).expand(4, 5), T_SHARED),
    "expanded_col": (lambda dt: tramp((4,), dt).unsqueeze(1).expand(4, 5), T_SHARED),
    "overlapping_windows": (lambda dt: tramp((10,), dt).as_strided((3, 4), (1, 1), 0), T_SHARED),
    "channels_last": (lambda dt: tramp((2, 3, 4, 5), dt).permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2), T_SHARED),
    "size1_odd_stride": (lambda dt: tramp((2, 3), dt).unsqueeze(1).as_strided((2, 1, 3), (3, 100, 1), 0), T_SHARED),
    "highrank_8d_permuted": (lambda dt: tramp((2,) * 8, dt).permute(1, 0, 3, 2, 5, 4, 7, 6), T_SHARED),
    "empty_3d": (lambda dt: tramp((2, 0, 3), dt), T_SHARED),
    "empty_view": (lambda dt: tramp((4, 6), dt)[:, 3:3], T_SHARED),
    "requires_grad_leaf": (lambda dt: requires_grad(tramp((2, 3), dt)), T_FLOAT),
    "requires_grad_nonleaf": (lambda dt: requires_grad(tramp((2, 3), dt)) * 2, T_FLOAT),
    "conj": (lambda dt: tramp((2, 3), dt).conj(), T_COMPLEX),
    "conj_imag_neg": (lambda dt: tramp((2, 3), dt).conj().imag, T_COMPLEX),
    "transposed_widened": (lambda dt: tramp((3, 4), dt).t(), [torch.bfloat16, torch.complex64]),
}


def tensor_record(t):
    """The tensor's own layout, which TorchSharp must reproduce for the same recipe.

    Args:
        t: the tensor.

    Returns:
        A JSON-ready dict: dtype, shape, strides, storage_offset, is_contiguous, requires_grad.
    """
    return {
        "dtype": str(t.dtype).replace("torch.", ""),
        "shape": list(t.shape),
        "strides": list(t.stride()),
        "storage_offset": int(t.storage_offset()),
        "is_contiguous": bool(t.is_contiguous()),
        "requires_grad": bool(t.requires_grad),
    }


def to_numpy_record(t, force):
    """What t.numpy(force=force) returns (flags, element strides, dtype, sharing) or raises.

    Args:
        t: the tensor.
        force: PyTorch's force flag (detach, cpu, resolve_conj, resolve_neg first).

    Returns:
        A JSON-ready dict with "ok" False and the error, or the array's record.
    """
    try:
        a = t.numpy(force=force)
    except Exception as ex:  # noqa: BLE001 - every refusal is an answer to record
        return {"ok": False, "error": type(ex).__name__, "message": str(ex).splitlines()[0]}
    rec = numpy_record(a)
    rec["ok"] = True
    rec["dtype"] = str(a.dtype)
    rec["shares"] = bool(a.size > 0 and t.numel() > 0 and address(a) == t.data_ptr())
    return rec


def main():
    """Records every state of both catalogues and writes the fixture.

    Returns:
        0 (the process exit code).
    """
    out = {"numpy": np.__version__, "torch": torch.__version__, "numpy_states": {}, "torch_states": {}}
    for name, (recipe, dtypes) in NUMPY_STATES.items():
        rows = {}
        for dt in dtypes:
            a = recipe(dt)
            rows[dt] = {"numpy": numpy_record(a), "from_numpy": from_numpy_record(a)}
        out["numpy_states"][name] = rows
    for name, (recipe, dtypes) in TORCH_STATES.items():
        rows = {}
        for dt in dtypes:
            t = recipe(dt)
            row = {"tensor": tensor_record(t), "numpy": to_numpy_record(t, force=False)}
            if not row["numpy"]["ok"]:
                row["numpy_force"] = to_numpy_record(t, force=True)
            rows[str(dt).replace("torch.", "")] = row
        out["torch_states"][name] = rows
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    n_np = sum(len(v) for v in out["numpy_states"].values())
    n_t = sum(len(v) for v in out["torch_states"].values())
    print(f"numpy {np.__version__}, torch {torch.__version__}: {n_np} array cases, {n_t} tensor cases -> {os.path.normpath(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
