# `ToNDArrayView` built a zero-copy view over an UNVALIDATED `__array_interface__` — memory-unsafe where numpy rejects

**Status:** FIXED on branch `experimental` (2026-08-13). `ViewViaArrayInterface` now performs the
same three validations numpy performs before trusting an `__array_interface__`, so a hostile or buggy
exporter is refused with a `NotSupportedException` instead of yielding a view over an invalid address.

**Found:** 2026-08-13, red-teaming the interop package from a packaged (`.nupkg`) consumer.
**Observed on:** pythonnet 3.0.5 · CPython 3.12.12 · numpy 2.4.2 · net8.0 · Windows 11.

---

## Overview

`NDArrayPythonInterop.ToNDArrayView` imports a non-contiguous numpy array (and any object exposing the
NumPy array-interface protocol) through `ViewViaArrayInterface`, which reads
`obj.__array_interface__['data'][0]` (a raw integer address), `['shape']` and `['strides']` and builds
a zero-copy NumSharp view directly over them. It performed **none** of the consistency checks numpy
performs on the same dict, so three classes of malformed interface — each of which **numpy rejects with
a `ValueError`** — produced a NumSharp view over invalid memory:

| Malformed `__array_interface__` | numpy | NumSharp (before) |
|---|---|---|
| `'data': (0, False)` (NULL pointer, non-empty array) | `ValueError: data is NULL but array contains data` | **builds a view over address 0** → read faults deep in `UnmanagedStorage.GetDouble` |
| `'shape': (-2,)` (negative dimension) | `ValueError: negative dimensions are not allowed` | **builds a view with a negative extent** (flows to a negative buffer count) |
| `'strides'` longer than `'shape'` (e.g. `(8,8,8)` for shape `(2,2)`) | `ValueError: mismatch in length of strides and shape` | **builds a corrupt view** (`Shape.Strides.Length` ≠ ndim) over a garbage pointer |
| `'strides'` shorter than `'shape'` (e.g. `(8,)` for shape `(2,2)`) | `ValueError: mismatch in length of strides and shape` | raw `IndexOutOfRangeException` from the stride loop |

This is the memory-unsafe opposite of the package's stated contract. The interop is otherwise
scrupulous about safety — it refuses writes through read-only sources, marks broadcast views
non-writeable, guards `resize`, sweeps orphaned exports — yet it trusted an arbitrary pointer, shape
and strides from a Python dict without a single check. For a *non-null* garbage pointer
(`strides_gt_shape` above uses `999999`) the consequence is a genuine access violation / memory
corruption on read **or write**, not a caught exception.

The sibling PEP 3118 path (`ViewViaBufferStrides`, for non-numpy strided exporters) is not affected in
the same way: there the pointer and strides come from a real CPython `memoryview`, which is internally
consistent by construction, and that path already checks `strides.Length == shape.Length` and the
itemsize. The gap was specific to `ViewViaArrayInterface`, which takes raw dict values.

## Reproduction

Any Python object can carry a hand-written `__array_interface__`; a buggy third-party library can emit
one by accident. Minimal:

```python
class B: pass
b = B()
B.__array_interface__ = {'shape': (3,), 'typestr': '<f8', 'data': (0, False), 'version': 3}
# numpy: np.asarray(b) -> ValueError: data is NULL but array contains data
```

```csharp
using var b = /* the object above */;
using var v = NDArrayPythonInterop.ToNDArrayView(b, allowReadonly: true);  // BEFORE: returns a view
double x = v.GetDouble(0);                                                 // BEFORE: faults / AV
```

## Expected

Match numpy: reject a malformed `__array_interface__` before any view is built.

## Actual (before the fix)

`ToNDArrayView` returned an `NDArray` whose storage pointed at address 0 (or `999999`, or had a
negative extent / mismatched strides). The failure was deferred to the first element access — a
`NullReferenceException` in the best case (NULL) and an access violation / silent corruption in the
worst (non-null garbage pointer, read **or write**).

## Root cause

`ViewViaArrayInterface` (`src/NumSharp.Interop.pythonnet/NDArrayPythonInterop.Import.cs`) read
`data[0]`, `shape` and `strides` and proceeded straight to window-normalization and
`WrapExternal(...)` without validating (a) that the pointer is non-NULL for a non-empty array,
(b) that every dimension is non-negative, or (c) that `len(strides) == len(shape)`.

## Fix

Three guards added at the top of `ViewViaArrayInterface`, each mirroring the numpy check and message,
each throwing the package's house `NotSupportedException` ("cannot express this as a view"):

- non-negative dimensions (`"negative dimensions are not allowed"`);
- non-NULL `data` pointer for a non-empty array (`"data is NULL but the array is non-empty"`);
- `strides` length equals `shape` length (`"mismatch in length of strides and shape"`).

Honest inputs are unaffected: a real numpy transpose / slice / reversed / broadcast view still imports
zero-copy, and every non-numpy strided `memoryview` still imports through `ViewViaBufferStrides`.

## How it is gated

`test/NumSharp.Interop.UnitTests/ArrayInterfaceValidationTests.cs` (5 tests): one per malformed class
asserting a `NotSupportedException` whose message matches the numpy wording, plus a control proving a
valid strided interface still imports zero-copy AND aliases (write-through visible in numpy). The full
interop suite (239 tests) passes with the guards in place.

## Related

- `docs/bugs/pythonnet-decoder-cache-poisoning.md` — the other confirmed interop gap (upstream
  pythonnet; register the codec before the first `As<NDArray>()`).
- `src/NumSharp.Interop.pythonnet/NDArrayPythonInterop.Import.cs` — `ViewViaArrayInterface` (fixed),
  `ViewViaBufferStrides` (already guarded).
