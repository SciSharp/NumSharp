---
paths:
  - "src/NumSharp.Core/View/Shape.Axes.cs"
  - "src/NumSharp.Core/Exceptions/AxisError.cs"
  - "src/NumSharp.Core/Exceptions/AxisOutOfRangeException.cs"
  - "src/NumSharp.Core/Backends/Default/Math/Reduction/ReductionArguments.cs"
  - "test/NumSharp.Tests/Manipulation/AxisArgumentParity.Test.cs"
---

# Axis arguments

Every axis argument goes through NumPy's `normalize_axis_index` / `normalize_axis_tuple`, ported once. Gates: the
type-strict, zero-excuse `axis_args.jsonl` oracle tier, the `evaluate.jsonl` tier's C10 block and
`Manipulation/AxisArgumentParity.Test.cs`.
- `Shape.check_and_adjust_axis(ndims, axis, msgPrefix)` (`View/Shape.Axes.cs`; `DefaultEngine`'s public static of that
  name forwards to it) is NumPy's `check_and_adjust_axis_msg`: the range test runs on the axis AS GIVEN, then the wrap,
  and the error is `AxisError` (NumPy's type, an `ArgumentOutOfRangeException`) with NumPy's text, `"{msgPrefix}: axis 5
  is out of bounds for array of dimension 2"` (`new AxisError(axis, ndim, msgPrefix)`). NumPy names the argument where it
  normalizes several: moveaxis' `source` / `destination`, swapaxes' and diagonal's `axis1` / `axis2`, cross' `axisa` /
  `axisb` / `axisc`, average's and trim_zeros' `axis`, unstack's `source`.
- `Shape.normalize_axis_tuple(axis, ndim, argname, allow_duplicate)` is NumPy's function: a NEW array (the caller's is
  never rewritten), every axis range-checked before the repeat check, a repeat NumPy's `ValueError "repeated axis"` /
  `"repeated axis in \`{argname}\` argument"` (never silently removed).
- `AxisOutOfRangeException` is `[Obsolete]` and derives from `AxisError`: nothing in NumSharp raises it.

Rules every caller follows:
- Never wrap then test. `while (axis < 0) axis += ndim` accepts `-ndim - 1` (wrapping twice, to the last axis) and
  never terminates for a negative axis of a 0-d input (`np.count_nonzero(0-d, axis=-2)` would hang until the oracle's
  watchdog kills the run).
- Check before every shortcut. `ReductionArguments.CheckReduceAxis` (PyUFunc_GenericReduction's integer branch: a 0-d
  input takes 0 / -1 and nothing else) runs first in prod / max / min / nanmax / nanmin, whose 0-d and one-element
  shortcuts would otherwise answer any axis; likewise cumsum / cumprod / argmax / argmin / compress.
- 0-d inputs: the ufunc reductions, `count_nonzero` and `squeeze` accept axis 0 / -1 of a 0-d input (no axis selected)
  and report every other axis against rank 0; the `PyArray_CheckAxis` functions (argmax / argmin / argsort / cumsum /
  cumprod / take / compress / repeat / packbits / unpackbits) ravel a 0-d input first and report rank 1; `np.sort`,
  `np.rollaxis`, the tuple forms and `np.linalg.norm` / `vector_norm` (a tuple reduce) refuse every axis.
- Python's IndexError where NumPy reads `a.shape[axis]` first: `np.split` / `np.array_split` (before the sections
  check) and `np.linalg.cross` raise `IndexError "tuple index out of range"`; `np.fft.irfftn(a, axes=())` raises it on the
  caller's tuple ("list index out of range" only for the defaulted axes of a 0-d input).
- NumPy's ValueError / TypeError around the axis: a non-unit squeeze axis ("cannot select an axis to squeeze out which
  has size not equal to one"), two equal diagonal / trace axes, rot90's axes, split's sections, stack's shape mismatch,
  the stacking family's empty list ("need at least one array to stack" / "…to concatenate"; atleast_1d/2d/3d of no
  arrays is the empty tuple), diff / unwrap / ma.diff of a 0-d input, the quantile family's q range. The boolean subtract
  / negative refusals are NumPy's `TypeError`, and the quantile family's complex input is `TypeError "a must be an array
  of real numbers"`, raised first.
- Values the rules decide: `np.linalg.vector_norm(x, axis=tuple)` is NumPy's tuple branch (the axes merged, the VECTOR
  norm; a 2-tuple never reaches norm's matrix branch), and the quantile family's EMPTY tuple reduces no axis.
  `np.rollaxis` with nothing to move and `np.expand_dims(a, ())` return a fresh view, never the input. np.average over a
  tuple axis is one add.reduce over the axis set, never a mean of means (`reductions.md`); `np.ptp` over a tuple naming
  every axis takes the flat reductions; default-order `vector_norm` / `norm` of a float32 / float64 operand streams
  `Sum(x * x, axis)` through np.evaluate (add.reduce's schedule, same bits, no temporary).
- Axes NumPy uses as Python LIST indices or values raise Python's errors. np.pad's dict `pad_width` (NumPy 2.3+) is
  NumPy's dict branch: each width is matched first (anything but an int or a 2-tuple of ints is `typing.assert_never`'s
  `AssertionError`; a typed C# array is an ndarray and refused), then `seq[axis] = …` (a key outside `[-ndim, ndim)` is
  `IndexError "list assignment index out of range"`, every key of a 0-d array is one), then `np.asarray(seq)`'s
  `TypeError` (a 0-d array's empty dict, every row given bools, a width past int64), then the negative width's
  `ValueError` after every key. `np.linalg.tensorsolve`'s axes go through `allaxes.remove(k)`, so a negative axis or one
  past the rank is `ValueError "list.remove(x): x not in list"`.
- np.evaluate's reductions follow their NumPy function's axis rule: the ufunc kinds (and count_nonzero / nansum /
  nanprod / nanmean) take axis 0 / -1 of a 0-d child as no axis; Mean / Var / Std refuse it (`_count_reduce_items`);
  Average names its argument ("axis: …").
- One house convention stays in the strict tier: NumPy's broadcast `ValueError` ("shape mismatch: objects cannot be
  broadcast…", np.cross' operands) is NumSharp's `IncorrectShapeException`, library-wide.
