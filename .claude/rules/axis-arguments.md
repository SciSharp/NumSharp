---
paths:
  - "src/NumSharp.Core/View/Shape.Axes.cs"
  - "src/NumSharp.Core/View/NDAxis.cs"
  - "src/NumSharp.Core/Exceptions/AxisError.cs"
  - "src/NumSharp.Core/Exceptions/AxisOutOfRangeException.cs"
  - "src/NumSharp.Core/Backends/Default/Math/Reduction/ReductionArguments.cs"
  - "test/NumSharp.Tests/Manipulation/AxisArgumentParity.Test.cs"
  - "test/NumSharp.Tests/APIs/KeptAxisOverloads.cs"
  - "test/NumSharp.Tests.Build.Analyzer/AxisOverloadResolutionTests.cs"
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
- `NDAxis` (`View/NDAxis.cs`, the owner's OD1) is NumPy's axis argument as one type: a function whose NumPy default is
  `axis=None` takes `NDAxis? axis = null`, one whose axis is required `NDAxis axis` (np.expand_dims, np.moveaxis). It
  converts implicitly from `int` (lifted: an `int?` too, null = None), `int[]` (a null array is None for `NDAxis?` and an
  ArgumentNullException for `NDAxis`), `(int, int)` and `(int, int, int)`, and has public constructors for the same
  values. It only HOLDS the axes as given: each function normalizes them itself, because NumPy's functions read an axis
  four ways - `normalize_axis_tuple` (`NDAxis.Normalize`; flip, expand_dims, moveaxis, the quantile family, average,
  trim_zeros, size, gradient, roll with `allow_duplicate`: every range, then "repeated axis"), the ufunc reduction (sum /
  prod / max / min / all / any / ptp / the nan reductions / the sum behind count_nonzero: the out / where converters,
  every range, then "duplicate value in 'axis'", and the INTEGER axis 0 / -1 of a 0-d input), `_count_reduce_items`
  (np.mean / var / std: each item's range from the count, the repeat only from the sum after its out check, so
  `np.mean(a, (0, 0), out=read_only)` is the read-only error and `np.var`'s the repeat) and `PyArray_ConvertMultiAxis`
  (squeeze: each item's range and repeat in order, so `(0, 0, 5)` is the repeat); np.argmax / argmin take ONE integer
  (a tuple of any length is `TypeError "'tuple' object cannot be interpreted as an integer"`, before the out), and so do
  np.nanargmax / nanargmin, whose NumPy statements reduce the NaN mask over the tuple first (`np.all(mask, axis)`: its
  AxisError, "duplicate value in 'axis'" and "All-NaN slice encountered" come before the TypeError; an integer or empty
  input has no mask) and chararray.argmax / argmin (the TypeError before the finalize error). `IsTuple`
  tells `1` from `(1,)` (Python's `1 != (1,)`); `default(NDAxis)` and `new NDAxis()` are the EMPTY tuple. Trap: the null
  literal converts to `NDAxis` through the `int[]` operator, so `flag ? null : someAxis` is typed `NDAxis` and throws -
  write `(NDAxis?)null`. NumSharp.Build hands an `NDAxis` axis to a `[ReducesThrough]` collapse as it is
  (`NDSubclass.CollapseAxes`, the weaver's `EmitAxisValue`), so `matrix._collapse` keeps a tuple's kept-dims matrix (the
  1 x 1 matrix for `(0, 1)`, the whole matrix for `()`); an `int` / `int?` axis keeps `Collapse`.
- The KEPT INTEGER OVERLOADS (the owner's API-parity batch-6 decision, "keep the int and int[] overloads ... for
  allllll overloads", after a resolution proof): every public np / ndarray / chararray / np.fft / np.linalg member that
  takes NumPy's axis as `NDAxis?` / `NDAxis` has an `int axis` and an `int[] axis` overload beside it, WITHOUT defaults
  (a parameter before the axis loses its default too: norm's `ord`, trim_zeros' `trim`, gradient's spacing), each a
  documented one-line forward that casts to the NDAxis member (every answer, error, layout and subclass result is that
  member's; the woven matrix arm is reached through it). A one-integer axis (argmax / argmin / nanargmax / nanargmin,
  ndarray and chararray methods included) keeps `int? axis` instead (no int[]: NumPy refuses a tuple). moveaxis keeps
  `(int, int)` and `(int[], int[])`. The engine (`TensorEngine` members) and numpy.matrix (its static-type `int`
  overloads) are outside the decision. The binding rules this gives, proven at C# 12 (net8.0) and C# 14 (net10.0) on
  mimics (scratchpad probe_axis) and pinned against the real members at C# 12 by
  `Build.Analyzer/AxisOverloadResolutionTests.cs`: an int-typed argument (a literal, an int / short / byte / char
  variable) binds `int`; an `int[]` and a bare `null` bind `int[]` (null converts to `int[]` better than to `NDAxis?`; a
  null array is None); every other spelling (tuples, NDAxis values, an `int?` variable, an omitted axis) binds the NDAxis
  member. Why the kept integer is `int`, never `int?`, beside an `int[]`: with `int?`, `f(a, null)` is CS0121 ambiguous.
  Why a one-integer axis keeps `int?`: with a plain `int`, `f(a, default)` would silently be axis 0. The one spelling the
  three overloads refuse is `f(a, default)` (CS0121, `int` vs `int[]`) - a compile error, never a silent answer. A
  NumPy-signature test or tool counts overloads WITHOUT the kept forwards: `KeptAxisOverloads.WithoutKeptIntegerForms`
  (tests), `drop_kept_axis_forwards` (tools/api-parameter-parity/signature_diff.py, coverage/signature_parity.py) - a
  structural rule (integer-typed axis parameters AND an NDAxis sibling with otherwise the same parameter names and types),
  never a list, so a stray integer overload without its NDAxis member still counts.
- The reductions route a tuple by `ReductionArguments.TryIntegerForm`: a set of ONE axis takes the integer route and a
  set of EVERY axis the None route (NumPy builds the same flags, so the exact schedules and the text routes are reused);
  every other tuple (two axes or more short of every axis, the empty tuple of an input with dimensions, or a tuple NumPy
  refuses) takes the function's NumPy statements over the set, which raise in NumPy's order
  (`Default.Reduction.AxisSet.cs`, `reductions.md`).

Rules every caller follows:
- Never wrap then test. `while (axis < 0) axis += ndim` accepts `-ndim - 1` (wrapping twice, to the last axis) and
  never terminates for a negative axis of a 0-d input (`np.count_nonzero(0-d, axis=-2)` would hang until the oracle's
  watchdog kills the run).
- Check before every shortcut. `ReductionArguments.CheckReduceAxis` (PyUFunc_GenericReduction's integer branch: a 0-d
  input takes 0 / -1 and nothing else) runs first in prod / max / min / nanmax / nanmin, whose 0-d and one-element
  shortcuts would otherwise answer any axis; likewise cumsum / cumprod / argmax / argmin / compress.
- 0-d inputs: the ufunc reductions, `count_nonzero` and `squeeze` accept the INTEGER axis 0 / -1 of a 0-d input (no axis
  selected; a tuple `(0,)` is an AxisError, `()` selects nothing) and report every other axis against rank 0; the `PyArray_CheckAxis` functions (argmax / argmin / argsort / cumsum /
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
