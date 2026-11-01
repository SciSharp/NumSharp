---
paths:
  - "src/NumSharp.Core/Matrixlib/**"
  - "src/NumSharp.Core/Creation/np.{asmatrix,bmat}.cs"
  - "src/NumSharp.Core/View/Shape.Matrix.cs"
  - "test/NumSharp.Tests/Matrixlib/**"
  - "test/oracle/gen_matrix*.py"
---

# `numpy.matrix` as `NDMatrix`

`np.matrix(data, dtype=None, copy=True)` returns an `NDMatrix`, a port of `numpy/matrixlib/defmatrix.py`
(`src/NumSharp.Core/Matrixlib/`). The CLR type is `public partial class NDMatrix : NDArray` (the house `ND*` prefix);
`np.matrix(...)`, `np.asmatrix(...)` and `np.bmat(...)` return it, and `coverage/object_surfaces.py` maps `numpy.matrix`
to `NumSharp.NDMatrix` (all members but `ctypes` / `dump` / `dumps`, which ndarray lacks too).

## Static type versus the object

- Members and operators follow the static type. NumPy finds the subclass at run time; NumSharp binds the members and
  operators declared on `NDMatrix` at compile time. The same object through an `NDArray` reference is a plain 2-D array
  to the operators, the indexers and the `NDArray` members: `a * a` is elementwise, `a[0]` drops an axis, `a.sum(0)` has
  N elements. Every kernel and helper is written against `NDArray`, and a run-time dispatch would change the meaning of
  that code for a caller who passed a matrix in. `NDMatrix.Clone()` hides `NDArray.Clone()` rather than overriding it for
  the same reason.
- The `np.*` functions follow the object: the matrix boundary (`Matrixlib/NDMatrixBoundary.cs`). A public function that
  receives an `NDMatrix` object through its `NDArray` parameter returns the class and shape NumPy 2.4.2 returns for a
  `numpy.matrix` operand; the result is an `NDMatrix` typed `NDArray` (cast it for the matrix members, no copy).
  - Rules, one statement per public overload: `if (NDMatrixBoundary.Sees(a)) return NDMatrixBoundary.<Rule>((a, ...),
    static s => f(s.a, ...));`. `Wrapped` is `__array_wrap__` / subclass allocation (ufuncs, `np.transpose`,
    `np.concatenate`, `np.dot`, `np.linalg.inv`: a 1-d result becomes a 1 × N row, more than two long axes raise
    `shape too large to be a matrix.`, so `np.stack((m, m))` raises). `Reduced` is the matrix METHOD NumPy calls
    (`np.sum(m, 0)` is a 1 × N matrix, a scalar without an axis, and `keepdims: true` raises `TypeError: matrix.sum() got
    an unexpected keyword argument 'keepdims'`). `np.sum` / `np.prod` / `np.mean` / `np.var` / `np.std` hand the method
    `axis`, `dtype`, `out` (and `ddof`, after `correction=` folded into it), so each other keyword passed (`keepdims`,
    `initial`, `where`, `mean`) is its TypeError, the first in the function's signature order; `ReducedInto` is their
    `out=`: a matrix out gets the method's answer, a plain out receives the values and then raises
    `AttributeError: 'numpy.ndarray' object has no attribute '_collapse'`. `Plain` is `asarray` inside (`np.cross`,
    `np.einsum`, the whole `np.fft`, `np.linalg.norm`). `WrappedAll` / `WrappedPair` / `WrappedTriple` /
    `WrappedFirstTwoOfFour` are the functions with several results (`np.split`, `np.divmod`, `np.linalg.qr`: Q and R;
    `eig` / `eigh`: plain eigenvalues, matrix eigenvectors; `svd`: U and Vh; `lstsq`: x and the residuals). `WrappedIf` is
    `np.block`, whose class follows NumPy's route (repeated concatenation returns a matrix, the single-allocation route
    past 2*512*512 elements a new ndarray).
  - Per-function facts from NumPy's source: `np.linalg.solve` and `lstsq` take their class from `b`; `np.clip` from the
    array or a bound; `np.iscomplex` is a matrix for a complex matrix and a plain array otherwise; `np.r_` / `np.c_` /
    `np.block` cannot give a matrix entry a third axis (`np.block([[[m]]])` raises "axis -3 is out of bounds for array of
    dimension 2").
  - The boundary is the outermost public call. A per-thread depth counter is raised by every rule around a second call
    of the function, and `Sees` is false while it is raised: nested public calls, `NDArray` members implemented by
    `np.*` functions and `NDMatrix` members compute with plain-array semantics. The state is packed in a tuple and handed
    to a `static` lambda (a capturing lambda would allocate at method entry on every call, plain arrays included).
  - A function without a rule opens a region (`using var plain = NDMatrixBoundary.PlainFor(a, ...)`): its result is the
    plain array NumPy's `asarray`-based functions return, and the safe answer where NumPy's is not implemented. A
    container stores the plain view (`NDMatrixBoundary.PlainOf`): numpy.ma (the masked array's data and mask in its
    constructor, and `AsData`) and the samplers' converted parameters (`RandomParam.Convert`), because a region cannot
    cover calls made after it closed.
  - The contract is measured: no method of the library may reach the matrix rule of a public function at depth zero.
    `FuzzCorpusTests.MatrixOperand_LibraryNeverReachesAMatrixRule` replays every corpus case with a 2-D operand (the
    ordinary, numpy.ma and index tiers) with those operands typed as matrices while
    `NDMatrixBoundary.AuditInternalCallers` is on, and requires the set of library callers to stay empty. A new finding
    is closed with a region or a rule.
  - Limits (`[Misaligned]`, pinned in `Matrixlib/NDMatrixBoundary.Tests.cs`): a function whose C# return type is a typed
    array cannot return a matrix (`np.isclose`, gap MO01; the typed `np.all` / `np.any` overloads, `np.nonzero` and
    `np.flatnonzero` are plain in NumPy too); `np.gradient` and `np.meshgrid` return plain arrays in their result structs
    (gaps MO06 / MO07, `OpenBugs.MatrixOperand.cs`); numpy.ma has no `masked_matrix` (NumPy's `ma.array(m).sum(axis=0)`
    is 1 × N); the `numpy.polynomial` package computes plain; a result is always a matrix VIEW of the plain result (no
    `flags.owndata` parity for the functions NumPy allocates as the subclass). Where matrix semantics leak into NumPy's
    own Python code, NumSharp does not reproduce it: gaps MO02 to MO05, `np.linalg.vector_norm(m)` (an array of `|x|` in
    NumPy), `np.average(m, axis=1, weights=ndarray)` (an N × N matrix in NumPy), `np.apply_along_axis` when `func1d`
    depends on the slice's rank (NumPy hands it a 1 × N matrix). Open (`OpenBugs.NpLayer.cs`, NL13): the max / min
    family with a matrix operand computes `initial=` / `where=` and refuses a matrix `out=`, where NumPy calls the
    method (its keyword TypeError; the method's answer in the out).
  - Cost: plain arrays pay one type test per call; a matrix operand pays the rule (the scope that reclaims the hidden
    plain result, the second call, the matrix view), about 300 ns, the same as the `NDMatrix` operators.

## Scalars

- A 0-d `NDMatrix` is a scalar, not a matrix. The 15 C# scalar types convert implicitly to a 0-d `NDMatrix`, so `m * 2`,
  `m + 1.5` and `m[mask] = 0` compile and bind the `(NDMatrix, NDMatrix)` operators; `*` with a 0-d operand scales.
  Members whose NumPy result is always a scalar return a 0-d `NDArray` (`m[0, 0]`, `m.sum()`); the general indexer
  (`this[params object[]]`, the one C# form of `matrix.__getitem__`) returns a 0-d `NDMatrix` for a full integer index.
- An explicit 0-d ARRAY is not a scalar to `*`. NumPy's `matrix.__mul__` tests `isinstance(other, ndarray)` before
  `isscalar(other)`, so `m * np.array(2)` is `np.dot(m, asmatrix(np.array(2)))`, the product with a 1 × 1 matrix:
  `ValueError: shapes (2,3) and (1,1) not aligned` unless `m` has one column. NumSharp has one 0-d type for both roles,
  so `np.array<T>(T)` and `np.asarray<T>(T)` set a per-object mark, `NDArray.IsExplicitZeroDimArray`
  (`Backends/NDArray.ZeroDimArray.cs`; `np.array(nd)` / `np.asarray(nd)` of a marked array carry it over), and
  `NDMatrix.Product` reads it: marked on the right → the 1 × 1 product; marked on the left → `np.dot(a0, m)`, the scalar
  route (a new owner, `0 + s * x`). Unmarked means scalar: every computed 0-d value scales (`m * m.sum()`,
  `m * np.dot(v, v)`, `m * np.linalg.norm(a)`, `m * a[0, 0]`, `m * NDArray.Scalar(2.0)`). A flags rule cannot do this:
  many computed scalars (`np.dot(v, v)`, `np.linalg.det`, `np.linalg.norm`, `z + z`) are writeable owners exactly like
  `np.array(2)`.
- Library code builds a computed scalar with `NDArray.Scalar(x)`, never with `np.array(x)`.
- Divergent (`[Misaligned]`): a 0-d array built another way (`np.zeros(())`, `x.reshape(())`, `np.squeeze(x)`)
  scales, where NumPy raises; NumSharp's own kernels allocate their 0-d results with those functions, so marking them
  would make `m * np.linalg.norm(a)` raise.

## Results, indexing and assignment

- Result forms (`NDMatrix.cs`): `ViewOf(a)` is `a.view(matrix)` (never an owner); `From(r)` is a result NumPy allocates
  as a matrix (copy, astype, dot, take: the matrix takes the result's storage over and owns when it did);
  `FromWrapped(r)` is a ufunc result `__array_wrap__` returns as a view (operators, reductions, cumsum, clip: never an
  owner); `FromIndexing` is `matrix.__getitem__`'s rule. All apply the finalize rule `Shape.AsMatrix()`
  (`View/Shape.Matrix.cs`): 0-d → `(1,1)`, 1-d → `(1,N)`, more than two axes longer than 1 →
  `ValueError("shape too large to be a matrix.")`.
- Indexing (`NDMatrix.Indexing.cs`): a 1-d result is a COLUMN when `len(index) > 1 and isscalar(index[1])` (entry 1 by
  position: `m[..., 0]` is a column; a 0-d array is not a scalar), else a row; other ranks are returned as they are
  (`m[newaxis]` is a 3-d matrix object, as in NumPy). An index array or mask gives a matrix that VIEWS the gathered array
  (NumPy's `wrap_out_array`); a 0-d integer array index gives an owning copy (`HAS_SCALAR_ARRAY`). Index errors carry
  NumPy's texts (`ValidateBasic`).
- Assignment: matrix has no `__setitem__`, but ndarray's assignment gets the subclass view through its own `__getitem__`
  for an index of integers and slices, so the value is broadcast into the row or COLUMN: `m[:, 0] = [7, 8, 9]` raises
  `could not broadcast input array from shape (3,) into shape (3,1)`. One element is stored through the dtype's setitem
  (an array value is refused; bool takes a one-element array's truth value; complex128 raises TypeError). The general
  indexer takes an `NDMatrix` or a number (`m[row, column]` any `NDArray`); `__setitem__(index, object)` takes any
  value.

## Operators

- `*` and `**` (`NDMatrix.Operators.cs`): `m * x` is `np.dot(m, asmatrix(x))` (a 1-d `x` becomes a ROW), `x * m` is
  `np.dot(x, m)`. Each binary operator has five forms (matrix/matrix, matrix/array, array/matrix, matrix/object,
  object/matrix).
- Scaling is `np.dot`'s scalar route: the dtype is `promote_types` of both operands (not the weak-scalar ufunc rule).
  For float32 / float64 / complex128 the result is a new C-contiguous owner and each element is `0 + scalar * x`, so a
  product of -0.0 is +0.0 (`?axpy` into zeros), except a one-element result (the direct product keeps -0.0). Every other
  dtype is the multiply ufunc, wrapped.
- The float scaling has three routes with one result (`Scale`): a C-contiguous float32 / float64 operand of the loop
  dtype (`m * 2.0`) runs one direct pass over raw memory (`ScaleContiguous`, the polynomial package's `PolyAffineDouble` /
  `PolyAffineSingle`: multiply then add, never an FMA); another operand of the loop dtype (`m.T`, a stepped view, a
  complex matrix) runs one fused `np.evaluate` pass from `FusedScaleMinElements` (4,096 for float views) /
  `FusedScaleMinElementsComplex` (512) elements on; the rest runs the multiply ufunc and then adds the zero in place with
  the same raw kernel. All three give the bytes of `np.add(np.multiply(x, s, dtype: loop), 0)`, except the payload of a
  NaN that meets a NaN scalar (NumPy's comes from the BLAS `?axpy` kernel, which no route models). complex128 has no
  direct pass: that kernel multiplies in the other operand order, which can round the last bit differently.
- A plain 0-d `NDArray` on the LEFT stands for a NumPy scalar and is the elementwise ufunc (`m.sum() * m`). C# has no
  `**`: `__pow__(int | object)`, `__ipow__`, `__rpow__`; also `__imul__`. Reflected comparisons evaluate with the matrix
  as the first ufunc operand (the order of NumPy's broadcast error text).

## Reductions and construction

- Reductions (`NDMatrix.Reductions.cs`): `sum / prod / mean / var / std / any / all / max / min` keep two axes
  (`keepdims=True`, then `_collapse`); `argmax / argmin / ptp` use `_align` (axis 0 → row, 1 → column, a valid negative
  axis → `ValueError("unsupported axis")`). `(int axis, ...)` returns `NDMatrix`, `(int? axis = null, ...)` returns
  `NDArray` (the 0-d scalar without an axis). A matrix reduction is always NumPy's ARRAY form: a float16 mean rounds
  twice, and `m.std(dtype: np.int64)` raises the sqrt cast error where `np.std` of the plain array truncates. With `out=`
  the loop is `promote_types(out.dtype, operand dtype)` without the small-integer widening
  (`reducelike_promote_and_resolve`), stored with an unsafe cast: an int8 sum into a bool out is the truth value of the
  int8 sum. `var` / `std` hand `out` to their last sum only (`TensorEngine.ReduceVar` / `ReduceStd` with `out=`:
  `DefaultEngine.VarStdNumPy`'s `out`).
- Constructor (`NDMatrix.Construction.cs`): the four branches of `matrix.__new__` in NumPy's order. A matrix literal
  (`"1 2; 3 4"`, `"[1.5, 2; 3 4j]"`) is `_convert_from_string` plus a port of `ast.literal_eval` for one token (four
  number bases, `_`, floats, imaginary numbers, signs, parentheses, `True` / `False`; `SyntaxError` and `malformed node`
  errors). A sequence goes through `UnmanagedStorage.FromSequence`; with a `dtype` each element is stored by
  `PyArray_Pack` (`ReductionInitial.TryPack`): a Python int out of range raises OverflowError, a complex number into a
  real dtype raises TypeError.

## Divergences and the library rules it relies on

- Divergences: the static typing above; a C# literal keeps its C# type (`matrix(int8) * 2` is int32 where NumPy's Python
  int gives int64); no tuple axes; the address in `ast` error texts is zero; complex128 scaling by a complex scalar is
  the ufunc product, which can differ from OpenBLAS `zaxpy` in the last bit.
- `np.asmatrix` returns a new view, never its operand. `np.dot(1-D, 2-D)` and `np.matmul` with a 1-D operand return an
  owner (the inserted axis is dropped in place); `np.dot(N-D, 1-D)` reports a mismatch in `np.dot`'s words.
  `NDArray.PrepareAssignedValue` is NumPy's index-array value rule (`selection-indexing.md`).
- A C# scalar literal operand (`m > 0.75`, `m * 2.0`) allocates a 0-d temporary that is left to the finalizer, and in a
  hot loop the garbage collector, not a lock, is the cost (every finalizable temporary is promoted with what it
  references, so the collections are generation 1 and pause the thread). Hold the 0-d operand (`NDArray s = 0.75`) or
  dispose it.

## Gates

- The oracle tier `matrix.jsonl` (`test/oracle/gen_matrix_oracle.py`, valueclass `matrix`, zero excuses): values, the
  result CLASS (`expected.cls`: matrix / ndarray / scalar), the memory facet (flags, strides, view / copy / identity
  against each operand) and NumPy's error type and text, over 14 source layouts and the dtypes; operands rebuild as
  `NDMatrix` (`"matrix": true`). It is replayed once per scaling route: `Matrix` (production: a counter proves the direct
  pass ran), `Matrix_FusedScaling` (direct pass off, fused pass forced) and `Matrix_TwoPassScaling` (both off), through
  `NDMatrix.DisableDirectScale` / `FusedScaleMinElementsTestOverride`. 111 cases are carved into
  `gaps/matrix_gaps.jsonl` under six gap ids; every one is an `NDArray` member or `np.*` function below the class. Pins in
  `OpenBugs.Matrix.cs`: a fancy assignment into a C-contiguous view at an offset throws (MX07), and the result layouts of
  np.cumsum / np.cumprod (MX02), ndarray.round of a broadcast operand (MX04) and np.conjugate of a transposed strided view
  (MX05); MX01 (the fancy getter's layout) and MX09 (complex128 np.prod's zero sign) are pinned by
  `OpenBugs.FancyIndex.cs` and `OpenBugs.Reductions.cs`.
- The matrix-operand tier (`test/oracle/gen_matrix_operand_oracle.py`, `oracle.md`): the public `np.*` functions with a
  matrix operand; gaps MO01–MO05 carved into `gaps/matrix_operand{,_host}_gaps.jsonl`.
- Unit tests: `Matrixlib/NDMatrix.Tests.cs` (NumPy's `test_defmatrix.py` ported plus the design pins;
  `Scaling_EveryRoute_GivesTheUfuncBits` holds the three routes to the ufunc reference past every threshold) and
  `Matrixlib/NDMatrixBoundary.Tests.cs` (the functions outside the matrix-operand truth table, whose rule comes from
  NumPy's source); the LAPACK functions' matrix operands are `test/NumSharp.Tests.Interop/MatrixOperandLapackTests.cs`.
  The leak completeness gate audits every declared member (`LeakCatalogue.Matrix.cs`).
- Benchmarks: warm every route twice with a pause, and time two routes in one process with interleaved rounds (a cell
  warmed once reads tier-0 times).
