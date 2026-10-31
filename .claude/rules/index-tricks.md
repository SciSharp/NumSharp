---
paths:
  - "src/NumSharp.Core/Creation/np.{r_,c_,ogrid,mgrid,meshgrid}.cs"
  - "src/NumSharp.Core/Indexing/np.{ix_,s_,indices}.cs"
  - "src/NumSharp.Core/Indexing/NDArray.IndexTricks.cs"
  - "test/NumSharp.Tests/Creation/np.{r_,ogrid,mgrid,meshgrid}*.cs"
  - "test/NumSharp.Tests/Indexing/np.{ix_,indices}*.cs"
---

# Index tricks: `r_`, `c_`, `ix_`, `s_`, `index_exp`, the grids and `indices`

NumPy's `_index_tricks_impl.py`; tests `Creation/np.r_.Test.cs`, `Indexing/np.ix_.Test.cs`; oracle cases `r_`/`c_`/`ix_`
in the `manip` tier. Sources `Creation/np.{r_,c_}.cs`, `Indexing/np.{ix_,s_}.cs`.
- A Python slice literal is a STRING in C#: `np.r_[0:5]` is `np.r_["0:5"]`. A string containing `':'` is a slice
  expression; a string without one is a NumPy special directive (`"r"`, `"c"`, `"-1"`, `"0,2"`, `"1,2,0"`). One string
  may hold several comma-separated slices (`np.r_["1:3, 5:8"]`), and a `Slice` / `Slice[]` object works as an entry, so
  `np.s_[…]` composes into `np.r_[…]`.
- `np.r_[…]` / `np.c_[…]` are static properties returning `RClass`/`CClass` (`AxisConcatenator` subclasses with a
  `this[params object[] key]` indexer, the port of NumPy's `__getitem__`). `r_` concatenates along axis 0; `c_` is
  `r_["-1,2,0", …]` (entries upgraded to ≥ 2-D with the 1s post-pended, a 1-D entry becomes a column, stacked on the last
  axis). Slice entries become `arange(start, stop, step)` or, for an imaginary step (`"-1:1:6j"`), `linspace(start, stop,
  N)` with the stop inclusive. Integer-ness follows the LITERAL (`"0:5"` is int64, `"0.0:5"` float64): the slice fields
  are the Python numbers they spell (`np.AxisConcatenator.SliceSpec`: digits are an int, kept exactly at any size,
  anything else a float), shared with np.mgrid / np.ogrid, so an integer slice takes arange's int route
  (`np.r_["9007199254740993:9007199254740995"]` has NumPy's two elements, a zero step is `division by zero`). A missing
  stop re-reads start as stop (`np.r_["2:"]` is `[0, 1]`, `np.r_["5::2"]` is `[0, 2, 4]`). Directives: axis,
  `axis,ndmin`, `axis,ndmin,trans1d` (NumPy's two upgrade paths: the array branch's `defaxes` permutation and the slice
  branch's `swapaxes(-1, trans1d)`), a silently ignored fourth field, and `"r"`/`"c"` matrix coercion (NumPy's
  `matrix(res)`, a C-ordered copy, so an F-ordered concatenation comes back C-contiguous; an `NDMatrix` typed `NDArray`).
  The indexer's statements are `NDArray.AxisConcatenate` (`Indexing/NDArray.IndexTricks.cs`). Errors: `special
  directives must be the first entry.`, `unknown special directive` / `unknown special directive '0,q'` (the comma form
  quotes).
- `ndmin` is uncapped (NumSharp has no 64-dimension ceiling; NumPy raises `ndmin must be <= ndmax (64)`), and cheap:
  `NDArray.AtLeastNdView` prepends every axis in one `Storage.Alias` over `Shape.PrependDimensions` (O(ndim)); a prepended
  axis takes stride `dimensions[0] * strides[0]` (0 for a 0-d source), so strides, flags and the writeable bit equal a
  per-axis `expand_dims` loop. An `ndmin` too large to allocate a shape for raises `OutOfMemoryException`; a non-positive
  `ndmin` is a no-op.
- `np.ix_(params object[])` → `NDArray[]` (an open mesh). The dtype is preserved (no integer validation; a float or int8
  sequence fails only at the later indexing call), except that an EMPTY non-ndarray input (an empty `object[]` or
  collection; a typed C# array or `Memory<T>` is an ndarray and keeps its dtype) becomes int64; a bool sequence is a mask
  (`nonzero`). Results are NEW views that write through (one sequence's too, as NumPy's same-shape reshape is
  `PyArray_View`: never the operand itself, never an array that owns its data), built with `expand_dims` (not
  `reshape`) so a stride-0/read-only operand keeps its strides and its read-only flag. `ValueError("Cross index must be 1
  dimensional")`, which a null sequence also raises.
- `np.s_[…]` / `np.index_exp[…]` capture an index expression; `arr[X]` ≡ `arr[np.s_[X]]` for every X. Their three
  indexers return the two tuple types `NDArray`'s one indexer takes whole: `this[string]` and `this[params Slice[]]` →
  `Slice[]` for basic expressions (`int` and `string` convert to `Slice`), and `this[params object[]]` → `object[]` as
  soon as an entry cannot be a `Slice` (an `NDArray` index or mask, `int[]`/`long[]`/`bool[]`, an `IList`, a
  basic/advanced mix). Both feed `arr[…]` directly (a `Slice[]` reaches the indexer by array covariance and takes its
  basic-index body), an `NDArray<T>` included (its indexer overrides `NDArray`'s and types the result). `s_` and
  `index_exp` are identical in C# (NumPy's differ only by `maketuple`, which C# cannot observe).
- Weak scalars by C# type: `bool`, the eight integer primitives, `float`, `double` and `Complex` are weak (the C# literal
  is the Python literal); `char`, `Half`, `decimal` and every `NDArray` (0-d included) are strong
  (`np.r_[i8, NDArray.Scalar(1L)]` is int64, `np.r_[i8, 1L]` int8). A weak entry is stored by the adopted dtype's setitem
  (`NDArray.AsStoreValue`, NumPy's `PyArray_Pack` with the LP64 texts): an integer that does not fit raises
  `OverflowException("Python integer 1000 out of bounds for int8")`, past the C long "Python int too large to convert to
  C long" (`np.r_[i8, ulong.MaxValue]`); a weak float saturates to ±inf. An all-literal key whose integer does not fit
  int64 lifts the default to uint64. One complex width: `result_type(float32, 1j)` is complex128.
- Matched, no need to re-prove: `-0.0` keeps its sign; NaN/±inf/subnormals round-trip; the weak-integer boundaries are
  exact (`i8+127` fits, `+128` raises); `9e18:9e18+2` collapses to empty; huge slice expressions raise
  `OutOfMemoryException`; NumPy's error order holds. `np.r_`/`np.c_` are stateless singletons (a directive is copied to
  locals per call).
- Divergences: a `null` entry throws (NumPy builds an object array); `np.ix_("abc")` is accepted as a char sequence;
  a sub-1e-15 slice step reports `np.arange`'s "step can't be 0". No `bmat` branch: NumPy routes a lone bare string key
  into `matrixlib.bmat`, which resolves words against the caller's Python frame (and raises `NameError` for every
  literal), so a lone string is read like any other entry.
- `np.mgrid` / `np.ogrid` are NumPy's `nd_grid.__getitem__` statements, the array object's `NDArray.MGrid` /
  `NDArray.OGrid` (`Indexing/NDArray.IndexTricks.cs`, beside np.r_'s): a key of one slice is the except branch
  (`GridSingleSlice`: arange, or for an imaginary step `arange(0, n) * step + start`, a fresh array), a key of several
  the try branch: every size first, in key order (`GridAxisSize`: `int(abs(step))` / `math.ceil((stop - start) / step)`
  in Python's arithmetic over the slice's Python numbers), `typ = result_type(*num_list)` (`GridMeshDtype`, through
  `DTypePromotion.ResultType` over the weak Python scalar classes), then the lines `i * step + start` (`GridFillLine`: the
  step and start converted as NumPy's ufunc calls convert them, a product and a sum, never fused). So a zero step is
  Python's `division by zero` / `float division by zero`, NaN / infinite / past-int64 sizes raise NumPy's texts, and an
  int64 mesh's start or step past int64 is NEP 50's conversion error (the LP64 text). Gates: the snapshot
  (`tools/facade-move/snapshot.cs`) and `Creation/np.{mgrid.NumPyPort,r_.PythonNumbers}.Test.cs`.
- `np.ogrid[...]` (`Creation/np.ogrid.cs`, `OGridClass`): the open mesh, with the same slice grammar
  (`AxisConcatenator.ParseSliceToken`/`SliceSpec`) and no directives. One slice returns a 1-D array (for an imaginary
  step the stop is inclusive, not linspace's forced endpoint); several return N arrays each of extent 1 on every axis but
  its own (`arange(size) * step + start` reshaped). The whole mesh shares one dtype (int64 iff every field of every slice
  is a Python int, else float64; an imaginary step floats the mesh). A single slice may omit its stop unless its
  imaginary step has more than one point (Python's TypeError, as in NumPy); a multi-slice may not, but for an imaginary
  step of one point (NumPy leaks an `AttributeError`; NumSharp raises a `ValueError` naming the slice, at the same
  point). `OGridResult` converts implicitly to `NDArray` (one slice) or `NDArray[]` and deconstructs
  (`var (y, x) = np.ogrid["0:3", "0:5"];`). Gate: `Creation/np.ogrid.Test.cs`.
- `np.mgrid[...]` (`Creation/np.mgrid.cs`, a property): one slice returns a 1-D array; several return ONE stacked array
  `(N, size_0, …, size_{N-1})`, built as `np.indices(sizes, typ)` with each non-trivial axis (start ≠ 0, step ≠ 1,
  imaginary) overwritten by its 1-D line broadcast across the grid (`np.copyto`), so `np.mgrid[0:a, 0:b]` is one fused
  `indices` fill. NumPy sizes each axis by `ceil((stop-start)/step)`, which may be negative: `mgrid[5:0, 0:3]` raises
  `"negative dimensions are not allowed"` where `ogrid[5:0, 0:3]` gives an empty axis. `MGridResult` converts to
  `NDArray`, deconstructs (`var (x, y) = np.mgrid["0:5", "0:3"];`) and indexes the per-axis grids. Gate:
  `Creation/np.mgrid.Test.cs`.
- `np.indices(dimensions, dtype, sparse)` is `NDArray.Indices` / `IndicesSparse` (`Backends/NDArray.Numeric.cs`), NumPy's
  `res = empty((N,) + dimensions, dtype)` then one `arange(dim, dtype)` per axis assigned with broadcasting: the
  allocation's ValueError comes first (a negative dimension), an arange's TypeError next (a text dtype at any length,
  bool past two elements, even in an empty grid), a text dtype's empty tuple is an empty text array, and an int64-lane
  grid keeps the IL fill kernel. The sparse grid is views of its aranges (`flags.owndata` False), and a negative dimension
  there is an empty axis (the reshape's unknown dimension), as NumPy's. Gate `Indexing/np.indices.NumPyPort.Test.cs`.
- `np.meshgrid(x1, x2[, x3], string indexing="xy", bool sparse=false, bool copy=true)` plus `meshgrid(NDArray[] xi, …)`
  (`Creation/np.meshgrid.cs`): `"xy"` swaps the first two axes ((N, M) for 2-D) against `"ij"` ((M, N)); `sparse` keeps
  the open-mesh `(1,…,Ni,…,1)` shapes; `copy=false` returns broadcast views. Each grid keeps its input's dtype. A bad
  `indexing` raises `ValueError "Valid values for \`indexing\` are 'xy' and 'ij'."`. `MeshgridResult` converts to
  `NDArray[]`, deconstructs and indexes. Gate: `Creation/np.meshgrid.Test.cs`.
