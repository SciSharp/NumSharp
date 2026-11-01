---
paths:
  - "src/NumSharp.Core/Creation/**"
  - "src/NumSharp.Core/Manipulation/**"
  - "src/NumSharp.Core/Backends/Default/ArrayManipulation/**"
  - "src/NumSharp.Core/Backends/Default/Creation/Default.CoreFunctionBase.cs"
  - "src/NumSharp.Core/Backends/Default/Manipulation/Default.{SetOps,TrimZeros}.cs"
  - "src/NumSharp.Core/Backends/NDArray.Numeric.cs"
  - "src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.{Ctors,Buffer,Coercion}*.cs"
  - "src/NumSharp.Core/Exceptions/AllocationGuard.cs"
  - "test/NumSharp.Tests/Creation/**"
  - "test/NumSharp.Tests/Manipulation/**"
---

# Creation and manipulation

## `np.arange`

`np.arange(start, stop, step, dtype)` is NumPy's `PyArray_ArangeObj` (`ctors.c`: `UnmanagedStorage.ArangeObj`,
`Backends/Unmanaged/UnmanagedStorage.Ctors.cs`). No ufunc runs, so no TensorEngine member is involved.
- The length is `ceil((stop - start) / step)` in Python's arithmetic on the bounds: the `double` overloads are Python
  floats, the `long` / `int` overloads Python ints (exact, with the correctly rounded true division). A quotient that
  underflows to zero counts one element for +0 and none for -0. A ceiling of exactly `2**63` is NumPy's x86 empty array
  (C's cast gives INT64_MIN there; .NET 9+ saturates, so the case is decided explicitly).
- `start` and `start + step` are stored through the dtype's `setitem` slot (NumPy's errors: `Python integer 130 out of
  bounds for int8`), and the rest is filled by its `fill` slot in the dtype's own arithmetic (bool's fill refuses more
  than two elements).
- NumPy's dtype checks run first (a text dtype is NumPy's TypeError "arange() not supported for inputs with DType …"),
  then Python's ZeroDivisionError for a zero step (`DivideByZeroException`); a NaN length is "arange: cannot compute
  length", a length past int64 "Maximum allowed size exceeded".
- Gates: the `arange_full` oracle tier (zero excuses) and `Creation/np.arange.NumPyPort.Test.cs`.

## `np.ndarray`

`np.ndarray(shape, dtype=float64, buffer=None, offset=0, strides=None, order='C')` is NumPy's low-level constructor
(`ndarray.__new__`, `array_new` in `arrayobject.c`: `Creation/np.ndarray.cs` converts the C# arguments and calls
`UnmanagedStorage.ArrayNew`, `Backends/Unmanaged/UnmanagedStorage.Ctors.cs`).
- Without a buffer it is `np.empty`'s allocation: uninitialized, C or F strides; an empty result has all-zero strides,
  as NumPy allocates it; explicit byte `strides` give an owning array under that layout, and a stride of 0 stays
  writeable.
- With a buffer it lays the array over the buffer's bytes as `dtype` without copying: `offset` is a byte offset,
  `strides` are byte strides, the shape `(-1,)` takes every whole element, and a larger buffer is read from its start.
- The checks run in NumPy's order (buffer, order, strides length, `PyArray_CheckStrides` with C's wrapping arithmetic,
  `-1`/too small, dimensions) with NumPy's texts.
- Buffers (`Backends/Unmanaged/NDBuffer.cs`, the port of `PyArray_BufferConverter`):
  - a C- or F-contiguous `NDArray` (else `ValueError: ndarray is not contiguous`) or its `.data` memoryview (else
    `BufferError: memoryview: underlying buffer is not contiguous`) is borrowed: the result counts its references on the
    buffer array's block, and its base collapses to the owner;
  - a C# array of any rank whose elements hold no references, `ArraySegment<byte>` / `Memory<byte>` (writable) and
    `ReadOnlyMemory<byte>` (read-only for good, NumPy's `bytes`) are pinned until the last array over them is released
    or collected (a failed call frees the pin at once);
  - `Half`/`char`/`decimal` export a read-only copy of their bytes (NumPy scalars);
  - anything else is `TypeError: a bytes-like object is required, not 'list'`, with the Python type name.
- The result is one storage (`UnmanagedStorage.OverBuffer`) over the buffer array's own slice when dtype and element
  grid match, else over a forwarding alias of the occupied span (`ArraySlice.WrapShared(NPTypeCode, …)`).
- Stricter than NumPy: after NumPy's checks pass, an exact byte-range check refuses the four inputs NumPy lays outside
  the buffer (a negative offset, `-1` with a negative stride, an offset near 2^63, strides over an empty buffer).
- Divergences (`[Misaligned]`): a stride that splits an element is `NotSupportedException` (a stride that addresses no
  element is stored truncated); a non-native dtype over a buffer decodes into a new array.
- An `[Obsolete(error: true)]` overload makes `np.ndarray(shape, dtype, buffer, 'F')` a compile error: a `char` fourth
  argument would otherwise bind `long offset`.
- `flags.aligned` is computed from memory (`UnmanagedStorage.IsDataAligned`: NumSharp's whole-element strides make the
  storage's base address decide), so an odd-offset buffer and every view of it report unaligned,
  `setflags(align=True)` raises NumPy's ValueError there, and `np.require(…, 'A')` copies.
- Gates: the `ndarray_ctor.jsonl` oracle tier (values plus a "layout" facet of flags and byte strides, zero excuses) and
  `Creation/np.ndarray.Test.cs`.

## The `as*` family

- `asarray_chkfinite(a, dtype=None, order='K')` is `asarray`, then `ValueError("array must not contain infs or NaNs")`
  when a float-family dtype (Half/Single/Double/Complex, NumPy's `typecodes['AllFloat']`; Decimal, integers and bool skip
  the check) holds an inf or NaN. The check is a fused single-pass NaN-poison SIMD reduction
  (`Backends/Kernels/FiniteScan.cs`: `acc += v - v` adds +0 for a finite value and an absorbing NaN otherwise; AVX2
  gather and a reversed-contiguous fast path for strided and negative-stride views). Its statements are
  `NDArray.asarray_chkfinite` (`Manipulation/NDArray.FunctionBase.cs`).
- `require(a, dtype=None, requirements=None)` parses the C/F/A/W/O/E flags and their aliases (a single string iterates
  by character like NumPy, so `"F_CONTIGUOUS"` as one string raises), resolves an order, and copies only when a remaining
  ALIGNED/WRITEABLE/OWNDATA flag is unsatisfied (ALIGNED fails after `setflags(align=False)` or over a caller's buffer at
  a misaligning byte offset; WRITEABLE for broadcast views; OWNDATA for views).
- `asmatrix(data, dtype=None)` is `matrix(data, dtype, copy=False)` and returns an `NDMatrix` (`matrix.md`): a new view of
  an array, never the operand itself. 0-D → (1,1), 1-D → (1,N), 2-D unchanged; more than 2-D drops length-1 axes and must
  land on 2-D, else `ValueError("shape too large to be a matrix.")`. It also parses matrix literals (`"1 2; 3 4"`).

## `np.array` / `np.asanyarray` of a C# object list

They are NumPy's coercion of a Python list (`NDArray.asanyarray` in `Creation/NDArray.Multiarray.cs`, over
`Backends/Unmanaged/UnmanagedStorage.Coercion{,.Sequence,.Text}.cs`).
- An `object[]`, a jagged `object[][]`, an `object[,]` grid (a list of rows), a tuple and a `List<object>` / `ArrayList`
  read as Python values: a C# integer is a Python int, a `float` / `double` a Python float, `null` None, a typed C# array
  or an `NDArray` an ndarray item, `Half` / `char` / `decimal` NumPy scalars.
- Without a dtype NumPy's dtype is discovered (`[1, 2]` int64, `[1, 2.5]` float64, `[2**64-1, 1]` float64; None or a
  Python int past 64 bits is NumPy's object array, refused). With one, every scalar item is stored by the class's
  `setitem` (NumPy's `PyArray_Pack`: 300 into int8 raises `OverflowException`, a complex into float64 `TypeError`, None
  NaN or `int()`'s TypeError) and every array item is cast.
- Two walks serve it. Lists holding text (or under a text request) take the text builder
  (`UnmanagedStorage.Coercion.Text.cs`: discovery through `DiscoverDTypeFromPyObject` / `FindScalarDescriptor` /
  `DTypePromotion.PromoteTypes`, the stores through `DTypeMeta.SetFreshItems`, whose `StringDType` run reserves its arena
  once). The rest take the sequence walk (`UnmanagedStorage.FromSequence(seq, requested)`), which is also the array_like
  reader of numpy.polynomial, numpy.char, numpy.matrix, np.histogramdd and np.gradient.
- An ndarray item of a text dtype in a list is refused (the sequence walk has no text cast); an empty object grid keeps
  its own dims (`object[0, 3]` is `(0, 3)`); the order is checked before anything is converted.
- Gates: `Creation/np.array.PythonList.Test.cs` and the coercion family's snapshot (`tools/facade-move/snapshot.cs`).

## `np.full` / `np.full_like` / `np.ones` / `np.eye`

They are the array object's: `NDArray.full` / `full_like` / `ones` (`Backends/NDArray.Numeric.cs`, numeric.py's
`a = empty(...); copyto(a, fill_value, casting='unsafe')`) and `NDArray.eye` (`Backends/NDArray.TwoDimBase.cs`), over
the storage-lane fills `UnmanagedStorage.FullCore` / `FullFillElement` / `OnesCore` (`UnmanagedStorage.Ctors.cs`).
- The fill value is read as NumPy's `asarray(fill_value)` without a dtype: a 0-d array keeps its dtype, a `BigInteger` is
  a Python int (int64, uint64, or the object-array refusal), except that a C# number keeps its C# type, as `np.asarray(7)`
  does (`np.full(3, 7)` is int32 where NumPy's is int64).
- An empty result converts nothing but a Python int, float or complex (NumPy's copyto packs those before the copy and
  casts the rest per element written).
- The ones of np.ones and np.eye are the Python int 1 stored by the class's `setitem` (np.eye of a
  `StringDType(coerce=False)` raises, as NumPy's does); the `StringDType` routes ask `HasReferences`.
- np.eye checks in np.zeros' order (the order's letter, the device, the C-or-F rule, the dimensions); the `_like` forms
  check the order before the device.
- Gates: the constructors' snapshot (`tools/facade-move/snapshot.cs`) and
  `Creation/np.{eye.NumPyOrder,like.OrderBeforeDevice,full.FillValues}.Test.cs`.

## `np.frombuffer`

NumPy's `PyArray_FromBuffer`, one body for every kind of buffer. The overloads (`Creation/np.frombuffer.cs`) are NumPy's
`array_frombuffer`: they convert their C# arguments (an `NPTypeCode`: `Empty` is the default, `String` is NumPy's
unsized `np.str_`, refused; a dtype name through `np.dtype`, a null name being None, float64) and call
`UnmanagedStorage.FromBufferObject` (every buffer np.ndarray takes: a `byte[]` or typed C# array, a `MemoryView`, a
`Memory<byte>`, an `ArraySegment<byte>`), `FromBufferPointer` (an address) or `FromBufferSpan` (a `ReadOnlySpan<byte>`:
a copy) in `Backends/Unmanaged/UnmanagedStorage.Buffer.cs`.
- The checks run in NumPy's order: REFCHK (`HasReferences`: "cannot create an OBJECT array from memory buffer"),
  ISUNSIZED, the buffer (a C# null; a memoryview must be C-contiguous: `BufferError`), the offset, then the count (the
  buffer-size rule only for a negative count; a count past the buffer is refused, never capped).
- The result is np.ndarray's view: it does not own its data, it is writeable exactly when the buffer is (a read-only
  memoryview's result for good), and a segment or memory ends at its own end.
- A non-native descriptor is decoded into a native copy by the class's `copyswapn` (NumPy keeps the order in the
  descriptor), one SIMD pass per contiguous run (`Utilities/NDByteOrder.cs`).
- The generic overload's default dtype is its element type (a NumSharp extension: NumPy's is float64).
- Gates: the snapshot (`tools/facade-move/snapshot.cs`) and
  `Creation/np.frombuffer.{PyArrayFromBuffer,DtypeNames}.Test.cs`.

## `np.linspace` / `np.logspace` / `np.geomspace`

NumPy 2.4.2's `numpy/_core/function_base.py`: `Creation/np.{linspace,logspace,geomspace}.cs` are the facades,
`TensorEngine.Linspace` / `Logspace` / `Geomspace` (`Backends/TensorEngine.CoreFunctionBase.cs`) the members,
DefaultEngine's fills in `Backends/Default/Creation/Default.CoreFunctionBase.cs`. `start`, `stop` and `base` are scalars
(`double`); array-like values are not supported. Otherwise full parameter parity: `num=50`, `endpoint=true`,
`base=10.0` (logspace only), `dtype=null`, `axis=0` (int and long `num` overloads).
- `linspace(start, stop, num, endpoint, dtype, device)` runs NumPy's statements in NumPy's order: the count's ValueError
  first, then the device, then `arange(num) * step + start` with NumPy's three observable branches (one sample is
  `0 * (stop - start) + start`: NaN for an infinite or NaN end, +0.0 for a start of -0.0; a step that underflows to zero
  takes gh-5437's `(i / div) * delta + start`; `y[-1] = stop` for two or more samples). Integer dtypes are FLOORED, a text
  dtype is formatted by `astype` (`dtype='U5'` → `['0.0', '0.25', ...]`), and the result is NumPy's float64 view
  (OWNDATA false) wherever NumPy's in-place ramp is one.
- `axis` changes no value for scalar inputs (the output is always 1-D) but is validated like NumPy's trailing
  `moveaxis(y, 0, axis)`: only 0 and -1 are in range, else
  `AxisError("destination: axis {a} is out of bounds for array of dimension 1")`, reporting the original axis.
  geomspace's `axis=-1` returns NumPy's moveaxis view (OWNDATA false) unless the final dtype cast copies. `num < 0` →
  `ValueError("Number of samples, {num}, must be non-negative.")`.
- `logspace` is `power(base, linspace(start, stop, num, endpoint))`, then `.astype(dtype)`. It always computes in
  float64 and only then casts, so an integer dtype truncates toward zero (`linspace` floors), and `dtype=complex128` is
  real + 0j, not a complex-domain computation. Bit-exact at every dtype (`Math.Pow` is `npy_pow`, and the exponents are
  NumPy's linspace values: `LinspaceValue`, the branches above). A negative base with fractional exponents yields NaN.
  One fused pass (`LogspaceCore`), no intermediate `linspace` array.
- `geomspace` is ported op for op: `out_sign = sign(start)`, rotate `start`/`stop` onto the positive real axis,
  `logspace(log10(start), log10(stop), num, endpoint, base=10)`, overwrite the endpoints, `result *= out_sign`. The real
  path is bit-exact (decreasing, all-negative and mixed-sign inputs included: `geomspace(-1, 1)` → `[-1, nan, nan, 1]`).
  Zero endpoints raise `ValueError("Geometric sequence cannot include zero")` first (before `num`, before `axis`); a text
  dtype then fails NumPy's next statements, still before `num` and `axis` (`StringDType`: `result_type(start, stop,
  float(num), zeros((), dtype))`'s `DTypePromotionError`; `S` / `U`: the ends promote to text and `sign(start)` has no
  string loop, NumPy's UFuncTypeError text).
- A `Complex` start/stop overload and a real input with `dtype=complex128` compute in the complex128 domain (the interior
  differs: `geomspace(1, 8, 4, dtype=complex)[1] == 1.9999999999999998`), through NumSharp's `npy_cpow` port and complex
  log10: accurate within the ≤ 3-ULP complex envelope but not byte-reproducible, so those cases are allclose unit tests
  and stay out of the byte corpus.
- `Math.Pow` is the speed ceiling: no bit-exact SIMD `pow` exists.
- Gates: `Creation/np.logspace.geomspace.Test.cs` and the `creation` oracle tier.

## Set routines

`intersect1d`, `union1d`, `setxor1d`, `setdiff1d` (NumPy's `_arraysetops_impl.py`) are compositions of
`unique`/`sort`/`argsort`/`concatenate`, ported line for line, so promotion, NaN handling and edge cases match by
construction. `Manipulation/np.{union1d,intersect1d,setxor1d,setdiff1d}.cs` are the facades; `union1d`, `setxor1d` and
`setdiff1d` are the engine members `TensorEngine.Union1d` / `Setxor1d` / `Setdiff1d`
(`Backends/Default/Manipulation/Default.SetOps.cs`).
- `union1d(ar1, ar2)` = `unique(concatenate(ravel(ar1), ravel(ar2)))`.
- `intersect1d` has three overloads that reproduce NumPy's bare-array-or-tuple return: `intersect1d(ar1, ar2)` and
  `intersect1d(ar1, ar2, assume_unique)` return `NDArray`; `intersect1d(ar1, ar2, assume_unique=false,
  return_indices=false)` returns `NDArray[]{values, comm1, comm2}` with both keywords optional, so
  `np.intersect1d(a, b, return_indices: true)` ports verbatim (C# betterness binds the two-argument call to the bare
  form). The tuple path uses a stable argsort, so the first occurrence in each input wins.
- `setxor1d` returns the values in exactly one input; `setdiff1d(ar1, ar2, assume_unique=false)` returns ar1's values
  absent from ar2 (built on `isin(..., invert=true)`), keeping ar1's dtype and, under `assume_unique`, its duplicates.
- Empty inputs give NumPy's float64 empty (from the empty concatenate).
- NaN never matches itself: a shared NaN is not in `intersect1d`, and each NaN survives `setxor1d`/`union1d`/`setdiff1d`.
- A surviving float32/float64 NaN is NumPy's positive quiet NaN (`0x7ff8000000000000` / `0x7fc00000`): NumPy's SIMD
  sort rewrites every NaN to it, while float16 and complex128 NaNs keep their bits. `np.sort` writes that NaN
  (`AxisSort.SortLineF32/F64`), but `unique` keeps payloads, so `union1d`/`setxor1d`/`setdiff1d` run a guarded rewrite
  on a float32/float64 result (`DefaultEngine.CanonicalizeSetOpNaN`: `np.where(np.isnan(r), canonical, r)`, entered
  only when `np.any(np.isnan(r))`). `isin` (a bool result) and `intersect1d` (no shared NaN) need none.
- Gates: `Manipulation/np.setops.Test.cs` and the `groupa` tier.

## Views and reshuffles

- `flip`/`fliplr`/`flipud` return O(ndim) views (stride negation plus a base-offset shift through `Storage.Alias`), the
  same as the `m[..., ::-1, ...]` slice for every layout (a flip of a read-only broadcast stays read-only).
  `flip(m, axis: null)` flips all axes; `flip(m, int[])` is the tuple form (an empty array flips nothing; axes are
  normalized NumPy-style: the full `AxisError` pass, reporting the original negative axis, then
  `ValueError("repeated axis")`, so `(0,0,5)` raises AxisError). 0-d: `flip(m)` returns the scalar (NumPy's `m[()]`),
  `flip(m, 0)` raises AxisError. `fliplr` needs ndim ≥ 2 and `flipud` ndim ≥ 1 (`ValueError("Input must be >= 2-d." /
  ">= 1-d.")`).
- `rot90(m, k=1, axes=(0,1))` rotates by `90°·k` from the first axis toward the second, as a view composition of flips
  and a transpose (it shares memory, read-only when the source is). `k` is taken mod 4 (negatives wrap). Validation
  (`ValueError`, NumPy's own type and texts, not normalize_axis_index's AxisError): `len(axes) != 2` → "len(axes) must
  be 2."; `axes[0] == axes[1] || |axes[0]-axes[1]| == ndim` → "Axes must be different." (checked before the range, so
  `axes=(0,2)` on 2-D reports "different"); else out of range.
- `trim_zeros(filt, trim="fb", axis=null)` crops leading/trailing all-zero hyperplanes (NumPy 2.2+'s N-D bounding box).
  `trim` is case-insensitive and validated (`ValueError "unexpected character(s) in \`trim\`: '…'"`); `axis=null` trims
  every axis, an int or `int[]` only those (NumPy's `normalize_axis_tuple(axis, ndim, argname="axis")`: every axis
  range-checked first, `AxisError "axis: axis 2 is out of bounds for array of dimension 2"`, then `ValueError "repeated
  axis in \`axis\` argument"`); an empty axis list or a 0-d input returns the input. Always a view; all-zero/empty inputs
  collapse the trimmed axes to length 0. The bounding box is separable, so it takes the first/last non-zero index from
  `np.nonzero` (1-D) or `np.nonzero(np.any(filt, other_axes))`, touching only the trimmed axes. A test of the values
  decides the view, so the body is the engine's: `TensorEngine.TrimZeros`
  (`Backends/Default/Manipulation/Default.TrimZeros.cs`).
- `permute_dims(a, axes=null)` is `transpose`. `matrix_transpose(x)` swaps the two innermost axes and needs ndim ≥ 2
  (`ArgumentException "Input array must be at least 2-dimensional, but it is {ndim}"`); `ndarray.mT` does the same but
  raises `"matrix transpose with ndim < 2 is undefined"`. Read-only inputs stay read-only.
- Transpose axes (shared by `transpose`/`permute_dims`/`swapaxes`/`moveaxis`/`rollaxis`): only `axes=null` reverses; an
  explicit empty `axes` matches only a 0-d array (`ValueError "axes don't match array"` for ndim ≥ 1); wrong length and
  repeated axes raise `ValueError` ("axes don't match array" / "repeated axis in transpose"); out-of-range axes
  `AxisError`. `np.moveaxis` names its arguments (`AxisError "source: axis 5 …"` / `"destination: …"`, `ValueError
  "repeated axis in \`source\` argument"`), `np.swapaxes` too (`"axis1: …"` / `"axis2: …"`), and `np.rollaxis(a, axis,
  start)` raises NumPy's message-only `AxisError "'start' arg requires -2 <= start < 3, but 4 was passed in"` (start
  reported after its negative wrap). Every transpose is a view, an empty array's too (gate
  `Manipulation/np.transpose.Empty.Test.cs`; the body is `Backends/Default/ArrayManipulation/Default.Transpose.cs`).

## Stacking and joining

Tests in `Creation/np.{block,column_stack,concat,unstack}.Test.cs`.
- `concat` is NumPy 2.0's alias of `concatenate`, with every overload (array form with `axis`/`out`/`dtype`/`casting`,
  tuple-arity forms).
- `unstack(x, axis=0)` returns views (write-through), each `x.shape` minus the axis, built by dropping the axis from
  dims/strides and advancing the offset. A 1-D input gives 0-d views; a 0-d input raises
  `ValueError("Input array must be at least 1-d.")`.
- `column_stack(tup)`: inputs below 2-D become columns through a single-alias `(N,1)` view (NumPy's
  `array(a, ndmin=2).T`), then `concatenate(axis=1)`; ndim ≥ 2 passes through.
- `block(arrays)` ports `numpy/_core/shape_base.py`: the depth-check walk (NumPy's ValueError texts for depth mismatch
  and empty lists; `TypeError` for `ITuple`s), `_atleast_nd` padding views, and both algorithms behind NumPy's threshold
  `list_ndim * final_size > 2*512*512`: repeated concatenation (intermediates disposed eagerly) or the single-allocation
  `_block_slicing` (per-leaf `NDIter.Copy`; `order='F'` if all inputs are F and not all C). Nesting: `object[]`,
  `NDArray[]`, 1-D primitive arrays and `IList` are Python lists; `NDArray`, scalars and rank ≥ 2 rectangular arrays are
  leaves; `np.block(x)` at depth 0 returns a copy. A C# scalar leaf keeps its C# type (`10` is int32, a Python int
  int64).
- `concatenate`'s layout vote: F output only when every input is F-contiguous and not all are also C-contiguous (inputs
  that are both, such as `(N,1)` columns, give C, as `PyArray_CreateMultiSortedStridePerm`). When every per-outer slab
  is the same 1/2/4/8/16 bytes (the `(N,1)` column pattern), it interleaves with constant-size loads and stores.
- `UnmanagedStorage.GetData<T>()` narrows to the shape's `[offset, offset+size)` window before returning or casting
  (zero-copy, writes through), so a contiguous view at a non-zero offset returns only its own elements.
- `np.resize(a, new_shape)` fills the output with repeated copies of `a` in C order (an exact-sized doubling byte tile;
  an empty source or zero size gives zeros; always C-contiguous; any layout is raveled first). `ndarray.resize(new_shape,
  refcheck=true)` works in place: it grows with zeros, shrinks by truncation, and operates on the raw contiguous buffer
  (an F-contiguous resize relabels memory column-major). It mirrors NumPy's guards (`IncorrectShapeException`): single
  segment only; when the byte size changes it must own its data and, under `refcheck`, not be shared
  (`IArraySlice.IsUniquelyReferenced`, the ARC refcount); `refcheck: false` bypasses. A same-size resize is a pure
  in-place reshape. `Manipulation/np.resize.cs` and `Manipulation/NDArray.resize.cs` are the facades over
  `UnmanagedStorage.ResizeRepeating` (`Backends/Unmanaged/UnmanagedStorage.ItemSelection.cs`) and
  `UnmanagedStorage.Resize` (`Backends/Unmanaged/UnmanagedStorage.Reshaping.cs`, NumPy's `PyArray_Resize`).
- `ndarray.tolist()` is NumPy's `recursive_tolist` for every dtype (`Manipulation/NDArray.tolist.cs`): one
  `List<object>` per axis position, walked by the strides from the array's first element (`Shape.offset`), sized once
  (`CollectionsMarshal.SetCount`), each innermost row filled through its span by one
  `IArraySlice.GetItems(start, stride, span)` call. `GetItems` is the run form of `GetIndex(long)`: a default interface
  method (`GetIndex` per element) that the three library slices override with exactly `GetIndex`'s values
  (`ArraySlice<T>` boxes from memory, a `Half` from its 16-bit pattern; `StringArraySlice` calls the codec per element;
  `VStringArraySlice` holds the allocator's lock once per run). Gates: `Manipulation/TolistTests.cs`,
  `Manipulation/TolistWalkTests.cs` (the per-element walk as the reference over every storage and layout kind).

## Allocation and reshape guards

Ported from NumPy's C (gates `Backends/AllocationGuardTests.cs`, `Manipulation/ReshapeErrorParityTests.cs`,
`Manipulation/ManipulationErrorParityTests.cs`).
- `AllocationGuard` (`Exceptions/AllocationGuard.cs`) is the dimension-check block of `PyArray_NewFromDescr_int`. A size
  computed by multiplication cannot be trusted until its overflow is checked, and the failure is silent by construction:
  a wrapped byte count looks like a small allocation, not an error. The running product starts at itemSize (the
  quantity checked is bytes, so `np.zeros(2^60)` float64 is rejected while the same shape of int8 is merely
  `OutOfMemoryException`, NumPy's `MemoryError`), and a zero dimension does not short-circuit the scan
  (`np.zeros((0, 2^62))` float64 still reports "array is too big"). Dims are scanned left to right, so the error follows
  NumPy's order (`(-1, 2^62)` is `negative dimensions are not allowed`, `(2^62, -1)` is `array is too big; ...`). There
  are two wrap points, the byte count (`(2^31, 2^31)`) and the element count itself (`(2^62, 2^62)`), so the guard keys
  off the dimensions, never `shape.size`. The fills behind `np.full` / `np.full_like` / `np.ones` allocate from
  `shape.size` directly and carry the call explicitly (`UnmanagedStorage.FullCore` / `OnesCore`, `NDArray.full_like`);
  `UnmanagedMemoryBlock<T>`'s two allocating constructors hold a byte-count backstop.
- `ndarray.resize` follows `PyArray_Resize_int`, which differs on three points: it breaks at the first zero dim
  (`a.resize((0,-1))` is legal and gives shape `(0,-1)`), its text omits "are" (`negative dimensions not allowed`), and
  an overflowing product is a `MemoryError`. It allocates by element count and relabels.
- `Shape.Reshape` is `_fix_unknown_dimension` + `convert_shape_to_string`. Any negative dim is the unknown one
  (`reshape(-3)` of 6 elements is `(6,)`, `reshape(3,-5)` is `(3,2)`). The zero product is tested before the division
  (`s_known == 0 || size % s_known != 0`). Every rejection but the multiple-unknown one carries NumPy's text
  `cannot reshape array of size {n} into shape {s}`, where `{s}` follows `convert_shape_to_string`: leading unknowns are
  dropped (`reshape(-1,0)` → `(0)`), later ones read `newaxis` (`reshape(0,-1)` → `(0,newaxis)`), and a one-element shape
  closes with `,)` (`reshape(0)` → `(0,)`). The type is `IncorrectShapeException`; the multiple-unknown case is
  `ValueError` ("can only specify one unknown dimension").
- `np.expand_dims(a, int)` validates the axis against the output ndim (`a.ndim + 1`) and reports it as given, a 0-d input
  included. The guard lives at the `np.*` boundary, not in `Shape.ExpandDimension`, whose internal callers (keepdims
  reductions, fancy-index shaping, `asmatrix`) are not bound by NumPy's axis contract.
- Matched, no need to re-prove: the values of `split`/`array_split`/`hsplit`/`vsplit` × {0, negative, empty},
  `repeat`/`tile`/`roll` × {0, negative, empty axis}, `squeeze`, `moveaxis`, `swapaxes`.
- Text divergences (each typed and texted on both sides): `np.split(a, 0)` raises "number sections must be
  larger than 0." where NumPy leaks `ZeroDivisionError: integer modulo by zero`; `repeat(-1)` and `tile(-1)` texts
  differ in wording; `np.broadcast_to(a, (2^62, 6))` builds the view where NumPy refuses with `iterator is too large`.
  The axis errors of this family (`split(..., axis=oob)`'s leaked `IndexError`, `squeeze`) are NumPy's
  (`axis-arguments.md`).

## `np.copyto`

`np.copyto(dst, src, casting='same_kind', where=True)` is NumPy's `array_copyto` → `PyArray_AssignArray` /
`PyArray_AssignRawScalar`, statement for statement (`Manipulation/np.copyto.cs`, the facades over
`UnmanagedStorage.CopytoAssignArray` / `CopytoAssignRawScalar` in `Backends/Unmanaged/UnmanagedStorage.Assign.cs`).
- Arguments: the casting name first (`ValueError` "casting must be one of …", `TypeError` for null), then the mask's dtype
  (NumPy's "Cannot cast array data|scalar from dtype('int64') to dtype('bool') according to the rule 'safe'", as
  `ArgumentException`). A C# integer / `float` / `double` / `Complex` binds `copyto<T>` and is NumPy's weak Python scalar
  (`npy_find_descr_for_scalar`, stored with `PyArray_Pack`'s range checks: `np.copyto(int8, 1000)` raises "Python integer
  1000 out of bounds for int8", never wraps; `'equiv'` refuses any dtype but the scalar's own default). A `BigInteger` is
  a Python int of any size. `bool`/`Half`/`char`/`decimal` and every `NDArray` (0-d too) are strong: a 0-d int64 into
  int8 wraps, as `np.array(1000)` does.
- Checks in NumPy's order: a 0-d source goes to AssignRawScalar ("Cannot cast scalar from …"); a redundant self-copy
  (same data, dtype, shape, strides) returns before the writeable check (`np.copyto(b, b)` on a read-only view is legal);
  read-only → "assignment destination is read-only"; the cast rule (`InvalidCastException`, NumPy's text); an overlapping
  source is copied first unless dst is 1-D with both runs in the same direction; the source broadcast, with its leading
  extent-1 axes beyond dst's rank stripped ("could not broadcast input array from shape (2,3) into shape (3,)"); a 0-d
  mask (after the source check; false copies nothing); the mask broadcast, never stripped ("could not broadcast where
  mask from shape (1,3) into shape (3,)").
- The masked copy is a bit copy of dst's element width (`DirectILKernelGenerator.MaskedAssign.cs`: one kernel per width
  1/2/4/8/16 serves all 15 dtypes) over NumPy's prepared iteration (`Backends/Iterators/RawArrayIter.cs` =
  `PyArray_PrepareThreeRawArrayIter`: axes by dst's |stride|, negative dst strides flipped, coalescing) plus the 1-D
  backward rule. An IL walker drives a run stage: a SIMD blend for contiguous runs (`ConditionalSelect`, AVX2 `vpmovzx`
  mask widening, a hoisted vector for a broadcast source), a branch-free select off running pointers for strided runs
  and tails, memmove / fill for a constant-true run. With more than one prepared axis the operands cannot overlap, so
  the innermost axis is the one where most operands are contiguous (NumPy's order wins ties;
  `UnmanagedStorage.CopytoNumPyOrderOnly` is the A/B hook). A source of another dtype, or an overlapping one NumPy
  copies first, is staged in dst's memory order (one element per distinct source element, on the stack up to 16 KB, else
  a pooled buffer) through NDIter.Copy's cast kernels.
- The unmasked copy is `NDIter.Copy`, whose overlap rule is NumPy's for every caller (slice assignment too): copy first
  for a multi-axis destination or an opposite-direction 1-D one; otherwise NumPy's raw 1-D order
  (`DirectILKernelGenerator.AssignRaw1D`), which reads its own writes when the strides differ (`a[3:19:2] = a[6:14]`).
- Divergences (house conventions): the exception types (TypeError → `ArgumentException` for the mask,
  `InvalidCastException` for casts; OverflowError → `OverflowException`); a 0-d non-bool mask always raises (NumPy takes
  `where=1`); LP64 texts for a Python int past int64; one complex width.
- Gates: `copyto_full.jsonl` (zero excuses) and `Manipulation/np.copyto.NumPyPort.Test.cs`.
