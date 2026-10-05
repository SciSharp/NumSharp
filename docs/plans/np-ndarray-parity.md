# `np.ndarray` — audit against NumPy 2.4.2, fix plan and progress

> The view-Shape plan (`docs/plans/view-shape-open-bugs.md`, complete 2026-10-05) left one follow-up for this
> function: "NumSharp's own `np.ndarray(shape, dtype, buffer, order)` defaults to `order='F'` and float32 (NumPy:
> `order=None`, i.e. C, and float64), and ignores `order` without a buffer". The audit below compares every
> parameter and every observable result of `np.ndarray` with NumPy. It found more than the two defaults: the buffer
> is not read as bytes, `offset=` and `strides=` do not exist, and the result's flags and errors differ.
>
> Each item is fixed only after three proofs: NumPy's behavior, NumSharp's divergence, and that the divergence is a
> parity bug and not a documented `[Misaligned]` choice.
>
> This plan covers item D7 of `docs/plans/api-parameter-parity.md` (`offset=`, `strides=`), which waited for the
> view-Shape work. That plan belongs to another session and is not edited here.

Source of truth (`refs/numpy`, 2.4.2):
- `numpy/_core/src/multiarray/arrayobject.c`: `array_new` (`ndarray.__new__`) and `PyArray_CheckStrides`;
- `numpy/_core/src/multiarray/conversion_utils.c`: `PyArray_IntpConverter` (shape), `PyArray_BufferConverter`
  (buffer), `PyArray_OptionalIntpConverter` (strides), `PyArray_OrderConverter` (order);
- `numpy/_core/src/multiarray/ctors.c`: `PyArray_NewFromDescr_int` and `_array_fill_strides`;
- `numpy/_core/src/common/mem_overlap.c`: `offset_bounds_from_strides`;
- `numpy/_core/src/common/array_assign.c`: `raw_array_is_aligned`.

Probes (2026-10-05): NumPy 2.4.2 (win-amd64) and NumSharp at `ec230a8b`, the same 185-case matrix on both sides.
The scripts and their outputs are kept with the session (`nd_probe.py`, `nd_probe_before.cs`, `nd_diff.py`). Result
at `ec230a8b`: 21 cases match, 89 differ, and 75 cannot be expressed with NumSharp's signature.

## Progress

Legend: `[ ]` open, `[~]` in progress, `[x]` done. P = NumPy's behavior proved, D = NumSharp's divergence proved,
B = confirmed as a parity bug, F = fixed and gated.

| # | Item | P | D | B | F | Commit |
|---|---|---|---|---|---|---|
| N1 | `dtype` defaults to float32 (NumPy: float64) | [x] | [x] | [x] | [x] | this change |
| N2 | `order` defaults to `'F'` (NumPy: `None`, i.e. C) | [x] | [x] | [x] | [x] | this change |
| N3 | Without a buffer, `order` is ignored and not validated | [x] | [x] | [x] | [x] | this change |
| N4 | With a buffer, `dtype` is ignored (NumPy reads the buffer's BYTES as `dtype`) | [x] | [x] | [x] | [x] | this change |
| N5 | Buffer kinds: only `System.Array`; no `NDArray`, `ndarray.data`, NumSharp slice; wrong errors | [x] | [x] | [x] | [x] | this change |
| N6 | Buffer size: too small and larger buffers (B1.10, B1.11) | [x] | [x] | [x] | [x] | this change |
| N7 | No `offset=` parameter | [x] | [x] | [x] | [x] | this change |
| N8 | No `strides=` parameter | [x] | [x] | [x] | [x] | this change |
| N9 | Shape `-1` with a buffer | [x] | [x] | [x] | [x] | this change |
| N10 | The result over a buffer reports `OWNDATA` and no base | [x] | [x] | [x] | [x] | this change |
| N11 | Flags: `ALIGNED`, writeable zero strides, empty strides | [x] | [x] | [x] | [x] | this change |
| N12 | Error order | [x] | [x] | [x] | [x] | this change |
| N13 | The wrap constructors (`new NDArray(Array, Shape, char)`) share N6 | [x] | [x] | [x] | [x] | this change |

All 13 items are fixed together, because they are one function's argument handling (the commit that lands this plan
is the one to cite; its message lists every item). Status: the 185-case probe matches NumPy on 167 cases (21 before);
the other 18 are the documented divergences of §3. Gates: the `ndarray_ctor.jsonl` oracle tier (5,919 cases, zero
excuses) and `Creation/np.ndarray.Test.cs` (39 tests).

---

## 1. NumPy's `ndarray.__new__`

Signature: `ndarray(shape, dtype=float, buffer=None, offset=0, strides=None, order=None)`.

`PyArg_ParseTupleAndKeywords` runs the six converters in this order: `shape`, `dtype`, `buffer`, `offset`,
`strides`, `order`. Then `array_new` does the following:

1. `order == 'F'` (either case) selects the F layout. `'C'`, `'A'`, `'K'` and `None` select C. Any other string is
   `ValueError: order must be one of 'C', 'F', 'A', or 'K' (got 'Q')`.
2. `dtype=None` is float64 (`NPY_DEFAULT_TYPE`).
3. With `strides`: a length other than the shape's is `ValueError: strides, if given, must be the same length as
   shape`. Then `PyArray_CheckStrides(itemsize, nd, numbytes, offset, dims, strides)` must hold, else
   `ValueError: strides is incompatible with shape of requested array and size of buffer`. Without a buffer,
   `numbytes` is `prod(shape) * itemsize` and `offset` is 0.
4. Without a buffer: `PyArray_NewFromDescr_int` allocates `prod(shape) * itemsize` bytes, UNINITIALIZED (as
   `np.empty`). It fills C or F strides, or copies the given strides. A zero-size array gets ALL-ZERO strides (the
   given strides too). `offset` is ignored.
5. With a buffer:
   - `shape == (-1,)` (one dimension, exactly -1) is `(len(buffer) - offset) // itemsize` (C division);
   - else, without `strides`, `len(buffer) < offset + itemsize * prod(shape)` is `TypeError: buffer is too small for
     requested array`. A larger buffer is allowed: the array uses a prefix;
   - the array starts at `buffer + offset`, with C or F strides from `_array_fill_strides` (which does not multiply
     zero extents in, so `(3, 0)` is `(8, 8)`), or the given strides.
6. `PyArray_NewFromDescr_int` checks the dimensions: `ValueError: negative dimensions are not allowed`, `ValueError:
   array is too big; ...`.

The buffer converter (`PyArray_BufferConverter`) accepts any object with the buffer protocol: `bytes` (read-only),
`bytearray`, `memoryview`, an `ndarray` that is C- or F-contiguous (else `ValueError: ndarray is not contiguous`),
`array.array`, a NumPy scalar. Anything else is `TypeError: a bytes-like object is required, not 'list'`. A read-only
buffer gives a read-only result whose `WRITEABLE` flag cannot be set again.

The result over a buffer does not own its data (`OWNDATA` false) and its `base` is the buffer object. `ALIGNED` is
computed from the data pointer and the strides (`raw_array_is_aligned`: a zero-size array is aligned; an axis of
extent 1 does not count).

NumPy does not check every bound. Three inputs give an array that reads OUTSIDE the buffer (probed):
- a negative `offset` without `strides` (`np.ndarray((2,), buffer=ba, offset=-8)`);
- `shape=-1` with a negative stride (`np.ndarray(-1, buffer=ba, strides=(-8,))`: the stride check runs while the
  dimension is still -1);
- an `offset` near 2^63 (`offset + nbytes` overflows the too-small check).

## 2. Comparison (NumPy 2.4.2 against NumSharp at `ec230a8b`)

NumSharp's signature: `ndarray(Shape shape, DType dtype = null, Array buffer = null, char order = 'F')`. Without a
buffer it returns `new NDArray(dtype, shape, order)` (zero-filled, `order` ignored). With a buffer it returns
`new NDArray(buffer, shape, order)` (the wrap constructor: dtype = the buffer's element type).

| Facet | NumPy 2.4.2 | NumSharp at `ec230a8b` | Probe cases | Verdict |
|---|---|---|---|---|
| Parameters | `shape, dtype, buffer, offset, strides, order` | `shape, dtype, buffer, order` | S01-S19, B05-B11 | N7, N8 |
| `dtype` default | float64 | float32 | A01-A16 | N1 |
| `order` default | `None` (C) | `'F'` | A01, N01 | N2 |
| `order='F'` without a buffer | F strides `(8, 16)` | C strides | A07-A10, A17F | N3 |
| invalid `order` without a buffer | `ValueError` | accepted | A15 | N3 |
| lowercase `order` | accepted | accepted | A13 | match |
| `dtype` with a buffer | the buffer's bytes read as `dtype` | ignored; the element type of the buffer array | B01, B48-B52, N08, N13 | N4 |
| buffer kinds | any buffer object | `System.Array` only | N02-N11 | N5 |
| non-buffer object | `TypeError: a bytes-like object is required, not 'list'` | `NotSupportedException` (object[]) | B54-B56 | N5 |
| non-contiguous `ndarray` buffer | `ValueError: ndarray is not contiguous` | not expressible | N04, N06 | N5 |
| buffer too small | `TypeError: buffer is too small for requested array` | `ArgumentException` | B03, B42 | N6 |
| buffer larger than needed | prefix used | `ArgumentException` | B04 | N6 |
| `offset=` | byte offset into the buffer; ignored without one | no parameter | A26-A28, B05-B11 | N7 |
| `strides=` | byte strides, checked by `PyArray_CheckStrides` | no parameter | S01-S19, B24-B40 | N8 |
| shape `-1` with a buffer | inferred | `ArgumentException` | B12-B17, B58-B60 | N9 |
| `OWNDATA`, `base` with a buffer | false, the buffer | true, none | B43, N01, N14 | N10 |
| values without a buffer | uninitialized | zero-filled | A01 | N1 (allocation) |
| strides of a zero-size result | all zero without a buffer; `_array_fill_strides` with one | C strides with zero extents multiplied in | A04-A06, A12, S09, B44-B46 | N11 |
| `ALIGNED` | from the data pointer and strides | always true | B07, B39, B40 | N11 |
| zero strides (`(0, 8)`) | writeable | (not expressible) | S06, S15, B30, B65 | N11 |
| error order | converters, strides length, `CheckStrides`, too small, dimensions | `order` checked after the dimensions | E01-E12 | N12 |
| invalid dtype string | `TypeError` | `NotSupportedException`, same text | A18, E02, E09 | documented house rule (`.claude/CLAUDE.md`, dtype system) |
| `complex64` | supported | no complex64 dtype (#569) | A17 complex64 | inherent |
| non-native byte order (`'>i4'`) | big-endian dtype | native dtype | A17 >i4 | inherent (NumSharp buffers are native) |

## 3. Decisions

**Signature.** `ndarray(Shape shape, DType dtype = null, object buffer = null, long offset = 0, long[] strides = null,
char order = 'C')`, NumPy's names and positions.
- `buffer` is `object`, as NumPy's buffer protocol is dynamic. An `NDArray`-typed parameter would bind every C# array
  through the implicit `Array -> NDArray` conversion, which copies, so the result would never share the caller's
  memory.
- `order` is a `char` with the default `'C'`: the house spelling of NumPy's `None` here (`np.asarray` uses `'K'` for
  its `None`). A multi-character order string is a NumPy `ValueError` that C# cannot express.
- The old fourth positional parameter was `order`. A C# `char` converts implicitly to the new `long offset`, so
  `np.ndarray(s, d, b, 'F')` would compile and silently read `offset = 70`. An `[Obsolete(error: true)]` overload
  `ndarray(Shape, DType, object, char offset)` catches that call at compile time and tells the caller to name
  `order:`.

**Buffers in C#** (`Backends/Unmanaged/NDBuffer.cs`, the port of `PyArray_BufferConverter`). NumPy's buffer protocol
maps to:
- `NDArray` (C- or F-contiguous; else `ValueError: ndarray is not contiguous`) and `np.MemoryView` (`ndarray.data`;
  over a non-contiguous array, CPython's `BufferError: memoryview: underlying buffer is not contiguous`): BORROWED. The
  bytes are read in memory order, writeability carries over, and the base collapses to the buffer array's owner, as
  `PyArray_SetBaseObject` does. The result counts its references on the buffer array's own memory block, so disposing
  the buffer array while the result lives frees nothing.
- `IArraySlice` (NumSharp's raw buffer): borrowed, writable.
- `System.Array` of any rank whose element type holds no references (`byte[]`, `double[,]`, a user struct): PINNED for
  as long as an array over it lives, writable. `ArraySegment<byte>` and `Memory<byte>` (writable) and
  `ReadOnlyMemory<byte>` (read-only for good, NumPy's `bytes`): pinned through `Memory<T>.Pin()`.
- `Half`, `char`, `decimal`, the C# values NumSharp treats as NumPy scalars: a read-only copy of the value's bytes, as
  NumPy's scalar types export (`np.ndarray((2,), np.uint8, buffer=np.float16(1.5))` is `[0, 62]`).
- Anything else: NumPy's `TypeError: a bytes-like object is required, not 'list'` with the Python type name of the C#
  value (`str`, `bool`, `int`, `float`, `complex`, `tuple`, `dict`, `set`, `list`; a reference-element array is a
  `list`).

The base of a result over a foreign buffer (a C# array, a `Memory`, a scalar) is a storage over the whole buffer that
does not own its data (`ExternalBase`), standing for NumPy's non-array base object. Its memory block owns the pin and
frees it when the last array over the buffer is released or collected; a call that fails after the pin frees it at
once (`NDBuffer.Release`).

**One storage per result.** The view is ONE storage (`UnmanagedStorage.OverBuffer`): over the buffer array's own slice
when the dtype and the element grid match (an ordinary view), else over an alias of exactly the occupied span whose
reference counting forwards to the buffer's owner (`ArraySlice.WrapShared(NPTypeCode, …)`). A buffer pinned here gets a
root slice typed as the result's dtype (`ArraySlice.WrapOwned`), so the common case shares it too.

**ALIGNED is computed from memory** (`UnmanagedStorage.IsDataAligned`). NumSharp's strides and offsets are whole
elements and every itemsize is a multiple of its alignment, so NumPy's `raw_array_is_aligned` reduces to the storage's
base address (an empty array is aligned). `flags.aligned` is the shape bit (cleared by `setflags(align=False)`) AND this;
`setflags(align=True)` raises NumPy's `ValueError: cannot set aligned flag of mis-aligned array to True` on misaligned
memory; `np.require(…, 'A')` copies an array that is not aligned (it assumed every array was). Every view of a
misaligned array shares its address, so the flag reaches every view without each view path computing it (probed:
NumPy's `[1:]`, `.T`, `reshape` of an offset-3 float64 array are unaligned, its copy aligned). This also corrects
`np.frombuffer(b, offset=3)` and odd-offset `getfield`, which reported ALIGNED.

**The wrap constructors** (`new NDArray(Array | IArraySlice, Shape, char)` and their `NDArray<T>` twins) follow NumPy's
size rule too (N13): a buffer smaller than the shape is `TypeError: buffer is too small for requested array`, a larger
one is read from its start (a prefix that shares the caller's memory), the lone `-1` takes every element, another
negative dimension is NumPy's ValueError. They keep their C# conveniences (the element type is the dtype, a jagged or
multi-dimensional array is flattened).

**What NumSharp refuses or reports differently.** Documented divergences, pinned by `[Misaligned]` tests in
`Creation/np.ndarray.Test.cs`:
- The four inputs NumPy lays OUTSIDE the buffer: a negative offset under a non-empty array, the shape `-1` with a
  negative stride, an offset near 2^63, and strides over an empty buffer. NumSharp checks the exact byte range
  (`Shape.ExactByteExtent`) after NumPy's own checks pass, and refuses with the strides `ValueError` when strides were
  given, else `ValueError: offset must be non-negative and no greater than buffer length (N)` (NumPy's `np.frombuffer`
  text) for a negative offset, else the too-small `TypeError`. An EMPTY array touches no memory, so every offset NumPy
  accepts for one is kept.
- A byte stride that addresses elements and splits one (`strides=(6, 2)` for int32): NumSharp stores strides in
  elements and cannot represent it (`NotSupportedException`). A byte OFFSET that splits an element is supported (the
  data pointer is misaligned and `ALIGNED` is false, as in NumPy). A stride that addresses no element (an axis of extent
  1, or any axis of an empty array) is accepted and stored truncated, so `strides` reports `(24, 0, 8)` where NumPy
  reports `(24, 4, 8)` (NumPy's own tests build `strides=(8, 80, 4)` on `(10, 5, 1)`).
- A non-native dtype with a buffer (`'>i4'`): the bytes are decoded into a new native array (the `np.frombuffer`
  precedent), so the result does not share the buffer and owns its data.
- `complex64` (#569), a multi-character order string, a bad dtype string's exception type (NumSharp's dtype parser
  raises `NotSupportedException` with NumPy's text, the house rule), and the order of a bad shape and a bad dtype (C#
  converts the dtype string at the call site, before `np.ndarray` runs).

## 4. Items

**N1, N2, N3 (defaults, order without a buffer).** The defaults are NumPy's: float64 and C order. Without a buffer the
array is NumPy's `np.empty` allocation (uninitialized, C or F strides from `Shape.Adopt(dims, order)`), and the order is
validated (`OrderResolver.ResolveBufferOrder`, NumPy's text) before anything else but the buffer. An empty allocation
has all-zero strides (`ctors.c`: "Make sure all the strides are 0"), which `np.zeros`/`np.empty` do not yet follow
(`OpenBugs.EmptyStrides.cs`).

**N4, N5 (dtype and buffer kinds).** The buffer's BYTES are read as the dtype; see "Buffers in C#".

**N6, N9, N12 (size, shape -1, error order).** NumPy's checks in `array_new`'s order: shape `None`, the buffer, the
order, the strides' length, `PyArray_CheckStrides` (`Shape.CheckStrides`, C's wrapping arithmetic so the verdict is
NumPy's even where its arithmetic overflows), the shape `-1` (`(len - offset) / itemsize`, C division) or the too-small
check (wrapping), then the dimensions (`AllocationGuard.CheckDimensions`: negative, too big). Then NumSharp's exact
bounds check and the element-stride conversion.

**N7, N8 (offset, strides).** `offset` is a byte offset into the buffer, ignored without one (as in NumPy). `strides` are
BYTE strides (NumPy's unit, which `ndarray.strides` reports), checked against the buffer, or against the array's own
size at offset 0 without one; a stride of 0 keeps the array writeable (NumSharp's flags call it a read-only broadcast,
so `WRITEABLE` is set back). Explicit strides without a buffer allocate an array that owns its memory under that
layout (`new UnmanagedStorage(slice, layout)`).

**N10, N11 (OWNDATA, base, flags).** A result over a buffer does not own its data and has a base; `WRITEABLE` is the
buffer's writability (a read-only buffer cannot be made writeable: `WriteProtected` on the foreign base, or the read-only
owner array); `ALIGNED` is computed from memory (§3).

**N13 (wrap constructors).** See §3.

**Performance** (NPY/NS, P-cores pinned, best of 9 rounds after a 300 ms warm-up, create + dispose per call; NumPy
2.4.2 `timeit`). Over a NumSharp array the call is 1.8–2.8× NumPy (`ndarray (2,3)` 2.33, `-1` as int32 1.82, a
1,000,000-element array 2.40, its memoryview 2.80): one view storage over the buffer array's own slice. Over a C#
`byte[]` it is 1.0–1.4× (`(2,3)` 1.00, offset + F 1.38, reversed strides 1.32, `ReadOnlyMemory` 1.18): the call pins the
array (`GCHandle`, ~23 ns alloc + free) and builds the base storage and the memory block that owns the pin, which
NumPy's `base` semantics need; the NDArray construction floor (~130 ns) is the rest. Explicit strides without a buffer:
1.76. Without a buffer the call IS `np.empty`'s allocation and costs the same (`(2,3)` 0.93, empty `(0,3)` 0.87, an
80 KB F array 0.40, 8 MB 0.10–0.19): NumSharp's allocator floor, shared by every allocating function (an 80 KB
`np.empty` takes ~1.1 µs where a raw `NativeMemory.Alloc` takes 82 ns, and 8 MB takes 40–126 µs against NumPy's
8–13 µs). That floor is a follow-up for the allocator (`SizeBucketedBufferPool`), not this function.

**Follow-ups found here.**
- The allocator floor above (large and mid-size fresh allocations).
- `np.zeros`/`np.empty`/`np.ones`/`np.full`/`np.arange(0)` still report C strides for an empty array (NumPy: all zero);
  only `np.ndarray` follows NumPy (`OpenBugs.EmptyStrides.cs`).
- Item D7 of `docs/plans/api-parameter-parity.md` (`offset=`, `strides=`) is done by this plan; that plan belongs to
  another session and was not edited.
