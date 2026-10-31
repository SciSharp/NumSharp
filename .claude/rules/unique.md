---
paths:
  - "src/NumSharp.Core/Manipulation/np.unique*.cs"
  - "src/NumSharp.Core/Manipulation/NDArray.unique.cs"
  - "src/NumSharp.Core/Backends/Default/Manipulation/Default.Unique*.cs"
  - "test/NumSharp.Tests/Manipulation/*unique*.cs"
---

# `np.unique` and the Array-API unique family

`np.unique(ar, return_index=false, return_inverse=false, return_counts=false, axis=null, equal_nan=true, sorted=true)`
returns `np.UniqueResult`, a `readonly struct` standing in for both NumPy shapes:
- it converts implicitly to `NDArray` (the values) and to `NDArray[]` (the present outputs in NumPy's field order
  `[values, index?, inverse?, counts?]`); `[k]` is the k-th present output (`np.unique(ar)[k]` is an OUTPUT; the k-th
  unique value is `.values[k]`); it deconstructs (`var (values, counts) = np.unique(ar, return_counts: true)`); its
  fields `values`/`indices`/`inverse_indices`/`counts` are NumPy's namedtuple names (null when not requested);
- it forwards the common NDArray surface to `values` (`size`/`shape`/`Shape`/`ndim`/`dtype`/`typecode`/`array_equal`/
  `GetAtIndex`/`GetAtIndex<T>`/`GetDouble`/`GetSingle`/`GetBoolean`/`GetInt32`), and the test `Should()` extension has a
  `UniqueResult` overload (`FluentExtension.cs`);
- the instance methods `NDArray.unique(int? axis=null, …)` → `NDArray` and `NDArray.unique(bool return_index, …)` →
  `NDArray[]` keep their return types. Every form is one call into an engine member: `TensorEngine.Unique` for
  `np.unique` (after its chararray / matrix boundaries) and for both instance members (the bare one takes element 0),
  `UniqueValues` / `UniqueCounts` / `UniqueInverse` / `UniqueAll` for the Array-API functions. DefaultEngine's overrides
  are `Backends/Default/Manipulation/Default.Unique.cs`, its kernels the partials
  `Default.Unique.{Kwargs,Hash,Radix,Text,VString}.cs`.
- `sorted` is accepted and is a no-op: the output is always sorted (NumPy's `sorted=False` hash order is
  platform-specific; identical as a set). An out-of-range axis raises `AxisError` ("axis {a} is out of bounds for array
  of dimension {n}"), reporting the original axis.
- The axis path compares each sub-array as a NumPy structured scalar (NumPy consolidates the axis into a void dtype):
  `equal_nan` has no effect there (the void kind is not in `_unique1d`'s `"cfmM"` guard), every NaN sub-array is
  distinct, and a signed-zero sub-array collapses (`-0.0 == +0.0`, the surviving zero's sign following input order).
  The slab equality is IEEE `==` and the order IEEE `<`/`>` with a stable tie-break by original index (NumPy's mergesort
  argsort). Never `.Equals`/`CompareTo` there: `NaN.Equals(NaN)` is true and `(-0.0).Equals(0.0)` is false. The values
  are always NumPy's view of the gathered rows (`reshape_uniq`'s `np.moveaxis` runs at axis 0 too, so `flags.owndata` is
  False), and a C-contiguous operand at an offset (an `np.split` / `np.array_split` / `np.unstack` child) is read from
  its own first element.
- `[Misaligned]`: for float16 + axis + NaN, NumPy 2.4.2's structured sort puts NaN first (unlike its own 1-D float16 sort
  and its float32/64 structured sorts); NumSharp keeps NaN last for every float width (the unique rows and counts match).
- Gates: `Manipulation/np.unique.AxisEdgeCases.Test.cs` (`UniqueResult_*`, `BareReturn_*`) and `NDArray.unique.Test.cs`.

The flat routes are NumPy's `_unique1d` statements. NumPy sorts in place when neither an index nor an inverse is asked
(`ar.sort(); aux = ar`), and otherwise argsorts and gathers (`perm = ar.argsort(kind='mergesort' if return_index else
'quicksort'); aux = ar[perm]`); the run starts are `aux[1:] != aux[:-1]` (IEEE: a NaN differs from everything,
`-0.0 == +0.0`), `equal_nan` collapses one run from the first NaN (complex: the first value with a NaN part), and the
index is `perm[mask]`. NumSharp's routes:
- An index or an inverse (any float dtype), float16 every call and complex128 counts run the argsort statements
  (`DefaultEngine.uniqueFlatArgsorted<T>`) over `AxisSort.ArgSort`, one stable order for every kind (NumPy's mergesort
  order): a collapsed NaN run's value and index are the same NaN's, a run of zeros keeps the first zero in input order,
  and complex values come in `CDOUBLE_LT`'s order (no NaN part lexicographic, then `R+nanj` by `R`, then `nan+Rj` by
  `R`, then `nan+nanj`). float16 values alone take it too because NumPy's float16 in-place sort is its scalar
  `quicksort_<half_tag>` off AVX512_SPR / ICL hosts, an insertion sort (stable) for lines of up to 16 elements.
- float32 / float64 values alone and counts alone are NumPy's in-place sort: the NaNs partitioned to the tail and
  written as NumPy's sort NaN, the positive quiet NaN, for two elements or more (`WriteSortNaNs`); a one-element line
  keeps its bits.
- complex128 values alone take the hash path (below) for both `equal_nan` settings.
- Past `Array.MaxLength` elements the long-indexed fallback runs the same statements on each route
  (`uniqueFlatArgsortedLong<T>` / `uniqueFlatArgsortedLongComplex`); past 2^28 complex elements, where the hash bails,
  the argsort route gives the hash path's answer (`ArgsortedHashAnswer`: under `equal_nan` the collapsed class is its
  first value in C order, the one NumPy's hash set keeps).
- Remaining divergences, all of NumPy's own order between EQUAL values: float32 / float64 values alone and counts alone
  (x86-simd-sort's order of `-0.0` / `+0.0`, CPU-dependent); float16 and complex128 lines of 17 elements or more without
  an index (NumPy's scalar introsort: float16 values alone, counts and an inverse; complex128 counts and an inverse),
  where NumPy may keep another zero than the first in input order; the complex hash path's table order.
- Gates: `Manipulation/np.unique.{ArgsortRoute,ComplexHash,OffsetView}.Test.cs` and `OpenBugs.SortUniqueNaN.cs` (a
  regression guard, no `[OpenBugs]` left).

`unique_values`, `unique_counts`, `unique_inverse`, `unique_all` (NumPy 2.x) wrap the same machinery with
`equal_nan=False` (each NaN distinct) and `axis=None`:
- results are named-field structs with NumPy's field names (`UniqueAllResult{values,indices,inverse_indices,counts}`,
  `UniqueCountsResult{values,counts}`, `UniqueInverseResult{values,inverse_indices}`; each converts implicitly to
  `NDArray[]`, deconstructs and indexes); `unique_values` returns a bare `NDArray`;
- `values` keeps the input dtype; `indices`/`inverse_indices`/`counts` are int64; `inverse_indices` takes the original
  input shape (`np.take(values, inverse_indices)` reconstructs). `np.unique`'s flat path returns `(1,)` for a 0-d input,
  so `ReshapeInverseToInput` restores `()` for these wrappers;
- `[Misaligned]` (`unique_values` only): NumPy's `_unique_hash` returns integer and complex dtypes in hash-bucket order
  (not reproducible); NumSharp returns them sorted. Float32/64 are not in NumPy's hash map, so they are sorted and
  bit-exact.
- The hash fast path (`Default.Unique.Hash.cs`): the ten integer-family dtypes (bool, byte, sbyte, int16, uint16, int32,
  uint32, int64, uint64, char) and complex use an open-addressing table (a byte-size-keyed generic hash, load factor 0.5,
  a splitmix64 finalizer), then sort the small unique set, as NumPy hashes integers and complex and sorts floats. Floats,
  half and decimal keep the sort path: hashing regresses all-unique input about 2×, and no cheap cardinality sample
  tells the cases apart. The splitmix64 finalizer is load-bearing: a single Fibonacci multiply plus a low-bits index
  clusters integer-valued double bit patterns (`1.0`, `2.0`, `4.0`: the discriminating bits are high) into one bucket.
- Complex takes the hash for `np.unique`'s values alone (either `equal_nan`) and for `np.unique_values`, as NumPy's
  `_unique1d` takes `_unique_hash` there, and for `np.unique_counts`, where NumPy sorts in place instead: the hash with
  counts gives the same sorted values and counts, its representative of equal values being the first in C order. Its
  distinct values are sorted afterwards (by radix from 2,048 on; the values with a NaN part in NumPy's sort order), and a
  line of 32,768 elements or more whose first sixteenth is over 90 % distinct goes to the argsort route instead, faster
  there with the same answer (`ComplexHashCheckpointMinLength`). NumPy's table order of EQUAL values with a NaN part
  (`nan+0j` and `nan-0j`) is MSVC's `std::unordered_set` iteration order, not reproduced: NumSharp keeps input order.
- The sub-1.5× cells (high-cardinality counts, `unique_inverse`, float values/counts) are bounded by NumPy's SIMD
  `vqsort`, which NumSharp's scalar sort core lacks.
- Gate: `Manipulation/np.unique_values.Test.cs`.
