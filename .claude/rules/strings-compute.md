---
paths:
  - "src/NumSharp.Core/Backends/Strings/{NDStringComparison,NDStringTruth,NDStringArithmetic,NDStringRepeatReduction,NDTextNumericReduction,NDVStringCells,NDVStringComparison,NDVStringArithmetic,NDVStringReduction,NDVStringStatistics,NDVStringStatistics.Quantile}.cs"
  - "src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Text*.cs"
  - "src/NumSharp.Core/Backends/Default/Sorting/{TextSort,TextArgReduce,TextSearch,TextKeyer,TextDistinct,VStringSort,VStringSearch,VStringDistinct}.cs"
  - "src/NumSharp.Core/Backends/Default/Manipulation/Default.{Unique,Isin}.*.cs"
  - "test/NumSharp.Tests/Backends/Strings/{StringComparison,StringSort,StringSearch,StringSets,StringDTypeCompute,StringDTypeOrder}*.cs"
---

# Computing on text: comparisons, arithmetic, reductions, ordering

Over `S`, `U` and `StringDType`, byte-identical to NumPy 2.4.2 (the storage and the refuse-or-compute rule:
`strings.md`; the named `np.strings` functions: `strings-methods.md`).

## The text kernels and their drivers

- The comparisons and truth values and the concatenation / repetition of `S` / `U` (`np.add` / `+` over two strings of
  one class, `np.multiply(text, int, out=)`, `np.strings.add` / `np.strings.multiply`) run IL text kernels
  (`Backends/Kernels/ILKernelGenerator.Text{,.Emit,.Concat}.cs`): element widths are emit-time constants and every element
  is fixed-size chunk loads / stores, so no per-element branch follows its length (NumPy's own repeat loop runs a
  variable-size memcpy per copy). The drivers are `Backends/Strings/NDString{Comparison,Truth,Arithmetic}.cs`.
- An iteration of several rows that does not coalesce (a column slice `m[:, :3]`, a column-broadcast comparand, a stepped
  N-D view) hands BLOCKS of rows to the rows form of a text kernel (`NDIterRef.ForEachRows` with an `NDInnerLoop2DFunc`,
  `Backends/Iterators/NDIter.Execution.Rows.cs`; kernels in `ILKernelGenerator.Text.Rows.cs`), so the kernel prologue runs
  once per block, not per row. A `where=` mask into a bool output runs the masked comparison kernel, which computes every
  element and blends `out = mask ? result : out` branch-free instead of one kernel call per mask run (a cast `out=` keeps
  NDIter's per-run ARRAYMASK route: its write-masked copy-back is what leaves a masked-off element of a cast output
  untouched). The switches `RowsDisabled` / `MaskedBlendDisabled` on `NDStringComparison` (and
  `NDStringTruth.RowsDisabled`) take the per-row / per-run routes for A/B tests (process-wide: such a test is
  `[DoNotParallelize]`).

## `StringDType` computes

- The six comparisons over (`T`, `T`) / (`T`, `U`) and their operators, the truth readers and `np.isnan`, `add` /
  `multiply` / `maximum` / `minimum`, `np.sum` / `np.cumsum` / `np.max` / `np.min` / `np.argmax` / `np.argmin` /
  `np.trace`, the statistics that refuse after computing, `np.char.add` / `multiply` and numpy.ma over `T`, with NumPy's
  NA rules: a NaN-like NA makes every comparison False (`!=` too), propagates, and orders last; a string NA is its text; a
  `None`-like NA equals an NA, and ordering or arithmetic with it raises. Descriptor resolution is NumPy's: the operands'
  common NA object, the first operand's `coerce`; a `T` out is the loop's own output, any other out receives the default
  `StringDType()` through the output cast, so an NA is `''` in a `U` out.
- Code: `Backends/Strings/NDVString{Cells,Comparison,Arithmetic,Reduction,Statistics}.cs` and
  `ILKernelGenerator.Text.VString.cs` (IL kernels over NDIter). Design rules: two short cells compare INSIDE their cells
  as a masked 128-bit big-endian key, with no branch on the data (one vector compare first sends a text holding a NUL to
  the strncmp path, below); a pair outside its cells decides by the first eight bytes inline; everything else is out of
  line (`ComparePairSlow` and the `NoInlining` arena forms: inlined beside the short forms they made the JIT spill
  registers on every element); the truth readers read only byte 15 and the size field (four cells a step on AVX2); a
  fresh output reserves its arena ONCE from a size pass and appends at the cursor (`VStringAllocator.AppendFresh`);
  `maximum` / `minimum` reserve the larger text of each pair and skip the pass over cell-only inputs.
- `NpyString_cmp` is strncmp over the shorter size, then the sizes: a NUL both texts share ENDS the byte comparison
  (`'a\0c' == 'a\0b'`, `'a\0c' < 'a\0bb'`, `'a\0' < 'ab'`), and every `T` comparison, sort, search, extreme, set routine
  and order statistic inherits it; `U` compares every code point. `NDVStringCells.Order` / `StrncmpStop` compare 32 / 16
  bytes a step (`~(a == b) | (a == 0)`; the step that would run past the text moves back to end on its last byte, and 8–15
  bytes are two SWAR words, `WordStop`). The fast paths test for a NUL only where it can change the answer: `Equal` only
  for a pair of one size whose bytes differ, `Less` / the 128-bit key one vector compare per short pair (`ShortHasNul`),
  `TryFirstWord` through `WordHasNul`; `VStringSort` keys on each text's comparison prefix (`NulLengthInPlace`);
  `VStringDistinct` drops duplicates by byte content inside each strncmp-equal run.

## Text sorts (`S` / `U`)

`Backends/Default/Sorting/TextSort.cs`, key kernels `Backends/Kernels/ILKernelGenerator.Text.Sort.cs`, the keywords in
`Backends/Default/Sorting/SortKind.cs`.
- What NumPy does: `np.sort` / `np.argsort` / `ndarray.sort` over `S` / `U` run `string_quicksort_` /
  `string_aquicksort_` (`npysort/quicksort.cpp`), a median-of-3 introsort that insertion-sorts every range of at most 16
  elements and falls back to `string_aheapsort_` once its depth budget (`2 · msb(n)`) is spent. It is NOT stable, so an
  argsort's order of equal strings depends on the whole run. The heapsort kind gives the default's answer, the stable
  kinds the unique stable order, and `np.partition` / `np.argpartition` have no text partition (NumPy sorts). A
  `StringDType` array refuses the heapsort kind (`RuntimeError: unsupported sort kind`).
- Keys: each element's first units as one big-endian `ulong` (`S`: 8 bytes; `U`: 8 units of 8 bits, else 16 / 21 / 32
  bits by the widest unit present), from IL key kernels with managed twins; EXACT when the key covers the element.
- Argsort routes (one permutation whichever runs; the test hooks `ShortcutDisabled`, `DirectPortDisabled`,
  `PackedWindowDisabled`, `TailPortDisabled` switch each off, and counters prove which ran):
  - a line of at most 16 elements: a branch-free vector count-rank (NumPy's insertion sort is the stable order); rows
    that lie end to end are keyed 4,096 elements per kernel call, strided rows one row at a time;
  - exact keys, unstable kind: the introsort REPLAYED over key records when the line has at most 128 elements, its
    duplicates are certain (more elements than the window of varying key bits holds values), or a 128-key sample repeats;
    over window-packed records (`window << 32 | index`, one ulong) when the varying bits span at most 32, else over
    (key, index) pairs;
  - keys that do not cover the strings, unstable kind, at most 16,384 elements, 1 to 8 sample repeats: the introsort
    replayed over (prefix key, index) records whose compare goes past the key on a tie (NumPy's own string compare);
  - otherwise the stable order (a stable LSD radix over the key bytes that vary, then a tie pass that keys the next units
    of each run of equal keys), and then for the unstable kinds either the distinct-strings shortcut (no equal strings:
    the stable order IS NumPy's answer) or dense ranks replayed over packed records.
- Value sorts and the stable kind take the stable order (any order of equal strings gives the same values) and gather.

## Text searches

`Backends/Default/Sorting/TextArgReduce.cs`, `TextSearch.cs`, the shared prefix keys `TextKeyer.cs`.
- What NumPy does: `STRING_argmax` / `UNICODE_argmax` replace the running best only with a STRICTLY greater element
  (smaller for argmin): the first extreme wins. `np.nanargmax` / `np.nanargmin` of text are argmax / argmin. `out=` must
  cast 'safe' to intp (bool through uint32; uint64 and floats are TypeError), checked after the empty-line ValueError.
  `np.searchsorted` converts both operands to their common descriptor (a number becomes text; over `T`,
  `PyArray_PromoteTypes(keys, haystack)`, keys FIRST: the first operand's `coerce`) and runs the GENERIC `npy_binsearch`
  with its monotonic carry; a sorter value outside the haystack raises only when reached. Over a SORTED haystack every
  correct search gives NumPy's answer; over an unsorted one only its exact comparison sequence does. `np.digitize` of
  text searches the bins converted to text (`digitize(['10','3','25'], [5, 20])` is `[0 2 2]`).
- argmax / argmin: lines in blocks of 1,024, block winners merged strictly. Per block the common prefix is skipped
  (inline whole-vector compares), then the GATHER filter (`U` + AVX2: the unit after the prefix of 8 elements a gather; a
  tie decided in one masked 256-bit compare for elements up to 32 bytes; each lane re-tested against the CURRENT best,
  since an earlier lane of the group can raise it) or the KEYS filter (`S`, no AVX2, or a call whose best string recurs
  one element in eight: the gather filter pays a full compare per copy). Short rows (≤ 16) are keyed end to end and
  scanned branch-free. Non-C N-D flat reductions walk lines along the smallest stride, merged by value then C index.
- searchsorted: DIRECT (few keys: `m·8·log2(n) ≤ n`): NumPy's loop on the strings, the head compared first (a `U`
  string's first unit, an `S` string's first 8 bytes big-endian), the next step's two candidates prefetched above 1 MB.
  KEYED: the prefix shared by every element and key is skipped, one packing chosen from both arrays keys them; when the
  haystack's keys never decrease in search order (through a sorter whose values are all in range; equal keys' tails
  compared when the keys are not exact) the SORTED route answers with bounds (a galloping MERGE for sorted keys, else
  four branch-free halving searches interleaved); otherwise NumPy's exact loop over the keys, as BRANCHES (over an
  unsorted haystack branches beat masks; masks lose on sorted keys and categorical data, whose paths the predictor
  learns).
- `np.histogram2d` / `np.histogramdd` of a text sample cast the columns to the stacked text dtype FIRST, into a new
  array, as NumPy's stacking does (an `S` column beside a `U` one with a non-ASCII byte raises `UnicodeDecodeError` before
  any bin error), and search the numeric edges as text.
- Floors: arg-reductions over fresh 100K text read every cache line, and few keys into a large haystack wait on a
  dependent load per step (NumPy's loop has the same walk).

## Text set routines and statistics

NumPy's statements over the sort and search engines: `Backends/Default/Manipulation/Default.Unique.Text.cs`,
`Backends/Default/Sorting/TextDistinct.cs`, `Default.Isin.cs` (`IsinTextFlat`), the set routines' own files,
`np.lexsort` / `np.sort_complex`, `Backends/Default/Statistics/Default.Histogram.cs` (`HistogramCumulative`),
`Statistics/QuantileText.cs`
with `Default.Quantile.Functions.cs` (`QuantileOfText`).
- np.unique: the flat path (axis None or a 1-D input) with flags takes ONE stable text argsort for every flag
  (return_index needs NumPy's mergesort first occurrences; inverse and counts do not depend on the order of equal
  strings), then NumPy's mask / cumsum / nonzero / diff statements. Values only: NumPy hashes (`unique.cpp`
  `unique_string`, FNV-1a and memcmp) and sorts the set; `TextDistinct` runs the same pass (open addressing, the 32-bit
  hash beside each slot's first-occurrence index) and sorts only the distinct strings, until more than max(256, n / 16)
  are distinct: then the flattened copy is sorted whole. The axis path orders rows by `np.lexsort` over their columns
  (NumPy's structured sort compares field by field), whole-row `not_equal` + `any` for the mask, `reshape_uniq`'s
  moveaxis view back.
- Order excuse: `np.unique_values` (always `sorted=False`) and `np.unique(sorted=False)` without flags return NumPy's
  `std::unordered_set` order, which depends on the C++ library; NumSharp returns the values sorted, and the string oracle
  reorders them into NumPy's order before the full comparison (`OrderExcuse` / `InNumPyOrder`), so set, dtype and layout
  are still compared.
- isin and the set routines: `_isin`'s loop path below `10 * len(ar1) ** 0.145` test elements (`ar1 == a`: text against a
  number, or `S` against `U`, has no loop, so the mask stays) and its sort path (unique, concatenate (numbers written as
  text, bytes decoded as ASCII with NumPy's `UnicodeDecodeError`), a mergesort argsort); `assume_unique`'s answer over
  duplicates is reproduced; `intersect1d`'s `return_indices` are the mergesort argsort's first occurrences.
- `np.lexsort` is a stable argsort per key (text's default kind is NumPy's introsort, so the call names `kind:
  "stable"`). `np.sort_complex` sorts the TEXT, then parses (`complex() arg is a malformed string` after the sort).
- np.histogram of text always takes NumPy's cumulative path: per 65,536-element block `np.sort` and the search of the
  edges (a number edge is searched as its text, so a count can be NEGATIVE); with weights a default-kind argsort (NumPy's
  tie order decides the float sum), cumsum and take; the running total is updated in place in its ntype (bool weights
  raise NumPy's same_kind add error).
- np.quantile / np.percentile of text: `_ureduce` (the axes, keepdims' `out[(Ellipsis,) + index_out]`, the merge
  `reshape(kept + (-1,))` that NumPy refuses for a 0-long kept axis), `q.ndim > 2`, the method, then `_quantile`'s take
  path for the selection methods and for 'linear' with an integer q: the partition is a full sort, then
  `PyArray_TakeFrom`'s checks in order (out's shape, the safe cast from out's dtype, the empty-axis IndexErrors, the
  unsafe writeback). The interpolating methods raise the subtract refusal, or over an empty axis the IndexError of the
  first neighbour `_get_indexes` clamps. keepdims over a 0-d result: Python's str / bytes tuple-index TypeErrors.

## Ordering `StringDType`

`np.sort` / `np.argsort` / `ndarray.sort` / `np.partition` / `np.argpartition` / `np.lexsort`, `np.searchsorted` /
`np.digitize`, `np.unique` and the Array-API unique family, `np.isin`, the four set routines, `np.sort_complex`,
`np.histogram`, `np.median`, the quantile family with its nan forms, and numpy.ma's `sort` / `argsort` / `median` /
`unique`, with NumPy's ordering of missing values: a NaN-like NA after every text in input order, a string NA as its
text, a `None`-like NA raising at the first comparison that meets it (a line of one element compares nothing).
- Code: `Backends/Default/Sorting/VString{Sort,Search,Distinct}.cs`, `Manipulation/Default.{Unique,Isin}.VString.cs`,
  `Backends/Strings/NDVStringStatistics.Quantile.cs`.
- Design: a stable radix of 8-byte prefix keys with a tie pass (NumPy's stable kind); NumPy's introsort replayed over
  (rank, index) records only where equal texts make the default kind's order observable; `np.lexsort` replays
  `npy_atimsort` when two or more NaN-like NAs meet (its asymmetric compare reverses adjacent NAs); `np.unique`'s
  values-only route hashes (`TextTable`: cell texts hashed from their two size-masked words; stale bytes past the size
  never count) unless the birthday estimate from the repeats of its first window (an eighth of the array, 256 to 4,096
  texts) says more than half the array is distinct, then argsorts the array itself; the order statistics run NumPy's
  Python statements (`_lerp`'s `b - a` of the neighbours' Python objects, `_ureduce`'s out / keepdims indexing,
  `apply_along_axis`'s first-row buffer dtype). The nan forms remove NaN-like NAs in `_remove_nan_1d`'s order
  (`RemoveNaN1D`: the partition keeps tied texts in arrival order). A masked `T` reduction's result layout lets the
  `where=` mask vote (`Shape.ReductionOutputShape(dims, input, axis, where)`); a `T` comparison with `where=` into an
  out of another dtype computes its bool temporary under the mask.
- NumPy 2.4.2 defects the generator avoids and the unit tests pin with NumSharp's answers (`StringDTypeOrderTests`,
  `[Misaligned]`): it CRASHES in `np.lexsort` over a non-contiguous `T` key and in the default-kind sort when its
  introsort spends the depth budget (the heapsort fallback reads the method context as an array: `np.intersect1d` of two
  unique outputs reaches it); it misorders arena texts in the argsort of a non-contiguous view (`axis=None`); it loads
  `searchsorted`'s haystack and `np.histogram`'s search with the keys' allocator (an arena haystack text raises
  `MemoryError`); an empty text times a negative count hangs. A `None` NA row result of `np.apply_along_axis` is an object
  array in NumPy, `NotSupportedException` here. NumPy's values-only `np.unique` hashes the FULL bytes and sorts
  unstably, so two strncmp-equal texts survive in an order the C++ library picks: the soak's `TieExcuse` accepts an
  element that compares EQUAL to NumPy's.

## Gates

- String oracle sections: `strcomp` / `strtruth` (comparisons and truth), `strarith` (concatenation / repetition),
  `strsort`, `strsearch`, `strsets`, `tcomp` / `tarith` / `treduce` (`StringDType` computes), `tsort` (ordering `T`),
  `qfamily` / `qprec` (the quantile family, `quantile.md`); every case strict.
- Unit: `Backends/Strings/StringComparisonTruthTests.cs`, `StringArithmeticTests.cs`, `StringRowsAndMaskTests.cs`,
  `StringSortTests.cs`, `StringSearchTests.cs`, `StringSetsTests.cs`, `StringDTypeComputeTests.cs` (raw-cell and
  arena-reservation tests included), `StringDTypeOrderTests.cs`.

## Traps

- A sorting benchmark that repeats ONE input lets the branch predictor memorize a branchy sort of up to ~1,000 elements,
  NumPy's included (NumPy argsorts the same 1,000 random `U8` strings 2.6× faster than fresh ones): time sorting cells
  over a pool of fresh inputs on BOTH sides (16 different inputs cycled per cell, the median call), and set routing
  thresholds from fresh-input medians. A harness that warms each cell just before timing it measures its first cells
  before the loops that run once per call (the radix passes) reach tier-1 code: warm every cell, pause, then time.
- A branch-free binary search is NOT always faster: it serializes the loads a predictable branchy search speculates
  past; measure sorted keys, categorical data and unsorted haystacks, not only random keys. The JIT does not turn selects
  inside loops into cmov (`x = c ? a : b` stays a branch on .NET 10): build masks with
  `-(long)Unsafe.As<bool, byte>(ref c)` and check the disassembly. A gather lane mask taken against a group's starting
  best can admit a lane an earlier lane of the group has since beaten.
- A variable-length `Buffer.MemoryCopy` in a hot loop costs ~20 ns a call (the hash's 1–7 byte tail): three fixed-size
  loads were 3× faster.
- A benchmark that disposes a result which IS an argument (`np.histogram`'s `bin_edges` is the edges array) frees the
  caller's input. The string oracle rebuilds a 1-D numeric operand as a view (OWNDATA False), so a result that is the
  operand itself needs `plain()` in the generator.
