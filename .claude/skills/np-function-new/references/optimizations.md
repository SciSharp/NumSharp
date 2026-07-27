# House Optimization Catalog

These techniques are already **in the codebase**. Skim this before designing any kernel: reuse and
extend rather than reinvent, and keep new kernels consistent with these idioms. When NumPy's
implementation uses a technique not listed here (pairwise summation, binsearch bound-carry, …),
understand what it buys before replacing it — there is a reason for everything NumPy does, and
"our version is simpler" is not parity.

## A. Specialization & code generation

- **Runtime IL emission per cache key** — DynamicMethod generates a kernel once per (op, dtypes,
  layout) and the JIT compiles it to native; subsequent calls hit a ConcurrentDictionary lookup.
- **Per-startup SIMD width baking** — VectorBits resolved once via IsHardwareAccelerated; the
  emitted IL targets exactly one of V128/V256/V512 with no runtime width branch.
- **Layout-specialized kernel paths** — Distinct kernels for SimdFull / SimdScalarLeft /
  SimdScalarRight / SimdChunk / General instead of one kernel with runtime layout branches; layout
  is part of the cache key.
- **Signature collapse for fast paths** — Contig kernels drop stride/shape args; scalar-broadcast
  kernels take `T scalar` not `T*`; cuts indirection and shrinks the IL body.
- **Helper-call vs inline-IL choice** — When an op has a tidy generic-constrained C# helper
  (e.g. `CumSumHelperSameType<T>`), the kernel emits a single Call and lets the JIT inline; only
  complex bodies inline the IL loop themselves.
- **Negative cache for unsupported combos** — `_castUnsupported`/`_maskedCastUnsupported` record
  dtype pairs that fail IL gen so retries are O(1) instead of re-attempting emission.

## B. Loop shaping

- **4×–8× unrolling with independent accumulators** — Body processes 4–8 vectors per iteration into
  4–8 separate accumulators; breaks the carried dependency so the CPU dispatches multiple SIMD
  ops/cycle.
- **Three-stage loop** — Unrolled SIMD body + 1-vector remainder + scalar tail; handles any count
  without padding.
- **Inner-contig runtime dispatch** — Inside strided kernels, compare each operand's stride to its
  element size; branch into the SIMD inner body when all match, else strided.
- **Cache-friendly loop ordering** — IKJ in MatMul so the inner SIMD walk is over sequential
  `B[k,:]` memory; `A[i,k]` is broadcast once and reused across all j.
- **Chunk-driven cursor instead of element-driven** — When exposing iteration, let the iterator
  hand out a whole inner loop and walk it with pointer arithmetic, touching the engine only at a
  chunk boundary; a contiguous array becomes ONE engine call. Measured 3× over calling
  `Iternext()` per element (`np.nditer<T>`).
- **Hand out `ref T` / `Span<T>`, not a wrapper object** — The per-element managed view is usually
  the whole cost of an iteration API: NumSharp's boxed `nditer` `it[0]` loop measures 59 ms on 100K
  float64 where the same walk yielding `ref double` is 0.167 ms (~350×) and a `Span<T>` chunk fed to
  `Vector<T>` is 0.027 ms. A `ref struct` enumerator with `ref T Current` gets this with no
  allocation, no boxing and no interface dispatch, because C#'s `foreach` is pattern-based and needs
  no interface. See design-recipes → "exposing an engine `ref struct` to USER code".

## C. SIMD primitives

- **Mask→uint via ExtractMostSignificantBits** — Convert a Vector mask to packed bits in a uint —
  the universal building block for All/Any/NonZero/CountTrue/CopyMasked.
- **Bit-scan loop (TrailingZeroCount + `bits &= bits-1`)** — Materialize lane indices from a packed
  mask one at a time without a per-lane branch; standard idiom for sparse-extract.
- **Self-equality NaN mask** — `Equals(v, v)` produces lanes true for non-NaN (NaN ≠ NaN); used to
  zero/count out NaNs in NaN-aware reductions.
- **Branchless ConditionalSelect** — Per-lane gating without a branch; used by Where and masked
  cross-dtype copy.
- **Scalar pre-broadcast** — `Vector.Create(scalar)` hoisted into a local before the loop so the
  body reuses it; used by scalar-broadcast variants of binary/where/clip.
- **Op-specific identity seeding** — Reduction accumulators pre-loaded with 0 (Sum), 1 (Prod),
  MinValue (Max), MaxValue (Min) — which also defines the empty-array result.
- **Tree merge + horizontal halving** — Multi-accumulator finalization: `acc0 op= acc1;
  acc2 op= acc3; acc0 op= acc2`, then horizontal reduce across the lanes.
- **Early-exit on mask state** — All/Any/IsAllZero return immediately when the packed bits hit the
  terminal pattern, skipping the rest of the array.
- **Vectorized index discovery, scalar scatter** — Even when the data store can't be vectorized
  (gather/scatter limits), the mask scan that finds the indices is fully SIMD.
- **AVX2 gather for strided float/double** — Strided axis reductions use intrinsic gather when the
  dtype is gather-capable.
- **Width-adaptive emit via GetVectorContainerType()** — One emission function picks
  Vector{128|256|512} methods through a cache; the same source path covers all widths.

## D. Memory & pointer

- **Cpblk IL intrinsic** — Same-type contiguous copy emits the CLR block-memcpy opcode directly
  instead of a loop.
- **Incremental coord advance** — Outer-dim walks update offsets by adding strides rather than
  recomputing via flat → div/mod per element.
- **Pre-computed dim strides in stack array** — Axis kernels pre-build output-dim strides on the
  stack so each output index → input offset is O(ndim) muladds, no divmods.
- **Pointer/stride prologue hoisting** — Inner-loop factory snapshots `dataptrs[i]` and
  `strides[i]` into locals once at the top so the loop body works against locals, not memory loads.
- **Pre-size-then-fill** — np.nonzero runs an IL-emitted popcount first to size the output buffer,
  then a second IL-emitted bit-scan kernel writes indices; avoids the "alloc max-size temp"
  pathology.

## E. Algorithmic

- **Two-pass algorithms** — ArgMax (find value → find index), Var/Std (mean → squared diffs),
  masked-copy (count → place). First pass enables vectorization; second pass exploits the known
  result.
- **Monotonic-bound carry** — searchsorted carries the lower bound L from the previous iteration
  when consecutive keys ascend, mirroring NumPy's binsearch.cpp.
- **Short-circuit prescan** — Quick SIMD all-zero check on a boolean mask short-circuits the whole
  np.where(cond) pipeline when the condition is fully false.
- **Type-promotion-aware path skip** — SIMD reduction skipped when input != accumulator
  (e.g. sum(int32)→int64) because Vector<T> can't widen lanes; falls to scalar IL.
- **Two-tier inner-loop API** — Callers choose Tier 3A (raw IL body) for full control or Tier 3B
  (scalar/vector body lambdas wrapped in the standard 4×-unrolled shell) for boilerplate
  elimination.

## F. Cross-type bridging

- **Decimal-via-double bridge** — All transcendental decimal ops emit
  decimal→double→Math.*→decimal inline IL.
- **Bool-mask lane expansion** — 1-byte mask widened through a WidenLower chain to match the
  1/2/4/8-byte data lane width before ConditionalSelect.
- **Magnitude comparison for Complex** — ArgMax/ArgMin on Complex compares |z|, since Complex has
  no native ordering.

## G. NumPy semantic compliance

- **NumPy-overflow shift semantics** — Branch on `shift >= bitWidth` returns 0 (or -1 for
  signed-negative right shift) instead of C#'s `x << (n & 31)` masking.
- **Sign-preserving zero in Modf** — Explicit fixup so `modf(-0.0) = (-0.0, -0.0)` and
  `modf(+inf) = (+0.0, +inf)` per the C standard.
- **Vacuous truth for empty reductions** — `all([])=True`, `any([])=False`, identity-valued
  Sum/Prod/Max/Min for empty arrays.
- **NEP50-aligned accumulator types** — Reduction kernels promote int32→int64 for
  Sum/Prod/CumSum, dropping out of SIMD when needed.

## H. Reflection & caching

- **MethodInfo cache (fail-fast at type load)** — Math.*, Vector*.*, Decimal.* reflection resolved
  in static initializers with `?? throw`; emission never pays GetMethod cost.
- **Width-resolved generic method cache** — `VectorMethodCache.V(VectorBits, clrType)` returns the
  right Vector{W}<T> type and `Generic(VectorBits, name, clrType, paramCount)` the right method
  handle.
- **ConcurrentDictionary.GetOrAdd keyed by structural value** — All kernel caches use struct keys
  with stable Equals/GetHashCode; thread-safe lazy init via GetOrAdd.
