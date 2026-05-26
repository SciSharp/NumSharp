---
name: np-function
description: Implement a NumPy np.* function in NumSharp with full API parity, optimizations, and variation coverage (NumPy 2.4.2 source of truth).
argument-hint: <np.function_name or description>
---

When user requests /np-function, you are to follow these instructions carefully!:

# np-function command

We are looking to support NumPy's np.* to the fullest. we are aligning with NumPy 2.4.2 as source of truth and are to provide exact same API (np.* overloading) as NumPy does.
This session we focusing on: """$ARGUMENTS"""
You job is around interacting with np.* functions - no more than one unless they are closely related.

np.* / function's high-level development cycle is defined as follows:

## 1. Read, investigate, learn and experiment
Read how NumPy (src\NumPy\) implemented the np functions you are about to implement - noting all parameters and overloads.
NumPy is the source of truth and if NumPy does A, we do A but in NumSharp's C# way.

### Definition of Done:
- At the end of step (1) step you understand to 100%:
    - How the np function works internally in NumPy and reacts to inputs / parameters.
    - What parameters the np function accepts and what modes the function works in.
    - Understand what optimizations are used by NumPy and what optimizations can we use.
- Understand how would be the best integration to our existing infrastructure.
    - Do we use ILKernelGenerator or NpyIter to implement the loop.
    - Do not implement struct kernel.
## 2. Implement np method/s
- Implement np methods to the fullest, integrating into our existing infrastructure and patterns.
- Our implementation might differ from NumPy's because NumPy uses C++ macros while we generate IL methods during runtime to achieve peak performance and cpu acceleration. But any input given to NumPy will produce same output with complete parity.
- Our implementation must provide same parameters as the NumPy function and support all dtypes NumSharp currently supports.
- Do not create a function per dtype/NPTypeCode or if-else/switch-case per dtype/NPTypeCode to call a specialized path.
- Do not use struct kernel pattern.
- Do utilize IL generation (ILKernelGenerator) and/or NpyIter to implement the function, including fast paths. 
- Any loops must be implemented via NpyIter or via ILKernelGenerator.

## Tools:
### Asserting, Validating, Comparing, Experimenting and Probing
"dotnet run <<'EOFDOTNET'" and "python <<'EOFPYTHON'" both can be used to asserting, validating, comparing, test and confirm how behaviors, edge cases, parameter variations, happyflow, unhappyflow are acting based on given input/s.
These cli functions allow rapid development and experimentation.
Specifying '#:project' and other '#' with paths must be absolute path.

### Benchmarking
Use "dotnet run <<'EOFDOTNET'" and "python <<'EOFPYTHON'" to produce professional benchmarks.

#### Benchmarking Rules of Thumbs
- We must be at-least x1.5 as fast as NumPy at all variations of execution extensively and modes possible extensively (all dtypes, all parameters combinations, see "Variations for Asserting, Validating, Comparing and Experimenting").
- There is a reason towards why NumPy does

## Optimizations and Implementation
Our codebase uses and follows the following techniques:

### A. Specialization & code generation

- Runtime IL emission per cache key — DynamicMethod generates a kernel once per (op, dtypes, layout) and the JIT compiles it to native; subsequent calls hit a ConcurrentDictionary lookup.
- Per-startup SIMD width baking — VectorBits resolved once via IsHardwareAccelerated; the emitted IL targets exactly one of V128/V256/V512 with no runtime width branch.
- Layout-specialized kernel paths — Generate distinct kernels for SimdFull / SimdScalarLeft / SimdScalarRight / SimdChunk / General instead of one kernel with runtime layout branches; layout becomes part of
the cache key.
- Signature collapse for fast paths — Contig kernels drop stride/shape args; scalar-broadcast kernels take T scalar not T*; cuts indirection and shrinks the IL body.
- Helper-call vs inline-IL choice — When an op has a tidy generic-constrained C# helper (e.g. CumSumHelperSameType<T>), the kernel emits a single Call and lets the JIT inline; only complex bodies inline the
IL loop themselves.
- Negative cache for unsupported combos — _castUnsupported/_maskedCastUnsupported record dtype pairs that fail IL gen so retries are O(1) instead of re-attempting emission.

### B. Loop shaping

- 4x-8x unrolling with independent accumulators — Body processes 4-8 vectors per iter into 4-8 separate accumulators; breaks the carried dependency so the CPU dispatches 4-8 SIMD ops/cycle.
- Three-stage loop — Unrolled SIMD body + 1-vector remainder + scalar tail; handles any count without padding.
- Inner-contig runtime dispatch — Inside strided kernels, compare each operand's stride to its element size; branch into the SIMD inner body when all match, else strided.
- Cache-friendly loop ordering — IKJ in MatMul so the inner SIMD walk is over sequential B[k,:] memory; A[i,k] is broadcast once and reused across all j.

### C. SIMD primitives

- Mask→uint via ExtractMostSignificantBits — Convert a Vector mask to packed bits in a uint — the universal building block for All/Any/NonZero/CountTrue/CopyMasked.
- Bit-scan loop (TrailingZeroCount + bits &= bits-1) — Materialize lane indices from a packed mask one-at-a-time without a per-lane branch; standard idiom for sparse-extract.
- Self-equality NaN mask — Equals(v, v) produces lanes that are true for non-NaN (NaN ≠ NaN); used to zero/count out NaNs in NaN-aware reductions.
- Branchless ConditionalSelect — Per-lane gating without a branch; used by Where and masked cross-dtype copy.
- Scalar pre-broadcast — Vector.Create(scalar) hoisted into a local before the loop so the body re-uses it instead of reloading; used by scalar-broadcast variants of binary/where/clip.
- Op-specific identity seeding — Reduction accumulators are pre-loaded with 0 (Sum), 1 (Prod), MinValue (Max), MaxValue (Min) — also defines the empty-array result.
- Tree merge + horizontal halving — Multi-accumulator finalization: acc0 op= acc1; acc2 op= acc3; acc0 op= acc2, then horizontal reduce across the lanes.
- Early-exit on mask state — All/Any/IsAllZero return immediately when the packed bits hit the terminal pattern, skipping the rest of the array.
- Vectorized index discovery, scalar scatter — Even when the data store can't be vectorized (gather/scatter limits), the mask scan that finds the indices is fully SIMD.
- AVX2 gather for strided float/double — Strided axis reductions use intrinsic gather when the dtype is gather-capable.
- Width-adaptive emit via GetVectorContainerType() — One emission function picks Vector{128|256|512} methods through a cache; the same source code path covers all widths.

### D. Memory & pointer

- Cpblk IL intrinsic — Same-type contiguous copy emits the CLR block-memcpy opcode directly instead of a loop.
- Incremental coord advance — Outer-dim walks update offsets by adding strides rather than recomputing via flat → div/mod per element.
- Pre-computed dim strides in stack array — Axis kernels pre-build output-dim strides on the stack so each output index → input offset is O(ndim) muladds, no divmods.
- Pointer/stride prologue hoisting — Inner-loop factory snapshots dataptrs[i] and strides[i] into locals once at the top so the loop body works against locals, not memory loads.
- Pre-size-then-fill — np.nonzero runs an IL-emitted popcount first to size the output buffer, then a second IL-emitted bit-scan kernel writes indices; avoids the "alloc max-size temp" pathology.

### E. Algorithmic

- Two-pass algorithms — ArgMax (find value → find index), Var/Std (mean → squared diffs), masked-copy (count → place). First pass enables vectorization; second pass exploits the known result.
- Monotonic-bound carry — searchsorted carries the lower bound L from the previous iteration when consecutive keys ascend, mirroring NumPy's binsearch.cpp.
- Short-circuit prescan — Quick SIMD all-zero check on a boolean mask short-circuits the whole np.where(cond) pipeline when the condition is fully false.
- Type-promotion-aware path skip — SIMD reduction skipped when input != accumulator (e.g. sum(int32)→int64) because Vector<T> can't widen lanes; falls to scalar IL.
- Two-tier inner-loop API — Callers choose between Tier 3A (raw IL body) for full control or Tier 3B (scalar/vector body lambdas wrapped in the standard 4×-unrolled shell) for boilerplate elimination.

### F. Cross-type bridging

- Decimal-via-double bridge — All transcendental decimal ops emit decimal→double→Math.*→decimal inline IL.
- Bool-mask lane expansion — 1-byte mask is widened through WidenLower chain to match the 1/2/4/8-byte data lane width before ConditionalSelect.
- Magnitude comparison for Complex — ArgMax/ArgMin on Complex compares |z|, since Complex has no native ordering.

### F. NumPy semantic compliance

- NumPy-overflow shift semantics — Branch on shift >= bitWidth returns 0 (or -1 for signed-negative right shift) instead of C# x << (n & 31) masking.
- Sign-preserving zero in Modf — Explicit fixup so modf(-0.0) = (-0.0, -0.0) and modf(+inf) = (+0.0, +inf) per C standard.
- Vacuous truth for empty reductions — all([])=True, any([])=False, identity-valued Sum/Prod/Max/Min for empty arrays.
- NEP50-aligned accumulator types — Reduction kernels promote int32→int64 for Sum/Prod/CumSum, dropping out of SIMD when needed.

### G. Reflection & caching

- MethodInfo cache (fail-fast at type load) — Math.*, Vector*.*, Decimal.* reflection resolved in static initializers with ?? throw; emission never pays GetMethod cost.
- Width-resolved generic method cache — VectorMethodCache.V(VectorBits, clrType) returns the right Vector{W}<T> type and Generic(VectorBits, name, clrType, paramCount) returns the right method handle.
- ConcurrentDictionary.GetOrAdd keyed by structural value — All kernel caches use struct keys with stable Equals/GetHashCode; thread-safe lazy init via GetOrAdd.


## Variations for Asserting, Validating, Comparing and Experimenting
These variations are the range of possabilities of inputs that we need to follow NumPy's output based on inputs for complete parity.
Total: ~44 distinct variations — 25 single-array layouts, 6 pairwise paths, 8 per-operand flags, 8 iteration flags, 4 composite execution paths.

### A. Single-array layouts

- C-contiguous — Row-major, stride[-1]==1 and stride[i]==shape[i+1]*stride[i+1]; baseline fast path via IsContiguous.
- F-contiguous — Column-major, stride[0]==1; 1-D arrays are both. Detected via IsFContiguous.
- Strided / non-contiguous — Arbitrary strides, neither C nor F; built via step slicing or axis swap.
- Transposed — Strides permuted by .T / swapaxes / moveaxis; usually non-contig.
- Negative-stride view — Reversed slicing ([::-1]); strides are signed-negative.
- Simple slice — offset!=0, not broadcast; fast GetOffsetSimple path (IsSimpleSlice).
- Sliced + composed — a[1:5].T, a[1:3][:,None,:]; offset combined with permutation or broadcast.
- Broadcasted — stride=0 with dim>1 (BROADCASTED flag); read-only per NumPy.
- Scalar-broadcast — All strides zero (IsScalarBroadcast); load value once and reuse.
- Partial broadcast — Some axes stride=0, others not; common (1,N)→(M,N) case.
- Scalar (0-d) — ndim==0, size==1, no strides.
- 0-D view from integer indexing — a[0,0,0] shares storage; distinct from np.array(5.0) which owns.
- 1-element 1-D — ndim==1, size==1; ambiguous against 0-d in some paths.
- Empty — size==0 (e.g. np.zeros((0,3))); reductions must return identity.
- Empty + composed — np.zeros((0,3))[::2,:]; rare but must not crash.
- NewAxis-inserted dim — a[None,:] adds dim=1, stride=0; not flagged broadcast since dim=1.
- Singleton dim (dim=1) — Stride is moot; NumPy treats as contig.
- Higher-rank (5+D) — Stack-allocated coord/stride arrays in kernels may have bounds.
- Stride > bufferSize — Negative-stride views can have offset + stride*(dim-1) >= bufferSize.
- Reshape view vs copy — Reshape returns a view when contig allows, materializes otherwise.
- Fancy-indexed result — Always a fresh C-contig owning array, never a view.
- Boolean-mask result — Always a contig owning copy.
- Read-only / non-writeable — IsWriteable==false (set on broadcast views); writes throw.
- Non-owning view — OwnsData==false; writes alias the parent.
- Aligned — ALIGNED flag; always true for managed allocs but a real NumPy axis.

### B. Pairwise (binary-op) paths — MixedTypeKernelKey.Path

- SimdFull — Both operands C-contig same dtype; SIMD baseline.
- SimdScalarRight — RHS is 0-d / scalar-broadcast, LHS is array.
- SimdScalarLeft — LHS is 0-d / scalar-broadcast, RHS is array.
- SimdChunk — Inner dim contig for both, outer strided.
- General — Arbitrary strides on either side; coordinate iteration.
- Mixed dtypes — Orthogonal axis: same layout, different LHS/RHS/result dtypes (NEP50 promotion).

### C. Per-operand variations — NpyIterOpFlags

- Aliased operands — Same buffer on both sides (a + a, out=a); no non-aliasing assumption.
- Overlapping views — Two views with partial overlap (a[1:] and a[:-1]); writes can clobber unread reads.
- In-place output (out=) — Output aliases an input; loop order must respect read-before-write.
- Reduction operand — Output has stride=0 along the reduction axis (REDUCE flag).
- Write-masked operand — WRITEMASKED: write only where mask is true.
- Virtual operand — VIRTUAL: no backing array, computed on demand.
- Buffered / casting operand — CAST / FORCECOPY / HAS_WRITEBACK: type conversion needs a temp.
- Read-only operand — READ without WRITE; matters for output selection.

### D. Iteration-level variations — NpyIterFlags

- Coalesced dimensions — Consecutive axes with matching strides collapsed; ndim=4 may arrive as ndim=1.
- IDENTPERM vs NEGPERM — Axis iteration order: identity vs flipped (negative stride on some axis).
- External loop (EXLOOP) — Kernel sees only the inner axis; outer loop driven by iterator.
- Ranged iteration (RANGE) — Partial traversal of a subset.
- GROWINNER — Inner-loop length varies across outer iterations.
- GATHER_ELIGIBLE — Strided inner axis but dtype supports AVX2 gather.
- EARLY_EXIT — Op supports short-circuit (All/Any/IsAllZero).
- PARALLEL_SAFE — Outer loop has no cross-iteration dependency.

### E. NpyIter composite execution paths

- Source-broadcast + dest-contig — Common reduction shape.
- Source-contig + dest-strided — Writing into a sliced output.
- Buffer-required path — Dtype mismatch or alignment forces NpyIter to insert a temp; kernel sees contig but indirect.
- Reused reduce loops — REUSE_REDUCE_LOOPS: inner-loop kernel runs against successive output positions without re-derivation.

