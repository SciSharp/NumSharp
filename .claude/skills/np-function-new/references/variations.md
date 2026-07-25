# The Variation Matrix — the input space a parity claim covers

"np.foo matches NumPy" is a claim over this whole space, not over `np.arange(10)`. Phase 1 probes
the families that can reach the op; Phase 4 tests them. **51 distinct variations** — 25 single-array
layouts, 6 pairwise paths, 8 per-operand flags, 8 iteration flags, 4 composite execution paths.

Concrete builders for the layout families live in `test/oracle/layout_catalog.py` (the fuzz-corpus
builders — 26 single + 9 pair + 5 where recipes); wiring an op into the oracle (Phase 4) exercises
them automatically. Families C–E are exercised through NDIter tests and the `operand`/`layout`
benchmark subsystems.

## A. Single-array layouts (25)

- **C-contiguous** — Row-major, `stride[-1]==1` and `stride[i]==shape[i+1]*stride[i+1]`; baseline
  fast path via `IsContiguous`.
- **F-contiguous** — Column-major, `stride[0]==1`; 1-D arrays are both. Detected via
  `IsFContiguous`.
- **Strided / non-contiguous** — Arbitrary strides, neither C nor F; built via step slicing or axis
  swap.
- **Transposed** — Strides permuted by `.T` / swapaxes / moveaxis; usually non-contig.
- **Negative-stride view** — Reversed slicing (`[::-1]`); strides are signed-negative.
- **Simple slice** — `offset!=0`, not broadcast; fast GetOffsetSimple path (`IsSimpleSlice`).
- **Sliced + composed** — `a[1:5].T`, `a[1:3][:,None,:]`; offset combined with permutation or
  broadcast.
- **Broadcasted** — stride=0 with dim>1 (`BROADCASTED` flag); read-only per NumPy.
- **Scalar-broadcast** — All strides zero (`IsScalarBroadcast`); load value once and reuse.
- **Partial broadcast** — Some axes stride=0, others not; the common `(1,N)→(M,N)` case.
- **Scalar (0-d)** — `ndim==0`, `size==1`, no strides.
- **0-D view from integer indexing** — `a[0,0,0]` shares storage; distinct from `np.array(5.0)`
  which owns.
- **1-element 1-D** — `ndim==1`, `size==1`; ambiguous against 0-d in sloppy paths.
- **Empty** — `size==0` (e.g. `np.zeros((0,3))`); reductions must return identity.
- **Empty + composed** — `np.zeros((0,3))[::2,:]`; rare but must not crash.
- **NewAxis-inserted dim** — `a[None,:]` adds dim=1, stride=0; not flagged broadcast since dim=1.
- **Singleton dim (dim=1)** — Stride is moot; NumPy treats as contig.
- **Higher-rank (5+D)** — Stack-allocated coord/stride arrays in kernels may have bounds.
- **Stride > bufferSize** — Negative-stride views can have `offset + stride*(dim-1) >= bufferSize`.
- **Reshape view vs copy** — Reshape returns a view when contiguity allows, materializes otherwise.
- **Fancy-indexed result** — Always a fresh C-contig owning array, never a view.
- **Boolean-mask result** — Always a contig owning copy.
- **Read-only / non-writeable** — `IsWriteable==false` (set on broadcast views); writes throw.
- **Non-owning view** — `OwnsData==false`; writes alias the parent.
- **Aligned** — `ALIGNED` flag; always true for managed allocs but a real NumPy axis.

## B. Pairwise (binary-op) paths (6) — `MixedTypeKernelKey.Path`

- **SimdFull** — Both operands C-contig same dtype; SIMD baseline.
- **SimdScalarRight** — RHS is 0-d / scalar-broadcast, LHS is array.
- **SimdScalarLeft** — LHS is 0-d / scalar-broadcast, RHS is array.
- **SimdChunk** — Inner dim contig for both, outer strided.
- **General** — Arbitrary strides on either side; coordinate iteration.
- **Mixed dtypes** — Orthogonal axis: same layout, different LHS/RHS/result dtypes (NEP50
  promotion).

## C. Per-operand variations (8) — `NDIterOpFlags`

- **Aliased operands** — Same buffer on both sides (`a + a`, `out=a`); no non-aliasing assumption.
- **Overlapping views** — Two views with partial overlap (`a[1:]` and `a[:-1]`); writes can clobber
  unread reads.
- **In-place output (`out=`)** — Output aliases an input; loop order must respect
  read-before-write.
- **Reduction operand** — Output has stride=0 along the reduction axis (REDUCE flag).
- **Write-masked operand** — WRITEMASKED: write only where mask (ARRAYMASK) is true. Enforced ONLY
  at buffered copy-back (NumPy parity); unbuffered = kernel contract.
- **Virtual operand** — VIRTUAL: null operand, allocate-equivalent in NumPy 2.x (real backing
  array, dtype request discarded → common dtype).
- **Buffered / casting operand** — CAST / FORCECOPY / HAS_WRITEBACK: type conversion needs a temp.
- **Read-only operand** — READ without WRITE; matters for output selection.

## D. Iteration-level variations (8) — `NDIterFlags`

- **Coalesced dimensions** — Consecutive axes with matching strides collapsed; ndim=4 may arrive
  as ndim=1.
- **IDENTPERM vs NEGPERM** — Axis iteration order: identity vs flipped (negative stride on some
  axis).
- **External loop (EXLOOP)** — Kernel sees only the inner axis; outer loop driven by iterator.
- **Ranged iteration (RANGE)** — Partial traversal of a subset.
- **GROWINNER** — Inner-loop length varies across outer iterations.
- **GATHER_ELIGIBLE** — Strided inner axis but dtype supports AVX2 gather.
- **Early exit** — short-circuit (All/Any/IsAllZero) is a KERNEL property
  (`SupportsEarlyExit`/`ShouldExit`), not an iterator flag.
- **PARALLEL_SAFE** — iteration range splittable across workers: no REDUCE operand, ≤1 WRITE
  operand with COPY_IF_OVERLAP-resolved overlap (`IsParallelSafe`).

## E. NDIter composite execution paths (4)

- **Source-broadcast + dest-contig** — Common reduction shape.
- **Source-contig + dest-strided** — Writing into a sliced output.
- **Buffer-required path** — Dtype mismatch or alignment forces NDIter to insert a temp; kernel
  sees contig but indirect.
- **Reused reduce loops** — REUSE_REDUCE_LOOPS: inner-loop kernel runs against successive output
  positions without re-derivation.
