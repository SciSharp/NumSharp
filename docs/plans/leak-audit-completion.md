# Leak-audit coverage completion: handover

Branch: `leak-audit-completion` (off `journey4` @ 256224d4). **Work in progress, not merged.** The
extended gate is intentionally NOT on `journey4` yet: it goes red until the np.ma and histogram leaks
below are fixed.

## Goal (from Eli)

"Audit the leak oracle test for coverage, use the inventory tool to find missings, and make the coverage
complete and whole." The leak oracle is `test/NumSharp.Tests.Oracle/Fuzz/UndisposedIntermediateTests.cs` +
`ScopeAudit.cs` + `NativeAllocationChokepointTests.cs` (categories FuzzMatrix + ScopeAudit).

## Audit results (2026-09-23, inventory tool = `coverage/NumSharp.Tools.ApiInventory`)

- Inventory: 8 [ModuleName] modules = 847 method names + 91 properties + 42 fields.
- Before this branch the sweep measured 138,660 ordinary cases (clean) but **265 of 961 module members
  were never leak-measured**: ndarray 94, np 75, np.ma 62, np.linalg 21, np.random 13.
- Excluded families: `ma_*` (68,860 cases), `index_*` (12,426), error paths (2,441 expects_throw; now
  measured: **0 leaks**), LAPACK (276 cases throw without a backend).
- **np.ma leaks in 36,696 of 68,860 cases**: 84 of 149 ops, up to 29 buffers/call (std 29, median 28,
  var 22, average 12, anom 11, mean 9, domained unary 4-7, domained binary 8, every reduction >= 1).
  Clean: plain Binary (add/multiply/...), domain-free Unary (sin/exp/...), shape ops.
- **Histogram family leaks** (no corpus rows): histogram 4/call, density 8, weights 7, "auto" 11,
  histogram_bin_edges 4, histogramdd 13, histogram2d 11. apply_along_axis 1/call.
- NDW012 (static analyzer) already flags 214 sites: NDMaskedArray.cs 176, np.histogram.cs 20,
  np.histogramdd.cs 14, + NDArray.Indexing.cs 1, poly1d.cs 1, logspace/geomspace 1 each.
- Harness bug (fixed on this branch): the bypass freshness range used `Addr(op) - Offset*isz`, but
  `NDArray.Address` is the BASE for Alias-built corpus operands, so the range was shifted down by
  the offset (latent: 0 disagreements over 137,634 results). Now `Storage.InternalArray.Address`.
- Drift: Fuzz/README.md:153 still says the retired `KnownEscapeFamilies_AreFixed` [OpenBugs] pin holds
  debt red; the chokepoint allowlist pins NDIter.cs=2 but 1 site remains (since 15154b00).

## What this branch adds (all in test/NumSharp.Tests.Oracle/Fuzz/)

| File | State |
|---|---|
| `ScopeAudit.cs` | + `MeasureConfirmedTraffic`/`MeasureConfirmed` (screen, settle, confirm). Done. |
| `UndisposedIntermediateTests.cs` | partial; `SharedSweep` (Lazy, one sweep per process); `SweepResult` with per-family counts + `MeasuredByOp`/`ErrorMeasuredByOp`/`ThrewByOp`; `SweepAccumulator`; per-family non-vacuity floors (verify the index/error floor numbers after the first real run: guessed index success > 5,400, error paths > 8,600); error paths measured; `CoverageKey` (rnd:/grnd:); base-range fix; `AssertNoUnclassifiedEscapes` helper. Compiled OK before the catalogue files were added. |
| `UndisposedIntermediateTests.Families.cs` | ma_* and index_* replays, `DisposeAny`, `MaskedSingletons()` (nomask + masked-constant arrays must be in every never-dispose set, since mask_or/getmask/.mask return them). Compiled OK. |
| `UndisposedIntermediateTests.Backend.cs` | `Corpus_BackendOps_LeaveNoUndisposedIntermediates`: whole ordinary corpus with OpenBLAS (threads=1), Inconclusive if none loads; `BackendOnlyOpKeys`. Compiled OK. |
| `IndexOracleTests.cs` | 6 helpers widened private -> internal (+docs) for the index replay. |
| `OracleSurfaceCoverageTests.cs` | `EquivalentAliases`/`NdarrayAliases`/`MaAliases` widened to internal (shared resolution). |
| `LeakCatalogue.cs`, `.NDArray.cs`, `.Masked.cs` | ~450 direct-invocation entries + `LeakFixture` (+ `Ranges` for the bypass check). `LeakCase.Throws` + `T(...)` = an always-throwing member measured as an ERROR path (a Throws entry that starts returning fails as stale). COMPILES. |
| `LeakSurfaceCoverageTests.cs` | `LeakSurface.Enumerate()` + `EveryInventoryMember_IsLeakAudited`. COMPILES; **not run yet** and still credits catalogue ids / properties STATICALLY (see step 3). |
| `UndisposedIntermediateTests.Catalogue.cs` | NEW (session 2): `SharedCatalogue` (Lazy) + `Catalogue_EveryEntry_LeavesNoUndisposedIntermediates` (warm, confirmed measure, freshness-vs-fixture bypass check, backend entries under OpenBLAS threads=1, harness errors fail), `DirectRunResult`, `TryEnableLeakBackend`, `FreshBytesAny`, `BaseRange`. |
| `UndisposedIntermediateTests.Properties.cs` | NEW (session 2): `SharedPropertyReads` + `EveryPropertyAndField_Read_LeavesNoUndisposedIntermediates`. The TARGET is built INSIDE the region (so a getter that caches an allocation on its owner is caught at the owner's disposal); each region reads twice (write-back in between for settable ones) — parts both reads return are owner-held, the rest are fresh and disposed, also on the error path; MissingBackendException reads are re-measured with OpenBLAS. **GREEN**: 288 members, 539 reads + 15 error paths. |

## Session 2 progress (2026-09-23, worktree `.claude/worktrees/leak-audit`)

Steps 1-2 DONE. Step 3 partly done: the property/field gate is GREEN; the catalogue gate is red ONLY on the
np.ma families (48) and the histogram family (4 ops) — every other catalogue leak is fixed. The full
corpus sweep (`Corpus_AllOps`, incl. ma_*/index_*/error paths), the backend pass and the completeness gate
have NOT been run yet in this session (ma_* replay takes ~13 min until the ma leaks are fixed — run the
catalogue/property tests alone meanwhile:
`dotnet test test/NumSharp.Tests.Oracle -c Release --no-build --framework net10.0 --filter "Name=Catalogue_EveryEntry_LeavesNoUndisposedIntermediates|Name=EveryPropertyAndField_Read_LeavesNoUndisposedIntermediates"`, ~1 s).
The worktree needs `src/NumSharp.Interop.OpenBLAS/runtimes/win-x64/` copied from the main tree (gitignored)
for the backend entries/pass.

Core fixes landed on this branch (each found by the new gates; all documented in place):
- `ndarray.shape`/`Shape` SETTERS (NDArray.cs `SetShapeInPlace`): NumPy's `array_shape_set` over the
  existing `ReshapeCore` port — adopt the no-copy view's dims/strides, else `AttributeError` verbatim.
  The old path COPIED a non-contiguous array in place (`Storage.Reshape` -> CloneData + SetInternalArray),
  stranding 2 buffers per assignment (ARC ref stayed on the old block) and diverging from NumPy.
- `NDArray.ReplaceData` (all 6 overloads): `MoveArcReference` moves the array's counted reference to the new
  buffer. Before: old buffer stranded AND Dispose released a reference on the adopted buffer it never took
  (`x.ReplaceData(nd)` = use-after-free risk for nd's owners).
- `ToJaggedArray`: no densified `GetData<T>()` temp (leak) and rank-2 reads by coordinate — it returned
  WRONG values for a transposed view and read OUT OF BOUNDS on a column slice (`-3e-241`).
- `ndarray.tofile` binary: releases its C-order copy. `np.ogrid`: `[NDScoped]` NdGridLines.
- `poly1d`: constructor detaches its field (was `scope.Returns` -> an enclosing scope released a live
  polynomial's coefficients), `FromFresh`/`WrapQuotient` release every intermediate behind
  operators/deriv/integ, `Call(double)`, `==`, indexer get/set, `np.polyval(poly1d, poly1d)` seed; `np.polymul`
  now disposes its temporary polynomials.
- `np.random.multivariate_normal` (both overloads + `ComputeSvdTransform`): 7 raw scratch blocks freed in
  finally (`FreeScratch`, `DangerousFree` idiom from dirichlet).
- `np.apply_along_axis` releases `buff`/`inarr`; `np.apply_over_axes` releases superseded expand_dims views;
  `np.vectorize` signature mode releases its broadcast views. OWNERSHIP CONTRACT (plan decision, documented
  in each XML doc): arrays a user CALLBACK returns — and slices handed to it — stay the caller's (never
  disposed), because a callback may return an array it keeps. So catalogue entries for callback-taking APIs
  use non-allocating (view / held) callbacks. Eli may want to revisit: a pooled-NDScope freshness probe
  around each callback could consume only arrays created DURING the callback (cheap; changes the contract).

Pre-existing bugs found, NOT fixed (catalogue measures them as error paths with the reason in the label):
`NDArray.AsStringArray` parses a legacy ToString layout (always throws); `GetStringAt`/`SetStringAt` pass ndim
coordinates where GetString/SetString need ndim-1 (always throw); `NDArray.Normalize` is NotImplemented.

## Next steps (in order)

1. np.ma leaks (`src/NumSharp.Core/Ma/NDMaskedArray.cs`) — hand-written scopes as below. Catalogue members
   leaking now include `NDMaskedArray.{all,any,anom,argmax,argmin,argsort,choose,compress,compressed,count,
   cumprod,cumsum,dot,max,mean,min,nonzero,prod,product,ptp,putmask,sort,std,sum,tobytes,trace,
   unshare_mask,var,op_Division,op_Modulus}` and `np.ma.{apply_along_axis,apply_over_axes,fromfunction,
   clump_unmasked,flatnotmasked_contiguous,flatnotmasked_edges,notmasked_contiguous,notmasked_edges,
   compress_nd,compress_rowcols,nonzero,putmask,masked_object,convolve,correlate,corrcoef,cov,polyfit}`.
   The weaver refuses class carriers (`tools/NumSharp.Build/ScopeWeaver.cs` `ImplementsCarrierInterface`
   requires `IsValueType`), so use HAND-WRITTEN scopes: `using var scope = NDScope.Open(); ... return
   Yield(scope, result);` with a private `Yield(NDScope, NDMaskedArray)` that `Returns` `_data` and `_mask`
   (skip NDMaskedConstant). Start with the shared helpers `Unary`/`Binary`/`DomainedBinary`/`ReduceIdentity`/
   `Scan`/`Map1`/`MapSeq`/`MaskPropagate`/`AverageCore`/`MedianAxisMasked`, then each still-leaking public
   method. Field stores (`_mask =` sites) must `NDScope.Detach` the new array (the poly1d lesson: a field egress
   is Detach, never Returns). np.ma.apply_* : same callback ownership contract as np.apply_* (entries need
   non-allocating callbacks — the current int[] entry uses np.ma.sum, switch it like np.apply_over_axes').
   `unshare_mask` replaces `_mask` — decide who owns the superseded mask.
2. Histogram family (`src/NumSharp.Core/Statistics/np.histogram*.cs`): `[NDScoped]` (HistogramResult is an
   INDArrayCarrier; check Histogram2dResult/HistogramddResult implement it). Catalogue: histogram 2..11,
   bin_edges 4..7, histogramdd 13, histogram2d 11 per call.
3. Completeness gate: make `LeakSurfaceCoverageTests.Resolve` credit catalogue ids and properties/fields only
   when `SharedCatalogue`/`SharedPropertyReads` actually MEASURED them (success OR error path; plus their
   `BackendSkipped` sets when no backend loads), and credit BackendOnlyOpKeys only from a shared backend-sweep
   result (make `Corpus_BackendOps` a Lazy too). Add `poly1d` (its operators are NOT enumerated today —
   `OperatorOwners` only lists ndarray/NDArray<T>/NDMaskedArray), DType and any other object owner with
   `op_*` members to `OperatorOwners`, then add catalogue entries for them. Add `FlatIterator.Item`/`NpzFile.Item`
   and other object indexers the gate reports.
4. Run the whole `ScopeAudit` category (net10 + net8): `Corpus_AllOps` (ordinary + ma + index + errors),
   backend pass, catalogue, properties, completeness. Verify the guessed floors (index success > 5,400; error
   paths > 8,600) and tighten all floors to ~5% under the measured counts.
5. Run the full Oracle suite + `test/NumSharp.Tests` (the shape-setter / ReplaceData / poly1d / ToJaggedArray
   changes touch user-visible behaviour; `NDArray.MakeGeneric.Test.cs` uses the Shape setter on a contiguous
   array, still valid). Add unit pins: shape setter AttributeError + no-copy adopt (NumPy probed 2.4.2: t.T.shape
   =(4,3) ok strides (8,32); (12,)/(2,6) AttributeError; strided [:, ::2] (2,6)/(12,)/(3,2,2) ok), ReplaceData ref
   moves, ToJaggedArray on transposed/sliced views.
6. Docs: README "Scope gate" section (families, catalogue, property gate, completeness gate, stale
   `KnownEscapeFamilies_AreFixed` pin line), the NDIter.cs chokepoint pin (2 -> 1), `.claude/skills/oracle`;
   then merge the branch into journey4.

## Cost notes

- Replaying ma_* with today's leaks took 12m47s (every leaking case forces a GC settle), which is over
  CI's 10-minute Oracle step. After the ma fixes it should drop to seconds, like the ordinary sweep (11 s).
- Evidence (probe tests + logs) is in session 2f2daee7's scratchpad `evidence/`.
