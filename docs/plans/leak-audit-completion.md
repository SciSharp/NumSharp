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
| `LeakCatalogue.cs`, `.NDArray.cs`, `.Masked.cs` | ~360 direct-invocation entries + `LeakFixture`. **NOT YET COMPILED**; expect some signature fixes. |
| `LeakSurfaceCoverageTests.cs` | `LeakSurface.Enumerate()` (inventory rule + operators + object_surfaces.py owners + NDMaskedArray + NDArray<T>) and `EveryInventoryMember_IsLeakAudited` (+ self-retiring maps). **NOT YET COMPILED.** |

## Next steps (in order)

1. `dotnet build test/NumSharp.Tests.Oracle -c Release` and fix the catalogue/gate compile errors.
2. Write `UndisposedIntermediateTests.Catalogue.cs` with two tests:
   - `Catalogue_EveryEntry_LeavesNoUndisposedIntermediates`: build `LeakFixture`; for each entry run
     warm, then `MeasureConfirmedTraffic(() => DisposeAny(entry.Run(fx), fx.Keep))`; record via
     `SweepAccumulator.Record(entry.Api, null, entry.Label, ...)`; a throwing entry is a HARNESS error
     (fail); backend entries run only after `OpenBlasEngine.Enable(threads: 1)` succeeds (skip + report
     otherwise); finish with `AssertNoUnclassifiedEscapes(r, "scope-audit/catalogue")`.
   - `EveryPropertyAndField_Read_LeavesNoUndisposedIntermediates`: for every `SurfaceKind.Property`/
     `Field` in `LeakSurface.Enumerate()` read on a target (null for static; `np.random`, `np.ma`,
     `np.fft`, `np.emath` singletons; ndarray on a FRESH array + a transposed view per property; object
     owners on fixture instances). Detect fresh results with TWO pre-reads (objects identical across
     both are stable and never disposed); in the region dispose only non-stable parts via `DisposeAny`;
     for settable properties also round-trip write the value just read. A throwing getter is an error
     path (still measured).
3. Run the Oracle ScopeAudit category and fix the harness until only REAL leaks remain.
4. Fix the leaks (zero-leak rule: fix, never allowlist):
   - np.ma (`src/NumSharp.Core/Ma/NDMaskedArray.cs`): the weaver refuses class carriers
     (`tools/NumSharp.Build/ScopeWeaver.cs` `ImplementsCarrierInterface` requires `IsValueType`), so use
     HAND-WRITTEN scopes: `using var scope = NDScope.Open(); ... return Yield(scope, result);` with a
     private `Yield(NDScope, NDMaskedArray)` that `Returns` `_data` and `_mask` (skip NDMaskedConstant).
     Start with the shared helpers `Unary`/`Binary`/`DomainedBinary`/`ReduceIdentity`/`Scan`/`Map1`/
     `MapSeq`/`MaskPropagate`/`AverageCore`/`MedianAxisMasked`, then each still-leaking public method
     (composers like `ptp` need their own scope). Field stores (`_mask =` at lines ~929/941/1155/4501/
     4531/4544) must `NDScope.Detach` the new array, or an enclosing scope frees a live mask. Do NOT scope
     `apply_along_axis`/`apply_over_axes` (user callbacks may keep arrays).
   - Histogram family (`src/NumSharp.Core/Statistics/np.histogram*.cs`): `[NDScoped]` (HistogramResult
     is an INDArrayCarrier; check Histogram2dResult/HistogramddResult implement it).
   - Whatever else the catalogue / read gate finds.
5. Tighten floors, README "Scope gate" section (families, catalogue, completeness gate, stale pin line),
   the NDIter.cs chokepoint pin (2 -> 1), and `.claude/skills/oracle` docs; then commit to journey4.

## Cost notes

- Replaying ma_* with today's leaks took 12m47s (every leaking case forces a GC settle), which is over
  CI's 10-minute Oracle step. After the ma fixes it should drop to seconds, like the ordinary sweep (11 s).
- Evidence (probe tests + logs) is in session 2f2daee7's scratchpad `evidence/`.
