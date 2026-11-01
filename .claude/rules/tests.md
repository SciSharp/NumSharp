---
paths:
  - "test/**"
---

# Tests

Test categories and the filter commands are in the root CLAUDE.md; the oracle has its own rule (`oracle.md`).

## Keeping the suite fast

NumSharp.Tests runs in parallel, one test per worker (`Utilities/TestParallelization.cs`:
`[assembly: Parallelize(Workers = 0, Scope = ExecutionScope.MethodLevel)]`), then every `[DoNotParallelize]` test alone;
`-- MSTest.Parallelize.Workers=1` on the `dotnet test` line runs it serially. The other test projects (Oracle, Interop,
Analyzer) run serially.
- A test that touches process-wide state carries `[DoNotParallelize]` (the full list and the reasons are on
  `TestParallelization`): process-wide counters (pool takes/returns, `GC.CollectionCount`, kernel/program cache sizes),
  forced collections (one suspends every worker), `OpenBlasEngine.Enable`/`Disable` (`TensorEngine.Blas`), environment
  variables, `Console.SetOut`, `np.multithreading`, `np.ma.masked_print_option`, a non-`[ThreadStatic]` engine hook
  (`ComparisonRun.Disabled`, `DefaultEngine.CastLoopBlockElements`, `DirectILKernelGenerator.ClipStreamingOverride`), the
  GLOBAL `np.random` when the test expects its stream (seed a private `np.random.RandomState(seed)` instead: the same
  stream, nothing to opt out of), a wall-clock assertion (measure the FASTEST of several batches), and a race test
  calibrated to own the cores. Method level means a sibling test of the SAME class runs concurrently too.
- A new engine test hook is `[ThreadStatic]`. A bool or counter starts false/0 on every thread, but a `[ThreadStatic]`
  field's initializer runs on ONE thread only, so a hook with another default needs a separate override field.
- A raw pointer (`.Address`) does not root its array: keep the array alive (`GC.KeepAlive`) past the pointer's last read.
- Host settings: `NumSharp.Tests.runsettings` (through `RunSettingsFilePath`) batches results (`BatchSize` 1000; at
  VSTest's default 10 the host serializes every in-progress test on every start and result under a lock), and the csproj
  runs the server GC with 8 heaps. The parallel phase is CPU-bound; its queue follows the compiler's class order
  (namespaces reverse-alphabetical, so `Backends` comes last). Other load on the machine moves a run by seconds, so
  compare interleaved runs; a TRX (`--logger trx`) has each test's start and end, which separates the phases.
- Never force a FULL GC per measurement: a forced full collection marks the whole live heap, which in a full run is mostly
  MSTest's per-test bookkeeping, so a GC-drain loop that is instant alone costs a minute in the suite and grows with every
  test. Tests that read process-global counters use `test/NumSharp.Tests/Utilities/GcQuiescence.cs` (linked into the
  Oracle): `CollectYoung()` for per-window hygiene; `OpenWindow()`/`Undisturbed()` to prove no collection touched the
  window (including a background GC, whose `CollectionCount` moves at its START but whose finalizers run at its END);
  `CollectFull()` once per test at most. Lifetime tests asserting what a collection does to objects they just built use
  `CollectSince(epoch)` ("still reachable": young is exact while no collection ran since the epoch, full otherwise) and
  `WaitCollected(weakRef)` ("now collectable": young first, full as the fallback). A class whose counter windows need a
  clean backlog drains FULL once in `[ClassInitialize]` and young per test. Temp directories holding memmaps go through
  `MappedFileCleanup.DeleteDirectory`.
- Concurrent-collection storms have two budgets (`test/NumSharp.Tests/Collections/StressBudget.cs`): each call site spells
  `StressBudget.Pick(full: <authored>, ci: <per-push>)`. Per-push runs the CI budget; `NUMSHARP_TEST_STRESS=full` restores
  the authored workload, and the nightly `collections-stress` job in `fuzz-soak.yml` runs it. A wall-clock storm gets a
  short per-push window only after calibration: re-introduce the defect it pins and confirm every per-push run still
  fails, pinned to 4 CPUs and unpinned. Size a per-push LIVE SET under the large-object threshold (every interior removal
  copies the whole generation; past 4,096 entries the compact dictionary's index is ≥ 128 KB, so each copy is a gen-2
  collection). A budget shrink is calibrated like a window, at the old and new budget, pinned and unpinned. Traps: a
  MUTATED run is faster than a clean one (a reader that sees the violation throws and leaves, so its writers finish
  sooner: time only clean same-build A/Bs), and shrinking the op count can silently drop a cadence-gated operation
  (FullSurfaceChaos' mid-storm `Clear`: budget the cadence masks with the ops).
- Hot product paths use `[GeneratedRegex]`, never the static `Regex.Match/Split/Replace`: on .NET 8 the static helpers
  share a 15-entry cache whose eviction thrashes once it is full of stale entries (every string slice `nd["1:3"]` would
  re-parse its pattern). Idle polls in stress tests call `Thread.Yield()`, not `Thread.SpinWait`: on .NET 8 a `SpinWait`
  loop is an FCALL the GC cannot suspend, so blocking collections stall.
- A strided view is not a scalar-path probe: the unary kernels gather strided inputs into the vector kernel, so
  "contiguous vs `::2`" compares the vector kernel with itself. Compare against the scalar entry point
  (`NDFloatMath.X(float)`) directly, and confirm with a planted scalar-only and vector-only slip that each assertion can
  fail.
- Oracle coverage gates read `CorpusSurvey`; replays stream `FuzzCorpus.Open`. Never re-parse the whole corpus into
  `FuzzCorpus.Case` objects for a coverage question (op keys, layouts, dtypes, params, outcome kinds): `CorpusSurvey.Files`
  is a header-only scan built once per process (extend `SurveyCase` and its fidelity test if a gate needs a new field). A
  replay that visits each case once enumerates `FuzzCorpus.Open(file)` (count up front, one parsed case alive at a time):
  a materialized whole-tier `List<Case>` is promoted through the GC generations, which costs more than the parse.
- Test hosts run with `TieredPGO=false` (a short JIT-bound run never pays back PGO's instrumented tier; set in the three
  test csprojs) and opt out of EcoQoS: `TestHostQualityOfService` (a `[ModuleInitializer]`, Windows-only) disables
  execution-speed power throttling, because Windows parks a windowless `testhost` on a hybrid CPU's E-cores. A/B
  single-threaded suites pinned to the P-cores (affinity `0xFFFF` on the dev box); unpinned runs mix two populations.
- A test that can never return must FAIL, not hang. NumPy's rejection samplers never accept a draw at some values their
  checks reject (`logseries` at `p = 1`; `zipf` at `a <= 1` or NaN), so a regressed check hangs instead of failing, and a
  local `dotnet test` has no timeout (CI's test steps have `timeout-minutes`). Oracle replays have the per-case
  `CaseWatchdog` (`Fuzz/README.md` → Gate semantics), which ends the host with `FailFast`, naming the case, after 2 min
  (`NUMSHARP_ORACLE_CASE_TIMEOUT_SECONDS`; 0 disables); unit tests that drive those bounds use `[Timeout]` or an explicit
  deadline thread. Give any new test that feeds a sampler its rejected boundary the same guard. Diagnose a hung testhost
  with `dotnet-dump collect -p <pid>`, then `analyze` with `clrthreads` (the Cooperative thread is the spinner),
  `setthread <n>` + `clrstack`, and `dso` → `dumpobj` on the `FuzzCorpus+Case` for its `Id`. A zombie testhost also locks
  its build output (`bin/Debug`): build Release until it is gone.
- Two shell traps: a quoted `--filter` passed from Git Bash through `cmd //c` reaches dotnet with literal quotes (read it
  from a file in the `.cmd` script); `start /b /wait X.cmd` from Git Bash runs X under `cmd /K` (wrap it in `cmd /c`).

## Memory-safety gates: fenced memory and guard pages

A kernel that READS past its operand gives right values (the extra bytes are discarded), and a sentinel margin shows
writes only, so a value test, an oracle case and a sentinel test all pass over such a read until a buffer ends where
mapped memory ends. Two tools make it fail:
- `FencedMemory` (`test/NumSharp.Tests/Utilities/FencedMemory.cs`; Windows, Inconclusive elsewhere): native memory with
  an inaccessible page directly below its first byte and directly above its last (the region is reserved whole, only the
  data pages committed). `fence.Place(bytes, atEnd)` gives the address of a block against the low or the high page, and
  `np.frombuffer((IntPtr)p, bytes, dtype)` makes it an array, so the public API runs over it. A violation ends the test
  host ("Test host process crashed") with the stack of the kernel that left its operand.
- `NUMSHARP_DEBUG_GUARD_PAGES=1` (the pool's diagnostic mode): every pooled buffer ends at an inaccessible page (none
  below). Run a class, the unit suite or the oracle replay with it after writing pointer code:
  `NUMSHARP_DEBUG_GUARD_PAGES=1 dotnet test --no-build -c Release --framework net10.0 --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"`.
  The tests that count pool takes and returns fail in this mode, because it bypasses the pool (in `NumSharp.Tests`:
  `NoOperationDefersItsBufferRelease`, `EightThreads_MixedOps_ZeroErrors_ZeroStrandSlope`, `ScopedSurface_ZeroStrandsPerCall`,
  `TupleCarrier_Polydiv_ZeroStrands`; in the oracle project the `…LeaveNoUndisposedIntermediates` gates and
  `OperatorNot_ReclaimsUntypedMakeGenericWrapper`); a CRASH is the finding. The whole unit suite and the whole oracle
  replay run under guard pages without a crash.
- Rules:
  - A stride kernel is visible only on a view that ENDS at its buffer's last element: `a[::2]` of an even-length buffer
    leaves one element behind the view, which absorbs a one-element over-read. Test `a[1::2]`, `a[2::3]` and the
    reversed forms that start at the last element too.
  - A vector load of N elements reads N elements: a deinterleave of two loads covers one element more than the group
    needs; a gather with a full mask does not.
  - A test whose two sides share a wrong operand proves nothing (an assertion that reads the same wrong mask on both
    sides passes).
- Trap (.NET 8): the .NET 8 runtime's TIER-0 code boxes a `Half` loaded through a pointer (`object o = *(Half*)p;`, also
  `Half h = *p; return h;` and `Unsafe.ReadUnaligned<Half>(p)` boxed) with a read past its two bytes; a float16 at the end
  of mapped memory ends the process with "Fatal error. Internal CLR error. (0x80131506)" and no useful stack. .NET 10 and
  optimized code (`DOTNET_TieredCompilation=0`) do not. Box a `Half` built from its bits (`UnmanagedStorage.HalfAt`:
  `BitConverter.UInt16BitsToHalf(*(ushort*)p)`). A fenced test hits it only while the getter is still tier-0 code, so it
  shows when the test runs ALONE (never in the full parallel run): run a fenced test by itself on net8.0 too.
- Mutation scripts: a test host that ends in the middle of a run still prints a summary line for the tests that finished
  ("Passed! - Failed: 0, Passed: 1, Total: 1"), so an out-of-bounds mutant can read as a survivor. Compare the summary's
  total with a clean run's, or look for "aborted".

## The OpenBugs category

`[OpenBugs]` tests are known-failing reproductions, excluded from CI; remove the attribute when the bug is fixed.
- The files: `OpenBugs.cs` (general), `OpenBugs.Bitmap.cs`, `OpenBugs.ApiAudit.cs`, `OpenBugs.ILKernelBattle.cs`
  (regression pins only), `OpenBugs.Random.cs`, `OpenBugs.BroadcastReduce.cs` (regression pins only), `OpenBugs.Char.cs`
  (regression pins only; `gen_oracle.char_tier` carves the same combos out of the corpus), `OpenBugs.NpLayer.cs` (the
  np-layer dtype plan's divergences, pinned and deferred, NL1 to NL11: np.copyto into StringDType under `where=`
  converting masked-off source elements, complex np.median without `_median_nancheck`, the flat float32 / float64 np.prod
  in SIMD lane order, 1-D np.polyval's complex zero, np.nested_iters / np.einsum argument order, np.array of a ragged
  jagged C# array, the subclass boundary's depth reaching user callbacks, NL8: which NaN a run of NaNs keeps in a
  DEBUG build, where the JIT's reduce step differs from Release (CI runs Release), NL9: np.polyder's term count and its
  missing last product, NL10: np.einsum_path's IndexOutOfRangeException for an ellipsis without its operand, NL11:
  np.isscalar of a Half, a BigInteger and a Type object), `OpenBugs.MatrixOperand.cs`
  (np.gradient and np.meshgrid return plain arrays in their result structs where NumPy returns matrices, MO06 / MO07),
  `OpenBugs.Reductions.cs` (how a reduction STARTS: flat `cumsum` loses a leading `-0.0`; the managed size-1 `dot`/`inner`
  operands take the opposite of NumPy's scalar branch; complex one-element `prod` skips the `1+0j` identity; `median`
  returns the middle element instead of its `np.mean`; complex `cumprod` multiplies element 0 by `1+0j`; the default
  `einsum`'s zero-filled accumulator for bare products; the N-D size-1 flat `cumprod` returning 0-d; which NaN a flat sum
  of several NaNs keeps (and a complex128 short run's NaN per part); an axis sum in a cast loop whose pinned rows are
  longer than `np.getbufsize()`; the plain C-contiguous sum along an outer axis keeping a column's first NaN where NumPy
  keeps the later row's (group 15); raw-bit asserts, because `0.0 == -0.0`; the reversed / broadcast operands' sums and
  means and np.average over a tuple axis are fixed, their tests in `Math/np.sum.ReversedBroadcast.Test.cs` and
  `Statistics/np.average.TupleAxes.Test.cs`), `OpenBugs.FancyIndex.cs` (fancy indexing against NumPy's
  three `mapping.c` routes: an N-D index array that is not C-contiguous beside a slice / ellipsis / integer is read in
  memory order, and the result LAYOUT: NumPy's transposed MapIter view for an index behind a slice, the index arrays'
  order for a lone index or arrays on every axis, the source's order for the other axes of a non-C source, including
  `Generator.permutation` of a non-C 3-D source), `OpenBugs.EmptyStrides.cs` (a zero-size array NumPy allocates has
  all-zero strides, and a slice keeps its parent's strides on an extent-0 axis), `OpenBugs.PolynomialNaN.cs` (the
  polynomial step-table integral kernels keep the wrong NaN per statement: `lagint` float16 overflow, `hermeint` float16
  NaN in `k`, `lagint` float64 `scl=+inf`; raw-bit asserts, win-amd64), `OpenBugs.SortUniqueNaN.cs` (regression pins
  only: the unique family's value routes write NumPy's sort NaN, the positive quiet NaN, for float32 / float64, and
  `np.sort`'s `stable`/`mergesort` keep every NaN's bits), `OpenBugs.Matrix.cs` (the array library below numpy.matrix: a
  fancy assignment into an offset view throws, and three result-layout gaps: cumprod, round of a broadcast operand,
  conjugate of a transposed strided view; the imag-of-a-real-F-array pin is a regression test), `OpenBugs.ErrorParity.cs`
  (regression pins only: the stacking family's empty-list texts, the moveaxis / rollaxis / swapaxes / squeeze AxisError
  texts, moveaxis' length mismatch as NumPy's ValueError), `OpenBugs.NaNOperandOrder.cs`, `OpenBugs.Surface.cs`,
  `OpenBugs.MaSurface.cs`, `OpenBugs.Relayout.cs`, `OpenBugs.ViewFlags.cs`.
- The LIFETIME OpenBugs (lost GC anchors) live in the Oracle project, because the bug exists only in optimized code and
  needs the lifetime child process (`oracle.md`): none is open; a new open gap gets a `Lifetime_LT<nn>_*` test beside the
  aggregate `Lifetime_KnownGaps` (and `LifetimeOracleTests.Scaled.cs`'s `LifetimeScaled_KnownGaps`).
- Auditing the category: run `TestCategory=OpenBugs&TestCategory!=HighMemory` (`[LargeMemoryTest]` adds the OpenBugs
  category to memory-heavy NON-bug tests) in Release AND Debug on both TFMs; a test that passes in all four has its tag
  removed, its comments rewritten as a regression pin, and a `<summary>` of what it guards. Kept on purpose:
  `T2_All_Contiguous_Is_Much_Slower_Than_NumPy` (a wall-clock limit that fails in a Debug build) and
  `BufferReleaseSweepTests.KnownDeferredReleases_StillDeferring` (the tag is the empty backlog's switch).
- Discovery trap: MSTest's `TestClassAttribute` is NOT inherited, so a class with `[TestMethod]`s and no `[TestClass]` of
  its own is never discovered: no filter runs it, tagged or not, and nothing reports it. Two such classes are
  `[LargeMemoryTest]`-only (`DirectILKernelGenerator_LargeArray_BattleTest`, `Unmanaged/AllocationTests`). Find them by
  reflection over the built `NumSharp.Tests.dll` (types declaring `[TestMethod]` methods without the attribute), not by
  grep.
