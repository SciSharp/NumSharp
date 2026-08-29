# `examples/NDScoping` — design

A single, runnable example project that showcases **every level and layer of NDScope memory
reclamation** in NumSharp — from the unscoped baseline, through the hand-written `NDScope.Open()`
pattern, up to the build-time `[NDScoped]` / `[NDScopedAsync]` weaver and the compile-time analyzer
that guards it. Each demo *proves* what it claims by observing `NDArray.IsDisposed`, so the console
output is evidence, not narration.

> Status: IMPLEMENTED. This document is the spec; the runnable project lives beside it (`dotnet run`
> prints ALL GREEN, `dotnet run -- --stress` too, and `CounterExamples/show-analyzer.sh` prints
> ANALYZER-OK). Nothing here changes library behaviour — it only exercises the existing public
> surface (`src/NumSharp.Core/Backends/NDScope.cs`, `NDScopedAttribute`, `NDScopedAsyncAttribute`,
> `INDArrayCarrier`) plus the source-mode weaver/analyzer wiring (`tools/NumSharp.Build`,
> `tools/NumSharp.Build.Analyzer`).
>
> Two small deltas from the spec below, both forced by the environment: a `.editorconfig` silences
> NDW012/NDW013 (the newer NDArray-leak analyzer would otherwise flag Level 0's deliberate baseline
> and the driver's observation code — the example proves reclamation at runtime via `IsDisposed`
> instead), and `level D` degrades to printing the `show-analyzer.sh` command when the host's spawned
> `bash` is a WSL shell without a usable `dotnet` (it runs the script directly where `bash`+`dotnet`
> are POSIX, e.g. CI). The verb never drags the example's exit code to failure.

---

## 1. Why this example exists

NumSharp is not refcounted: a dropped `NDArray`'s pooled buffer is reclaimed only when its finalizer
runs, which in a tight loop lags the allocation rate and forces cold allocations. `NDScope` closes
that gap — it makes a composition method a **deterministic reclamation point**. There are four ways
to reach for it, at increasing levels of automation, and each has several *layers* (the shapes of
result it must hand back intact). Newcomers need one place that shows all of them side by side,
running, with the reclamation made visible. That is this project.

The guiding principle: **every demo is falsifiable.** A demo that claims "the temporaries are
reclaimed at the method boundary" ends by asserting `temp.IsDisposed == true` and
`result.IsDisposed == false`, and prints `OK`/`FAIL`. If the weaver or the scope ever regressed, the
example would go red — so it doubles as a living, readable gate beside the formal ones
(`NDScopeTests`, `NDScopeWeaveTests`, `verify_build_package.sh`).

---

## 2. The taxonomy — "levels and layers"

**Levels** = *how much is automated* (the vertical axis). **Layers** = *which result-shape /
egress vocabulary* a level must handle (the horizontal axis). The example is organized by level; each
level walks its layers.

| Level | Name | Mechanism | Who writes it |
|------|------|-----------|---------------|
| **0** | Unscoped | nothing — finalizer backstop | the baseline / the problem |
| **A** | Manual | `using var scope = NDScope.Open();` + `scope.Returns(…)` | you, by hand |
| **B** | Woven sync | `[NDScoped]` — weaver injects Level A into the compiled IL | you write the attribute; the build writes the scope |
| **C** | Woven async | `[NDScopedAsync]` — state-machine / deferral weave | ditto, for `await`-suspending shapes |
| **D** | Guarded | the Roslyn analyzer — NDW00x build errors on a wrong/unsupported target | the compiler, before you run anything |
| **Z** | The seam | `OpenOrResume`/`Suspend`/`DisposeSlot`/`ExitIterator`/`ReturnsTask` | **the weaver only** — shown to explain the mechanism, never hand-written |

**Layers** (the egress vocabulary), shared by Levels A–C:

| Layer | Result shape | Egress |
|-------|-------------|--------|
| L-arr | `NDArray` / `NDArray<T>` | `Returns<T>(T)` |
| L-many | `NDArray[]` | `Returns<T>(T[])` |
| L-tup | `ValueTuple` of 2–4 NDArrays | `Returns<T1,T2[,T3[,T4]]>(…)` (typed, no box) |
| L-ituple | any other `ValueTuple`/`Tuple` (arity 5–8, mixed, reference tuple) | `Returns(ITuple)` |
| L-carrier | a result struct implementing `INDArrayCarrier` | `carrier.YieldTo(scope)` |
| L-buffer | bare `IArraySlice` / `UnmanagedStorage` | `Returns(slice)` / `Returns(storage)` (counted ref) |
| L-out | `out NDArray` / `out NDArray[]` parameter | `p = scope.Returns(temp)` |
| L-scalar | `void` / scalar / `string` | scope only (nothing to yield) |
| L-iter | synchronous iterator (`IEnumerable<NDArray>`, `yield`) | per-`yield` egress; Level B only |
| L-async | `Task`/`ValueTask`/async/async-iterator | Level C only |
| L-pass | a passthrough (input returned unchanged) | `Returns` is a provable no-op |

Cross-cutting concerns that are not a single layer but pervade all of them get their own section
(§8): **nesting**, **hot-loop `Attach`/`Detach`**, **the exception path**, **threading**, and
**granularity**.

---

## 3. The observable — how a demo proves reclamation

All demos use one public, honest signal: `NDArray.IsDisposed` (public on `NDArray`). No internals, no
reflection — exactly what an external consumer could write.

The shape every demo takes (`Instrumentation/ReclamationProbe.cs`):

```csharp
// A demo hands its INTERNAL temporaries to `probe` as it makes them, and returns its RESULT.
// The probe then asserts: every temp reclaimed, the result still alive, and (optionally) a value.
public readonly record struct Report(string Name, bool TempsReclaimed, bool ResultAlive, string Detail);

public static Report Run(string name, Func<List<NDArray>, NDArray> body, double? expect = null)
{
    var temps = new List<NDArray>();
    var result = body(temps);                       // the demo runs; the scope closes inside it
    bool reclaimed = temps.Count > 0 && temps.TrueForAll(t => t.IsDisposed);
    bool alive     = result is not null && !result.IsDisposed;
    bool value     = expect is null || Math.Abs(result.GetDouble(0) - expect.Value) < 1e-9;
    return new Report(name, reclaimed, alive && value, …);
}
```

- **Level 0** flips the expectation: right after the call the temps are **not** disposed (finalizer
  backstop); a forced `GC.Collect(); GC.WaitForPendingFinalizers();` then reclaims them — the demo
  prints both observations to make the contrast concrete.
- **Levels A–C** expect `TempsReclaimed == true` **immediately** after the call (deterministic), and
  `ResultAlive == true`.
- For **async** the probe has a `Task`-returning overload; for **iterators** an `IEnumerable`
  overload that also checks each yielded element is alive and owned by the consumer.
- An optional **GC-pressure** mode (`--stress`) re-runs every demo 25× with forced full GCs between
  sweeps (the `verify_build_package.sh` step-15 trick): a mis-scoped result whose buffer was
  reclaimed early surfaces as pool-reuse corruption, i.e. a wrong value.

The probe never touches `NDScope.Current`/`Track` (internal) — the demos append their temps
explicitly, which is exactly how the async/iterator tests do it and keeps the example an
external-style consumer.

---

## 4. Project layout

```
examples/NDScoping/
├── README.md                     # 1-screen: what it is, `dotnet run`, the guided-tour index, and the
│                                 #   real-consumer (NuGet) snippet next to the in-repo (source) one
├── DESIGN.md                     # this document
├── NDScoping.csproj              # the runnable console app — source-mode weaver + analyzer wired in
├── Program.cs                    # the driver: parses args, runs each level, prints a PASS/FAIL table
├── Instrumentation/
│   └── ReclamationProbe.cs       # Run(...) helpers + the Report table renderer
├── Level0_Unscoped.cs            # the baseline / the problem (temps linger until the finalizer)
├── Level1_HandScope.cs           # NDScope.Open() by hand — every egress layer + passthrough
├── Level2_Scoped_Sync.cs         # [NDScoped] — every synchronous layer + iterator + property + idempotence
├── Level3_ScopedAsync.cs         # [NDScopedAsync] — every async/Task layer + deferral + cross-await survival
├── Carriers.cs                   # the INDArrayCarrier result struct(s) used by Levels 1–3
├── LevelZ_Seam.cs                # the low-level seam, hand-driven — "the weaver writes this, you never do"
├── CrossCutting.cs               # nesting · Attach/Detach hot loop · exception path · threading · granularity
└── CounterExamples/              # Level D — the ANALYZER, shown by a project that is EXPECTED TO FAIL
    ├── README.md                 # the expected NDW00x error per shape + the one-liner to reproduce
    ├── NDScoping.CounterExamples.csproj   # analyzer applied; NOT in the solution; excluded from CI
    ├── BadShapes.cs              # NDW002/003/005/006/009/010/011 — one attributed method per code
    └── show-analyzer.sh          # builds the project, asserts each NDW code appears → turns "fails to build" green
```

`Level4_Carriers` is folded into `Carriers.cs` (a support file, not a level). The counter-examples
live in their own **separate project** because they must not compile — see §7.

---

## 5. Build wiring (the example teaches this too)

The example is in-repo, so it consumes the weaver + analyzer in **source mode** — which is itself
instructive (it shows the machinery a real project gets from the NuGet package). `NDScoping.csproj`:

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFrameworks>net8.0;net10.0</TargetFrameworks>
    <Nullable>disable</Nullable>
    <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
    <LangVersion>latest</LangVersion>
    <!-- source-mode weaver: point the imported target at the freshly built tool -->
    <NumSharpBuildToolDll>$(MSBuildThisFileDirectory)..\..\tools\NumSharp.Build\bin\$(Configuration)\net8.0\NumSharp.Build.dll</NumSharpBuildToolDll>
  </PropertyGroup>

  <ItemGroup>
    <ProjectReference Include="..\..\src\NumSharp.Core\NumSharp.Core.csproj" />
    <!-- build the weaver TOOL (do not link it) so $(NumSharpBuildToolDll) exists at weave time -->
    <ProjectReference Include="..\..\tools\NumSharp.Build\NumSharp.Build.csproj"
                      ReferenceOutputAssembly="false" Private="false" />
    <!-- apply the compile-time ANALYZER (the standard source-mode analyzer reference) -->
    <ProjectReference Include="..\..\tools\NumSharp.Build.Analyzer\NumSharp.Build.Analyzer.csproj"
                      ReferenceOutputAssembly="false" OutputItemType="Analyzer" />
  </ItemGroup>

  <!-- the packaged weave target, imported directly (the source-mode recipe) -->
  <Import Project="..\..\tools\NumSharp.Build\build\NumSharp.Build.targets" />
</Project>
```

`README.md` puts the **real-consumer** form right beside it, so a reader learns both:

```xml
<!-- A normal consumer just installs the package — the weaver AND the analyzer come with it: -->
<PackageReference Include="NumSharp.Build" Version="…" PrivateAssets="all" />
```

`NDScopeWeaveTests`' invariant applies to the example too: every `[NDScoped]`/`[NDScopedAsync]`
method in the built assembly must carry an `NDScope` local. `Program.cs` reflects over its own
assembly and prints that coverage as the last line — the example proves *it itself was woven*.

---

## 6. Level-by-level content

Each row below is one demo method. "Observed" is what the probe asserts; that assertion is the demo's
whole point.

### Level 0 — Unscoped (`Level0_Unscoped.cs`) — the problem

| Demo | Body | Observed |
|------|------|----------|
| `Unscoped_TempsLinger` | a 3-step composition with **no** scope; append each temp | right after return: temps **NOT** disposed; after a forced GC + finalizers: temps disposed. Prints both → motivates every level below. |

### Level A — Manual `NDScope.Open()` (`Level1_HandScope.cs`)

Every layer, written by hand, so the reader sees exactly what the weaver will later inject.

| Demo | Layer | Egress shown | Observed |
|------|------|--------------|----------|
| `Hand_Single` | L-arr | `return scope.Returns((a*b + c));` | temps gone, result alive |
| `Hand_Array` | L-many | `scope.Returns(new[]{ … })` | temps gone, both elements alive |
| `Hand_Tuple2` | L-tup | `scope.Returns((q, r))` | temps gone, both alive |
| `Hand_ITuple` | L-ituple | `scope.Returns((ITuple)(a, 5, b))` then return the tuple | NDArray members alive, scalar untouched |
| `Hand_Carrier` | L-carrier | `result.YieldTo(scope)` on a `PairResult` (see `Carriers.cs`) | both fields alive |
| `Hand_Buffer` | L-buffer | `return scope.Returns(nd.Storage.InternalArray);` | returned buffer survives the scope's Release of the NDArray sharing it |
| `Hand_Out` | L-out | `r = scope.Returns(temp);` (bool + `out NDArray`) | out-value alive, internals gone |
| `Hand_Scalar` | L-scalar | `return t.GetDouble(0);` (scope-only) | temps gone; scalar correct |
| `Hand_Passthrough` | L-pass | `return scope.Returns(input);` (input built before the scope) | `Returns` is a **no-op**; input untouched (rule R2 for free) |
| `Hand_Exception` | cross | throw mid-body inside the `using` | temps reclaimed **on the exception path** too |

### Level B — `[NDScoped]` (`Level2_Scoped_Sync.cs`) — the weaver

The *same* layers, but the source keeps its 100 % original body and carries only the attribute. The
demo pairs each with its Level-A twin so the reader sees "identical behaviour, zero boilerplate."

| Demo | Layer | Signature (attribute + body, unscoped) | Observed |
|------|------|----------------------------------------|----------|
| `Scoped_Single` | L-arr | `[NDScoped] static NDArray F(NDArray a, List<NDArray> t)` | same as `Hand_Single` |
| `Scoped_Array` | L-many | `[NDScoped] static NDArray[] F(…)` | same as `Hand_Array` |
| `Scoped_Tuple` | L-tup | `[NDScoped] static (NDArray,NDArray) F(…)` | same as `Hand_Tuple2` |
| `Scoped_Carrier` | L-carrier | `[NDScoped] static PairResult F(…)` | same as `Hand_Carrier` |
| `Scoped_Buffer` | L-buffer | `[NDScoped] static IArraySlice F(…)` | same as `Hand_Buffer` |
| `Scoped_Out` | L-out | `[NDScoped] static bool F(…, out NDArray r)` | same as `Hand_Out` |
| `Scoped_Void` | L-scalar | `[NDScoped] static void F(…, out NDArray r)` | out alive, internals gone |
| `Scoped_Property` | L-arr | `[NDScoped] static NDArray Prop => …;` | property getter is woven |
| `Scoped_Iterator` | L-iter | `[NDScoped] static IEnumerable<NDArray> F(…) { … yield … }` | each yielded element owned by the consumer (alive); hoisted state reclaimed at the end / at `Dispose()` on early `break` |
| `Scoped_Idempotent` | cross | `[NDScoped]` **and** a hand-written `using var scope = NDScope.Open()` | the weaver **skips** an already-scoped body (no double-wrap); behaves once |

A short `[Reflection]` panel prints, for two of these, the `NDScope` local the weaver added (and, for
the iterator, the `<>ndscope` field on the compiler state machine) — the reader *sees* the injection.

### Level C — `[NDScopedAsync]` (`Level3_ScopedAsync.cs`) — the async weaver

| Demo | Layer | Signature | Observed |
|------|------|-----------|----------|
| `Async_Task` | L-async | `[NDScopedAsync] static async Task<NDArray> F(…)` | temps survive the `await`; reclaimed at completion; result alive |
| `Async_ValueTask` | L-async | `[NDScopedAsync] static async ValueTask<NDArray> F(…)` | ditto |
| `Async_Void` | L-async | `[NDScopedAsync] static async void F(…)` | fire-and-forget; reclaimed at completion (observed via a signal) |
| `Async_Iterator` | L-async | `[NDScopedAsync] static async IAsyncEnumerable<NDArray> F(…)` | each `await foreach` element alive; state reclaimed at the end |
| `Async_NonAsyncTask` | L-async | `[NDScopedAsync] static Task<NDArray> F(…) => Compute…;` (**no** `async`) | a COMPLETED task yields now; an INCOMPLETE one **defers** reclamation to completion — the in-flight callee's operand is protected (demo holds a `SemaphoreSlim`, checks the operand is alive *while in flight*, then reclaimed after completion) |
| `Async_ValueTaskPreserve` | L-async | non-`async` `ValueTask<NDArray>` return | the incomplete VT is `Preserve()`d; the caller receives the multi-observable form and reads the right value |
| `Async_CrossAwaitSurvival` | cross | `await`s a slow callee that reads a temp handed to it | proves the temp is NOT reclaimed at the suspend point (a scope-per-segment bug would dispose it) |

### Level Z — the seam (`LevelZ_Seam.cs`) — mechanism only

A single demo that hand-drives the exact protocol the woven `MoveNext` performs —
`OpenOrResume(ref slot)` → `Suspend` before a simulated schedule → resume on another thread →
`Returns` the result → `DisposeSlot` — mirroring `NDScopeAsyncTests`. Header comment in bold:
**"This is what the `[NDScopedAsync]` weaver emits. You never write it. It is here only to show the
machinery underneath Level C."** Proves: temps survive suspension, reclaimed at `DisposeSlot`, and the
same scope object spans a cross-thread resume.

---

## 7. Level D — the analyzer (`CounterExamples/`) — a project that must NOT compile

The compile-time guardrails cannot live in a runnable project, because their whole point is that the
build fails. They get a **sibling project that is deliberately un-buildable**, excluded from the
solution and from CI, plus a script that turns "the build fails" into a green assertion.

`CounterExamples/BadShapes.cs` — one attributed method per diagnostic, each with the expected code in
a comment:

| Method | Attribute + shape | Expected error |
|--------|-------------------|----------------|
| `RefEgress` | `[NDScoped] void M(ref NDArray a)` | **NDW002** hidden `ref` egress |
| `BadCarrier` | `[NDScoped] List<NDArray> M(…)` | **NDW003** unsupported carrier |
| `NoBody` | `[NDScoped] extern NDArray M(…)` | **NDW005** no body |
| `SetterOnly` | `[NDScoped]` on a set-only property | **NDW006** setter-only |
| `AsyncUnderSync` | `[NDScoped] async Task<NDArray> M(…)` | **NDW009** async under `[NDScoped]` |
| `SyncUnderAsync` | `[NDScopedAsync] NDArray M(…)` | **NDW010** sync under `[NDScopedAsync]` |
| `HasBoth` | `[NDScoped][NDScopedAsync] …` | **NDW011** both attributes |

`CounterExamples/show-analyzer.sh` (the demo): `dotnet build` the project, assert the build **fails**
and that each of the seven `error NDWxxx` codes appears in the output; print `ANALYZER-OK`. This is
the same layer split `verify_build_package.sh` step 11 uses — the analyzer catches these at
CoreCompile and preempts the weaver.

`CounterExamples/NDScoping.CounterExamples.csproj` wires **only the analyzer** (no weave needed —
it never gets past compile) via the source-mode `OutputItemType="Analyzer"` reference, and carries
`<IsPackable>false</IsPackable>` + a comment that it is intentionally excluded from `SciSharp.NumSharp.sln`
and `-p:SkipNDScopeWeave` has no effect on it (the analyzer is independent of the weave).

`README.md` there shows the expected output verbatim, so a reader who does not want to run it still
sees exactly what the analyzer says and why.

---

## 8. Cross-cutting demos (`CrossCutting.cs`)

The contracts that are not a single result-shape but govern correct use at every level:

| Demo | Shows | Observed |
|------|-------|----------|
| `Nesting` | inner scoped call whose result the outer caller drops | the ENCLOSING scope reclaims the dropped inner result (`Returns` re-parents into the parent) |
| `HotLoop_Naive` vs `HotLoop_Attach` | a caller loop that receives scoped results | naive: results accumulate (caller owns them, scope can't help across calls); `NDScope.Attach(r)` inside the caller's own scope adopts each received result → reclaimed per iteration. **The two-audience rule made concrete.** |
| `Detach_ForCache` | a scoped method that caches an array into a `static` field | `NDScope.Detach(nd)` removes it from the scope so it OUTLIVES the boundary; the rest of the method's temps are still reclaimed |
| `Granularity` | scope a CALL vs a caller LOOP | scoping the loop holds thousands of temps simultaneously (pool-bucket overflow); scoping the call reclaims each — a timing/`TotalAllocated`-style contrast in the note, IsDisposed count in the assert |
| `Threading` | a `[ThreadStatic]` scope + a parallel region | arrays built on worker threads see no ambient scope (finalizer fallback) unless each worker opens its own — printed, not asserted-red (it is correct behaviour, not a bug) |

These map 1:1 to the "Ownership rules become structural", "Granularity", "Threading" and "Nesting"
paragraphs of `NDScope`'s XML doc — the example is the executable form of that doc.

---

## 9. The driver & output (`Program.cs`)

```
dotnet run                     # run every level, print the table
dotnet run -- level A          # one level (0 | A | B | C | D | Z | cross)
dotnet run -- --stress         # 25× sweeps with forced GCs between (mis-scope → corrupted value)
dotnet run -- --explain        # print each demo's one-line "what this shows" without running
```

Expected shape of the output (illustrative):

```
NDScoping — every level and layer of NDScope reclamation
────────────────────────────────────────────────────────────────
LEVEL 0  Unscoped (the problem)
  Unscoped_TempsLinger      temps after return: ALIVE   after GC: reclaimed   ⇒ finalizer backstop
LEVEL A  Manual  NDScope.Open()
  Hand_Single      temps reclaimed ✓   result alive ✓   value ✓   OK
  Hand_Array       …                                             OK
  … (10 demos)
LEVEL B  [NDScoped]  (woven)
  Scoped_Single    temps reclaimed ✓   result alive ✓   value ✓   OK   (woven: NDScope local present)
  Scoped_Iterator  yielded elements owned ✓   state reclaimed ✓          OK
  … (10 demos)
LEVEL C  [NDScopedAsync]  (woven)
  Async_Task           temps survived await ✓   reclaimed at completion ✓   OK
  Async_NonAsyncTask   operand alive in-flight ✓   reclaimed after ✓         OK
  … (7 demos)
LEVEL Z  the seam (mechanism)
  Seam_HandDriven  cross-thread scope ✓   reclaimed at DisposeSlot ✓        OK
CROSS-CUTTING
  Nesting · HotLoop(Attach) · Detach · Granularity · Threading             OK
────────────────────────────────────────────────────────────────
weave coverage: 21/21 attributed methods carry an NDScope local            OK
ALL GREEN
```

`Program.cs` returns non-zero if any demo is not `OK`, so the example is runnable AND assertive. The
analyzer (Level D) is reached via `dotnet run -- level D`, which shells `CounterExamples/show-analyzer.sh`
and reports `ANALYZER-OK` / the codes it saw (it does not build the counter-examples into itself).

---

## 10. Solution / CI integration & non-goals

- **`NDScoping.csproj`** is added to `SciSharp.NumSharp.sln` (like `NeuralNetwork.NumSharp`), so it
  builds with the solution and stays honest.
- **`CounterExamples/` is NOT added to the solution** — it must fail to compile. It is exercised only
  by its own `show-analyzer.sh`, and the example's `README` says so in bold.
- CI: optional. If wired, it is `dotnet run --project examples/NDScoping -- --stress` (must print
  `ALL GREEN`) plus `examples/NDScoping/CounterExamples/show-analyzer.sh` (must print `ANALYZER-OK`).
  Neither is required for the release gate; they are a readable complement to the formal gates.
- **Non-goals:** performance numbers (that is `benchmark/`), NumPy parity (that is the oracle),
  exhaustive dtype coverage (the demos use `float64` for legibility — the reclamation contract is
  dtype-agnostic), and re-teaching the weaver internals (that is `DISPOSAL-GUIDELINES.md` /
  `docs/website-src/docs/ndscoped.md`, which this example links to).

---

## 11. Signing / house rules

Follow the existing example convention (`examples/NeuralNetwork.NumSharp`): inherit repo signing
(`Directory.Build.props` supplies `Open.snk`), `TargetFrameworks=net8.0;net10.0`, `LangVersion=latest`.
The example uses only the **public** surface (`NDScope`, the two attributes, `INDArrayCarrier`,
`NDArray.IsDisposed`) — no `InternalsVisibleTo`, so it reads exactly like third-party consumer code.

---

## 12. Implementation checklist

1. `NDScoping.csproj` + `Program.cs` driver + `Instrumentation/ReclamationProbe.cs`.
2. `Level0` … `LevelZ` + `Carriers.cs` + `CrossCutting.cs` — the demo catalogs in §6/§8.
3. `CounterExamples/` project + `BadShapes.cs` + `show-analyzer.sh` + its `README`.
4. `README.md` — the guided tour, the `dotnet run` verbs, and the source-mode-vs-package-mode csproj
   pair.
5. Add `NDScoping.csproj` (only) to `SciSharp.NumSharp.sln`.
6. Verify: `dotnet run -- --stress` prints `ALL GREEN`; `show-analyzer.sh` prints `ANALYZER-OK`; the
   weave-coverage line reports full coverage.

Cross-references: `src/NumSharp.Core/Backends/NDScope.cs` (the API + its XML doc, which this example
makes executable), `DISPOSAL-GUIDELINES.md`, `docs/website-src/docs/ndscoped.md`,
`tools/NumSharp.Build` + `tools/NumSharp.Build.Analyzer`, and the formal gates `NDScopeTests` /
`NDScopeWeaveTests` / `NDScopeAsyncTests` / `verify_build_package.sh`.
```
