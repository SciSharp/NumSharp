# NDScoping — a runnable tour of NDScope memory reclamation

NumSharp is not refcounted: a dropped `NDArray`'s pooled buffer is reclaimed only when its finalizer
runs, which in a tight loop lags the allocation rate and forces cold allocations. **`NDScope` closes
that gap** — it makes a composition method a *deterministic* reclamation point. This example shows
**every level and layer** of that mechanism, side by side, running, with the reclamation made
visible: each demo asserts its claim via the public `NDArray.IsDisposed` and prints `OK`/`FAIL`, so
the console output is *evidence, not narration*. It doubles as a living gate beside the formal ones
(`NDScopeTests`, `NDScopeWeaveTests`, `verify_build_package.sh`).

> Full spec: [`DESIGN.md`](DESIGN.md). Background: [`../../DISPOSAL-GUIDELINES.md`](../../DISPOSAL-GUIDELINES.md)
> and the API's own XML doc in `src/NumSharp.Core/Backends/NDScope.cs`, which this example makes executable.

## Run it

```bash
cd examples/NDScoping
dotnet run -- examples           # START HERE: the guided tutorial (Examples.cs) — leak -> [NDScoped]
dotnet run                       # run every level, print the PASS/FAIL table + weave coverage
dotnet run -- level A            # one level: 0 | A | B | C | D | Z | cross
dotnet run -- --stress           # 25x sweeps with forced GCs between (a mis-scope -> a wrong value)
dotnet run -- --explain          # print each level's one-line "what this shows" without running
dotnet run -- level D            # the analyzer demo (shells CounterExamples/show-analyzer.sh)
```

**New to NDScope? Read [`Examples.cs`](Examples.cs)** — a single, top-to-bottom, heavily-commented
file that runs one small computation (operators, `np.*`, broadcasting, two chained temporaries) five
ways: leaking, prevented by hand, with `NDScope`, with `[NDScoped]`, and with `[NDScopedAsync]` — so
you see the exact same code go from leak to leak-free. The `Level*` files below are the exhaustive
matrix; `Examples.cs` is the introduction.

The last line of a full run reflects over the built assembly and reports **weave coverage** — the
example proving *it itself was woven* (every `[NDScoped]`/`[NDScopedAsync]` method carries an
`NDScope` local). A non-zero exit code means some demo, or the coverage check, was not `OK`.

## The taxonomy — levels × layers

**Levels** = how much is automated. **Layers** = which result-shape / egress vocabulary a level handles.

| Level | Mechanism | File |
|------|-----------|------|
| **0** Unscoped | nothing — the finalizer backstop (the problem) | `Level0_Unscoped.cs` |
| **A** Manual | `using var scope = NDScope.Open();` + `scope.Returns(…)` | `Level1_HandScope.cs` |
| **B** Woven sync | `[NDScoped]` — the weaver injects Level A into the compiled IL | `Level2_Scoped_Sync.cs` |
| **C** Woven async | `[NDScopedAsync]` — the state-machine / deferral weave | `Level3_ScopedAsync.cs` |
| **D** Guarded | the Roslyn analyzer — NDW00x build errors on a bad target | `CounterExamples/` |
| **Z** The seam | `OpenOrResume`/`Suspend`/`DisposeSlot` — the weaver only | `LevelZ_Seam.cs` |

Layers (shared by A–C): `NDArray`, `NDArray[]`, a `ValueTuple` of NDArrays, any `ITuple`, an
`INDArrayCarrier` result struct (`Carriers.cs`), a bare `IArraySlice`, an `out` parameter, a
scalar/`void`, a synchronous iterator (B) / async shapes (C), and a passthrough (`Returns` is a
no-op). Cross-cutting concerns — nesting, the caller-side `Attach`/`Detach` hot loop, the exception
path, threading, granularity — are in `CrossCutting.cs`.

## How the demos prove reclamation

Every demo hands the probe (`Instrumentation/ReclamationProbe.cs`) its internal temporaries and its
result, and the probe asserts **every temp reclaimed** (`IsDisposed == true`), **the result still
alive** (`IsDisposed == false`), and — where applicable — a correct value. Level 0 flips the
expectation and observes GC reachability with `WeakReference` (the finalizer path drops the buffer
without ever setting `IsDisposed`). No internals, no reflection into buffers — exactly what an
external consumer could write.

## The build wiring (source mode vs package mode)

This example is in-repo, so it consumes the weaver + analyzer in **source mode** — which is itself
instructive: it shows the machinery a real project gets from the NuGet package. See
[`NDScoping.csproj`](NDScoping.csproj): a `ProjectReference` builds the weaver tool (not linked), a
second one applies the analyzer (`OutputItemType="Analyzer"`), and the packaged weave target is
`<Import>`ed directly.

A normal consumer installs one package — the weaver AND the analyzer come with it, and the import is
automatic:

```xml
<PackageReference Include="NumSharp.Build" Version="…" PrivateAssets="all" />
```

Either way, `[NDScoped]` is inert without the weaver (the method just runs unscoped, the pre-weave
finalizer backstop) — so adding or removing the package never changes results, only *when* the
transients' buffers are reclaimed.

## Non-goals

Performance numbers (that is `benchmark/`), NumPy parity (that is the oracle), and exhaustive dtype
coverage (the demos use `float64` for legibility — the reclamation contract is dtype-agnostic). The
example teaches the *shape* of reclamation; it does not re-teach the weaver internals (that is
`DISPOSAL-GUIDELINES.md`).
