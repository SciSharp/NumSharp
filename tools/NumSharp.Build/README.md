# NumSharp.Build

Build-time IL weaver for [NumSharp](https://www.nuget.org/packages/NumSharp)'s `[NDScoped]`
deterministic memory reclamation. Installing this package is the whole opt-in — it wires a
post-compile step into **your** project's build, and it is **not a dependency**: it ships MSBuild
targets + a tool only (no `lib/`, no dependency entries), and `dotnet add package` installs it with
`PrivateAssets="all"`, so it never flows into your own package's dependency graph.

## What it does

A NumSharp composition method allocates a handful of intermediate `NDArray`s, keeps one as its
result, and drops the rest. Dropped arrays normally wait on the finalizer to return their pooled
buffers — in a tight loop that lag means cold allocations. Mark the method `[NDScoped]` and keep the
body exactly as you wrote it:

```csharp
using NumSharp;

[NDScoped]
public static NDArray Normalize(NDArray a)
{
    var mean = np.mean(a);           // transient
    var std  = np.std(a);            // transient
    var centered = a - mean;         // transient
    return centered / std;           // the one you keep
}
```

At build time the method is rewritten to open an `NDScope` — the transients are reclaimed the moment
the method exits (exception paths included) while the result survives. Your source keeps its 100 %
original body; the scope exists only in the compiled IL.

**Synchronous iterators work too, under `[NDScoped]`.** An `IEnumerable<NDArray>` /
`IEnumerator<NDArray>` (`yield return`) method is woven through its compiler state machine: one scope
spans the whole enumeration, `yield return`ed elements survive (the consumer owns them), and hoisted
state is reclaimed at the end of iteration or the enumerator's `Dispose()` (a `foreach` `break`
included).

**Async methods, async iterators and `Task`/`ValueTask` returns use `[NDScopedAsync]`.** An
`async Task<NDArray>` / `async ValueTask<NDArray>` method, an `IAsyncEnumerable<NDArray>` async
iterator, or a non-`async` method returning `Task`/`ValueTask` is marked `[NDScopedAsync]` (the
companion attribute for shapes that suspend across `await`). The async ones weave through the state
machine — one scope spans the logical invocation, suspended across `await`s and resumed on whatever
thread continues, so temps handed to an in-flight awaited callee stay alive and everything is
reclaimed when the invocation completes; a non-`async` `Task`/`ValueTask` return yields a completed
task's result immediately and defers reclamation to an incomplete task's completion. Marking a method
with the wrong attribute is a build error (NDW009/NDW010), never a silent unwoven ship.

**A retained argument is protected with `[NDScopedExit]`.** A scope reclaims every array it tracked
except the return value and `out` params — so an array handed to something that *keeps* it (a field
store, a long-lived collection, a captured closure) would be disposed under the retainer. Mark the
retaining parameter and the weaver detaches the argument from the caller's scope at the callee's
entry:

```csharp
public NDArray Weights { get; private set; }
public void Adopt([NDScopedExit] NDArray w) => Weights = w;   // w survives the caller's scope
```

It works with or without a method-level scope attribute, covers `NDArray` / `NDArray[]` / tuples of
NDArrays (a property setter's `value` counts), and is a no-op when there is no ambient scope. An
unsupported parameter type is a build error (NDW014). A raw public-field store has no parameter to
annotate — route it through a setter or call `NDScope.Detach` by hand.

**A companion Roslyn analyzer catches mistakes at compile time — and it ships with NumSharp
itself, not with this package.** The `NumSharp` package carries the analyzer (under its
`analyzers/dotnet/cs/`, applied automatically to any `PackageReference` compile), so it is already
active in every project that can use the attributes — before this weaver is ever installed. It
reports a wrong or unsupported target as a build ERROR — in the editor, before the weave runs: the
wrong attribute (NDW009 = an `async`/`Task` method under `[NDScoped]`, NDW010 = a plain sync method
or synchronous iterator under `[NDScopedAsync]`, NDW011 = both attributes), a hidden `ref NDArray`
egress (NDW002), an unsupported carrier return (NDW003), a body-less method (NDW005), or a
setter-only property (NDW006). The IL-only checks (an unrecognized state machine, a tail-call, an
out-of-date NumSharp) stay with the weaver post-compile. A SOURCE-mode consumer (ProjectReference to
NumSharp.Core + imported targets — no package, so no auto-apply) opts the analyzer in by referencing
the analyzer project with `OutputItemType="Analyzer"`, or by setting `$(NumSharpBuildAnalyzerDll)`
at the built analyzer DLL — the parallel to `$(NumSharpBuildToolDll)`.

With the package **absent**, `[NDScoped]` (which ships in NumSharp itself) is inert metadata: the
method runs unscoped and transients fall back to the finalizer. Adding or removing the package never
changes results — only *when* buffers are reclaimed.

## Install

```bash
dotnet add package NumSharp.Build
```

Requires a project that references NumSharp (that is where `NDScope` and the attribute live) and a
.NET 8+ host runtime for the build tool (it rolls forward to any newer major).

## Escape hatches

| flag | effect |
|---|---|
| `-p:SkipNDScopeWeave=true` | build without weaving (the attribute is inert) |
| `-p:NDScopeWeaveILVerify=true` | additionally run `dotnet-ilverify` on the woven output |

## Documentation

Full reference — which return shapes are woven (tuples, `NDArray[]`, result-struct carriers via
`INDArrayCarrier`, `out` params), the `NDScope` API, hand-scoping, and the `NDW00x` build errors:
<https://scisharp.github.io/NumSharp/docs/ndscoped.html>

Part of the [SciSharp STACK](https://scisharp.github.io/SciSharp/).
