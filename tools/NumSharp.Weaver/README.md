# NumSharp.Weaver

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

With the package **absent**, `[NDScoped]` (which ships in NumSharp itself) is inert metadata: the
method runs unscoped and transients fall back to the finalizer. Adding or removing the package never
changes results — only *when* buffers are reclaimed.

## Install

```bash
dotnet add package NumSharp.Weaver
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
