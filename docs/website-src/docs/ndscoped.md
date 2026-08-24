# `NumSharp.Weaver` — build-time `[NDScoped]` memory reclamation

`NumSharp.Weaver` is the optional build-time package that makes a composition method eagerly return
its transients' pooled buffers instead of waiting on the finalizer. Mark a method `[NDScoped]` and,
at build time, the weaver rewrites it to open an `NDScope` — an ambient reclamation scope that tracks
every `NDArray` the method constructs and disposes the ones it does not return the moment the method
exits, on the exception path included. **Your source keeps its 100 % original body**; the scope is
injected into the compiled IL, so the reclamation is invisible in the code and free of boilerplate.

NumSharp.Core already uses this weaver on its own build — dozens of `np.*` composition methods carry
`[NDScoped]`. This package packages the same transform so you can apply it to *your* composition
methods, and `NumSharp.Core` stays 100 % managed with nothing to install: with the package absent, an
`[NDScoped]` method simply runs unscoped and its transients fall back to the finalizer — the exact
pre-weave behaviour.

**On this page:** [Why it exists](#why-this-package-exists) · [Quick start](#quick-start) ·
[What the weaver injects](#what-the-weaver-injects) · [Which returns are woven](#which-returns-are-woven) ·
[The `NDScope` API](#the-ndscope-api) · [Hand-scoping](#hand-scoping) ·
[How the build step works](#how-the-build-step-works) · [When to use it](#when-to-use-it) ·
[Escape hatches](#escape-hatches) · [Troubleshooting](#troubleshooting)

> Mono.Cecil transform, invoked after each per-`TargetFramework` compile · net8.0 / net10.0.
> Behaviour is covered by gates: [`NDScopeTests`][gate-contract] (the scope contract),
> [`NDScopeWeaveTests`][gate-coverage] (every attributed method was woven),
> [`NDScopeWeaveCarrierTests`][gate-carrier] and [`NDScopeWeaveNestingTests`][gate-nesting]
> (tuple / carrier / storage egress and nesting), and [`StrongNameTests`][gate-sign] (identity
> survives the re-sign).

## Why this package exists

A dropped `NDArray`'s buffer is reclaimed only when its finalizer runs. In a tight loop the finalizer
lags the allocation rate, so most calls miss the warm buffer pool and pay a cold `NativeMemory.Alloc`
plus a first-touch page fault, on top of ~950 ns of finalization per dropped result. NumPy does not
have this problem: it is refcounted, so an intermediate array's memory is freed **deterministically**
the instant the last reference drops.

A NumSharp composition — `np.cross`, `np.linalg.norm`, a polynomial fit, your own helper built out of
`np.*` calls — allocates a handful of intermediate `NDArray`s, keeps one as its result, and drops the
rest. Under the finalizer backstop those dropped intermediates linger. `[NDScoped]` closes the gap:
it makes the method's boundary a **deterministic reclamation point**, so every transient it owns is
returned to the pool the moment the method exits — matching NumPy's refcount discipline at the one
place it matters, the composition boundary, without asking the caller to dispose anything.

This is the same lever NumSharp.Core pulls internally. The package lets you pull it in your code.

## Quick start

```bash
dotnet add package NumSharp.Weaver
```

Referencing the package is the whole opt-in: it wires the weave step into your build. Now mark any
method that owns `NDArray` transients `[NDScoped]` and keep the body exactly as you wrote it —

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

At build time the method is rewritten to what the hand-written pattern spells — a scope opened at the
top, the whole body wrapped in `try`/`finally`, the result routed through `scope.Returns(...)` — so
`mean`, `std` and `centered` are reclaimed on return while the result survives (see
[What the weaver injects](#what-the-weaver-injects)). The *input* `a` is never touched: it was
constructed before the scope opened, so it is never tracked.

If the `NumSharp.Weaver` package is **absent**, the `[NDScoped]` attribute is inert and `Normalize`
runs exactly as its source reads — unscoped, the finalizer backstop. Adding or removing the package
never changes results, only *when* the transients' buffers come back.

<!-- Tests: NDScopeTests.TempsReclaimed_ResultSurvives; NDScopeWeaveTests.EveryScopedMethod_WasWoven -->

## What the weaver injects

Per `[NDScoped]` method, the transform is exactly the hand-written scope pattern, applied to the
compiled IL:

```csharp
[NDScoped]                                          // what you WRITE — the original body, unchanged
public static NDArray<bool> logical_and(NDArray x1, NDArray x2)
{
    var b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);
    var b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
    return (b1 & b2).MakeGeneric<bool>();
}

// what the built assembly CARRIES (logically):
public static NDArray<bool> logical_and(NDArray x1, NDArray x2)
{
    using var scope = NDScope.Open();               // 1. prologue, OUTSIDE the protected region
    try                                             // 2. the whole original body becomes the try
    {
        var b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);
        var b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
        return scope.Returns((b1 & b2).MakeGeneric<bool>());   // 3. every ret routed through Returns
    }
    finally { scope.Dispose(); }                    // 4. dispose reclaims every tracked, un-yielded array
}
```

Three properties make the rewrite trivially safe:

- **The CLR forbids `ret` inside a protected region.** Wrapping the body in `try`/`finally` forces
  every original return through a rewritable seam; bodies that already contain `try`/`using` compile
  their inner returns to `leave`-chains, and only the real `ret` instructions are touched (mutated in
  place, so branches and nested-handler boundaries that referenced them stay valid).
- **`scope.Returns(x)` is the one egress rule.** It re-tracks the yielded array into the *parent*
  scope (so an enclosing scope still reclaims it if the caller drops it) and is a provable no-op for
  an array the scope never tracked — an input passthrough, a caller-supplied `out`. So *"wrap every
  egress in `Returns`"* is a safe blanket rule; the weaver applies it for you.
- **Inputs are never tracked.** They were constructed before the scope opened, so a passthrough
  (`ravel()` returning its operand) or a caller-owned array is never at risk.

`out NDArray` / `out NDArray[]` parameters are yielded the same way before each successful return
(their contents are undefined after a throw, so only the success path escapes them).

<!-- Tests: NDScopeTests.OutParameter_Egress_ViaReturns; NDScopeWeaveNestingTests -->

## Which returns are woven

The weaver dispatches on the method's return shape. Every carrier of `NDArray`s NumSharp returns is
handled; a shape whose egress it cannot see is a **build error**, not a silent miss.

| return / signature | weaver action |
|---|---|
| `NDArray` / `NDArray<T>` | value routed through `scope.Returns<T>(T)` at every `ret` |
| `NDArray[]` / `NDArray<T>[]` | `scope.Returns<T>(T[])` — a tuple-of-arrays result (`nonzero`, `meshgrid`, `split`, …) |
| `ValueTuple` of 2–4 `NDArray`s (`(NDArray, NDArray)` — `modf` / `qr` / `svd` / `lstsq` / …) | each component yielded through the matching strongly-typed `Returns<T1,…>` overload (no boxing) |
| any other `ValueTuple` / `Tuple` (arity up to 8, a non-`NDArray` component, or a reference `Tuple`) | `scope.Returns(ITuple)` — yields every `NDArray` component, skips the rest |
| a result-struct carrier implementing `INDArrayCarrier` (`UniqueResult`, `MeshgridResult`, `PolyfitResult`, …) | `retVar.YieldTo(scope)` via a boxing-free `constrained.callvirt` — the struct's own method re-parents each `NDArray` it holds |
| bare `IArraySlice` / `UnmanagedStorage` (a lower-layer buffer, not wrapped in an `NDArray`) | `scope.Returns(slice/storage)` takes a **counted ARC reference** so the scope's reclamation of an intermediate `NDArray` sharing the buffer can't free it under the caller |
| `void` / scalar (`long`, `bool`, `double`, `string`, enums, …) | scope only; `out NDArray` / `out NDArray[]` params still escaped |
| `ref NDArray`(`[]`) parameter | **error NDW002** — a hidden egress the weaver can't see; scope by hand |
| an unsupported carrier (a bespoke reference type, a collection, or a struct that does NOT implement `INDArrayCarrier`) | **error NDW003** — its members would be handed back disposed; add `INDArrayCarrier`, or scope by hand |
| iterator (`yield`) / `async` | **error NDW004** — the visible body is a state-machine stub whose real egress lives in `MoveNext` |

A result struct opts in by implementing the `INDArrayCarrier` interface — one explicit method that
yields each `NDArray` it holds:

```csharp
public readonly struct MyResult : INDArrayCarrier
{
    public NDArray Values { get; }
    public NDArray Counts { get; }

    void INDArrayCarrier.YieldTo(NDScope scope)   // called by the weaver at each return
    {
        scope.Returns(Values);
        scope.Returns(Counts);
    }
}
```

The interface exists because a struct's members can live behind **private** fields (auto-property
backing fields), and the CLR grants a nested type access to its enclosing type's privates but not the
reverse — so the woven method cannot read them directly. The struct yields its own members from the
inside; the weaver just calls `YieldTo`.

<!-- Tests: NDScopeWeaveCarrierTests (tuple / ITuple / carrier / storage egress); NDScopeWeaveTests.EveryScopedMethod_WasWoven -->

## The `NDScope` API

The runtime type the weaver targets. You rarely touch it directly — the attribute does — but it is
public, so you can hand-scope (below) or manage a scope yourself.

| member | purpose |
|---|---|
| `NDScope.Open()` | opens a scope on the current thread; nests (the previous scope resumes on dispose) |
| `scope.Returns(x)` / `Returns(x[])` | yield the result (or a tuple of arrays): re-tracks it into the **parent** scope, so an enclosing scope still reclaims a dropped inner result. The one egress rule. |
| `scope.Returns((a, b[, …]))` / `Returns(ITuple)` | the `ValueTuple` / `Tuple` egress — yields each `NDArray` component |
| `scope.Returns(slice)` / `Returns(storage)` | protect a returned bare `IArraySlice` / `UnmanagedStorage` with a counted reference |
| `NDScope.Attach(nd)` | adopt an array the scope did not construct into the current scope (the hot-loop pattern — attach a received result instead of a per-item `using`) |
| `NDScope.Detach(nd)` | permanently un-track an array (for one being cached into a static / long-lived field) |
| `scope.Dispose()` | dispose every tracked, un-yielded array and reinstate the parent scope |

Two properties are load-bearing:

- **Tracked disposal is ordinary ARC release** — the buffer frees only at refcount 0. So releasing a
  base whose *view* was yielded never corrupts: the yielded view keeps the buffer alive. It is the
  same safety a hand-written `Dispose` gives, with the bookkeeping automated.
- **Scopes are `[ThreadStatic]` and nest per thread.** A scope is opened, used and disposed on **one**
  thread (NumSharp's API is synchronous — a scope must not span `await`). Arrays constructed on other
  threads (parallel kernel workers) see no scope and fall back to the finalizer; a parallel region
  that wants eager reclamation opens its own scope inside each worker body.

## Hand-scoping

When a method's egress isn't expressible as a return value plus `out` params — a `ref NDArray` flow,
or a mid-method handback that must free before a later allocation — write the scope by hand instead of
the attribute. The attribute and a hand-written scope are the same thing; the weaver **skips** a
method that already opens an `NDScope` (idempotence), so a hand-scoped method may still carry
`[NDScoped]` without double-wrapping.

```csharp
public static NDArray WeightedMean(NDArray a, NDArray w)
{
    using var scope = NDScope.Open();
    var num = np.sum(a * w);                 // two transients: the product and the sum
    var den = np.sum(w);
    return scope.Returns(num / den);         // the one egress, yielded
}
```

## How the build step works

The weave runs as an MSBuild target after each per-`TargetFramework` compile, on the intermediate
assembly, before it is copied to the output:

- **A Mono.Cecil transform** rewrites every `[NDScoped]` method into the IL above. Instructions are
  only inserted or mutated in place, so the original sequence points survive and source stepping
  still lands on the right lines — the rewritten portable **PDB** is preserved.
- **Re-signing.** IL rewriting invalidates the compile-time strong-name signature, so the weaver
  **re-signs** the assembly with the project key. Identity (public-key token, version binding) is
  unchanged — [`StrongNameTests`][gate-sign] gates it.
- **Idempotent & incremental.** A body that already opens an `NDScope` is skipped, so re-weaving is a
  no-op; the target is incremental on a per-`TargetFramework` marker, so an unchanged assembly is not
  re-woven.

A **coverage gate** ([`NDScopeWeaveTests`][gate-coverage]) reflects over the shipped assembly and
asserts every `[NDScoped]` method carries a local of type `NDScope` — a necessary consequence of
scoping. It turns a silently un-run weave (a broken target, or a `-p:SkipNDScopeWeave=true` build)
red instead of letting those methods quietly revert to the finalizer backstop.

## When to use it

- **Scope a call, not a caller loop.** The temporaries a scope holds are all alive simultaneously, and
  batch-disposing thousands of same-size buffers overflows their pool bucket. Put `[NDScoped]` on the
  *composition method*, not on a loop that calls it a million times.
- **Reach for it on transient-owning compositions** — a method built from several `np.*` calls that
  keeps one result and drops the rest. A single-kernel wrapper or a pure view / passthrough owns no
  transients, so scoping it buys nothing (and the weaver leaves such shapes as a plain scope).
- **Hot loops should still `using` the results they receive.** `[NDScoped]` reclaims a method's
  *internal* transients; the *result* it hands you is yours to dispose (or `NDScope.Attach` into your
  own scope). The two-audience contract — a callee that cleans up after itself, a caller that cleans
  up what it receives — is unchanged.

This mirrors [buffer ownership](buffering.md#ownership-who-frees-the-memory): the scope is the
deterministic-reclamation layer above the raw ARC-refcounted buffers that page describes, and it is
woven with the same [IL-emission machinery](il-generation.md) NumSharp uses for its kernels.

## Escape hatches

| flag | effect |
|---|---|
| `-p:SkipNDScopeWeave=true` | build **without** weaving — every `[NDScoped]` method runs unscoped (the finalizer backstop, the pre-weave status quo). Nothing else changes; the attribute is inert. |
| `-p:NDScopeWeaveILVerify=true` | additionally run `dotnet-ilverify` on the woven output (opt-in, so machines without the tool still build). The weave adds **zero** ILVerify findings over the unwoven baseline. |

## Troubleshooting

**A `[NDScoped]` build error.** The weaver refuses a shape whose egress it cannot see, rather than
mis-weave:

| code | meaning | fix |
|---|---|---|
| `NDW002` | a `ref NDArray`(`[]`) parameter — a hidden egress | [hand-scope](#hand-scoping) and yield it explicitly |
| `NDW003` | an unsupported carrier return (a bespoke reference type, a collection, or a struct without `INDArrayCarrier`) | implement `INDArrayCarrier` on the struct, or hand-scope |
| `NDW004` | an iterator (`yield`) or `async` method — the body is a state-machine stub | hand-scope inside the real body, or don't scope it |
| `NDW005` | the method has no body (abstract / extern) | remove the attribute |
| `NDW006` | `[NDScoped]` on a setter-only property | put the attribute on the getter accessor |
| `NDW007` | the body contains a tail-call prefix | (the C# compiler does not emit these; refuse rather than mis-weave) |

**An `[NDScoped]` method still allocates cold buffers in a loop.** Check the coverage gate — a method
that carries the attribute but no `NDScope` local was not woven (a `-p:SkipNDScopeWeave=true` build,
or the package/target missing). And remember the scope reclaims the method's *internal* transients:
the result you receive is still yours to `using` / `Dispose` / `NDScope.Attach` in the hot loop.

**Results changed after adding the package.** They should not — the weave is byte-neutral (it changes
*when* buffers are reclaimed, never the values). If you see a difference, it is a bug; the value is
covered by NumSharp's differential-fuzz gate against NumPy.

<!-- Tests: NDScopeWeaveTests.EveryScopedMethod_WasWoven; NDScopeTests; NDScopeWeaveCarrierTests; NDScopeWeaveNestingTests -->

[gate-contract]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Lifetime/NDScopeTests.cs
[gate-coverage]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Lifetime/NDScopeWeaveTests.cs
[gate-carrier]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Lifetime/NDScopeWeaveCarrierTests.cs
[gate-nesting]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Lifetime/NDScopeWeaveNestingTests.cs
[gate-sign]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Assembly/StrongNameTests.cs
