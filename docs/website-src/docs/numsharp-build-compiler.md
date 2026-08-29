# NumSharp.Build Compiler — build-time `[NDScoped]` memory reclamation

`NumSharp.Build` is the optional build-time compiler package that makes a composition method eagerly
return its transients' pooled buffers instead of waiting on the finalizer. Mark a method `[NDScoped]`
and, at build time, the weaver rewrites it to open an `NDScope` — an ambient reclamation scope that
tracks every `NDArray` the method constructs (whatever C# construct constructs it — an operator, a
cast, an initializer, a helper call) and disposes the ones it does not return the moment the method
exits, on the exception path included. **Your source keeps its 100 % original body**; the scope is
injected into the compiled IL, so the reclamation is invisible in the code and free of boilerplate.

NumSharp.Core already uses this compiler on its own build — **265+ members** carry `[NDScoped]`
(effectively the whole hot `np.*` entry surface, plus ~30 `[NDScopedCovered]` helpers riding their
callers' scopes). This package packages the same transform so you can apply it to *your* composition
methods, and
`NumSharp.Core` stays 100 % managed with nothing to install: with the package absent, an `[NDScoped]`
method simply runs unscoped and its transients fall back to the finalizer — the exact pre-weave
behaviour.

`[NDScoped]` covers **synchronous** methods and **synchronous iterators** (`yield return`). For a
method that suspends across `await` — an `async` method, an async iterator, or a non-`async` method
returning `Task`/`ValueTask` — use the companion **`[NDScopedAsync]`** attribute instead: it weaves
the compiler state machine (or the deferral egress) so temps survive awaits and are reclaimed at the
invocation's completion. Choosing the wrong attribute is a build error, not a silent miss — see
[Which returns are woven](#which-returns-are-woven).

**On this page:** [Why it exists](#why-this-package-exists) · [Quick start](#quick-start) ·
[What the weaver injects](#what-the-weaver-injects) · [Which members can carry the attribute](#which-members-can-carry-the-attribute) ·
[Which returns are woven](#which-returns-are-woven) ·
[Retained arguments (`[NDScopedExit]`)](#retained-arguments--ndscopedexit) ·
[The `NDScope` API](#the-ndscope-api) · [Hand-scoping](#hand-scoping) ·
[How the build step works](#how-the-build-step-works) · [When to use it](#when-to-use-it) ·
[Leak detection (NDW012 / NDW013)](#leak-detection--ndw012--ndw013) ·
[Escape hatches](#escape-hatches) · [Troubleshooting](#troubleshooting)

> Mono.Cecil transform, invoked after each per-`TargetFramework` compile · net8.0 / net10.0.
> Behaviour is covered by gates: [`NDScopeTests`][gate-contract] (the scope contract),
> [`NDScopeWeaveTests`][gate-coverage] (every attributed method was woven),
> [`NDScopeWeaveCarrierTests`][gate-carrier] and [`NDScopeWeaveNestingTests`][gate-nesting]
> (tuple / carrier / storage egress — recursive component dispatch included — and nesting),
> [`NDScopeAsyncTests`][gate-async] (the async/iterator seam), [`NDScopeExitTests`][gate-exit]
> (retained arguments), and [`StrongNameTests`][gate-sign] (identity survives the re-sign).

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
dotnet add package NumSharp.Build
```

Referencing the package is the whole opt-in: it wires the weave step into your build, and it is
**not a dependency** — the package ships MSBuild targets and a build tool only (no `lib/`, no
dependency entries), and `dotnet add package` installs it with `PrivateAssets="all"` automatically,
so it never appears in your own package's dependency graph. The companion **Roslyn analyzer** ships
with the `NumSharp` package itself (not this one), so it is already active the moment you reference
NumSharp: it flags a wrong or unsupported target (an `async` method under `[NDScoped]`, an
unsupported return, …) as a build error in the editor — before the weave runs — so mistakes surface
immediately, weaver installed or not. Now mark any method that owns `NDArray` transients
`[NDScoped]` and keep the body exactly as you wrote it —

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

If the `NumSharp.Build` package is **absent**, the `[NDScoped]` attribute is inert and `Normalize`
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

`out` parameters are yielded the same way before each successful return (their contents are undefined
after a throw, so only the success path escapes them) — the full carrier vocabulary: `out NDArray`(`<T>`),
`out NDArray[]`, an `out` `ValueTuple`/`Tuple` of supported shapes (boxed through `Returns(ITuple)`),
an `out` `INDArrayCarrier` struct (`YieldTo` through the byref, no box), and an `out`
`IArraySlice`/`UnmanagedStorage` (counted-ref protection). An `out` shape that carries NDArrays but
cannot be yielded — a `List<NDArray>`, a `Task<NDArray>`, a multi-dimensional array, a `T : NDArray` —
is **build error NDW015**, never a silent sweep.

<!-- Tests: NDScopeTests.OutParameter_Egress_ViaReturns; NDScopeWeaveNestingTests -->

## Which members can carry the attribute

`[NDScoped]`/`[NDScopedAsync]` target **methods and properties** (`AttributeTargets.Method |
Property`), and "method" is broader than it sounds:

| member | supported? |
|---|---|
| ordinary static/instance methods | ✔ woven |
| **user-defined operators** (`operator +`, …) and **conversion operators** (`op_Implicit` / `op_Explicit`) | ✔ woven — they are static methods; NumSharp's own binary-operator overloads are scoped exactly this way |
| property **getters** (property-level attribute resolves to the getter; NDW006 if setter-only) and accessors directly | ✔ woven |
| **local functions** | ✔ woven — the attribute lands on the compiler-generated method and the weaver collects it |
| `async` methods / async iterators / `yield return` iterators | ✔ woven through their state machines (see below) |
| **constructors / finalizers** | ✘ not an attribute target (C# rejects it) — and a ctor's egress is typically a *field*, which the weave cannot express; [hand-scope](#hand-scoping) with `_field = scope.Returns(...)` |
| lambdas | ✘ not attributable — a lambda that runs *while* the scope is open is covered by the ambient scope; one that **escapes and runs later** (a stored delegate, deferred LINQ) runs unscoped (finalizer backstop) |

Two runtime facts make the coverage syntax-agnostic: tracking hooks `NDArray` **construction** (the
single `InitializeArc` funnel), so any C# construct that creates an array under the scope — a cast
operator, an object/collection initializer, a collection expression, a `Deconstruct`, LINQ executing
inside the method — is tracked with no per-construct support; and inputs were constructed *before*
the scope opened, so they are structurally untouchable whatever syntax reads them.

## Which returns are woven

The weaver dispatches on the method's return shape. Every carrier of `NDArray`s NumSharp returns is
handled; a shape whose egress it cannot see is a **build error**, not a silent miss.

> **Two attributes, one per scoping model.** `[NDScoped]` weaves **synchronous** methods (every row
> below except the `[NDScopedAsync]`-tagged ones) **and synchronous iterators** (`yield return`).
> `[NDScopedAsync]` weaves the shapes that suspend across `await` or defer disposal to a task's
> completion: **`async` methods, async iterators, and non-`async` `Task`/`ValueTask` returns**. A
> method has exactly one scoping model — marking it with the wrong attribute is a build **error**
> (NDW009 = an async/Task shape under `[NDScoped]`; NDW010 = a plain synchronous method or
> synchronous iterator under `[NDScopedAsync]`; NDW011 = both attributes on one method), never a
> silent unwoven ship.

| return / signature | weaver action |
|---|---|
| `NDArray` / `NDArray<T>` | value routed through `scope.Returns<T>(T)` at every `ret` |
| `NDArray[]` / `NDArray<T>[]` | `scope.Returns<T>(T[])` — a tuple-of-arrays result (`nonzero`, `meshgrid`, `split`, …) |
| `ValueTuple` of 2–4 `NDArray`s (`(NDArray, NDArray)` — `modf` / `qr` / `svd` / `lstsq` / …) | each component yielded through the matching strongly-typed `Returns<T1,…>` overload (no boxing) |
| any other `ValueTuple` / `Tuple` (any arity — Rest-packed 8+ flatten — a non-`NDArray` component, or a reference `Tuple`) | `scope.Returns(ITuple)` — each component dispatches by its **runtime type**: bare `NDArray`s, `NDArray[]` elements, **nested tuples (recursively)**, `INDArrayCarrier` structs and bare buffers are all yielded; components carrying no `NDArray` are skipped. A component that carries NDArrays in a shape the dispatch cannot see through (`List<NDArray>`, `Task<NDArray>`, `T : NDArray`) is **error NDW003** |
| a result-struct carrier implementing `INDArrayCarrier` (`UniqueResult`, `MeshgridResult`, `PolyfitResult`, …) | `retVar.YieldTo(scope)` via a boxing-free `constrained.callvirt` — the struct's own method re-parents each `NDArray` it holds |
| bare `IArraySlice` / `UnmanagedStorage` (a lower-layer buffer, not wrapped in an `NDArray`) | `scope.Returns(slice/storage)` takes a **counted ARC reference** so the scope's reclamation of an intermediate `NDArray` sharing the buffer can't free it under the caller |
| `void` / scalar (`long`, `bool`, `double`, `string`, enums, …) | scope only; `out` params of any supported carrier shape still escaped |
| **`[NDScopedAsync]`** — non-`async` `Task` / `ValueTask`[`<T>`] (`T` any supported shape above, or none) | `scope.ReturnsTask` / `ReturnsValueTask` — a completed task's result is yielded immediately; an incomplete task **defers reclamation to its completion** (the in-flight callee may still hold tracked temps); an incomplete `ValueTask` is `Preserve()`d and the caller receives the multi-observable form |
| **`[NDScopedAsync]`** — `async` method (`Task`, `Task<T>`, `ValueTask`[`<T>`], `void`, custom task-like) | woven through the compiler **state machine**: one scope spans the logical invocation — suspended before every `await`'s continuation is scheduled, resumed on the thread that continues — so temps survive awaits and everything is reclaimed at `SetResult` / `SetException`, the result routed through `Returns` first |
| **`[NDScoped]`** synchronous iterator (`IEnumerable`[`<T>`] / `IEnumerator`[`<T>`]) / **`[NDScopedAsync]`** async iterator (`IAsyncEnumerable<T>`) | state-machine weave; every `yield return`ed element is routed through `Returns` (the **consumer** owns it), hoisted state is reclaimed at the end of iteration or at the enumerator's `Dispose()` (early `break` included) |
| `ref` / `in` parameter over **any** NDArray-carrying shape (`ref NDArray`(`[]`), a tuple with an `NDArray` component, a carrier struct, a `List<NDArray>`, a `T : NDArray`) | **error NDW002** — a hidden egress the weaver can't see; scope by hand |
| an unsupported carrier (a bespoke reference type, a collection, a struct that does NOT implement `INDArrayCarrier`, a **nested task** `Task<Task<…>>`, or a tuple with an unsupported ND-carrying component) — returned directly, inside a `Task<T>`, produced by an `async` method, or `yield return`ed | **error NDW003** — its members would be handed back disposed; add `INDArrayCarrier`, or scope by hand |
| an `out` parameter whose NDArray-carrying shape the out-escape cannot yield (`out List<NDArray>`, `out Task<NDArray>`, `out NDArray[,]`, `out T` with `T : NDArray`) | **error NDW015** — the caller would receive it with its arrays already swept; scope by hand |

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
inside; the weaver just calls `YieldTo`. The opt-in is **struct-only** by design (record structs
included): a *class* wrapping NDArrays is aliasable and mutable, so "sweep the temps, re-parent the
members at return" has no sound ownership story — a class carrier is NDW003 whatever it implements;
hand-scope it or restructure it as a result struct. A conversion operator on your type does not
change this: `return myWrapper;` invokes no conversion, so the weaver classifies the *declared*
return type.

<!-- Tests: NDScopeWeaveCarrierTests (tuple / ITuple / carrier / storage egress); NDScopeWeaveTests.EveryScopedMethod_WasWoven -->

<a id="retained-arguments--ndscopedexit"></a>

## Retained arguments — `[NDScopedExit]`

A scope auto-protects the **return value** and **`out` parameters** — everything else it tracked is
swept at exit. So an array handed to something that *keeps* it — a field store, a property, a
long-lived collection, a captured closure — would be disposed under the retainer. Mark the retaining
**parameter** and the weaver detaches the argument from the caller's ambient scope at the callee's
entry:

```csharp
public NDArray Weights { get; private set; }
public void Adopt([NDScopedExit] NDArray w) => Weights = w;   // w survives the caller's scope
```

- Works **with or without** a method-level scope attribute (the callee runs inside the caller's
  ambient scope, so the injected `NDScope.Detach(param)` reaches it with zero call-site plumbing),
  and is a no-op when no scope is open.
- Covers `NDArray`(`<T>`), `NDArray[]`, and `ValueTuple`/`Tuple` of NDArrays — **nested tuples and
  `NDArray[]` components detach recursively**.
- A **property setter's `value` parameter** counts — which also covers the *object-initializer*
  spelling (`new Wrapper { Prop = temp }` compiles to a setter call), and a **constructor parameter**
  covers `new Wrapper(temp)`.
- A raw **public-field** store (`obj.field = temp`) has no parameter to annotate — route it through a
  setter or call `NDScope.Detach` by hand.
- An unsupported parameter type — `ref`/`out`/`in`, a scalar, a bare buffer, an `INDArrayCarrier`
  struct, or a tuple with an ND-carrying component `Detach` cannot see through — is **error NDW014**.

Detachment trades eager reclamation for survival: the retainer now owns the array (dispose it there,
or let the finalizer backstop reclaim it).

<!-- Tests: NDScopeExitTests (Detach overloads incl. recursive tuples; callee-detaches semantics) -->

## The `NDScope` API

The runtime type the weaver targets. You rarely touch it directly — the attribute does — but it is
public, so you can hand-scope (below) or manage a scope yourself.

| member | purpose |
|---|---|
| `NDScope.Open()` | opens a scope on the current thread; nests (the previous scope resumes on dispose) |
| `scope.Returns(x)` / `Returns(x[])` | yield the result (or a tuple of arrays): re-tracks it into the **parent** scope, so an enclosing scope still reclaims a dropped inner result. The one egress rule. |
| `scope.Returns((a, b[, …]))` / `Returns(ITuple)` | the `ValueTuple` / `Tuple` egress — yields every `NDArray` the tuple carries (components dispatch by runtime type: bare arrays, `NDArray[]` elements, nested tuples recursively, carrier structs, bare buffers) |
| `scope.Returns(slice)` / `Returns(storage)` | protect a returned bare `IArraySlice` / `UnmanagedStorage` with a counted reference |
| `NDScope.Attach(nd)` | adopt an array the scope did not construct into the current scope (the hot-loop pattern — attach a received result instead of a per-item `using`) |
| `NDScope.Detach(nd)` / `Detach(nd[])` / `Detach(ITuple)` | permanently un-track an array (for one being cached into a static / long-lived field) — the array/tuple overloads detach every element, nested tuples and `NDArray[]` components recursively; also the egress `[NDScopedExit]` weaves |
| `scope.Dispose()` | dispose every tracked, un-yielded array and reinstate the parent scope |

Two properties are load-bearing:

- **Tracked disposal is ordinary ARC release** — the buffer frees only at refcount 0. So releasing a
  base whose *view* was yielded never corrupts: the yielded view keeps the buffer alive. It is the
  same safety a hand-written `Dispose` gives, with the bookkeeping automated.
- **Scopes are `[ThreadStatic]` and nest per thread.** A scope is opened, used and disposed on **one**
  thread — a **hand-written** scope must not span `await`; only the weaver's state-machine seam may
  carry a scope across suspensions (it uninstalls the scope before each continuation is scheduled
  and re-installs it on the resuming thread). Arrays constructed on other
  threads (parallel kernel workers) see no scope and fall back to the finalizer; a parallel region
  that wants eager reclamation opens its own scope inside each worker body.

## Hand-scoping

When a method's egress isn't expressible as a return value plus `out` params — a `ref`/`in` flow over
an ND-carrying shape (NDW002), an unsupported carrier like a class wrapper or `List<NDArray>`
(NDW003/NDW015), a constructor whose egress is a field, or a mid-method handback that must free
before a later allocation — write the scope by hand instead of the attribute. The attribute and a
hand-written scope are the same thing; the weaver **skips** a method that already opens an `NDScope`
(idempotence), so a hand-scoped method may still carry `[NDScoped]` without double-wrapping.

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
- **Cross-assembly resolution.** In your project `NDScope` and the attribute live in the
  *referenced* NumSharp assembly, so the target hands the weaver the compiler's own reference list
  (a response file of `@(ReferencePathWithRefAssemblies)`); the weaver resolves the types through
  it and emits cross-assembly member references. NumSharp.Core's self-weave needs none of this —
  there the types are in-module.
- **Re-signing.** IL rewriting invalidates a compile-time strong-name signature, so on a
  strong-named project the weaver **re-signs** the assembly with the project's own key; identity
  (public-key token, version binding) is unchanged — [`StrongNameTests`][gate-sign] gates it for
  NumSharp itself. An unsigned project is simply written unsigned; delay- and public-signed
  projects are left as-is (they carry no valid signature to restore).
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

## Leak detection — NDW012 &amp; NDW013

Two diagnostics catch the two ways an `NDArray` transient can go unreclaimed at build time, so a leak
surfaces in the editor and on every build instead of only as a runtime finalizer cost.

<a id="ndw012"></a>

### NDW012 — an NDArray is created but never disposed or scoped

The `NumSharp` package ships a Roslyn analyzer that **warns** when a method creates an
`NDArray` (or any carrier of one — `NDArray[]`, a `ValueTuple`/`Tuple` of NDArrays, an
`INDArrayCarrier` result struct) and then never returns it, assigns it to an `out`/`ref` parameter,
stores it, disposes it, or yields it through an `NDScope` — so its pooled buffer is left to the
finalizer. Referencing NumSharp is the whole opt-in (no weaver install needed); NumSharp.Core runs
the same analyzer on its own build.

```csharp
public NDArray Bad(NDArray a, NDArray b)
{
    var t = a + b;              // ⚠ NDW012: 't' is created but only read, never disposed/returned
    np.add(a, b);               // ⚠ NDW012: the result is dropped on the floor
    return np.sum(a * b);       // ⚠ NDW012: the 'a * b' temporary is handed to np.sum and never reclaimed
}
```

**The fixes the message points at** — any one of them clears the warning:

- mark the method **`[NDScoped]`** (or **`[NDScopedAsync]`**) — the weaver reclaims every transient it
  constructs, and a passthrough/aliased input is a provable no-op, so this is the safe blanket fix;
- open an **`NDScope`** by hand and yield the result through `scope.Returns(...)`;
- **dispose** the value with a `using` declaration or `.Dispose()`;
- hand it to an `out`/`ref` parameter, a field, or another API (a legitimate egress).

A method that is already `[NDScoped]`/`[NDScopedAsync]`, or that opens an `NDScope` itself, is
**exempt** — nothing it constructs leaks. Returning the value, storing it, or passing it to a
non-NumSharp API are all egress and never flagged.

**Two limits, by design.** The analysis is per-method: a private helper whose temporaries are actually
reclaimed by a **caller's ambient `[NDScoped]` scope** (the "scope the boundary, not the helpers"
pattern) is still flagged in isolation — scope the boundary (which covers the helper), or mark the
helper **`[NDScopedCovered]`** (an analyzer-only, runtime-inert assertion that it always runs under a
caller's open scope — the sanctioned fix). And it deliberately does **not** flag a temporary that is only ever read through
a member call (`var t = a + b; return t.sum();`): so much of NumSharp is fluent reinterpret/view chains
(`x.MakeGeneric<T>()`, `x.reshape(...)`, `x[...]`) whose result shares the receiver's buffer that
telling a view apart from a fresh result syntactically is impossible, and following the receiver is the
choice that keeps those chains from flooding with false positives. Tune it per project in
`.editorconfig` (`dotnet_diagnostic.NDW012.severity = none|suggestion|warning|error`); on NumSharp.Core
turn it off entirely with `-p:EnableNDArrayLeakAnalyzer=false`.

<a id="ndw013"></a>

### NDW013 — [NDScoped] used, but the weaver is not installed

`[NDScoped]`/`[NDScopedAsync]` are **inert without the `NumSharp.Build` package**: the method keeps its
original body, no scope is injected, and every transient it drops is left to the finalizer — a silent
leak. NDW013 catches exactly that. It ships in **NumSharp itself** (a `build/NumSharp.targets` warning),
not in the weaver package, precisely so it can fire when the weaver is **absent** — the attributes live
in NumSharp, so the guard is present whenever they could be used:

```
warning NDW013: MyProject uses [NDScoped]/[NDScopedAsync] but the NumSharp.Build package is not
installed (or weaving is disabled), so the attributes are INERT ... Install the weaver with
'dotnet add package NumSharp.Build' ...
```

It stays silent the moment the weaver is installed and weaving (the weaver's targets set
`$(NumSharpBuildActive)=true`; under `-p:SkipNDScopeWeave=true` the attributes really are inert, so the
warning correctly fires). Opt out with `-p:NumSharpDisableWeaverMissingWarning=true`, by adding `NDW013`
to `$(MSBuildWarningsAsMessages)`, or — the intended fix — by installing the weaver.

## Escape hatches

| flag | effect |
|---|---|
| `-p:SkipNDScopeWeave=true` | build **without** weaving — every `[NDScoped]` method runs unscoped (the finalizer backstop, the pre-weave status quo). Nothing else changes; the attribute is inert. |
| `-p:NDScopeWeaveILVerify=true` | additionally run `dotnet-ilverify` on the woven output (opt-in, so machines without the tool still build). The weave adds **zero** ILVerify findings over the unwoven baseline. |

## Troubleshooting

**A `[NDScoped]` / `[NDScopedAsync]` build error.** A shape whose egress cannot be seen is refused
rather than mis-woven. Two layers report it: the **NumSharp package** ships a **Roslyn analyzer** that
flags the SOURCE-detectable mistakes at **compile time** — in the editor, with a red squiggle, and as
a build error that runs before (and preempts) the weave — and the **IL weaver** (this package) reports
the rest post-compile.
The analyzer covers **NDW002, NDW003, NDW005, NDW006, NDW009, NDW010, NDW011, NDW015** (the
`[NDScoped]`-target gate) plus **NDW012** (the [leak warning](#ndw012), a separate always-on analyzer);
the weaver covers the IL-only **NDW001, NDW004, NDW007, NDW008, NDW014** (a resolution failure, an
unrecognized state machine, a tail-call, a NumSharp too old for the async seam, or a bad
`[NDScopedExit]` parameter). **NDW013** also ships in NumSharp itself
so it can warn when `[NDScoped]` is used [without the weaver installed](#ndw013).
Same code, same fix, whichever layer fires.

> **A `PackageReference` consumer gets the analyzer automatically** — NuGet applies it from the
> **NumSharp** package's `analyzers/dotnet/cs/`, so referencing NumSharp alone is enough (no weaver
> install required). A **source-mode consumer** — one that references NumSharp.Core by
> `ProjectReference` and `<Import>`s `build/NumSharp.Build.targets` directly, so no package delivers
> the analyzer — turns it on the same way it points the tool at a built weaver: either reference the
> analyzer project with
> `<ProjectReference "…/NumSharp.Build.Analyzer.csproj" OutputItemType="Analyzer" ReferenceOutputAssembly="false" />`,
> or set `<NumSharpBuildAnalyzerDll>…/NumSharp.Build.Analyzer.dll</NumSharpBuildAnalyzerDll>` at a
> built analyzer DLL (the parallel to `<NumSharpBuildToolDll>`). Without either, a source-mode build
> still fails on a bad target — via the **weaver** post-compile — just not in the editor.

| code | meaning | fix |
|---|---|---|
| `NDW001` | the project has `[NDScoped]`/`[NDScopedAsync]` methods but `NumSharp.NDScope` could not be resolved from the assembly or its references | reference the `NumSharp` package (the attributes and the scope live there); if you invoke the weaver by hand, pass `--refs` with the compile's reference list |
| `NDW002` | a `ref`/`in` parameter over any NDArray-carrying shape (`ref NDArray`(`[]`), a tuple with an `NDArray` component, a carrier struct, a `List<NDArray>`) — a hidden egress | [hand-scope](#hand-scoping) and yield it explicitly |
| `NDW003` | an unsupported carrier return (a bespoke reference type, a collection, a struct without `INDArrayCarrier`, a nested `Task<Task<…>>`, or a tuple with an unsupported ND-carrying component) | implement `INDArrayCarrier` on the struct, or hand-scope |
| `NDW004` | an async/iterator state machine the weaver does not recognize (not compiled by Roslyn/C#: no `MoveNext`, no `<>t__builder` / `<>2__current`) | compile the method with the C# compiler, or don't scope it (real C# async/iterator methods weave fine) |
| `NDW005` | the method has no body (abstract / extern) | remove the attribute |
| `NDW006` | `[NDScoped]`/`[NDScopedAsync]` on a setter-only property | put the attribute on the getter accessor |
| `NDW007` | the body contains a tail-call prefix | (the C# compiler does not emit these; refuse rather than mis-weave) |
| `NDW008` | an async/iterator/Task-shaped target, but the referenced NumSharp predates the async scope seam | update the `NumSharp` package |
| `NDW009` | an `async` method, async iterator, or `Task`/`ValueTask`-returning method marked `[NDScoped]` | mark it `[NDScopedAsync]` (that attribute owns the shapes that suspend across `await`) |
| `NDW010` | a plain synchronous method or a **synchronous** iterator marked `[NDScopedAsync]` | mark it `[NDScoped]` (synchronous bodies and synchronous iterators are its job) |
| `NDW011` | a method carries BOTH `[NDScoped]` and `[NDScopedAsync]` | keep only the one that matches the method's scoping model |
| [`NDW012`](#ndw012) | an NDArray is created but never returned, out/ref'd, stored, disposed, or yielded to an `NDScope` — a transient left to the finalizer | mark the method `[NDScoped]`/`[NDScopedAsync]`, dispose it (`using`/`.Dispose()`), yield it via `scope.Returns(...)`, or hand it to an egress; tune via `.editorconfig` (`dotnet_diagnostic.NDW012.severity`) |
| [`NDW013`](#ndw013) | `[NDScoped]`/`[NDScopedAsync]` used but the `NumSharp.Build` package is not installed (or `-p:SkipNDScopeWeave=true`) — the attributes are inert and the temporaries leak | install `NumSharp.Build`, or remove the attributes and dispose by hand; suppress with `-p:NumSharpDisableWeaverMissingWarning=true` |
| `NDW014` | `[NDScopedExit]` on an unsupported parameter type (a `ref`/`out`/`in`, a scalar, a bare buffer, an `INDArrayCarrier` struct, or a tuple with an ND-carrying component `Detach` cannot see through) | hand-detach with `NDScope.Detach` |
| `NDW015` | an `out` parameter whose NDArray-carrying shape the out-escape cannot yield (`out List<NDArray>`, `out Task<NDArray>`, `out NDArray[,]`, `out T` with `T : NDArray`) | [hand-scope](#hand-scoping) and yield the final value explicitly |

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
[gate-async]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Lifetime/NDScopeAsyncTests.cs
[gate-exit]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Lifetime/NDScopeExitTests.cs
[gate-sign]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests/Assembly/StrongNameTests.cs
