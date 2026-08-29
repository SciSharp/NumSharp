# NumSharp Disposal Guidelines

*How to dispose transient `NDArray` intermediates in the backend — correctly, and when it's worth
it. Every rule here is backed by a runnable experiment (see Appendix B).*

`NDArray` is `IDisposable`. Disposing a **transient** result returns its unmanaged buffer to the
`SizeBucketedBufferPool` **synchronously**, instead of waiting on the finalizer — which at small N /
in hot loops is a 1.6×–3× speedup. But disposing the **wrong** object is a *silent memory-corruption
bug* that value-only tests will not catch. This document is the standard for getting the win without
the bug.

---

## Contents

1. [TL;DR](#1-tldr)
1½. [The standard instrument — NDScope](#the-standard-instrument--ndscope)
2. [Why this exists — the problem and the measured stakes](#2-why-this-exists)
3. [The mechanism — finalizer lag vs. synchronous Return](#3-the-mechanism)
4. [Correctness invariants — how disposal actually works (ARC, verified)](#4-correctness-invariants)
5. [Decision Flow 1 — *should* I dispose this intermediate? (R1/R2/R3)](#5-decision-flow-1)
6. [Decision Flow 2 — *which form*: `using` vs `Dispose()` vs `try/finally`](#6-decision-flow-2)
7. [The dilemmas — the judgment calls, resolved](#7-the-dilemmas)
8. [The bug catalog — what a wrong migration looks like](#8-the-bug-catalog)
9. [Worked examples — the migrated sites](#9-worked-examples)
10. [Two audiences — backend authors vs. callers](#10-two-audiences)
11. [Decision Flow 3 — is this site worth migrating?](#11-decision-flow-3)
12. [Decision Flow 4 — the verification gate (mandatory)](#12-decision-flow-4)
13. [Anti-patterns / do-not](#13-anti-patterns)
14. [Appendix A — ARC internals reference](#appendix-a)
15. [Appendix B — the experiments behind every claim](#appendix-b)

---

## 1. TL;DR

- **The standard instrument is `NDScope`** (next section), and the standard way to APPLY it is
  the `[NDScoped]` attribute: the build-time IL weaver injects the scope (open, try/finally
  dispose, every egress yielded) so the method keeps its 100% original body. Every NDArray
  constructed under the scope — however deep the helper-call tree — is reclaimed at exit; inputs
  were constructed before the scope opened and are structurally untouchable. Hand-write the
  scope only for shapes the weaver rejects (a struct without `INDArrayCarrier`, `ref` flows,
  an ND-carrying collection/task egress).
- **The rules the scope automates** (and which still govern any hand-written dispose):
  (**R1**) never dispose what you return, (**R2**) never dispose an input you were given,
  (**R3**) dispose after the last use (through any view too).
- **For hand-written disposal** (engine internals, mid-method handback): `using` when the
  intermediate dies at method exit; a **direct `Dispose()`** when it dies mid-method or
  per-loop-iteration; `try/finally` only when a throw between create and dispose is *both
  plausible and costly*.
- **Exception-safety here protects only the optimization, not correctness.** A skipped dispose
  just falls back to the finalizer (the status quo) — no leak, no double-free, no corruption.
- **Views are safe by refcount** — disposing a base that still has a live view does **not** free the
  buffer. This is what makes both the scope and hand disposal safe: releasing a wrapper whose view
  escaped never corrupts.
- **Every migration must be verified byte-neutral** (§12). A wrong dispose can return the correct
  value while corrupting the caller's input — values won't catch it.
- **Never automate LIVENESS INFERENCE.** `NDScope` automates *registration* only — deadness is
  still declared by the programmer (everything not `Returns`ed is dead), and disposal is ordinary
  refcounted `Dispose`. The reverted eager-reclaim ring guessed liveness from refcounts and
  corrupted memory; that class of automation stays banned.

---

## The standard instrument — `NDScope`

`NDScope` (`src/NumSharp.Core/Backends/NDScope.cs`) is an ambient reclamation scope — the
TorchSharp `DisposeScope` pattern. Every `NDArray` constructed on the current thread while a scope
is open is tracked by it (one hook in `NDArray.InitializeArc`, the single funnel every ctor
passes); disposing the scope disposes everything that was not yielded via `Returns`.

```csharp
public static NDArray<bool> logical_and(NDArray x1, NDArray x2)
{
    using var scope = NDScope.Open();
    var b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);   // original body, untouched
    var b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
    return scope.Returns((b1 & b2).MakeGeneric<bool>());
}
```

**Why the rules become structural.** Inputs are constructed BEFORE the scope opens → never
tracked → R2 needs no `ReferenceEquals` guard, and a passthrough (`ravel()`/`atleast_2d()`
returning its operand, a caller's `@out`) is automatically protected. The return value is the one
egress and is yielded (R1). `Returns` on an untracked array is a **provable no-op** (the
back-pointer says "not mine"), so *"wrap every egress in `Returns`"* is a safe blanket rule.

**The API.**
- `NDScope.Open()` — opens (nests; per-thread pooled).
- `scope.Returns(x)` / `scope.Returns(x[])` — yield the result (or a tuple of views); re-tracks
  into the PARENT scope, so an enclosing scope still reclaims a dropped inner result. Also the
  egress for `out`-parameters: `result = scope.Returns(temp);`.
- `scope.Returns((a, b))` / `Returns((a, b, c))` / `Returns((a, b, c, d))` — the ValueTuple egress
  (yields each component). A result-struct carrier instead implements `INDArrayCarrier.YieldTo(scope)`
  (calling `scope.Returns` on each member). Both are what the weaver targets for a tuple / struct
  return — see "The decision table" below.
- `NDScope.Detach(x)` — permanently un-track (for an array being cached into a static/long-lived
  field from inside a scoped call). Overloads `Detach(x[])` / `Detach(ITuple)` un-track a whole
  array/tuple of results, and are the runtime the **`[NDScopedExit]`** parameter attribute weaves —
  see "Retained arguments" below.
- `NDScope.Attach(x)` — adopt an untracked array into the CURRENT scope (the caller-side
  hot-loop pattern: attach a received result instead of a per-result `using`), or move one
  between scopes; no-op with no open scope.

**Threading.** The scope stack is `[ThreadStatic]`: open/use/dispose on ONE thread (debug-asserted
on `Dispose` AND on the mutating `Returns`/`Attach`/`Detach`, so a cross-thread scope touch trips in
Debug rather than racing a `List`; never span an `await`). Worker threads see no scope and fall back
to the finalizer backstop — safe; a parallel region wanting eager reclaim opens a scope inside each
worker body. A result yielded on a thread with no parent scope is fully detached and can be handed
to any other thread.

**Disposal ordering.** The weaver's try/finally and every `using` dispose LIFO — the fast path, where
the disposed scope IS the innermost and pops straight to its parent. `Open()` returns a raw
`IDisposable`, though, so a hand-managed scope CAN be disposed out of order; `Dispose` tolerates it by
splicing the scope out of the middle of the thread's chain, so `t_current` is never left pointing at —
or through — a disposed scope. (Correctness of the reclamation is unaffected either way; this only
keeps the ambient stack coherent under misuse.)

**Granularity.** Scope a CALL, not a caller loop: a scope holds its temps alive until exit, and
batch-disposing thousands of same-size buffers overflows their pool bucket (the excess is freed,
not pooled — measured). Hot loops still `using` the results they receive (§10's two-audience
contract is unchanged). Where a mid-method handback genuinely matters (free a big transient before
the next allocation), nest an inner block scope or keep a hand-written `Dispose()`.

**Gate.** `test/NumSharp.Tests/Lifetime/NDScopeTests.cs` (contracts: tracking, yield/no-op/
re-parenting, out-params, view-over-base ARC, exception paths, Detach, Attach, pooling,
out-of-order dispose, zero-strand counters) and `NDScopeStressTests.cs` (8-thread mixed-op load,
nested scopes under concurrency, cross-thread hand-off, forced-GC antagonist), on top of the
`BufferReleaseSweepTests` slope sweep and the FuzzMatrix byte-exactness gate.

---

## The weaver — `[NDScoped]`

The scope pattern is INJECTED AT BUILD TIME, so source files keep their 100% original bodies.
A method (or property accessor) marked `[NDScoped]` (`src/NumSharp.Core/Backends/
NDScopedAttribute.cs`, public — the attribute is consumer-facing) is rewritten post-compile by
`tools/NumSharp.Build` — a Mono.Cecil console tool invoked by the `NDScopeWeave` target in
`NumSharp.Core.csproj` after each per-TFM `CoreCompile` — into exactly the IL the hand-written
pattern produces. The SAME transform ships to consumer projects as the **`NumSharp.Build`
NuGet package** (tools + MSBuild targets only, `PrivateAssets="all"` on install — a weaver on
the project it is installed on, never a runtime dependency; gate:
`tools/verify_build_package.sh`, docs: `docs/website-src/docs/ndscoped.md`); there the weaver
resolves `NDScope`/`INDArrayCarrier` out of the REFERENCED NumSharp assembly via the compile's
reference list:

```csharp
[NDScoped]                                          // what you write: the ORIGINAL body
public static NDArray<bool> logical_and(NDArray x1, NDArray x2)
{
    var b1 = x1.typecode == NPTypeCode.Boolean ? x1 : (x1 != 0);
    var b2 = x2.typecode == NPTypeCode.Boolean ? x2 : (x2 != 0);
    return (b1 & b2).MakeGeneric<bool>();
}
// what the built assembly carries: scope = NDScope.Open() prologue; the whole body as the try
// of a try/finally whose finally is scope.Dispose(); every ret rewritten to route the value
// through scope.Returns(value); each `out NDArray` parameter's FINAL value yielded before each
// successful return (on a throw the finally reclaims; out contents are undefined after a throw).
```

**Why the rewrite is trivially safe:** the CLR forbids `ret` inside a protected region, so once
the body is wrapped in try/finally every original return is forced through a rewritable seam —
bodies that already contain try/using compile their inner returns to leave-chains ending at
outermost `ret`s, and only those real `ret` instructions are touched (mutated IN PLACE, so
branches and nested-handler boundaries that referenced them stay valid).

**The decision table** (enforced at weave time — a violation is a BUILD ERROR):

| return / signature | weaver action |
|---|---|
| `NDArray` / `NDArray<T>` | value routed through `Returns<T>(T)` at every ret |
| `NDArray[]` / `NDArray<T>[]` | `Returns<T>(T[])` (tuple results — nonzero etc.) |
| `ValueTuple` of 2..4 NDArrays (`(NDArray, NDArray)` — `modf`/`polydiv`/`qr`/`eig`/`svd`/`lstsq`/…) | each component yielded through the matching `Returns<T1,…>` tuple overload (no box) |
| any OTHER `ValueTuple`/`Tuple` (any arity — Rest-packed 8+ flatten — a non-NDArray component, or a reference `Tuple`) | `Returns(ITuple)` — every component dispatches by its RUNTIME type: bare NDArrays, `NDArray[]` elements, NESTED tuples (recursively), carrier structs and bare buffers are all yielded; components carrying no NDArray are skipped (boxed on the boundary; negligible off the hot path). A component carrying NDArrays in a shape the dispatch cannot see through (`List<NDArray>`, `Task<NDArray>`, `T : NDArray`) is **error NDW003** |
| result-struct carrier implementing `INDArrayCarrier` (`UniqueResult`, `MeshgridResult`, `PolyfitResult`, …) | `retVar.YieldTo(scope)` via boxing-free `constrained.callvirt` — the struct's own method re-parents each NDArray it holds (so private fields are reached from INSIDE the struct) |
| bare `IArraySlice` / `UnmanagedStorage` return (a lower-layer buffer NOT wrapped in an NDArray) | `Returns(slice)`/`Returns(storage)` takes a **counted ARC reference** (`TryAddRef`) so the scope's deterministic `Release` of an intermediate NDArray sharing the buffer can't free it under the caller (a bare buffer is otherwise an UNCOUNTED alias); the ref is abandoned, so the block's finalizer reclaims it on unreachability |
| `void` / scalar (`long`, `bool`, `double`, string, enums, …) | scope only; `out` params of any supported carrier shape still escaped |
| `out` param of a supported carrier shape — `NDArray`(`<T>`), `NDArray[]`, a `ValueTuple`/`Tuple` of supported shapes (nested included), an `INDArrayCarrier` struct, a bare `IArraySlice`/`UnmanagedStorage` | final-value escape before each successful return (tuples box through `Returns(ITuple)`; a carrier's `YieldTo` runs `constrained.` through the byref, no box; buffers take the counted ref) |
| `out` param of an ND-carrying shape the escape cannot yield (`out List<NDArray>`, `out Task<NDArray>`, `out NDArray[,]`, `out T` with `T : NDArray`) | **error NDW015** — the caller would receive it with its arrays already swept; scope by hand |
| **`[NDScopedExit]`** on a by-value `NDArray`/`NDArray[]`/tuple-of-NDArrays PARAMETER — an argument the callee RETAINS (nested tuples and `NDArray[]` components detach recursively) | `NDScope.Detach(param)` injected at the method's ENTRY (the compiler stub, for async/iterator), so the CALLER's ambient scope will not dispose the argument. Independent of the method's own scope model (works with or without `[NDScoped]`). Covers property setters (`[param: NDScopedExit]` on `value`) and any method/ctor parameter you own; a raw public-field store (`obj.field = a`) has no parameter to annotate — use a property setter or a hand `NDScope.Detach`. An unsupported parameter type (a `ref`/`out`/`in`, a scalar, a bare buffer, an `INDArrayCarrier` struct, or a tuple with an ND-carrying component `Detach` cannot see through) is **error NDW014** |
| **`[NDScopedAsync]`** NON-async `Task`/`ValueTask`[`<T>`] (T = any supported row above, or none) | `ReturnsTask`/`ReturnsValueTask` at every ret + `CloseUnlessDeferred` in the finally: a COMPLETED task's result is yielded immediately; an INCOMPLETE one **defers the scope's disposal to the task's completion** (the in-flight callee may still be using tracked temps handed to it) and yields the result there; an incomplete `ValueTask` is `Preserve()`d first (single-consumption — the caller receives the multi-observable form) |
| **`[NDScopedAsync]`** `async` method (`Task`, `Task<T>`, `ValueTask`[`<T>`], `void`, custom task-like) | woven through the STATE MACHINE — see "Async & iterators" below |
| **`[NDScoped]`** synchronous iterator (`IEnumerable`[`<T>`]/`IEnumerator`[`<T>`]) / **`[NDScopedAsync]`** async iterator (`IAsyncEnumerable<T>`) | woven through the state machine; `yield return`ed elements routed through `Returns` (the consumer owns them) |
| `ref`/`in` param over ANY NDArray-carrying shape (`ref NDArray`(`[]`), a tuple with an NDArray component, a carrier struct, a `List<NDArray>`, a `T : NDArray`) | **error NDW002** — hidden egress; scope by hand |
| unsupported carrier (a bespoke reference type, a collection, a result struct WITHOUT `INDArrayCarrier`, a nested `Task<Task<…>>`, or a tuple with an ND-carrying component the `Returns(ITuple)` dispatch cannot see through) — returned directly, inside a `Task<T>`, as an async result, or `yield return`ed | **error NDW003** — the weaver cannot see every NDArray, so members would be handed back disposed; add `INDArrayCarrier` to the struct, or scope by hand |
| a state machine not compiled by Roslyn/C# (missing `MoveNext`/`<>t__builder`/`<>2__current`) | **error NDW004** — refused loudly, never mis-woven |
| async/iterator/Task-shaped target against a NumSharp predating the seam | **error NDW008** — update the NumSharp package |
| an `async`/iterator/`Task`-shaped method marked `[NDScoped]`, or a plain synchronous method / synchronous iterator marked `[NDScopedAsync]`, or a method carrying BOTH | **error NDW009 / NDW010 / NDW011** — the WRONG attribute (or both); each error names the correct one |

**Two attributes.** `[NDScoped]` weaves synchronous methods and synchronous iterators;
`[NDScopedAsync]` weaves the shapes that suspend across `await` or defer to a task's completion —
`async` methods, async iterators, and non-`async` `Task`/`ValueTask` returns. Both drive the SAME
state-machine/deferral machinery below (a synchronous iterator uses the invocation-scope seam exactly
as an async method does); the split is which attribute the method carries, and choosing the wrong one
is a loud build error (NDW009/NDW010/NDW011), never a silent unwoven ship.

**Two layers report a bad target.** The `NumSharp` package ships the Roslyn analyzer
(`tools/NumSharp.Build.Analyzer`, staged into NumSharp's `analyzers/dotnet/cs/` by
`NumSharp.Core.csproj → _NumSharpCorePackAnalyzer` — so it is present the moment the attributes
are, weaver installed or not); the `NumSharp.Build` package adds the IL weaver and ships no
analyzers/ of its own. The analyzer reports every SOURCE-detectable rejection at COMPILE time — in
the editor and before the weave — and preempts the weaver: NDW002 (ref egress), NDW003 (unsupported
carrier), NDW005 (no body), NDW006 (setter-only property), NDW009/NDW010/NDW011 (wrong attribute /
both). The IL-only rejections stay with the weaver post-compile: NDW001 (NDScope unresolved), NDW004
(an unrecognized state-machine shape), NDW007 (a tail-call), NDW008 (a NumSharp too old for the
async seam). The analyzer mirrors the weaver's `Classify` exactly, so the two never disagree; where
they cannot cheaply agree — a real iterator vs a method merely returning a bare `IEnumerable` — the
analyzer defers to the weaver rather than risk a false positive. Gate:
`tools/verify_build_package.sh` step 11 (11a analyzer, 11b weaver) + step 18 (analyzer via the
NumSharp package alone).

**Async & iterators — the state-machine weave.** An async or iterator method's visible body is a
stub; the real code — and every egress — lives in the compiler state machine's `MoveNext`, which
runs once per synchronous SEGMENT, each possibly on a different thread. A `[ThreadStatic]` scope
cannot simply span that, and a scope-per-segment is UNSOUND (it would reclaim temps an in-flight
awaited callee still uses — `await Bar(t)` holds `t` across the suspension). So the weave gives the
state machine **one scope per logical invocation**, held in a weaver-added field (`<>ndscope`):

- `MoveNext` prologue: `scope = NDScope.OpenOrResume(ref this.<>ndscope)` — opens on the first
  segment, RE-INSTALLS the suspended scope (re-stamping its owning thread) on every resumption.
- Immediately BEFORE each `Await[Unsafe]OnCompleted`: `suspended = true; NDScope.Suspend(scope)` —
  the scope must be off this thread's chain BEFORE the builder has the continuation, because after
  that call the continuation may already be resuming on another thread (suspending in the finally
  would race it; `suspended` is a LOCAL for the same reason — the finally must not read scope state
  a concurrent resumption mutates).
- Immediately BEFORE the builder's `SetResult`/`SetException`: the result (if any) is yielded
  through the ordinary `Returns`/`YieldTo` egress, then `NDScope.DisposeSlot(ref slot)` — disposal
  happens BEFORE the completion signal because the signal can run the CALLER's continuation inline
  on this very thread, and that code must not execute under (and track into) a dying scope. The
  finally (`if (!suspended) DisposeSlot`) remains as the backstop.
- Iterators: suspension IS `return true`, and the consumer only regains control after `MoveNext`
  returns, so the finally decides single-threadedly (`retVar ? Suspend : DisposeSlot`); every
  `stfld <>2__current` is routed through `Returns` (the consumer owns yielded elements); the
  enumerator's `Dispose()` gets `DisposeSlot` before each ret — `foreach { break; }` reclaims
  deterministically. Async iterators additionally run `ExitIterator(ref slot, hasMore)` BEFORE
  `promise.SetResult(hasMore)` (the consumer can re-enter `MoveNext` the instant that signal lands).

Consequences: hoisted locals and parameters need NO special handling (parameters were constructed
before the scope opened and are never tracked; hoisted locals stay correctly tracked because the
scope spans the whole invocation); temps live until the METHOD completes, not the segment — a
long-lived async loop that wants per-iteration reclamation opens an inner `using var s =
NDScope.Open()` WITHIN a segment (a hand scope must still never span an `await`); results/yielded
elements re-parent into whatever scope is ambient at the completion/yield point (the sync-nesting
chain works when completion is inline; a genuinely cross-thread completion leaves the result
untracked — finalizer backstop if dropped). An abandoned-without-Dispose enumerator or a
never-completing async invocation falls to the ordinary GC/finalizer backstop. Verified end-to-end
(struct AND class state machines, generic ones included, 200-way concurrent stress, ILVerify clean)
by the async consumer battery in `tools/verify_build_package.sh`; the seam's runtime contracts are
pinned by `NDScopeAsyncTests`.

**Lower-layer buffer returns — scope of support.** The `IArraySlice`/`UnmanagedStorage` row above
covers a method that RETURNS a bare buffer (the return is protected). What is deliberately NOT done is
auto-TRACKING intermediate bare buffers for eager reclamation: the scope's reclamation unit is the
NDArray, which owns the single counted ARC reference on its buffer, and a bare buffer takes no counted
reference of its own (an "uncounted alias" — see the `~NDArray` ABANDON contract). Hooking bare-buffer
CONSTRUCTION to track them would double-count against the NDArray's reference, so intermediate bare
buffers stay finalizer-managed (their block's `~Disposer` reclaims on unreachability) — which is what
they already are. In practice compositions own NDArrays, not bare buffers, so their transients are
already reclaimed transitively when the scope releases the owning NDArrays.

**Idempotence & incrementality.** A body that already opens an `NDScope` is skipped
("already-scoped"), so re-weaving the same assembly is a no-op; the MSBuild target is
additionally incremental via a per-TFM `NDScopeWeave.marker` in obj. Note the return-shape check
runs BEFORE the already-scoped check: `[NDScoped]` on a hand-scoped method whose return is an
UNSUPPORTED carrier (NDW003) is an error, not a skip — the attribute would be a lie there — while a
hand-scoped method with a supported return (incl. a tuple or an `INDArrayCarrier` struct) is simply
skipped.

**Strong naming & symbols.** IL rewriting invalidates the compile-time signature, so the weaver
re-signs with the repo `Open.snk` (`sn -vf` valid; `StrongNameTests` gates identity) and
rewrites the portable PDB — original sequence points survive (instructions are only inserted or
mutated in place), so source stepping still lands on the right lines.

**Coverage gate.** `test/NumSharp.Tests/Lifetime/NDScopeWeaveTests.cs` reflects over the SHIPPED
assembly and asserts every `[NDScoped]` method (and property accessor) carries a local of type
`NDScope` — a necessary consequence of scoping, whether woven or hand-written. It turns a silently
un-run weave (a broken `NDScopeWeave` target, or a `-p:SkipNDScopeWeave=true` build) red instead of
letting those methods quietly revert to the finalizer backstop, and a non-vacuity floor guards against
the attribute being stripped. The tuple and `INDArrayCarrier` rows have in-tree consumers —
`np.polydiv` (tuple) and `np.unique_counts`/`unique_inverse`/`unique_all` (result structs), pinned
behaviorally by `NDScopeWeaveCarrierTests` (members survive, dropped results re-parent into an
enclosing scope, internal temps leave zero strands). The `out`-parameter egress rows have **no
in-tree consumer** (no `[NDScoped]` method currently takes an out-carrier param); their runtime
semantics are pinned by `NDScopeTests.OutParameter_Egress_ViaReturns` (hand-written scope) plus the
recursive-`Returns(ITuple)`/carrier/buffer pins in `NDScopeWeaveCarrierTests`, and the emitted IL is
a structural twin of the return-value `Returns` path — correct by construction, kept for the shape's
completeness.

**Escape hatches.** `-p:SkipNDScopeWeave=true` builds without weaving (attributed methods then
simply run unscoped — the finalizer backstop, the pre-migration status quo);
`-p:NDScopeWeaveILVerify=true` additionally runs the `dotnet-ilverify` global tool on the woven
output (kept opt-in so machines without the tool still build; measured: the weave adds ZERO
ILVerify findings over the unwoven baseline in both configurations).

**When to hand-scope instead of the attribute:** an UNSUPPORTED carrier the weaver rejects with
NDW003/NDW015 (an object/collection member, an ND-carrying `List`/`Task`/`T : NDArray` component —
a result struct instead just implements `INDArrayCarrier` and stays woven, and any tuple of
supported shapes, nested included, is woven too), `ref`/`in` flows over ND-carrying shapes, a
mid-method handback that genuinely must free before a later allocation (nest an inner block scope),
or any body where the egress isn't expressible as return-value + out-params.

**Retained arguments — `[NDScopedExit]`.** The scope auto-protects only two egresses — the return
value and `out` params. A tracked array handed to something that **keeps** it (a field/property
store, a long-lived collection, a closure/task outliving the call) is otherwise STILL disposed at
scope exit, dangling the retained reference — a use-after-free. (Passing to something that only
*reads* the array — every `np.*` op, the overwhelming case — is safe: the scope disposes it after,
which is the whole point of "one scope covers the helper tree.") Mark the RETAINING parameter
`[NDScopedExit]` and the weaver injects `NDScope.Detach(param)` at the callee's entry; because the
scope is `[ThreadStatic]` and the callee runs inside the caller's scope, the detach reaches the
caller's scope with no call-site plumbing, so the argument survives (falling to the finalizer
backstop unless the retainer disposes it — **survival, not eager reclamation**). It is a no-op with
no ambient scope, so an `[NDScopedExit]` method is always safe to call. It composes with any method
shape (with or without `[NDScoped]`/`[NDScopedAsync]`), covers `NDArray`/`NDArray[]`/tuple-of-NDArrays
parameters, and covers property setters (a setter's `value` is a parameter). What it does NOT cover:
a raw public-field store (`obj.field = a` — no parameter to annotate; use a property or a hand
`NDScope.Detach`), a parameter you cannot annotate (a BCL sink like `List<NDArray>.Add` — hand-detach
the temp), and a hand-scoped body (the weaver leaves those untouched, so their retained params are
the author's to detach). The alternative to the attribute is what it automates: `NDScope.Detach(x)`
at the retention site, or building the retained array BEFORE the scope opens (never tracked).

---

## 2. Why this exists

A dropped `NDArray`'s buffer is reclaimed only when its finalizer runs. In a tight loop the finalizer
lags the allocation rate, so most calls miss the warm pool and pay a cold `NativeMemory.Alloc` +
first-touch page fault, plus ~950 ns of finalization per dropped result. NumPy is refcounted and
frees instantly. Disposing closes that gap. The stakes, measured on this repo (Release, best-of-rounds):

| Workload | Not disposed | Disposed | Pool hit-rate |
|----------|--------------|----------|---------------|
| Comparison ops `a<b`/`a==b` @ N=1000 (geomean vs NumPy) | **0.53×** | **`using` 1.22× / `try-finally` 1.11×** | 27–29% → **100%** |
| `np.ptp(a,0)` small hot-loop | 3149 ns/call | **1930 ns/call (1.63×)** | 38.7% → **70.6%** |

The win is **concentrated at small N and in hot loops**. At large N the allocation is amortized: the
time win shrinks, though the GC-pressure win remains. (See Appendix B for the harnesses.)

---

## 3. The mechanism

```
NOT disposed:  result dropped ──▶ GC ──▶ ~NDArray finalizer ──▶ Release ──▶ Return(buffer)   [LATE]
Disposed:      Dispose() ──▶ Release ──▶ refcount 0 ──▶ Return(buffer)  + SuppressFinalize    [NOW]
```

`Dispose()` (`Backends/NDArray.cs`) is idempotent, calls `Storage.InternalArray.Release()`, and
`GC.SuppressFinalize`s the object. `Release()` (`Backends/Unmanaged/UnmanagedMemoryBlock\`1.cs`)
decrements the block's refcount and, **only on the 0-transition**, hands the buffer to
`SizeBucketedBufferPool.Return`. So disposing gives warm-pool reuse next iteration *and* removes the
finalization cost. The `~NDArray()` finalizer remains as a safety net for anything not disposed.

---

## 4. Correctness invariants

Disposal is **automatic reference counting (ARC)**. These invariants are what make the standard safe;
each is verified (Appendix B):

- **I1 — `Dispose()` decrements a refcount and frees only at 0.** Not "frees on first dispose."
- **I2 — Views share that refcount.** `UnmanagedStorage.Alias(shape)` reuses the *same* underlying
  block (`r.InternalArray = InternalArray`), and the view's ctor calls `InitializeArc() → TryAddRef()`.
  A `reshape`/slice/`T`/`expand_dims` bumps the shared count.
- **I3 — ⇒ disposing a base while a view is alive does NOT free the buffer.** After `t.Dispose()`
  with a live view `v`: `t.IsReleased == v.IsReleased == false`, and `v` still reads correct data.
  Only disposing the *last* reference frees it. *(This is why explicit `Dispose` is safe where the
  reverted auto-reclaim ring was not: the ring freed at refcount ≤1 via `IsUniquelyReferenced`;
  `Dispose` frees at 0.)*
- **I4 — Double-dispose is safe** (idempotent; `SuppressFinalize` and the finalizer never collide).
- **I5 — Fresh vs. view.** Elementwise ops, reductions, `astype` (**even same-dtype — it copies**),
  `sqrt`/`power`/… return **fresh, owned** buffers (`@base == null`). `reshape`/`T`/slice/`squeeze`/
  `expand_dims`/`broadcast_to`/`diag(2-D)`/`real` return **views** (`@base != null`).
- **I6 — The ONLY way to corrupt memory is to release the *last* reference while the data is still
  needed** — directly (use-after-dispose) or via the *caller* (you disposed what they still hold).
  Everything in §5 and §8 follows from this one fact.

---

## 5. Decision Flow 1 — *should* I dispose this intermediate?

For each local `var t = <np.* op or operator>;`:

```
                 ┌───────────────────────────────────────────────────────────────┐
                 │  Is `t` a fresh, locally-created np.* / operator result?       │
                 │  (not a parameter, field, or alias of one)                     │
                 └───────────────┬───────────────────────────┬───────────────────┘
                          No  (input/field)              Yes │
                                 │                            │
                                 ▼                            ▼
                    ┌──────────────────────┐   ┌───────────────────────────────────────────┐
                    │ DO NOT dispose  (R2) │   │ Is `t` — or a VIEW derived from `t` —      │
                    │ You don't own it.    │   │ returned, stored in a field, or passed out?│
                    └──────────────────────┘   └───────────┬───────────────────┬───────────┘
                                                     Yes    │                No │
                                                            ▼                   ▼
                                            ┌──────────────────────┐  ┌──────────────────────────────┐
                                            │ DO NOT dispose  (R1) │  │ Are ALL uses of `t` (and of   │
                                            │ Never dispose the    │  │ any view of it) already done? │
                                            │ value you return.    │  └────────┬──────────────┬───────┘
                                            └──────────────────────┘      No   │           Yes│
                                                                               ▼              ▼
                                                                  ┌────────────────────┐  ┌──────────────┐
                                                                  │ Move the dispose    │  │ DISPOSE.     │
                                                                  │ AFTER the last use  │  │ Go to Flow 2 │
                                                                  │ (R3), then re-check │  │ for the form.│
                                                                  └────────────────────┘  └──────────────┘
```

**The three rules, named:**

- **R1 — Never dispose what you return.** Not `t`, and not a value that *is* `t` (e.g. a `squeeze`/
  `reshape` view of `t` that becomes the result). The return value is the caller's to own.
- **R2 — Never dispose an input.** Parameters, fields, or anything aliased to them belong to the
  caller. Watch loop seeds like `NDArray acc = a;` — the first iteration's `acc` *is* the input.
- **R3 — Dispose after the last use, counting uses through views.** If a `reshape`/`T` of `t` is
  still read later, `t`'s last use is *there*, not at `t` itself.

**When in doubt, don't.** The finalizer still reclaims it — you only forgo the optimization, never
correctness.

---

## 6. Decision Flow 2 — *which form*

```
        ┌──────────────────────────────────────────────────────────┐
        │ Does `t` live until method exit (dies at end of scope)?   │
        └───────────────┬──────────────────────────┬───────────────┘
                    Yes │                        No │  (dies mid-method, or per-loop-iteration)
                        ▼                            ▼
              ┌───────────────────┐   ┌──────────────────────────────────────────────┐
              │ using var t = …;  │   │ Can a throw between create and last-use both  │
              │ (DEFAULT)         │   │ HAPPEN and MATTER (real cleanup needed)?      │
              └───────────────────┘   └───────────────┬──────────────────┬───────────┘
                                                  Yes  │               No │
                                                       ▼                  ▼
                                        ┌──────────────────────────┐  ┌──────────────────────┐
                                        │ try { … }                │  │ t.Dispose();          │
                                        │ finally { t.Dispose(); } │  │ (direct, at the exact │
                                        └──────────────────────────┘  │  point `t` dies)      │
                                                                      └──────────────────────┘
```

| Form | Use when | Example site |
|------|----------|--------------|
| **`using var t = …;`** (default) | `t`'s lifetime **is** the enclosing scope. Exception-safe for free, reads clean. | `np.ptp` `maxRes`/`minRes`; `np.corrcoef` `stddev` |
| **direct `t.Dispose();`** | `t` dies **mid-method** (free its buffer before later allocations) or **per-loop-iteration** — where `using` can't express the lifetime. | `np.cov` `avg` (freed before the GEMM); `np.ptp` multi-axis loop |
| **`try { … } finally { t.Dispose(); }`** | disposal must span a non-scoped/conditional lifetime, `t` is assigned conditionally, OR a throw between create and last-use both happens and requires cleanup. | rare for pooled intermediates; common for real resources (streams, handles) |

**Prefer `using`.** Drop to direct `Dispose()` only for the mid-method / per-iteration lifetimes;
drop to `try/finally` only when its extra ceremony buys real cleanup-on-throw (see Dilemma D3).

---

## 7. The dilemmas

The judgment calls this standard exists to resolve. Each is stated as a genuine tension, then resolved.

### D1 — Performance vs. corruption risk
Disposing wins at small-N/hot-loops (1.6×–3×), but a wrong dispose is a *silent* use-after-free that
can **return the correct value** while corrupting the caller's input (§8, BUG C). **Resolution:**
treat disposal as an optimization applied to *hot* sites only, gated by mandatory byte-neutral
verification (§12). Correctness-first: when unsure, don't dispose.

### D2 — `using` vs. direct `Dispose()`
`using` is clean and exception-safe but pins the lifetime to the scope; a direct `Dispose()` fits
mid-method / per-iteration lifetimes but is skipped if code throws first. **Resolution:** choose by
*lifetime* (Flow 2), not by "which feels safer." Forcing a mid-method dispose into `using` **defeats
the point** — it holds the buffer to method exit instead of handing it back before the next
allocation.

### D3 — What does exception-safety actually buy *here*?
`using`/`try-finally` guarantee `Dispose()` runs on the throw path. But for a **pooled transient**,
the only consequence of *not* disposing is that the `~NDArray()` finalizer reclaims it later — the
pre-migration status quo. **Resolution:** in this design, exception-safety protects **the
optimization, not the memory**. There is no leak (finalizer backstop), no double-free (idempotent +
`SuppressFinalize`), no corruption. So do **not** pay `try/finally` ceremony for a pooled intermediate
whose worst exception-path outcome is "reclaimed a bit later." Reserve `try/finally` for genuine
resources (files, sockets, native handles) where a skipped dispose is a permanent leak. *(In the
backend these ops throw only on programmer errors — bad shape/dtype/axis — validated up front, which
abort the whole call anyway.)*

### D4 — Views: chase them or ignore them?
By I3, disposing a base while a view is alive doesn't free the buffer — so *forgetting* a view is
never a corruption bug. But then the buffer isn't reclaimed until the view finalizes, so the
optimization is only *partial*. **Resolution:** for **correctness** you may ignore views; for
**benefit** you must release the *last* reference — dispose the owner **and** any still-live views of
it (see the `corrcoef` example in §9, which `using`s `stddev` *and* its two reshape views).

### D5 — Ownership ambiguity (does this op alias the input?)
Some ops *might* return a view/self instead of a fresh buffer (`astype` with `copy` semantics,
`atleast_2d`, `ravel`, `ascontiguousarray`). If such a result aliases an **input**, disposing it
violates R2. **Resolution:** verify per-site. Check `result.@base == null` (owns) and, when it
matters, that `result.Storage.Address != input.Storage.Address`. *(Verified: `np.atleast_2d(m).astype(sameDtype)`
copies — the result is independent of `m`. But do not assume this for every op; confirm.)* When
unsure, don't dispose.

### D6 — The return value is sacrosanct, even when you'd love to reclaim it
You often build a result via a `squeeze`/`reshape` **view** of an intermediate (`np.cov` ends with
`np.squeeze(c)`). You cannot dispose `c` — the returned view aliases it (R1). **Resolution:** the
method disposes only what it *owns and keeps internal*; the result belongs to the caller. If prompt
reclamation matters, that's the caller's job (D7).

### D7 — Diminishing returns / partial migration
`np.ptp`'s migration reached only 70% pool hit-rate because the returned `diff` still finalizes — the
method disposed the 2 consumed intermediates but must hand back the 3rd. **Resolution:** the library
disposes what it owns; the **caller** disposes what it receives. A hot loop calling `np.ptp` should
`using` the result to approach 100%. This is a two-audience contract (§10), not a library shortfall.

---

## 8. The bug catalog

All three are real, deterministic corruptions (Appendix B). They are R1/R2/R3 violations — the *form*
(`using` vs direct) does not matter; disposing the **wrong object** is the bug.

### BUG A — dispose too early (release the last ref, then use)
```csharp
var t = a - b;
t.Dispose();          // refcount 0 -> buffer returned to pool
var s = t.sum();      // ❌ use-after-free: reads a reused buffer -> wrong value (e.g. 7)
```
Violates **R3**. Fix: dispose after the last use.

### BUG B — dispose the value you return
```csharp
NDArray Double(NDArray x) { using var t = x * 2.0; return t; }   // ❌ `using` disposes the return
// caller: r.IsReleased == true -> reads garbage
```
Violates **R1**. Fix: `return x * 2.0;` — never `using`/dispose a returned local.

### BUG C — dispose an input (the insidious one)
```csharp
NDArray acc = a;                       // acc starts as the INPUT
foreach (var ax in axes) {
    var prev = acc;
    acc = np.amax(acc, ax, keepdims: true);
    prev.Dispose();                    // ❌ iteration 1 frees the caller's `a`
}
return acc;                            // returns the CORRECT answer...
// ...but the caller's `a` is now IsReleased == true — a use-after-free planted in the caller
```
Violates **R2**. The method **returns the right value**, so value tests pass; only the caller's *next*
use of `a` breaks. Fix: guard the input — `if (!ReferenceEquals(prev, a)) prev.Dispose();`. This class
is why §12's UAF audit is mandatory.

---

## 9. Worked examples

Real backend sites — the boundary methods now carry `[NDScoped]` and the weaver injects what
these examples spell out by hand (each example shows the SEMANTICS the woven IL carries; only the
carrier-struct sites — `AverageCore`, the `unique` result-struct wrappers — still write the scope
in source). The hand-written forms below them remain the tool for engine internals and mid-method
handback. (All proven byte-neutral: FuzzMatrix bit-exact + the Lifetime slope sweep at zero
strands.)

**A) The scope default — original body + two lines** — `np.ptp` overload 1:
```csharp
using var scope = NDScope.Open();
var maxRes = np.amax(a, axis, keepdims);   // tracked; reclaimed at exit
var minRes = np.amin(a, axis, keepdims);
var diff = maxRes - minRes;
return scope.Returns(@out is null ? diff.MarkReductionScalar() : WriteOrReturn(diff, @out));
// ^ diff is yielded on the no-out path; a caller-supplied @out is untracked -> Returns is a no-op
```

**B) One scope covers a whole helper tree** — `np.isin`:
```csharp
using var scope = NDScope.Open();                       // at the public boundary
...
NDArray member = ComputeMembership(ar1, ar2);           // helpers below are scope-free and pristine:
NDArray result = invert ? np.logical_not(member) : member;   // their casts/sorts/masks all track here
return scope.Returns(result.reshape(outShape));
```

**C) Yielding a view over a tracked base** — `np.cov`:
```csharp
using var scope = NDScope.Open();
NDArray X = np.atleast_2d(m).astype(resultType);        // original composition, untouched
...
return scope.Returns(np.squeeze(c));   // the view is yielded; c's tracked wrapper releases at exit,
                                       // ARC keeps the buffer alive through the view (I3)
```
Where the old mid-method handback genuinely matters (freeing a big transient BEFORE the next
allocation), nest an inner block scope around just that stretch — the inner `Returns` re-parents
its result outward.

**D) The loop seed that IS the input** — `np.ptp` overload 2, now scope-covered:
```csharp
using var scope = NDScope.Open();
NDArray maxRes = a, minRes = a;                    // seeds ARE the input — never tracked, so the
foreach (var ax in sortedDesc) {                   // scope cannot touch them (R2 is structural)
    maxRes = np.amax(maxRes, ax, keepdims: true);  // each intermediate reduction is tracked
    minRes = np.amin(minRes, ax, keepdims: true);
}
NDArray diff = maxRes - minRes;
...
return scope.Returns(WriteOrReturn(diff, @out));
```
The hand-written equivalent of this loop needed a `ReferenceEquals(prev, a)` guard on every
iteration — the exact bug class (BUG C) the scope design retires.

---

## 10. Two audiences

Reclamation is a contract between the library and its callers:

- **Backend authors** dispose the intermediates a method *owns and keeps internal* (§9). They must
  **never** dispose the return value (R1) or an input (R2).
- **Callers / hot-loop code** dispose the *results they receive*. This is the other half of the
  optimization — e.g. the comparison-operator win (0.53× → 1.22×) comes entirely from the caller
  wrapping the mask:
  ```csharp
  for (…) { using var mask = a < b; total += CountTrue(mask); }   // caller-side `using`
  ```
  A method like `np.ptp` can only reach ~70% pool reuse on its own; the caller `using`-ing the result
  closes the rest (D7).

---

## 11. Decision Flow 3 — is this site worth migrating?

```
      ┌────────────────────────────────────────────────────────────────┐
      │ Does the method allocate transient np.* intermediates it OWNS,  │
      │ AND get called in hot loops / at small N?                       │
      └───────────────┬────────────────────────────────┬───────────────┘
                  No  │                             Yes │
                      ▼                                 ▼
        ┌──────────────────────────────┐   ┌────────────────────────────────────┐
        │ Low priority. The finalizer  │   │ Migrate: apply Flow 1 per           │
        │ already reclaims; the win is │   │ intermediate, Flow 2 per form,      │
        │ small on cold / large-N.     │   │ then the verification gate (Flow 4).│
        │ Migrate opportunistically.   │   └────────────────────────────────────┘
        └──────────────────────────────┘
```

**Priority sites:** composition-heavy reductions/statistics called in loops — `ptp`, `cov`,
`corrcoef`, `std`/`var` engine internals, `median`/`percentile`, norm/reduction compositions.
**Migrate incrementally** — one method per change.

**Coverage note (audited).** The hot reduction/statistics surface is covered directly OR
transitively: `std`/`var` delegate to the scoped `ReduceStd`/`ReduceVar`, and
`median`/`percentile`/`quantile` plus their `nan*` twins delegate to the scoped
`QuantileEngine.Compute`, so they own no transients of their own. The linear-algebra, core-math, polynomial, and
N-D FFT COMPOSITIONS that DO own transients are now scoped too: `einsum` (its `EinsumContract`
core — the writeable single-operand view path stays writeable and writes through under the scope,
pinned by `EinsumContractionTests.ViewPath_*`), `cross`, `kron`, `outer`, `tensordot`,
`linalg.matrix_power`, `linalg.norm`, `linalg.multi_dot`, `diff`, `ediff1d`, `vander`, the
polynomial family (`poly`, `polyval`, `polyder`, `polyint`, `polyadd`/`polysub` via `PolyAddSub`,
`polymul`), and the eight N-D FFTs (`fft2`/`ifft2`/`fftn`/`ifftn`/`rfft2`/`rfftn`/`irfft2`/`irfftn`
— each a per-axis 1-D composition; byte-exactness re-confirmed by the `fft.jsonl` fuzz tier).
Deliberately NOT scoped (documented non-targets): single-kernel ufuncs and views/passthroughs (no
owned transients), the kernel-driven `nanmean`/`nanstd`/`nanvar`, and the THIN product wrappers
`inner`/`vdot`/`vecdot`/`matvec`/`vecmat` (≈1 `dot` + a reshape — marginal). **Carrier returns are
now WEAVABLE** — the weaver yields a `ValueTuple` of NDArrays through the `Returns` tuple overloads
and a result struct through its `INDArrayCarrier.YieldTo`, so `[NDScoped]` works on them exactly like
a bare `NDArray` return: `np.polydiv` (tuple) and the `unique_counts`/`unique_inverse`/`unique_all`
result-struct wrappers are woven (the hand-scoped `AverageCore`/`np.unique` are equivalent and may be
converted opportunistically). The remaining carrier compositions — `meshgrid`/`mgrid`/`ogrid` (grid
result structs, all of which already implement `INDArrayCarrier`), `polyfit` (`PolyfitResult`), and
the factorization tuples `svd`/`qr`/`eig`/`eigh`/`lstsq`/`slogdet` (also backend-only) — need only
the `[NDScoped]` attribute added if a profile shows them hot. Still DEFERRED (low-priority per §11):
the **one-shot structural** array ops `pad`/`insert`/`delete`/`block` and the `r_`/`c_` construction
indexers (not hot-loop / small-N) — migrate opportunistically.

---

## 12. Decision Flow 4 — the verification gate (mandatory)

A dispose migration must be **byte-neutral** (same outputs) and must **not free live data**.

```
   1. BYTE-NEUTRAL DIFFERENTIAL
      Run the SAME exhaustive script against the un-migrated build and the migrated build.
      diff(out_before, out_after)  ── must be EMPTY ───────────────────────────────┐
                                                                                    │
   2. EXISTING GATE                                                                 │
      dotnet test --filter "TestCategory=FuzzMatrix"  + the method's unit tests     │
      ── must be GREEN (the migration changes NO values) ─────────────────────────┤
                                                                                    │
   3. UAF AUDIT (catches BUG C, which values miss)                                  │
      After the method returns, assert for every input param AND the return value:  │
      !arr.Storage.InternalArray.IsReleased   (true == use-after-free) ────────────┤
                                                                                    ▼
                                                              all pass ──▶ SHIP.  any fail ──▶ revert/fix.
```

The differential is the strong proof: identical bytes across every dtype × shape × parameter path
means the dispose changed *nothing observable*. The UAF audit is what catches the class where values
alone lie (BUG C). *(The audit reads internal state — `Storage`/`InternalArray.IsReleased` — so it
lives in test/debug code that has friend access.)*

---

## 13. Anti-patterns

- **Do NOT automate liveness inference.** A per-thread eager-reclaim ring that freed at GC-detection
  was memory-unsafe (it freed at refcount ≤1 while a view was live) and was reverted. `NDScope` is
  NOT that: it automates *registration* only — the programmer still declares deadness (everything
  not `Returns`ed) and disposal is ordinary refcounted `Dispose`.
- **Do NOT scope a caller loop.** A scope holds its temps to exit; thousands of same-size buffers
  alive at once overflow their pool bucket and get freed instead of pooled. Scope one call.
- **Do NOT forget `Returns` on an egress** — a yielded-but-not-`Returns`ed return value is BUG B by
  automation. The blanket rule is safe and mechanical: wrap every `return` (and every `out`
  assignment) — an untracked value passes through as a no-op.
- **Do NOT mass-migrate without the gate.** The value is in the hot sites; the risk is a silent UAF.
  Earn each migration with §12.
- **Do NOT `using`/dispose a returned local** (R1) or **an input** (R2) — including via a view.
- **Do NOT force a mid-method lifetime into `using`** — it defeats the buffer handback (D2).
- **Do NOT add `try/finally` around a pooled-intermediate dispose "for safety"** — the exception path
  only forgoes the optimization; the finalizer is the backstop (D3). Save `try/finally` for real
  resources.
- **Do NOT assume an op returns a fresh buffer** — confirm `@base == null` before disposing a result
  that might alias an input (D5).

---

## Appendix A — ARC internals reference

*(File paths as of this writing; behaviour is what matters.)*

- **`Backends/NDArray.cs`** — `Dispose()` (idempotent; `Release()` + `SuppressFinalize`), `~NDArray()`
  (safety-net finalizer), `InitializeArc()` (`TryAddRef` on construction), `@base` (null ⇒ owns data).
- **`Backends/Unmanaged/UnmanagedMemoryBlock\`1.cs`** — the `Disposer`: `Release()` decrements
  `_refCount` and, on the 0-transition, `ReleaseUnmanagedResources()` → `SizeBucketedBufferPool.Return`;
  `~Disposer()` finalizer backstop; `IsReleased`, `IsUniquelyReferenced`, `TryAddRef`.
- **`Backends/Unmanaged/Pooling/SizeBucketedBufferPool.cs`** — `Take`/`TakeZeroed`/`Return`, and the
  diagnostics `Hits`/`Misses`/`Returns`/`ResetCounters()`/`Clear()` used by the verification harnesses.
- **`Backends/Unmanaged/UnmanagedStorage.Cloning.cs`** — `Alias(shape)` builds a view over the **same**
  block (`r.InternalArray = InternalArray`), which is why views share the refcount (I2).
- **`Backends/Unmanaged/Interfaces/IArraySlice.cs`** — `TryAddRef()`, `Release()`, `IsReleased`,
  `IsUniquelyReferenced` (the surface the UAF audit reads).

---

## Appendix B — the experiments behind every claim

Runnable POCs (`dotnet run -c Release <file>.cs`; Release is mandatory — Debug taints hand-written
kernels ~2×). Point `#:project` at `src/NumSharp.Core`.

| Claim | Experiment |
|-------|------------|
| I3 (views protected), I4 (double-dispose), I5 (astype copies), clean-dispose value-safety | `verify2.cs` (F1/F4/F5/F6), `probe_view_arc.cs` |
| BUG A / BUG B / BUG C corrupt deterministically | `verify2.cs` (F2/F3), `bug_migration.cs` |
| §12 differential — migration is byte-neutral (19 cases) | `check_main.cs` vs `check_wt.cs` → `diff` empty |
| §2 benefit — `ptp` 1.63×, hit-rate 38.7%→70.6% | `bench_ptp_main.cs` vs `bench_ptp_wt.cs` |
| §2 benefit — comparison ops 0.53×→1.22× (caller-side `using`) | `poc_dispose.cs` |
| The 5 migrations | `migrations.patch` (`git apply` to reproduce) |

**Observing a free:** `SizeBucketedBufferPool.Returns` increments when a buffer is actually returned
(refcount hit 0). **Observing a use-after-free:** `arr.Storage.InternalArray.IsReleased == true` on an
array a caller still holds. Both are how the harnesses turn "did it free?" into an assertion.

---

*This document is a coding standard, not library code. It describes how to migrate hot backend sites
to eager disposal safely; it prescribes no automatic behaviour.*
