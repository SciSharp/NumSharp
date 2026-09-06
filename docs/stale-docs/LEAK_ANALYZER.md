# NDW012 — the NDArray leak analyzer: supported cases & what NDScope handles automatically

`NDArrayLeakAnalyzer` (`tools/NumSharp.Build.Analyzer/`) is a Roslyn flow analysis that reports a
**warning** — NDW012, never an error — for an NDArray a method **creates but never reclaims or hands
off**, so its pooled unmanaged buffer is left to the finalizer instead of being returned promptly (the
exact cost `[NDScoped]` exists to remove; see `DISPOSAL-GUIDELINES.md` and the website's
`numsharp-build-compiler.html#ndw012`). It ships **inside the NumSharp package** (`analyzers/dotnet/cs/`, packed by
`_NumSharpCorePackAnalyzer`), so every `PackageReference` consumer gets it with no extra install; it
also runs on NumSharp.Core's own build (toggle: `-p:EnableNDArrayLeakAnalyzer=false`; Core adds NDW012
to `$(WarningsNotAsErrors)` so a leak can never break a build even under `TreatWarningsAsErrors`).
Its sibling `NDArrayHolderAnalyzer` (same DLL, same toggle, NDW016/NDW017 likewise in
`$(WarningsNotAsErrors)`) carries ownership across the field boundary this pass stops at — a TYPE that
stores NDArrays must be disposable and dispose them, and an NDArray-owning disposable is itself an
owned value — see §10.

Every case below is **probe-verified and pinned**: the fixture files in
`test/NumSharp.Tests.Build.Analyzer.Fixtures/` carry a `// [NDW012]` tag on each line that must warn,
the exact-match gate fails on both a **missing and an unexpected** diagnostic, and
`test/NumSharp.Tests.Build.Analyzer/` (125 in-process + 4 build tests, green on net8.0 and net10.0)
adds metamorphic pairs, invariant sweeps, and a 100-seed property fuzz on top. The backlog and its
history live in `test/NumSharp.Tests.Build.Analyzer/COVERAGE_PLAN.md`.

---

## 1. The model

The analysis is **per method body** (one pass per method / accessor / constructor / local function),
over the ORIGINAL operation tree (pre-lowering, so async and iterator bodies analyze like plain ones).
It tracks every **owned NDArray value** — a value the method itself constructed — and classifies where
each one goes:

| Disposition | Meaning | Outcome |
|---|---|---|
| **Reclaimed** | disposed or yielded to an `NDScope` | clean |
| **Escapes** | returned / stored / handed off — someone else's to reclaim | clean |
| **Leaks** | dropped, dead, or fed to a NumSharp op that never disposes its inputs | **NDW012** |

Owned carrier types: `NDArray` (and subclasses — `NDArray<T>`, consumer subclasses), `NDArray[]`, a
`ValueTuple` containing any NDArray (any nesting), and an `INDArrayCarrier` struct (including a
consumer's own). Resolution is by **symbol**, not spelling — aliases (`using ND = NumSharp.NDArray`),
nullable annotations, fully-qualified names and global usings all resolve; with NumSharp not
referenced at all the analyzer no-ops.

A local that is ever assigned a **plain input** (parameter, field, property, another local) is an
*alias* and is never tracked — disposing it could free the **caller's** buffer (rule R2 throughout).
For a tracked local, **any single reclaiming or escaping use saves it** — the analysis is deliberately
path-insensitive (see §6, dispose-in-catch).

## 2. What counts as a producer (an "owned" value)

**Direct producers:**

| Expression | Example |
|---|---|
| object creation | `new NDArray(typeof(double), new Shape(n))` |
| binary / unary operator | `a + b`, `-a`, `a & b` |
| any call returning an owned type | `np.add(a, b)`, `NDArray.Scalar(5.0)`, a local helper returning `NDArray`/`NDArray[]`/tuple |

**Composite producers** — a container literal owns what its elements own. Producer **iff at least one
element is itself a producer** (it holds all elements at once; a container of plain inputs owns
nothing — R2):

| Expression | Dropped ⇒ | Of aliases ⇒ |
|---|---|---|
| tuple literal `(a + b, c - d)` (any nesting, mixed too) | NDW012 | `(a, b)` silent |
| array creation `new[] { a + b }` | NDW012 | `new[] { a, b }`, `new NDArray[5]` silent |
| C# 12 collection expression `[a + b]` | NDW012 | `[a, b]` silent |

**Conditional producers** — a conditional holds only ONE branch. Producer **iff every completing
branch produces** (a single aliasing branch makes demanding reclamation unsafe — the same conservatism
as the mixed `if/else` rule); a **`throw` branch is vacuous** (it never completes, so it can never be
the held value):

| Expression | Dropped ⇒ |
|---|---|
| `p ? a + b : c - d` | NDW012 |
| `n switch { 0 => a + b, _ => c - d }` | NDW012 |
| `(a + b) ?? (c - d)` | NDW012 |
| `p ? a + b : throw new …` / a `switch` with throw arms | NDW012 |
| `p ? a + b : a` (mixed), `x ?? (a + b)`, `p ? a : throw …` | silent (conservative) |

> The any-element / every-branch asymmetry is deliberate — do not unify the two rules. Any-element on
> conditionals flags mixed ternaries (R2 false positives); every-element on tuples goes silent on
> `(temp, alias)` drops, which are real leaks (`np.setops` had one).

**NOT producers** (view reads — excluded so fluent view chains don't drown the signal): property reads
(`a.T`, `a.flat`), indexer reads (`a["1:3"]`, `a[mask]`), implicit scalar conversions (`NDArray t = 5`),
and `NDScope.Returns/Attach/Detach` results (reclamation plumbing).

## 3. The leak verdicts (what actually warns)

| Case | Example |
|---|---|
| dead local | `var t = a + b;` (never used again) |
| discarded result | `np.add(a, b);` — also expression-bodied: `=> np.add(a, b);` |
| explicit discard | `_ = a + b;`, `_ = p ? a + b : c - d;` |
| **arg to a non-consuming NumSharp op** | `return np.sum(a * b);` flags `a * b` — a NumSharp method that RETURNS an NDArray never disposes its NDArray inputs (R2), so the temp is stranded. Also through a container: `np.concatenate(new[] { a + 1.0, c }, 0)` |
| operand of an operator | the inner `a + b` of `return a + b + c;` (one temp per extra link) |
| loop temp | `for (…) { var t = a + b; }` |
| conditionally-assigned dead local | `NDArray t; if (p) t = a + b; else t = c - d;` (unused) |
| compound-assigned dead local | `var t = a + b; t += c;` (unused — seed AND result die) |
| abandoned-by-throw | `var t = a + b; throw …;` |
| deconstruction with an unused side | `var (q, r) = Split(a);` and into EXISTING locals `(q, r) = Split(a);` — and of a tuple literal `var (q, r) = (a+1.0, b-1.0);` |
| dropped composite/conditional | every table row in §2 |

Only the **outermost** leaking expression of a chain is reported (a nested owned sub-expression whose
owning ancestor also leaks is subsumed) — one diagnostic per actual leak.

## 4. The escapes (all clean — someone else owns it now)

| Family | Cases |
|---|---|
| return / yield / await | `return x;` (incl. via a local, tuples, arrays, conditionals), `yield return a + b;`, `var v = await ComputeAsync(a + b);` |
| out / ref / in | assignment to an `out` parameter; `Foo(ref t)` / `Foo(out t)` / `Foo(in t)` |
| **stores** | static/instance **property** setter, static/instance **field**, object-initializer `new W { Prop = a + b }`, array element `arr[0] = …`, dictionary/indexer `map[k] = …`, **NumSharp's own indexer-set** `a["1:3"] = b + c`, `ref`-local store, compound into a field `_f += a`, deconstruction into fields — the storing TYPE now owns it, which is what NDW016/NDW017 (§10) hold it to |
| ctor args | `new Wrapper(a + b)` (assumed to take ownership), `: base(a + b)` |
| foreign calls | any argument to a **non-NumSharp** method (assumed to consume/observe), incl. `params` elements, collection `Add`, local-function args; a **void** NumSharp op too (`np.copyto(dst, a + b)` — the non-consuming rule keys on the RETURN type) |
| observation | `Console.WriteLine(a + b)`, string interpolation `$"{a + b}"` (result isn't owned) |
| closures / delegates | lambda capture (`Action f = () => Use(t);`), a lambda RETURNING its temp (`Func<NDArray> f = () => a + b;`) |
| views | the receiver of a call/property/indexer **follows the result upward**: `(a + b).reshape(1, -1)` returned, `(a + b)?.reshape(…)`, `x.MakeGeneric<T>()` chains — the view shares the buffer, so if the result escapes the receiver is not a leak |
| consumption | `foreach (var x in Split(a))` — a produced `NDArray[]`/tuple/carrier is a managed container; the loop owns each element's fate (conservative). NOT a produced NDArray or owning disposable (`foreach (var r in a + b)`, `foreach (var x in np.nditer(a))`): C# disposes the enumerator, never the enumerable — those leak (§10) |

## 5. The reclaims (all clean — deterministically disposed)

| Spelling | Notes |
|---|---|
| `using var t = a + b;` | using declaration |
| `using (var t = a + b) { }` | using statement |
| `using (a + b) { }` / `using ((IDisposable)(a + b)) { }` | bare owning expression, cast included |
| `t.Dispose();` | anywhere — including `try/finally`, **catch-only** (path-insensitive by design), and inside a **local function** (local-function bodies are part of the outer method's operation tree); `t.DisposeAsync()`, `t.Close()` and NumSharp's `it.close()` (an `np.nditer`) are the same reclaim |
| `t?.Dispose();` | conditional dispose |
| `((IDisposable)t).Dispose();` | ANY parameterless `Dispose` counts — the receiver walk only ever arrives from an owned value, so the static type of the call is irrelevant |
| `scope.Returns(x)` / `NDScope.Attach(x)` / `NDScope.Detach(x)` | handed to the scope machinery |

## 6. Exemptions (the whole method is skipped)

| Exemption | Why |
|---|---|
| `[NDScoped]` / `[NDScopedAsync]` | the weaver injects an ambient scope that reclaims every transient (§9) — includes **property-level** attributes (the getter is exempt via its associated symbol) |
| `[NDScopedCovered]` | the author's assertion that this helper ALWAYS runs under a caller's open scope — the per-method analysis has no call graph to see it (the sanctioned fix for the ambient over-report, §8) |
| a hand-opened `NDScope.Open()` / `OpenOrResume()` anywhere in the body | its own scope owns reclamation — even when the `Open` sits in a nested block (matches the weaver's whole-method skip) |

Not exempt (deliberately): a **local function** inside a non-scoped method is its own analyzed block —
its leaks report; a `[NDScoped]` attribute ON a local function draws no gate diagnostic (symbol
actions don't visit local functions — deferred to the weaver).

## 7. Body kinds analyzed

All of them: plain methods, **async** methods, **iterators**, instance and **static constructors**,
**finalizers**, **expression-bodied** members, property **getters**, user-defined **operators**, and
**local functions**. Lambdas are analyzed as part of their enclosing method's block (see SC-5 below
for the scoped-method consequence).

## 8. Known limitations — pinned, not silent

Every divergence is pinned in `KnownLimitationScenarios.cs` under `[TestCategory("Misaligned")]` so a
future improvement flips a tag and turns the pin **red** (noticed, never silent).

**False negatives** (real leak, currently silent — untagged pins):

| Case | Why it stays |
|---|---|
| reassignment orphan: `var t = a + b; t = c - d; return t;` | catching the first value needs per-assignment (SSA) tracking; the analysis is per-local |
| mixed coalesce: `x ?? (a + b)` dropped | leaks only when `x` is null — the every-branch rule's residue |
| `out var` results dropped | whether the callee handed out a FRESH buffer or an alias is invisible per-method |
| `is`-pattern locals: `if ((a+b) is NDArray t)` | rare idiom, not worth the plumbing |
| lambda temp inside a `[NDScoped]` method (SC-5) | the lambda rides the enclosing exemption, yet may run after the scope closes |
| accumulator intermediates: `acc = acc + x` in a loop | the seed is an input alias (conservative clean); intermediates need SSA |

**False positives** (kept alive, but flagged — tagged pins):

| Case | Why it stays |
|---|---|
| view-of-temp: `return np.reshape(a + b, …);` | the returned VIEW keeps the temp's buffer alive, but fresh-vs-view is syntactically undecidable; the receiver-follow choice already absorbs the fluent chains (411→213 first-cut FP reduction), this is the residue |
| ambient-covered helper (AM-1) | a helper whose temps are reclaimed by its `[NDScoped]` CALLER's scope — per-method analysis cannot see call-graph coverage. Fix: mark it `[NDScopedCovered]` |

One deliberate concession inside the receiver-follow: `var t = a + b; return t.sum();` is clean even
though `sum()` returns a fresh buffer — a fresh-returning receiver is syntactically identical to a
view-returning one, and following the receiver is what keeps the pervasive view chains quiet.

## 9. What NDScope actually handles automatically

The runtime mechanism: inside a woven `[NDScoped]` method (or a hand-opened scope), **every NDArray
constructed while that scope is the innermost open one on the thread is attached to it** — including
arrays built by plain helpers it calls, transitively. The woven return path detaches the result(s)
via `scope.Returns(...)`, and scope close disposes everything still attached. So within a scoped
boundary, *every* leak class in §3 is handled automatically — dead locals, discarded results, operator
temps, tuple/array drops, args to non-consuming ops — with no `using`/`Dispose` written by hand. That
is why the analyzer exempts scoped methods outright.

**Quantified on NumSharp.Core today** (reflection over the built assembly + `git grep`, 2026-08-29):

| Mechanism | Count | Meaning |
|---|---|---|
| `[NDScoped]` members (woven) | **265** (245 public, 20 internal) | each gets an automatic ambient scope; effectively the whole hot `np.*` entry surface |
| `[NDScopedAsync]` | 0 | Core has no async API surface (the weaver supports it for consumers) |
| `[NDScopedCovered]` helpers | **30** | analyzer-only assertion: ride the caller's ambient scope (no weave, zero runtime cost) |
| hand-opened `NDScope.Open()` | **3** | manual scopes where weaving isn't suitable |
| NDW012 on a fresh Core rebuild | **1** | the single documented indexer false positive — everything else is scoped, covered, or precisely disposed |

**History of that "1", as the analyzer sharpened** — the numbers tell the coverage story:

- First cut: **411** flagged sites — receivers of fluent view chains dominated; the receiver-follow
  fix cut it to **213**.
- Measured at the disposal-sweep start: **149** unique sites, in three buckets — (1) ambient-covered
  helpers (already handled automatically at runtime; the warning was a per-method-visibility
  artifact — these became `[NDScopedCovered]`), (2) **real leaks in un-scoped leaf APIs**
  (`np.insert`, `np.take`, `np.all`, `np.polyfit`, … — fixed by scoping the entry or precise
  disposal), (3) other FP classes (view-of-input, fluent in-place mutator used as a statement).
  Roughly 60 sites sat in files that already had `[NDScoped]` somewhere; 89 in files with none.
- Post-sweep: **1**.
- When the composite-producer rules landed (§2), Core briefly went 1 → **3**: the analyzer had found
  **two more real leaks** the old rules missed — `np.setops.CanonicalizeSetOpNaN`'s canon scalar
  (ternary of two `NDArray.Scalar`s handed to `np.where`; ambient-covered → `[NDScopedCovered]`) and
  `np.interp.PreparePeriodic`'s wrap temps (inside a `new[]{…}` handed to `np.concatenate`; entries
  unscoped → genuinely stranded → `using` fix). Back to **1**.

The division of labor this leaves: **the ambient scope is the default** (265 woven boundaries + 30
covered helpers = the overwhelming majority of Core's reclamation, fully automatic), **precise
`using`/`Dispose` is the exception** (hot dispatchers where scope setup isn't wanted, and unscoped
leaf paths like interp's periodic branch), and the analyzer is the gate that keeps the two honest —
anything that falls through both shows up as a warning on the very next build.

**The EGRESS side now matches the analyzer's carrier model too** (closed 2026-08-29; probe-verified
live, then pinned). The leak side of §9's promise was always complete — tracking happens at
construction, so ANY dropped shape is reclaimed — but the *yield* side had holes: a scoped method
could return a shape the analyzer models as an owned carrier ("a `ValueTuple` containing any NDArray,
**any nesting**") whose NDArrays the egress machinery then missed, handing the caller **disposed**
arrays from a gate-clean, successfully-woven method. All closed:

- `Returns(ITuple)` / `Detach(ITuple)` now dispatch each component by its RUNTIME type —
  nested tuples (recursively), `NDArray[]` elements, `INDArrayCarrier` structs and bare
  `IArraySlice`/`UnmanagedStorage` components are all seen through (previously only bare `NDArray`
  components were; `((a, b), c)` returned `a`, `b` disposed). Covers sync returns, async
  `SetResult`, `yield return`ed elements, task-carried results, and `[NDScopedExit]` params in one
  fix — every route funnels through these two methods.
- The `out`-parameter escape covers the full carrier vocabulary (was: bare `NDArray`/`NDArray[]`
  only — an `out (NDArray, NDArray)` or `out` carrier struct was silently swept); an ND-carrying
  `out` shape it cannot yield is the new **build error NDW015**, never a sweep.
- **NDW002 widened** to `ref`/`in` over ANY NDArray-carrying shape (an ND tuple, a carrier struct, a
  `List<NDArray>`, a `T : NDArray` — all previously slipped through silently); a general tuple with
  an ND-carrying component the dispatch cannot see through (`List<NDArray>`, `Task<NDArray>`) and a
  nested `Task<Task<NDArray>>` are **NDW003** now instead of weaving-then-corrupting; the gate
  analyzer mirrors all of it and no longer false-positives on a custom `[AsyncMethodBuilder]`
  task-like async method (the weaver validates the real result type at IL time).

Gates: `NDScopeWeaveCarrierTests` (recursive ITuple dispatch ×4), `NDScopeExitTests` (recursive
detach ×2), `NDScopeAsyncTests` (task-carried nested tuple), `GateNegativeTests` (28 new weaver-parity
rows incl. the NDW015/NDW002 matrix and the custom-task-like pin).

## 10. Ownership at the TYPE level — NDW016 / NDW017 and the contagion rule

§4 lists "stores" as a clean escape: a value written into a field or property is the storing type's
to reclaim. Until 2026-09-04 nothing checked that the storing type ever did, so ownership silently
evaporated at every field boundary. `NDArrayHolderAnalyzer` (`tools/NumSharp.Build.Analyzer/
NDArrayHolderAnalyzer.cs`, over the shared `OwnershipModel.cs`) closes the hand-off with two more
**warnings**, and the same model makes owning types *contagious* in the per-method pass:

| Diagnostic | Where | Verdict |
|---|---|---|
| **NDW016** | the type's name | a class/struct declares instance **holders** (see below) but implements neither `IDisposable` nor `IAsyncDisposable` (a `ref struct`: has no `public void Dispose()`) |
| **NDW017** | each member | the type is disposable, but this holder is not disposed on any path from `Dispose()` / `Dispose(bool)` / `DisposeAsync()` / `DisposeAsyncCore()` |

**Holders** — an instance field, or an auto-property (positional record properties included; a
computed property is a view, not storage), whose type *holds NDArrays*, defined structurally:

| Holds | Examples |
|---|---|
| NDArray-like | `NDArray`, `NDArray<T>`, a consumer subclass, `T : NDArray` |
| arrays, any rank | `NDArray[]`, `NDArray[,]`, `NDArray[][]` |
| tuples with a holding component | `(NDArray a, int n)`, `List<(string, NDArray)>` |
| generics instantiated over a holding type, and types nested in them | `List<NDArray>`, `Dictionary<K, NDArray>` (and its `ValueCollection`), `Lazy<NDArray>`, `Task<NDArray>`, a consumer's `Box<NDArray>`, `Cell<NDArray>` |
| `INDArrayCarrier` structs | `UniqueResult`, a consumer's own carrier |
| **another type that stores NDArrays** (own members or base type) — the contagion | a class storing a `Batch`, NumSharp's `np.NDIterator` (its `operands`/`value` are visible), `NpzFile` |

Never holders: delegates, pointers, `WeakReference<T>`, comparers/`IComparable`/`IEquatable`,
`IObservable`/`IObserver`/`IProgress`, unconstrained type parameters, `object`/`IDisposable`, and
any type from an assembly that cannot even name NDArray (the BCL is answered without a member scan).
Static members never make an instance an owner. **`[NDBorrowed]`** (`src/NumSharp.Core/Backends/
NDBorrowedAttribute.cs`, runtime-inert like `[NDScopedCovered]`) excludes a member ("it references an
array owned elsewhere"), or a whole class/struct ("every array it references is borrowed" — which also
removes the type from the contagious set even when it is disposable). Exempt outright: static classes,
interfaces, `[NDBorrowed]` types, and `INDArrayCarrier` implementers (transient result packaging the
scope yields through). Metadata types are never reported; they are only *consulted*, and since private
storage is invisible there, every visible instance field/property/indexer counts as evidence.

**Contagion, precisely.** Holding is structural, so a type that stores NDArrays but is not (yet)
disposable is still a holder in its owner — the owner's NDW016/NDW017 message says "make `Inner`
disposable first", and a chain resolves one level per fix (pinned by `ContagionChain_ResolvesOneLevelAtATime`).
For the per-method pass, the contagious set is narrower: an **NDArray-owning disposable** — a
disposable class/struct/ref struct that holds NDArrays and is not `[NDBorrowed]` — becomes an owned
value in `IsOwnedNDArrayType`, so `new Batch(a + b);`, a dead `var batch = MakeBatch(a);`,
`var it = np.nditer(a);` and an `Owner[]` literal all draw NDW012; `using`/`Dispose()`/`Close()`/
NumSharp's `close()` reclaim; return/store/hand-off escape. A non-disposable holder instance is *not*
an owned value (nothing the method could reclaim — the type is what warns). And `foreach` now
leaks a produced **NDArray-like or owning-disposable collection** (`foreach (var r in a + b)`,
`foreach (var x in np.nditer(a))`): C# disposes the enumerator it obtains, never the enumerable —
a produced `NDArray[]`/tuple/carrier stays conservatively consumed (a managed container whose
elements the loop body owns).

**The NDW017 path analysis** (per type, `RegisterSymbolStartAction` + one block walk per member body
+ `RegisterSymbolEndAction`): roots are the type's own `Dispose`/`Dispose(bool)`/`DisposeAsync`/
`DisposeAsyncCore` (explicit interface implementations included; a finalizer is deliberately not a
root); reachability follows same-type callees (methods, property accessors, delegates over methods;
local functions are part of the body). A body disposes holder `m` when: a `Dispose`/`DisposeAsync`/
`Close`/`close` call's receiver **roots at** `m` — through conversions, `?.`, tuple/field/property
reads (`_pair.Item1`, `_map.Values`), indexers/elements (`_list[i]`, `_arr[i]`), call results
(`_list.First()`), and **derived locals** (a local/loop variable/deconstructed name assigned from `m`,
transitively, to a fixed point); `m` is a `using` resource; `m` (or a component of it) is handed as an
argument to ANY method/constructor — gated on the argument's own type holding NDArrays, so
`Log(x.size)` hands off a scalar and `Log(_a.ToString())` a call result, neither counting; or a call on
`m` carries a lambda that disposes (`_list.ForEach(x => x.Dispose())`). Not disposal: `_a = null`,
`Clear()`, a finalizer, an unreachable helper. Messages carry the contract (`IDisposable` /
`IAsyncDisposable` / both / the `Dispose()` pattern), the member and its kind, and one of two hints:
inherited-Dispose (names the base; override `Dispose(bool)`) or make-the-inner-type-disposable-first.

**Measured on NumSharp.Core (2026-09-04, fresh rebuild):** first run 18 NDW016 types + 9 NDW017
members + 9 new NDW012 sites. Triage: 20 `[NDBorrowed]` annotations on genuine borrowers
(`FlatIterator`, `MemoryView`, `NDRefIter<T>`/`NDChunkIter<T>`, `NDEnumerate`(`<T>`), `NDFlatIterator`,
`NDExpr`'s `ArrayNode`/bind context, `NDArrayFlags`, the debugger proxy, `_Unsafe`/`_Pinning`,
`r_`'s `Operand`, `SplitContext`, `IndexOp`/`PreparedIndex`, `NpzFile.BagObj`, `NDIterRef`'s operand
slots, `NDIterator.Current` + its foreach wrapper, `NDArray.TrackingScope`, `Broadcast._ops`,
`NpzFile._cache` — the arrays it hands out are the reader's); **three real ownership gaps fixed**:
`poly1d` owned its coefficient array (yielded into it via `scope.Returns`) yet was not disposable — now
`IDisposable`, its copy-constructor copies (two owners of one array would double-dispose), its
`operator +/-/*//(poly1d, NDArray)` normalization temps and `polyval`'s Horner intermediates are
released; `Broadcast` built `broadcast_to` views it never released — `Dispose()` (what `foreach` calls)
now releases them and lazily rebuilds on the next access; `IndexCollector` stranded its outgrown
buffer on growth and on the trim path — now `IDisposable`, handing its storage out on a perfect fit;
and two unscoped entries (`np.isreal`/`np.iscomplex`) handing an `np.imag` view + a `Scalar(0)` temp to
`np.equal` are `[NDScoped]`. Core is back to **exactly 1 NDW012** (the indexer false positive), 0
NDW016, 0 NDW017.

**Limitations (pinned or documented):** a holder released through an object the type merely
*registered* it with elsewhere (`_disposables.Add(_a)` in a constructor, `_disposables.Dispose()` in
Dispose) reads as NDW017 — mark it `[NDBorrowed]` or dispose it directly; the hand-off rule is an
over-approximation in the other direction (`Console.WriteLine(_a)` counts as disposal); per-instance
ownership (a type that sometimes owns, sometimes borrows) is not expressible — annotate the common
case; a metadata type's private storage is invisible, so a library holder with only private fields is
not contagious unless it exposes an NDArray-typed member; the receiver-follow concession (§8) applies
to holder instances too (`return batch.Data.size;` saves `batch`).

## 11. Verification

- **Exact-match fixtures** (11 scenario files: 8 NDW012 + `HolderTypeScenarios.cs` (NDW016),
  `DisposePathScenarios.cs` (NDW017), `ContagionScenarios.cs` (NDW012 under contagion); ~115
  warn-tagged + ~150 clean scenarios): a missing AND an unexpected diagnostic both fail; tags live
  only in comment tails (keep bracketed `[NDWxxx]` out of prose — the marker parser scans for exactly
  that shape).
- **Metamorphic pairs**: the same body with one edit (`using`, `return`, `[NDScoped]`, sink, cast-
  dispose, alias elements/branches) must flip warn↔clean — catches "warns for the wrong reason";
  `OwnershipMetamorphicTests` does the same for types (add `IDisposable` → NDW016 becomes NDW017, add
  the dispose call → clean, `[NDBorrowed]` member/type, static, computed property, carrier,
  ref-struct pattern, contagion chain resolving one level at a time, `using` a dropped holder, `close()`).
- **Property fuzz (RB-5)**: 100 deterministic seeded bodies (3–8 statements from a 9-template grammar)
  asserted to produce EXACTLY as many NDW012 as leaky statements; `OwnershipPropertyFuzzTests` generates
  120 seeded TYPES (1–6 members from a 14-template holder/non-holder grammar, disposable or not, a
  random subset of holders disposed through a random Dispose shape) and asserts NDW016 == non-disposable
  holder types, NDW017 == undisposed holders, NDW012 == 0.
- **Robustness**: malformed source, unresolved symbols, 400-statement bodies, recursion, generics,
  the no-NumSharp no-op; for types: cycles with and without arrays, 25-deep contagion chains, generic
  definition vs instantiation, nested/partial types, 300-member types, 200 types, aliases, nullable.
- **Suite**: 216 in-process + 4 `AnalyzerBuild` tests, green net8.0 + net10.0; Core's own build is the
  live gate (§9's "1", now with NDW016/NDW017 at 0).

Compile floor: the analyzer targets **Roslyn 4.8** (the .NET 8 SDK — what consumers' compilers are
guaranteed to load). That is why collection expressions are matched by **syntax**
(`CollectionExpressionSyntax`) rather than `ICollectionExpressionOperation` (absent in 4.8): the
syntax match works on a 4.8 host (unrecognized operation) and newer hosts (typed operation) alike.

Related: `DISPOSAL-GUIDELINES.md` (the ownership rules), `docs/website-src/docs/numsharp-build-compiler.md` (the
user-facing `[NDScoped]`/NDW012/NDW013 page), `test/NumSharp.Tests.Build.Analyzer/COVERAGE_PLAN.md`
(the scenario backlog and its design guardrails — §10 there lists the do-not-regress rules).
