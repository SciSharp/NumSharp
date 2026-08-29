# NDW012 — the NDArray leak analyzer: supported cases & what NDScope handles automatically

`NDArrayLeakAnalyzer` (`tools/NumSharp.Build.Analyzer/`) is a Roslyn flow analysis that reports a
**warning** — NDW012, never an error — for an NDArray a method **creates but never reclaims or hands
off**, so its pooled unmanaged buffer is left to the finalizer instead of being returned promptly (the
exact cost `[NDScoped]` exists to remove; see `DISPOSAL-GUIDELINES.md` and the website's
`ndscoped.html#ndw012`). It ships **inside the NumSharp package** (`analyzers/dotnet/cs/`, packed by
`_NumSharpCorePackAnalyzer`), so every `PackageReference` consumer gets it with no extra install; it
also runs on NumSharp.Core's own build (toggle: `-p:EnableNDArrayLeakAnalyzer=false`; Core adds NDW012
to `$(WarningsNotAsErrors)` so a leak can never break a build even under `TreatWarningsAsErrors`).

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
| **stores** | static/instance **property** setter, static/instance **field**, object-initializer `new W { Prop = a + b }`, array element `arr[0] = …`, dictionary/indexer `map[k] = …`, **NumSharp's own indexer-set** `a["1:3"] = b + c`, `ref`-local store, compound into a field `_f += a`, deconstruction into fields |
| ctor args | `new Wrapper(a + b)` (assumed to take ownership), `: base(a + b)` |
| foreign calls | any argument to a **non-NumSharp** method (assumed to consume/observe), incl. `params` elements, collection `Add`, local-function args; a **void** NumSharp op too (`np.copyto(dst, a + b)` — the non-consuming rule keys on the RETURN type) |
| observation | `Console.WriteLine(a + b)`, string interpolation `$"{a + b}"` (result isn't owned) |
| closures / delegates | lambda capture (`Action f = () => Use(t);`), a lambda RETURNING its temp (`Func<NDArray> f = () => a + b;`) |
| views | the receiver of a call/property/indexer **follows the result upward**: `(a + b).reshape(1, -1)` returned, `(a + b)?.reshape(…)`, `x.MakeGeneric<T>()` chains — the view shares the buffer, so if the result escapes the receiver is not a leak |
| consumption | `foreach (var x in Split(a))` — the loop owns each element's fate (conservative) |

## 5. The reclaims (all clean — deterministically disposed)

| Spelling | Notes |
|---|---|
| `using var t = a + b;` | using declaration |
| `using (var t = a + b) { }` | using statement |
| `using (a + b) { }` / `using ((IDisposable)(a + b)) { }` | bare owning expression, cast included |
| `t.Dispose();` | anywhere — including `try/finally`, **catch-only** (path-insensitive by design), and inside a **local function** (local-function bodies are part of the outer method's operation tree) |
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

## 10. Verification

- **Exact-match fixtures** (8 NDW012 scenario files, ~44 warn-tagged + ~75 clean scenarios): a missing
  AND an unexpected diagnostic both fail; tags live only in comment tails (keep bracketed `[NDWxxx]`
  out of prose — the marker parser scans for exactly that shape).
- **Metamorphic pairs**: the same body with one edit (`using`, `return`, `[NDScoped]`, sink, cast-
  dispose, alias elements/branches) must flip warn↔clean — catches "warns for the wrong reason".
- **Property fuzz (RB-5)**: 100 deterministic seeded bodies (3–8 statements from a 9-template grammar)
  asserted to produce EXACTLY as many NDW012 as leaky statements.
- **Robustness**: malformed source, unresolved symbols, 400-statement bodies, recursion, generics,
  the no-NumSharp no-op.
- **Suite**: 125 in-process + 4 `AnalyzerBuild` tests, green net8.0 + net10.0; Core's own build is the
  live gate (§9's "1").

Compile floor: the analyzer targets **Roslyn 4.8** (the .NET 8 SDK — what consumers' compilers are
guaranteed to load). That is why collection expressions are matched by **syntax**
(`CollectionExpressionSyntax`) rather than `ICollectionExpressionOperation` (absent in 4.8): the
syntax match works on a 4.8 host (unrecognized operation) and newer hosts (typed operation) alike.

Related: `DISPOSAL-GUIDELINES.md` (the ownership rules), `docs/website-src/docs/ndscoped.md` (the
user-facing `[NDScoped]`/NDW012/NDW013 page), `test/NumSharp.Tests.Build.Analyzer/COVERAGE_PLAN.md`
(the scenario backlog and its design guardrails — §10 there lists the do-not-regress rules).
