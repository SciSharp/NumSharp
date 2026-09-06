# NumSharp analyzer — coverage-expansion plan

A plan to harden and broaden `NumSharp.Tests.Build.Analyzer` so the **NDW012 leak analyzer**
(`NDArrayLeakAnalyzer`), the **NDW002–011 `[NDScoped]`-target gate** (`NDScopedTargetAnalyzer`), and
the **NDW013 weaver-missing** MSBuild guard are exercised across the control-flow, data-flow, type,
and configuration space — with each new case landing as a fixture tag + an in-process assertion.

> This is a planning document, not a spec. Each item is a concrete, self-contained scenario with the
> expected diagnostic and the technique to assert it. Work top-down by priority; check items off in
> §8. New scenarios go into `NumSharp.Tests.Build.Analyzer.Fixtures/` (tagged `// [NDWxxx]`) and are asserted by
> the exact-match tests, or as inline-source cases in the relevant test class.

---

## 1. Current baseline (what is already covered)

- **NDW012 leaks** (`LeakScenarios.cs`, 6): dead local, dead-local-then-scalar-return, arg-to-`np`-op
  (`np.sum(a*b)`), discarded result (`np.add(a,b);`), local-fed-to-`np`-op, constructed-and-dropped.
- **NDW012 clean** (11): returned, `using` decl, explicit `.Dispose()`, hand-scoped + `scope.Returns`,
  hand-scoped + `NDScope.Attach`, `out` param, field store, foreign-API sink, input alias, fluent view
  chain, `[NDScoped]`.
- **Carriers** (`CarrierScenarios.cs`, 4): dropped tuple, deconstruction with an unused side, dropped
  `NDArray[]`, dropped `INDArrayCarrier` struct; + clean returned tuple/array and `[NDScoped]` carrier.
- **Gate** (`GateScenarios.cs`, 7): NDW002/003/005/006/009/010/011 (NDW005 is an `extern`; the abstract
  declaration beside it is a clean contract).
- **Inheritance** (`InheritanceScenarios.cs` + `InheritanceAnalyzerTests`, `WeaverInheritanceBuildTests`):
  overrides/implementations inherit `[NDScoped]`/`[NDScopedAsync]`/`[NDScopedCovered]` (class chain incl.
  a non-declaring level, generic instantiation, implicit/explicit/base-interface implementations,
  property-level → getter) → NDW012-exempt; explicit attribute on the override wins; the gate reports a
  hidden egress on the declaration AND the override; cross-assembly declarations (compiled to a
  reference) inherit too and draw the analyzer's NDW013 only when `build_property.NumSharpBuildActive`
  is declared and not `true`; the real-build test weaves an attribute-free consumer (7 inherited
  targets, incl. an async state machine and an inherited `[NDScopedExit]`) and runs the woven code.
- **The weaver itself, in-process** (`WeaverTestHarness`: Roslyn-compile a fixture with a portable PDB →
  `ScopeWeaver.WeaveAssembly` → Mono.Cecil inspection → load + execute + assert `IsDisposed`):
  `WeaverHarnessSelfTests` (teeth: unattributed untouched, attributed woven with scope local /
  try-finally Dispose / Open / Returns and real reclamation, idempotence, hand-scoped skip, NDW002/003/
  004/009/010/014/015, Release + Debug IL, re-signing with `Open.snk`, byte-identical re-weave);
  `WeaverInheritanceTests` (the walk, shape by shape: chains through non-declaring levels, `new`
  hiding vs override, sealed/abstract overrides, explicit-wins, generic instantiations + open generic
  derived + nested generic args + overload discrimination + by-ref/pointer/params/optional signatures,
  implicit/explicit/struct/generic/default-implementation/base-interface/hiding-interface members,
  20-level interface chains, property-level vs accessor-level attributes, covariant returns, async and
  iterator overrides, inherited `[NDScopedExit]` by position with own marks winning, NDW011 not
  propagated, NDW002/006/009 at the override with provenance, cross-assembly declarations, an
  unresolvable base, nested types, idempotence, 12×40 hierarchies, the reflection `inherit:true`
  coverage invariant; pinned limitations: an interface listed on a derived class implemented by a base
  method, static abstract members); `WeaverShapeTests` (every return/egress/body shape on BOTH Release
  and Debug IL, executed, temps reclaimed, results alive; resultless tasks; out carriers; 300-local /
  long-branch bodies; 64-thread parallel execution; values under forced GCs); `WeaverAnalyzerParityTests`
  (both resolvers over the same fixtures agree per method, and each matches its own output — the scope
  local, the NDW012 placement); `WeaverHierarchyFuzzTests` (40 seeded random chains + interface vs a
  reference model, weaver AND analyzer); `WeaverCliTests` (the tool PROCESS: exit codes, rsp, `--snk`,
  `--verbose`, NDW001 — unreachable in-process because Cecil resolves through the host's
  TRUSTED_PLATFORM_ASSEMBLIES).
- **Contract**: NDW012 severity == Warning + enabled-by-default; a real leak is a Warning not an Error;
  gate diagnostics are Errors (`AnalyzerContractTests`).
- **Harness**: teeth (real leak warns / clean is silent / broken source reported / `[NDScoped]` exempt),
  non-vacuity floors, per-gate-code presence, NDW013 build test (`AnalyzerBuild`).

**Not yet covered:** almost all of the control-flow, escape-nuance, scope-nuance, type-system, and
configuration space below.

---

## 2. Techniques (the tools this plan uses)

| # | Technique | Where | Use for |
|---|-----------|-------|---------|
| T1 | **Marker fixtures + exact-match** | `Fixtures/*.cs` + `AssertExactAsync` | the default — every "must warn / must stay clean" scenario, asserted to the line |
| T2 | **Inline-source assertions** | `HarnessSelfTests`-style `RunAsync(src,…)` | quick targeted single cases, severity, counts |
| T3 | **Metamorphic pairs** | new `MetamorphicTests` | prove the DELTA: identical body, one variant adds `using`/`return`/`[NDScoped]` → flips warn↔clean. Catches "flags for the wrong reason" |
| T4 | **Weaver-parity cross-check** | new `WeaverParityTests` | a shape the weaver would successfully weave must be gate-clean; a shape it rejects must be gated. Reflect over `ScopeWeaver.Classify` vs `TypeHelpers` |
| T5 | **Severity / config** | `AnalyzerContractTests` + editorconfig | NDW012 stays a warning; editorconfig `severity=none/suggestion/error` changes output; `WarningsNotAsErrors` |
| T6 | **Robustness / fuzz** | new `RobustnessTests` | malformed/partial code, huge bodies, generated method bodies → analyzer must not crash + invariants hold (returned ⇒ never flagged) |
| T7 | **Adversarial known-limitation pins** | `[TestCategory("Misaligned")]` | pin the CURRENT (imperfect) behavior of view-vs-fresh and ambient-scope so a change is noticed, not silent |
| T8 | **NDW013 MSBuild matrix** | `Ndw013BuildTests` | multi-TFM, `SkipNDScopeWeave`, `ExcludeAssets`, token-in-string, suppression knobs |

---

## 3. NDW012 leak-analyzer coverage gaps

### 3.1 Control flow & data flow  *(P0 — highest value; the analyzer is a dataflow pass)*

| id | scenario | expected | why / technique |
|----|----------|----------|-----------------|
| CF-1 | reassignment orphans the first value: `var t = a + b; t = c - d; return t;` | **warn on `a + b`** (orphaned), clean on `c - d` (returned via `t`) | the first owning value is overwritten, never reclaimed. Verifies per-assignment tracking, not just per-local. T1 |
| CF-2 | conditional assignment, both owning, dropped: `NDArray t; if (p) t = a + b; else t = c - d;` (t unused) | **warn** (t leaks) | T1 |
| CF-3 | conditional assignment, one branch aliases input: `NDArray t; if (p) t = a + b; else t = a; return t;` | **clean** (returned; and the alias branch must not make it R2-unsafe) | mixed owning/alias → currently dropped from owning set (conservative). Pin the CURRENT behavior. T1/T7 |
| CF-4 | ternary owning, dropped: `var t = p ? a + b : c - d;` (unused) | decide + pin (tuple-literal-style path may currently MISS this) | T1/T7 — likely a **gap to fix** (ternary of temps) |
| CF-5 | loop-created temp: `for (int i=0;i<n;i++){ var t = a + b; }` | **warn** (leaks each iteration) | T1 |
| CF-6 | accumulator reassign: `NDArray acc = a; foreach(var x in xs) acc = acc + x; return acc;` | pin: seed is input (alias, clean); the intermediate `acc + x` values are orphaned each iter | the classic reduction pattern; the intermediates leak but `acc` seed is R2-safe. Decide warn-on-intermediate vs conservative-clean, then pin. T1/T7 |
| CF-7 | `using` STATEMENT (not declaration): `using (var t = a + b) { … }` | **clean** | `IUsingOperation` resource path (distinct from `using var`). T1 |
| CF-8 | `using` on a bare owning expr: `using (a + b) { }` | **clean** | no local; the resource IS the owning value | T1 |
| CF-9 | try/finally dispose: `var t = a+b; try { … } finally { t.Dispose(); }` | **clean** | T1 |
| CF-10 | explicit discard: `_ = a + b;` | decide + pin (currently the discard target may read as an assignment → clean, but it IS a leak) | likely a **gap** — an explicit discard drops the buffer. T2/T7 |
| CF-11 | `switch` expression producing temps: `var t = x switch { 0 => a+b, _ => c-d };` (unused) | pin | T1/T7 |
| CF-12 | thrown-before-use: `var t = a+b; throw new Exception(); ` (unreachable use) | **warn** (still constructed, dropped) | T2 |

### 3.2 Escape / consume nuances  *(P1)*

| id | scenario | expected | why / technique |
|----|----------|----------|-----------------|
| EC-1 | `ref`/`out` argument to another method: `Foo(ref t)` / `Foo(out t)` where `t` owning | **clean** (write-back egress) | T1 |
| EC-2 | `in` parameter: `Foo(in t)` | **clean** (still handed off) | T1 |
| EC-3 | `params` array element: `Bar(a+b, c-d)` where `Bar(params NDArray[])` | **clean** (handed to a consuming API) | T1 |
| EC-4 | collection add: `list.Add(a+b);` / `new List<NDArray>{ a+b }` | **clean** (stored) | T1 |
| EC-5 | dictionary/indexer store: `map[k] = a+b;` | **clean** | T1 |
| EC-6 | lambda capture: `Action f = () => Use(t);` (t owning) | **clean** (captured — can't track) | T1 |
| EC-7 | returned from a local function / lambda: `NDArray L() => a+b;` | **clean** inside L (returned) | separate operation block. T1 |
| EC-8 | yielded: `yield return a + b;` in an iterator | **clean** (consumer owns it) | needs the enclosing method to be a non-scoped iterator. T1 |
| EC-9 | awaited: `var v = await ComputeAsync(a+b);` | pin (arg to a non-NumSharp async method ⇒ clean) | T1 |
| EC-10 | boxed/observed: `Console.WriteLine(a+b);` / `$"{a+b}"` | **clean** (returns non-NDArray ⇒ observed, not owned) | avoids logging FPs. T1 |
| EC-11 | tuple element returned: `return (a+b, c-d);` | **clean** (both escape via the returned tuple) | T1 (already implicit in CarrierScenarios helpers — make explicit) |
| EC-12 | stored to a `ref` local: `ref var r = ref field; r = a+b;` | pin | rare. T2/T7 |
| EC-13 | passed to a **NumSharp view function** `np.reshape(a+b, s)` returned | **known limitation**: currently WARNS on `a+b` (view of the temp, buffer kept alive by the returned view) | document as accepted FP (fresh-vs-view undecidable). T7 |

### 3.3 Owning-expression detection  *(P1)*

| id | scenario | expected | why / technique |
|----|----------|----------|-----------------|
| OW-1 | property view NOT a candidate: `var t = a.T;` (unused) | **clean** (properties/`.T`/`.flat` are excluded to avoid view noise) | pin the exclusion. T1 |
| OW-2 | indexer view: `var t = a["1:3"];` / `a[mask]` (unused) | pin (receiver-followed / excluded) | T1/T7 |
| OW-3 | chained operators, multiple temps: `return a + b + c;` | **warn on the inner `a + b`** (one temp), clean on the outer (returned) | T1 |
| OW-4 | `NDArray.Scalar(x)` dropped | **warn** | a constructed owned scalar. T2 |
| OW-5 | generic `NDArray<T>` result dropped: `var t = (a & b).MakeGeneric<bool>();` (unused) | **warn** (the MakeGeneric wrapper is dropped) | T2 |
| OW-6 | compound assign on an input: `a += b;` (a is a parameter) | **clean** (a is the caller's; the new value is stored back into a) | pin — must NOT warn (R2). T1/T7 |
| OW-7 | implicit scalar conversion: `NDArray t = 5;` dropped | pin (is `5` → NDArray owned here?) | probe + pin. T2/T7 |

### 3.4 Scope / exemption nuances  *(P1)*

| id | scenario | expected | why / technique |
|----|----------|----------|-----------------|
| SC-1 | `[NDScoped]` on a **property getter** with a leaking body | **clean** (exempt) | accessor exemption via `AssociatedSymbol`. T1 |
| SC-2 | hand-scope via `using (NDScope.Open()) { … }` STATEMENT (not `using var`) | **clean** (exempt) | T1 |
| SC-3 | `NDScope.Open()` in a NESTED block only, leak OUTSIDE it | pin: currently exempts the WHOLE method | document (matches the weaver's whole-method skip). T7 |
| SC-4 | **local function** inside a non-scoped method that leaks | **warn** inside the local function (own operation block) | verify local functions are analyzed. T1 |
| SC-5 | **lambda** body that leaks inside a scoped method | pin (lambda runs later; ambient scope may be gone) | probe + pin. T2/T7 |
| SC-6 | `[NDScopedAsync]` async method that leaks a temp | **clean** (exempt) | T1 |
| SC-7 | `NDScope.Detach(t)` then drop | pin (detached ⇒ intentionally un-tracked) | T2 |

### 3.5 The ambient-scope over-report  *(P2 — document, do not "fix")*

| id | scenario | expected | technique |
|----|----------|----------|-----------|
| AM-1 | a private helper that builds temps, called only from a `[NDScoped]` boundary | **warns** (per-method analysis cannot see ambient coverage) | pin as `[TestCategory("Misaligned")]`; this is the documented limitation. T7 |
| AM-2 | the same helper marked `[NDScoped]` | clean | shows the fix. T1 |

---

## 4. Gate analyzer (NDW002–011) coverage gaps  *(P1)*

| id | scenario | expected |
|----|----------|----------|
| GT-1 | `ref NDArray[]` parameter | NDW002 |
| GT-2 | `in NDArray` parameter | NDW002 (an `in` NDArray-carrying is a hidden egress the same way `ref` is) — **probe** the current behavior first |
| GT-3 | supported carriers must be CLEAN: `(NDArray,NDArray,NDArray)` (arity 3), a 5-tuple (ITuple path), `IArraySlice`/`UnmanagedStorage`, `NDArray<T>` | **no gate error** (these ARE woven) — the negative gate cases, currently absent |
| GT-4 | `Task<NDArray[]>` / `ValueTask<NDArray>` / `ValueTask` under `[NDScopedAsync]` | clean (supported task shapes) |
| GT-5 | `IAsyncEnumerable<NDArray>` under `[NDScopedAsync]` | clean (async iterator) |
| GT-6 | `IEnumerable<NDArray>` iterator under `[NDScoped]` | clean (sync iterator — must NOT be NDW010) |
| GT-7 | `Task<List<NDArray>>` under `[NDScopedAsync]` | NDW003 (unsupported carrier inside the task) |
| GT-8 | `[AsyncMethodBuilder]` custom task-like under `[NDScopedAsync]` | clean (async method) |
| GT-9 | attribute on a **local function** / explicit-interface method | probe + pin |
| GT-10 | expression-bodied property getter with the attribute + unsupported return | NDW003 |

> **Technique note (T4):** the gate is a *mirror* of `ScopeWeaver.Classify`. Add a reflection test that,
> for a curated set of return types, asserts `TypeHelpers.IsSupportedCarrier` agrees with the weaver's
> `Classify != null` — so the two never drift.

---

## 5. Type-system & resolution robustness  *(P2)*

| id | scenario | expected |
|----|----------|----------|
| TY-1 | a **consumer subclass** of `NDArray` (`class MyND : NDArray`) leaked | warn (subclass is NDArray-like) |
| TY-2 | a **consumer's own** `INDArrayCarrier` struct dropped | warn |
| TY-3 | nullable annotations: `NDArray? t = a+b;` dropped | warn |
| TY-4 | aliased using: `using ND = NumSharp.NDArray;` then `var t = a+b;` dropped | warn (resolution by symbol, not name) |
| TY-5 | fully-qualified `NumSharp.np.sum(a*b)` returned | warn on `a*b` |
| TY-6 | NumSharp **not referenced** at all | analyzer no-ops (KnownTypes null) — no crash, no diagnostics |
| TY-7 | `global using NumSharp;` (implicit usings) | resolves the same |

---

## 6. Robustness & non-crash  *(P2)*  — technique T6

- RB-1 malformed / incomplete method body (missing `}`, unresolved symbols) → **no analyzer crash**, compile errors reported by the harness, no bogus NDW.
- RB-2 a very large generated method (e.g. 2,000 statements building/returning temps) → completes, no crash.
- RB-3 recursive local functions → no infinite walk (the `FlowOf` 64-step guard).
- RB-4 generic host type / generic method → resolves.
- RB-5 a **property-based generator**: emit N random small method bodies from a grammar of {create, return, dispose, pass-to-op, store} and assert the invariants *"a value that is returned/out/stored/disposed is never NDW012"* and *"a value only ever read and dropped IS NDW012"*. Shrinks a violation to the minimal body. (Mirrors the differential-fuzz philosophy in `test/NumSharp.Tests.Oracle`.)

---

## 7. NDW013 (MSBuild guard) matrix  *(P1)*  — technique T8, all `[TestCategory("AnalyzerBuild")]`

> **2026-09-05 — the analyzer is now the PRIMARY NDW013 reporter** (per member, in-process:
> `Ndw013AnalyzerTests` — own `[NDScoped]`/`[NDScopedAsync]`, inherited from this or another assembly,
> `[NDScopedExit]` own/inherited/setter/constructor, contracts, rejections, the knobs). The MSBuild text
> scan below is the FALLBACK for a compile without the analyzer (a bare `<Reference>` to the assembly, or
> an analyzer item removed by hand — NOT `ExcludeAssets="analyzers"`, which the SDK does not honour for a
> package's analyzers, observed on 10.0.101) and yields when `NumSharp.Build.Analyzer` is in `@(Analyzer)`
> — pinned by `Ndw013BuildTests.Ndw013_ComesFromTheAnalyzer_WhenItIsApplied_AndTheScanYields` and
> `verify_build_package.sh` step 17/17b. The rows below describe the fallback; N13-5's string-literal
> false positive is the scan's alone.

- N13-1 **multi-TFM** consumer (`net8.0;net10.0`) → warns once per TFM.
- N13-2 `-p:SkipNDScopeWeave=true` **with** the weaver installed → **fires** (weaving disabled ⇒ attributes inert).
- N13-3 `-p:NumSharpDisableWeaverMissingWarning=true` → silent.
- N13-4 `NDW013` in `$(MSBuildWarningsAsMessages)` → downgraded to a message.
- N13-5 the token `NDScoped` appears only in a **string literal** (no real attribute) → **known false-positive** of the ASCII scan; pin as `[Misaligned]`.
- N13-6 a project with the weaver **and** `[NDScoped]` → silent (the common healthy case).
- N13-7 `[NDScopedAsync]`-only consumer without the weaver → fires (the shared-prefix scan catches it).

---

## 8. Prioritized backlog & tracking

**P0 (land first — core dataflow correctness):** CF-1, CF-4, CF-5, CF-6, CF-10, plus the metamorphic
harness (T3). These probe the parts of `FlowOf`/local-tracking most likely to hide a real bug (CF-1/CF-4/
CF-10 are suspected gaps — write the fixture, watch it fail, decide fix-vs-pin).

**P1:** §3.2 escape nuances, §3.4 scope nuances, §4 gate negatives (GT-3/4/5/6 especially — the analyzer
has **no** "supported shape stays clean" gate cases today), §7 NDW013 matrix, T4 weaver-parity.

**P2:** §3.5 ambient-scope pins, §5 type-system, §6 robustness/fuzz.

| Tier | Items | Status |
|------|-------|--------|
| P0 | CF-2, CF-5, CF-10, CF-12 (warn), CF-3, CF-6, CF-7, CF-8, CF-9 (clean), MetamorphicTests (T3) | ☑ landed |
| P0 (pinned) | CF-1, CF-4, CF-11 (documented false negatives, `KnownLimitationScenarios` under Misaligned) | ☑ pinned |
| P1 | EC-1…EC-12, OW-1…OW-7, SC-1…SC-4/6/7, GT-1…GT-7/GT-9/GT-10, WeaverParityTests (T4) | ☑ landed |
| P1 (pinned) | EC-13, SC-5 (Misaligned) | ☑ pinned |
| P1 (deferred) | GT-8 (custom `[AsyncMethodBuilder]` — weaver-internals verification needed; the analyzer would gate it NDW003, matching `IsTaskLike`'s 4-shape contract), N13-1/N13-4/N13-5 (multi-TFM / WarningsAsMessages / token-in-string) | ☐ |
| P2 | AM-1/AM-2, TY-1…TY-7, RB-1…RB-4 + the returned/disposed invariant sweep | ☑ landed |
| P2 (deferred) | RB-5 (full property-based generator + shrinker — the invariant DataRows cover its spirit) | ☐ |

**Landed this pass (fixtures `ControlFlow`/`Escape`/`Owning`/`Scope`/`GateNegative`/`KnownLimitation`
Scenarios.cs; tests `ControlFlow`/`EscapeOwning`/`Scope`/`GateNegative`/`Metamorphic`/`Robustness`/
`TypeSystem`/`KnownLimitation`Tests.cs + `FixtureFacts` + `RunWithoutNumSharpAsync` + N13-3/N13-7 build
tests):** 101 in-process + 4 `AnalyzerBuild` tests, green net8.0 + net10.0. **One analyzer FIX shipped
— CF-10:** an explicit discard `_ = a + b;` now reads as a leak (a discard drops the buffer; it is not a
store), zero false positive on `_ = a;` (a bare input owns nothing). Every scenario was probe-verified
against the analyzer's actual behavior BEFORE tagging (§9 step 1), so each fixture pins reality, not a
guess.

**Second pass — escape/store edges + composite producers (113 + 4 tests green net8.0/net10.0):**
- **Store escapes probed clean and pinned** (`EscapeScenarios` + `EveryStoreEgress_IsClean`): static /
  instance property setter, instance field on another object, object-initializer assignment, array
  element, NumSharp's own indexer-set (`a["1:3"] = b + c`), compound-assign into a field, base-ctor
  argument, void NumSharp op (`np.copyto`), local-function argument, string interpolation,
  conditional-access receiver.
- **Analyzer FIX — composite producers** (`ProducesOwnedNDArray`): a **tuple literal / array creation**
  is a producer iff AT LEAST ONE element is itself a producer (it holds all elements at once; a
  tuple/array of plain inputs owns nothing — R2). A **conditional value** (ternary / switch expression /
  coalesce) is a producer iff EVERY branch produces (it holds only one branch; a single aliasing branch
  makes reclaiming unsafe — the same conservatism as the CF-3 if/else rule). This closed CF-4 and CF-11
  (moved out of the Misaligned pins into tagged warns), caught the dropped/mixed/nested tuple-literal,
  dropped array-literal, deconstructed-literal-unused-side, discarded-ternary and both-owning-coalesce
  cases, and kept every alias/escape twin silent (probe-verified, 34 scenarios).
- **The fix found two REAL leaks in Core** (proof of value): `np.setops.CanonicalizeSetOpNaN`'s `canon`
  scalar (ternary of two `NDArray.Scalar` producers handed to `np.where`) — ambient-covered by the
  `[NDScoped]` set-op entries, marked `[NDScopedCovered]`; and `np.interp.PreparePeriodic`'s two
  wrap-extend temps (`xpLast - period`, `xpFirst + period` inside the `np.concatenate(new[]{…})`
  argument) — genuinely stranded (interp's entries are unscoped), fixed with a `using` declaration.
  Core is back to exactly ONE NDW012 (the documented indexer FP); interp/setops unit tests stay green.
- **Remaining pinned FN:** the MIXED coalesce `x ?? (a + b)` (leaks only when `x` is null — the
  every-branch rule's deliberate residue, pinned in `KnownLimitationScenarios`).

**Third pass — body kinds, exotic reclaim spellings, RB-5 (125 + 4 tests green net8.0/net10.0):**
- **Body-kind coverage** (`BodyKindScenarios` + `BodyKindTests`): async methods, iterators,
  instance/static constructors, finalizers, expression-bodied members, unscoped property getters, and
  user-defined operators are all analyzed and warn on a leak; a lambda RETURNING its temp is a
  legitimate escape (its consumer owns the value).
- **Four more analyzer FIXES**, each probe-verified with its negative twin:
  1. `IsDisposeCall` no longer requires an owned containing type — `((IDisposable)t).Dispose()` was a
     FALSE POSITIVE (the receiver walk only ever arrives from an owned value, so any parameterless
     `Dispose` on it disposes the same buffer).
  2. **C# 12 collection expressions** (`NDArray[] g = [a + b];`) join the any-element producer rule —
     matched by SYNTAX (`CollectionExpressionSyntax`), because the analyzer compiles against Roslyn 4.8
     (the .NET 8 SDK floor) which has the language feature but not `ICollectionExpressionOperation`;
     the syntax match works on both a 4.8 host (None-operation) and newer hosts (typed operation).
  3. **Deconstruction into EXISTING locals** (`(q, r) = Split(a);`) — the target mentions counted as
     USES, masking the unused side's leak; `IsDeclarationReference` now treats a local-ref inside a
     deconstruction-assignment TARGET as a write.
  4. **`throw` branches are vacuous** for the every-branch conditional rule (`p ? a+b : throw …` owns
     its result on every path that yields one); an alias completing branch still wins the conservatism.
- **Reclaim-spelling pins:** `t?.Dispose()`, cast-dispose, catch-only dispose (any single reclaiming
  use saves the local — deliberately path-insensitive), `using` on a cast bare expression, dispose
  inside a LOCAL FUNCTION (part of the outer operation tree). Compound-assign local pair (dropped
  warns / returned clean). Temp-array-into-`np.concatenate` warns; alias-array + foreach-consumption
  stay clean.
- **RB-5 LANDED** (`PropertyFuzzTests`): a deterministic seeded generator (100 seeds × 3–8 statements
  from a 9-template grammar) asserts NDW012 count == the exact number of leaky statements per body —
  the "escapes are never flagged / drops always are" invariants under arbitrary interleaving, with
  per-seed replay and body dump on failure.
- **New pinned FNs:** `out var` results (callee-fresh vs alias is per-method-undecidable) and
  declaration-pattern locals (`expr is NDArray t` — rare, not worth the plumbing).

**Fourth pass — the RUNTIME/weaver side of §9's promise (the egress gap; 153 + 4 tests green
net8.0/net10.0):** this pass fixed NDScope/ScopeWeaver, not the leak analyzer — probing showed the
analyzer's carrier model ("a tuple of NDArrays, any nesting" = owned) was AHEAD of what a scope
actually yielded, so a gate-clean `[NDScoped]` method could hand back DISPOSED arrays:
- **Runtime fix** (`NDScope.cs`): `Returns(ITuple)` and `Detach(ITuple)` dispatch each component by
  RUNTIME type (nested tuples recursively, `NDArray[]` elements, `INDArrayCarrier` structs, bare
  buffers) — one fix covering sync returns, async `SetResult`, yielded elements, task-carried
  results (`YieldBoxed`) and `[NDScopedExit]` params. Probe: 11 live GAP reproductions → all green.
- **Weaver fix** (`ScopeWeaver.cs`): the `out`-param escape now emits the full carrier vocabulary
  (tuple → box+`Returns(ITuple)`, carrier → `constrained. YieldTo` through the byref, buffer →
  counted ref); **NDW015** (new) gates an ND-carrying `out` shape it cannot yield; **NDW002**
  widened to `ref`/`in` over ANY `CarriesNDArray` shape; a general tuple return validates its
  components (a `List<NDArray>`/`Task<NDArray>`/`T : NDArray` component → NDW003) and a nested
  `Task<Task<NDArray>>` is refused; `ClassifyExitParam` applies the same component rule (NDW014).
- **Gate-analyzer mirror** (T4): `TypeHelpers.CarriesNDArray`/`IsSupportedOutCarrier` + recursive
  tuple-component validation in `IsSupportedCarrier`; NDW015 descriptor; NDW002 widened; a custom
  `[AsyncMethodBuilder]` task-like async method is now DEFERRED to the weaver instead of the old
  NDW003 false positive (GT-8 closed — the weaver validates the SM's real result type at IL time).
- **New tests:** `GateNegativeTests` +28 (tuple-component/task-nesting DataRows, the
  `RefOutParamParity` matrix, `OutGenericConstrainedToNDArray_IsGated`,
  `CustomTaskLikeAsyncMethod_IsNotGated`); `NDScopeWeaveCarrierTests` +4 recursive-ITuple pins;
  `NDScopeExitTests` +2 recursive-detach pins; `NDScopeAsyncTests` +1 task-carried nested tuple.
  Verified end-to-end on live-woven probe assemblies (16 runtime checks + a 12-shape gate matrix,
  ILVerify clean on every woven subject); `[NDScoped]` on LOCAL FUNCTIONS confirmed already woven
  (GT-9's "deferred to the weaver" is genuinely handled — the generated method carries the
  attribute and `CollectTargets` finds it).

---

## 9. Workflow for adding a scenario

1. **Probe first.** For any "decide + pin" item, write the fixture line (or an inline `RunAsync` case)
   and RUN it — the analyzer's *current* behavior tells you whether it is a gap to fix or behavior to
   pin. (Writing `CarrierScenarios.DeconstructedOneSideLeaks` this way already found and fixed a real
   bug — the deconstruction declaration-reference-as-use masking.)
2. **Fix or pin.** A genuine miss/false-positive → fix `NDArrayLeakAnalyzer` (add the closure/flow case)
   and keep the fixture green. An intractable case (fresh-vs-view, ambient-scope) → keep the fixture but
   move the assertion to `[TestCategory("Misaligned")]` and document *why* here.
3. **Tag + assert.** Add `// [NDWxxx]` on the expected line (`AssertExactAsync` gates it exactly), or an
   inline assertion in the right test class. Keep clean-scenario counts in the non-vacuity floors.
4. **Never let a doc tag masquerade as a marker.** Keep bracketed `[NDWxxx]` out of prose comments (the
   marker parser scans the comment tail for exactly that shape).

---

## 10. Design guardrails (do not regress)

- **NDW012 is a WARNING, never an Error** (pinned by `AnalyzerContractTests`; `WarningsNotAsErrors` on
  Core). A leak must never break a build.
- **Receiver of a call/property/indexer FOLLOWS the result upward** — do not revert to "receiver ⇒ leak"
  (it floods NumSharp's `MakeGeneric`/`reshape` view chains; see the 411→213 note in the analyzer's
  memory).
- **The gate mirrors the weaver's `Classify`** — any new supported/rejected shape must be reflected in
  BOTH `ScopeWeaver` and `TypeHelpers`, guarded by the T4 parity test.
- **The analysis is per-method by design.** Ambient-scope coverage is invisible to it; that is a
  documented limitation, not a bug to chase with heuristics that risk false negatives.
- **Composite-producer rules are asymmetric ON PURPOSE.** A tuple/array literal produces if ANY element
  produces (it holds all elements simultaneously — a produced element is definitely in the dropped
  value); a conditional (ternary/switch/coalesce) produces only if EVERY branch produces (it holds one
  branch — a possibly-aliasing branch makes reclaiming unsafe, mirroring CF-3). Do not "simplify" the
  two to one rule: any-element on conditionals flags mixed ternaries (R2 false positives), and
  every-element on tuples goes silent on `(temp, alias)` drops (real leaks — `np.setops` had one).

---

## 11. The ownership pass — NDW016 / NDW017 and contagion (LANDED 2026-09-04)

`NDArrayHolderAnalyzer` (+ the shared `OwnershipModel`) closes the hand-off NDW012's "store" escape
opens: a TYPE that stores NDArrays must be disposable (NDW016) and must dispose every holder on its
Dispose path (NDW017); an NDArray-owning disposable is itself an owned value for NDW012 (contagion),
and a `foreach` over a produced NDArray / owning disposable leaks it. Opt-out: `[NDBorrowed]` on a
member or a type. Full semantics: `docs/LEAK_ANALYZER.md` §10.

| family | fixture / test | what it pins |
|---|---|---|
| holder vocabulary (warn) | `HolderTypeScenarios.cs` (25+ tagged types) | field, auto/init property, positional record, arrays (any rank), List/Dictionary/nested generic, tuple, carrier field, Lazy/Task, NDArray<T>, `T : NDArray`, consumer `Box<NDArray>`, struct, ref struct, abstract, many-members message, contagious disposable member + list of them, non-disposable holder member, `np.NDIterator`, `NpzFile`, cyclic pair with an array |
| exemptions (clean) | same file | static-only, static+scalar, computed property, `[NDBorrowed]` member/type, delegates, comparers, weak refs, observers/progress, unconstrained generic, object/IDisposable, carrier struct, disposing IDisposable/IAsyncDisposable, ref-struct Dispose pattern, `[NDBorrowed]` disposable + its holder, NumSharp's `FlatIterator` (metadata attribute), scalars-only, derived-adds-nothing, self-reference, cycle without arrays, interface/enum/delegate, generic definition |
| dispose paths (warn) | `DisposePathScenarios.cs` (14 tagged members) | forgets one of two, `= null`, `Clear()`, foreach-read-only, Reset-only, unreachable helper, finalizer-only, `Log(_a.ToString())`, inherited Dispose without override, contagious member forgotten, non-disposable inner (hint), auto-property, `DisposeAsync` forgets, tuple sibling, struct |
| dispose spellings (clean) | same file (40+ types) | direct/conditional/this/cast/as-cast/guarded/try-finally, Dispose(bool) pattern, explicit interface, helper / helper chain / static-helper arg / external-helper arg / property setter / delegate / local function, local copy, `Interlocked.Exchange`, `using` statement/declaration, override of base Dispose(bool), array foreach/for/`Array.ForEach`, list foreach/`ForEach`/indexer/`Parallel.ForEach`, dictionary values/pairs/deconstructed/LINQ, nested foreach, tuple components/deconstruction, `Lazy.Value`, carrier field, auto-property, contagious member + list, `close()`, async + `DisposeAsyncCore` pattern, struct, ref struct, record, partial across parts, `[NDBorrowed]` sibling, nested type |
| contagion (NDW012) | `ContagionScenarios.cs` (9 tagged) | dropped/dead/factory/discarded holder instance, holder array literal, `np.nditer` dropped / walked by foreach, produced NDArray walked by foreach, temp handed to `np.nditer`; clean: using/dispose/return/store/sink/using-statement/`close()`/using-then-foreach, `[NDBorrowed]` disposable, non-disposable wrapper, NDArray[] foreach, alias foreach, `[NDScoped]` twin |
| metamorphic | `OwnershipMetamorphicTests` (19 pairs) | IDisposable turns 016→017, dispose call →clean, IAsyncDisposable, `[NDBorrowed]` member/type, static, computed, carrier, ref-struct pattern, chain resolving one level at a time, contagion stopping at `[NDBorrowed]`, `using` a dropped holder, holder non-disposable moves the verdict to the type, `close()`, foreach-over-produced until `using`, helper reachability, override of inherited Dispose, foreach-dispose of a collection |
| property fuzz | `OwnershipPropertyFuzzTests` (120 seeded types) | NDW016 == non-disposable holder types, NDW017 == undisposed holders, NDW012 == 0, over a 14-template grammar × 3 Dispose shapes |
| robustness | `OwnershipRobustnessTests` (19) | cycles ± arrays, self-reference, 25-deep chain, disposing chain, generic definition vs `Box<NDArray>`/`Box<int>`, alias, nullable, nested, partial, 300 members, 200 types, static/interface/enum/delegate, no-NumSharp no-op, malformed, unresolved member type, arrays of holders, `Cell<NDArray>`, other-instance helper |
| contract | `AnalyzerContractTests` | NDW016/NDW017 are Warnings, enabled, anchored help links; the three analyzers own disjoint ids |

**Guardrails added by this pass:**
- **The hand-off rule is type-gated.** An argument counts as a hand-off of a holder only when the
  argument's OWN type holds NDArrays and it is not a call result — `Log(x.size)` on a loop element and
  `Log(_a.ToString())` are not disposals. Dropping the gate silently cleared `ForeachWithoutDispose`.
- **Only root `HoldsNDArrays` answers are memoized when a cycle cut them.** A nested answer computed
  while its type was on the visiting stack may be inexact; memoizing it turned a real holder clean.
- **Metadata types are consulted, never reported**, and only from assemblies that can name NDArray.
- **`INDArrayCarrier` types are exempt from NDW016/NDW017 on purpose** — they are the transient result
  vocabulary the scope yields through; demanding IDisposable there contradicts the carrier contract.
