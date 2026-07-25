---
name: np-function-new
description: >-
  The house workflow for implementing a NumPy np.* function in NumSharp with full NumPy 2.4.2
  parity — probe-first behavior matrix, kernel/NDIter integration rules, the all-15-dtype + layout
  DOD, then the oracle fuzz gate, benchmark, docs and commit. Use this whenever the user asks to
  implement, add, port, complete, audit, or fix an np.* function ("implement np.foo", "add np.bar",
  "port numpy's baz", "np.qux is missing / NotImplemented", "make <fn> match numpy", "add
  out=/where=/axis= to <fn>"), even when they don't say "parity" — ALL np.* API work goes through
  this workflow.
argument-hint: <np.function_name or description>
---

# np.* Implementation Workflow (NumPy 2.4.2 parity)

**Session scope:** the function(s) in: """$ARGUMENTS""" — one np.* function, or one closely-related
family that ships together (flip/fliplr/flipud counts as one). Do not drift beyond it — with two
sanctioned exceptions: **partial invocations** (a probe-only pass, a phases-1–2 design study, a
parity audit, an experiment across candidates) are legitimate — state which phases you are running
and stop there; and a **dependency extension** surfaced by the Phase-2 dependency audit is in-scope
when the function cannot reach parity without it.

## Mental model

NumPy 2.4.2 is the **source of truth** — not its docs, not memory: the **live interpreter** plus the
clone at `src/numpy/`. If NumPy does A, NumSharp does A, in C#. The mechanics may differ (NumPy uses
C templates/macros; NumSharp emits IL at runtime for SIMD-specialized kernels), but every input must
produce the same output value, dtype, shape, writeability, view/copy identity, and error text.
There is a reason for everything NumPy does — before deviating from NumPy's approach, articulate
what it buys (precision, overflow semantics, cache locality, pairwise-summation error bounds) and
prove your design preserves it. Breaking NumSharp changes to reach NumPy parity are acceptable.

## Phase map — do them in order, each phase gates the next

| # | Phase | Exit criterion |
|---|-------|----------------|
| 1 | Probe | Behavior matrix: you can predict NumPy's exact output for any input |
| 2 | Design | Integration decision: composition vs engine, kernel contract, file placement |
| 3 | Implement | Builds clean; spot-parity probes agree |
| 4 | Verify | Unit tests green + FuzzMatrix tier green (or divergence documented) |
| 5 | Benchmark | NPY/NS measured and reported honestly |
| 6 | Document + commit | CLAUDE.md API list updated; one extensive commit |

### Phase 1 — Probe (NumPy source + live behavior)

- Read the real implementation in `src/numpy/` — parameter handling, overload dispatch, edge-case
  branches, error paths. Grep `numpy/_core/` and `numpy/lib/` for the Python layer, `src/` C where it
  bottoms out.
- Run **live NumPy 2.4.2** (`python_run` heredocs — templates in `references/probing.md`) and build
  the behavior matrix: signature + defaults (`inspect.signature`), per-dtype result dtype and values,
  edge cases (empty, 0-d, 1-element, NaN/±inf/-0.0, negative/repeated/out-of-range axis, bool input,
  python-scalar vs np-scalar NEP50 weak promotion), **error messages verbatim**, view-vs-copy
  (`np.shares_memory` + writeability + write-through), and the layout families in
  `references/variations.md` that apply to this op.
- Never implement from documentation or recall alone — probe.

**Exit DOD:** 100% understanding — how the function reacts to every parameter/mode, what NumPy's
optimizations are, and which of ours apply.

### Phase 2 — Design the integration

- **Match the function's shape to a recipe first** — `references/design-recipes.md` maps the
  recurring shapes (pure composition, new ufunc, algorithm-selection param, index-consuming op,
  data-dependent-output kernel) to their house designs. Pattern 1 (compose existing np.*) vs
  Pattern 2 (TensorEngine + kernels) is the coarse cut: prefer composition when it reaches
  performance; drop to kernels for hot loops. Match NumPy's *implementation structure*, not just
  behavior — if NumPy is clean and the NumSharp area is spaghetti, refactor toward NumPy's design.
- **Dependency audit** (composition designs): NumPy's own source names your composition targets —
  verify each NumSharp counterpart supports the MODES NumPy uses (array-valued params? `axis=`?
  `dtype=`?), not merely that it exists. A missing mode is an explicit, recorded scope decision:
  extend the dependency first (preferred), or ship documented partial parity. Recipe in
  `references/design-recipes.md`.
- **View-first:** if NumPy returns a view, NumSharp must too. Exemplars: `Manipulation/np.flip.cs`
  (stride negation + base-offset shift via `Storage.Alias` — O(ndim), no data movement),
  `Manipulation/np.rot90.cs` (pure flip+transpose view composition),
  `Manipulation/np.trim_zeros.cs` (separable bounding box beating NumPy's own algorithm).
- **Kernel contract choice** (project CLAUDE.md Q&A has the full distinction):
  - whole-array kernel that walks its own dims/strides → `DirectILKernelGenerator` partial in
    `Backends/Kernels/Direct/`;
  - per-chunk inner loop driven by the iterator → `ILKernelGenerator` (root), run via
    `NDIterRef.Execute`.
- Skim `references/optimizations.md` — the catalog of techniques already in the codebase — before
  designing any kernel, and reuse rather than reinvent.
- **File placement:** API entry `src/NumSharp.Core/<Area>/np.<name>.cs` (Area ∈ Creation,
  Manipulation, Math, Statistics, Logic, LinearAlgebra, Selection, IO…); engine parts
  `Backends/Default/<Area>/DefaultEngine.<Op>.cs`; tests
  `test/NumSharp.UnitTest/<Area>/np.<name>.Test.cs`.

### Phase 3 — Implement (the hard rules)

- Full parameter parity: NumPy's names, order, defaults — reachable the way NumPy reaches them.
  Elementwise ufuncs take the NumPy overload shape
  `f(x[, x2], NDArray out = null, NDArray where = null, NPTypeCode? dtype = null)`; the plumbing
  exists (`Backends/Default/Math/DefaultEngine.UfuncOut.cs`) — do not reinvent it. A brand-NEW
  ufunc is a ~12-file wiring (enum → IL emission → engine → np API → NDExpr fusion → DecimalMath →
  oracle → benchmark): follow the checklist in `references/new-ufunc.md`, keyed off the ATan2
  archetype.
- All 15 dtypes handled, or explicitly rejected with the probed NumPy error. Char and Decimal have
  no NumPy analog — decide their behavior deliberately and document it.
- **No struct-kernel pattern** (generic-struct-implements-IKernel specialization). The codebase
  standardized on IL emission with layout-keyed caches; a second specialization mechanism fragments
  the architecture.
- **No hand-written per-dtype implementations** and no if-else/switch-per-dtype selecting
  specialized code paths. Dtype specialization belongs in kernel cache keys — the IL emitter
  generates per-(op, dtype, layout) native code from ONE emission path. The only sanctioned dtype
  switch is dispatch-to-generic plumbing (`case NPTypeCode.X: return Process<X>(...)`).
- **Element-traversal loops only via NDIter or IL kernels.** O(ndim) loops over dims/strides (view
  math) are plain C#.
- NEP50 type promotion; error texts verbatim from the Phase-1 probe; broadcast views stay read-only.

### Phase 4 — Verify (the parity gates)

- **Spot-parity first:** run identical cases through `dotnet_run` (NumSharp) and `python_run`
  (NumPy) and diff — values, dtype, shape, flags, error text (`references/probing.md` has the
  side-by-side pattern).
- **Unit tests from real probed NumPy output** (MSTest v3), covering every applicable family in
  `references/variations.md`: layouts (contiguous/F/strided/negative-stride/sliced/broadcast/0-d/
  empty), dtype sweep, edge cases, error cases.
- **THE gate — wire the op into the differential-fuzz oracle.** Invoke the **`oracle`** skill: add
  the `gen_oracle.py` job + the `OpRegistry.cs` case, regenerate (pinned `numpy==2.4.2`), run the
  tier. Char rides the uint16 proxy automatically; Decimal needs a `gen_decimal_oracle.cs` entry.
- **Exit DOD:** tier green. Real divergences get fixed; intended ones get documented in
  `MisalignedRegistry` — never silently excused.

### Phase 5 — Benchmark

- Invoke the **`benchmark`** skill: add the C# `[Benchmark]` + NumPy twin, smoke them, report
  **NPY/NS** (`>1` = NumSharp faster; higher is better).
- Timing scripts MUST run `dotnet run -c Release - < script.cs` — Debug taints hand-written hot
  loops ~2× (file-based apps compile `#:project` NumSharp.Core in Debug).
- Aspire to beat NumPy — the codebase routinely lands 1.5×+ on SIMD-capable dtypes with contiguous
  layouts. The shipping floor: **every cell < 1.0 is understood** — either fixed (learn NumPy's
  technique first) or documented as expected (Half/Complex/Decimal scalar paths; O(1) view ops
  measure dispatch overhead, not throughput).

### Phase 6 — Document + commit

- Update `.claude/CLAUDE.md` → "Supported np.* APIs": add the name to its category list; if the
  function has notable semantics, add a per-function paragraph in the house pattern (see the
  flip/rot90/trim_zeros paragraphs — semantics, validation texts, view/copy nature, perf claim,
  file pointer).
- One `git add … && git commit` with an extensive conventional-commit message (what, why, parity
  evidence, perf numbers).

## DOD checklist (the whole function)

- [ ] Behavior matrix probed against live NumPy 2.4.2 (never docs/memory alone)
- [ ] Signature parity: parameter names, order, defaults
- [ ] All 15 dtypes handled or explicitly rejected with the probed error
- [ ] Layouts: contiguous, F, strided/transposed, negative-stride, sliced/offset, broadcast
      (read-only), 0-d, empty
- [ ] NEP50 promotion + edge cases + verbatim error texts match
- [ ] View where NumPy views, copy where NumPy copies
- [ ] Unit tests generated from real NumPy output
- [ ] Oracle: op in `gen_oracle.py` + `OpRegistry.cs`, FuzzMatrix tier green
- [ ] Benchmark pair added; NPY/NS reported; every <1.0 cell explained
- [ ] CLAUDE.md API list updated; extensive commit

## Gotchas

- **Debug taint:** any timing not run `-c Release` is ~2× slow on hand-written loops — never quote it.
- **`#:` directive paths must be absolute** in `dotnet_run` heredoc scripts.
- **Internals access** from scripts: `#:property AssemblyName=NumSharp.DotNetRunScript` (matches
  `InternalsVisibleTo`) + `#:property AllowUnsafeBlocks=true` + `#:property PublishAot=false`.
- **Pin `numpy==2.4.2`** for every probe and corpus regeneration — verify `np.__version__` first.
- **0-d vs 1-element 1-D are different animals** (NumPy's `m[()]` scalar returns, axis errors on
  0-d) — probe both, always.
- **Index-typed outputs are Int64** — NumPy's `intp` is int64 on 64-bit platforms (NumPy 2.x even
  on Windows); house precedent `Default.NonZero.cs`. Never Int32.
- **NumPy's input gates are often the cast machinery talking**, not bespoke messages
  (`Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'`) —
  route validation through the equivalent machinery so texts match for free
  (`references/design-recipes.md`).
- **`(1)` is not a tuple** and other Python-literal traps — when the op touches headers/parsing,
  read the File I/O section of project CLAUDE.md first.

## References

- `references/probing.md` — probe templates: python_run behavior matrix, ufunc loop introspection,
  dtype-pair grids, validation-order probes, dotnet_run internals template, side-by-side parity
  diff, error-text capture, timing rules.
- `references/design-recipes.md` — function-shape → house-design table + recipes: dependency
  audit, index-consuming ops, data-dependent output, cast-machinery validation, algorithm-selection
  params, non-ufunc dtype=, tuple returns, NaN-in-set-ops.
- `references/new-ufunc.md` — the ~12-touchpoint checklist for a brand-new elementwise ufunc
  (the ATan2 archetype), loop-signature policy from `ufunc.types`, the two rejection error texts.
- `references/optimizations.md` — the house optimization catalog (specialization, loop shaping,
  SIMD primitives, memory, algorithmic, bridging, semantic compliance, caching).
- `references/variations.md` — the 51-variation input space a parity claim covers (layouts,
  pairwise paths, operand flags, iteration flags, composite paths).
- Skills: **`oracle`** (the FuzzMatrix gate — Phase 4), **`benchmark`** (the perf harness — Phase 5).
- Project `.claude/CLAUDE.md` — DOD, architecture, kernel Q&A, supported-API catalog.
