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

**Drawing the family boundary.** When the request names "and neighbouring functions", or a concept
rather than a function, size the family BEFORE Phase 1. NumPy's own source file is the strongest
signal: names defined together upstream generally share the abstraction and the helpers, so
splitting them means implementing the shared primitive twice — the tri/diag cluster all lives in
`_twodim_base_impl.py`, and `tril`/`triu` are literally `where(tri(...), m, 0)`. When two readings
of the request differ materially in work (a 5-function reading vs a 13-function one), put the
options to the user instead of picking silently. Scope is theirs, it is cheap to ask before
probing, and expensive to redo after.

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
| 5 | Benchmark | NPY/NS measured and reported honestly; any optimization re-verified through Phase 4 |
| 6 | Document + commit | CLAUDE.md API list updated; one extensive commit |

The map is a ratchet with one exception: **Phase 5 can un-gate Phase 4**, because optimizing means
editing the implementation. Treat a Phase-5 code change as re-entering Phase 4, not as polish.

**Audit mode** — two triggers, one playbook (`references/audit.md`): the op already exists
("verify/modernize/why does np.foo differ"), or you just shipped it and are sweeping its edges
("do a pass over edge cases", "find missing support"). Both reorder the phases — inventory existing
coverage, signature-diff, probe OUTSIDE the corpus envelope, classify findings (wins included). The
post-ship sweep is not a re-run of Phase 4; its entire subject is what the gates you just turned
green cannot express.

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
- **The docstring is not the contract; the body is.** `np.diag`'s docstring promises "a copy of its
  k-th diagonal" and the 2-D branch returns a **read-only view** — `shares_memory` says so. Probe
  view-vs-copy on every branch even when the prose sounds settled.
- **Incidental consequences of NumPy's spelling are still behaviour.** `tril` squares a 1-D input
  into an `(n, n)` result only because the body reads `tri(*m.shape[-2:])`, and a 0-d input escapes
  as `tri()`'s own missing-argument `TypeError`. Reproduce such **leaked implementation errors
  verbatim** — house precedent is the `mmap_mode` texts in project CLAUDE.md. You cannot derive
  either behaviour from the documentation; you read them off the composition.
- **For tuple returns, probe identity and arity, not just values.** `diag_indices` returns THE SAME
  array object in every slot (`(idx,) * ndim` — a write through one is visible through all), and
  `mask_indices`' arity follows its callback's rank rather than `n`. Both are observable, so both
  are parity.

**Exit DOD:** 100% understanding — how the function reacts to every parameter/mode, what NumPy's
optimizations are, and which of ours apply.

### Phase 2 — Design the integration

- **Match the function's shape to a recipe first** — `references/design-recipes.md` maps the
  recurring shapes (pure composition, new ufunc, algorithm-selection param, index-consuming op,
  data-dependent-output kernel, **NumPy object/protocol API**) to their house designs. Not every
  np.* name is "array in → array out": iterators and context managers (`nditer`, `ndindex`,
  `broadcast`, `printoptions`) are objects whose parity is protocol semantics, and they have their
  own recipe. Pattern 1 (compose existing np.*) vs
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
- **Prove the new coverage has teeth before trusting it green.** A tier that passes on the first run
  is as consistent with "not wired up" as with "correct" — `OpRegistry`'s `default:` throws, so a
  missing case fails loudly, but a mis-keyed param or an op that never got emitted just sits silent.
  Break it on purpose once: rename your `OpRegistry` case, confirm the tier goes red on YOUR cases,
  revert. Two minutes, and it converts "green" from a hope into evidence.
- **Watch output-size blow-up when joining a layout × dtype tier.** Those tiers multiply every job
  by ~26 layouts × ~15 dtypes, so an op whose output is superlinear in its input balloons the
  committed corpus — `diagflat` squares its input, turning a 24-element operand into a 576-element
  expectation. Cap such jobs by operand size and give the larger/strided shapes a dedicated tier.
- **When the op has no corpus-comparable result, say so instead of forcing a tier.** The corpus
  compares `(dtype, shape, bytes)`, so an op that returns no array — an iteration protocol, a
  context manager, an object exposing a cursor — cannot ride it. That is a narrow exemption, not a
  general escape: if your op returns an NDArray in any mode, it belongs in the corpus. When it
  genuinely doesn't, record the reason in the CLAUDE.md paragraph so a later reader sees a decision
  rather than an oversight, and make the unit tests the gate (probed protocol transcripts:
  order, cursor position, view/copy identity, verbatim errors).
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
- **"Understood" means located, not guessed.** Decompose the slow loop into its layers and time each
  — the algorithm alone, then the algorithm plus whatever it allocates per element — and only then
  micro-benchmark the construction path that dominates. Sessions have found the engine 28× FASTER
  than NumPy while the op measured 0.17×, the entire gap being a per-element object; that diagnosis
  changes both the fix and the honest framing, and you cannot reach it from the aggregate number.
  When the located cost turns out to be core-wide (object construction, engine resolution,
  refcounting), take whatever win is local and in-scope, then write the residue down with its cause
  — quietly shipping it, or unilaterally rewriting a core constructor, are both worse.
- **When two strategies trade allocation against writes, measure both across a size sweep** rather
  than reasoning from instruction counts. Fresh large allocations arrive already zeroed from the OS,
  so a "wasteful" pre-zeroing `np.zeros` can cost nothing while an "obviously cheaper" write-once
  pass touches twice the memory. One measured case inverted sharply at 64 MiB — write-once 6.0 ms vs
  11.0 ms just below it, 18.9 ms vs 13.5 ms just above. Encode the measured crossover with its
  evidence in a named constant, not a guessed round number.
- **Don't geomean across cost classes.** An O(1) view, an O(N) fill and an O(√N) sparse write mean
  different things, and averaging them yields a number describing nothing. Separate them in the
  benchmark class and report them apart — it also keeps you honest about which sub-1.0 cells are
  dispatch overhead on a tiny op versus real algorithmic loss.

**Phase 5 writes code, so it re-opens Phase 4.** This is the one place the phase map lies: every
other phase only adds to the last, but an optimization is a new **branch** — on size, on layout, on
operand type — and Phase 4's gates only cover the forms the corpus can express. A real regression
shipped this way: a scalar fast path added to `fill_diagonal` routed every non-`NDArray` value
through an `IConvertible` conversion, so `int[]`, `List<int>` and `int[,]` began throwing
`InvalidCastException` — while the unit tests, the tier, and the hand-written parity harness all
stayed green, because every one of them passes that argument as an `NDArray`. After any change made
for speed, re-run the unit tests AND the tier, then probe what they structurally cannot see:

- **The acceptance surface of polymorphic parameters.** A corpus serializes each parameter to ONE
  canonical form, so the other accepted forms — C# arrays, collections, non-`IConvertible` scalars
  like `Half`/`Complex`, `null` — are invisible to it. Enumerate what the parameter's declared type
  actually admits and probe each. Then close the hole properly: where you can, widen the oracle to
  hand the argument over in a NON-canonical form (passing a raw `long[]` instead of wrapping it in
  `np.array`) so the general path is gated from then on, and add unit tests for the rest.
- **Both sides of a size- or layout-selected branch.** A threshold means every small test you
  already wrote exercises one half of the code. Add a case that crosses it — a large-allocation
  path that forgets to zero its output produces allocator garbage no small test can see.

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
- [ ] Oracle: op in `gen_oracle.py` + `OpRegistry.cs`, FuzzMatrix tier green — or the op returns no
      array at all and the exemption is recorded in CLAUDE.md
- [ ] Benchmark pair added; NPY/NS reported; every <1.0 cell explained
- [ ] Every Phase-5 optimization re-verified: unit tests + tier re-run, acceptance surface of each
      polymorphic parameter probed, both sides of any size/layout-selected branch tested
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
- **Corpus-green is not full parity** — every fuzz tier has a size/layout envelope (the reduce
  tier's operands are ≤ 24 elements, below NumPy's pairwise-summation blocking). Effects that
  engage past a threshold — accumulation trees, buffering — need probes outside the envelope
  (`references/audit.md`).
- **NumPy's input gates are often the cast machinery talking**, not bespoke messages
  (`Cannot cast array data from dtype('float64') to dtype('int64') according to the rule 'safe'`) —
  route validation through the equivalent machinery so texts match for free
  (`references/design-recipes.md`).
- **`(1)` is not a tuple** and other Python-literal traps — when the op touches headers/parsing,
  read the File I/O section of project CLAUDE.md first.
- **`NDArray`'s `==` / `!=` are ELEMENTWISE** — they return `NDArray<bool>`, so a null check like
  `if (op[i] == null)` is not a null check. Use `is null` / `is not null`, which are pattern matches
  and bypass operator overloads. Symptom when you get it wrong: `CS0266 cannot convert
  NDArray<bool> to bool`, or worse, an `&&` that compiles into an elementwise `&`.
- **Shapes and strides are `long[]`, not `int[]`** (`nd.shape`, `Shape.Dimensions`, `Shape.Strides`)
  — project CLAUDE.md's internals table still says int[] in places, so confirm the member's real
  type before designing a signature around it. Two `params` overloads (`long[]` and `int[]`) also
  make the zero-argument call ambiguous, and `int[]` does NOT widen to `long[]` by array covariance:
  the workable pair is `params long[]` (individual int literals widen into it) plus a NON-params
  `int[]` overload.
- **`dotnet run <path.cs>` caches per script path** — a probe can replay a stale build and look like
  your fix failed. `references/probing.md` has the symptom and the two one-line fixes; reach for it
  whenever a result doesn't move after a change you can see in the source.
- **Normalize line endings before diffing NumSharp output against NumPy output.** The C# side writes
  CRLF and Python writes LF, so `diff` marks EVERY line changed and a perfect 165/165 match reads as
  165/165 failures. Pipe both through `tr -d '\r'` first. Watch too for harness-only formatting
  differences masquerading as parity bugs (complex printed `(1.0+0.0j)` vs NumPy's `(1+0j)` is the
  same value) — fix the harness, not the implementation.
- **Test-harness dtype traps:** `np.arange` yields **Int64**, so `GetAtIndex<int>` / `GetValue<int>`
  on its results trips a `Debug.Fail` inside the test host; coordinate element access is
  `GetValue<T>(i, j)` (there is no two-argument `GetData<T>`); and `BeOfValues` has no loop for
  Half/Decimal, so assert those dtypes via `count_nonzero` or `GetValue<T>` instead. These surface
  as test failures that look like implementation bugs and are not.
- **The working tree may not be yours alone.** Another session writing into the same checkout will
  break your build from files you never touched. Don't repair or revert someone else's in-flight
  work: identify whose file it is (`git status`, timestamps), wait or work around it, and at commit
  time `git add` your OWN paths explicitly — never `git add -A` — then check the staged diff is only
  yours before committing.

## References

- `references/probing.md` — probe templates: python_run behavior matrix, ufunc loop introspection,
  dtype-pair grids, validation-order probes, dotnet_run internals template, side-by-side parity
  diff, error-text capture, timing rules.
- `references/design-recipes.md` — function-shape → house-design table + recipes: dependency
  audit, object/protocol APIs (iterators, context managers), wrapping an engine `ref struct` in a
  long-lived object, mirroring NumPy's wrapper-vs-core layering, index-consuming ops, data-dependent
  output, cast-machinery validation, algorithm-selection params, non-ufunc dtype=, tuple returns,
  NaN-in-set-ops.
- `references/new-ufunc.md` — the ~12-touchpoint checklist for a brand-new elementwise ufunc
  (the ATan2 archetype), loop-signature policy from `ufunc.types`, the two rejection error texts.
- `references/audit.md` — auditing an existing op **and sweeping the one you just shipped**
  ("do a pass over edge cases" lands here): coverage inventory, signature diff, the
  outside-the-envelope probe dimensions (acceptance surface of polymorphic params, both sides of
  size-selected branches, param combos the tier skips, extreme scalars, `null`), finding
  classification, and the evidence tables from the five-function audit and the 13-function
  post-ship sweep.
- `references/optimizations.md` — the house optimization catalog (specialization, loop shaping,
  SIMD primitives, memory, algorithmic, bridging, semantic compliance, caching).
- `references/variations.md` — the 51-variation input space a parity claim covers (layouts,
  pairwise paths, operand flags, iteration flags, composite paths).
- Skills: **`oracle`** (the FuzzMatrix gate — Phase 4), **`benchmark`** (the perf harness — Phase 5).
- Project `.claude/CLAUDE.md` — DOD, architecture, kernel Q&A, supported-API catalog.
