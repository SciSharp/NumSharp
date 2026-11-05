# np.evaluate over text (S, U, StringDType)

`np.evaluate` compiles an `NDExpr` tree into one fused pass. Until this plan it refused every text operand (decision D20
of `strings-phase2.md`: "text leaves are a later extension"). This plan brings text into the trees: every node takes
text exactly where NumPy's ufunc has a text loop, and the result equals NumPy's unfused statements byte for byte.

## 0. State

| Phase | Scope | Status |
|---|---|---|
| E1 (MVP) | text leaves (S / U / StringDType arrays, 0-d, a C# `string` as NumPy's `str`) in the six comparisons anywhere in a tree; their bool results feed every numeric node, reduction and keyword | done (`2ab907a0`) |
| E2 (next release) | the truth value of text (`LogicalAnd/Or/Xor`, `Where`'s condition, `Any` / `All` / `CountNonzero`), `IsNaN` over StringDType, the np.strings ufuncs and searches as factories (`StrLen`, the nine predicates, `Find` / `RFind` / `Count` / `Index` / `RIndex`, `StartsWith` / `EndsWith`), text → number `Cast` | done (the owner's "Continue with phase E2", 2026-11-05) |
| E3 (next release) | text RESULTS: `Where` with text branches (np.where's promotion), `Add` (concatenation), `Multiply` (StringDType repeat; S / U need `out=`), `Maximum` / `Minimum` over StringDType, a bare text root; text reductions (`ArgMax` / `ArgMin` over S / U / T, `Max` / `Min` / `Sum` over StringDType); NumPy's own error for every other node with a text operand | planned |

## 1. Decisions (owner, 2026-11-04, design-ask pages)

| # | Question | Answer (verbatim) | Consequence |
|---|---|---|---|
| D1 | How a text node runs inside np.evaluate: block-staged / materialize / inline | "Measure performancewise" | measured in §3: whole-array text stages + one fused numeric pass |
| D2 | Which text nodes in this round | "Everything NumPy does on text" — "But I want phases where we do minimal MVP and keep other phases for next release" | E1 now; E2 and E3 for the next release |
| D3 | Spelling of the np.strings factories | "PascalCase like NDExpr" | `NDExpr.StrLen`, `NDExpr.IsAlpha`, `NDExpr.StartsWith`, `NDExpr.Find`, ... (E2) |

## 2. What NumPy computes on text (2.4.2, probed)

- Comparisons (`equal` … `greater_equal`) over S·S, U·U, T·T, T·U, U·T, and against a Python `str` / `bytes` of the same
  class; S·U and text·number have no loop (`UFuncTypeError`, the operand classes named).
- `logical_and` / `logical_or` / `logical_xor` take any text mix through the text's truth (`'' → False`); `logical_not`
  of text has NO loop. `np.where(text, x, y)` and `astype(bool)` read the truth too. `np.any` / `np.all` /
  `np.count_nonzero` of text compute (truth), while `text != 0` has no loop.
- `isnan` over StringDType only (its NaN-like NA); `isinf` / `isfinite` have no text loop.
- `np.strings.str_len` (int64) and the nine predicates are ufuncs (`isdecimal` / `isnumeric`: no S loop);
  `find` / `rfind` / `count` (int64), `startswith` / `endswith` (bool), `index` / `rindex` (raise `ValueError: substring
  not found`) are functions over private ufuncs.
- `add` concatenates (S+S → `S(wa+wb)`, U+U → `U(wa+wb)`, T+T / T+U → T); `multiply` repeats (StringDType without
  `out=`; S / U raise "The 'out' kwarg is necessary when using the string multiply ufunc directly..." without one);
  `maximum` / `minimum` exist for StringDType only.
- `np.where` promotes its branches (U with S → U, U with int64 → `<U21`, U with T → T).
- Reductions: `argmax` / `argmin` over S / U / T; `max` / `min` / `sum` over StringDType (S / U: no `maximum` loop;
  `sum` raises "the resolved dtypes are not compatible with add.reduce").

## 3. Architecture measurement (D1)

`(names == "paris") & (age > 30)`, `age` int64, pinned to two P-cores (0x14), best of 7 after a paused double warm-up,
every executor verified bit-identical to the unfused result first. Prototypes: `benchmark/fusion/probes/evaltext_arch_probe.cs`
(round 1: unfused, materialize, staged, inline) and `evaltext_arch2_probe.cs` (round 2: materialize, lite, staged).

- **staged**: one composite inner loop; per block of 4,096 elements the text kernel writes an L1 scratch, then the fused
  numeric kernel reads it (trivial loop or NDIter).
- **lite**: the text node's own kernel over the whole operands into a pooled temporary (trivial loop, the rows kernel for
  narrow rows, NDIter), then the numeric remainder's fused program over the temporary.
- **materialize**: `np.equal` into a temporary, then `np.evaluate` of the rest (the engine-call form of lite).
- **inline** (proxy): staged with a forced-scalar numeric kernel, what emitting the text compare inside the fused kernel
  costs (a text element is not a SIMD lane).

| case (µs) | NumPy | unfused | materialize | lite | staged | inline |
|---|---|---|---|---|---|---|
| U6 contiguous 1K | 19.4 | 9.2 | 3.1 | 2.8 | 2.3 | 3.0 |
| U6 contiguous 100K | 2,395 | 163 | 176 | 170 | 171 | 232 |
| U6 contiguous 1M | 24,915 | 3,028 | 2,522 | 2,488 | 2,600 | 3,070 |
| S6 contiguous 1K | — | 3.4 | 2.7 | 2.3 | 1.9 | 2.5 |
| U6 rows of 8 (`x2d[:, :8]`) 100K | — | 504 | 590 | 587 | 775 | 802 |
| U6 rows of 8, 1M | — | 12,284 | 12,139 | 12,013 | 13,927 | 15,681 |
| S6 rows of 8, 100K | — | 334 | 404 | 406 | 515 | 515 |
| T contiguous 1K / 100K / 1M | — | 7.5 / 657 / 6,434 | 6.3 / 608 / 6,384 | 6.0 / 610 / 6,308 | 5.4 / 607 / 6,375 | 6.1 / 750 / 6,895 |
| T rows of 8, 100K / 1M | — | 1,842 / 18,575 | 1,240 / 17,843 | 1,293 / 17,785 | 1,174 / 16,252 | 1,412 / 17,519 |
| strided `x[::2]` (U / S / T, every n) | — | | ≈ | ≈ | ≈ | ≈ |

Reading: the text kernel is 82–97 % of the call (D20's measurement holds). Inline, against staged in the same run, is
25–54 % slower on contiguous operands at 100K+ (S 47–54 %, U 25–40 %, StringDType 10–24 %: its numeric part runs
scalar) and within −4…+16 % on strided and narrow-row operands. Staged wins only the fixed cost of a small contiguous
call (0.4–0.6 µs at 1K, 1.1–1.2×) and StringDType's narrow rows (no rows kernel there, 1.1×); lite wins narrow S / U
rows by 8–24 % at 100K / 1M (lite/staged 0.76–0.92: it keeps the text kernels' rows route) and ties everywhere else.
On the U6 contiguous cells NumPy was timed on, every design is 6–14× NumPy.

**Decision: lite.** Equal performance in aggregate, and the decisive property: a text node runs the engine's own text
call (`NDStringUfuncs.BinaryGuard` and friends) over whole arrays, so (1) NumPy's errors, NA rules and descriptor
resolution come from the code the string oracle already gates; (2) a program never bakes a text width or class, so the
program cache needs no descriptor in its key; (3) every evaluate route (the trivial loop, NDIter, `where=`, `out=` casts,
the streamed and folded reductions, `Compile()`) runs unchanged over the stage results. A staged fast path for small
contiguous calls stays possible later (it would sit behind the same lowering).

## 4. Design

- **Leaves.** `NDExprBindContext.IndexOf` accepts text arrays; `implicit operator NDExpr(string)` makes NumPy's `str`
  (`NDArray.PythonStrLiteral`: a 0-d `<U{len}` marked as a Python str), so `NDExpr.Equal(city, "paris")` and
  `(NDExpr)city < "m"` compile. A literal is an input, not IL: one program serves every text value.
- **Lowering** (`NDExpr.Text.cs`, at program build, after binding): each node lowers its children (`LowerText`). A node
  that computes on text becomes a STAGE and leaves a new positional input of the stage's result dtype in its place; the
  remainder is an ordinary numeric tree over the numeric inputs and the stage results. A node with a text operand that the
  phase does not compute raises `StringComputationNotSupportedException` naming the phase.
- **Stages** (E1: the comparisons) run in evaluation order per call through the engine's guard
  (`NDStringUfuncs.BinaryGuard(ufunc, x1, x2)`): it computes NumPy's result or raises NumPy's error. A stage operand is
  an input array, a Python scalar (`ConstNode` → `NDArray.PythonLiteral`) or a stand-in 0-d array of a numeric sub-tree's
  dtype (a comparison of text with a number never computes, so its numeric side only has to name its class).
- **Remainder.** Resolved per call through the positional program cache (`GetProgram(numericInputs)`: dtype signature and
  0-d parameters of the stage results) and run by `EvaluateCore` with the call's `out=` / `where=` / `dtype=` / `casting=`
  / `order=`. A remainder that is a bare stage result returns the stage's array (NumPy's own ufunc result) when no keyword
  shapes the result.
- **Truth tests.** The factories `Any` / `All` / `CountNonzero` mark their `x != 0` (`ComparisonNode.TruthTest`, part of
  the node identity), so the reduction reads the truth of a text `x` while a user's `NotEqual(text, 0)` keeps NumPy's
  no-loop error.
- **E2: truth stages.** NumPy reads a string's truth through its bool CAST (the logical ufuncs, `np.where`'s condition,
  `np.any` / `np.all`) or its NONZERO function (`np.count_nonzero(a)`; with an axis or keepdims NumPy sums `a != ''` for
  S / U and `a.astype(bool)` for StringDType, so an NA of an empty-text NA object counts there and not without an axis —
  probed). A text operand of `LogicalAnd` / `LogicalOr` / `LogicalXor` or `Where`'s condition becomes a truth stage
  (`NDStringTruth.Truth`, the cast) and the node stays fused over its bool result — the engine's own `logical_and` over text
  is exactly that (the guard's `LogicalOverTruth`). The reduction lowers its truth test the same way, by the route of its
  reader (`NDExprTruthRule`), and stays the ordinary reduction over the bool result (axis, keepdims and the 0-d `axis 0 /
  -1` rule NumPy lets through included). NumPy casts inside its iterator, where a broadcast operand abstains from the
  result's layout: a truth stage of a broadcast text operand computes over the un-broadcast view and broadcasts the bool
  result back (a dense K-order cast of a row broadcast is F-contiguous; the fused result would come out F where NumPy's is C).
- **E2: statement stages.** `IsNaN` of text is `TensorEngine.IsNan` (NaN-like StringDType NA; NumPy's refusal for S / U);
  `Cast` of text to a number is `astype` (NumPy's parse, its own statement in NumPy); the numpy.strings factories
  (`NDExpr.Strings.cs`: `StringsNode`, never IL) are `NDStringMethods.StrLen` / `Predicate` / `FindLike`. A tree with a
  `StringsNode` takes the text plan even with no text input (`NDExprBindContext.HoldsStageOnlyNode`), so the stage raises
  NumPy's own no-loop error for `StrLen(int array)`.
- **E2: computed operands.** A stage operand may be any numeric sub-tree (a search's `start`, the number a comparison
  meets): renumbered over the remainder inputs it reads and evaluated per call into the temporary NumPy's unfused sequence
  holds (`MaterializeChildNumPyLayout`, the fmax / fmin stages' route), so its values, layout and its own errors come first
  as in NumPy's statement order (E1's 0-d stand-in for a comparison's computed number is gone). A constant-only sub-tree
  (`-(NDExpr)3`) is evaluated over a 0-d dummy and marked a Python scalar when its typing is weak.

## 5. Gates

- Unit tests from NumPy output: `test/NumSharp.Tests/Backends/Iterators/NDEvaluateTextTests.cs` (every comparison over
  S / U / T and their mixes, literals on both sides, broadcasting, strided / F / transposed / reversed / 0-d / empty
  operands, reductions over comparisons, `out=` / `where=` / `dtype=` / `order=`, `Compile()` handles, NumPy's errors,
  the refusals naming their phase; E2: the truth readers over S / U / T with every StringDType NA kind, both
  `count_nonzero` routes, axis forms and the 0-d axis rule, the transposed and broadcast layouts; `IsNaN`; the
  numpy.strings factories with every `start` / `end` form, their errors and NA rules; `Cast` to numbers; the keywords and
  handles over a strings stage; one program for every literal substring).
- E2 in the string oracle section `evaltext` (cases after E1's, so E1's ids are unchanged; NumPy reference = the
  unfused statements): the truth readers in every core layout (G), the StringDType NA kinds including an empty-text NA
  object (H), text classes mixed in one logical node and `IsNaN`'s refusal of S / U (I), the numpy.strings functions in
  every core layout with eight `start` / `end` forms, fused and reduced (J), each function's NA rule (K), `Cast` from
  integer and float texts (L), NumPy's refusals (M). A second layout excuse, np.evaluate's own rule as over numbers: an
  elementwise result np.evaluate allocates is C- or F-ordered, where NumPy's next statement keeps a permuted temporary's
  K order (the t3 layout's 3-D transpose); kept only for a C-ordered result, so a bare stage result's layout stays
  compared.
- Benchmark, E2 (`benchmark/fusion/probes/evaltext_e2_bench.cs` / `.py`, the table joined by `evaltext_sheet.py`; the
  method of E1's). Trees: truth_and `LogicalAnd(names, age > 30)`, truth_where `Where(names, price * 2, price)`,
  truth_count `CountNonzero(names)`, strlen `(StrLen(names) > 5) & (age > 30)`, starts
  `CountNonzero(StartsWith(names, "pa") & (age > 30))`, find `Where(Find(names, "o") >= 0, price, 0)`, isalpha
  `IsAlpha(names)` (a bare stage), cast `Cast(digits, int64) * 2`. Cells: NPY/NS (np.evaluate µs).

  | text | layout | n | truth_and | truth_where | truth_count | strlen | starts | find | isalpha | cast |
  |---|---|---|---|---|---|---|---|---|---|---|
  | U | contig | 1,000 | ✅ 72.3× (2.9) | ✅ 80.3× (2.7) | ✅ 1.3× (2.2) | ✅ 1.5× (3.8) | ✅ 1.3× (7.4) | ✅ 1.5× (9.9) | ✅ 5.1× (4.7) | ✅ 8.7× (32.7) |
  | U | contig | 100,000 | ✅ 132.3× (158) | ✅ 138.6× (153) | ✅ 2.6× (85.4) | ✅ 4.5× (223) | ✅ 2.7× (471) | ✅ 1.1× (1,839) | ✅ 6.3× (389) | ✅ 9.5× (2,984) |
  | U | contig | 1,000,000 | ✅ 95.4× (2,166) | ✅ 81.1× (2,599) | ✅ 1.8× (1,462) | ✅ 2.9× (3,874) | ✅ 2.4× (5,020) | ✅ 1.1× (21,933) | ✅ 6.1× (4,059) | ✅ 9.4× (30,164) |
  | U | strided | 1,000 | ✅ 49.1× (4.3) | ✅ 52.2× (4.2) | 🟡 1.0× (2.8) | ✅ 1.2× (5.1) | ✅ 1.3× (8.1) | ✅ 1.4× (10.3) | ✅ 5.0× (4.7) | ✅ 9.2× (31.6) |
  | U | strided | 100,000 | ✅ 67.7× (313) | ✅ 81.8× (260) | ✅ 1.6× (137) | ✅ 2.5× (377) | 🟡 0.8× (1,488) | ✅ 1.0× (1,968) | ✅ 6.2× (399) | ✅ 8.6× (3,289) |
  | U | strided | 1,000,000 | ✅ 46.1× (4,599) | ✅ 48.4× (4,400) | ✅ 1.3× (2,585) | ✅ 2.3× (5,574) | ✅ 2.1× (6,573) | ✅ 1.2× (20,207) | ✅ 2.2× (11,014) | ✅ 9.5× (30,215) |
  | S | contig | 1,000 | ✅ 23.9× (2.2) | ✅ 24.6× (2.2) | ✅ 1.4× (1.9) | ✅ 2.1× (2.8) | ✅ 1.1× (8.1) | ✅ 1.6× (9.4) | ✅ 4.1× (4.1) | ✅ 3.8× (28.3) |
  | S | contig | 100,000 | ✅ 49.1× (99.6) | ✅ 54.6× (92.8) | ✅ 3.6× (59.1) | ✅ 6.5× (127) | ✅ 2.0× (583) | ✅ 1.1× (1,817) | ✅ 5.1× (349) | ✅ 3.9× (2,709) |
  | S | contig | 1,000,000 | ✅ 47.8× (1,035) | ✅ 42.9× (1,278) | ✅ 3.5× (612) | ✅ 6.0× (1,771) | ✅ 2.1× (5,991) | ✅ 1.3× (16,951) | ✅ 5.4× (3,352) | ✅ 4.0× (27,558) |
  | S | strided | 1,000 | ✅ 14.5× (3.8) | ✅ 15.9× (3.6) | ✅ 1.1× (2.4) | ✅ 1.4× (4.2) | ✅ 1.0× (9.1) | ✅ 1.6× (9.2) | ✅ 4.1× (4.2) | ✅ 3.8× (28.7) |
  | S | strided | 100,000 | ✅ 25.5× (212) | ✅ 31.0× (177) | ✅ 3.5× (60.4) | ✅ 3.5× (249) | ✅ 1.8× (655) | ✅ 1.1× (1,803) | ✅ 4.3× (419) | ✅ 4.1× (2,696) |
  | S | strided | 1,000,000 | ✅ 19.0× (2,820) | ✅ 21.8× (2,680) | ✅ 2.9× (730) | ✅ 3.5× (3,298) | ✅ 1.9× (7,025) | ✅ 1.3× (18,482) | ✅ 5.3× (3,410) | ✅ 4.1× (27,623) |
  | T | contig | 1,000 | ✅ 1.8× (2.3) | ✅ 2.4× (2.3) | ✅ 1.7× (2.0) | ✅ 2.9× (4.7) | ✅ 2.3× (12.4) | ✅ 3.0× (14.4) | ✅ 8.5× (5.0) | ✅ 2.7× (33.0) |
  | T | contig | 100,000 | ✅ 2.2× (103) | ✅ 2.7× (118) | ✅ 4.9× (60.3) | ✅ 4.5× (281) | ✅ 2.9× (811) | ✅ 1.7× (2,256) | ✅ 10.3× (407) | ✅ 2.6× (3,215) |
  | T | contig | 1,000,000 | ✅ 1.3× (1,945) | ✅ 2.9× (2,260) | ✅ 3.8× (791) | ✅ 4.4× (3,412) | ✅ 3.0× (8,160) | ✅ 1.8× (22,800) | ✅ 10.0× (4,157) | ✅ 2.7× (31,815) |
  | T | strided | 1,000 | 🟡 1.0× (4.6) | ✅ 1.3× (4.4) | ✅ 1.1× (3.2) | ✅ 2.4× (5.8) | ✅ 2.3× (12.7) | ✅ 3.0× (14.1) | ✅ 8.2× (5.2) | ✅ 2.6× (33.6) |
  | T | strided | 100,000 | ✅ 1.0× (275) | ✅ 1.5× (242) | ✅ 2.4× (124) | ✅ 3.3× (396) | ✅ 2.6× (897) | ✅ 1.6× (2,371) | ✅ 10.2× (409) | ✅ 2.7× (3,194) |
  | T | strided | 1,000,000 | 🟡 0.9× (4,217) | ✅ 1.8× (4,070) | ✅ 1.4× (2,281) | ✅ 3.1× (5,148) | ✅ 2.7× (9,117) | ✅ 1.8× (23,621) | ✅ 9.7× (4,402) | ✅ 2.9× (31,339) |

  Reading: 114 of the 144 cells are at least 1.5× NumPy, 26 between 1× and 1.5×, 4 below 1× (U strided starts at 100K
  0.80×, StringDType strided truth_and at 1K / 1M 0.98× / 0.95×, U strided truth_count at 1K 0.97×). The truth readers
  over S / U are 15–139× NumPy (NumPy's fixed-width bool cast builds a Python object per element). Where a cell is under
  1.5×, the cause is not the stage mechanism — a bare stage costs the engine call it runs (isalpha 0.93–1.0× of
  `np.strings.isalpha`, its noise both ways) — but one of three things, the first two outside E2's code:
  - the engine's own text function is near NumPy there: NumSharp's UNFUSED statements are 0.9–1.6× NumPy for find
    (np.strings.find, U / S at 100K+), 0.74× for starts over strided U at 100K, 1.1–1.4× for count_nonzero over strided
    text and about 1.4× for StringDType's truth cast — a stage runs exactly those functions, so it cannot beat them;
  - np.evaluate's numeric remainder kernels: `bool & (int64[::2] > 30)` (E1's open item: truth_and / strlen over strided
    operands) and a `Where` whose int64 comparison feeds a float64 select, which branches per element — over find's
    random mask the fused remainder loses to the unfused np.where (0.46× at 100K over a random mask, 2.0× faster over a
    patterned one; `benchmark/fusion/probes/evaltext_where_mask_probe.cs`), so the fused find tree is 0.70–0.86× of its
    unfused statements on contiguous data;
  - at 1K, NumPy's calls take 2–10 µs, NumSharp's text functions about the same, and np.evaluate's entry adds about
    0.3–0.5 µs over the bare engine call.
  Fused is faster than NumSharp's own unfused statements in 87 of the 144 cells.
- String oracle section `evaltext` (`test/oracle/gen_strings_oracle.py`, NumPy reference = the unfused statements):
  trees over text and numeric operands in the corpus layouts, strict.
- Benchmark (E1 as built; `benchmark/fusion/probes/evaltext_bench.cs`, its NumPy twin `evaltext_bench.py`, the table
  joined by `evaltext_sheet.py`): pinned to two P-cores (0x14), best of 7
  after a paused double warm-up, every fused result checked against NumSharp's unfused statements first. Trees: filter
  `(names == "paris") & (age > 30)`, count `CountNonzero(names == "paris")`, where `Where(names < "m", price * 2, price)`,
  compare `names == "paris"`; `names` over eight city names, `age` int64, `price` float64, strided = `[::2]` of a 2n
  array. Cells: NPY/NS (np.evaluate µs).

  | text | layout | n | filter | count | where | compare |
  |---|---|---|---|---|---|---|
  | U | contig | 1,000 | ✅ 5.2× (3.8) | ✅ 6.2× (2.9) | ✅ 7.4× (2.9) | ✅ 8.2× (2.2) |
  | U | contig | 100,000 | ✅ 14.0× (172) | ✅ 17.8× (133) | ✅ 11.0× (275) | ✅ 18.2× (129) |
  | U | contig | 1,000,000 | ✅ 10.1× (2,485) | ✅ 14.3× (1,690) | ✅ 7.8× (3,797) | ✅ 14.0× (1,705) |
  | U | strided | 1,000 | ✅ 4.8× (4.5) | ✅ 6.3× (3.0) | ✅ 5.5× (4.1) | ✅ 8.5× (2.2) |
  | U | strided | 100,000 | ✅ 7.9× (324) | ✅ 14.0× (170) | ✅ 4.0× (767) | ✅ 14.5× (163) |
  | U | strided | 1,000,000 | ✅ 5.2× (4,954) | ✅ 7.2× (3,310) | ✅ 3.2× (9,369) | ✅ 7.5× (3,155) |
  | S | contig | 1,000 | ✅ 7.0× (2.7) | ✅ 7.3× (2.5) | ✅ 6.8× (3.1) | ✅ 10.2× (1.7) |
  | S | contig | 100,000 | ✅ 18.1× (132) | ✅ 25.5× (92.1) | ✅ 11.1× (267) | ✅ 26.2× (88.6) |
  | S | contig | 1,000,000 | ✅ 18.2× (1,347) | ✅ 26.4× (888) | ✅ 9.5× (3,011) | ✅ 26.1× (892) |
  | S | strided | 1,000 | ✅ 5.1× (3.9) | ✅ 7.4× (2.4) | ✅ 5.3× (4.0) | ✅ 10.6× (1.6) |
  | S | strided | 100,000 | ✅ 10.3× (235) | ✅ 26.3× (89.1) | ✅ 4.1× (729) | ✅ 26.8× (86.3) |
  | S | strided | 1,000,000 | ✅ 9.6× (2,644) | ✅ 27.5× (856) | ✅ 3.9× (7,575) | ✅ 28.1× (829) |
  | T | contig | 1,000 | ✅ 2.9× (6.5) | ✅ 2.8× (6.1) | ✅ 3.2× (6.4) | ✅ 3.2× (5.2) |
  | T | contig | 100,000 | ✅ 2.2× (607) | ✅ 2.2× (569) | ✅ 4.4× (404) | ✅ 2.2× (563) |
  | T | contig | 1,000,000 | ✅ 2.2× (6,380) | ✅ 2.1× (5,937) | ✅ 3.8× (4,517) | ✅ 2.2× (5,737) |
  | T | strided | 1,000 | ✅ 2.4× (7.7) | ✅ 2.7× (6.2) | ✅ 2.7× (7.4) | ✅ 3.1× (5.3) |
  | T | strided | 100,000 | ✅ 1.9× (727) | ✅ 2.2× (581) | ✅ 1.8× (848) | ✅ 2.2× (577) |
  | T | strided | 1,000,000 | ✅ 1.9× (7,830) | ✅ 2.1× (6,007) | ✅ 2.0× (9,046) | ✅ 2.1× (6,026) |

  Reading: every cell beats NumPy, the lowest StringDType's strided `where` at 100K (1.8×); S / U run 3.2–28× NumPy,
  StringDType 1.8–4.4× (NumPy's StringDType comparison takes about half its `U` time, NumSharp's 1.9–4.4× its own `U`
  time). Against NumSharp's own unfused statements: `where` gains 1.1–2.0× (one pass over the branches); count and
  compare tie at 100K / 1M (0.95–1.09×) and pay np.evaluate's entry at 1K (0.2–0.5 µs); filter gains 1.0–1.26× at 1K
  and loses 0.63–0.99× at 100K / 1M (strided 0.63–0.95×). Filter's loss is its numeric remainder `stage & (age > 30)`,
  not the text stage: the numeric-only tree `m & (age > 30)` runs at 0.31–0.91× of its unfused statements, the same on
  the base commit 78b28c2b (§6, open).

## 6. State log

- 2026-11-04/05 — decisions D1–D3; NumPy matrix (§2); architecture measured (§3); E1 implemented (`2ab907a0`):
  `NDExpr.Text.cs` (the `string` leaf, `LowerText` on every node, `NDExprTextStage` / `NDExprTextPlan`),
  `DefaultEngine.Evaluate.Text.cs` (the per-call driver), `NDExprProgram.TextPlan` / `IsReduction`, the D20 refusals
  removed (binding, positional operands), `ComparisonNode.TruthTest` for the factories' `x != 0`. Gates:
  `NDEvaluateTextTests` (21 cases from NumPy output), the 440 earlier NDEvaluate/NDExpr tests unchanged but one (the
  check order now ends with the tree's own refusal), the string oracle section `evaltext` (2,949 cases, strict; the
  layout of np.evaluate's AXIS reductions is excused — a keepdims result is a reshaped view and a permuted child reduces
  in C order, np.evaluate's own reduction rule, pre-existing and the same over numbers — values, dtype and shape compared).
- 2026-11-05 — end-to-end benchmark (§5): 72 cells (S / U / T × contiguous / strided × 1K / 100K / 1M × four trees),
  every one faster than NumPy (1.8–28×). Re-gated on the final tree: the 461 NDEvaluate / NDExpr tests and the 53 strings
  doc tests on net8.0 and net10.0; the unit suite (CI filter) on net10.0 (20,951 passed) and net8.0 (20,950 passed, two
  runs; a first run met the suite-only intermittent `LongIndexedFallback_GivesTheManagedOutputs` failure known since
  2026-11-01 and then a test-host crash, neither in E1's code and neither again); the unit suite and `evaltext` under
  `NUMSHARP_DEBUG_GUARD_PAGES=1` on both (only the four pool-counting tests fail, as that mode bypasses the pool; no
  crash); the whole oracle project on net10.0 (382 passed; the 9 failures are the `[OpenBugs]` gap corpora, and the 2
  random-surface tests that look for the checkout from their output folder pass when run inside it); `evaltext` on
  net8.0.
- 2026-11-05 — E2 implemented (the owner's "Continue with phase E2"): the truth readers (truth stages, the broadcast
  rule), IsNaN, Cast to a number, the numpy.strings factories (`NDExpr.Strings.cs`), computed stage operands (E1's 0-d
  stand-in replaced). Gates: `NDEvaluateTextTests` 29 cases (E1's 21, two of them rewritten), the string oracle section
  `evaltext` 5,736 cases (E2's 2,787 after E1's; a second layout excuse, np.evaluate's own C / F result rule against a
  permuted temporary, 9 cases), the benchmark above (114 / 144 cells at 1.5× NumPy or more).
- Open (pre-existing, not text): np.evaluate's axis reductions return a keepdims result as a view (OWNDATA False) and a
  permuted child's reduction C-ordered, where NumPy's np.sum / np.any allocate the keepdims shape in K order.
- Open (pre-existing, not text): np.evaluate's fresh elementwise result is C- or F-ordered (F only when every input is
  strictly F-contiguous), where NumPy keeps a permuted operand's K order (`np.evaluate((NDExpr)x * 2)` of a 3-D
  transpose is C, NumPy's `x * 2` keeps the permutation).
- Open (pre-existing, not text): np.evaluate's fused `Where` whose condition compares an int64 operand and whose branches
  are float64 branches per element: over a random mask it is 0.46× of the unfused np.where at 100K (2.0× faster over a
  patterned mask; `benchmark/fusion/probes/evaltext_where_mask_probe.cs`).
- Open (engine performance, owner's call): the text functions the stages run are under 1.5× NumPy in places — np.strings.find
  over U / S at 100K+ (0.9–1.6×), startswith over strided U (0.74× at 100K), count_nonzero over strided text (1.1–1.4×),
  StringDType's truth cast (about 1.4×).
- Open (pre-existing, not text): np.evaluate's fused kernel for `m & (age > 30)` (a bool and an int64 comparison, the
  filter's remainder) is slower than the unfused statements: 0.64× contiguous / 0.34× strided at 100K, 0.91× / 0.52× at
  1M on the base commit 78b28c2b, the same on E1 (`benchmark/fusion/probes/evaltext_remainder_probe.cs`); the text
  filter cells inherit it.
