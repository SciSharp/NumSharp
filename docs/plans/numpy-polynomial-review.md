# Review of U3 (cb64deb1) and the coverage oracle join (ea32af61): continuation plan

Status as of 2026-09-26, branch `journey4`, HEAD `af3a70a1`. The review was asked for as: "review what you
already implemented, see that it is clean and byte perfect as oracle enforces". It covers two commits:

- `cb64deb1`: the numpy.polynomial evaluation family (U3), plan [numpy-polynomial.md](numpy-polynomial.md).
- `ea32af61`: the per-API NumPy-oracle parity join in `coverage/`.

One real parity bug was found in U3 (section 2). The coverage join is correct, but it has a weak replay guard
and a few documentation gaps (section 4). Nothing from this review is committed yet except this file.

## 1. What is verified (on af3a70a1, isolated worktree)

The shared tree does not build: another session has uncommitted `RandomSampling` work in it
(`RandomConstraints.cs`, `Generator.Broadcast.cs`, `RandomBroadcast.cs`). So every run used a clean detached
worktree at HEAD, built Release for both TFMs:
`C:\Users\ELI\AppData\Local\Temp\claude\K--source-NumSharp\fdefc0e8-1f2c-45b6-893d-51e9d363bc9e\scratchpad\wt_review`.

| Gate | Result |
|---|---|
| `python coverage/generate_coverage.py` | join clean (0 problems), 251,744 contracts. Headline 492/560 oracle-verified (87.9 %). Expanded 779/2519. 45 available headline APIs have no contract. The 28 new Generator samplers (fa4bf379) are verified-direct via `grnd`; `Generator.multivariate_normal` is available but unverified. |
| FuzzMatrix, net8.0 + net10.0 | 137 passed, 4 skipped (the OpenBLAS tiers: BlasBackendDelta, MatmulParity, LinalgParity, Corpus_BackendOps_LeaveNoUndisposedIntermediates), 0 failed. Polyeval tier green. |
| Polynomial unit tests, both TFMs | 50 passed, 1 skipped (`Frequency_ParabolicAndPolynomialCompanionAreFunctional`, not U3). |
| Fresh U3 probe, 27,856 cases, both TFMs | 27,855 byte-exact (NaN tokenized, as the polyeval tier compares). 1 divergence, section 2. |

What the fresh probe covers:

- T1: uint64 x at the float64 rounding ties above 2**63, every vector-remainder length, three layouts, every
  series dtype. This is the .NET 8 double-rounding class that ebe18b7c fixed in `EmitConvertTo`, which the U3
  emitter calls.
- T2: int64 x near +-2**63.
- T3: weak Python ints at float64 / float32 / float16 ties.
- T4: 0-d uint64 / int64 x.
- T5: uint64 `val2d` / `grid2d`.
- R: 15,000 seeded random cases with full-mantissa values. They cover every base, every kind (`val`, per-point,
  N-D, coefficient layouts, weak scalars, `val2d`/`val3d`/`grid2d`/`grid3d`/`valnd`), 13 x 13 dtypes, x layouts
  and mixed ordinate dtypes.

The committed tier's moderate values are short dyadic rationals, so most products there are exact; the random
full-mantissa values are what exercise rounding order.

## 2. The finding: NumPy has a THIRD complex multiply, and U3 reproduces only two

The divergent case is `hermite_e.hermeval2d(x, y, c)`:

- x is complex128 of shape (1,), bits `(0xbfc6edc46f5f2c30, 0xbffa97b3ccf45646)`;
- y is float32 of shape (1,), `[0.82701397]`;
- c is uint32 of shape (4, 1), `[[4],[10],[39],[25]]`.

The real part differs by 1 ULP: NumPy gives `0xc057379f5d0bdbaf`, NumSharp `...bdbb0`. It reduces to one call:
`hermeval(x, c_f64 of shape (4,1))` gives `...baf` in NumPy, while the same series with shape `(4,)` gives
`...bb0` (NumSharp's answer).

Root cause, from NumPy's source:

- `refs/numpy/numpy/_core/src/umath/loops_arithm_fp.dispatch.c.src` holds the complex `@TYPE@_multiply`. Its
  vector body (`simd_cmul`) is skipped for `b_sdst == 0` (line 348), which goes to `loop_scalar` (line 511).
  That loop is `a_r*b_r - a_i*b_i`, `a_r*b_i + a_i*b_r`, compiled by MSVC in the AVX2+FMA dispatch target, so
  it is CONTRACTED.
- `refs/numpy/numpy/_core/src/umath/ufunc_object.c`, `try_trivial_single_output_loop`: this returns -2
  (use NpyIter) when the non-0-d operands differ in ndim. 0-d operands are exempt.
- `refs/numpy/numpy/_core/src/multiarray/nditer_constr.c` line 1594 has `if (bshape == 1) strides[iop] = 0;`.
  A single-element iteration therefore hands the inner loop stride 0 for EVERY operand, the output included.

So NumPy 2.4.2 on win-amd64 has three complex128 products:

| Path | re | im |
|---|---|---|
| scalarmath (NumPy scalars) | `ar*br - ai*bi`, all rounded (naive) | `ar*bi + ai*br`, naive |
| ufunc vector `simd_cmul` (every array op, size >= 1, trivial or NpyIter) | `fma(ar, br, -(ai*bi))` | `fma(ar, bi, ai*br)` |
| ufunc `loop_scalar`: NpyIter single-element iteration, reached when the non-0-d operands differ in ndim and the broadcast result has size 1 | `fma(ar, br, -(ai*bi))` | **`fma(ai, br, ar*bi)`** |

This was measured in `scratchpad/oracle/cmul_forms.py` over 6,000 random pairs:

- the trivial loop and the vector body match simd every time;
- scalarmath matches naive every time;
- the `(1,1) x (1,)` single-element product has re always equal to simd's form and im always equal to
  `fma(ai, br, round(ar*bi))`.

Where U3 meets it: only `NDPolyEval.Evaluate`'s N-D-series, per-point-x branch (`coefOperand && PerPoint`),
when all four conditions hold:

- the result has size 1;
- the loop is complex (tx or tc is Complex; the step tables hold no complex constants);
- `ndim(c[k])` differs from `x.ndim`. Here `ndim(c[k]) = (c.ndim - 1) + (tensor ? x.ndim : 0)`, so with
  `tensor=True` the coefficient side is ALWAYS deeper;
- the multiply is complex x complex between a coefficient-derived value and an x-derived value.

Products with a real-cast operand agree under every form, so only genuinely complex pairs discriminate.

**Beyond U3 (to verify, likely library-wide).** `loop_scalar` is plain `np.multiply` machinery, so NumSharp's
general complex multiply is probably affected as well. `NDComplexMath.Multiply` always uses the simd form
(commit dbc0b3b3). NumPy would then differ from it by 1 ULP (imag, then everything downstream) on two
kinds of call:

- a complex `*` of two non-0-d arrays with different ndims and a single-element broadcast result
  (`np.array([[a]]) * np.array([b])`);
- the genuine src/dst overlap case that also goes to `loop_scalar`.

The memory note for dbc0b3b3 calls that tail "naive". The measurement above says it is contracted (re: simd
form, im: `fma(ai, br, ar*bi)`). This is a separate item from the U3 fix: step 10 of section 3.

Which multiplies take the third form, with dc = ndim(c[k]) and dx = x.ndim:

- **dc > dx** (every `tensor=True` case, and `tensor=False` with a deeper series): every `c1 * <x-derived>`
  and polyval's `c0 * x`. The form is constant across steps.
- **dc < dx** (`tensor=False`, x deeper): only the products whose c1 is still the raw `c[-1]` take it. That is
  the first Clenshaw step (nc >= 3), or the final `c1*x` for nc == 2. Once c1 has absorbed x it has x's ndim, and
  the trivial loop takes over (simd form). Horner never takes it (`c0 = c[-1] + x*0` already has x's ndim).
- **dc == dx**: never (the current behaviour, correct).

Rule check: `scratchpad/oracle/unit_rank_rule.py` replays each basis's NumPy lines on Python complex values,
tracks each value's ndim and picks the form per multiply. It reproduced NumPy bit for bit for poly, cheb, herm
and herme, in all 8 shape configurations (including the dc < dx step-dependent one), for nc in
{1,2,3,4,5,7,12}, 150 random each.

**leg/lag: not yet confirmed.** They mismatched in EVERY configuration, the equal-ndim ones included. That
points to a bug in the emulator, not in the rule: its `div` uses Python's `/`. NumPy's `CDOUBLE_divide` by the
real `nd` is Smith's algorithm: `rat = 0`, `scl = 1/nd`, `out = (in1r + in1i*rat)*scl`, i.e. a RECIPROCAL
MULTIPLY. U3's kernel already implements this (`PolyLaneOps.s_cDivPrep`/`s_cDivBy`, corpus-gated). Fix the
emulator's `div` and re-run before implementing.

Separately, a known #569 lane appeared while probing. A grid whose ordinate is a weak Python complex, meeting a
float16/float32 series, has a complex64 INTERMEDIATE; a later float64 ordinate then lifts it to complex128.
NumPy's values then have complex64 precision (sometimes NaN from complex64 overflow); NumSharp has one complex
width and computes in complex128. Example: `chebgrid2d(1+2j, y_f64, c_f32)`. This is not a U3 bug, and the
committed tier never emits it. The probe skips it (`grid_has_complex64`). Optional: pin it as a `[Misaligned]`
unit test.

## 3. Next steps for the U3 fix (in order)

1. **Confirm the rule for leg/lag.** Replace the emulator's `div` with NumPy's Smith branch:
   `|in2r| >= |in2i|` gives `rat = in2i/in2r`, `scl = 1/(in2r + in2i*rat)`, `re = (in1r + in1i*rat)*scl`,
   `im = (in1i - in1r*rat)*scl`; otherwise the mirrored branch. Expect all 336 configurations exact.
   If leg/lag still differ with equal ndims, stop and re-derive.
2. **Kernel key.** Add a field to `PolyEvalKey`, e.g. `enum PolyUnitBroadcast : byte { None, CoefDeeper,
   PointsDeeper }`. Its doc must say it is NumPy's NpyIter single-element iteration with stride 0 and
   `loop_scalar`. `NDPolyEval.Evaluate` sets it in the `coefOperand && PerPoint` branch when
   `y.size == 1 && (tx == Complex || tc == Complex)` and dc != dx; otherwise it is None, so no new kernels
   exist for other calls. Thread it through `GetPolyEvalKernel` / `CompilePolyEval` / the DynamicMethod name.
3. **Rank tracking in the emitter.** Give `PolyValue` a NumPy-ndim tag, tracked only in this mode. Use
   representative ndims, because only comparisons matter:
   - CoefDeeper: Ck = 2, X/X2 = 1;
   - PointsDeeper: Ck = 1, X/X2 = 2;
   - weak constants 0; op result = max; `Convert`/`CopyTo` keep the tag.

   In `PolyEmitter.Emit`, a `BinaryOp.Multiply` at `NPTypeCode.Complex` whose operand tags are both non-zero
   and different emits a new helper, `PolyLoopScalarComplexMultiply(a, b)`:
   `new Complex(Math.FusedMultiplyAdd(a.Re, b.Re, -(a.Im*b.Im)), Math.FusedMultiplyAdd(a.Im, b.Re, a.Re*b.Im))`.
   Document it next to `PolyNaiveComplexMultiply`, with a `MethodInfo` cached the same way. `PolyLoadCoef`
   tags Ck; the per-point `env.X` is tagged X; `Pre` (x2) inherits X through the op.
4. **Peel to the joint fixpoint.** `PolyTyping.PeelCount` must also iterate the ndim state `(n0, n1)` in
   PointsDeeper mode: it goes `(dc,dc) -> (dc,dx) -> (dx,dx)`, i.e. 2 steps for Clenshaw; Horner stays 0.
   - The peel cache `s_polyPeel` and `PolyTyping.Class` must include the mode.
   - In the loop body, assert the carried tags do not change, next to the existing dtype assert. A wrong peel
     then throws instead of silently mixing forms.
   - `ResultType` is unaffected, because more straight steps past the dtype fixpoint do not change dtypes.
5. **One point, scalar only.** In this mode emit no vector parts and no U-chain stage, exactly as `ScalarMath`
   does (`vecB`/`vecL` false; `Part` returns `rest`). Performance is irrelevant for a size-1 result.
6. **Oracle.** Add a gen_polyeval block (`test/oracle/gen_oracle.py`) for single-element broadcasts:
   - shapes: c tails `(1,)`, `(1,1)`; x `(1,)`, `(1,1)`, `(1,1,1)`; `tensor` True/False;
   - dtypes: complex x with f64/c128 series, and real x with c128 series;
   - every base; nc in {1,2,3,4,5,7};
   - `val2d`/`val3d`/`grid2d` with `(1,)` ordinates, which reach this mode in their second/third pass.

   Values must be random full-mantissa, not `_poly_fill`'s moderate ones, or the forms will not discriminate
   (use a seeded `np.random.default_rng`). Then:
   - regenerate with `python test/oracle/gen_oracle.py polyeval`;
   - raise the `polyeval.jsonl` floor in `FuzzCorpusTests.MinCases` if one exists;
   - run `dotnet build` to copy the corpus.

   Also add unit tests with NumPy-probed literal bits: the (4,1) vs (4,) pair above, plus one PointsDeeper case
   where the forms change between steps.
7. **Mutation check.**
   - Force the simd form in this mode: the new corpus cells must go red.
   - Force `loop_scalar` for equal ndims: the existing tier must go red.
   - Drop the peel extension: the loop-body assert must throw.
8. **Docs.**
   - The Emitter.cs header "COMPLEX MULTIPLY: TWO NUMPY SEMANTICS" becomes three, with the source lines above.
   - `docs/plans/numpy-polynomial.md`: add a U3 traps line.
   - `.claude/CLAUDE.md`'s polynomial traps: the "Two complex multiplies" bullet becomes three.
   - Every new member gets full XML docs (user rule).
9. **Re-run everything** on both TFMs: the probe (section 5), FuzzMatrix and the polynomial unit tests.
10. **Separate follow-up: the library-wide multiply** (section 2, "Beyond U3").
    - Probe NumSharp `np.multiply` on complex128 for `(1,1) x (1,)`, `(1,) x (1,1,1)` and an in-place overlap
      (`np.multiply(a[1:], a[:-1], out=a[1:])`) against NumPy's bits.
    - If they differ, it is a core fix, ported with the same helper and the same selection rule (trivial loop
      vs NpyIter single element / overlap), plus its own oracle cells in the `binary` / `out_where` tiers.
    - Keep it out of the U3 commit.

## 4. Coverage-join cleanups (ea32af61)

1. **Replay guard is weaker than documented.** `coverage/oracle_evidence.py::_replayed_files` counts ANY
   `"<name>.jsonl"` literal in `FuzzCorpusTests*.cs` / `IndexOracleTests.cs`. That includes the `MinCases`
   floor table (`["aliasing.jsonl"] = 70`), so a corpus file whose only mention is a floor entry would pass as
   "replayed".
   - Match replay CALLS instead: `RunCorpus("…")`, `RunHostLibmCorpus("…")`, `RunMaCorpus("…")` and
     `RunHostLibmMaCorpus("…")` (FuzzCorpusTests.Ma.cs). Inspect `IndexOracleTests.cs` for its own loader
     call.
   - Keep the `"regressions"` directory special case.
   - Add a synthetic test in `coverage/test_oracle_evidence.py` where a file named only in a floor table is
     NOT replayed.
   - Then update the docstring and `coverage/README.md`'s strictness list.
2. **Missing docs (user rule):**
   - JSDoc for `isOracleVerified`, `oracleTitle`, `oracleSummary` and `oracleSection` in
     `docs/website-src/docs/coverage-support-dashboard.md`;
   - docstrings for `identity_predicate`'s nested `resolve` / `identical`;
   - a docstring for `CommittedCorpusTests.evidence`;
   - comments on the `coverage/test_dashboard.cjs` helpers `direct` / `aliasVia` / `oracleRows`.
3. **Stale comment in U3.** `ILKernelGenerator.Polynomial.Lanes.cs` line 31 says `ConvertToDouble` was
   "verified against the scalar cast on 16M wide-magnitude values". Before ebe18b7c that scalar cast
   (`conv.r.un`) double-rounded on .NET 8 above 2**63, so it is the wrong reference. Reword: the vector
   conversion rounds once (hi/lo split), as NumPy does, and probe T1 gates it at the ties. Also scan the U3
   files for TODOs, dead code and other stale comments.
4. **Re-verify:**
   - `python -m unittest discover -s coverage` (expect 32+);
   - `node --test coverage/test_dashboard.cjs` (15);
   - `python coverage/generate_coverage.py --check` (byte-identical re-render);
   - the mutation runner `scratchpad/oracle/mutate.py` (16/16 killed at ea32af61; add a mutant for the tightened
     guard).

## 5. Tools and how to reproduce (scratchpad; not in the repo)

The scratchpad is `C:\Users\ELI\AppData\Local\Temp\claude\K--source-NumSharp\fdefc0e8-1f2c-45b6-893d-51e9d363bc9e\scratchpad`.
It is a temp dir, so if it is gone, rebuild the tools from the descriptions here.

- `oracle/probe_polyeval.py`: the fresh probe (blocks T1-T5 + R, section 1). It imports gen_oracle's helpers,
  so it emits the polyeval schema verbatim. Usage:
  `NS_ORACLE_ROOT=<repo or worktree> python probe_polyeval.py <out.jsonl> [seed=20260926] [n_random=15000]`.
  Replay: copy the output to
  `<worktree>/test/NumSharp.Tests.Oracle/bin/Release/<tfm>/Fuzz/corpus/regressions/`, then run
  `dotnet test test/NumSharp.Tests.Oracle/NumSharp.Tests.Oracle.csproj -c Release --no-build --framework <tfm> --filter Name=FuzzRegression`.
  The committed tree has no regressions folder, so never copy the probe into the source tree.
  Two probe traps, both fixed:
  - `rng.choice` over huge Python ints builds a float64 array, and `int()` of `2**64-1` from it is `2**64`
    (no C# primitive). Index instead.
  - The complex64 grid intermediates (section 2) must be skipped.
- `oracle/cmul_forms.py`: classifies NumPy's complex product per ufunc path against the naive / simd / swapped
  forms (exact `Fraction` arithmetic).
- `oracle/unit_rank_rule.py`: the rule emulator, per basis, with ndim tracking. Its `div` bug is the item in
  section 3, step 1.
- `oracle/mutate.py`: the coverage-join mutation runner from ea32af61.
- `wt_review/`: a detached worktree at af3a70a1, Release-built for both TFMs. Remove it with
  `git worktree remove` when the review is committed.

## 6. Done when

- NumPy's third form is reproduced.
- The new single-element-broadcast oracle cells are byte-exact on both TFMs.
- The probe replays 27,856/27,856.
- The coverage cleanups are in, with their tests green.
- Only this review's files are committed (`git commit --only`); the parallel session's RandomSampling WIP is
  never included.
