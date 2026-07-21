# Triaging a divergence (a red FuzzMatrix case)

When a corpus case doesn't bit-match, the harness reports the divergent cell(s) and **auto-shrinks** the failure to
a minimal (often 1-element) repro. Your job: classify it as a **real bug**, an **intended difference**, or a
**harness/generator mistake**. Never leave a red case silently excused.

## What "bit-exact" means here

`BitDiff` compares the raw C-contiguous bytes of the result, plus dtype and shape, with two deliberate relaxations:
- **NaN is tokenized** — any NaN matches any NaN of the same width (NumPy and NumSharp needn't agree on NaN payloads
  unless a test specifically pins them).
- **Decimal compares by canonical VALUE**, so `1.0m ≡ 1.00m` (scale differences don't count as divergence).

Everything else — including signed zero (`-0.0` ≠ `0.0`), integer wrap, float rounding — must match to the bit.

## Decision tree

1. **Reproduce and read the shrunk case.** The harness prints the operand(s), the op+params, NumPy's expected
   bytes, and NumSharp's actual bytes. Rebuild the exact call in a `dotnet run` script (see the project
   `dotnet-run-script` guidance) and run the SAME call in Python against numpy 2.4.2. Confirm which side is "right"
   — or whether the question is ill-posed, see 5.

2. **NumSharp is wrong → it's a real bug.** Fix the op. If you can't fix it now, carve the shrunk case into an
   `[OpenBugs]` reproduction (`OpenBugs.cs`, or a focused file like `OpenBugs.DtypeCoverage.cs` / `OpenBugs.Char.cs`)
   so CI excludes it but it's tracked and un-silenced. Do NOT excuse a real bug in `MisalignedRegistry`.

3. **The difference is intended and defensible → excuse it in `MisalignedRegistry.cs`.** This is for documented,
   deliberate NumSharp-vs-NumPy differences (e.g. a dtype NumSharp handles differently by design, an error-text
   divergence). Add an entry keyed to the op/case with a one-line rationale. The gate then treats it as expected,
   not a pass and not a failure — and it shows up in the divergence ledger, never silently.

4. **The generator/registry is wrong → fix the corpus, not the excuse.** Common causes:
   - Wrong `OpRegistry` mapping (e.g. routed to the wrong overload, or read the wrong param key).
   - A job whose Python lambda and C# case don't actually compute the same thing.
   - A params dict that doesn't capture everything the C# side needs to reconstruct the call.
   Fix it, regenerate the tier, re-run.

5. **Neither side is "right" → NumPy's answer is host-dependent.** Before blaming either side, check whether the
   expectation is even reproducible. Re-run the same Python on a *different* host (or just compare the vectorized
   and scalar paths — `np.array([x]*8).astype(dt)[0]` vs `np.<src>(x).astype(dt)`). If NumPy contradicts itself,
   the case is asserting undefined behaviour, not a contract, and no implementation can pass it. The known family
   is **float→integer conversion of NaN / ±inf / out-of-range values** (`astype`, and `reciprocal` on an integer
   dtype containing 0 — NumPy computes it as `(T)(1.0/0)`), where glibc/gcc and MSVC disagree. Fix the *generator*
   so it stops emitting the undefined value class — do **not** excuse it in `MisalignedRegistry` (the divergence
   isn't a NumSharp behaviour) and do **not** retune NumSharp to one host. See `Fuzz/README.md` →
   "Host-dependent values"; `fuzz_random.py` already defuses this class and `assert_portable` audits it.

## Error parity

Cases can also assert **error parity** — NumPy raising must correspond to NumSharp raising. The generator records
raising cases it chooses to keep (many are skipped by `try/except`); the harness checks NumSharp raises too. If
NumSharp succeeds where NumPy raised (or vice versa), that's a divergence to classify like any other. Verbatim
error-message matching is generally handled by dedicated unit tests, not the byte corpus.

## The known teardown crash is NOT a divergence

A full `TestCategory=FuzzMatrix` run can end "Test host process crashed" (`AccessViolation`) after every test
reported Passed. That's an intermittent teardown crash, not a red case. Re-run the specific `FuzzCorpusTests` class
(it exits 0 cleanly) to confirm the tier is actually green.

## Ledger

The complete, human-readable divergence ledger is `test/NumSharp.UnitTest/Fuzz/README.md`. Keep it and
`MisalignedRegistry.cs` in sync when you excuse something.
