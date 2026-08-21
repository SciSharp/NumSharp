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
   divergence, a bounded transcendental ULP gap). The mechanism is a branch in `MisalignedRegistry.Classify(...)`
   that returns a one-line reason string (`null` = not excused = red); the runner counts and PRINTS each excused
   reason per tier, so it is never silent. Three rules, all learned the hard way:
   - **Scope the branch to the exact `(op, dtype, kind)` cell.** A blanket "any complex value diff" once excused a
     gross complex-matmul regression. Match the op name, the `tc`, the `DivergenceKind`, and (for ULP gaps) a tight
     tolerance — anything broader lets a neighbouring regression through.
   - **Guard every ULP/near-miss branch with `diffs.Count > 0`.** `diffs.All(...)` is VACUOUSLY true on an empty
     diff list, so an unrelated divergence (error-text, wrong arity) would otherwise be silently excused as
     "within N ULP".
   - **Pin the scope from BOTH sides** in `test/NumSharp.Tests.Oracle/OpenBugs.FuzzGate.cs`
     (`MisalignedRegistryTightnessTests`): a paired NOT-excused test (a gross regression in the neighbouring cell →
     `null`) and STILL-excused test (the documented divergence → non-null), so a future re-broadening turns red.
   The gate then treats it as expected, not a pass and not a failure — keep the human-readable ledger
   `test/NumSharp.Tests.Oracle/Fuzz/README.md` in sync.

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

Cases can also assert **error parity** — NumPy raising must correspond to NumSharp raising. Two tiers, deliberately
different in strength:
- **Weak ("threw something"):** `errors.jsonl`, plus any case flagged `expects_throw` with no recorded exception.
  NumSharp must throw *anything*; type and message are not checked. NumSharp succeeding where NumPy raised (or vice
  versa) is a divergence to classify like any other.
- **Message parity:** `errors_full.jsonl`, plus any case carrying `error: {type, text}` (recorded verbatim at
  generation time — the cells every value tier skips). `CheckError` (in `FuzzCorpusTests.Kinds.cs`) holds NumSharp to
  BOTH the exception type (via the NumPy-class → .NET-type map `ErrorTypeMap`; identical names like
  `ValueError`/`AxisError` always match) AND the message verbatim, after `NormalizeMessage` strips .NET's
  `" (Parameter 'x')"` framing. **So verbatim error text IS gated by the corpus now** — this is no longer only a
  unit-test concern.

A message mismatch routes through `MisalignedRegistry` as an `ErrorText` divergence, so a documented wording gap is
excused-but-printed, not silently accepted.

## The known teardown crash is NOT a divergence

A full `TestCategory=FuzzMatrix` run can end "Test host process crashed" (`AccessViolation`) after every test
reported Passed. That's an intermittent teardown crash, not a red case. Re-run the specific `FuzzCorpusTests` class
(it exits 0 cleanly) to confirm the tier is actually green.

## Host-pinned tiers go Inconclusive, not red

`matmul_parity` and `random_parity_host` record bytes reproducible only on a specific host — the exact BLAS build +
DYNAMIC_ARCH kernel + thread count NumPy used, or the win-amd64 CRT libm. When the host doesn't match (checked via
`MatmulParityPin` by BLAS binary SHA-256 / core name, or an OS check), the tier asserts **`Inconclusive`**, never
red — a machine without NumPy's wheel has nothing to be wrong about. Seeing "Inconclusive" on these two tiers
off-host is expected, not a failure; regenerate the corpus on your host (`python gen_oracle.py matmul_parity` /
`random_parity`) to gate against your machine.

## Ledger

The complete, human-readable divergence ledger is `test/NumSharp.Tests.Oracle/Fuzz/README.md`. Keep it and
`MisalignedRegistry.cs` in sync when you excuse something.
