# NumSharp's resize refusal carries NumPy's embedded line wraps — single-line message matching silently fails

**Status:** not a defect — deliberate byte-level NumPy parity. Recorded here because it is a
standing trap for anyone matching the message: test assertions, greps, log pipelines, and docs
that quote it as one line all miss, and the failure mode looks like the message simply isn't
there.

**Found:** 2026-07-22 — as the only two test failures of the interop documentation gate's first
run. **Observed on:** NumSharp `interop-pythonnet` @ `d970424d` · net8.0 & net10.0; message
verified against NumPy 2.4.2 source.

---

## Overview

When `ndarray.resize(...)` must reallocate and the buffer is shared with another array (and
`refcheck` is on), NumSharp throws `IncorrectShapeException` with NumPy's message reproduced
**byte-for-byte — including the two newlines NumPy's C source embeds mid-sentence**:

```text
cannot resize an array that references or is referenced
by another array in this way.
Use the np.resize function or refcheck=False
```

The first line break falls *inside the sentence*, between "referenced" and "by". Any matcher
written from the rendered sentence — `"...references or is referenced by another array..."` with
a space — does not match the real string, because the real string has `\n` where the reader's eye
put a space.

## Reproduction

```csharp
var nd = np.arange(8).astype(NPTypeCode.Double);
var secondReference = nd["2:"];          // any second view of the block (or a live Python export)

nd.resize(new Shape(16));                // IncorrectShapeException, the 3-line message above
```

The same message guards a buffer pinned by a Python-side export
(`NDArrayPythonInterop.ToNumpy(nd)` held by Python → `nd.resize` refuses identically).

## Expected (by the naive matcher)

```csharp
.Should().Throw<IncorrectShapeException>()
    .WithMessage("*cannot resize an array that references or is referenced by another array in this way*");
```

## Actual

Both gates that used the single-line wildcard failed on the suite's first run:

```text
Failed Contract_RefcheckGuard_SeesOtherReferencesToTheBlock
  Expected exception message to match the equivalent of "*cannot resize an array that references
  or is referenced by another array in this way*", but "cannot resize an array that references or is referenced
  by another array in this way.
  Use the np.resize function or refcheck=False" does not.
```

The wildcard `*` was irrelevant — the mismatch is the pattern's literal space against the
message's literal `\n`.

## Root cause (why the message is shaped this way)

NumSharp's project rule is to match NumPy exactly, error texts included. NumPy's own C source
wraps the string across source lines **with explicit `\n`s in the literal**, so the runtime
`ValueError` genuinely contains them — this is not a formatting artifact of any printer:

- **NumPy:** `src/numpy/numpy/_core/src/multiarray/shape.c` (vendored NumPy 2.4.2), the
  `refcheck` branch around lines 101–105:

  ```c
  "cannot resize an array that "
  "references or is referenced\n"
  "by another array in this way.\n"
  "Use the np.resize function or refcheck=False"
  ```

  (A sibling variant at lines ~82–85 — used on a different path — wraps once and ends
  `"...in this way. Use the np.resize function."`.)

- **NumSharp:** `src/NumSharp.Core/Manipulation/NDArray.resize.cs` (lines ~96–99) reproduces the
  refcheck variant with the same `\n`s.

Corroborating trivia: NumPy's own docstring machinery dodges the wrap too —
`numpy/_core/_add_newdocs.py:4248` quotes the error as
`ValueError: cannot resize an array that references or is referenced ...`, truncating with an
ellipsis exactly at the newline.

Note the contrast with the *other* resize refusal, which is single-line and trap-free:
`cannot resize this array: it does not own its data` (the non-owning-view guard, same file,
line ~92).

## Workaround — the house matching pattern

Never span the wrap. Assert (or grep) **fragments that each live on one line**:

```csharp
.Should().Throw<IncorrectShapeException>()
    .WithMessage("*cannot resize an array that references or is referenced*")
    .WithMessage("*by another array in this way*");
```

For shell pipelines, match a single fragment (`grep "references or is referenced"`), or use a
multiline-capable matcher with the exact `\n`s.

Rule of thumb this incident produced for future gate authors: **before pinning any NumPy-parity
error text, check the vendored `src/numpy` C string for embedded `\n`** — if the literal wraps,
assert fragments.

## Where it is handled today

- Gates asserting fragments:
  `test/NumSharp.Interop.UnitTests/DocExamples.InteropIndexPage.cs` →
  `Contract_RefcheckGuard_SeesOtherReferencesToTheBlock`;
  `test/NumSharp.Interop.UnitTests/DocExamples.PythonnetNumpyPage.cs` →
  `Troubleshooting_SymptomsAreVerbatim`, `Lifetime_ALiveConversionLocksResizing_BothSides`;
  `test/NumSharp.Interop.UnitTests/DocExamples.NpFrombufferPage.cs` →
  `LiveView_LocksTheSourceAgainstResize`.
- The interop docs (`docs/website-src/docs/interop/index.md`, `pythonnet-numpy.md`) quote the
  sentence inline with the wrap collapsed — deliberate, for readability in prose and table cells;
  the transcripts and gates carry the truth, and a reader greping docs by the first fragment
  still lands on the right rows.

## Related

- `docs/bugs/pythonnet-decoder-cache-poisoning.md` — the other discovery from the same
  gate-writing session
- `src/NumSharp.Core/Manipulation/NDArray.resize.cs` — both refusal messages
- Project principle (see `.claude/CLAUDE.md`): error texts are part of the NumPy-parity surface —
  the fidelity that creates this trap is intentional and must not be "fixed" by unwrapping the
  message.
