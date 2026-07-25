# Specification — the interop documentation set

**Status:** agreed, pre-writing. This document fixes *how* the four interop pages are written
before a word of them exists. It is the contract; the pages are the deliverable.

---

## 1. Scope

Four pages, replacing the entire existing `docs/website-src/docs/interop/` folder.

| # | File | Subject |
|---|---|---|
| 1 | `docs/website-src/docs/interop/index.md` | The interop landing page: what a NumSharp bridge is, the contract every bridge implements, the bridge table |
| 2 | `docs/website-src/docs/interop/pythonnet-numpy.md` | The `NumSharp.Interop.pythonnet` package: verbs, view-vs-copy, layout fidelity, lifetime, codec, GIL, dtypes, versions |
| 3 | `docs/website-src/docs/interop/np-frombuffer.md` | Reaching **any** Python library through the buffer protocol — `ToMemoryView` + `np.frombuffer` and friends |
| 4 | `docs/website-src/docs/interop/numpy-net.md` | Coexistence and migration with SciSharp's `Numpy` / `Numpy.Bare` (Numpy.NET) |

### 1.1 Clean slate

The four files currently in that folder — `index.md`, `pythonnet.md`, `zero-copy-model.md`,
`numpy-net.md` — are **deleted**, not rewritten. No sentence, table, heading or example is carried
over. They are prior art, not source material. The new pages are written from:

- the package source (`src/NumSharp.Interop.pythonnet/`, 7 files, 2,469 lines),
- the test suite (`test/NumSharp.Interop.UnitTests/`, 21 files),
- the measurements recorded in §9 of this spec, taken live this session.

`zero-copy-model.md` has no successor page: the view-vs-copy decision belongs inside page 2, and
page 3 restates the part of it a buffer-protocol reader needs.

### 1.2 Table of contents

`docs/website-src/docs/toc.yml` — the `Interoperability` node is rewritten to:

```yaml
- name: Interoperability
  href: interop/index.md
  expanded: false
  items:
  - name: Python & numpy (pythonnet)
    href: interop/pythonnet-numpy.md
  - name: Any library via np.frombuffer
    href: interop/np-frombuffer.md
  - name: Numpy.NET
    href: interop/numpy-net.md
```

---

## 2. Page anatomy

Every page is assembled from the same parts, in this order. Parts marked *optional* appear only
when the material warrants them.

```
# Title — one-line promise

Thesis paragraph.

**On this page:** [link] · [link] · [link]

> Verification banner (blockquote).

---

## Section                       ← as many as the material needs
### Subsection                   ← question, or noun for reference
    claim block
    claim block
---

## Troubleshooting               ← optional; page 1 has none
## Claims ledger
## See also
```

### 2.1 Section count

**There is no fixed tier count.** Sections are content-driven: a page gets as many `##` sections as
its material genuinely divides into, ordered so a reader can stop reading at any section boundary
and still have something usable. The first section after the banner is always the shortest path to
working code.

### 2.2 Headings

| Level | Rule |
|---|---|
| `#` | Exactly one, first line. Form: `Subject — the promise in one clause`. |
| `##` | Sentence case. Names a body of material, not a tier. |
| `###` | **A noun or verdict phrase naming the topic** ("Raw layout access", "Read-only sources", "The cost curve"), for behavioural and reference subsections alike. A question is the rare exception, kept only where it is genuinely what a reader would search ("Why not just `ToNumpy()`?") — at most one or two per page. |
| `####` | Avoid. If a `###` needs subdivision it is two `###`s. |

Whatever the heading's form, the subsection opens with the answer — **first sentence, verdict in
bold**:

```markdown
### Copy behaviour

**Nothing is copied — and the gap is two orders of magnitude.** `np.frombuffer` reports
`OWNDATA = False`, and numpy itself agrees the memory is shared.
```

Never open a subsection by restating the question, by "In this section we…", or by throat-clearing.

### 2.3 Thesis paragraph

Three to six sentences directly under the `#`. States what the page is about, why it exists, and
what the reader will be able to do. No links, no lists, no code. For page 3 it also states the
self-containment promise (§6).

### 2.4 On-this-page row

One line, `**On this page:** ` followed by `##`-section links joined by ` · `. Only `##` sections,
never `###`. Omitted when a page has fewer than four sections.

### 2.5 Verification banner

A blockquote directly under the on-this-page row, on every page:

```markdown
> Verified on CPython 3.12.12 · numpy 2.4.2 · pythonnet 3.0.5 · net8.0/net10.0.
> Every claim below is reproduced by a test in `NumSharp.Interop.UnitTests`.
```

The version list names the stack the claims were actually measured on. It is prose — nothing fails
if it drifts — but it is updated whenever the verification box changes. Page 4 extends the line
with `Numpy.Bare 3.11.1.33`.

### 2.6 Depth

Claim blocks are the skeleton, not the whole body. Around them the page carries the explanatory
depth of a conventional documentation site: name the API members involved, take a construction
apart piece by piece, state what each moving part is for and why the design refuses what it
refuses. The added detail lives in prose between claim and code — never as comment padding inside
the code — and it obeys §10: mechanisms named, quantities stated, no filler.

---

## 3. The claim block

The paragraph-level unit of every page. A claim block is:

1. **The claim**, as a bold lead sentence (or the bold answer to the `###` question).
2. **The code**, a runnable block (§7).
3. **The observed output**, in a ```` ```text ```` block — real bytes from a real run, never invented.
4. **The gate line**, naming the test that keeps the claim honest.

````markdown
**No — and the gap is two orders of magnitude.** `np.frombuffer` reports `OWNDATA = False`, and
numpy itself agrees the memory is shared.

```csharp
using PyObject mv = nd.ToMemoryView();
scope.Set("mv", mv);
scope.Exec("a = np.frombuffer(mv, '<f8').reshape(2, 3)");
```

```text
a.flags['OWNDATA']       False
np.shares_memory(a, x)   True
a[0,0] = -5.0    ->      nd == [[-5, 1, 2], [3, 4, 5]]
```

<sub>See here [`Frombuffer_SharesTheBuffer`][gate]</sub>
````

Parts 2–4 are omitted where they'd be noise (a one-line definitional claim needs no proof block),
but **any claim about behaviour, cost or compatibility carries at least a gate line.**

### 3.1 The gate line

Format, verbatim:

```markdown
<sub>See here [`TestMethodName`][gate]</sub>
```

- Prefix is `See here`.
- Link text is the **test method name only**, in backticks — not the class, not a sentence.
- Links are **reference-style**: `[gate]` is defined once, at the bottom of the page — one
  definition per target test file (`[gate]`; `[gate-<suffix>]` when a page cites more than one
  file). Renders identically to an inline link and keeps ~30 repetitions of a 120-character URL
  out of the markdown source.
- The link target is the **file** on `master`, never a line anchor (`#L123` rots on every edit):
  `[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/<File>.cs`
- Multiple gates for one claim: comma-separated links inside one `<sub>`.
- A claim proven by a self-skipping test (§8.3) gets a trailing `†`, explained once per page in the
  claims ledger.

---

## 4. Claims ledger

The last table on every page, before *See also*. One row per behavioural claim the page makes, in
page order.

```markdown
## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | `np.frombuffer` over a NumSharp memoryview shares memory | `OWNDATA=False`, `shares_memory=True` | [`Frombuffer_SharesTheBuffer`][gate] |
| 2 | PyTorch reads the same address | `t.data_ptr() == a.ctypes.data` | [`Torch_SharesTheDataPointer`][gate] † |

† Self-skipping: reports Inconclusive when the Python package is absent.
```

The **Evidence** column holds the observable — a flag value, a pointer comparison, an exception
message, a measured time — not a restatement of the claim. If a row's evidence cannot be written as
an observable, the claim is not a claim and comes out of the page.

---

## 5. Reference furniture

### 5.1 Tables

Dense, aligned, no filler columns. Header row in sentence case. Numeric columns right-aligned
(`|---:|`). Status cells use `✅` / `❌` / `⚠` only — never "Yes"/"No"/"Partial" prose.

Every table that enumerates a closed set (dtypes, layouts, modes, versions) must be **exhaustive
for that set** or say what it omits in the caption line beneath it.

### 5.2 Troubleshooting

A two-column `| Symptom | Cause & fix |` matrix, on pages 2, 3 and 4. Symptoms are quoted verbatim
from the actual exception or Python error text, in backticks. Fixes are imperative and single-step.
Rows are ordered by how often a reader hits them, not alphabetically.

### 5.3 See also

Bulleted, two to four entries, each `- [Page title](file.md) — what it gives you that this page doesn't`.

---

## 6. Cross-linking

| Page | May assume |
|---|---|
| 1 `index.md` | nothing |
| 2 `pythonnet-numpy.md` | nothing |
| 3 `np-frombuffer.md` | **nothing** — see below |
| 4 `numpy-net.md` | page 2 (links forward for depth, but restates the GIL rule and the wrap/unwrap idiom inline) |

**Page 3 stands fully alone.** A reader arriving from a search engine must be able to use it without
opening page 2. It therefore restates, in its own words and at its own depth: the contiguity gate,
writeability, the lifetime model, the resize lock, and the GIL rule. Links to page 2 are always
phrased as *go deeper*, never as *go first*. Duplication between pages 2 and 3 is expected and
correct; the two must not contradict, which the shared test gates enforce.

---

## 7. Code

### 7.1 Languages and fences

| Fence | Use |
|---|---|
| ```` ```csharp ```` | Everything a reader types into C#. |
| ```` ```python ```` | Code that genuinely lives on the Python side (a snippet a user would paste into a `.py`), not `scope.Exec` strings. |
| ```` ```text ```` | Observed output, exception messages, console transcripts. |
| ```` ```bash ```` | `dotnet add package`, `dotnet test`. |
| ```` ```xml ```` | `PackageReference` fragments. |
| ```` ```yaml ```` | toc / workflow fragments. |

### 7.2 Rules

- **Every C# block compiles and runs** as written, modulo the elisions in §7.3. It is copied from,
  or into, the corresponding `DocExamples` test.
- Namespaces are shown once per page, in the first block that needs them.
- `PyObject`s are created and disposed inside a `using (Py.GIL())` scope in every block that touches
  them — the pages must not teach a lifetime bug.
- Never `dynamic`. The package's whole design is direct `PyObject` calls; the samples reflect that.
- Comments in samples explain *why*, at most one per two lines, aligned to a common column when
  three or more appear consecutively.

### 7.3 Permitted elisions

- `var scope = Py.CreateScope(); scope.Exec("import numpy as np");` may be assumed after the first
  block that shows it.
- `// …` on its own line for omitted, irrelevant body.
- Nothing else. No pseudo-code, no `<your value here>` placeholders in runnable positions.

---

## 8. The proof gate

The law this repository already runs on (commit `98e6045a`): **test the claims, never the prose.**
Tests read no markdown, parse no files, and assert no wording. A page may be rewritten freely; it
can only break the build by being false.

### 8.1 One test class per page

| Page | Class | File |
|---|---|---|
| 1 | `DocExamples_InteropIndexPage` | `DocExamples.InteropIndexPage.cs` |
| 2 | `DocExamples_PythonnetNumpyPage` | `DocExamples.PythonnetNumpyPage.cs` |
| 3 | `DocExamples_NpFrombufferPage` | `DocExamples.NpFrombufferPage.cs` |
| 4 | `DocExamples_NumpyNetPage` | `DocExamples.NumpyNetPage.cs` |

Test method names are the claim, not the mechanism: `Frombuffer_SharesTheBuffer`,
`MemoryView_RefusesNonContiguous`, `Torch_SharesTheDataPointer`. Each carries an XML-doc summary
quoting the sentence of the page it gates.

### 8.2 What must be gated

Unconditionally, with numpy and the standard library only:

- every C# code block on the page,
- every row of every behavioural table (layout matrices, dtype maps, mode tables),
- every exception message quoted in Troubleshooting,
- every documented table that duplicates runtime knowledge (e.g. the Python→pythonnet matrix) —
  encoded as test data and compared against the runtime source of truth, so the two copies cannot
  disagree,
- every measured number that appears as a *ratio or ordering* claim (see §8.4).

### 8.3 Third-party libraries

`torch`, `PIL`, `pyarrow`, `pandas`, `polars`, `cv2` claims are gated by **self-skipping** tests:

```csharp
[TestMethod]
public void Torch_SharesTheDataPointer()
{
    SkipUnless("torch");     // Assert.Inconclusive($"python package '{module}' is not installed")
    …
}
```

`SkipUnless(string module)` is added to `InteropTestBase`: it attempts the import under the GIL and
calls `Assert.Inconclusive` on failure. These tests are real proof where the package exists and
silent where it doesn't; CI stays green on a bare image. Their claims carry `†` in the ledger.

### 8.4 Numbers

- **Absolute timings never appear as gated claims.** A test may not assert `0.0069 ms`.
- Timings appear in the pages as measured figures with their stack named (§2.5) and are refreshed
  when re-measured.
- The *ordering* claim behind a timing table — "the view path does not scale with n and the copy
  path does" — is gated by a test that measures both at two sizes and asserts the ratio direction,
  with margin.
- Counts of things the code owns (dtypes mapped, layouts exported, exporter varieties viewable) are
  gated as exact constants, since the code, not the machine, determines them.

---

## 9. Evidence base

Measured this session on CPython 3.12.12 · numpy 2.4.2 · pythonnet 3.0.5 · Windows 11 · net10.0.
These are the facts the pages are written from; each becomes a claim block or a table row.

### 9.1 Conversion cost, 1,000,000 float64, best of 5

| Verb | Time | Scales with n |
|---|---:|---|
| `ToMemoryView` | 0.0069 ms | no |
| `ToNumpy` | 0.0088 ms | no |
| `ToNDArrayView` | 0.0273 ms | no |
| `ToNDArray` (copy) | 0.6755 ms | yes |
| `ToNumpyCopy` | 1.1266 ms | yes |

### 9.2 `ToMemoryView` vs `ToNumpy` by layout

| Layout | `ToMemoryView` | `ToNumpy` |
|---|---|---|
| C-contiguous `(4,6)` | ✅ 192 bytes | ✅ strides `(48,8)` |
| row slice `b[1:3]` | ✅ 96 bytes | ✅ `(48,8)` |
| column slice `b[:, ::2]` | ❌ `InvalidOperationException` | ✅ `(48,16)` |
| transpose `b.T` | ❌ | ✅ `(8,48)` |
| reversed `b[::-1]` | ❌ | ✅ `(-48,8)` |
| F-order | ❌ | ✅ `(8,32)` |
| broadcast `(3,)→(2,3)` | ❌ | ✅ `(0,8)`, `WRITEABLE=False` |
| 0-d scalar | ✅ 8 bytes | ✅ `shape=()` |
| empty `(0,3)` | ✅ 0 bytes | ✅ `shape=(0,3)` |

### 9.3 Exported numpy view, structure

`nd.ToNumpy()` → `ndarray` with `OWNDATA=False`, `WRITEABLE=True`, base chain
`ndarray → c_char_Array_48`; strided exports base on numpy's `DummyArray` (via `as_strided`).
`np.frombuffer(mv, …)` → `OWNDATA=False`, base `memoryview`.

### 9.4 Import route census (11 probes)

| Source | Result |
|---|---|
| `np.arange(4, dtype='f4')` | view `Single[4]`, writable |
| `bytearray(b'abcd')` | view `Byte[4]`, writable |
| `array.array('i', …)` | view `Int32[3]`, writable |
| `(ctypes.c_double*3)(…)` | view `Double[3]`, writable |
| `memoryview(bytearray(…))[::2]` | view `Byte[4]`, writable |
| `io.BytesIO(…).getbuffer()` | view `Byte[8]`, writable |
| `np.arange(6).reshape(2,3).T` | view `Int64[3,2]`, writable |
| `np.broadcast_to(…, (2,3))` | view `Int64[2,3]`, **not** writable |
| `np.array([1+2j], dtype='c8')` | **copy**, widened to `Complex` |
| `np.array(['a','b'], dtype='U1')` | **copy**, narrowed to `Char` |
| `np.arange(4, dtype='>i4')` | **rejected** both paths — big-endian |

`np.frombuffer(bytes(...), …)` imports as a **non-writeable** view under `allowReadonly: true`, and
throws with copy guidance without it.

### 9.5 Third-party reach, all zero-copy

| Library | Version | Call | Observed |
|---|---|---|---|
| PyTorch | 2.12.1+cu126 | `torch.frombuffer(mv, dtype=torch.float32)` | `data_ptr` identical to `np.frombuffer`; `t[0]=42` lands in the `NDArray` |
| Pillow | 11.3.0 | `Image.frombuffer('RGB',(6,4),mv,'raw','RGB',0,1)` | `getpixel((3,2)) == (10,20,30)` |
| PyArrow | 23.0.1 | `pa.py_buffer(mv)` → `Array.from_buffers` | `to_numpy(zero_copy_only=True).ctypes.data == buf.address` |
| pandas | 2.3.3 | `pd.DataFrame(x, copy=False)` | `np.shares_memory(df.to_numpy(), x)` |
| polars | 1.38.1 | `pl.Series('v', x).to_numpy(allow_copy=False)` | shares |
| OpenCV | 4.13.0 | `cv2.circle(im, …)` | draws into NumSharp memory |
| stdlib | — | `struct.unpack_from`, `hashlib.sha256(mv)`, `zlib.compress(mv)`, `mv.cast('h')` | all direct; `mv.cast('h')[0]=77` writes through |

A PIL `Image` itself is **not** importable — its `__array_interface__['data']` is `bytes`, not a
`(pointer, readonly)` tuple, and it exports no PEP 3118 buffer. `np.asarray(im)` first; the result
imports as a read-only view.

### 9.6 The lock

While a NumSharp view leases a `bytearray`, `ba.append(1)` raises
`BufferError: Existing exports of data: object cannot be re-sized`; after `Dispose()` it succeeds.

### 9.7 Public dtype maps

`ToNumpyDtypeStr` / `ToBufferFormat`, complete:

| dtype | numpy | PEP 3118 |
|---|---|---|
| Boolean · Byte · SByte | `\|b1` · `\|u1` · `\|i1` | `?` · `B` · `b` |
| Int16 · UInt16 | `<i2` · `<u2` | `h` · `H` |
| Int32 · UInt32 | `<i4` · `<u4` | `i` · `I` |
| Int64 · UInt64 | `<i8` · `<u8` | `q` · `Q` |
| Char | `<u2` | `H` |
| Half · Single · Double | `<f2` · `<f4` · `<f8` | `e` · `f` · `d` |
| Complex | `<c16` | `Zd` |
| Decimal | *throws* | *throws* |

---

## 10. Voice

- **Second person, present tense, active.** "You hand the array to Python"; not "the array is handed".
- **Verdict first.** Bold the answer, then justify. Never build to a conclusion.
- **Name the mechanism.** "A `weakref.finalize` on the base buffer fires when the last Python view
  dies" beats "lifetimes are managed automatically".
- **Quantify or drop it.** "Fast" is not a claim; `0.0069 ms, flat in n` is.
- **Own the limits.** Every refusal in the package exists for a reason; the page gives the reason and
  the workaround in the same breath, so a limitation reads as a design decision, which it is.
- **No marketing.** No "seamless", "blazing", "simply", "just", "powerful", "easy".
- **No hedging.** No "should generally", "in most cases" — if it is conditional, name the condition.
- **No future tense about the library.** Document what ships, not what is planned.

### 10.1 Prohibited

| Never | Because |
|---|---|
| An invented output value | The `text` blocks are transcripts; a fabricated one is a lie the ledger cannot catch |
| A test name that does not exist | The gate line is a promise; a dead one is worse than none |
| An exact test count, suite size, or file line count | Nothing can keep it honest without policing prose (`98e6045a`) |
| A GitHub line anchor (`#L120`) | Rots on the next edit |
| Prose copied from the deleted pages | §1.1 |
| A claim with no observable | §4 |

---

## 11. Definition of done

A page is finished when:

1. Every `##` and `###` obeys §2.2.
2. Every behavioural claim has a gate line, and every gate line names a test that exists and passes.
3. The claims ledger has one row per behavioural claim, each with a real observable.
4. Every C# block appears, verbatim or trivially adapted, in the page's `DocExamples` class.
5. `dotnet test test/NumSharp.Interop.UnitTests` is green on `net8.0` and `net10.0`.
6. Page 3 has been read start to finish with pages 1, 2 and 4 unavailable, and nothing was missing.
7. The DocFX build produces no warnings for the page and `toc.yml` resolves.

---

## 12. Open

Nothing. The §2.1 interpretation — "as many sections as the material needs, no fixed tier count" —
was put to the user and confirmed; §2.1 stands as written. The reference-style gate links of §3.1
were ratified at the same time. The one deliberate deferral (page 3 naming its tests ahead of the
class existing) was closed when the full gate landed: all four `DocExamples_*` classes exist and
every named gate passes on net8.0 and net10.0.

Amended during the section-by-section revision pass: §2.2 — behavioural `###`s are noun/verdict
phrases, questions demoted to a rare exception; §2.6 (new) — conventional-documentation depth
around the claim blocks. Both ratified with the user before any page was touched.
