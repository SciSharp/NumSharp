# NumSharp.Interop.pythonnet — adversarial probe report

**Branch:** `experimental` (from `journey3`, worktree `K:/source/NumSharp-experimental`)
**Date:** 2026-08-13 · pythonnet 3.0.5 · CPython 3.12.12 (Anaconda) · numpy 2.4.2 · net8.0 · Windows 11
**Goal:** use numpy through pythonnet via the `NumSharp.Interop.pythonnet` NuGet package, then
red-team the integration — find bugs, gaps, and any way to break the lifetime/memory design.

## What was exercised

1. **Packed the real `.nupkg`** (`dotnet pack`) and **consumed it as a package** (not a project
   reference) from a standalone consumer (`experimental/Probe`), resolving `NumSharp 0.60.0` +
   `pythonnet [3.0.5,4.0.0)` transitively from a local feed. A round-trip smoke test passed through
   the packaged assembly — the packaging contract is sound.
2. Built an **arg-dispatched adversarial harness** (`experimental/adversarial-harness`, one embedded
   engine per process so shutdown/GIL/poison scenarios each get a clean interpreter). Scenarios:
   `battery leak threads aiface aiface2 strided drainidle resize mmap dtypeedge poison shutdown gil
   nullcrash`.

To reproduce: `dotnet run -c Release -- <scenario>` from the harness dir (needs the embedded Python;
the harness points pythonnet at `C:/Users/ELI/.claude/python/python312.dll`).

---

## Findings

### 1. (FIXED) `ToNDArrayView` built views over an UNVALIDATED `__array_interface__` — memory-unsafe

The one genuine, net-new defect. `ViewViaArrayInterface` read `data[0]` (a raw address), `shape` and
`strides` from an arbitrary Python object and built a zero-copy view over them without the checks numpy
performs. Calibrated directly against numpy 2.4.2 — **numpy rejects each; the interop accepted each:**

| malformed `__array_interface__` | numpy | NumSharp (before) |
|---|---|---|
| `data:(0,False)` (NULL, non-empty) | `ValueError: data is NULL…` | view over **address 0** → faults on read |
| `shape:(-2,)` (negative dim) | `ValueError: negative dimensions…` | view with a **negative extent** |
| `strides` longer than `shape` | `ValueError: mismatch in length…` | **corrupt** view (`Strides.Length` ≠ ndim) over garbage ptr |
| `strides` shorter than `shape` | `ValueError: mismatch in length…` | raw `IndexOutOfRangeException` |

For a non-NULL garbage pointer this is a true access violation / memory corruption on read **or
write** — the memory-unsafe opposite of the package's contract (which is otherwise scrupulous:
read-only refusal, non-writeable broadcasts, resize guards, orphaned-export sweeps). The sibling PEP
3118 path (`ViewViaBufferStrides`) is not affected — its pointer/strides come from a real `memoryview`
and it already length-checks.

**Fixed** in this branch: three guards in `ViewViaArrayInterface` mirroring numpy's checks and
messages (non-negative dims, non-NULL data for non-empty, `len(strides)==len(shape)`), throwing the
package's house `NotSupportedException`. Honest inputs unaffected. Full write-up:
`docs/bugs/pythonnet-array-interface-validation.md`. Gate:
`test/NumSharp.Interop.UnitTests/ArrayInterfaceValidationTests.cs` (5 tests). The full interop suite
(239 tests) passes with the guards in place.

### 2. (CONFIRMED, upstream) codec decoder-cache poisoning

Reproduced the documented trap: a `pyObj.As<NDArray>()` **before** `RegisterCodec()` permanently
poisons that (python-type, `NDArray`) pair for the engine session — it keeps failing even after
registration succeeds, while pairs first touched *after* registration work. This is pythonnet's
`PyObjectConversions` cache behaviour, not fixable from NumSharp; the rule is "register the codec
before the first `As<NDArray>()`." Already documented at
`docs/bugs/pythonnet-decoder-cache-poisoning.md`; confirmed still real on this stack.

### 3. (documented, validated) `requireGIL:false` without holding the GIL hard-crashes

Confirmed accurate: `ToNumpy(nd, requireGIL:false)` **while holding** the GIL works; **without** it,
the process dies with `AccessViolationException` inside `PyTuple_New` — exactly as the XML docs warn.
Not a bug; the documentation matches reality.

---

## Break attempts that FAILED to break it (the lifetime/memory design held)

Reported honestly — extensive stress found **no leak, no use-after-free, no deadlock, no counter
drift**:

- **Leak sweep** — 120 iterations × ~7 MB arrays for each of: export-view, export-copy, import-view,
  import-copy, and a **finalizer-drop** import path (views dropped without `Dispose`, released only by
  GC). Net working-set growth across all 600 conversions: **~40 MB** (baseline noise), `LiveExports`
  and `LiveImports` both returned to **0** every time.
- **Deferred-drain reliability** — a lease disposed with **no** subsequent conversion/GC still drained
  (thread-pool) in **33 ms**. The deferred design does not depend on a follow-up call.
- **Orphaned-export sweep** — an export held only by Python, then `PythonEngine.Shutdown()`: the pin
  was swept **115 ms** after shutdown (pythonnet runs no atexit, so `weakref.finalize` never fires at
  shutdown — the interop's own sweep releases it). No buffer leaked past the engine.
- **resize refcheck** — an exported source refuses size-changing `resize(refcheck:true)`; an import
  view (`owndata=false`) refuses size-changing resize. Both match numpy's guards.
- **Concurrency** — 8 threads × conversions each (export view+copy, import view+mutate): 0 errors,
  counters settled. No GIL/gate lock-order inversion observed.
- **mmap GIL-bounce** — 50 import/dispose cycles over `mmap`-backed memory (the documented
  deadlock-prone `PyBuffer_Release` path): clean, no hang.
- **Post-shutdown conversion** — a clean `InvalidOperationException`, not a crash.

## Correctness that held (no divergence from numpy)

- **All 15 dtypes** round-trip (Char→uint16; Decimal→`NotSupportedException`).
- **Every layout** exports with numpy-**exact** shape + values + **strides**: contiguous, row/col
  slice, `[::2,::2]`, `[::-1]`, `[::-1,::-1]`, `.T`, negative-column, 0-d scalar, Fortran + Fortran
  slice.
- **Import views** (numpy strided/reversed/transpose/broadcast **and** non-numpy `array.array`
  memoryviews `[::2]`/`[::-1]`/`[3:]`/`[1::3]`) read **and** write-through correctly.
- Read-only refusal + non-writeable views; broadcast → read-only numpy.
- big-endian / complex64-view / UCS-4-view / structured → clean rejection; complex64 & UCS-4 copies
  widen/narrow correctly.
- dtype widths (`l/L/q/Q/i/f/d`), float16, int64/uint64 extremes (incl. > 2^63) — bit-exact.
- Codec auto-marshal (`np.matmul(a,b)` on two encoded NDArrays, decoded back) — correct.

## Packaging observations (non-blocking)

- The nupkg is well-formed: correct transitive deps (`NumSharp`, `pythonnet`), README, net8/net10.
- It ships **no native/runtime assets** and requires the consumer to set `Runtime.PythonDLL` and call
  `PythonEngine.Initialize()` — by design and documented, but a fresh consumer gets no help locating
  Python (there is no discovery helper in the shipped package; the test project has one).
- The consumer project under `experimental/Probe` depends on a **local feed** (`_localfeed/`, not
  committed), so it is a documented artifact rather than a CI-buildable project.

## Net assessment

The integration is high-quality: correctness is numpy-exact across dtypes/layouts, and the
lifetime/memory design survived every stress thrown at it. The single real defect was an
input-validation gap on the `__array_interface__` import path (now fixed and gated). The other two
items are a confirmed upstream pythonnet limitation and a documented, accurate safety warning.
