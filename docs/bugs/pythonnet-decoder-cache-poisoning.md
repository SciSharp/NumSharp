# pythonnet caches decoder-lookup misses — a decode before `RegisterCodec()` poisons that type pair for the whole engine session

**Status:** upstream pythonnet behaviour, permanent per engine session, cannot be healed from
NumSharp code. Documented and gated; the rule for users is one line: **register the codec before
the process's first `As<NDArray>()`.**

**Found:** 2026-07-22, while writing the executable gates for the interop documentation set.
**Observed on:** pythonnet 3.0.5 · CPython 3.12.12 · numpy 2.4.2 · net8.0 & net10.0 · Windows 11.

---

## Overview

pythonnet's conversion pipeline resolves which registered decoder serves a conversion once per
**(Python type, CLR target type)** pair — and it memoizes the result *including the miss*. If
`pyObject.As<NDArray>()` runs before `NDArrayPythonInterop.RegisterCodec()`, the lookup finds no
decoder, the failure is cached, and that exact pair keeps failing for the rest of the engine
session — **even after registration succeeds**. Pairs first touched *after* registration work
normally, which makes the symptom deeply confusing: `RegisterCodec()` returns `true`, auto-encode
works, `As<NDArray>()` on a `bytes` object works, and `As<NDArray>()` on a numpy array throws.

This is not NumSharp-specific: the cache belongs to pythonnet's `PyObjectConversions` layer, so
the same trap applies to any `IPyObjectDecoder` registered late for any target type.

## Reproduction

Minimal, self-contained (`dotnet run` file-based app against
`src/NumSharp.Interop.pythonnet/NumSharp.Interop.pythonnet.csproj`):

```csharp
Runtime.PythonDLL = @"C:\Python312\python312.dll";
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();

using (Py.GIL())
{
    using var scope = Py.CreateScope();
    scope.Exec("import numpy as np");

    // 1. decode BEFORE registration — fails, and the miss is cached for (ndarray -> NDArray)
    using (PyObject a = scope.Eval("np.arange(3, dtype='f8')"))
        try { a.As<NDArray>(); } catch (InvalidCastException) { /* expected */ }

    // 2. register — returns true, the codec IS in the pipeline now
    NDArrayPythonInterop.RegisterCodec();

    // 3. the SAME pair still fails, forever this session
    using (PyObject a = scope.Eval("np.arange(3, dtype='f8')"))
        a.As<NDArray>();   // InvalidCastException — the poisoned pair

    // 4. a pair never tried before registration works fine
    using (PyObject b = scope.Eval("b'abcd'"))
        using (NDArray nd = b.As<NDArray>()) { }   // OK: Byte[4]
}
```

## Expected

Registering a decoder makes subsequent conversions to its target type succeed, regardless of
whether an earlier conversion was attempted and failed.

## Actual

Observed transcript (the controlled experiment, run twice — identical both times):

```text
1. BEFORE RegisterCodec: As<NDArray>(ndarray)   -> [InvalidCastException] 'numpy.ndarray' value cannot be converted to NumSharp.NDArray
2. RegisterCodec()                              -> True
3. AFTER RegisterCodec:  As<NDArray>(ndarray)   -> [InvalidCastException] 'numpy.ndarray' value cannot be converted to NumSharp.NDArray
4. AFTER RegisterCodec:  As<NDArray>(bytes)     -> OK Byte    (pair never tried pre-registration)
5. AFTER RegisterCodec:  As<NDArray>(bytearray) -> OK Byte
```

The exception shape: outer `System.InvalidCastException: cannot convert object to target type`,
inner `Python.Runtime.PythonException: 'numpy.ndarray' value cannot be converted to
NumSharp.NDArray`.

How it was actually found: an experiment script probed the no-codec behaviour of `As<NDArray>()`
*first* (to capture the failure text for a troubleshooting table), then registered the codec and
continued — and every later `As<NDArray>()` on numpy arrays crashed while `scope.Set(name, nd)`
auto-encoding worked perfectly in the same session. The asymmetry (encode fine, one decode pair
broken) is the fingerprint of this trap.

## Root cause

pythonnet's `PyObjectConversions` (the codec pipeline behind `PyObject.As<T>()`) resolves the
decoder for a conversion through a per-session cache keyed by the (Python type, target type)
pair. The resolution runs once, on first use of the pair, and the cached value is whatever the
lookup produced *at that moment* — including "no decoder". Registration appends to the decoder
list but does not invalidate the cache, so a pair whose first lookup predates registration keeps
its cached miss. The cache is cleared only by `PyObjectConversions.Reset()`, which pythonnet runs
during `PythonEngine.Shutdown()` — which is why the poisoning is exactly session-scoped.

Granularity, proven by observation: the key really is the *pair*, at Python-type precision. The
gate exploits this — ctypes array types carry their element type **and length** in the type name
(`c_short_Array_7` vs `c_short_Array_8` are different Python types), so poisoning one leaves the
other decodable.

## Workaround

- **Register at startup.** Call `NDArrayPythonInterop.RegisterCodec()` immediately after
  `PythonEngine.Initialize()`, before anything can decode. This is the documented rule
  (`docs/website-src/docs/interop/pythonnet-numpy.md`, "When must I register it?").
- Already poisoned? Nothing in-process heals it short of a full engine restart
  (`Shutdown()` → `Initialize()` — impractical in most apps because CPython + numpy do not
  re-initialize reliably), so treat it as: fix the registration order and restart the process.
- The **explicit verbs never involve the codec** and are immune: `ToNDArray` / `AsNDArray` /
  `ToNDArrayView` / `ToNumpy` / `ToNumpyCopy` / `ToMemoryView` all work regardless.
- Re-registration after a legitimate engine restart is handled: pythonnet clears all codecs
  during `Shutdown`, and `RegisterCodec()` knows to re-register in the new session.

## How it is gated

`test/NumSharp.Tests.Interop/DocExamples.PythonnetNumpyPage.cs` →
`Codec_RegisterBeforeFirstConversion_OrThePairIsPoisoned`. The test is dual-branch because
registration is process-global and test order is not guaranteed:

- If its probe decode (`c_short_Array_7`) **fails**, this test ran before any registration: it
  registers, proves the poisoned pair still fails, and proves the fresh pair
  (`c_short_Array_8`) succeeds — the full trap, reproduced in-suite.
- If the probe **succeeds**, another test registered first: it asserts `RegisterCodec()` returns
  `false` (idempotence) and that fresh pairs decode.

Both branches assert real behaviour; neither is vacuous; and the probe types' per-length
`tp_name`s guarantee the poisoning can never leak into any other test's pairs.

Documented for users in `pythonnet-numpy.md` (its own subsection + a Troubleshooting row keyed on
the verbatim inner-exception text) and cross-referenced from `numpy-net.md`'s Troubleshooting
(Numpy.NET's `NDarray.self` is a `numpy.ndarray`, so it decodes through the same pair).

## Upstream

A pythonnet fix would be for encoder/decoder registration to invalidate (or version-stamp) the
conversion-resolution cache. Worth filing against pythonnet if late registration is ever meant to
be supported; until then, "register before first use" is the contract. Verified on 3.0.5; the
pipeline design is shared across the 3.x line, but other versions were not explicitly probed.

## Related

- `docs/website-src/docs/interop/pythonnet-numpy.md` — the user-facing rule and transcript
- `docs/bugs/numpy-resize-message-line-wrap.md` — the second discovery from the same gate-writing
  session
- `src/NumSharp.Interop.pythonnet/NumpyCodec.cs` — the codec whose registration is order-sensitive
- `src/NumSharp.Interop.pythonnet/PythonRuntimeInterop.cs` — the per-session `CodecRegistered`
  flag and the shutdown handling that makes re-registration work
