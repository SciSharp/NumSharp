# System.Numerics.Tensors — NDArray as a Tensor&lt;T&gt;/TensorSpan&lt;T&gt;, and back

`NumSharp.Interop.System.Numerics.Tensors` connects `NDArray` to the BCL's
[`System.Numerics.Tensors`](https://learn.microsoft.com/dotnet/api/system.numerics.tensors) types —
`Tensor<T>`, `TensorSpan<T>` and `ReadOnlyTensorSpan<T>` — without copying. An array of **any** layout
(contiguous, sliced, transposed, strided, even broadcast) becomes a `TensorSpan<T>` over the very same
unmanaged bytes; a `Tensor<T>` produced anywhere comes back as an `NDArray` — an owning copy, or a
zero-copy view over the tensor's pinned buffer. It is a **conversion library**, the sibling of the
[ONNX Runtime](onnxruntime.md) and [ML.NET](mlnet.md) bridges: no `TensorEngine` seam, no
`[ModuleInitializer]`, **no native dependency**. This page is its reference: setup, the verbs, what
crosses zero-copy and what cannot, who frees what, dtypes, frameworks.

**On this page:** [From zero](#from-zero) · [The verbs](#the-verbs) · [Zero-copy: the rules](#zero-copy-the-rules) ·
[Lifetime](#lifetime-who-frees-what) · [Dtypes](#dtypes) · [Frameworks](#frameworks) · [Limits](#limits) ·
[Not a compute backend](#not-a-compute-backend) · [Claims](#claims-ledger)

> Verified on `System.Numerics.Tensors` 10.0.0 · net8.0/net10.0 · Windows, Linux, macOS.
> Every claim below is reproduced by a test in `NumSharp.Tests.Interop.System.Numerics.Tensors`
> (54 tests, green on net8.0 AND net10.0; no Python, no native code).

---

## From zero

```bash
dotnet add package NumSharp.Interop.System.Numerics.Tensors
```

That is the whole dependency — `System.Numerics.Tensors` is a purely managed BCL package and comes with
it. Nothing else to install, no native runtime to pick.

```csharp
using System.Numerics.Tensors;
using NumSharp;
using NumSharp.Interop.Tensors;

var nd = np.arange(6).reshape(2, 3).astype(NPTypeCode.Single);   // [[0,1,2],[3,4,5]]

// Zero-copy: view the NDArray as a TensorSpan and mutate it in place.
using (var h = nd.AsTensorSpan<float>())
{
    TensorSpan<float> span = h.Span;                 // shares nd's buffer, no copy
    for (nint i = 0; i < span.Lengths[0]; i++)
        for (nint j = 0; j < span.Lengths[1]; j++)
            span[new[] { i, j }] += 10f;             // writes straight into nd
}
// nd is now [[10,11,12],[13,14,15]]

// A Tensor<T> produced elsewhere, read back as an NDArray with no copy.
var output = Tensor.Create(new[] { 0.1f, 0.7f, 0.2f }, new nint[] { 3 });
using NDArray probs = output.AsNDArray();
long best = np.argmax(probs);                        // 1 — NumSharp reductions on the shared buffer
```

> **Experimental.** `Tensor<T>`/`TensorSpan<T>` are `[Experimental("SYSLIB5001")]` in the BCL. This package
> re-marks its own surface `[Experimental("NUMSHARP_TENSORS")]` so you acknowledge that churn **once**, at
> the seam. Add `<NoWarn>NUMSHARP_TENSORS</NoWarn>` to your project, or `#pragma warning disable
> NUMSHARP_TENSORS` around the calls.

## The verbs

Everything is packaging over five operations on the static `NDArrayTensorsInterop` class (also extension
methods). The naming is the house convention: **`As…` shares memory, `To…` copies.**

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → Tensors | `nd.AsTensorSpan<T>()` → `TensorSpanHandle<T>` | **zero-copy** `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` over the array's buffer; **any non-negative-stride layout**. The handle owns the buffer pin — take the span from `.Span` / `.ReadOnlySpan` inside a `using` |
| NumSharp → Tensors | `nd.ToTensor<T>()` → `Tensor<T>` | independent, dense **copy**; any layout, read in logical (C) order; no lifetime coupling |
| Tensors → NumSharp | `tensor.ToNDArray()` | fresh **owning** C-contiguous **copy** — the safe default |
| Tensors → NumSharp | `tensor.AsNDArray()` | **zero-copy view** over the tensor's pinned backing store; dense → a C-contiguous view, strided → a strided view (no densifying copy) |
| Tensors → NumSharp | `span.ToNDArray()` | **copy** of a `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` (a `ref struct` span cannot be leased) |

Plus the dtype maps (`ToTensorElementClrType`, `TypeCodeOf<T>`) and the counters
`NDArrayTensorsInterop.LiveExports` / `LiveImports`, which count live export handles and import leases so a
leak is observable.

```csharp
using var h = nd.AsTensorSpan<float>();          // ZERO COPY: pointer + lengths + strides
TensorSpan<float> s = h.Span;                    // mutable (writes through)
ReadOnlyTensorSpan<float> ro = h.ReadOnlySpan;   // read-only, always available

Tensor<float> t = nd.ToTensor<float>();          // dense copy, independent
using NDArray copy = t.ToNDArray();              // owning copy back
using NDArray view = t.AsNDArray();              // zero-copy view over t's buffer
```

### Why a handle, not the span itself

`TensorSpan<T>` is a `ref struct`: it cannot be stored in a field, captured in a lambda, or returned to
outlive the pin that keeps its memory valid. So `AsTensorSpan` returns a small `IDisposable`
`TensorSpanHandle<T>` that owns the ARC reference on the NumSharp buffer and rebuilds the span on each
`.Span` / `.ReadOnlySpan` access (unmanaged memory never moves, so the pointer is stable). Wrap it in a
`using` and take the span from it — the same shape as keeping a pinned buffer alive across a native call.

## Zero-copy: the rules

A `TensorSpan<T>` carries **explicit lengths and strides**, so unlike an ONNX tensor (row-major, no
strides), a strided NumSharp view can be shared as-is:

- **Every non-negative-stride layout shares.** Contiguous, an offset slice (`nd["2:5"]`), a transpose
  (`nd.T`), a stepped slice (`nd["::2"]`), a Fortran-contiguous array, and a **broadcast** view (stride-0
  dimensions) all cross zero-copy. A broadcast (or otherwise read-only) view is exposed through
  `.ReadOnlySpan` only — `.Span` throws, because writing through overlapping stride-0 lanes would corrupt
  data, exactly as NumSharp forbids writing a broadcast view.
- **A negative-stride view is the one refusal.** `System.Numerics.Tensors` forbids negative strides (its
  `TensorSpan` ctor throws *"Strides cannot be less than 0"*), so a reversed slice `nd["::-1"]` is refused
  with the fix in the message: `np.ascontiguousarray(nd)` / `nd.copy()`, or `ToTensor` (which reads any
  layout in logical order and copies).
- **A length-≤1 axis is normalized to stride 0.** NumSharp assigns a unit axis a nonzero element stride;
  `System.Numerics.Tensors` requires such an axis to have stride 0 (a nonzero stride there reads as
  over-claiming the buffer). The bridge normalizes it — safe, because a single-element axis is never
  stepped, so the addressing is identical.
- **A 0-d scalar crosses as a rank-1 `[1]` span**, and an **empty array as `TensorSpan<T>.Empty`** — the
  BCL's pointer ctor cannot express rank 0 and rejects a zero-length backing, so these two shapes take the
  documented fallback.
- **No 2 GB limit on `AsTensorSpan`.** The span fronts a native pointer with an `nint` length, unlike a
  `Memory<T>`-backed tensor; arrays over 2 GB share fine. (`ToTensor` / `ToNDArray` copy into a managed
  `T[]`, which is `int`-indexed — see [Limits](#limits).)

Under the hood the export is `new TensorSpan<T>(pointer, dataLength, lengths, strides)` on
`slice.Address + Shape.Offset × itemsize` with the element strides from `Shape.Strides`; the import pins
the tensor's backing array (`Tensor<T>.GetPinnedHandle()`) and wraps the pointer as a NumSharp memory
block with a release hook — the same "wrap foreign memory" primitive the whole [interop family](index.md)
is built on.

## Lifetime: who frees what

- **Exports.** `AsTensorSpan` takes an atomic reference on the NumSharp buffer, so the memory outlives the
  source array if it is disposed or collected while a span is in use. Disposing the `TensorSpanHandle<T>`
  drops the reference; a forgotten handle is released by a finalizer safety net. `LiveExports` counts open
  handles.
- **Imports.** `AsNDArray` leases the tensor's pinned backing through NumSharp's memory-block reference
  count: the lease fires (unpinning the array) when the *last* NumSharp view over the memory — slices
  derived from it included — is disposed or collected. The view roots the `Tensor<T>` so its backing cannot
  move underneath it, and does not own its data (`owndata == false`), so a size-changing `resize` refuses
  instead of detaching. `LiveImports` counts open leases.
- **No GIL, no interpreter, no native handles.** The verbs are plain managed calls; the one rule is the
  handle rule — **keep the `TensorSpanHandle<T>` alive while you use its span, and dispose it.**

## Dtypes

`System.Numerics.Tensors` containers are **unconstrained generics** over an unmanaged `T` — there is no
dtype enum and no fixed element-type table. So **all 15 NumSharp dtypes cross zero-copy as their own CLR
type**; nothing is refused and nothing is converted:

| NumSharp | Crosses as | Note |
|---|---|---|
| Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Single, Double | `bool`, `byte`, `sbyte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `float`, `double` | zero-copy |
| Half | `System.Half` | **directly** — no `Float16` wrapper (unlike ONNX); NaN payloads and subnormals cross untouched |
| Char | `char` | zero-copy, UTF-16 code units; **directional** — a `ushort` element type reads back as `UInt16`, never `Char` |
| Decimal | `System.Decimal` | zero-copy (a 16-byte unmanaged struct) — no conversion, unlike the ONNX/ML.NET bridges |
| Complex | `System.Numerics.Complex` | zero-copy (a 16-byte unmanaged struct) — no refusal, unlike the ONNX/ML.NET bridges |

`T` must be the array's own element type: a mismatch (`AsTensorSpan<int>()` on a `float` array) is refused
up front rather than silently reinterpreting the bytes.

## Frameworks

The package builds an assembly for **net8.0** and **net10.0**. **net9.0** and **net11.0** consumers are
served by the net8.0 / net10.0 assets through NuGet nearest-TFM selection and .NET roll-forward — that is
how 9 and 11 are supported without producing two more byte-identical assemblies. An opt-in
`-p:BuildAllTfms=true` additionally emits the net9.0/net11.0 assets (once those SDKs are installed).

The dependency is `System.Numerics.Tensors [10.0.0, 11.0.0)`. It is pinned to the 10.x line because
`Tensor<T>`/`TensorSpan<T>` are `[Experimental]` and their surface shifted across the 9.x previews; the
ctors this package binds are verified against 10.0.0. Any 10.x rolls forward; the upper bound keeps a
future 11.x (free to break the experimental API) from resolving automatically.

## Limits

- **Negative-stride views** cannot be shared zero-copy (the BCL forbids negative strides) — `ToTensor` /
  `ascontiguousarray` them.
- **`ToTensor` / `ToNDArray` copy into a managed array**, which is `int`-indexed: over `int.MaxValue`
  elements they throw. Share zero-copy with `AsTensorSpan` (a native pointer, no such limit) instead.
- **A `ref struct` span cannot be leased**: importing a `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` always
  copies (`span.ToNDArray()`). Only a `Tensor<T>` (a heap object whose backing can be pinned) supports the
  zero-copy `AsNDArray` view.
- **A 0-d scalar crosses as rank-1 `[1]`** and an **empty array as `TensorSpan<T>.Empty`** — the BCL's
  rank-0 / zero-length handling forces these two documented shapes.

## Not a compute backend

This bridge moves *data*, not *computation*. Do **not** route NumSharp's operations through
`System.Numerics.Tensors`' `TensorPrimitives` / `Tensor.Add` surface: NumSharp's own kernels are bit-exact
with NumPy (and frequently faster than `TensorPrimitives`), and swapping them out would trade that parity
away. `System.Numerics.Tensors` does not compute NumSharp operations — which is exactly why there is no
`TensorEngine` seam here. The bridge is for handing a NumSharp buffer to code that already speaks
`Tensor<T>`/`TensorSpan<T>`, and reading such tensors back.

## Claims ledger

| # | Claim | Gate |
|---|---|---|
| 1 | All 15 dtypes map to their own CLR type and round-trip; the map is directional (UInt16 ≠ Char) | [`DtypeMapTests`][gate] |
| 2 | `AsTensorSpan` shares memory both ways (span↔ndarray write-through) and shares every dtype's bytes 1:1 | [`ExportTests`][gate] |
| 3 | `ToTensor` is an independent dense copy that reads any layout (incl. negative stride) in logical order | [`ExportTests`][gate] |
| 4 | `ToNDArray` copies (dense + strided); `AsNDArray` is a zero-copy view, write-through, dense or strided | [`ImportTests`][gate] |
| 5 | Contiguous / offset / transposed / stepped / Fortran / broadcast / unit-axis / 3-D views all share zero-copy | [`LayoutTests`][gate] |
| 6 | A negative-stride view is refused with the materialize hint; materializing then exporting works | [`LayoutTests`][gate] |
| 7 | A broadcast view exposes `.ReadOnlySpan` only; `.Span` throws (read-only) | [`LayoutTests`][gate] |
| 8 | The export handle tracks `LiveExports`, dispose is idempotent, the source may die while a span lives | [`LifetimeTests`][gate] |
| 9 | The import lease releases only when the last view (derived slices included) dies; a forgotten handle/view is released by GC | [`LifetimeTests`][gate] |
| 10 | Special values (NaN sign/payload/signaling, ±inf, ±0, subnormals, every dtype extreme) survive every crossing bit-exact | [`SpecialValueFidelityTests`][gate] |
| 11 | NDArray → Tensor → NDArray and NDArray → TensorSpan → NDArray preserve values across all dtypes and layouts | [`RoundTripTests`][gate] |
| 12 | The examples on this page run as written | [`DocExampleTests`][gate] |

## See also

- [Interoperability overview](index.md) — the contract every bridge builds on
- [ONNX Runtime](onnxruntime.md) — the sibling bridge whose handle / lease lifetime model this one mirrors
- [ML.NET](mlnet.md) — the other conversion bridge (an `IDataView` over an `NDArray`)
- Package README: `src/NumSharp.Interop.System.Numerics.Tensors/README.md`

[gate]: https://github.com/SciSharp/NumSharp/tree/master/test/NumSharp.Tests.Interop.System.Numerics.Tensors
