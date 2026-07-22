# Numpy.NET — drive real numpy over NumSharp buffers

NumSharp and SciSharp's Numpy.NET solve the same problem from opposite ends: NumSharp
*reimplements* numpy in pure .NET; Numpy.NET *remote-controls* a real CPython numpy through
pythonnet. They meet in one process without conflict, because both speak pythonnet's `PyObject` —
so a NumSharp array can be driven by Numpy.NET's C# API with zero copies, and a Numpy.NET array can
feed NumSharp kernels the same way. This page shows the two handoff idioms, the one GIL rule
Numpy.NET adds, and how to run both libraries side by side while migrating in either direction.

**On this page:** [One process, one engine](#one-process-one-engine) ·
[The handoff](#the-handoff-wrap-and-unwrap) · [The GIL rule](#the-gil-rule-numpynet-adds) ·
[Slices, dtypes, compute](#slices-dtypes-and-compute-cross-the-boundary) ·
[Lifetime](#lifetime-across-three-facades) · [Coexistence & migration](#coexistence-and-migration) ·
[Troubleshooting](#troubleshooting) · [Claims](#claims-ledger)

> Verified on CPython 3.12.12 · numpy 2.4.2 · pythonnet 3.0.5 · Numpy.Bare 3.11.1.33 ·
> net8.0/net10.0. Every claim below is reproduced by a test in `NumSharp.Interop.UnitTests`.

---

## One process, one engine

### Which Numpy.NET package goes next to NumSharp?

**`Numpy.Bare` — it binds to whatever engine is already initialized.** Numpy.NET ships two
flavors: `Numpy` (bundles Python.Included, which downloads and manages its own embedded Python)
and `Numpy.Bare` (uses the Python that is already there). Since `NumSharp.Interop.pythonnet` apps
own their engine — you set `Runtime.PythonDLL` and call `PythonEngine.Initialize()` — Bare is the
one that composes:

```xml
<PackageReference Include="NumSharp.Interop.pythonnet" Version="0.60.0" />
<PackageReference Include="Numpy.Bare" Version="3.11.1.33" />
```

Never reference `Numpy` and `Numpy.Bare` together: the two packages ship the same types in
different assemblies, which is compile error CS0433. And the pythonnet versions unify by
themselves — `Numpy.Bare` declares pythonnet 3.0.1, NumSharp's floor is `[3.0.5, 4.0.0)`, and
NuGet resolves the higher floor for both.

Numpy.NET's lazy initialization then sees the running engine and simply imports numpy into it — no
second interpreter, no installer:

```csharp
using Numpy;                    // NDarray, Dtype, Numpy.np
using np2 = Numpy.np;           // NumSharp's np wins bare-name lookup; alias theirs

using (Py.GIL())
{
    using NDarray their = np2.arange(6);
    Console.WriteLine($"{their}  dtype={their.dtype}");
}
```

```text
np2.arange(6) = [0 1 2 3 4 5]   dtype=int64
our scopes still work: 1 + 1 = 2
LiveExports = 0   (Numpy.NET's own arrays involve no NumSharp pins)
```

<sub>See here [`SharedEngine_NumpyNetBootsOnOurs`][gate], [`Packaging_BareBindsAndPythonnetUnifies`][gate]</sub>

---

## The handoff: wrap and unwrap

Two idioms cover every crossing. Both are zero-copy; both go through the `PyObject` the two
libraries share.

### How do I drive a NumSharp array with Numpy.NET's API?

**Wrap the export: `new NDarray(nd.ToNumpy())`.** Numpy.NET's whole C# surface then operates over
NumSharp's buffer — reads, reductions, in-place fills:

```csharp
var ours = np.arange(6).astype(NPTypeCode.Double);

using (Py.GIL())
{
    using var wrapped = new NDarray(ours.ToNumpy());

    double sum = wrapped.sum().item<double>();   // Numpy.NET computes over NumSharp memory
    wrapped.fill(7.0);                           // ...and writes into it
}

Console.WriteLine(ours);
```

```text
wrapped.sum().item<double>() = 15
after wrapped.fill(7):  ours = [7. 7. 7. 7. 7. 7.]
```

<sub>See here [`Wrap_TheirApiDrivesOurBuffer`][gate]</sub>

### How do I run NumSharp kernels over a Numpy.NET array?

**Unwrap its `PyObject`: `their.self.AsNDArray()` for a view, `.ToNDArray()` for a copy.**
`NDarray.self` is the underlying numpy `PyObject`, and it imports like any other numpy array —
leased, shared, writable:

```csharp
NDarray their;
using (Py.GIL())
    their = np2.arange(8).astype(np2.float64);

NDArray view;
using (Py.GIL())
    view = their.self.AsNDArray();      // zero-copy lease over their buffer

view[1] = (NDArray)(-7.5);              // NumSharp writes...
using (Py.GIL())
    their.fill(3.25);                   // ...Numpy.NET writes...

var total = np.sum(view);               // ...NumSharp kernels see it all
```

```text
after view[1] = -7.5:      their.item<double>(1) = -7.5
after their.fill(3.25):    view = [3.25 3.25 3.25 3.25 3.25 3.25 3.25 3.25]
np.sum(view) = 26.0
```

<sub>See here [`Unwrap_OurKernelsRunOverTheirArray`][gate]</sub>

---

## The GIL rule Numpy.NET adds

**Numpy.NET contains no GIL management of its own — wrap every Numpy.NET call, including
`NDarray.Dispose()`, in `Py.GIL()`.** The library assumes the GIL stays held after
`PythonEngine.Initialize()`, which is true only until you call `BeginAllowThreads()`. An
application that combines it with NumSharp interop *does* call `BeginAllowThreads` (that is what
lets arbitrary threads convert), so the assumption breaks and the caller must supply the GIL:

```csharp
using (Py.GIL())                 // every np2.* call, every NDarray member, every Dispose
{
    using var a = np2.arange(4);
    using var b = a.reshape(2, 2);
}
```

NumSharp's own verbs are unaffected — they acquire the GIL themselves (see
[the pythonnet page](pythonnet-numpy.md)). The rule is Numpy.NET's alone, and forgetting it is an
access violation, not an exception you can catch.

<sub>See here [`NumpyNet_BootsOnTheSharedEngine`][gate-suite] — the whole suite runs under this rule</sub>

---

## Slices, dtypes and compute cross the boundary

### Do slices survive the crossing?

**Yes, in both directions, as views.** A Numpy.NET slice is a numpy view, and NumSharp leases it —
writes through the NumSharp side land in their *base* array. A strided NumSharp view exports with
its layout intact, and Numpy.NET reads it in logical order:

```text
their[2:8] -> NumSharp view: size=6, first element 2
  view[3] = -1.0    ->  their base[5] == -1.0
NumSharp b["1:3, ::2"] -> their shape (2, 3), sum() = 66
```

<sub>See here [`Slices_CrossInBothDirections`][gate]</sub>

### Do dtypes arrive exact?

**Yes — the numpy dtype is the NumSharp dtype's exact mapping, both ways.** Outbound, a wrapped
array reports the numpy name (`Double` → `float64`, `Single` → `float32`, `Int32` → `int32`,
`Int64` → `int64`, `Byte` → `uint8`, `Boolean` → `bool`); inbound, `their.astype(np2.int32)`
arrives as `NPTypeCode.Int32` with values intact. `Decimal` is the one dtype with no numpy
representation — it cannot be wrapped (see [the dtype table](pythonnet-numpy.md#dtypes)).

<sub>See here [`Dtypes_ArriveExact_BothWays`][gate]</sub>

### Can I trust their compute over my buffers?

**Cross-checked: Numpy.NET's `matmul` over wrapped NumSharp inputs equals NumSharp's own.**

```csharp
var a = (np.arange(4).astype(NPTypeCode.Double) / 2.0).reshape(2, 2);
var b = (np.arange(4).astype(NPTypeCode.Double) + 1.0).reshape(2, 2);

using (Py.GIL())
{
    using var wa = new NDarray(a.ToNumpy());
    using var wb = new NDarray(b.ToNumpy());
    using var product = np2.matmul(wa, wb);      // real numpy computes...
    NDArray result = product.self.ToNDArray();   // ...NumSharp takes the result
}
```

```text
np2.matmul(wa, wb)  (real numpy):
[[1.5 2. ]
 [5.5 8. ]]
product.mean() = 4.25

np.matmul(a, b)  (NumSharp's own):
[[1.5 2. ]
 [5.5 8. ]]
```

This is the migration safety net: any call you have not ported yet can run through Numpy.NET
against the same buffers, and the two implementations check each other.

<sub>See here [`Compute_TheirMatmulEqualsOurs`][gate]</sub>

---

## Lifetime across three facades

One buffer can be visible as a NumSharp `NDArray`, a Numpy.NET `NDarray`, and a plain numpy object
in a Python scope — simultaneously. The lifetime rules are the interop's usual ones, and the
Numpy.NET wrapper participates like any other Python-side reference:

- **Their wrapper pins our buffer.** Wrap a NumSharp array, drop every NumSharp reference, run the
  GC — Numpy.NET still reads valid memory (`LiveExports` stays 1). Dispose the wrapper (under the
  GIL) and the pin drains to 0.
- **Our lease outlives their wrapper.** Import `their.self` as a view, then `their.Dispose()` —
  the lease keeps the numpy array alive; reads and writes through the NumSharp view stay valid
  (`LiveImports` stays 1 until the view is disposed).
- **The registered codec sees through the wrapper.** `their.self.As<NDArray>()` decodes like any
  numpy array — under the default `Auto` mode, as a zero-copy view.

```text
wrapper alive, source collected:   LiveExports = 1, wrapped.sum() still = 20.0
their wrapper disposed:            view[0] = 11.5 still lands; LiveImports = 1
their3.self.As<NDArray>() -> Double; decoded[0] = -100 -> their3.item(0) == -100
```

<sub>See here [`Lifetime_TheirWrapperPinsOurBuffer_AndDrainsOnDispose`][gate], [`Lifetime_OurLeaseOutlivesTheirWrapper`][gate], [`Codec_TheirArraysDecode_BecauseTheyAreNdarrays`][gate]</sub>

---

## Coexistence and migration

The two libraries divide cleanly by what executes:

| | NumSharp | Numpy.NET |
|---|---|---|
| Executes | .NET IL (SIMD kernels) | real CPython numpy |
| Needs Python at runtime | no — pure .NET | yes, always |
| Array type | `NDArray` | `NDarray` (a `PyObject` wrapper) |
| Call overhead | none | one Python round-trip per call |
| API breadth | the NumSharp-implemented surface | everything numpy ships |

Which makes the migration paths symmetrical:

- **Moving Python code to .NET:** port callsites to NumSharp one at a time; anything not yet
  ported runs through Numpy.NET *over the same buffers* via the wrap idiom — no copies, no
  divergent state, and the cross-check pattern above verifies each ported call against real numpy.
- **A Numpy.NET codebase adopting NumSharp:** unwrap with `their.self.AsNDArray()` where you want
  Python-free kernels, hot paths without round-trip overhead, or deployment without an interpreter.

Both directions can stop halfway and stay there: the handoff idioms are cheap enough (a wrap is a
pointer exchange) that a permanently mixed codebase is a reasonable end state, not a failure to
finish.

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| Compile error CS0433 (type exists in both assemblies) | `Numpy` and `Numpy.Bare` are referenced together — they ship the same types. Keep exactly one; next to this interop, `Numpy.Bare` |
| `np` is ambiguous between NumSharp and Numpy | Both libraries expose an `np` class. Alias theirs: `using np2 = Numpy.np;` |
| Access violation inside a Numpy.NET call | The GIL rule: with `BeginAllowThreads` active, every Numpy.NET call — including `NDarray.Dispose()` — must run inside `Py.GIL()` |
| Numpy.NET starts downloading a Python | You referenced the `Numpy` flavor (Python.Included). Use `Numpy.Bare`, which binds to the engine you initialized |
| `'numpy.ndarray' value cannot be converted to NumSharp.NDArray` | `As<NDArray>()` on `their.self` needs the codec, registered before the process's first decode — see [the codec trap](pythonnet-numpy.md#when-must-i-register-it) |
| `cannot resize this array: it does not own its data` | The NumSharp side of the handoff is a view over numpy's memory. `np.require(view, null, "O")` for an owning copy |

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | Numpy.Bare binds to the already-initialized engine — no second interpreter | `np2.arange(6)` works; our scopes keep working; `LiveExports == 0` | [`SharedEngine_NumpyNetBootsOnOurs`][gate] |
| 2 | The pythonnet versions unify; the types come from `Numpy.Bare` | loaded pythonnet ≥ 3.0.5 with Numpy.Bare functional; assembly identity | [`Packaging_BareBindsAndPythonnetUnifies`][gate] |
| 3 | `new NDarray(nd.ToNumpy())` lets Numpy.NET compute over and write into NumSharp memory | `sum == 15`; `fill(7.0)` lands in the `NDArray` | [`Wrap_TheirApiDrivesOurBuffer`][gate] |
| 4 | `their.self.AsNDArray()` is a shared, writable lease | writes cross both ways; `np.sum(view)` correct | [`Unwrap_OurKernelsRunOverTheirArray`][gate] |
| 5 | Slices cross as views in both directions | write through our view of their slice hits their base; strided export sums correctly | [`Slices_CrossInBothDirections`][gate] |
| 6 | Dtypes arrive exact in both directions | numpy names outbound; `NPTypeCode` inbound | [`Dtypes_ArriveExact_BothWays`][gate] |
| 7 | Their `matmul` over wrapped inputs equals NumSharp's | element-wise comparison of both products | [`Compute_TheirMatmulEqualsOurs`][gate] |
| 8 | Their wrapper alone keeps NumSharp memory alive; disposing it drains the pin | `LiveExports` 1 → 0 around wrapper disposal | [`Lifetime_TheirWrapperPinsOurBuffer_AndDrainsOnDispose`][gate] |
| 9 | Our lease outlives their disposed wrapper | reads/writes valid after `their.Dispose()`; `LiveImports == 1` | [`Lifetime_OurLeaseOutlivesTheirWrapper`][gate] |
| 10 | The codec decodes Numpy.NET arrays (they are `numpy.ndarray`) | `As<NDArray>()` yields a shared view | [`Codec_TheirArraysDecode_BecauseTheyAreNdarrays`][gate] |

The deeper behavioural suite for this pairing — lifetime edge cases, codec interleaving, the
GIL discipline every test runs under — is [`NumpyNetInteropTests`][gate-suite].

---

## See also

- [Python & numpy (pythonnet)](pythonnet-numpy.md) — the bridge underneath: verbs, layouts,
  lifetime, the codec and its registration-order trap, dtypes, versions
- [Any library via np.frombuffer](np-frombuffer.md) — reaching Python libraries that want bytes,
  no numpy (or Numpy.NET) required
- [Interoperability](index.md) — the contract underneath every NumSharp bridge

[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/DocExamples.NumpyNetPage.cs
[gate-suite]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/NumpyNetInteropTests.cs
