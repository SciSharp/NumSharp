# Interoperability — one buffer, every ecosystem's API

A NumSharp array is raw unmanaged memory plus four numbers that describe how to read it: a base
address, element strides, an offset and a dtype. That is the same convention numpy, Python's buffer
protocol, Arrow and every other strided-array system speak — so interop is not translation, it is
introduction: hand the description across the boundary, agree on who frees the memory, and both
sides work on the same bytes. This page states the contract that makes that safe and maps the
bridges built on it.

**On this page:** [The contract](#the-contract) · [The bridges](#the-bridges) ·
[Claims](#claims-ledger)

> Verified on CPython 3.12.12 · numpy 2.4.2 · pythonnet 3.0.5 · net8.0/net10.0.
> Every claim below is reproduced by a test in `NumSharp.Interop.UnitTests` — this page's own
> gates run without Python, because the contract is NumSharp's alone.

---

## The contract

Three NumSharp capabilities make a bridge possible. Everything the bridge pages document —
zero-copy views, leases, locks — reduces to these.

### How does a view cross without copying?

**Because a layout is four numbers, and NumSharp exposes all four.** Any strided window — a slice,
a transpose, a reversed axis — is the same base pointer with different strides and offset:

```csharp
var nd = np.arange(24).reshape(4, 6).astype(NPTypeCode.Double);
var window = nd["1:3, ::2"];
```

```text
window.Storage.Address   == nd.Storage.Address     (views share the base pointer)
window.Shape.Strides     == [6, 2]                 (element strides, not bytes)
window.Shape.Offset      == 6
window.typecode          == Double
```

A bridge multiplies the element strides by the item size, adds the offset to the pointer, and any
strided-array consumer can address the window exactly. No elements move.

<sub>See here [`Contract_RawLayoutAccess_ExposesAddressStridesOffsetAndDtype`][gate]</sub>

### How does foreign memory become an `NDArray`?

**Through one primitive: wrap a pointer with a release hook.** This is what every import path is
made of — Python buffers, mmaps, memory another runtime owns:

```csharp
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

byte* ptr = (byte*)NativeMemory.Alloc(6);            // any foreign allocation
bool released = false;
Action onLastReferenceReleased = () => released = true;

var nd = new NDArray(new UnmanagedStorage(
    new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(ptr, 6, onLastReferenceReleased)),
    new Shape(2, 3)));
```

NumSharp kernels run over the foreign memory in place, writes land in it, and the hook fires when
NumSharp is done with it:

```text
nd.GetByte(1, 2)  reads ptr[5]        np.sum(nd) runs over the foreign buffer
nd[0, 0] = 200        ->  ptr[0] == 200
nd.Dispose()          ->  released == true
```

<sub>See here [`Contract_ExternalMemoryWrapping_TheDocumentedPrimitive`][gate]</sub>

### When exactly does the release hook fire?

**On the last reference to the memory block — original or derived view, disposed or collected.**
The block is atomically reference-counted; a slice taken from a wrapped array holds the same block,
so disposing the original frees nothing while the slice lives. The refcount decides, not disposal
order — and the GC finalizer is the safety net when nothing was disposed at all:

```text
derived = nd["2:"]; nd.Dispose()   ->  hook NOT fired; derived still reads valid memory
derived.Dispose()                  ->  hook fired
(no Dispose at all, GC runs)       ->  hook fired by the finalizer safety net
```

The same references feed NumSharp's resize guard: while a second view (or an export to Python)
holds the block, `nd.resize(...)` refuses with NumPy's own wording — `cannot resize an array that
references or is referenced by another array in this way`.

<sub>See here [`Contract_ReleaseHook_FiresOnTheLastReference_IncludingDerivedViews`][gate], [`Contract_ReleaseHook_AlsoFiresByGarbageCollection`][gate], [`Contract_RefcheckGuard_SeesOtherReferencesToTheBlock`][gate]</sub>

### Who owns wrapped memory?

**Whoever you say — and the bare wrap says NumSharp does, which is usually wrong for a bridge.**
A bare wrap claims ownership: a growing `resize` succeeds by reallocating into fresh NumSharp
memory, silently detaching from the foreign pointer (and firing the release hook). A bridge that
must stay attached aliases the storage instead, which gives the array numpy's `owndata == False`
semantics — exactly what the pythonnet import path does:

```csharp
var attached = new NDArray(
    new UnmanagedStorage(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(p, 8, () => { })),
                         Shape.Vector(8))
        .Alias(new Shape(8)));
```

```text
bare wrap:  resize(16) succeeds — and the address changes; the hook fires
aliased:    resize(16) throws IncorrectShapeException:
            cannot resize this array: it does not own its data
```

<sub>See here [`Contract_BareWrapClaimsOwnership_AliasIsWhatKeepsItAttached`][gate]</sub>

---

## The bridges

Every bridge below implements the contract above; each page states its own claims and carries its
own gates.

| Bridge | Ships in | Zero-copy | Page |
|---|---|---|---|
| **numpy, in process** — `NDArray` ⇄ `numpy.ndarray`, every layout, both directions | `NumSharp.Interop.pythonnet` | ✅ views both ways | [Python & numpy (pythonnet)](pythonnet-numpy.md) |
| **Any Python library** — PyTorch, Pillow, Arrow, OpenCV, stdlib — via the buffer protocol | `NumSharp.Interop.pythonnet` | ✅ `memoryview` / PEP 3118 | [Any library via np.frombuffer](np-frombuffer.md) |
| **Numpy.NET coexistence** — drive real numpy's C# API over NumSharp buffers | + `Numpy.Bare` | ✅ `PyObject` handoff | [Numpy.NET](numpy-net.md) |
| **`.npy` / `.npz` files** — `np.save` / `np.load`, byte-for-byte identical to NumPy's own writer | `NumSharp` (core) | — files, not memory | [NumPy compliance](../compliance.md) |

Start with the page whose *consumer* matches yours: numpy code → the pythonnet page; a library
that wants bytes (or no numpy at all) → the frombuffer page; an existing Numpy.NET codebase → the
Numpy.NET page; data at rest → the file formats, whose writer is byte-exact against `np.save`
itself.

<sub>See here [`Bridges_ThePythonnetPackage_ShipsTheFourVerbs`][gate], [`NpyOracleTests`][gate-npy]</sub>

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | A strided window is the base pointer plus strides/offset/dtype — no copy | same `Storage.Address`; strides `[6, 2]`; offset `6` | [`Contract_RawLayoutAccess_ExposesAddressStridesOffsetAndDtype`][gate] |
| 2 | Foreign memory wraps into a working `NDArray` with a release hook | kernels + writes on the foreign buffer; hook fires on `Dispose` | [`Contract_ExternalMemoryWrapping_TheDocumentedPrimitive`][gate] |
| 3 | The hook fires on the *last* reference, derived views included | disposing the original leaves the hook unfired until the slice goes | [`Contract_ReleaseHook_FiresOnTheLastReference_IncludingDerivedViews`][gate] |
| 4 | The GC finalizer is a safety net when nothing was disposed | hook observed fired after collection | [`Contract_ReleaseHook_AlsoFiresByGarbageCollection`][gate] |
| 5 | The resize guard sees every reference to the block | verbatim NumPy-worded refusal while a second view lives | [`Contract_RefcheckGuard_SeesOtherReferencesToTheBlock`][gate] |
| 6 | Bare wraps own and may detach; aliased wraps refuse to detach | address change vs `does not own its data` refusal | [`Contract_BareWrapClaimsOwnership_AliasIsWhatKeepsItAttached`][gate] |
| 7 | The pythonnet package really ships the four verbs | assembly identity + method presence | [`Bridges_ThePythonnetPackage_ShipsTheFourVerbs`][gate] |
| 8 | `.npy`/`.npz` output is byte-identical to NumPy's own | 286-case oracle of real `np.save` output, replayed bit-exact | [`NpyOracleTests`][gate-npy] |

---

## See also

- [Python & numpy (pythonnet)](pythonnet-numpy.md) — the reference bridge: four verbs, layouts,
  lifetime, codec, GIL, dtypes, versions
- [Any library via np.frombuffer](np-frombuffer.md) — the buffer protocol route to libraries that
  never touch numpy
- [Numpy.NET](numpy-net.md) — running SciSharp's numpy binding over NumSharp memory
- [Buffering & Memory](../buffering.md) — how NumSharp's own storage, slices and reference
  counting work underneath all of this

[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/DocExamples.InteropIndexPage.cs
[gate-npy]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.UnitTest/IO/NpyOracleTests.cs
