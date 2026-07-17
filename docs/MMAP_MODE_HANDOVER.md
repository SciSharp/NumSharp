# Handover — `np.load(mmap_mode=…)` for NumSharp

> **✅ IMPLEMENTED (2026-07-17).** This design shipped: `NpyFormat.OpenMemmap` +
> `ResolveMmapMode` in `IO/NpyFormat.cs`, dispatch rewired in `APIs/np.load.cs`, 16 tests in
> `test/NumSharp.UnitTest/IO/NpyMemmapTests.cs` (green on net8.0 + net10.0), and every claim below
> re-verified against NumPy 2.4.2. The document is kept as the design record and the parity
> reference. Notable as-built refinements vs the plan: the read-only-shape install reuses the
> existing `UnmanagedStorage.SetShapeUnsafe` (no new infra was needed — §5's "one infra gap" was
> already covered), and `w+` raises the same `TypeError` **without** reproducing NumPy's file
> truncation (a destructive bug we decline to copy).

Status date: 2026-07-17 · Branch: `worktree-npsave` · Scope: `np.load` / `NpyFormat`
(`APIs/np.load.cs`, `IO/NpyFormat.cs`).

This specifies the follow-up for the two `mmap_mode` divergences found while auditing the
`np.load`/`save` family (the other five were fixed in `6b5dc184`; these were deliberately left
for a dedicated design because they need a real memory-mapping backend):

- **#1 — `np.load(<npz>, mmap_mode=…)` is wrongly rejected.** NumPy ignores `mmap_mode` for zip
  archives and returns the `NpzFile`; NumSharp throws `NotImplementedException` because
  `CheckMmapMode` runs *before* file-type detection.
- **#4 — the accepted mode set is wrong.** NumPy accepts **8** mode strings; NumSharp accepts 4
  and rejects the long-form aliases as invalid.

Both are symptoms of the same root cause: **`mmap_mode` is validated and acted on in the wrong
place, and the mapping backend was never built.** This document is the recipe to build it, with
every design assumption proven by probing NumPy 2.4.2 and by running the .NET / NumSharp
mechanism end-to-end (see §10).

> **Everything below is proven, not assumed.** NumPy behavior is from the installed 2.4.2 source
> (`_core/memmap.py`, `lib/_format_impl.py`, `lib/_npyio_impl.py`) plus live probes; the .NET and
> NumSharp mechanisms were each executed and the outputs are pasted verbatim in §10.

---

## 0. TL;DR — what to build

1. **Delete the up-front `CheckMmapMode` calls** from the three `load(…)` overloads. `mmap_mode`
   must only be inspected on the **`.npy`** branch of `LoadCore` — npz and pickle ignore it
   entirely (even an *invalid* value), which is exactly what NumPy does.
2. **Add `NpyFormat.OpenMemmap(string path, string mode, long maxHeaderSize)`** that reads the
   header, maps the file with `MemoryMappedFile`, wraps the view pointer in an
   `UnmanagedMemoryBlock<T>` whose free-callback releases the map, and returns an `NDArray`.
   The wrapping ctor and lifetime hook **already exist** — see §5.
3. **Support the 4 modes that actually work through `np.load`**: `r` / `readonly` (read-only),
   `r+` (read-write, persists), `c` (copy-on-write). Reproduce NumPy's errors for the other 4
   (`w+`, `write`, `copyonwrite`, `readwrite`) — they *validate* but then fail downstream in
   NumPy too (§2, §3). This is a NumPy quirk we match, not a feature we add.
4. **mmap needs a real path.** The `Stream` and `byte[]` overloads must reject `mmap_mode` on the
   npy branch with NumPy's fileobj error, but still ignore it for npz.
5. **Reject the un-mappable dtypes** (big-endian, `<U1`/Char, `<c8`/complex64) — NumSharp
   *converts* those on a normal read, which a zero-copy view cannot do.

---

## 1. Why — the two findings, reproduced

```
                              NumPy 2.4.2                 NumSharp (today)
np.load(npz, mmap_mode="r")   NpzFile (mmap ignored)      NotImplementedException   ← #1
np.load(npy, mmap_mode="readonly")  memmap (== 'r')       ArgumentException         ← #4
np.load(npy, mmap_mode="zzz") ValueError [8-mode msg]     ArgumentException [4-mode msg]
```

Current NumSharp (`APIs/np.load.cs`):

```csharp
private static readonly string[] ValidMmapModes = { "r", "r+", "w+", "c" };   // ← 4, wrong set

public static object load(string file, string mmap_mode = null, …) {
    CheckEncoding(encoding);
    CheckMmapMode(mmap_mode);      // ← runs for EVERY file kind, before detection → both bugs
    …
}
// CheckMmapMode: ArgumentException if not in the 4; NotImplementedException if it is.
```

---

## 2. Proven NumPy behavior matrix

### 2a. The 8 accepted mode strings (and which actually work through `np.load`)

`numpy.memmap` (`_core/memmap.py:12-15`):

```python
valid_filemodes    = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+", "w+"]
mode_equivalents = {"readonly": "r", "copyonwrite": "c", "readwrite": "r+", "write": "w+"}
```

The validation error lists `valid_filemodes + list(mode_equivalents)` — this exact order:

```
mode must be one of ['r', 'c', 'r+', 'w+', 'readonly', 'copyonwrite', 'readwrite', 'write'] (got 'zzz')
```

But **through `np.load` only four of the eight succeed** (probed, §10-A):

| mmap_mode      | result through `np.load` on a `.npy`      | semantics            |
|----------------|-------------------------------------------|----------------------|
| `r`            | ✅ memmap, `WRITEABLE=False`               | read-only            |
| `readonly`     | ✅ memmap, `WRITEABLE=False` (aliases `r`) | read-only            |
| `r+`           | ✅ memmap, `WRITEABLE=True`, **persists**  | read-write           |
| `c`            | ✅ memmap, `WRITEABLE=True`, **no persist**| copy-on-write        |
| `copyonwrite`  | ❌ `ValueError: invalid mode: 'copyonwriteb'` | (should be `c`)   |
| `readwrite`    | ❌ `ValueError: invalid mode: 'readwriteb'`   | (should be `r+`)  |
| `write`        | ❌ `ValueError: invalid mode: 'writeb'`       | (should be `w+`)  |
| `w+`           | ❌ `TypeError: object of type 'NoneType' has no len()` | (create mode) |

The write-through persistence was proven by writing `[0,0]=999` through the map and reloading the
file normally (§10-A): `r+` → disk shows 999; `c` → disk still 0.

### 2b. Where `mmap_mode` is validated — **only on the `.npy` branch**

`np.load` reads the magic first, then dispatches (`lib/_npyio_impl.py:475-485`). `mmap_mode` is
consumed **only** in the `.npy` branch (`format.open_memmap`). npz and pickle never see it.
Proven (§10-B):

```
npz + mmap='garbage'  -> OK NpzFile     (ignored, NOT validated)
npz + mmap='r'        -> OK NpzFile     (ignored)
npy + mmap='garbage'  -> ValueError [8-mode msg]   (validated, in memmap)
garbage + mmap='gbg'  -> UnpicklingError            (pickle branch; mmap ignored)
```

**Consequence for the design:** the up-front `CheckMmapMode` is wrong twice — it rejects valid
modes on npz *and* validates invalid modes on npz/pickle. Move it into the npy branch.

### 2c. mmap needs a real path (fileobj is rejected)

Proven (§10-B):

```
np.load(<BytesIO npy>, mmap_mode='r')     -> TypeError: expected str, bytes or os.PathLike object, not BytesIO
np.load(<open file handle>, mmap_mode='r')-> ValueError: Filename must be a string or a path-like object.  Memmap cannot use existing file handles.
```

So `NumSharp.load(Stream, mmap_mode)` / `load(byte[], mmap_mode)` **on an npy** must raise the
`ValueError` above (a stream/`byte[]` has no mappable file); on an **npz** they still ignore it.

---

## 3. NumPy's exact mechanism (so the parity quirk is intentional, not copied blind)

`open_memmap` (`lib/_format_impl.py:891-995`) routes on **`'w' in mode`**:

```python
if 'w' in mode:                       # WRITE branch — expects to CREATE the file
    ...
    with open(os.fspath(filename), mode + 'b') as fp:   # open(file, 'copyonwriteb') → invalid mode!
        _write_array_header(fp, d, version)             # d['shape'] is None → len(None) TypeError
        offset = fp.tell()
else:                                 # READ branch
    with open(os.fspath(filename), 'rb') as fp:
        version = read_magic(fp); _check_version(version)
        shape, fortran_order, dtype = _read_array_header(fp, ...)
        offset = fp.tell()            # the 64-byte-aligned data offset (128 for small headers)
order = 'F' if fortran_order else 'C'
if mode == 'w+': mode = 'r+'          # only exact 'w+' is remapped
marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order, mode=mode, offset=offset)
```

The asymmetry in §2a is now explained: **`copyonwrite`, `readwrite`, `write` all contain the
letter `w`**, so `'w' in mode` misroutes them into the write branch, where `open(file, mode+'b')`
is an invalid file-open string. `readonly` has no `w`, reaches `numpy.memmap`, and `mode_equivalents`
aliases it to `r`. `w+` reaches the write branch and dies on `len(shape=None)` because `np.load`
passes no `shape`.

`numpy.memmap.__new__` then does the page-alignment (`_core/memmap.py:283-296`):

```python
start = offset - offset % mmap.ALLOCATIONGRANULARITY   # round the map start DOWN to granularity
bytes = int(offset + size * _dbytes)
array_offset = offset - start                          # how far into the map the data begins
mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)
self = ndarray.__new__(subtype, shape, dtype, buffer=mm, offset=array_offset, order=order)
self.offset = offset
```

**This is the crux the `CheckMmapMode` sketch worried about** — and **.NET solves it for you**
(§4): `MemoryMappedViewAccessor.PointerOffset` *is* `array_offset`.

---

## 4. The .NET mechanism — proven end-to-end

`MemoryMappedFile` + `MemoryMappedViewAccessor` + `SafeMemoryMappedViewHandle.AcquirePointer`
gives a raw `byte*`, in all three access modes. Proven (§10-C), full script + output:

```
READ  values : [0,1,2,3,4,5] PointerOffset=0          (mapped whole file from 0)
R+W   on-disk: [999,1,2,3,4,5]                          (write-through persists after Flush)
CoW   on-disk: [0,1,2,3,4,5]  (in-mem was 777)          (copy-on-write does NOT persist)
```

Access-mode mapping (create the `MemoryMappedFile` **and** the view with the same access):

| mmap_mode | FileStream access | MemoryMappedFileAccess | writeable? | persists? |
|-----------|-------------------|------------------------|------------|-----------|
| `r`/`readonly` | `FileAccess.Read`      | `Read`         | no  | —   |
| `r+`      | `FileAccess.ReadWrite` | `ReadWrite`    | yes | yes |
| `c`       | `FileAccess.Read`      | `CopyOnWrite`  | yes | no  |

**Alignment is handled by .NET.** `CreateViewAccessor(offset, size, access)` maps from an
allocation-granularity-aligned base and exposes `PointerOffset` as the delta back to your
requested `offset` — the exact `array_offset` trick. Proven (§10-D):

```
view@offset=128: PointerOffset=128  values=[0,1,2,3,4,5]
```

So either approach works: **(a)** map the whole file from 0 (`CreateViewAccessor(0,0,…)`,
`PointerOffset==0`) and index `basePtr + dataOffset`, or **(b)** `CreateViewAccessor(dataOffset, n*itemsize, …)` and index `basePtr + PointerOffset`. Approach (a) is simplest.

---

## 5. NumSharp wiring — proven, and the sketch's premise is outdated

The `CheckMmapMode` sketch says *"today `UnmanagedMemoryBlock` assumes it owns its allocation."*
**That is no longer true.** The exact hook already exists (``UnmanagedMemoryBlock`1.cs``:78):

```csharp
/// Construct with externally allocated memory and a custom `dispose` function. Claims ownership.
public UnmanagedMemoryBlock(T* start, long count, Action dispose)
```

It builds an `AllocationType.External` `Disposer` whose `ReleaseUnmanagedResources()` calls your
`_dispose()` **exactly once** (single-shot `_freed` guard) and participates in the block's ARC
refcount — so the map is released when the last NDArray reference drops, or force-released via
`block.Free()`. The whole path was run end-to-end (§10-C, part 4):

```csharp
Action free = () => { view.SafeMemoryMappedViewHandle.ReleasePointer();
                      view.Dispose(); mmf.Dispose(); fs.Dispose(); };
var block = new UnmanagedMemoryBlock<int>(dataPtr, count, free);   // External alloc + free-callback
var nd    = new NDArray(new ArraySlice<int>(block), new Shape(2,3));
// -> nd.flat == [0,1,2,3,4,5]; block.Free() runs `free` (releases view+map+file). Proven.
```

Read-only enforcement uses the same mechanism broadcasts use — `Shape.WithFlags` (proven
`IsWriteable=False`, §10-C):

```csharp
Shape readonlyShape = shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
```

Fortran-order stays **zero-copy**: `reshape(shape[::-1]).T` shares the same `Storage.Address`
(proven §10-E — same pointer, mutation visible through the transpose). So an F-order file maps the
flat buffer and transposes, exactly like `NpyFormat.ReadArray`'s `reshape(reversed).T`, with no
materialization.

---

## 6. The design

### 6a. `NpyFormat.OpenMemmap` (new)

```csharp
// IO/NpyFormat.cs
public static NDArray OpenMemmap(string path, string mode, long maxHeaderSize = MaxHeaderSize)
{
    // 1. Resolve the mode exactly as numpy.memmap does, and reproduce open_memmap's 'w' misroute
    //    for parity (see §8 for the decision on whether to keep this bug-for-bug).
    string fileMode = ResolveMmapMode(mode);   // throws the verbatim NumPy errors; see below

    // 2. Read the header to learn shape / fortran_order / dtype / dataOffset.
    //    (r/c open Read; r+ opens ReadWrite — but the header read is identical.)
    using (var probe = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
    {
        FormatVersion ver = ReadMagic(probe); CheckVersion(ver);
        HeaderData h = ReadArrayHeader(probe, ver, maxHeaderSize);
        long dataOffset = probe.Position;

        // 3. Reject what cannot be a zero-copy view (§7).
        if (h.Dtype.HasObject) throw …"Object arrays cannot be loaded…";
        if (h.Dtype.NeedsSwap) throw new NotSupportedException(
            "Big-endian .npy files cannot be memory-mapped: NumSharp byte-swaps to native on read, " +
            "which a zero-copy view rules out. Load without mmap_mode.");
        if (h.Dtype.Conversion != ElementConversion.None) throw new NotSupportedException(
            $"'{DtypeDescrOf(h)}' is stored at a different width than its NumSharp type (Char/complex64) " +
            "and is converted on read, so it cannot be memory-mapped. Load without mmap_mode.");

        // 4. Map + wrap (type-dispatched T; only mappable dtypes reach here).
        return MapAndWrap(path, fileMode, h, dataOffset);
    }
}
```

`ResolveMmapMode` — mirror `numpy.memmap` + the `open_memmap` misroute, verbatim errors:

```csharp
private static string ResolveMmapMode(string mode)
{
    // open_memmap routes on 'w' in mode BEFORE memmap's alias table, so the aliases containing
    // 'w' misroute into the (broken, shape-less) create path. Reproduce NumPy's observable errors.
    if (mode == "w+")                       throw new TypeError("object of type 'NoneType' has no len()");
    if (mode.Contains('w'))                 // 'write','copyonwrite','readwrite'
        throw new ValueError($"invalid mode: '{mode}b'");

    switch (mode)                            // memmap.mode_equivalents (the no-'w' entries) + valid_filemodes
    {
        case "readonly": return "r";
        case "r": case "c": case "r+": return mode;
        default:
            throw new ValueError(
                "mode must be one of ['r', 'c', 'r+', 'w+', 'readonly', 'copyonwrite', 'readwrite', 'write'] " +
                $"(got '{mode}')");
    }
}
```

`MapAndWrap` — the 15-way dtype switch (only the mappable subset is reachable), each arm building
`UnmanagedMemoryBlock<T>(ptr, count, free)`:

```csharp
private static unsafe NDArray MapAndWrap(string path, string fileMode, HeaderData h, long dataOffset)
{
    (FileAccess fa, MemoryMappedFileAccess mma, bool writeable) = fileMode switch {
        "r"  => (FileAccess.Read,      MemoryMappedFileAccess.Read,        false),
        "r+" => (FileAccess.ReadWrite, MemoryMappedFileAccess.ReadWrite,   true),
        "c"  => (FileAccess.Read,      MemoryMappedFileAccess.CopyOnWrite, true),
        _    => throw new InvalidOperationException()  // ResolveMmapMode already filtered
    };

    var fs   = new FileStream(path, FileMode.Open, fa, FileShare.ReadWrite);
    var mmf  = MemoryMappedFile.CreateFromFile(fs, null, 0, mma, HandleInheritability.None, leaveOpen: false);
    var view = mmf.CreateViewAccessor(0, 0, mma);
    byte* basePtr = null;
    view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
    byte* dataPtr = basePtr + view.PointerOffset + dataOffset;
    Action free = () => { view.SafeMemoryMappedViewHandle.ReleasePointer(); view.Dispose(); mmf.Dispose(); fs.Dispose(); };

    long count = h.Count;                    // product(shape); 0-d => 1
    NDArray flat = h.Dtype.TypeCode switch { // one arm per mappable dtype (bool,i1..u8,i2..u8,f2,f4,f8,c16)
        NPTypeCode.Int32  => new NDArray(new ArraySlice<int>   (new UnmanagedMemoryBlock<int>   ((int*)   dataPtr, count, free)), FlatShape(count)),
        NPTypeCode.Double => new NDArray(new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)dataPtr, count, free)), FlatShape(count)),
        …  // remaining mappable dtypes
        _ => throw new NotSupportedException($"{h.Dtype.TypeCode} cannot be memory-mapped.")
    };

    NDArray nd = ShapeFromHeader(flat, h);   // C-order: reshape(shape); F-order: reshape(reversed).T (zero-copy)
    if (!writeable)
        SetShapePreservingFlags(nd, nd.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
    return nd;
}
```

> **One infra gap to close:** installing a non-writeable Shape. `UnmanagedStorage.Shape`'s setter
> routes through `Reshape`, which **recomputes** `_flags` and would drop the cleared `WRITEABLE`.
> Add a small internal `SetShapePreservingFlags` (or a `Reshape` overload that keeps an explicit
> flag mask) — the broadcast path already constructs storage with a pre-cleared shape, so mirror
> that. `Shape.WithFlags` itself is proven to produce `IsWriteable=false`.

### 6b. `np.load.cs` changes

```csharp
// DELETE the three up-front CheckMmapMode(mmap_mode) calls.

// In LoadCore, thread the mmap_mode + a path (null for stream/bytes) through, and act on it ONLY
// in the npy branch:
private static object LoadCore(Stream stream, bool ownStream, bool allowPickle, long maxHeaderSize,
                               string mmapMode, string pathOrNull)
{
    …
    if (NpyFormat.IsNpzFile(stream))
        return new NpzFile(stream, ownStream, allowPickle, maxHeaderSize);   // mmapMode IGNORED (parity)

    if (NpyFormat.IsNpyFile(stream))
    {
        if (mmapMode != null)
        {
            if (pathOrNull == null)      // Stream / byte[] overload → NumPy's fileobj rejection
                throw new ValueError("Filename must be a string or a path-like object.  " +
                                     "Memmap cannot use existing file handles.");
            return NpyFormat.OpenMemmap(pathOrNull, mmapMode, allowPickle ? long.MaxValue : maxHeaderSize);
        }
        return NpyFormat.ReadArray(stream, allowPickle, maxHeaderSize);
    }
    …  // pickle branch: mmapMode IGNORED
}
```

`load(string file, …)` passes `pathOrNull: file`; `load(Stream)`/`load(byte[])` pass
`pathOrNull: null`. The `CheckEncoding` call stays where it is.

---

## 7. Un-mappable inputs (must reject, with a clear message)

A memmap is a **zero-copy reinterpretation of the file bytes**. Anything NumSharp *transforms* on a
normal read cannot be mapped:

| Input | Why it can't be mapped | Action |
|-------|------------------------|--------|
| Big-endian (`>i4`, `>c16`, …) | `NpyFormat` byte-swaps to native on read | `NotSupportedException` (or NumPy would map a byte-swapped dtype — NumSharp has none) |
| `<U1` (Char) | 4-byte UCS-4 in file ↔ 2-byte UTF-16 Char (`ElementConversion.Ucs4`) | `NotSupportedException` |
| `<c8` (complex64) | 2×float32 in file → 2×float64 `Complex` (`ElementConversion.Complex64`) | `NotSupportedException` |
| Object (`\|O`) | pickle stream | existing "Object arrays cannot be loaded…" |
| Decimal | never has a `descr` (can't be written) | unreachable |

The mappable set is therefore: `bool, i1, u1, i2, u2, i4, u4, i8, u8, f2, f4, f8, c16` (native-endian).
Everything else falls back to "load without mmap_mode".

Other edges (all cheap to handle):
- **0-d** (`shape=()`): `count=1`, one element mapped; `Shape` scalar. Works.
- **Empty** (`size=0`): `count=0` — `CreateViewAccessor(0,0,…)` still maps the header bytes; there
  are simply no elements to point at. Return an empty NDArray; consider skipping the map entirely.
- **allow_pickle=True** lifts `maxHeaderSize` to `long.MaxValue`, same as `ReadArray` (already in §6b).

---

## 8. Parity decision points (call these out to the maintainer)

1. **The 4 broken modes (`w+`, `write`, `copyonwrite`, `readwrite`).** NumPy *validates* them then
   fails downstream with specific errors (§2a). Two options:
   - **(A) Strict bug-for-bug parity** — reproduce the verbatim errors (`ResolveMmapMode` above
     does this). Matches the differential-fuzz philosophy; a user porting code sees the same
     failure. **Recommended** (and cheapest to pin in a test).
   - **(B) Fix NumPy's bug** — treat `copyonwrite`→`c`, `readwrite`→`r+`, `write`/`w+`→(reject,
     no shape) since the misroute is clearly unintended. More useful, but *diverges* from NumPy
     and would need a `[Misaligned]` note. Only do this if the maintainer wants NumSharp to be a
     superset here (as it already is for parenthesized-complex `fromfile`).
2. **`r+`/`c` writeability vs NumSharp's write kernels.** A mapped `r+` array is a genuine writable
   `NDArray` — in-place ops write through to disk on flush. Confirm the engine's in-place paths
   don't assume owned/poolable memory (they operate on `Storage.Address`, so they should be fine,
   but a mapped buffer must never be returned to `SizeBucketedBufferPool` — the `External` disposer
   guarantees this: its release calls `_dispose()`, never `Pool.Return`).
3. **Flush semantics.** NumPy's memmap flushes on `.flush()`/GC. Decide whether NumSharp exposes an
   explicit flush or flushes on dispose (`MemoryMappedViewAccessor.Flush()` before `Dispose`).

---

## 9. Test plan

- **Unit (`IO/`):** one test per working mode on a round-tripped `.npy` — `r`/`readonly` read
  correct values + `IsWriteable==false`; `r+` write-through persists (reload, assert); `c` write
  stays in memory (reload, assert unchanged). Assert the 4 broken modes throw the verbatim
  errors. Assert big-endian / Char / complex64 files reject with the §7 message.
- **Dispatch parity:** `load(npz, mmap_mode="r")` → `NpzFile`; `load(npz, mmap_mode="garbage")` →
  `NpzFile` (ignored); `load(npy, mmap_mode="garbage")` → the 8-mode `ValueError`;
  `load(Stream/byte[] npy, mmap_mode="r")` → the fileobj `ValueError`.
- **Lifetime:** map, drop all references, force GC, assert the file handle is released (open the
  path `FileShare.None` afterward, or assert via a dispose-flag as in §10-C part 4).
- **NpyOracle:** add mmap read cases to the oracle harness if a mapped read must be bit-identical
  to a normal read (it is — same bytes).

---

## 10. Proof appendix — probe scripts & verbatim outputs

### 10-A. NumPy mode semantics (`python`)

```python
for mode in ["r","r+","c","readonly","copyonwrite","readwrite","w+","write"]:
    p = fresh_int32_npy(); m = np.load(p, mmap_mode=mode); m[0,0]=999; m.flush(); del m
    print(mode, np.load(p)[0,0])   # persisted?
```
```
r          -> W=False, write raises "assignment destination is read-only"
r+         -> W=True,  on-disk 999 (persisted)
c          -> W=True,  on-disk 0   (NOT persisted)
readonly   -> W=False, write raises  (aliases r)
copyonwrite-> LOAD raised ValueError: invalid mode: 'copyonwriteb'
readwrite  -> LOAD raised ValueError: invalid mode: 'readwriteb'
w+         -> LOAD raised TypeError: object of type 'NoneType' has no len()
write      -> LOAD raised ValueError: invalid mode: 'writeb'
```

### 10-B. Where NumPy validates + fileobj rejection (`python`)

```
npz + mmap='garbage'  -> OK NpzFile          npy + mmap='garbage' -> ValueError [8-mode msg]
npz + mmap='r'        -> OK NpzFile          garbage + mmap='gbg' -> UnpicklingError
BytesIO npy + mmap='r'     -> TypeError: expected str, bytes or os.PathLike object, not BytesIO
open file npy + mmap='r'   -> ValueError: Filename must be a string or a path-like object.  Memmap cannot use existing file handles.
```

### 10-C. .NET `MemoryMappedFile` + NumSharp wrap (`dotnet_run`, abridged — full script in git history of this doc's commit)

```
data offset = 128
READ  values : [0,1,2,3,4,5] PointerOffset=0
R+W   on-disk: [999,1,2,3,4,5]              (write-through persists)
CoW   in-mem : 777
CoW   on-disk: [0,1,2,3,4,5]                (copy-on-write, not persisted)
WRAP  NDArray: shape=[2,3] data=[0,1,2,3,4,5]
WRAP  read-only via WithFlags: IsWriteable=False
WRAP  dispose-callback fired (view+map+file released) = True
```

Core of the wrap (the load-bearing lines):

```csharp
byte* bp=null; view.SafeMemoryMappedViewHandle.AcquirePointer(ref bp);
int* dataPtr = (int*)(bp + view.PointerOffset + dataOffset);
Action free = () => { view.SafeMemoryMappedViewHandle.ReleasePointer(); view.Dispose(); mmf.Dispose(); fs.Dispose(); };
var block = new UnmanagedMemoryBlock<int>(dataPtr, count, free);
var nd    = new NDArray(new ArraySlice<int>(block), new Shape(2,3));
// nd.flat -> [0,1,2,3,4,5]; nd.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE).IsWriteable -> False
// block.Free() -> free() runs exactly once (proven).
```

### 10-D. .NET aligns the view for you (`dotnet_run`)

```csharp
using var view = mmf.CreateViewAccessor(off, 6*sizeof(int), MemoryMappedFileAccess.Read);
int* p = (int*)(bp + view.PointerOffset);        // PointerOffset == off (== array_offset)
// -> "view@offset=128: PointerOffset=128 values=[0,1,2,3,4,5]"
```

### 10-E. Fortran-order `reshape().T` is zero-copy (`dotnet_run`)

```csharp
var flat = np.arange(6).astype(NPTypeCode.Int32);
var v = flat.reshape(3,2).T;
// flat.Storage.Address == v.Storage.Address : True
// flat[0]=42 -> v[0,0]==42 ; v.Fcontig=True, contiguous=False
```

---

## 11. Source map (where everything lives)

| Piece | Location |
|-------|----------|
| Up-front `CheckMmapMode` to delete; `ValidMmapModes` | `APIs/np.load.cs:13,69,336` |
| `LoadCore` dispatch (add npy-branch mmap) | `APIs/np.load.cs:274` |
| New `OpenMemmap` / `ResolveMmapMode` / `MapAndWrap` | `IO/NpyFormat.cs` (next to `ReadArray`) |
| Header read (`ReadMagic`, `ReadArrayHeader`, `HeaderData`, `DtypeInfo.NeedsSwap`, `ElementConversion`) | `IO/NpyFormat.cs` |
| Wrapping ctor (External alloc + free-callback) | ``Backends/Unmanaged/UnmanagedMemoryBlock`1.cs``:78 |
| `ArraySlice<T>(block)` → `NDArray(slice, shape)` | ``Backends/Unmanaged/ArraySlice`1.cs``:51, `Backends/NDArray.cs:287` |
| `Shape.WithFlags` (read-only) | `View/Shape.cs:569` |
| Non-writeable-shape install gap (`Reshape` recomputes flags) | `Backends/Unmanaged/UnmanagedStorage.cs:166` |
| NumPy reference | `_core/memmap.py`, `lib/_format_impl.py:891` (`open_memmap`), `lib/_npyio_impl.py:475` |
