# I/O with NumSharp

NumSharp reads and writes the same file formats NumPy does, and — for the `.npy`/`.npz` binary format — its output is **byte-for-byte identical** to `np.save`, not merely readable by NumPy. This page covers the three families of I/O:

- **`.npy` / `.npz`** — NumPy's native binary array format (the default choice).
- **Text** — `loadtxt` / `savetxt` for CSV/TSV and other delimited data.
- **Raw binary / bytes** — `fromfile` / `tofile` / `frombuffer` / `fromstring`.

> NumPy's own *I/O* fundamentals article is entirely about `genfromtxt`, which NumSharp does **not** implement (it requires structured/masked dtypes NumSharp has no analog for). This page instead documents the I/O surface NumSharp actually provides. See [What is not implemented](#what-is-not-implemented) at the end.

<!-- Tests: NumSharp.Tests.Documentation.FundamentalsIoDocTests — every code example on this page is executed and asserted in test/NumSharp.Tests/Documentation/FundamentalsIoDocTests.cs. Section → method(s):
     .npy save/load round-trip → Npy_SaveLoad_RoundTrip
     .npz dictionary keys + npz.f dot access → Npz_DictionaryKeys_And_DotAccess
     np.load returns object dispatched on kind → Load_ReturnsObject_DispatchedOnKind
     mmap_mode 'r' is read-only → Mmap_Mode_R_IsReadOnly
     savetxt/loadtxt round-trip → SaveTxt_LoadTxt_RoundTrip
     fromstring (+ binary-mode-removed throw) → FromString_ParsesNumbers; FromString_BinaryModeRemoved_Throws
     raw tofile/fromfile → Raw_TofileFromfile_RoundTrip -->

---

## `.npy` and `.npz` — the native format

### Writing

```csharp
np.save("array.npy", arr);                       // single array → .npy
np.savez("bundle.npz", a, b, c);                 // several arrays → .npz (named arr_0, arr_1, …)
np.savez_compressed("bundle.npz", a, b);         // deflate-compressed .npz

// named members via a dictionary:
np.savez("bundle.npz", new Dictionary<string, NDArray> { ["weights"] = w, ["bias"] = bias });
```

The writer is a port of NumPy 2.4.2's `_format_impl.py`, so a saved file is **exactly** what `np.save` would produce — format versions 1.0/2.0/3.0, 64-byte data alignment (mmap-ready), `fortran_order`, and the same header bytes. NumPy can read NumSharp's output and vice versa.

### Reading

```csharp
object loaded = np.load("array.npy");            // NDArray for .npy, NpzFile for .npz
NDArray arr   = np.load_npy("array.npy");        // typed — no cast
using NpzFile npz = np.load_npz("bundle.npz");   // typed — IDisposable

NDArray w = npz["weights"];                       // by key
NDArray b = npz.f.bias;                            // BagObj dot-access (NumPy's npz.f.name)
```

`np.load` returns `object` because NumPy's `np.load` is content-dependent (an array for `.npy`, an archive for `.npz`) and C# has no union type. Prefer the typed `np.load_npy` / `np.load_npz` when you know the kind; each gives a directed error if handed the wrong one. **`NpzFile` is lazy and cached, and must be disposed** (`using`).

### Memory-mapping large files

`mmap_mode` returns an `NDArray` backed by a memory-mapped view — zero-copy, released with the array:

```csharp
var big = (NDArray)np.load("huge.npy", mmap_mode: "r");   // read-only view
var rw  = (NDArray)np.load("huge.npy", mmap_mode: "r+");  // read-write, flushes to disk
```

Modes match what NumPy does through `np.load`: `"r"`/`"readonly"`, `"r+"` (read-write), `"c"` (copy-on-write). mmap needs a real file **path** (a stream/`byte[]` cannot be mapped), and NumSharp refuses to map files it would otherwise transform on read (big-endian, `<U1`/`<c8`) with a clear `NotSupportedException`.

### The dtype map

| NumPy descriptor | NumSharp dtype | Notes |
|------------------|----------------|-------|
| `\|b1`, `\|i1`, `\|u1` | Boolean, SByte, Byte | single-byte types use the `\|` prefix |
| `<i2`…`<u8`, `<f2`, `<f4`, `<f8` | Int16…UInt64, Half, Single, Double | direct |
| `<c16` | Complex | `<c8` (complex64) widens to Complex on read |
| `<U1` | Char | 2-byte UTF-16 ↔ 4-byte UCS-4; non-BMP rejected |
| — | Decimal | **`NotSupportedException`** — no NumPy dtype |

Big-endian files are byte-swapped to native on read. Object arrays, structured/subarray dtypes, `datetime64`/`timedelta64`, `S`/`U`(n>1)/`V`, `<f16`/`<c32` parse and then raise a precise message.

---

## Text I/O

### `np.savetxt` — write delimited text

```csharp
np.savetxt("out.csv", arr);                                  // default fmt "%.18e", space delimiter
np.savetxt("out.csv", arr, fmt: "%.3f", delimiter: ",");     // 2-D → one row per line
np.savetxt("out.csv", arr, fmt: "%d", header: "x,y", comments: "# ");
```

A port of NumPy 2.4.2's `savetxt`, **byte-identical to NumPy's output**. A 1-D array writes one value per line; a 2-D array one row per line; 0-D/≥3-D raise `ValueError`. `fmt` is a Python `%`-format spec (single spec repeated per column, a multi-`%` template, or a per-column list). Targets: a filename (`.gz` → gzip), a `Stream`, or a `TextWriter`.

### `np.loadtxt` — read delimited text

```csharp
var m = np.loadtxt("data.csv", delimiter: ",", skiprows: 1);
var cols = np.loadtxt("data.txt", usecols: new[] { 0, 2 }, dtype: np.float64);
```

Round-trips `savetxt` (byte-exact values). Parameters mirror NumPy: `dtype` (default float64), `comments`, `delimiter` (whitespace-runs when unset), `converters`, `skiprows`, `usecols`, `unpack`, `ndmin`, `max_rows`, `quotechar`. Inputs: filename (`.gz` transparent), `Stream`, `TextReader`, or `IEnumerable<string>`. Parsers match NumPy's C reader — bool via int, range-checked integers, `PyOS_string_to_double` float semantics, and lowercase-`j` complex.

### `np.fromstring` — parse numbers from a string

```csharp
np.fromstring("1 2 3 4", sep: " ");         // [1. 2. 3. 4.]
np.fromstring("1,2,3", np.int32, sep: ",");
```

Shares `fromfile`'s text parser. The **binary mode was removed in NumPy 1.22**, so an empty/`null` `sep` raises `ValueError("The binary mode of fromstring is removed, use frombuffer instead")`, matching NumPy.

---

## Raw binary I/O

```csharp
arr.tofile("data.bin");                       // raw C-order bytes, no header
var back = np.fromfile("data.bin", np.float64);

byte[] buf = File.ReadAllBytes("data.bin");
var view = np.frombuffer(buf, np.float32);    // reinterpret bytes (no copy of the format)
```

`tofile`/`fromfile` are the headerless raw path — you must know the dtype and byte order yourself (there is no self-describing header, unlike `.npy`). `np.frombuffer` reinterprets an existing .NET buffer as an array of a given dtype and is the general bridge for interop — see [Any library via np.frombuffer](../interop/np-frombuffer.md).

---

## Size limits

The `.npy`/`.npz` path streams in 256 KB chunks and round-trips files **larger than 4 GiB** (verified against NumPy at every 32-bit boundary). Array data lives in unmanaged memory, so no managed buffer scales with the array. The only 2 GB ceilings are the `byte[]`-returning convenience overloads (`np.save(arr)` → `byte[]`, `np.savez(...)` → `byte[]`); use the path/stream overloads above 2 GB.

---

## What is not implemented

| NumPy function | Status | Use instead |
|----------------|--------|-------------|
| `np.genfromtxt` | **not implemented** | `np.loadtxt` (needs no missing-value/structured support), or parse yourself |
| `np.fromregex` | **not implemented** | requires structured dtypes; parse with .NET regex + `np.array` |
| Structured/record `.npy` files | **read raises** | `.npy` files with a structured dtype are rejected — NumSharp has no structured dtype ([Structured arrays](structured-arrays.md)) |

Both `genfromtxt` and `fromregex` are built on NumPy's structured-dtype and masked-array machinery, which NumSharp does not model. Their non-structured subset is already covered by `loadtxt`.

---

## Common patterns

### Save/restore model weights

```csharp
np.savez("model.npz", new Dictionary<string, NDArray> { ["w1"] = w1, ["b1"] = b1 });
using var m = np.load_npz("model.npz");
var w1 = m["w1"];
```

### Interop with NumPy across languages

```csharp
np.save("shared.npy", arr);        // Python: np.load("shared.npy") reads it identically
```

### Stream a large file without loading it all

```csharp
var mapped = (NDArray)np.load("huge.npy", mmap_mode: "r");
var slice  = mapped["0:1000"];     // only the touched pages are read
```

---

## Troubleshooting

### "`np.load` returned `object`, I can't index it"
Cast it, or use the typed loader: `np.load_npy(path)` → `NDArray`, `np.load_npz(path)` → `NpzFile`.

### "`NpzFile` leaked / file stayed locked"
`NpzFile` is `IDisposable` and lazy. Wrap it in `using`.

### "Saving a Decimal array threw"
Decimal has no NumPy dtype, so it cannot be written to `.npy`. Cast to `Double` first (`arr.astype(np.float64)`), accepting the precision change.

### "`fromfile` gave garbage"
`fromfile`/`tofile` are headerless — you must pass the exact dtype the data was written with, and the byte order must match. Prefer `.npy` (`np.save`/`np.load`) for self-describing round-trips.

---

## API reference

| Function | Purpose |
|----------|---------|
| `np.save(path, arr)` | write one array to `.npy` (byte-exact with NumPy) |
| `np.savez(path, …)` / `np.savez_compressed(path, …)` | write an `.npz` archive (optionally compressed) |
| `np.load(path, mmap_mode, allow_pickle, max_header_size)` | read `.npy`/`.npz` → `object` |
| `np.load_npy(path)` / `np.load_npz(path)` | typed loads → `NDArray` / `NpzFile` |
| `np.savetxt(fname, X, fmt, delimiter, …)` | write delimited text (byte-exact) |
| `np.loadtxt(fname, dtype, delimiter, skiprows, usecols, …)` | read delimited text |
| `np.fromstring(s, dtype, count, sep)` | parse numbers from a string |
| `arr.tofile(path)` / `np.fromfile(path, dtype)` | raw headerless binary |
| `np.frombuffer(buffer, dtype)` | reinterpret a .NET buffer as an array |

---

## Related reading

- [Array creation](array-creation.md) — I/O is creation mechanism #4/#5.
- [Data types](../dtypes.md) — the dtype map and which types round-trip.
- [Any library via np.frombuffer](../interop/np-frombuffer.md) — the buffer bridge.
- [NumPy I/O guide](https://numpy.org/doc/stable/user/basics.io.html) — the upstream article (genfromtxt).
