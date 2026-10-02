# NumSharp.Interop.ParquetNet

Load **Apache Parquet** columns directly into NumSharp `NDArray`s — through the fully-managed
[Parquet.Net](https://github.com/aloneguid/parquet-dotnet), with **no Pandas.NET and no native code**.

Parquet is columnar, so a column maps almost 1:1 onto an `NDArray`. Parquet.Net's read API is
*buffer-based* (`ReadAsync<T>(field, Memory<T>)` decodes into a caller-supplied buffer), so this bridge
hands it a `Memory<T>` window over the NDArray's own unmanaged block: for a **required (non-null) column
the decoded values land straight in NDArray memory — no intermediate managed array and no extra copy.**

## Install

```
dotnet add package NumSharp.Interop.ParquetNet   # pulls NumSharp + the managed Parquet.Net
```

## Use — the NumSharp face (`ParquetFile`, `NpzFile`-style)

```csharp
using NumSharp;
using NumSharp.Interop.ParquetNet;

using ParquetFile pf = ParquetFile.Load("prices.parquet");
NDArray close  = pf["close"];      // decoded into raw memory on first access, then cached
NDArray volume = pf.f.volume;      // dot-access
string[] cols  = pf.Columns;       // loadable column names (schema only)
long rows      = pf.RowCount;

// projection — only these columns are ever read from disk / committed to memory
using var p2 = ParquetFile.Load("big.parquet", new ParquetLoadOptions { Columns = new[] { "a", "b" } });

// bounded-memory streaming, one row group at a time
foreach (ParquetRowGroup g in pf.RowGroups)
    using (g) Accumulate(g.Column("value"));
```

## Use — the Parquet.Net face (extensions on Parquet.Net's own types)

```csharp
using Parquet;   // the extensions live in Parquet.Net's namespace, so they light up via IntelliSense

// whole file → NDArrays (the analog of Parquet.Net's ReadParquetAsDataFrameAsync)
Dictionary<string, NDArray> cols = await stream.ReadParquetAsNDArraysAsync();

// one column off a reader you already opened
await using ParquetReader reader = await ParquetReader.CreateAsync(stream);
NDArray col = await reader.ReadColumnAsNDArrayAsync("close");

// inside your own per-row-group loop — one line different from what you write today
using ParquetRowGroupReader rg = reader.OpenRowGroupReader(0);
NDArray a = await rg.ReadAsNDArrayAsync(field);

// or decode into an NDArray you allocate, with Parquet.Net's own call:
await rg.ReadAsync<double>(field, nd.AsMemory<double>());
```

## Nulls

NumSharp has no native NA. The default policy is **`Fill`** — nulls become the option's `NullFill`, or
`default(T)` (0 / false) when none is given:

```csharp
NDArray q  = pf.Column("qty");                 // nulls -> 0
NDArray q2 = pf.Column("qty", nullFill: -1);   // nulls -> -1
using var pf2 = ParquetFile.Load(path, new ParquetLoadOptions { Nulls = NullHandling.Raise }); // throw on nulls
```

## Supported dtypes

The twelve Parquet.Net CLR column types that map 1:1 onto a NumSharp dtype:
`bool, byte, sbyte, short, ushort, int, uint, long, ulong, float, double, decimal`.

Rejected with a clear message (no NumSharp dtype, or not rectangular): `DateTime` / `DateOnly` /
`TimeOnly`, `BigDecimal`, `BigInteger`, `Interval`, `Guid`, `string` / `byte[]`, and any nested (struct /
map) or repeated (list / array) field. `Half`, `Char` and `Complex` have no Parquet.Net source on the read
path.

## Notes

- **Decode is mandatory.** Parquet columns are compressed/encoded on disk, so — unlike a `.npy` mmap —
  there is no zero-copy view over the *file* bytes. Zero-copy applies *after* decode (slices/transposes of
  the decoded NDArray), and "chunking" means *row groups*.
- Async-first (`ParquetReader` is async); the `ParquetFile.Load` / `Column` / `RowGroups` surface wraps the
  async calls, and `*Async` variants are available throughout.
- Column NDArrays outlive the `ParquetFile` — each owns its pooled unmanaged buffer via NumSharp's
  reference counting, independent of the file.
