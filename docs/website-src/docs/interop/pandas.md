# Pandas interoperability through the shared Python bridge

`NumSharp.Interop.pythonnet` includes a built-in `PandasPythonArrayAdapter` for Pandas
`DataFrame`, `Series`, `Index`, `ExtensionArray`, and derived types. It does not implement another
pointer bridge. The adapter selects Pandas' official [`to_numpy`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_numpy.html)
projection; `NDArrayPythonInterop` still owns dtype, shape, strides, writeability, copying, leases,
GIL policy, and shutdown behavior.

> Verified live on CPython 3.12.12 · NumPy 2.4.2 · pythonnet 3.0.5 ·
> **Pandas 3.0.5** · net8.0/net10.0. The claims on this page are exercised against the installed
> package by 29 tests in `PandasInteropTests`, `PandasInteropEdgeCaseTests`, and
> `PandasInteropRareScenarioTests`.

## Setup

Install Pandas into the interpreter embedded by pythonnet:

```bash
python -m pip install pandas==3.0.5
```

```csharp
Runtime.PythonDLL = "python312.dll"; // or PYTHONNET_PYDLL
PythonEngine.Initialize();

using (Py.GIL())
{
    dynamic pd = Py.Import("pandas");
}
```

There is no compile-time Pandas dependency and no Pandas .NET wrapper dependency.

## Pandas to NumSharp

Use the ordinary bridge verbs directly on the Pandas `PyObject`:

```csharp
using (Py.GIL())
{
    using PyObject series = scope.Eval("pd.Series([1, 2, 3], dtype='i8')");

    using NDArray view = series.AsNDArray(allowReadonly: true);
    using NDArray copy = series.ToNDArray();
}
```

| Verb | Contract |
|---|---|
| `pandasObject.AsNDArray(allowReadonly: true)` | zero-copy only when stable shared storage is proven; otherwise throws |
| `pandasObject.ToNDArray()` | independent C-contiguous snapshot; Pandas may coerce/materialize first |
| `pandasObject.As<NDArray>()` after `RegisterCodec()` | `Auto`: verified view first, then owning copy |

Pandas 3 commonly marks a shared NumPy projection read-only under its mandatory
[`Copy-on-Write`](https://pandas.pydata.org/docs/user_guide/copy_on_write.html) rules. Pass
`allowReadonly:true`; NumSharp returns `Shape.IsWriteable == false`, and guarded writes fail instead
of bypassing Pandas' policy. Calling bare `AsNDArray()` intentionally rejects that read-only source.

### Why `copy:false` is not enough

Pandas documents that `to_numpy(copy:false)` does **not** guarantee no copy. A homogeneous frame can
be one NumPy-backed block, while mixed columns need a common dtype and a materialized array. Nullable,
categorical, sparse, and other extension arrays also decide individually.

For View mode the adapter requests two independent official projections and requires
`numpy.shares_memory(first, second)`. If they do not overlap, no stable Pandas-owned projection can
be proven and View declines with guidance to use Copy or Auto. Empty arrays are accepted because
they have no addressable element whose sharing could be misrepresented. The selected projection
then enters the same locked-buffer / array-interface bridge used by NumPy and Torch.

```text
Pandas object
  → PandasPythonArrayAdapter.to_numpy()
  → stable-overlap check (View only)
  → NDArrayPythonInterop.ToNDArrayView / ToNDArray
  → existing dtype + layout + lease machinery
```

## NumSharp to Pandas

Pandas already accepts NumPy, so the existing NumSharp encoder is the correct route:

```csharp
NDArrayPythonInterop.RegisterCodec();

using (Py.GIL())
{
    scope.Set("values", nd); // implicit NDArray → NumPy view
    scope.Exec("series = pd.Series(values, copy=False)");
    scope.Exec("frame = pd.DataFrame(values.reshape(-1, 2), copy=False)");
}
```

`copy=False` is important when sharing is intended. Pandas 3 constructors otherwise copy NumPy
inputs by default. An explicit `copy=False` over an external NumSharp-backed NumPy array shares the
storage: NumSharp writes are visible to Pandas, and Pandas writes are visible to NumSharp. Pandas'
Copy-on-Write tracking cannot detach an owner outside Pandas, so this is a deliberate shared-memory
opt-in, not ordinary Pandas-to-Pandas CoW isolation.

## What shares and what copies

| Pandas source | View | Copy / Auto fallback |
|---|---|---|
| homogeneous numeric `DataFrame` | yes; often F-strided and read-only | detached same dtype |
| NumPy-backed numeric `Series` | yes; slices and negative strides preserved | detached |
| numeric `Index` / `RangeIndex` | yes; read-only | detached |
| nullable integer without missing values | yes when Pandas exposes its stable numeric buffer | detached |
| mixed numeric frame | no; common dtype requires materialization | yes, using Pandas' upcast |
| nullable integer with `pd.NA` | no | typically float64 with NaN |
| numeric categorical / sparse | no | categorical values / dense numeric values |
| complex64 | no NumSharp view | widens to NumSharp complex128 on copy |
| object, string, datetime, timedelta, timezone, `MultiIndex` values | unsupported | unsupported: no corresponding NumSharp dtype |

Axis labels, names, duplicate columns, and hierarchical row labels are Pandas metadata. The adapter
returns only the two-dimensional value matrix, exactly like `DataFrame.to_numpy()`.

## Codec policy

```csharp
NDArrayPythonInterop.RegisterCodec(new NumpyCodecOptions
{
    DecodeMode = NumpyCodecMode.Auto,
    DecodeArrayAdapters = true,
});
```

- `View` accepts only a proven stable projection.
- `Copy` always returns an owning NumSharp snapshot when the projected dtype is representable.
- `Auto` tries View and falls back to Copy.
- `DecodeArrayAdapters = false` excludes Pandas/Torch adapters without disabling native NumPy or
  buffer sources.

The adapter is registered automatically.
`PythonArrayAdapterRegistry.Register(PandasPythonArrayAdapter.Instance)` therefore returns `false`
because the name already exists.

## Edge behavior pinned by tests

- scalar, empty Series, zero-row frames, zero-column frames, empty RangeIndex, and 1×1 frames;
- C/F layouts, positive strides, reversed negative strides, and storage offsets;
- all 15 NumSharp source dtypes, including Char→uint16 and Decimal→float64 boundaries;
- NaN payloads, infinities, signed zero, unsigned maxima, and complex values;
- mixed blocks, nullable missing values, categorical, sparse, object and semantic time dtypes;
- duplicate labels and MultiIndex axes;
- Pandas subclasses and public extension-array base classes;
- derived NumSharp-view lifetime after every Pandas/Python wrapper is deleted;
- explicit caller-owned GIL mode, four-thread churn, disposed wrappers, failing overrides, and leak
  counters after every unsuccessful conversion;
- full NumSharp→Pandas→NumSharp pointer/shape/stride identity for positive, negative, offset,
  Fortran, and broadcast layouts;
- every nullable integer/unsigned/float/boolean extension dtype, both mask-free and with `pd.NA`;
- float16/float32, int64/uint64, real/complex, and object-producing column-promotion boundaries;
- big-endian integer/float/complex view rejection plus component-wise byte swapping on copy;
- imported-view resize refusal, owning-copy resize, NumSharp-export lifetime under Pandas ownership,
  malformed `to_numpy` returns, descending RangeIndex, and two-axis negative strides.

## Claims ledger

| # | Claim | Gate |
|---|---|---|
| 1 | validated runtime is Pandas 3.0.5 | [`Runtime_IsLatestStablePandas305`][gate] |
| 2 | built-in recognition covers frames, series, indexes, extension arrays and subclasses—not scalars | [`BuiltInAdapter_RecognizesFrameSeriesIndexExtensionAndSubclasses_NotScalars`][gate] |
| 3 | implicit NumSharp encoding plus explicit Pandas `copy=False` shares storage | [`NumSharpToPandas_ImplicitEncoderSharesInput_WhenCopyFalseIsExplicit`][gate] |
| 4 | numeric Series enters the existing read-only view bridge with exact pointer/layout | [`NumericSeries_UsesExistingReadonlyViewBridge_WithExactPointerAndLayout`][gate] |
| 5 | homogeneous frame preserves its F-strided matrix and pointer through implicit decode | [`HomogeneousDataFrame_ImplicitDecoderPreservesFortranStridesAndPointer`][gate] |
| 6 | numeric Index and nullable-without-missing extension storage are stable views | [`IndexAndNullableExtensionWithoutMissingValues_AreStableViews`][gate] |
| 7 | mixed numeric frame copies, detaches, and follows Pandas' common-dtype upcast | [`MixedNumericFrame_OrdinaryCopyAndImplicitAutoAreDetachedAndUpcastByPandas`][gate] |
| 8 | materializing numeric families reject View but succeed through Copy | [`MaterializingPandasObjects_RejectView_ButNumericCopiesPreservePandasValues`][gate-edge] |
| 9 | object/string/time/nested-index values fail without leases | [`ObjectStringDatetimeTimedeltaAndNestedIndex_AreDirectedUnsupportedFailures`][gate-edge] |
| 10 | complex64 copy-widens while View declines | [`Complex64ViewDeclines_AndCopyWidensToNumSharpComplex128`][gate-edge] |
| 11 | empty and degenerate ranks/shapes/dtypes are preserved | [`EmptyAndDegenerateShapes_PreserveRankShapeAndResolvedDtype`][gate-edge] |
| 12 | positive and negative Series strides preserve logical values | [`PositiveAndNegativeStridedSeries_PreserveLogicalOffsetsAndStrides`][gate-edge] |
| 13 | all 15 NumSharp source dtypes round-trip through a numeric Series | [`All15NumSharpDtypes_RoundTripThroughPandasNumericSeries`][gate-edge] |
| 14 | labels are metadata; duplicate/MultiIndex axes do not alter value order | [`AxisLabelsDuplicatesAndMultiIndex_AreMetadataOnly_ValuesKeepTwoDimensionalOrder`][gate-edge] |
| 15 | derived NumSharp views retain Pandas storage after Python wrappers die | [`DerivedNumSharpView_KeepsPandasStorageAliveAfterAllPythonWrappersDie`][gate-edge] |
| 16 | View/Copy/Auto and adapter-disable policies remain distinct | [`CodecModes_RespectPandasViewCopyAutoAndExplicitAdapterDisable`][gate-edge] |
| 17 | caller-owned GIL mode works for view and copy | [`ExplicitNoGilPolicy_WorksInsideOneOuterGil_ForPandasViewAndCopy`][gate-edge] |
| 18 | concurrent round-trips drain every export/import counter | [`ConcurrentPandasRoundTrips_AreThreadSafeAndDrainEveryLifetimeCounter`][gate-edge] |
| 19 | disposed and broken Pandas objects fail cleanly without leases | [`DisposedAndBrokenPandasObjects_FailCleanlyWithoutLeasingMemory`][gate-edge] |
| 20 | NumSharp layouts round-trip through Pandas with identical pointer, shape and strides | [`NumSharpLayouts_RoundTripThroughPandasWithoutLosingPointerShapeOrStrides`][gate-rare] |
| 21 | every mask-free nullable extension dtype exposes its resolved stable typed buffer | [`NullableExtensionDtypesWithoutMissingValues_ExposeEveryStableTypedBuffer`][gate-rare] |
| 22 | nullable extension masks reject View and resolve Copy dtype/NaN precisely | [`NullableExtensionDtypesWithMissingValues_RejectViewAndResolveCopyDtypePrecisely`][gate-rare] |
| 23 | mixed-column promotion follows Pandas, including uint64→float64 precision boundaries | [`MixedColumnPromotion_FollowsPandasCommonDtypeIncludingPrecisionBoundaries`][gate-rare] |
| 24 | big-endian integer, float and complex values reject View and byte-swap correctly on Copy | [`BigEndianNumericSeries_RejectView_AndCopyByteSwapsEveryScalarComponent`][gate-rare] |
| 25 | imported views cannot resize/detach while owning copies can | [`ImportedPandasViewCannotResizeOrDetach_WhileOwningCopyCan`][gate-rare] |
| 26 | Pandas retains a NumSharp export after the original NDArray is disposed | [`PandasObject_KeepsNumSharpExportAliveAfterOriginalArrayIsDisposed`][gate-rare] |
| 27 | Pandas mutations update a leased view while an owning copy stays detached | [`PandasMutationUpdatesLeasedView_ButOwningCopyRemainsDetached`][gate-rare] |
| 28 | malformed `to_numpy` return types fail without acquiring import leases | [`MalformedToNumpyReturnTypes_FailWithoutAcquiringImportLeases`][gate-rare] |
| 29 | descending RangeIndex and two-axis negative-stride frames preserve logical order | [`DescendingRangeIndexAndTwoAxisReorderedFrame_PreserveUnusualLogicalOrder`][gate-rare] |

[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/PandasInteropTests.cs
[gate-edge]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/PandasInteropEdgeCaseTests.cs
[gate-rare]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/PandasInteropRareScenarioTests.cs
