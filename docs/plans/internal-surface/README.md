# NumSharp.dll internal surface map

Every internal class and member of `NumSharp.dll` (`src/NumSharp.Core`), and which of them stop a consumer from
extending or inheriting NumSharp. Measured on `strings` at `b10afc49` (2026-11-07), the line 0.80.0 ships from.

| File | Holds |
|------|-------|
| [tables.md](tables.md) | The generated summary: totals, areas, findings by category, virtuality, friend usage, exact type tests, and every internal type by area with its summary and source link |
| [internal-types.tsv](internal-types.tsv) | One row per internal type (493): area, kind, file:line, member counts, the friend packages that name it, doc summary |
| [internal-members.tsv](internal-members.tsv) | One row per internal member (6,091): declared access, modifiers, C# signature, file:line, friend use, doc summary |
| [extensibility-findings.tsv](extensibility-findings.tsv) | One row per way a visible type is closed to a consumer (470) |
| [virtuality.tsv](virtuality.tsv) | Per visible non-sealed class: visible constructors, how much of its instance surface can be overridden, which members |
| [friend-internals-usage.tsv](friend-internals-usage.tsv) | Every internal type / member the six `InternalsVisibleTo` packages name, with sites and callers |
| [exact-type-checks.tsv](exact-type-checks.tsv) | The `GetType()` equality tests in Core's source |
| [probe/](probe/) | A consumer-side `NDArray` subclass run through 12 operations, and its NumPy 2.4.2 twin |

## How it is measured

[`tools/internal-surface-map/SurfaceMap.cs`](../../../tools/internal-surface-map/SurfaceMap.cs) reads the built
assembly with Mono.Cecil instead of grepping source, so every accessibility answer is the compiler's:

- **Reach** is effective: a public member of an internal type is internal, a `private protected` member is internal (only
  a subclass inside NumSharp.dll sees it), a nested type takes the narrowest access of its chain. Lambdas, iterators,
  backing fields, file-local types and records' synthesized members are excluded.
- **Locations** come from the portable PDB (moved from the first sequence point to the signature line); fields, constants
  and abstract members, which have no IL body, are found by a keyword-checked search of every file of their partial type.
  Every one of the 6,091 lines was checked to declare its member; the 34 implicit parameterless constructors C# supplies
  have no declaration of their own and link their type's.
- **Summaries** are the first sentence of each member's XML `<summary>`.
- **Friend usage** walks the IL of every `InternalsVisibleTo` friend that builds from `src/` (OpenBLAS, pythonnet,
  OnnxRuntime, MLNet, System.Numerics.Tensors, ParquetNet): calls, field accesses, signatures, locals, base types,
  overrides. Test projects, the benchmark and `TensorFlowNET.UnitTest` are friends too, but are not scanned.

Regenerate after building Core and the packages for net10.0:

```bash
dotnet build src/NumSharp.Core -f net10.0
for p in src/NumSharp.Interop.*/; do dotnet build "$p" -f net10.0; done
dotnet run tools/internal-surface-map/SurfaceMap.cs -- --repo . --out docs/plans/internal-surface --label "<branch> <commit>"
```

## The numbers

| | Public | Internal | Private (nested) | Total |
|---|---:|---:|---:|---:|
| Types (no protected nested type exists) | 524 | 493 | 282 | 1,299 |

| Internal members | Count |
|---|---:|
| Declared `internal` / `private protected` on one of 170 visible types | 1,908 |
| Non-private members of internal types | 4,183 |
| **Total** | **6,091** |

Where they are (internal types / their members / internal members declared on the area's visible types):

| Area | Types | Members | On visible types |
|---|---:|---:|---:|
| Backends/Strings | 68 | 811 | 0 |
| Utilities/Maths | 1 | 726 | 0 |
| Backends/Kernels | 109 | 542 | 494 |
| Backends/Default | 73 | 515 | 173 |
| Backends/Iterators | 48 | 302 | 264 |
| RandomSampling | 22 | 246 | 143 |
| Polynomial/Package | 26 | 212 | 47 |
| IO/TextReading | 24 | 130 | 0 |
| Backends/Subclass | 18 | 112 | 0 |
| Backends/Printing | 22 | 104 | 4 |

The one type in `Utilities/Maths` is
[`Operator`](../../../src/NumSharp.Core/Utilities/Maths/Operator.cs#L11): 726 members and no caller anywhere in `src/` or
`test/`, so it is dead code.

The public types carrying the most internal members are `DirectILKernelGenerator` (395 internal beside 110 visible),
`NDArray` (195 / 408), `DefaultEngine` (152 / 292), `UnmanagedStorage` (103 / 201) and `Shape` (68 / 104).

## Extension scenarios

| Scenario | Verdict | What decides it |
|---|---|---|
| [Replace or extend the compute engine](#engines-open) | **Open** | Engines are 92-100% overridable; `TensorEngine.Default` is settable |
| [Plug a BLAS / LAPACK / FFT provider](#engines-open) | **Open** | `IBlasBackend`, `LinalgEngine`, `FftEngine` are public and fully virtual |
| [Subclass `NDArray`](#ndarray-subclass-compiles-but-every-operation-drops-the-class) | **Compiles, class lost** | 11 of 11 operations return plain `NDArray` (NumPy: 0 of 11); protocol hooks internal; closed-world dispatch |
| [Subclass `NDMatrix` / `NDCharArray` / `NDMaskedArray`](#ndmatrix-ndchararray-ndmaskedarray-closed) | **Closed** | Every constructor is internal |
| [Specialise storage](#storage-wrap-yes-specialise-no) | **Wrap yes, specialise no** | `UnmanagedStorage`: 28 public or protected constructors, 0 of 159 members overridable |
| [Write a zero-copy bridge package](#a-bridge-package-what-our-own-packages-need) | **Partly** | Our six packages use 32 internals; foreign-memory booking, lifetime-safe aliases and StringDType cells have no public route |
| [Add a dtype](#user-dtypes-registration-public-storage-closed) | **Closed** | Registration is public, but `setitem` / `getitem` / `copyswapn` / `fill` / `nonzero` are `internal virtual` |
| [Write a bit generator](#random-closed) | **Closed** | `BitGenerator`: `private protected` constructor, `internal abstract` primitives |
| [Write a polynomial basis](#polynomial-series-closed) | **Closed** | `NDPolyBase`: `private protected` constructor; the six series classes are sealed |
| [Add an expression node](#expression-nodes-closed-call-out-open) | **Closed by design** | `NDExpr` has 7 internal abstract members; `NDExpr.Call(Delegate)` is the open route |

### Engines: open

[`TensorEngine`](../../../src/NumSharp.Core/Backends/TensorEngine.cs) has a protected constructor and 258 of its 267
visible instance members are abstract or virtual; `DefaultEngine` 254 of 277; `LinalgEngine`, `DefaultLinalgEngine`,
`FftEngine` and `DefaultFftEngine` all of theirs. [`TensorEngine.Default`](../../../src/NumSharp.Core/Backends/TensorEngine.cs#L50)
has a public setter, so every array created after an assignment binds a consumer's engine, and `NDArray.TensorEngine` moves
a single array. Nothing a consumer must implement is hidden: no engine-side interface or abstract class has an internal
abstract member.

The four internal virtual members of `TensorEngine` are all optional:
[`Evaluate(NDExprProgram, ...)`](../../../src/NumSharp.Core/Backends/TensorEngine.cs#L269) (a `CompiledExpression` runs
only on an engine derived from `DefaultEngine`; the base throws `NotSupportedException`), and `DotFunction1D`,
`VdotFunction1D`, `VdotReadsFirstInPlace` (`DefaultEngine`'s 1-D dot paths; a consumer overrides the public `Dot` /
`Vdot` instead).

### NDArray subclass: compiles, but every operation drops the class

`NDArray` has 14 public or protected constructors and a `protected internal Storage`, so a consumer can derive from it. The
[probe](probe/subclass_probe.cs) does, with `__array_priority__ = 20`, and checks the class of each result:

| Operation | NumSharp | NumPy 2.4.2 |
|---|---|---|
| `np.add(x, x)`, `x + 1`, `-x`, `np.sqrt(x)` | `NDArray` | `MyArray` |
| `x["0"]`, `x[":, 1:"]`, `x.T`, `x.reshape(3, 2)` | `NDArray` | `MyArray` |
| `np.sum(x, 0)`, `np.concatenate([x, x])`, `x.copy()` | `NDArray` | `MyArray` |
| `np.asanyarray(x)` | the same object | the same object |

Three things make it so:

1. **The protocol hooks are internal.**
   [`__array_wrap__`](../../../src/NumSharp.Core/Backends/NDArray.SubclassHooks.cs#L41), both
   [`__array_finalize__`](../../../src/NumSharp.Core/Backends/NDArray.SubclassHooks.cs#L55) overloads and
   [`_collapse`](../../../src/NumSharp.Core/Backends/NDArray.SubclassHooks.cs#L77) are `internal virtual`, as are
   [`NanMax` / `NanMin`](../../../src/NumSharp.Core/Backends/NDArray.NanFunctions.cs#L94) and
   [`ReprClassName`](../../../src/NumSharp.Core/Casting/NDArray.ToString.cs#L51). Only `__array_priority__` is public,
   and nothing reads it for a class the library does not know.
2. **The runtime dispatch is closed-world.**
   [`NDSubclass.KindOf`](../../../src/NumSharp.Core/Backends/Subclass/NDSubclass.cs#L82) tests `GetType()` exactly against
   `NDMatrix` and `NDCharArray`; any other class is a plain array.
3. **The woven dispatch is fixed at NumSharp's own build.**
   [`SubclassWeaver`](../../../tools/NumSharp.Build/SubclassWeaver.cs#L330) generates arms for the `NDArray` subclasses it
   finds in NumSharp.dll (an `internal const string __name__`); a consumer's assembly is not in that set.

Of the 271 visible instance members, 7 can be overridden: `__array_priority__`, the indexer, `ToString`, `Equals`,
`GetHashCode`, `Clone` and the finalizer. `NDArray<T>` is in the same position (13 public or protected constructors, 2 of 17
members overridable).

### NDMatrix, NDCharArray, NDMaskedArray: closed

Their only constructors are internal:
[`NDMatrix(UnmanagedStorage)`](../../../src/NumSharp.Core/Matrixlib/NDMatrix.cs#L180),
[`NDCharArray(UnmanagedStorage)`](../../../src/NumSharp.Core/Char/NDCharArray.cs#L165),
[`NDMaskedArray(NDArray, NDArray, NDArray)`](../../../src/NumSharp.Core/Ma/NDMaskedArray.cs#L124). `numpy.matrix`,
`numpy.char.chararray` and `numpy.ma.MaskedArray` are ordinary Python classes a user can subclass.

### Storage: wrap yes, specialise no

`UnmanagedStorage` has 28 public or protected constructors and none of its 159 visible instance members is virtual, so
a subclass changes nothing; an engine can still hand out its own storage from the virtual `GetStorage(DType)`. Its state
is internal:
`OwnsData`, `IsDataAligned`, `ItemBytes`, `WriteProtected`, `ExternalBase`, `_baseStorage`, `StringSlice`,
`VStringSlice`. Foreign memory can be wrapped publicly: `ArraySlice.Wrap<T>(void*, long)` (counts nothing) and
[`UnmanagedMemoryBlock<T>(T*, long, Action dispose)`](../../../src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock%601.cs#L103)
with `ArraySlice.FromMemoryBlock`.

### A bridge package: what our own packages need

Our six `InternalsVisibleTo` packages name 32 distinct internal types and members (47 package-by-member pairs) at 410 IL
sites. A third-party package of the same kind can reach none of them:

| Internal | Used by (sites) | Public route |
|---|---|---|
| `Shape.dimensions` / `strides` fields | OpenBLAS (127 / 74), Tensors, pythonnet | [`Shape.Dimensions` / `Strides`](../../../src/NumSharp.Core/View/Shape.cs#L515): a new array on every read |
| `Shape.offset` / `size` / `bufferSize` fields | OpenBLAS (59 / 6), Tensors, pythonnet | `Shape.Offset` / `Size` / `BufferSize`: the same values |
| [`Shape(dims, strides, offset, bufferSize)`](../../../src/NumSharp.Core/View/Shape.cs#L622), [`Shape.Adopt(dims)`](../../../src/NumSharp.Core/View/Shape.cs#L870) | ORT, Tensors, pythonnet / OpenBLAS (13) | `Shape(dims, strides)`: copies, and takes no offset or buffer size |
| `NDArray.GetTypeCode` | OpenBLAS (27) | `NDArray.typecode`: the same value |
| `np._FindCommonArrayType` | OpenBLAS (6) | `np.result_type` / `np.promote_types` |
| [`SizeBucketedBufferPool.BookForeign` / `ReturnForeign`](../../../src/NumSharp.Core/Backends/Unmanaged/Pooling/SizeBucketedBufferPool.cs#L696) | ORT, MLNet, Tensors, pythonnet | **None**: a third party's foreign buffers never pace the collection that finalizes their dropped holders |
| [`ArraySlice.WrapShared`](../../../src/NumSharp.Core/Backends/Unmanaged/ArraySlice.cs#L552) | ORT (14) | **None**: `Wrap` counts nothing, so an alias cannot keep its owner's buffer alive |
| `VStringAllocator`, `VStringParameters`, `UnmanagedStorage.VStringSlice`, `DType.VString` | pythonnet (53) | **None**: StringDType cells cannot be read or written zero-copy from outside |
| `NDArray.ForOverwrite` | pythonnet (1) | `np.empty` (on the default engine) |
| `ReductionInitial.PackStore` | ParquetNet (1, unreleased) | (a reduction helper) |

Where a public route exists it costs a third party at most an allocation per read (`Dimensions` / `Strides`) or a copy
(`Shape(dims, strides)`); the three **None** rows are capabilities only a friend has.

### User dtypes: registration public, storage closed

The registration path is public: a protected [`DTypeMeta`](../../../src/NumSharp.Core/DTypes/DTypeMeta.cs#L134)
constructor, [`DTypeRegistry.Register`](../../../src/NumSharp.Core/DTypes/DTypeRegistry.cs#L654) ("Registers a new DType
class (a user dtype)") and `DTypeRegistry.AddCastingImpl` over the public abstract `CastingImpl`. The element slots are
`internal virtual`: [`setitem`](../../../src/NumSharp.Core/DTypes/DTypeMeta.cs#L394) and
[`getitem`](../../../src/NumSharp.Core/DTypes/DTypeMeta.cs#L450) throw "no slot" in the base, and `SetFreshItems`,
`copyswapn`, `fill`, `nonzero`, `HasFill`, `HasScanFunc`, `HasFromStr`, `ByteOrderNotApplicable`, `OwnsTypeCode` and
`OwnsScalarType` cannot be overridden either. A registered user dtype can neither store nor read an element.

Opening the slots would not finish the job: storage is keyed by `NPTypeCode`, a closed enum that every kernel switches
on, and ufunc loops have no per-dtype registry (`ArrayMethod` serves casts only).

### Random: closed

[`BitGenerator`](../../../src/NumSharp.Core/RandomSampling/BitGenerator.cs#L32) has a `private protected` constructor,
`internal abstract` [`NextUInt64` / `NextUInt32`](../../../src/NumSharp.Core/RandomSampling/BitGenerator.cs#L242) and
`Name`, `private protected abstract` `CreateFromSeed` / `GetStateCore` / `SetStateCore`, and 9 `internal virtual` draws
and bulk fills. `Generator`, `MT19937`, `PCG64`, `PCG64DXSM`, `Philox`, `SFC64` and `SeedSequence` are sealed. NumPy's
`Generator` takes a user-provided bit generator that exports the `bitgen_t` capsule (NumPy's "Extending: New
BitGenerators"). `NumPyRandom` can be subclassed (4 public or protected constructors), but only `ToString` of its 239
members is virtual.

### Polynomial series: closed

[`NDPolyBase`](../../../src/NumSharp.Core/Polynomial/Package/NDPolyBase.cs#L138) has 23 `private protected` members, its
constructor among them, and `NDPolynomial`, `NDChebyshev`, `NDLegendre`, `NDLaguerre`, `NDHermite`, `NDHermiteE` and
`poly1d` are sealed. In NumPy the six series classes are ordinary subclasses of the abstract `ABCPolyBase`
(`numpy/polynomial/_polybase.py`), which any Python class can derive from too.

### Expression nodes: closed, call-out open

`NDExpr` has 7 `internal abstract` members (`BindArrays`, `InferType`, `HashStructure`, `StructureEquals`, `LowerText`,
`CanEmitVectorV2`, `ContainsReduce`) and its node classes are sealed, so a consumer writes no node of its own;
[`NDExpr.Call(Delegate, ...)`](../../../src/NumSharp.Core/Backends/Iterators/NDExpr.cs#L849) and the `MethodInfo` forms
are the public way to put user code in an expression.

### Exact type tests

58 `GetType()` equality tests are in Core's code; 5 can treat a subclass differently from `is`:
[`NDSubclass.KindOf`](../../../src/NumSharp.Core/Backends/Subclass/NDSubclass.cs#L82)'s two (the closed-world dispatch
above), `Slice.Equals` (`obj.GetType() != typeof(Slice)`: a `Slice` subclass never equals a `Slice`), and one each in
`Shape.Equals` (a struct: nothing derives from it) and `NDExpr`'s delegate-identity comparer (two delegates' classes). The
other 53 dispatch on CLR element types, where exact and `is` agree.

## What the internals are

The TSVs classify nothing; read by kind, the 6,091 members fall into five groups:

- **Machinery**: the IL kernel generators and their delegates (`Backends/Kernels`, 109 types plus 494 members on public
  kernel classes), NDIter and its buffers (`Backends/Iterators`), printing, text codecs and the string engine
  (`Backends/Strings`, `IO/TextReading`), polynomial and einsum caches. Nothing a consumer extends goes through them.
- **Implementations behind public functions**: 104 of `NDArray`'s internal methods (77 functions with their overloads)
  are the array-object functions that public `np.*` facades call (`concatenate`, `where`, `zeros_like`, `meshgrid`,
  `tril`, ...), as are `UnmanagedStorage`'s `Allocate` / `Copyto` / `FromSequence` / `MapIter*` and `Shape`'s layout
  math. The public route exists.
- **NumPy-semantics bookkeeping**: NEP 50 operand marks on `NDArray` (`WasPythonLiteral`, `Nep50Operand`,
  `WasScalarArgument`, ...), ownership tracking (`TrackingScope`, `TrackingIndex`), the subclass dispatcher
  (`Backends/Subclass`, 18 types).
- **Extension seams kept internal**: the list below.
- **Dead code**: `Operator` (726 members).

## Seams to decide on

Each row opens one scenario above. Nothing here is implemented; the order is by how much NumPy-shaped extension each
unlocks.

| # | Seam | Unlocks | Today |
|---|---|---|---|
| 1 | The subclass protocol: `__array_wrap__`, `__array_finalize__`, `_collapse` overridable outside, and a dispatch that routes any `NDArray` subclass (not only the two the weaver saw) | Consumer subclasses that keep their class, as in NumPy | `internal virtual`; `KindOf` exact-tests two classes |
| 2 | Protected constructors on `NDMatrix`, `NDCharArray`, `NDMaskedArray` (after 1) | Subclasses of NumPy's own subclasses | Internal constructors |
| 3 | A public bridge surface: foreign-byte booking, `WrapShared`, a `Shape` built from dims / strides / offset / buffer size, allocation-free dims / strides reads | Third-party zero-copy bridges on par with ours | Friend-only |
| 4 | `BitGenerator`'s primitives and seeding helpers | User bit generators | `internal abstract` / `private protected` |
| 5 | `NDPolyBase`'s constructor and helpers, unsealed series classes | User polynomial bases | `private protected`, sealed |
| 6 | `DTypeMeta`'s element slots | User dtypes that store and read values (computing needs a loop registry as well) | `internal virtual` |
| 7 | `TensorEngine.Evaluate(NDExprProgram, ...)` | Compiled expressions on engines not derived from `DefaultEngine` | `internal virtual` |
