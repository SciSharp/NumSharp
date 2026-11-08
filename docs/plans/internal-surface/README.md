# NumSharp.dll internal surface map

Every internal class and member of `NumSharp.dll` (`src/NumSharp.Core`), and which of them stop a consumer from
extending or inheriting NumSharp. The tables describe `strings` at `34e48e16` with the access changes of `687fb94f`
(2026-11-07), which opened the seams the owner decided on (see [Decisions](#decisions-2026-11-07)). The map from before
those changes is the first commit of this folder, `62570760`.

| File | Holds |
|------|-------|
| [tables.md](tables.md) | The generated summary: totals, areas, findings by category, virtuality, friend usage, exact type tests, and every internal type by area with its summary and source link |
| [internal-types.tsv](internal-types.tsv) | One row per internal type (489): area, kind, file:line, member counts, the friend packages that name it, doc summary |
| [internal-members.tsv](internal-members.tsv) | One row per internal member (5,939): declared access, modifiers, C# signature, file:line, friend use, doc summary |
| [extensibility-findings.tsv](extensibility-findings.tsv) | One row per way a visible type is closed to a consumer (405) |
| [virtuality.tsv](virtuality.tsv) | Per visible non-sealed class: visible constructors, how much of its instance surface can be overridden, which members |
| [friend-internals-usage.tsv](friend-internals-usage.tsv) | Every internal type / member the six `InternalsVisibleTo` packages name, with sites and callers |
| [exact-type-checks.tsv](exact-type-checks.tsv) | The `GetType()` equality tests in Core's source |
| [probe/](probe/) | A consumer-side `NDArray` subclass run through 12 operations, and its NumPy 2.4.2 twin |

## How it is measured

[`tools/internal-surface-map/SurfaceMap.cs`](../../../tools/internal-surface-map/SurfaceMap.cs) reads the built
assembly with Mono.Cecil instead of grepping source, so every accessibility answer is the compiler's:

- **Reach** is effective: a public member of an internal type is internal, a `private protected` member is internal (only
  a subclass inside NumSharp.dll sees it), a `protected internal` member is open to every subclass, and a nested type
  takes the narrowest access of its chain. Lambdas, iterators, backing fields, file-local types and records' synthesized
  members are excluded.
- **Locations** come from the portable PDB (moved from the first sequence point to the signature line); fields, constants
  and abstract members, which have no IL body, are found by a keyword-checked search of every file of their partial type.
  Every one of the 5,939 lines was checked to declare its member; the implicit parameterless constructors C# supplies have
  no declaration of their own and link their type's.
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

## Decisions (2026-11-07)

The owner decided the seams the first map listed, group by group; `687fb94f` applies them.

| Seam | The owner's words | Now |
|---|---|---|
| Subclass protocol | "Make all protected internal" | `NDArray.__array_wrap__`, both `__array_finalize__` overloads, `_collapse`, `NanMax`, `NanMin`, `ReprClassName` and the two `__name__` constants are `protected internal` (with their `NDMatrix` / `NDCharArray` overrides); `NewSubtypeZeroDim` is public, because a `protected internal` member cannot take an internal type |
| Subclass constructors | "make subclasses all public and their constructors"; `NDMaskedConstant`: "Public" | `NDMatrix(UnmanagedStorage)`, `NDCharArray(UnmanagedStorage)`, `NDMaskedArray(data, mask, fill_value)`, `NDMaskedConstant()`, `NDPolyFitResult(...)`, `NDArray<T>(Shape, bool, TensorEngine)` and `NDArray(DType, Shape, bool, TensorEngine)` are public; the last binds `TensorEngine.Default` for a null engine |
| Bridge surface | "Public" | `SizeBucketedBufferPool.BookForeign` / `ReturnForeign`, both `ArraySlice.WrapShared` forms, `VStringAllocator` with NumPy's documented `NpyString` API (`acquire_allocator`, `release_allocator`, `load`, `pack`, `pack_null`) and its 10 NumPy layout constants, `VStringArraySlice.Allocator`, `UnmanagedStorage.VStringSlice` / `StringSlice`, `Shape(dims, strides, offset, bufferSize)` |
| Bit generators | "Public and the core ones also virtual and classes have clean proper constructors"; follow-up: "Unseal the classes" | `BitGenerator`'s primitives are public (`abstract` / `virtual` as they were), its seeding and fill helpers public, its constructor `protected`; `Generator`, `MT19937`, `PCG64`, `PCG64DXSM`, `Philox` and `SFC64` are unsealed |
| Shape | "Shape - all public" | The raw fields `dimensions`, `strides`, `offset`, `size`, `bufferSize` are public (read-only by contract: the arrays are shared) |
| Engine hooks | "all public" | `TensorEngine.Evaluate(NDExprProgram, ...)`, `DotFunction1D`, `VdotFunction1D`, `VdotReadsFirstInPlace` are `public virtual`, as are `DefaultEngine`'s overrides |
| Iterators | "public" | The constructors of `np.NDEnumerate`, `np.NDEnumerate<T>`, `np.NDIndex` and `np.NDIterator` |
| Storage state | "public"; follow-up: "Public read, internal write" | `OwnsData`, `IsDataAligned`, `ItemBytes` are public; `WriteProtected` / `ExternalBase` are `public { get; internal set; }` |

Where an instruction met a technical limit, the call made:

- `NDExprProgram` and `NDEvaluateOptions` are public types with internal members, an opaque handle and bundle. Three of
  the program's fields have internal plan types, and the options' constructor would let a caller pass keywords the engine
  never validated. An engine overrides `Evaluate(program, ...)` to observe or reroute and calls `base`; an engine not
  derived from `DefaultEngine` cannot run a program itself.
- `UnmanagedStorage._baseStorage` stays internal: `BaseStorage` is its public read form, and a public setter is the
  use-after-free its own docs rule out.
- `Shape.Adopt` is gone: the owner's `journey4` commit `278e9862`, re-derived on `release-0.80.0`, replaces it with
  public `Shape(dims, order, copy)` and `Shape(dims, strides, copy)` constructors, NumPy's `copy` keyword being the
  public no-copy route.
- `DTypeMeta`'s element slots were not decided and stay internal.

## The numbers

| | Public | Internal | Private (nested) | Total |
|---|---:|---:|---:|---:|
| Types (no protected nested type exists) | 528 | 489 | 282 | 1,299 |

| Internal members | Count |
|---|---:|
| Declared `internal` / `private protected` on one of 164 visible types | 1,851 |
| Non-private members of internal types | 4,088 |
| **Total** | **5,939** |

Where they are (internal types / their members / internal members declared on the area's visible types):

| Area | Types | Members | On visible types |
|---|---:|---:|---:|
| Backends/Strings | 67 | 758 | 38 |
| Utilities/Maths | 1 | 726 | 0 |
| Backends/Kernels | 109 | 542 | 494 |
| Backends/Default | 73 | 515 | 169 |
| Backends/Iterators | 46 | 260 | 306 |
| RandomSampling | 22 | 246 | 64 |
| Polynomial/Package | 26 | 212 | 46 |
| IO/TextReading | 24 | 130 | 0 |
| Backends/Subclass | 17 | 112 | 0 |

The one type in `Utilities/Maths` is
[`Operator`](../../../src/NumSharp.Core/Utilities/Maths/Operator.cs#L11): 726 members and no caller anywhere in `src/` or
`test/`, so it is dead code.

The public types carrying the most internal members are `DirectILKernelGenerator` (395 internal beside 110 visible),
`NDArray` (187 / 416), `DefaultEngine` (148 / 296), `UnmanagedStorage` (96 / 208) and `Shape` (62 / 110).

## Extension scenarios

| Scenario | Verdict | What decides it |
|---|---|---|
| [Replace or extend the compute engine](#engines-open) | **Open** | `TensorEngine` 262 of 271 members overridable; `TensorEngine.Default` is settable |
| [Plug a BLAS / LAPACK / FFT provider](#engines-open) | **Open** | `IBlasBackend`, `LinalgEngine`, `FftEngine` are public and fully virtual |
| [Subclass `NDArray`](#ndarray-subclass-hooks-open-dispatch-closed) | **Hooks open, class still lost** | The protocol hooks can be overridden, but the dispatch routes two classes: 11 of 11 operations return plain `NDArray` (NumPy: 0 of 11) |
| [Subclass `NDMatrix` / `NDCharArray` / `NDMaskedArray`](#ndmatrix-ndchararray-ndmaskedarray-constructible) | **Constructible** | Public constructors; a subclass of `NDMatrix` / `NDCharArray` is plain to np.* functions, for the same dispatch reason |
| [Specialise storage](#storage-wrap-and-read-yes-specialise-no) | **Wrap and read yes, specialise no** | `UnmanagedStorage`: state readable, 0 of 166 members overridable |
| [Write a zero-copy bridge package](#a-bridge-package) | **Open** | The ONNX Runtime, ML.NET and Tensors bridges now use no internal; the rest have public routes or are allocator details NumPy keeps private too |
| [Add a dtype](#user-dtypes-registration-public-storage-closed) | **Closed** | Registration is public, but `setitem` / `getitem` / `copyswapn` / `fill` / `nonzero` are `internal virtual` |
| [Write a bit generator](#random-open) | **Open** | `BitGenerator`'s contract is public; `Generator` and the built-in generators are unsealed |
| [Write a polynomial basis](#polynomial-series-closed) | **Closed** | `NDPolyBase`'s constructor is `private protected` and the basis math is picked by an internal enum |
| [Add an expression node](#expression-nodes-closed-call-out-open) | **Closed by design** | `NDExpr` has 7 internal abstract members; `NDExpr.Call(Delegate)` is the open route |

### Engines: open

[`TensorEngine`](../../../src/NumSharp.Core/Backends/TensorEngine.cs) has a protected constructor and 262 of its 271
visible instance members are abstract or virtual; `LinalgEngine`, `DefaultLinalgEngine`, `FftEngine` and
`DefaultFftEngine` all of theirs. [`TensorEngine.Default`](../../../src/NumSharp.Core/Backends/TensorEngine.cs#L50) has a
public setter, so every array created after an assignment binds a consumer's engine, and `NDArray.TensorEngine` moves a
single array. No engine-side interface or abstract class has an internal abstract member, and none has an internal
virtual member left: `Evaluate(NDExprProgram, ...)` and the three 1-D dot hooks are public. The compiled program is
opaque outside NumSharp.dll, so an engine that does not derive from `DefaultEngine` can forward it but not run it.

### NDArray subclass: hooks open, dispatch closed

`NDArray` has 15 public or protected constructors and a `protected internal Storage`, and its subclass protocol is now
open to every subclass: [`__array_wrap__`](../../../src/NumSharp.Core/Backends/NDArray.SubclassHooks.cs#L50), both
`__array_finalize__` overloads, `_collapse`, `NanMax` / `NanMin` and `ReprClassName` are `protected internal` (overridden
from another assembly as `protected override`), beside the public `__array_priority__`. 14 of its 278 visible instance
members can be overridden.

The [probe](probe/subclass_probe.cs) still measures 11 of 11:

| Operation | NumSharp | NumPy 2.4.2 |
|---|---|---|
| `np.add(x, x)`, `x + 1`, `-x`, `np.sqrt(x)` | `NDArray` | `MyArray` |
| `x["0"]`, `x[":, 1:"]`, `x.T`, `x.reshape(3, 2)` | `NDArray` | `MyArray` |
| `np.sum(x, 0)`, `np.concatenate([x, x])`, `x.copy()` | `NDArray` | `MyArray` |
| `np.asanyarray(x)` | the same object | the same object |

The hooks are reachable, but nothing calls them for a consumer's class:

1. **The runtime dispatch is closed-world.**
   [`NDSubclass.KindOf`](../../../src/NumSharp.Core/Backends/Subclass/NDSubclass.cs#L82) tests `GetType()` exactly against
   `NDMatrix` and `NDCharArray`; any other class, a subclass of those two included, is a plain array.
2. **The woven dispatch is fixed at NumSharp's own build.**
   [`SubclassWeaver`](../../../tools/NumSharp.Build/SubclassWeaver.cs#L330) generates arms for the `NDArray` subclasses it
   finds in NumSharp.dll (a `__name__` constant); a consumer's assembly is not in that set.

The exception is `NanMax` / `NanMin`: `np.nanmax` / `np.nanmin` reach them by plain virtual dispatch, so a consumer's
override answers there.

### NDMatrix, NDCharArray, NDMaskedArray: constructible

Their constructors are public:
[`NDMatrix(UnmanagedStorage)`](../../../src/NumSharp.Core/Matrixlib/NDMatrix.cs#L185),
[`NDCharArray(UnmanagedStorage)`](../../../src/NumSharp.Core/Char/NDCharArray.cs#L171),
[`NDMaskedArray(NDArray, NDArray, NDArray)`](../../../src/NumSharp.Core/Ma/NDMaskedArray.cs#L130). The first two skip
NumPy's finalize rule (`np.asmatrix`, `np.char.array` apply it) and the third validates nothing but its data
(`np.ma.array` checks, broadcasts and converts); each documents that. A subclass of `NDMatrix` or `NDCharArray` keeps
the members it inherits, but np.* functions treat it as a plain array (the exact type tests above). `NDMaskedArray`
wraps an array instead of deriving from one, so a subclass of it is an ordinary masked array.

`NDMaskedConstant()` is public too. NumPy's `MaskedConstant()` returns the existing `np.ma.masked`, while a C#
constructor builds a new object, and the library recognises `np.ma.masked` by five type tests and nine reference
identity tests (var, std, average, the set operations). A second instance passes the first kind only.

### Storage: wrap and read yes, specialise no

`UnmanagedStorage` has 28 public or protected constructors and none of its 166 visible instance members is virtual, so
a subclass changes nothing; an engine can still hand out its own storage from the virtual `GetStorage(DType)`. Its state
is readable: `OwnsData`, `IsDataAligned`, `ItemBytes`, `StringSlice`, `VStringSlice`, `BaseStorage`, and
`WriteProtected` / `ExternalBase`, which only NumSharp sets (when it creates a storage over foreign memory). Foreign
memory can be wrapped publicly with `ArraySlice.Wrap<T>(void*, long)` (counts nothing),
[`UnmanagedMemoryBlock<T>(T*, long, Action dispose)`](../../../src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock%601.cs#L103)
or `ArraySlice.WrapShared` (an alias that keeps its owner's buffer alive).

### A bridge package

What our six `InternalsVisibleTo` packages needed is public now:
[`SizeBucketedBufferPool.BookForeign` / `ReturnForeign`](../../../src/NumSharp.Core/Backends/Unmanaged/Pooling/SizeBucketedBufferPool.cs#L701),
[`ArraySlice.WrapShared`](../../../src/NumSharp.Core/Backends/Unmanaged/ArraySlice.cs#L558), the
[`StringDType` allocator](../../../src/NumSharp.Core/Backends/Strings/VStringAllocator.cs#L91)'s `NpyString` API, and
`Shape`'s raw layout ([fields](../../../src/NumSharp.Core/View/Shape.cs#L71) and the
[view constructor](../../../src/NumSharp.Core/View/Shape.cs#L654)). The ONNX Runtime, ML.NET and System.Numerics.Tensors
bridges use no internal at all. The 16 internals still named (68 IL sites):

| Package | Internal | Sites | Public route |
|---|---|---:|---|
| OpenBLAS | `NDArray.GetTypeCode` | 27 | `NDArray.typecode`: the same value |
| OpenBLAS | `Shape.Adopt(dims)` | 13 | `Shape(dims, copy: false)`, which the 13 sites now call (`278e9862` on `release-0.80.0`) |
| OpenBLAS | `np._FindCommonArrayType` | 6 | `np.promote_types` / `np.result_type` |
| pythonnet | `VStringAllocator.isnull`, `size`, `newsize`, `TryReserve`, `ArenaBytesForDuplicate` | 9 | `load` (its return value is the null test, its `size` the length) and `pack`; the other three are the allocator's private arithmetic, private in NumPy too |
| pythonnet | `VStringParameters` (`coerce`, `HasNAObject`, `na_object`), `DType.VString` | 10 | `DType.coerce` / `DType.na_object` |
| pythonnet | `NDArray.ForOverwrite` | 1 | `np.empty` (on the default engine) |
| ParquetNet (unreleased) | `ReductionInitial` / `PackStore` | 2 | (a reduction helper) |

### User dtypes: registration public, storage closed

Not decided, unchanged. The registration path is public: a protected
[`DTypeMeta`](../../../src/NumSharp.Core/DTypes/DTypeMeta.cs#L134) constructor,
[`DTypeRegistry.Register`](../../../src/NumSharp.Core/DTypes/DTypeRegistry.cs#L654) ("Registers a new DType class (a user
dtype)") and `DTypeRegistry.AddCastingImpl` over the public abstract `CastingImpl`. The element slots are
`internal virtual`: [`setitem`](../../../src/NumSharp.Core/DTypes/DTypeMeta.cs#L394) and
[`getitem`](../../../src/NumSharp.Core/DTypes/DTypeMeta.cs#L450) throw "no slot" in the base, and `SetFreshItems`,
`copyswapn`, `fill`, `nonzero`, `HasFill`, `HasScanFunc`, `HasFromStr`, `ByteOrderNotApplicable`, `OwnsTypeCode` and
`OwnsScalarType` cannot be overridden either. A registered user dtype can neither store nor read an element. Opening the
slots would not finish the job: storage is keyed by `NPTypeCode`, a closed enum that every kernel switches on, and ufunc
loops have no per-dtype registry (`ArrayMethod` serves casts only).

### Random: open

[`BitGenerator`](../../../src/NumSharp.Core/RandomSampling/BitGenerator.cs) is a public contract: a subclass passes its
seed sequence to the `protected` constructor, implements `NextUInt64`, `NextUInt32`, `Name`, `CreateFromSeed`,
`GetStateCore` and `SetStateCore`, and may override `NextDouble`, `UnitToDouble`, `NextFloat`, `NextRaw` and the five
bulk fills for speed, with `SeedWords32` / `SeedWords64` (NumPy's `generate_state`) and the `FillDoubleFrom64` /
`FillUInt32From64` / `FillFloatFrom64` helpers to build them. `Generator`, `MT19937`, `PCG64`, `PCG64DXSM`, `Philox` and
`SFC64` are unsealed: a subclass of a built-in generator inherits its engine and overrides `CreateFromSeed` so that
`spawn` returns its own type. NumPy subclasses `Generator` and `PCG64` the same way, and takes user bit generators through
its `bitgen_t` capsule.

### Polynomial series: closed

[`NDPolyBase`](../../../src/NumSharp.Core/Polynomial/Package/NDPolyBase.cs#L138) has 23 `private protected` members, its
constructor among them, and `NDPolynomial`, `NDChebyshev`, `NDLegendre`, `NDLaguerre`, `NDHermite`, `NDHermiteE` and
`poly1d` are sealed. Opening the constructor would not help: the math of each basis is picked by the internal `PolyBasis`
enum inside `NDPolyClass`, whose constructor is internal too, and `NDPolyBase` has no abstract member a new basis could
implement. In NumPy the six series classes are ordinary subclasses of the abstract `ABCPolyBase`
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

The TSVs classify nothing; read by kind, the 5,939 members fall into five groups:

- **Machinery**: the IL kernel generators and their delegates (`Backends/Kernels`, 109 types plus 494 members on public
  kernel classes), NDIter and its buffers (`Backends/Iterators`), printing, text codecs and the string engine
  (`Backends/Strings`, `IO/TextReading`), polynomial and einsum caches. Nothing a consumer extends goes through them.
- **Implementations behind public functions**: 104 of `NDArray`'s internal methods (77 functions with their overloads)
  are the array-object functions that public `np.*` facades call (`concatenate`, `where`, `zeros_like`, `meshgrid`,
  `tril`, ...), as are `UnmanagedStorage`'s `Allocate` / `Copyto` / `FromSequence` / `MapIter*` and `Shape`'s layout
  math. The public route exists.
- **NumPy-semantics bookkeeping**: NEP 50 operand marks on `NDArray` (`WasPythonLiteral`, `Nep50Operand`,
  `WasScalarArgument`, ...), ownership tracking (`TrackingScope`, `TrackingIndex`), the subclass dispatcher
  (`Backends/Subclass`, 17 types).
- **Seams still closed**: the list below.
- **Dead code**: `Operator` (726 members).

## Still closed

| Seam | Why | What would open it |
|---|---|---|
| The subclass dispatch | `NDSubclass.KindOf` exact-tests two classes; the weaver's arms are built at NumSharp's own build | A dispatch that routes any `NDArray` subclass through its now-overridable hooks |
| `DTypeMeta`'s element slots | Not decided | `protected internal virtual` slots, plus a per-dtype loop registry for a user dtype to compute |
| `NDPolyBase` | The basis math is an internal enum, not a member a subclass implements | A design with virtual basis operations |
| `NDExpr` nodes | By design | (`NDExpr.Call` is the extension route) |
| `Shape.Adopt` | Replaced by public `Shape(dims, ..., copy)` constructors (`278e9862` on `journey4`, re-derived on `release-0.80.0`) | Done; `strings` still has `Shape.Adopt` |
