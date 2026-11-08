# NumPy ↔ NumSharp API coverage

Compared with NumPy **2.4.2** using NumSharp assembly **0.80.0.0**.

Expanded API availability (dashboard default): **41.6%** (2017 of 4844 APIs across 46 surfaces and 56 categories). **2823 missing**, **4 partial**.

The catalog includes **853 per-ufunc protocol methods** (each public `numpy.ufunc` method on every ufunc export of the compared namespaces, plus the methods numpy.ma's masked ufunc wrappers implement); **338** describe methods whose calls NumPy rejects for that ufunc. These count as exposed API contracts, not computable operations.

It also includes **1793 Python protocols** NumPy defines on its classes (operators, subscripting, conversions, iteration, `with`, pickling), each class listing the protocols it has; NumSharp provides **344**, through a member of the protocol's own name, the C# operator, indexer or cast of the same meaning, or a reviewed adapter. **20** are slots NumPy fills only to refuse every call (`del a[i]`, ordering comparisons on `ndarray.flags`).

Headline API availability: **95.2%** (533 of 560 default-scope APIs). Including partial mappings, **95.9%** are addressed.

| Surface | Available | Partial | Unsupported | Missing | Total | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| np.fft.* | 18 | 0 | 0 | 0 | 18 | 100.0% |
| np.linalg.* | 31 | 0 | 0 | 0 | 31 | 100.0% |
| ndarray.* | 66 | 1 | 0 | 3 | 70 | 94.3% |
| np.* | 357 | 3 | 0 | 20 | 380 | 93.9% |
| np.polynomial.* | 10 | 0 | 0 | 0 | 10 | 100.0% |
| np.random.* | 51 | 0 | 0 | 0 | 51 | 100.0% |

> Availability is based on the compiled public API: a NumPy API is available when NumSharp has a member of its name, or a reviewed alias names another member. It is not a blanket behavioral-parity claim; dtype, layout and edge-case parity require differential tests (the oracle section), and signature parity is measured separately (the signature section).

## NumPy-oracle parity evidence

**887,230** committed NumPy 2.4.2 oracle contracts (**101,439** error-message contracts) are replayed against NumSharp in CI; each must be bit-exact or a documented, registry-excused divergence. An API counts as **oracle-verified** when contracts resolve to it directly, or through an identity alias (the same NumPy object mapped to the same NumSharp member) or a reviewed pure-delegation alias.

Headline scope: **537** of 560 APIs oracle-verified (**95.9%**; 525 direct, 12 alias); **0** available APIs have no oracle contracts.

Expanded scope: **1615** of 4844 APIs oracle-verified (**33.3%**; 1591 direct, 24 alias); **406** available APIs have no oracle contracts.

| Source | Contracts | Error contracts | Host-pinned | Files | Keys | Replayed by |
|---|---:|---:|---:|---:|---:|---|
| Differential-fuzz corpus (`test/NumSharp.Tests.Oracle/Fuzz/corpus`) | 836,054 | 89,067 | 79,180 | 151 | 2350 | FuzzMatrix (FuzzCorpusTests*, IndexOracleTests) |
| .npy/.npz format oracle (`test/NumSharp.Tests/IO/corpus/npy_oracle.zip`) | 404 | 38 | 0 | 1 | 11 | NpyOracle (NpyOracleTests) |
| Array flags oracle (`test/NumSharp.Tests/Backends/corpus/flags_oracle.jsonl`) | 1,245 | 225 | 0 | 1 | 2 | FlagsOracleTests |
| Result-layout parity oracle (`test/NumSharp.Tests/Backends/corpus/layout_parity_oracle.jsonl`) | 305 | 0 | 0 | 1 | 35 | LayoutParityOracleTests |
| String oracle (numpy.strings calls) (`test/NumSharp.Tests.Oracle/Strings/corpus`) | 49,222 | 12,109 | 0 | 13 | 46 | StringOracleTests |

Host-pinned contracts are hard-gated only on the pinned host (win-amd64 CRT libm, or the content-hash-pinned OpenBLAS) and Inconclusive elsewhere.

Per surface (expanded scope; surfaces with nothing available or gated are omitted). *Verified* shares the coverage percentage's denominator: every API row of the surface.

| Surface | Total | Available | Oracle-verified | Direct | Alias | Unverified available | Verified |
|---|---:|---:|---:|---:|---:|---:|---:|
| broadcast | 10 | 9 | 8 | 8 | 0 | 1 | 80.0% |
| char | 118 | 52 | 0 | 0 | 0 | 52 | 0.0% |
| char.chararray | 192 | 127 | 0 | 0 | 0 | 127 | 0.0% |
| dtype | 40 | 34 | 22 | 22 | 0 | 12 | 55.0% |
| emath | 9 | 9 | 9 | 9 | 0 | 0 | 100.0% |
| np.fft.* | 18 | 18 | 18 | 18 | 0 | 0 | 100.0% |
| finfo | 21 | 14 | 13 | 13 | 0 | 1 | 61.9% |
| flatiter | 17 | 8 | 4 | 4 | 0 | 4 | 23.5% |
| iinfo | 9 | 6 | 5 | 5 | 0 | 1 | 55.6% |
| np.linalg.* | 31 | 31 | 31 | 28 | 3 | 0 | 100.0% |
| np.ma.* | 633 | 313 | 215 | 203 | 12 | 98 | 34.0% |
| matrix | 164 | 122 | 85 | 85 | 0 | 37 | 51.8% |
| ndarray.* | 154 | 107 | 70 | 70 | 0 | 38 | 45.5% |
| ndarray.flags | 25 | 20 | 14 | 14 | 0 | 6 | 56.0% |
| ndenumerate | 3 | 1 | 0 | 0 | 0 | 1 | 0.0% |
| ndindex | 2 | 1 | 0 | 0 | 0 | 1 | 0.0% |
| nditer | 34 | 30 | 24 | 24 | 0 | 6 | 70.6% |
| np.* | 380 | 357 | 360 | 351 | 9 | 0 | 94.7% |
| np.polynomial.* | 525 | 517 | 504 | 504 | 0 | 13 | 96.0% |
| np.random.* | 51 | 51 | 51 | 51 | 0 | 0 | 100.0% |
| random.BitGenerator | 11 | 5 | 4 | 4 | 0 | 1 | 36.4% |
| random.Generator | 50 | 46 | 46 | 46 | 0 | 0 | 92.0% |
| random.MT19937 | 12 | 6 | 5 | 5 | 0 | 1 | 41.7% |
| random.PCG64 | 13 | 7 | 6 | 6 | 0 | 1 | 46.2% |
| random.PCG64DXSM | 13 | 7 | 6 | 6 | 0 | 1 | 46.2% |
| random.Philox | 13 | 7 | 6 | 6 | 0 | 1 | 46.2% |
| random.RandomState | 54 | 50 | 50 | 50 | 0 | 0 | 92.6% |
| random.SFC64 | 11 | 5 | 4 | 4 | 0 | 1 | 36.4% |
| random.SeedSequence | 11 | 9 | 9 | 9 | 0 | 0 | 81.8% |
| strings | 148 | 46 | 46 | 46 | 0 | 0 | 31.1% |
| vectorize | 6 | 2 | 0 | 0 | 0 | 2 | 0.0% |

### Available headline APIs without oracle contracts

Parity for these is declared, not oracle-proven (some carry dedicated unit suites instead; see OracleSurfaceCoverageTests' sibling-owned and known-gap classifications).

_None: every available headline API has oracle contracts._

### Oracle-gated APIs the catalog does not credit

The corpus proves NumSharp reproduces these NumPy behaviours, yet no NumSharp member is credited under the NumPy name (add the member, or record a reviewed alias in overrides.json).

_None._

## Signature parity

Availability credits a NumPy API by NAME. Here every credited API's NumPy signature is compared with every C# spelling of it (module functions, members, constructors, extension methods and the typed `NDArray<T>`), from the API diff gap map of `tools/api-parameter-parity` (docs/plans/API_DIFF_GAP.md), which the generator runs on the same tree: a reflection dump, NumPy's runtime signatures completed from its `.pyi` stubs and numpydoc sections, and a compile probe of NumPy-style calls at C# 12 and C# 14. The rules are docs/plans/api-parameter-parity.md's: a positional-only NumPy parameter is matched by slot, a positional-or-keyword one by name and slot, a keyword-only one by name; `*args` needs a `params` array, and keywords NumPy takes through `**kwargs` are not differences. A call is **identical** when it carries no mark. Marks: **M** a NumPy parameter no overload offers (a rename counts: NumPy's keyword does not compile; M ◦, only parameters the plan excludes such as `like`, `subok` or a ufunc's `casting` / `order` / `signature`, is not a difference), **X** a parameter NumPy does not have, **O** more than one overload (NumPy has one signature), **T** a dtype taken as `NPTypeCode` / `Type` instead of `DType` (T ◇: picked by a generic type argument, `np.zeros<T>(shape)`), **B** a NumPy-style call that does not compile (`np.linspace(0, 1)`, `a.transpose(1, 0)`; ◐ at C# 12 only), **L** `null` refused where NumPy's default is None (◐ C# 12 only; ◦ only a `Shape` refuses it and `default` binds), **Q** an integer sequence refused where NumPy's stubs allow one (`axis=(0, 1)`), **N** credited through another name (NumPy's spelling does not compile). The closest overload (the fewest edits from NumPy's) adds positional-order, required-parameter, default-value and `*args` differences. A credited row the gap map does not compare keeps the inventory's own reflection comparison and is not compile-probed.

Headline scope: **244** of 560 APIs have NumPy's exact signature (**43.6%**); of the 519 credited calls compared, **275** differ; 2 credited calls were not compared; 4 compared calls were not compile-probed.

Expanded scope: **796** of 4844 APIs have NumPy's exact signature (**16.4%**); of the 1471 credited calls compared, **675** differ; 2 credited calls were not compared; 129 compared calls were not compile-probed.

| Mark | Meaning | Headline rows | Expanded rows |
|---|---|---:|---:|
| M | a NumPy parameter no overload offers | 48 | 223 |
| X | a parameter NumPy does not have | 50 | 87 |
| O | more than one overload | 211 | 407 |
| T ● | a dtype taken as NPTypeCode / Type | 6 | 14 |
| T ◇ | a generic type argument picks the dtype | 12 | 15 |
| B | a NumPy-style call does not compile | 17 | 30 |
| L | `null` refused for NumPy's None | 61 | 143 |
| Q | an integer sequence refused | 33 | 86 |
| N | credited through another name | 14 | 14 |
| order | a positional parameter at another slot (closest overload) | 10 | 51 |
| required | C# requires what NumPy defaults (closest overload) | 27 | 59 |
| default | another default value (closest overload) | 2 | 2 |
| varargs | `*args` without a `params` array (closest overload) | 7 | 12 |
| M ◦ | only plan-excluded parameters missing (not a difference) | 117 | 150 |
| unprobed | compared by the inventory's own reflection only (no compile probe) | 4 | 129 |

Per surface (expanded scope; surfaces without a credited call are omitted):

| Surface | APIs | Identical | Differs | Not compared | M | X | O | T | B | L | Q | N | Identical % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| broadcast | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10.0% |
| char | 118 | 52 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 44.1% |
| char.chararray | 192 | 64 | 27 | 0 | 5 | 4 | 13 | 2 | 1 | 0 | 10 | 0 | 33.3% |
| dtype | 40 | 2 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 5.0% |
| emath | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| np.fft.* | 18 | 14 | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 77.8% |
| flatiter | 17 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.9% |
| np.linalg.* | 31 | 27 | 4 | 0 | 0 | 1 | 4 | 0 | 0 | 1 | 1 | 0 | 87.1% |
| np.ma.* | 633 | 96 | 173 | 0 | 159 | 32 | 12 | 0 | 10 | 1 | 30 | 0 | 15.2% |
| matrix | 164 | 30 | 36 | 0 | 11 | 1 | 24 | 2 | 2 | 0 | 13 | 0 | 18.3% |
| ndarray.* | 154 | 24 | 32 | 0 | 5 | 4 | 17 | 4 | 2 | 0 | 10 | 0 | 15.6% |
| nditer | 34 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23.5% |
| np.* | 380 | 174 | 184 | 2 | 43 | 45 | 136 | 10 | 15 | 21 | 22 | 14 | 45.8% |
| np.polynomial.* | 525 | 231 | 66 | 0 | 0 | 0 | 60 | 6 | 0 | 0 | 0 | 0 | 44.0% |
| np.random.* | 51 | 2 | 49 | 0 | 0 | 0 | 48 | 1 | 0 | 39 | 0 | 0 | 3.9% |
| random.BitGenerator | 11 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 9.1% |
| random.Generator | 50 | 4 | 40 | 0 | 0 | 0 | 36 | 0 | 0 | 38 | 0 | 0 | 8.0% |
| random.MT19937 | 12 | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 16.7% |
| random.PCG64 | 13 | 2 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 15.4% |
| random.PCG64DXSM | 13 | 2 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 15.4% |
| random.Philox | 13 | 2 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 15.4% |
| random.RandomState | 54 | 2 | 47 | 0 | 0 | 0 | 46 | 1 | 0 | 37 | 0 | 0 | 3.7% |
| random.SFC64 | 11 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 9.1% |
| random.SeedSequence | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18.2% |
| strings | 148 | 43 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 29.1% |

### The API diff gap map in this artifact

The gap map (`tools/api-parameter-parity`, docs/plans/API_DIFF_GAP.md) counts C# spellings; the rows above count NumPy APIs. Every spelling with a NumPy counterpart joins the row of the API it spells (a constructor joins the class row, `NDArray<T>`'s members join `ndarray`'s, an alias or an inherited member adopts the spellings of the row with the same C# member and the same NumPy object); the columns count the same keys over the gap map and over the spellings the emitted rows carry. Compile probe: 7,825 calls, 493 fail at C# 12, 460 at C# 14.

| Count | Gap map | Carried by the artifact |
|---|---:|---:|
| C# spellings with a NumPy counterpart | 1293 | 1291 |
| M ● a NumPy parameter missing | 227 | 226 |
| M ◦ only plan-excluded parameters missing | 142 | 142 |
| X a parameter NumPy lacks | 91 | 90 |
| O more than one overload | 342 | 340 |
| T ● NPTypeCode / Type | 10 | 10 |
| T ◇ a generic dtype selector | 17 | 17 |
| B a NumPy-style call does not compile | 42 | 42 |
| B ◐ only at C# 12 | 0 | 0 |
| L ● null refused | 23 | 23 |
| L ◐ null refused at C# 12 only | 19 | 19 |
| L ◦ only Shape refuses null (default binds) | 73 | 73 |
| Q an integer sequence refused | 77 | 77 |
| no mark at all | 609 | 609 |

Spellings with a NumPy counterpart that no NumPy row carries:

| Spelling | Where | Reason |
|---|---|---|
| `np.random` / `get_bit_generator` | `numsharp.random.get_bit_generator` | NumPy has the name outside its __all__ and documentation; the catalog lists the NumSharp member as NumSharp-only |
| `np.random` / `set_bit_generator` | `numsharp.random.set_bit_generator` | NumPy has the name outside its __all__ and documentation; the catalog lists the NumSharp member as NumSharp-only |
| `np.lib.npyio.NpzFile` / `(constructor)` | not shown (`numpy.lib.npyio.NpzFile`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.random.MT19937` / `_legacy_seeding` | not shown (`numpy.random.MT19937._legacy_seeding`) | NumPy's name is private (a leading underscore): the catalog lists public NumPy APIs only, while NumSharp exposes the member publicly |

### Headline APIs whose signature differs (275)

| API | Marks | Missing | NumSharp-only parameters | Overloads | NPTypeCode / Type / generic | Closest overload | Compile probe |
|---|---|---|---|---:|---|---|---|
| [`np.fft.fftfreq`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftfreq.html#numpy.fft.fftfreq) | O |  |  | 3 |  |  |  |
| [`np.fft.fftshift`](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html#numpy.fft.fftshift) | O |  |  | 2 |  |  |  |
| [`np.fft.ifftshift`](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html#numpy.fft.ifftshift) | O |  |  | 2 |  |  |  |
| [`np.fft.rfftfreq`](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftfreq.html#numpy.fft.rfftfreq) | O |  |  | 3 |  |  |  |
| [`np.linalg.multi_dot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html#numpy.linalg.multi_dot) | O |  |  | 2 |  |  |  |
| [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html#numpy.linalg.norm) | OLQ |  |  | 2 |  |  | L: the None values in one call: ambiguous: `np.linalg.norm(NDArray, object, int?, bool)` / `np.linalg.norm(NDArray, object, int[], bool)` — `np.linalg.norm(a, ord: null, axis: null, keepdims: false)`<br>Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.linalg.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html#numpy.linalg.tensordot) | XO |  | `axesA`, `axesB` | 2 |  |  |  |
| [`np.linalg.vector_norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.vector_norm.html#numpy.linalg.vector_norm) | O |  |  | 2 |  |  |  |
| [`ndarray.all`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.all.html#numpy.ndarray.all) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.any`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.any.html#numpy.ndarray.any) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmax.html#numpy.ndarray.argmax) | O |  |  | 3 |  |  |  |
| [`ndarray.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmin.html#numpy.ndarray.argmin) | O |  |  | 3 |  |  |  |
| [`ndarray.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argpartition.html#numpy.ndarray.argpartition) | O |  |  | 3 |  |  |  |
| [`ndarray.argsort`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argsort.html#numpy.ndarray.argsort) | OT |  |  | 2 | `<T>(int axis)` |  |  |
| [`ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype) | OB |  |  | 2 |  | `order: NumPy #1, C# #2`, `casting: NumPy #2, C# #3`, `copy: NumPy #4, C# #1`, `C# requires order`, `C# requires copy` | B: `order` alone: needs `copy`<br>B: `casting` alone: needs `copy`<br>B: positionally up to `order`: no overload fits; the closest one wants `bool` for argument 2 — `a.astype(np.float64, 'K')` |
| [`ndarray.choose`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.choose.html#numpy.ndarray.choose) | O |  |  | 3 |  |  |  |
| [`ndarray.conj`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conj.html#numpy.ndarray.conj) | X |  | `out` | 1 |  |  |  |
| [`ndarray.conjugate`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conjugate.html#numpy.ndarray.conjugate) | X |  | `out` | 1 |  |  |  |
| [`ndarray.dot`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dot.html#numpy.ndarray.dot) | M | `out` |  | 1 |  |  |  |
| [`ndarray.getfield`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.getfield.html#numpy.ndarray.getfield) | OT |  |  | 2 | `<T>(int offset)` |  |  |
| [`ndarray.item`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html#numpy.ndarray.item) | OT |  |  | 10 | `<T>()`, `<T>(long index)`, `<T>(long i, long j)`, `<T>(long i, long j, long k)`, `<T>(long[] indices)` |  |  |
| [`ndarray.max`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.max.html#numpy.ndarray.max) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.mean`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.mean.html#numpy.ndarray.mean) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.min`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html#numpy.ndarray.min) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.partition`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.partition.html#numpy.ndarray.partition) | O |  |  | 3 |  |  |  |
| [`ndarray.prod`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.prod.html#numpy.ndarray.prod) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ravel.html#numpy.ndarray.ravel) | O |  |  | 2 |  | `C# requires order` |  |
| [`ndarray.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.repeat.html#numpy.ndarray.repeat) | O |  |  | 3 |  |  |  |
| [`ndarray.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape) | MO | `copy`, `order` |  | 6 |  | `C# requires order` |  |
| [`ndarray.resize`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html#numpy.ndarray.resize) | O |  |  | 3 |  |  |  |
| [`ndarray.searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.searchsorted.html#numpy.ndarray.searchsorted) | O |  |  | 3 |  |  |  |
| [`ndarray.setfield`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setfield.html#numpy.ndarray.setfield) | MX | `val (C#: value)` | `value` | 1 |  |  |  |
| [`ndarray.squeeze`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.squeeze.html#numpy.ndarray.squeeze) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.std`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.std.html#numpy.ndarray.std) | MQ | `correction` |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.sum`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.sum.html#numpy.ndarray.sum) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.take`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.take.html#numpy.ndarray.take) | XO |  | `index` | 2 |  |  |  |
| [`ndarray.tofile`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile) | O |  |  | 2 |  |  |  |
| [`ndarray.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.transpose.html#numpy.ndarray.transpose) | B |  |  | 1 |  | `*axes as int[] premute (no params array)` | B: several `*axes` values: no overload takes 2 arguments — `a.transpose(1, 1)` |
| [`ndarray.var`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.var.html#numpy.ndarray.var) | MQ | `correction` |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 1 |
| [`ndarray.view`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html#numpy.ndarray.view) | OT |  |  | 4 | `Type type`, `<T>()` | `C# requires dtype`, `C# requires type` |  |
| [`np.abs`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html#numpy.absolute) | O |  |  | 2 |  |  |  |
| [`np.absolute`](https://numpy.org/doc/stable/reference/generated/numpy.absolute.html#numpy.absolute) | O |  |  | 2 |  |  |  |
| [`np.acos`](https://numpy.org/doc/stable/reference/generated/numpy.acos.html#numpy.acos) | N |  |  | 1 |  |  |  |
| [`np.all`](https://numpy.org/doc/stable/reference/generated/numpy.all.html#numpy.all) | XOL |  | `nd` | 6 |  |  | L: `axis`: ambiguous: `np.all(NDArray, int[], bool)` / `np.all(NDArray, int?, NDArray, bool, NDArray)` (C# 12 only) |
| [`np.amax`](https://numpy.org/doc/stable/reference/generated/numpy.amax.html#numpy.amax) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.amin`](https://numpy.org/doc/stable/reference/generated/numpy.amin.html#numpy.amin) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.any`](https://numpy.org/doc/stable/reference/generated/numpy.any.html#numpy.any) | XOL |  | `nd` | 6 |  |  | L: `axis`: ambiguous: `np.any(NDArray, int[], bool)` / `np.any(NDArray, int?, NDArray, bool, NDArray)` (C# 12 only) |
| [`np.append`](https://numpy.org/doc/stable/reference/generated/numpy.append.html#numpy.append) | O |  |  | 2 |  |  |  |
| [`np.apply_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html#numpy.apply_along_axis) | O |  |  | 2 |  |  |  |
| [`np.apply_over_axes`](https://numpy.org/doc/stable/reference/generated/numpy.apply_over_axes.html#numpy.apply_over_axes) | O |  |  | 2 |  |  |  |
| [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange) | OBL |  |  | 11 |  | `C# requires stop`, `C# requires step`, `C# requires dtype` | B: `step` alone: no overload takes 2 arguments<br>B: `device` alone: no overload takes 2 arguments<br>L: `stop`: no overload fits; the closest one wants `float` for argument 1 |
| [`np.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html#numpy.argmax) | O |  |  | 3 |  |  |  |
| [`np.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html#numpy.argmin) | O |  |  | 3 |  |  |  |
| [`np.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html#numpy.argpartition) | O |  |  | 3 |  |  |  |
| [`np.argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html#numpy.argsort) | XOT |  | `nd` | 2 | `<T>(NDArray nd, int axis)` |  |  |
| [`np.around`](https://numpy.org/doc/stable/reference/generated/numpy.around.html#numpy.around) | MX | `a (C#: x)` | `dtype`, `x` | 1 |  |  |  |
| [`np.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array) | MXO | `object (C#: array, buffer, data, nd, scalar, strArray, value)`, `ndmax` | `array`, `buffer`, `data`, `nd`, `scalar`, `size`, `strArray`, `value` | 33 |  | `order: 'K' vs 'C'` |  |
| [`np.array2string`](https://numpy.org/doc/stable/reference/generated/numpy.array2string.html#numpy.array2string) | M | `formatter`, `legacy` |  | 1 |  |  |  |
| [`np.array_equal`](https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html#numpy.array_equal) | MX | `a1 (C#: a)`, `a2 (C#: b)` | `a`, `b` | 1 |  |  |  |
| [`np.array_repr`](https://numpy.org/doc/stable/reference/generated/numpy.array_repr.html#numpy.array_repr) | MX | `arr (C#: a)` | `a` | 1 |  |  |  |
| [`np.array_split`](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split) | XO |  | `indices` | 3 |  |  |  |
| [`np.asanyarray`](https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html#numpy.asanyarray) | MXOB | `copy` | `buffer` | 3 |  | `C# requires dtype`, `C# requires order` | B: `order` alone: no overload fits; the closest one wants `np.MemoryView` for argument 1 |
| [`np.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray) | XOL |  | `buffer`, `data`, `ndim` | 7 |  |  | L: `order`: argument 2: `<null>` -> `char` |
| [`np.asarray_chkfinite`](https://numpy.org/doc/stable/reference/generated/numpy.asarray_chkfinite.html#numpy.asarray_chkfinite) | L |  |  | 1 |  |  | L: `order`: argument 2: `<null>` -> `char` |
| [`np.ascontiguousarray`](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray) | XO |  | `buffer` | 2 |  |  |  |
| [`np.asfortranarray`](https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray) | XO |  | `buffer` | 2 |  |  |  |
| [`np.asin`](https://numpy.org/doc/stable/reference/generated/numpy.asin.html#numpy.asin) | N |  |  | 1 |  |  |  |
| [`np.asmatrix`](https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html#numpy.asmatrix) | O |  |  | 4 |  |  |  |
| [`np.atan`](https://numpy.org/doc/stable/reference/generated/numpy.atan.html#numpy.atan) | N |  |  | 1 |  |  |  |
| [`np.atan2`](https://numpy.org/doc/stable/reference/generated/numpy.atan2.html#numpy.atan2) | N |  |  | 1 |  |  |  |
| [`np.atleast_1d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d) | O |  |  | 4 |  |  |  |
| [`np.atleast_2d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d) | O |  |  | 4 |  |  |  |
| [`np.atleast_3d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d) | O |  |  | 4 |  |  |  |
| [`np.average`](https://numpy.org/doc/stable/reference/generated/numpy.average.html#numpy.average) | MOL | `returned` |  | 2 |  |  | L: `axis`: ambiguous: `np.average(NDArray, int?, NDArray, bool)` / `np.average(NDArray, int[], NDArray, bool)` |
| [`np.base_repr`](https://numpy.org/doc/stable/reference/generated/numpy.base_repr.html#numpy.base_repr) | O |  |  | 2 |  |  |  |
| [`np.binary_repr`](https://numpy.org/doc/stable/reference/generated/numpy.binary_repr.html#numpy.binary_repr) | O |  |  | 2 |  |  |  |
| [`np.bitwise_invert`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_invert.html#numpy.bitwise_invert) | N |  |  | 1 |  |  |  |
| [`np.bitwise_left_shift`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_left_shift.html#numpy.bitwise_left_shift) | ON |  |  | 2 |  |  |  |
| [`np.bitwise_right_shift`](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_right_shift.html#numpy.bitwise_right_shift) | ON |  |  | 2 |  |  |  |
| [`np.bmat`](https://numpy.org/doc/stable/reference/generated/numpy.bmat.html#numpy.bmat) | OB |  |  | 5 |  | `C# requires ldict` | B: `gdict` alone: needs `ldict` |
| [`np.broadcast_arrays`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays) | O |  |  | 2 |  |  |  |
| [`np.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to) | MXO | `array (C#: from)`, `shape (C#: against)` | `against`, `from` | 9 |  |  |  |
| [`np.can_cast`](https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html#numpy.can_cast) | MXOT | `from_ (C#: from, value)` | `from`, `value` | 20 | `NPTypeCode from`, `NPTypeCode to`, `Type from`, `Type to`, `<TFrom, TTo>(string casting)` |  |  |
| [`np.choose`](https://numpy.org/doc/stable/reference/generated/numpy.choose.html#numpy.choose) | O |  |  | 3 |  |  |  |
| [`np.concat`](https://numpy.org/doc/stable/reference/generated/numpy.concat.html#numpy.concat) | O |  |  | 9 |  |  |  |
| [`np.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html#numpy.concatenate) | O |  |  | 9 |  |  |  |
| [`np.copyto`](https://numpy.org/doc/stable/reference/generated/numpy.copyto.html#numpy.copyto) | O |  |  | 4 |  |  |  |
| [`np.count_nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero) | OBLQ |  |  | 2 |  | `C# requires axis` | B: `keepdims` alone: needs `axis`<br>L: `axis`: argument 2: `<null>` -> `int`<br>Q: `axis` as a sequence: no overload fits; the closest one wants `int` for argument 2 |
| [`np.cumprod`](https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html#numpy.cumprod) | MX | `a (C#: arr)`, `dtype (C#: typeCode)` | `arr`, `typeCode` | 1 |  |  |  |
| [`np.cumsum`](https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html#numpy.cumsum) | MX | `a (C#: arr)`, `dtype (C#: typeCode)` | `arr`, `typeCode` | 1 |  |  |  |
| [`np.cumulative_prod`](https://numpy.org/doc/stable/reference/generated/numpy.cumulative_prod.html#numpy.cumulative_prod) | MXN | `dtype`, `include_initial` | `typeCode` | 1 |  |  |  |
| [`np.cumulative_sum`](https://numpy.org/doc/stable/reference/generated/numpy.cumulative_sum.html#numpy.cumulative_sum) | MXN | `dtype`, `include_initial` | `typeCode` | 1 |  |  |  |
| [`np.delete`](https://numpy.org/doc/stable/reference/generated/numpy.delete.html#numpy.delete) | O |  |  | 7 |  |  |  |
| [`np.divmod`](https://numpy.org/doc/stable/reference/generated/numpy.divmod.html#numpy.divmod) | O |  |  | 2 |  |  |  |
| [`np.dsplit`](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html#numpy.dsplit) | XO |  | `indices` | 2 |  |  |  |
| [`np.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html#numpy.einsum) | O |  |  | 3 |  | `order: ... vs 'K'`, `casting: ... vs "safe"`, `*operands as string subscripts, NDArray[] operands (no params array)` |  |
| [`np.einsum_path`](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html#numpy.einsum_path) | O |  |  | 3 |  | `C# requires optimize`, `*operands as string subscripts, NDArray[] operands (no params array)` |  |
| [`np.empty`](https://numpy.org/doc/stable/reference/generated/numpy.empty.html#numpy.empty) | OTB |  |  | 8 | `<T>(int[] shape)`, `<T>(long[] shape)` | `C# requires dtype` | B: `device` alone: needs `dtype`<br>B: positionally up to `order`: no overload fits; the closest one wants `string` for argument 3 — `np.empty(new Shape(2), np.float64, 'C')` |
| [`np.empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html#numpy.empty_like) | OBL |  |  | 2 |  | `shape: NumPy #4, C# #2` | B: `order` alone: needs `dtype`<br>L: `shape`: null -> `Shape` (`default` binds) |
| [`np.equal`](https://numpy.org/doc/stable/reference/generated/numpy.equal.html#numpy.equal) | O |  |  | 3 |  |  |  |
| [`np.exp`](https://numpy.org/doc/stable/reference/generated/numpy.exp.html#numpy.exp) | O |  |  | 2 |  |  |  |
| [`np.exp2`](https://numpy.org/doc/stable/reference/generated/numpy.exp2.html#numpy.exp2) | O |  |  | 2 |  |  |  |
| [`np.expand_dims`](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html#numpy.expand_dims) | O |  |  | 3 |  |  |  |
| [`np.expm1`](https://numpy.org/doc/stable/reference/generated/numpy.expm1.html#numpy.expm1) | O |  |  | 2 |  |  |  |
| [`np.flip`](https://numpy.org/doc/stable/reference/generated/numpy.flip.html#numpy.flip) | OL |  |  | 2 |  |  | L: `axis`: ambiguous: `np.flip(NDArray, int?)` / `np.flip(NDArray, int[])` |
| [`np.float_power`](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html#numpy.float_power) | O |  |  | 2 |  |  |  |
| [`np.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html#numpy.floor_divide) | O |  |  | 2 |  |  |  |
| [`np.fmod`](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html#numpy.fmod) | O |  |  | 2 |  |  |  |
| [`np.frexp`](https://numpy.org/doc/stable/reference/generated/numpy.frexp.html#numpy.frexp) | MX | `out (C#: out1)`, `dtype` | `out1`, `out2` | 1 |  |  |  |
| [`np.frombuffer`](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html#numpy.frombuffer) | XOT |  | `address`, `array`, `byteLength`, `dispose`, `memory`, `segment` | 18 | `NPTypeCode dtype` |  |  |
| [`np.fromfile`](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html#numpy.fromfile) | XO |  | `stream` | 2 |  |  |  |
| [`np.frompyfunc`](https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc) | M | `identity` |  | 1 |  |  |  |
| [`np.fromstring`](https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html#numpy.fromstring) | O |  |  | 2 |  |  |  |
| [`np.full`](https://numpy.org/doc/stable/reference/generated/numpy.full.html#numpy.full) | MOT | `order` |  | 5 | `<T>(int[] shape, object fill_value)`, `<T>(long[] shape, object fill_value)` |  |  |
| [`np.full_like`](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy.full_like) | MOB | `shape` |  | 2 |  | `C# requires dtype`, `C# requires order` | B: `order` alone: needs `dtype` |
| [`np.geomspace`](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html#numpy.geomspace) | O |  |  | 4 |  |  |  |
| [`np.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html#numpy.gradient) | OQ |  |  | 5 |  | `*varargs as object spacing (no params array)` | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.greater`](https://numpy.org/doc/stable/reference/generated/numpy.greater.html#numpy.greater) | O |  |  | 3 |  |  |  |
| [`np.greater_equal`](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html#numpy.greater_equal) | O |  |  | 3 |  |  |  |
| [`np.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html#numpy.histogram) | OL |  |  | 4 |  |  | L: `density`: argument 2: `<null>` -> `bool` |
| [`np.histogram2d`](https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html#numpy.histogram2d) | OL |  |  | 6 |  |  | L: `density`: argument 3: `<null>` -> `bool` |
| [`np.histogram_bin_edges`](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges) | O |  |  | 4 |  |  |  |
| [`np.histogramdd`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html#numpy.histogramdd) | OL |  |  | 8 |  |  | L: `density`: argument 2: `<null>` -> `bool` |
| [`np.hsplit`](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html#numpy.hsplit) | XO |  | `indices` | 2 |  |  |  |
| [`np.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy.hstack) | M | `dtype`, `casting` |  | 1 |  |  |  |
| [`np.indices`](https://numpy.org/doc/stable/reference/generated/numpy.indices.html#numpy.indices) | MO | `sparse` |  | 3 |  |  |  |
| [`np.insert`](https://numpy.org/doc/stable/reference/generated/numpy.insert.html#numpy.insert) | XO |  | `value` | 12 |  |  |  |
| [`np.interp`](https://numpy.org/doc/stable/reference/generated/numpy.interp.html#numpy.interp) | O |  |  | 2 |  |  |  |
| [`np.intersect1d`](https://numpy.org/doc/stable/reference/generated/numpy.intersect1d.html#numpy.intersect1d) | O |  |  | 3 |  |  |  |
| [`np.iscomplex`](https://numpy.org/doc/stable/reference/generated/numpy.iscomplex.html#numpy.iscomplex) | MX | `x (C#: a)` | `a` | 1 |  |  |  |
| [`np.iscomplexobj`](https://numpy.org/doc/stable/reference/generated/numpy.iscomplexobj.html#numpy.iscomplexobj) | MX | `x (C#: a)` | `a` | 1 |  |  |  |
| [`np.isdtype`](https://numpy.org/doc/stable/reference/generated/numpy.isdtype.html#numpy.isdtype) | XO |  | `arr`, `kinds` | 15 |  |  |  |
| [`np.isreal`](https://numpy.org/doc/stable/reference/generated/numpy.isreal.html#numpy.isreal) | MX | `x (C#: a)` | `a` | 1 |  |  |  |
| [`np.isrealobj`](https://numpy.org/doc/stable/reference/generated/numpy.isrealobj.html#numpy.isrealobj) | MX | `x (C#: a)` | `a` | 1 |  |  |  |
| [`np.isscalar`](https://numpy.org/doc/stable/reference/generated/numpy.isscalar.html#numpy.isscalar) | MX | `element (C#: obj)` | `obj` | 1 |  |  |  |
| [`np.issubdtype`](https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html#numpy.issubdtype) | XOT |  | `arr` | 11 | `NPTypeCode arg1`, `NPTypeCode arg2`, `Type arg1`, `Type arg2` |  |  |
| [`np.ldexp`](https://numpy.org/doc/stable/reference/generated/numpy.ldexp.html#numpy.ldexp) | M | `dtype` |  | 1 |  |  |  |
| [`np.left_shift`](https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html#numpy.left_shift) | O |  |  | 2 |  |  |  |
| [`np.less`](https://numpy.org/doc/stable/reference/generated/numpy.less.html#numpy.less) | O |  |  | 3 |  |  |  |
| [`np.less_equal`](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html#numpy.less_equal) | O |  |  | 3 |  |  |  |
| [`np.lexsort`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort) | O |  |  | 2 |  |  |  |
| [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace) | MOB | `retstep`, `axis` |  | 4 |  | `dtype: NumPy #5, C# #4`, `C# requires num` | B: required parameters only: no overload takes 2 arguments — `np.linspace(1.0, 1.0)` |
| [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load) | XO |  | `bytes` | 3 |  |  |  |
| [`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html#numpy.loadtxt) | XO |  | `lines`, `reader`, `stream` | 4 |  |  |  |
| [`np.log`](https://numpy.org/doc/stable/reference/generated/numpy.log.html#numpy.log) | O |  |  | 2 |  |  |  |
| [`np.log10`](https://numpy.org/doc/stable/reference/generated/numpy.log10.html#numpy.log10) | O |  |  | 2 |  |  |  |
| [`np.log1p`](https://numpy.org/doc/stable/reference/generated/numpy.log1p.html#numpy.log1p) | O |  |  | 2 |  |  |  |
| [`np.log2`](https://numpy.org/doc/stable/reference/generated/numpy.log2.html#numpy.log2) | O |  |  | 2 |  |  |  |
| [`np.logspace`](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html#numpy.logspace) | O |  |  | 2 |  |  |  |
| [`np.max`](https://numpy.org/doc/stable/reference/generated/numpy.max.html#numpy.max) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.mean`](https://numpy.org/doc/stable/reference/generated/numpy.mean.html#numpy.mean) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.median`](https://numpy.org/doc/stable/reference/generated/numpy.median.html#numpy.median) | OL |  |  | 2 |  |  | L: `axis`: ambiguous: `np.median(NDArray, int?, NDArray, bool, bool)` / `np.median(NDArray, int[], NDArray, bool, bool)` |
| [`np.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid) | O |  |  | 3 |  | `*xi as NDArray x1, NDArray x2 (no params array)` |  |
| [`np.min`](https://numpy.org/doc/stable/reference/generated/numpy.min.html#numpy.min) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html#numpy.mod) | O |  |  | 2 |  |  |  |
| [`np.modf`](https://numpy.org/doc/stable/reference/generated/numpy.modf.html#numpy.modf) | MX | `out (C#: outFrac)` | `outFrac`, `outIntegral` | 1 |  |  |  |
| [`np.moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html#numpy.moveaxis) | O |  |  | 4 |  |  |  |
| [`np.nanargmax`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html#numpy.nanargmax) | O |  |  | 2 |  |  |  |
| [`np.nanargmin`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html#numpy.nanargmin) | O |  |  | 2 |  |  |  |
| [`np.nanmax`](https://numpy.org/doc/stable/reference/generated/numpy.nanmax.html#numpy.nanmax) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.nanmean`](https://numpy.org/doc/stable/reference/generated/numpy.nanmean.html#numpy.nanmean) | MBQ | `out`, `where` |  | 1 |  | `dtype: NumPy #2, C# #3`, `keepdims: NumPy #4, C# #2` | B: positionally up to `dtype`: no overload fits; the closest one wants `bool` for argument 3 — `np.nanmean(a, 1, np.float64)`<br>Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.nanmedian`](https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html#numpy.nanmedian) | OL |  |  | 2 |  |  | L: `axis`: ambiguous: `np.nanmedian(NDArray, int?, NDArray, bool, bool)` / `np.nanmedian(NDArray, int[], NDArray, bool, bool)` |
| [`np.nanmin`](https://numpy.org/doc/stable/reference/generated/numpy.nanmin.html#numpy.nanmin) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.nanpercentile`](https://numpy.org/doc/stable/reference/generated/numpy.nanpercentile.html#numpy.nanpercentile) | OL |  |  | 6 |  |  | L: `axis`: ambiguous: `np.nanpercentile(NDArray, double, int?, NDArray, bool, string, bool, NDArray)` / `np.nanpercentile(NDArray, double, int[], NDArray, bool, string, bool, NDArray)` |
| [`np.nanprod`](https://numpy.org/doc/stable/reference/generated/numpy.nanprod.html#numpy.nanprod) | MQ | `dtype`, `out`, `initial`, `where` |  | 1 |  | `keepdims: NumPy #4, C# #2` | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.nanquantile`](https://numpy.org/doc/stable/reference/generated/numpy.nanquantile.html#numpy.nanquantile) | OL |  |  | 8 |  |  | L: `axis`: ambiguous: `np.nanquantile(NDArray, double, int?, NDArray, bool, string, bool, NDArray)` / `np.nanquantile(NDArray, double, int[], NDArray, bool, string, bool, NDArray)` |
| [`np.nanstd`](https://numpy.org/doc/stable/reference/generated/numpy.nanstd.html#numpy.nanstd) | MBQ | `out`, `where`, `mean`, `correction` |  | 1 |  | `dtype: NumPy #2, C# #4`, `ddof: NumPy #4, C# #3`, `keepdims: NumPy #5, C# #2` | B: positionally up to `dtype`: no overload fits; the closest one wants `bool` for argument 3 — `np.nanstd(a, 1, np.float64)`<br>Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.nansum`](https://numpy.org/doc/stable/reference/generated/numpy.nansum.html#numpy.nansum) | MQ | `dtype`, `out`, `initial`, `where` |  | 1 |  | `keepdims: NumPy #4, C# #2` | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.nanvar`](https://numpy.org/doc/stable/reference/generated/numpy.nanvar.html#numpy.nanvar) | MBQ | `out`, `where`, `mean`, `correction` |  | 1 |  | `dtype: NumPy #2, C# #4`, `ddof: NumPy #4, C# #3`, `keepdims: NumPy #5, C# #2` | B: positionally up to `dtype`: no overload fits; the closest one wants `bool` for argument 3 — `np.nanvar(a, 1, np.float64)`<br>Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.nested_iters`](https://numpy.org/doc/stable/reference/generated/numpy.nested_iters.html#numpy.nested_iters) | O |  |  | 2 |  |  |  |
| [`np.not_equal`](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html#numpy.not_equal) | O |  |  | 3 |  |  |  |
| [`np.ones`](https://numpy.org/doc/stable/reference/generated/numpy.ones.html#numpy.ones) | OTB |  |  | 8 | `<T>(int[] shape)` | `C# requires dtype` | B: `device` alone: needs `dtype`<br>B: positionally up to `order`: no overload fits; the closest one wants `string` for argument 3 — `np.ones(new Shape(2), np.float64, 'C')` |
| [`np.ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html#numpy.ones_like) | MOB | `shape` |  | 2 |  | `C# requires dtype`, `C# requires order` | B: `order` alone: needs `dtype` |
| [`np.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad) | O |  |  | 10 |  |  |  |
| [`np.partition`](https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition) | O |  |  | 3 |  |  |  |
| [`np.percentile`](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html#numpy.percentile) | OL |  |  | 6 |  |  | L: `axis`: ambiguous: `np.percentile(NDArray, double, int?, NDArray, bool, string, bool, NDArray)` / `np.percentile(NDArray, double, int[], NDArray, bool, string, bool, NDArray)` |
| [`np.piecewise`](https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html#numpy.piecewise) | O |  |  | 3 |  |  |  |
| [`np.place`](https://numpy.org/doc/stable/reference/generated/numpy.place.html#numpy.place) | O |  |  | 2 |  |  |  |
| [`np.pow`](https://numpy.org/doc/stable/reference/generated/numpy.pow.html#numpy.pow) | ON |  |  | 4 |  |  |  |
| [`np.power`](https://numpy.org/doc/stable/reference/generated/numpy.power.html#numpy.power) | O |  |  | 4 |  |  |  |
| [`np.printoptions`](https://numpy.org/doc/stable/reference/generated/numpy.printoptions.html#numpy.printoptions) | M | `formatter`, `legacy`, `override_repr` |  | 1 |  | `sign: NumPy #8, C# #7`, `floatmode: NumPy #9, C# #8` |  |
| [`np.prod`](https://numpy.org/doc/stable/reference/generated/numpy.prod.html#numpy.prod) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.promote_types`](https://numpy.org/doc/stable/reference/generated/numpy.promote_types.html#numpy.promote_types) | OT |  |  | 4 | `NPTypeCode type1`, `NPTypeCode type2`, `Type type1`, `Type type2`, `<T1, T2>()` |  |  |
| [`np.ptp`](https://numpy.org/doc/stable/reference/generated/numpy.ptp.html#numpy.ptp) | OL |  |  | 2 |  |  | L: `axis`: ambiguous: `np.ptp(NDArray, int?, NDArray, bool)` / `np.ptp(NDArray, int[], NDArray, bool)` |
| [`np.put`](https://numpy.org/doc/stable/reference/generated/numpy.put.html#numpy.put) | MXO | `ind (C#: index, indices)`, `v (C#: value, values)` | `index`, `indices`, `value`, `values` | 2 |  |  |  |
| [`np.putmask`](https://numpy.org/doc/stable/reference/generated/numpy.putmask.html#numpy.putmask) | O |  |  | 2 |  |  |  |
| [`np.quantile`](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html#numpy.quantile) | OL |  |  | 8 |  |  | L: `axis`: ambiguous: `np.quantile(NDArray, double, int?, NDArray, bool, string, bool, NDArray)` / `np.quantile(NDArray, double, int[], NDArray, bool, string, bool, NDArray)` |
| [`np.ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html#numpy.ravel) | O |  |  | 2 |  | `C# requires order` |  |
| [`np.ravel_multi_index`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index) | XO |  | `coords`, `modes` | 3 |  |  |  |
| [`np.remainder`](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html#numpy.remainder) | O |  |  | 2 |  |  |  |
| [`np.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat) | O |  |  | 5 |  |  |  |
| [`np.require`](https://numpy.org/doc/stable/reference/generated/numpy.require.html#numpy.require) | O |  |  | 2 |  |  |  |
| [`np.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html#numpy.reshape) | MO | `order`, `copy` |  | 4 |  |  |  |
| [`np.result_type`](https://numpy.org/doc/stable/reference/generated/numpy.result_type.html#numpy.result_type) | OT |  |  | 8 | `NPTypeCode[] types`, `NPTypeCode type1`, `NPTypeCode type2`, `Type type1`, `Type type2` |  |  |
| [`np.right_shift`](https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html#numpy.right_shift) | O |  |  | 2 |  |  |  |
| [`np.roll`](https://numpy.org/doc/stable/reference/generated/numpy.roll.html#numpy.roll) | OQ |  |  | 2 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 3<br>Q: `shift` as a sequence: no overload fits; the closest one wants `long` for argument 2 |
| [`np.round`](https://numpy.org/doc/stable/reference/generated/numpy.round.html#numpy.round) | MXN | `a (C#: x)` | `dtype`, `x` | 1 |  |  |  |
| [`np.row_stack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack) | MN | `dtype`, `casting` |  | 1 |  |  |  |
| [`np.save`](https://numpy.org/doc/stable/reference/generated/numpy.save.html#numpy.save) | O |  |  | 4 |  |  |  |
| [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html#numpy.savetxt) | XO |  | `stream`, `writer` | 6 |  |  |  |
| [`np.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez) | O |  |  | 8 |  | `*args as NDArray[] args (no params array)` |  |
| [`np.savez_compressed`](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed) | O |  |  | 8 |  | `*args as NDArray[] args (no params array)` |  |
| [`np.searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy.searchsorted) | O |  |  | 3 |  |  |  |
| [`np.set_printoptions`](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions) | M | `formatter`, `legacy`, `override_repr` |  | 1 |  | `sign: NumPy #8, C# #7`, `floatmode: NumPy #9, C# #8` |  |
| [`np.size`](https://numpy.org/doc/stable/reference/generated/numpy.size.html#numpy.size) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.split`](https://numpy.org/doc/stable/reference/generated/numpy.split.html#numpy.split) | XO |  | `indices` | 3 |  |  |  |
| [`np.square`](https://numpy.org/doc/stable/reference/generated/numpy.square.html#numpy.square) | O |  |  | 2 |  |  |  |
| [`np.squeeze`](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html#numpy.squeeze) | XOLQ |  | `shape` | 3 |  | `C# requires axis` | L: `axis`: argument 2: `<null>` -> `int`<br>Q: `axis` as a sequence: no overload fits; the closest one wants `int` for argument 2 |
| [`np.stack`](https://numpy.org/doc/stable/reference/generated/numpy.stack.html#numpy.stack) | M | `out`, `dtype`, `casting` |  | 1 |  |  |  |
| [`np.std`](https://numpy.org/doc/stable/reference/generated/numpy.std.html#numpy.std) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html#numpy.sum) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.take`](https://numpy.org/doc/stable/reference/generated/numpy.take.html#numpy.take) | XO |  | `index` | 2 |  |  |  |
| [`np.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html#numpy.tensordot) | XO |  | `axesA`, `axesB` | 3 |  |  |  |
| [`np.tile`](https://numpy.org/doc/stable/reference/generated/numpy.tile.html#numpy.tile) | O |  |  | 2 |  |  |  |
| [`np.transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html#numpy.transpose) | MX | `axes (C#: premute)` | `premute` | 1 |  |  |  |
| [`np.trim_zeros`](https://numpy.org/doc/stable/reference/generated/numpy.trim_zeros.html#numpy.trim_zeros) | OLQ |  |  | 2 |  |  | L: the None values in one call: ambiguous: `np.trim_zeros(NDArray, string, int?)` / `np.trim_zeros(NDArray, string, int[])` — `np.trim_zeros(a, trim: "fb", axis: null)`<br>Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html#numpy.unravel_index) | XO |  | `index` | 2 |  |  |  |
| [`np.unwrap`](https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html#numpy.unwrap) | O |  |  | 2 |  |  |  |
| [`np.var`](https://numpy.org/doc/stable/reference/generated/numpy.var.html#numpy.var) | Q |  |  | 1 |  |  | Q: `axis` as a sequence: no overload fits; the closest one wants `int?` for argument 2 |
| [`np.vsplit`](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html#numpy.vsplit) | XO |  | `indices` | 2 |  |  |  |
| [`np.vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack) | M | `dtype`, `casting` |  | 1 |  |  |  |
| [`np.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where) | O |  |  | 5 |  | `C# requires x`, `C# requires y` |  |
| [`np.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html#numpy.zeros) | OTB |  |  | 8 | `<T>(int[] shape)`, `<T>(long[] shape)` | `C# requires dtype` | B: `device` alone: needs `dtype`<br>B: positionally up to `order`: no overload fits; the closest one wants `string` for argument 3 — `np.zeros(new Shape(2), np.float64, 'C')` |
| [`np.zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html#numpy.zeros_like) | MOB | `shape` |  | 2 |  | `C# requires dtype`, `C# requires order` | B: `order` alone: needs `dtype` |
| [`np.polyint`](https://numpy.org/doc/stable/reference/generated/numpy.polyint.html#numpy.polyint) | O |  |  | 2 |  |  |  |
| [`np.polyval`](https://numpy.org/doc/stable/reference/generated/numpy.polyval.html#numpy.polyval) | O |  |  | 2 |  |  |  |
| [`np.random.beta`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html#numpy.random.beta) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.binomial`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html#numpy.random.binomial) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.chisquare`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.chisquare.html#numpy.random.chisquare) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html#numpy.random.choice) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng) | OL |  |  | 13 |  | `C# requires seed` | L: `seed`: ambiguous: `NumPyRandom.default_rng(int[])` / `NumPyRandom.default_rng(long[])` (C# 12 only) |
| [`np.random.dirichlet`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.dirichlet.html#numpy.random.dirichlet) | OL |  |  | 5 |  |  | L: `size`: ambiguous: `NumPyRandom.dirichlet(double[], int[])` / `NumPyRandom.dirichlet(double[], long[])` (C# 12 only) |
| [`np.random.exponential`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html#numpy.random.exponential) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.f`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.f.html#numpy.random.f) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.gamma`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html#numpy.random.gamma) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.geometric`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html#numpy.random.geometric) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.get_state`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html#numpy.random.get_state) | O |  |  | 2 |  | `C# requires legacy` |  |
| [`np.random.gumbel`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html#numpy.random.gumbel) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.hypergeometric`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.hypergeometric.html#numpy.random.hypergeometric) | OL |  |  | 7 |  |  | L: `size`: null -> `Shape` |
| [`np.random.laplace`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.laplace.html#numpy.random.laplace) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.logistic`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.logistic.html#numpy.random.logistic) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.lognormal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html#numpy.random.lognormal) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.logseries`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.logseries.html#numpy.random.logseries) | OL |  |  | 6 |  |  | L: `size`: null -> `Shape` |
| [`np.random.multinomial`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html#numpy.random.multinomial) | O |  |  | 4 |  |  |  |
| [`np.random.multivariate_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html#numpy.random.multivariate_normal) | OL |  |  | 5 |  |  | L: `size`: ambiguous: `NumPyRandom.multivariate_normal(double[], double[*,*], int[])` / `NumPyRandom.multivariate_normal(double[], double[*,*], long[])` (C# 12 only) |
| [`np.random.negative_binomial`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.noncentral_chisquare`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.noncentral_f`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_f.html#numpy.random.noncentral_f) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html#numpy.random.normal) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.pareto`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html#numpy.random.pareto) | OL |  |  | 6 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.permutation`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html#numpy.random.permutation) | O |  |  | 3 |  |  |  |
| [`np.random.poisson`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html#numpy.random.poisson) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.power`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.power.html#numpy.random.power) | OL |  |  | 6 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.rand`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html#numpy.random.rand) | O |  |  | 2 |  |  |  |
| [`np.random.randint`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint) | OL |  |  | 4 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.randn`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html#numpy.random.randn) | OT |  |  | 2 | `<T>()` |  |  |
| [`np.random.random`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html#numpy.random.random) | O |  |  | 2 |  | `C# requires size` |  |
| [`np.random.random_integers`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_integers.html#numpy.random.random_integers) | L |  |  | 1 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.random_sample`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random_sample.html#numpy.random.random_sample) | O |  |  | 2 |  | `C# requires size` |  |
| [`np.random.ranf`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.ranf.html#numpy.random.ranf) | O |  |  | 2 |  |  |  |
| [`np.random.rayleigh`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rayleigh.html#numpy.random.rayleigh) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.sample`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.sample.html#numpy.random.sample) | O |  |  | 2 |  |  |  |
| [`np.random.seed`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html#numpy.random.seed) | OL |  |  | 8 |  | `C# requires seed` | L: `seed`: ambiguous: `NumPyRandom.seed(uint[])` / `NumPyRandom.seed(long[])` (C# 12 only) |
| [`np.random.set_state`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.set_state.html#numpy.random.set_state) | O |  |  | 3 |  |  |  |
| [`np.random.standard_cauchy`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_cauchy.html#numpy.random.standard_cauchy) | OL |  |  | 2 |  | `C# requires size` | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.standard_exponential`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_exponential.html#numpy.random.standard_exponential) | OL |  |  | 2 |  | `C# requires size` | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.standard_gamma`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html#numpy.random.standard_gamma) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.standard_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html#numpy.random.standard_normal) | OL |  |  | 2 |  | `C# requires size` | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.standard_t`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html#numpy.random.standard_t) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.triangular`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html#numpy.random.triangular) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html#numpy.random.uniform) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.vonmises`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.vonmises.html#numpy.random.vonmises) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.wald`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.wald.html#numpy.random.wald) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.weibull`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html#numpy.random.weibull) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |
| [`np.random.zipf`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.zipf.html#numpy.random.zipf) | OL |  |  | 3 |  |  | L: `size`: null -> `Shape` (`default` binds) |

### Credited headline calls not compared (2)

| API | Credited to | Reason |
|---|---|---|
| `np.astype` | `NumSharp.NDArray.astype` | credited through NumSharp.NDArray.astype, whose receiver is an argument of the NumPy call or the other way round |
| `np.shape` | `NumSharp.NDArray.Shape` | credited to a property |

## Highest-priority gaps

| API | Surface | Status | Category |
|---|---|---|---|
| [`ndarray.data`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html#numpy.ndarray.data) | ndarray | partial | Array attributes |
| [`np.cumulative_prod`](https://numpy.org/doc/stable/reference/generated/numpy.cumulative_prod.html#numpy.cumulative_prod) | np | partial | Reductions |
| [`np.cumulative_sum`](https://numpy.org/doc/stable/reference/generated/numpy.cumulative_sum.html#numpy.cumulative_sum) | np | partial | Reductions |
| [`np.shape`](https://numpy.org/doc/stable/reference/generated/numpy.shape.html#numpy.shape) | np | partial | Shape manipulation |
| [`ndarray.ctypes`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html#numpy.ndarray.ctypes) | ndarray | missing | Array attributes |
| [`ndarray.dump`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dump.html#numpy.ndarray.dump) | ndarray | missing | Array methods |
| [`ndarray.dumps`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dumps.html#numpy.ndarray.dumps) | ndarray | missing | Array methods |
| [`np.busday_count`](https://numpy.org/doc/stable/reference/generated/numpy.busday_count.html#numpy.busday_count) | np | missing | Date & time |
| [`np.busday_offset`](https://numpy.org/doc/stable/reference/generated/numpy.busday_offset.html#numpy.busday_offset) | np | missing | Date & time |
| [`np.datetime_as_string`](https://numpy.org/doc/stable/reference/generated/numpy.datetime_as_string.html#numpy.datetime_as_string) | np | missing | Date & time |
| [`np.from_dlpack`](https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html#numpy.from_dlpack) | np | missing | Array creation |
| [`np.fromfunction`](https://numpy.org/doc/stable/reference/generated/numpy.fromfunction.html#numpy.fromfunction) | np | missing | Array creation |
| [`np.fromiter`](https://numpy.org/doc/stable/reference/generated/numpy.fromiter.html#numpy.fromiter) | np | missing | Array creation |
| [`np.fromregex`](https://numpy.org/doc/stable/reference/generated/numpy.fromregex.html#numpy.fromregex) | np | missing | Input & output |
| [`np.genfromtxt`](https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt) | np | missing | Input & output |
| [`np.get_include`](https://numpy.org/doc/stable/reference/generated/numpy.get_include.html#numpy.get_include) | np | missing | Runtime & diagnostics |
| [`np.geterr`](https://numpy.org/doc/stable/reference/generated/numpy.geterr.html#numpy.geterr) | np | missing | Floating-point handling |
| [`np.geterrcall`](https://numpy.org/doc/stable/reference/generated/numpy.geterrcall.html#numpy.geterrcall) | np | missing | Floating-point handling |
| [`np.info`](https://numpy.org/doc/stable/reference/generated/numpy.info.html#numpy.info) | np | missing | Runtime & diagnostics |
| [`np.is_busday`](https://numpy.org/doc/stable/reference/generated/numpy.is_busday.html#numpy.is_busday) | np | missing | Date & time |
| [`np.isnat`](https://numpy.org/doc/stable/reference/generated/numpy.isnat.html#numpy.isnat) | np | missing | Date & time |
| [`np.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.ndim.html#numpy.ndim) | np | missing | Shape manipulation |
| [`np.seterr`](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html#numpy.seterr) | np | missing | Floating-point handling |
| [`np.seterrcall`](https://numpy.org/doc/stable/reference/generated/numpy.seterrcall.html#numpy.seterrcall) | np | missing | Floating-point handling |
| [`np.show_config`](https://numpy.org/doc/stable/reference/generated/numpy.show_config.html#numpy.show_config) | np | missing | Runtime & diagnostics |
| [`np.show_runtime`](https://numpy.org/doc/stable/reference/generated/numpy.show_runtime.html#numpy.show_runtime) | np | missing | Runtime & diagnostics |
| [`np.test`](https://numpy.org/doc/stable/reference/routines.testing.html#module-numpy.testing) | np | missing | Runtime & diagnostics |

## Case-insensitive near-misses

NumPy's public API is case-sensitive, so a NumSharp member is credited only when the spelling matches exactly. The generator additionally folds case to surface near-misses — in-scope NumPy APIs left *missing* for which NumSharp exposes a same-surface member differing only by case. These are **not** counted as available; rename to the exact NumPy spelling (or record a reviewed alias) to close them.

_None detected._

## Extended NumPy submodules and object contracts

These surfaces appear in the dashboard by default. `candidate` denotes numeric APIs and object contracts; `subsystem` denotes a separate object model such as masked/string/record arrays, ufuncs or NumPy's scalar types; `tooling` denotes Python tooling. Class existence is catalogued separately from member availability. The historical headline remains unchanged by these additions.

| Surface | Disposition | Available | Missing | Total | Notable missing |
|---|---|---:|---:|---:|---|
| numpy.__array_namespace_info__ | candidate | 0 | 5 | 5 | `capabilities`, `default_device`, `default_dtypes`, `devices`, `dtypes` |
| numpy.broadcast | candidate | 9 | 1 | 10 | `__next__` |
| numpy.char.chararray | candidate | 127 | 65 | 192 | `__abs__`, `__and__`, `__array__`, `__array_finalize__`, `__array_function__`, `__array_interface__` … |
| numpy.dtype | candidate | 34 | 6 | 40 | `__bool__`, `__class_getitem__`, `__mul__`, `__reduce__`, `__rmul__`, `__setstate__` |
| numpy.dtypes | candidate | 30 | 3 | 33 | `Complex64DType`, `ObjectDType`, `VoidDType` |
| numpy.dtypes.StringDType | candidate | 0 | 12 | 12 | `__eq__`, `__ge__`, `__gt__`, `__hash__`, `__le__`, `__lt__` … |
| numpy.emath | candidate | 9 | 0 | 9 |  |
| numpy.errstate | candidate | 0 | 3 | 3 | `__call__`, `__enter__`, `__exit__` |
| numpy.exceptions | candidate | 3 | 4 | 7 | `ComplexWarning`, `ModuleDeprecationWarning`, `RankWarning`, `VisibleDeprecationWarning` |
| numpy.exceptions.AxisError | candidate | 0 | 3 | 3 | `__str__`, `axis`, `ndim` |
| numpy.finfo | candidate | 14 | 7 | 21 | `__class_getitem__`, `__str__`, `iexp`, `machep`, `negep`, `nexp` … |
| numpy.flatiter | candidate | 8 | 9 | 17 | `__array__`, `__delitem__`, `__eq__`, `__ge__`, `__gt__`, `__le__` … |
| numpy.iinfo | candidate | 6 | 3 | 9 | `__class_getitem__`, `__str__`, `key` |
| numpy.matrix | candidate | 122 | 42 | 164 | `__abs__`, `__array__`, `__array_finalize__`, `__array_function__`, `__array_interface__`, `__array_namespace__` … |
| numpy.ndarray | candidate | 41 | 43 | 84 | `__abs__`, `__array__`, `__array_finalize__`, `__array_function__`, `__array_interface__`, `__array_namespace__` … |
| numpy.ndarray.flags | candidate | 20 | 5 | 25 | `__delitem__`, `__ge__`, `__gt__`, `__le__`, `__lt__` |
| numpy.ndenumerate | candidate | 1 | 2 | 3 | `__next__`, `iter` |
| numpy.ndindex | candidate | 1 | 1 | 2 | `__next__` |
| numpy.nditer | candidate | 30 | 4 | 34 | `__copy__`, `__delitem__`, `__len__`, `__next__` |
| numpy.polynomial | candidate | 550 | 8 | 558 | `__array__`, `__iter__`, `__len__`, `__pow__`, `__radd__`, `__rsub__` … |
| numpy.random.BitGenerator | candidate | 5 | 6 | 11 | `__getstate__`, `__reduce__`, `__setstate__`, `capsule`, `cffi`, `ctypes` |
| numpy.random.Generator | candidate | 46 | 4 | 50 | `__getstate__`, `__reduce__`, `__repr__`, `__setstate__` |
| numpy.random.MT19937 | candidate | 6 | 6 | 12 | `__getstate__`, `__reduce__`, `__setstate__`, `capsule`, `cffi`, `ctypes` |
| numpy.random.PCG64 | candidate | 7 | 6 | 13 | `__getstate__`, `__reduce__`, `__setstate__`, `capsule`, `cffi`, `ctypes` |
| numpy.random.PCG64DXSM | candidate | 7 | 6 | 13 | `__getstate__`, `__reduce__`, `__setstate__`, `capsule`, `cffi`, `ctypes` |
| numpy.random.Philox | candidate | 7 | 6 | 13 | `__getstate__`, `__reduce__`, `__setstate__`, `capsule`, `cffi`, `ctypes` |
| numpy.random.RandomState | candidate | 50 | 4 | 54 | `__getstate__`, `__reduce__`, `__repr__`, `__setstate__` |
| numpy.random.SFC64 | candidate | 5 | 6 | 11 | `__getstate__`, `__reduce__`, `__setstate__`, `capsule`, `cffi`, `ctypes` |
| numpy.random.SeedSequence | candidate | 9 | 2 | 11 | `__reduce__`, `__setstate__` |
| numpy.vectorize | candidate | 2 | 4 | 6 | `cache`, `excluded`, `otypes`, `pyfunc` |
| numpy.busdaycalendar | subsystem | 0 | 2 | 2 | `holidays`, `weekmask` |
| numpy.char | subsystem | 53 | 66 | 119 | `add.accumulate`, `add.at`, `add.outer`, `add.reduce`, `add.reduceat`, `add.resolve_dtypes` … |
| numpy.generic | subsystem | 0 | 923 | 923 | `bool.__and__`, `bool.__bool__`, `bool.__buffer__`, `bool.__class_getitem__`, `bool.__eq__`, `bool.__ge__` … |
| numpy.ma | subsystem | 324 | 326 | 650 | `add.accumulate`, `add.outer`, `add.reduce`, `arctan2.accumulate`, `arctan2.outer`, `arctan2.reduce` … |
| numpy.memmap | subsystem | 0 | 155 | 155 | `__abs__`, `__add__`, `__and__`, `__array__`, `__array_finalize__`, `__array_function__` … |
| numpy.rec | subsystem | 0 | 9 | 9 | `array`, `find_duplicate`, `format_parser`, `fromarrays`, `fromfile`, `fromrecords` … |
| numpy.rec.format_parser | subsystem | 0 | 1 | 1 | `dtype` |
| numpy.recarray | subsystem | 0 | 155 | 155 | `__abs__`, `__add__`, `__and__`, `__array__`, `__array_finalize__`, `__array_function__` … |
| numpy.record | subsystem | 0 | 125 | 125 | `__abs__`, `__add__`, `__and__`, `__array__`, `__array_interface__`, `__array_namespace__` … |
| numpy.strings | subsystem | 46 | 102 | 148 | `add.accumulate`, `add.at`, `add.outer`, `add.reduce`, `add.reduceat`, `add.resolve_dtypes` … |
| numpy.ufunc | subsystem | 0 | 655 | 655 | `__call__`, `__doc__`, `__name__`, `__repr__`, `__signature__`, `__str__` … |
| numpy.ctypeslib | tooling | 0 | 6 | 6 | `as_array`, `as_ctypes`, `as_ctypes_type`, `c_intp`, `load_library`, `ndpointer` |
| numpy.f2py | tooling | 0 | 2 | 2 | `get_include`, `run_main` |
| numpy.lib.NumpyVersion | tooling | 0 | 15 | 15 | `NumpyVersion`, `__eq__`, `__ge__`, `__gt__`, `__le__`, `__lt__` … |
| numpy.typing | tooling | 3 | 0 | 3 |  |
| numpy.version | tooling | 0 | 6 | 6 | `full_version`, `git_revision`, `release`, `short_version`, `version`, `version` |


## Expanded capability categories

| Category | Available | Partial | Missing | Total |
|---|---:|---:|---:|---:|
| Array API introspection | 0 | 0 | 5 | 5 |
| Array attributes | 15 | 1 | 1 | 17 |
| Array creation | 29 | 0 | 3 | 32 |
| Array flags | 20 | 0 | 5 | 25 |
| Array iterators | 49 | 0 | 17 | 66 |
| Array metadata & memory | 5 | 0 | 0 | 5 |
| Array methods | 32 | 0 | 2 | 34 |
| Array protocols | 41 | 0 | 43 | 84 |
| Bitwise operations | 13 | 0 | 0 | 13 |
| Complex numbers | 6 | 0 | 0 | 6 |
| Complex-domain math | 9 | 0 | 0 | 9 |
| Date & time | 1 | 0 | 7 | 8 |
| Differences & integration | 6 | 0 | 0 | 6 |
| Dtype & promotion | 9 | 0 | 0 | 9 |
| Dtype classes | 0 | 0 | 12 | 12 |
| Dtype objects | 34 | 0 | 6 | 40 |
| Exceptions & warnings | 0 | 0 | 3 | 3 |
| Exponentials & logarithms | 9 | 0 | 0 | 9 |
| Floating-point handling | 9 | 0 | 7 | 16 |
| Fortran bindings | 0 | 0 | 2 | 2 |
| Fourier transforms | 18 | 0 | 0 | 18 |
| Function utilities | 1 | 0 | 0 | 1 |
| Function wrappers | 2 | 0 | 4 | 6 |
| Hyperbolic functions | 9 | 0 | 0 | 9 |
| Indexing & selection | 22 | 0 | 0 | 22 |
| Input & output | 8 | 0 | 2 | 10 |
| Legacy random state | 50 | 0 | 4 | 54 |
| Legacy string operations | 52 | 0 | 0 | 52 |
| Linear algebra | 50 | 0 | 0 | 50 |
| Logic & comparison | 29 | 0 | 0 | 29 |
| Machine limits | 20 | 0 | 10 | 30 |
| Masked arrays | 313 | 0 | 320 | 633 |
| Math | 30 | 0 | 0 | 30 |
| Matrix objects | 122 | 0 | 42 | 164 |
| Memory-mapped arrays | 0 | 0 | 155 | 155 |
| Polynomials | 517 | 0 | 8 | 525 |
| Random | 51 | 0 | 0 | 51 |
| Random bit generators | 37 | 0 | 36 | 73 |
| Random generators | 46 | 0 | 4 | 50 |
| Random seeding | 9 | 0 | 2 | 11 |
| Record arrays | 0 | 0 | 287 | 287 |
| Reductions | 48 | 2 | 0 | 50 |
| Rounding | 7 | 0 | 0 | 7 |
| Runtime & diagnostics | 0 | 0 | 19 | 19 |
| Scalar types | 0 | 0 | 923 | 923 |
| Set operations | 10 | 0 | 0 | 10 |
| Shape manipulation | 47 | 1 | 1 | 49 |
| Sorting & searching | 16 | 0 | 0 | 16 |
| Statistics & histograms | 9 | 0 | 0 | 9 |
| String array objects | 127 | 0 | 65 | 192 |
| String operations | 46 | 0 | 0 | 46 |
| Text & formatting | 11 | 0 | 0 | 11 |
| Trigonometric functions | 18 | 0 | 0 | 18 |
| Universal function objects | 0 | 0 | 823 | 823 |
| Window functions | 5 | 0 | 0 | 5 |
| ctypes interop | 0 | 0 | 5 | 5 |

## Counting rules

The historical headline is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. The expanded API scope adds public extended-submodule callables and object methods/properties/protocols: the Python protocols (dunder members) NumPy defines, the members of NumPy's scalar types, and the members of every other public NumPy class. Types, constants, modules, and NumSharp-only APIs remain searchable in the full catalog but do not enter either API denominator. Aliases are separate exported names; inherited members are separate contracts on each object type, while NumPy's scalar types and the classes no object surface owns list each member once, at the class that defines it. A ufunc method row records the exposed protocol even when that ufunc rejects the operation (for example, unary reductions), and a protocol row records a slot NumPy fills only to refuse it. Matching a namespace function never credits a ufunc or a different object type's member. APIs NumPy keeps only for NumPy 1.x code and has itself deprecated (numpy.matlib, numpy.typing.NBitBase) are left out; numpy.lib and numpy.testing are catalogued but not shown, except numpy.lib.NumpyVersion.
