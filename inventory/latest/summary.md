# NumPy ↔ NumSharp API coverage

Compared with NumPy **2.5.4** using NumSharp assembly **0.81.0.0**.

Expanded API availability (dashboard default): **70.3%** (2747 of 3908 APIs across 47 surfaces and 56 categories). **1160 missing**, **1 partial**. Another **942** catalogued APIs are listed but not counted: **358** calls NumPy itself refuses, and **584** members a class NumSharp has no type for (memmap, recarray, record, MaskedRecords) inherits unchanged from a class whose own rows count them.

The catalog includes **853 per-ufunc protocol methods** (each public `numpy.ufunc` method on every ufunc export of the compared namespaces, plus the methods numpy.ma's masked ufunc wrappers implement); **338** describe methods whose calls NumPy rejects for that ufunc: they stay listed, outside the percentages.

It also counts **1483 Python protocols** NumPy defines on its classes (operators, subscripting, conversions, iteration, `with`, pickling), each class listing the protocols it has; NumSharp provides **983**, through a member of the protocol's own name, the C# operator, indexer or cast of the same meaning, or a reviewed adapter. Not counted: **20** slots NumPy fills only to refuse every call (`del a[i]`, ordering comparisons on `ndarray.flags`), and **290** inherited unchanged by a class NumSharp has no type for.

Headline API availability: **96.2%** (538 of 559 default-scope APIs). Including partial mappings, **96.4%** are addressed.

| Surface | Available | Partial | Unsupported | Missing | Total | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| np.fft.* | 18 | 0 | 0 | 0 | 18 | 100.0% |
| np.linalg.* | 31 | 0 | 0 | 0 | 31 | 100.0% |
| ndarray.* | 66 | 1 | 0 | 3 | 70 | 94.3% |
| np.* | 362 | 0 | 0 | 17 | 379 | 95.5% |
| np.polynomial.* | 10 | 0 | 0 | 0 | 10 | 100.0% |
| np.random.* | 51 | 0 | 0 | 0 | 51 | 100.0% |

> Availability is based on the compiled public API: a NumPy API is available when NumSharp has a member of its name, or a reviewed alias names another member. It is not a blanket behavioral-parity claim; dtype, layout and edge-case parity require differential tests (the oracle section), and signature parity is measured separately (the signature section).

## NumPy-oracle parity evidence

**921,345** committed NumPy 2.5.4 oracle contracts (**109,418** error-message contracts) are replayed against NumSharp in CI; each must be bit-exact or a documented, registry-excused divergence. An API counts as **oracle-verified** when contracts resolve to it directly, or through an identity alias (the same NumPy object mapped to the same NumSharp member) or a reviewed pure-delegation alias.

Headline scope: **539** of 559 APIs oracle-verified (**96.4%**; 528 direct, 11 alias); **0** available APIs have no oracle contracts.

Expanded scope: **1767** of 3908 APIs oracle-verified (**45.2%**; 1709 direct, 58 alias); **981** available APIs have no oracle contracts.

| Source | Contracts | Error contracts | Host-pinned | Files | Keys | Replayed by |
|---|---:|---:|---:|---:|---:|---|
| Differential-fuzz corpus (`test/NumSharp.Tests.Oracle/Fuzz/corpus`) | 838,505 | 89,968 | 77,916 | 152 | 2371 | FuzzMatrix (FuzzCorpusTests*, IndexOracleTests) |
| .npy/.npz format oracle (`test/NumSharp.Tests/IO/corpus/npy_oracle.zip`) | 404 | 38 | 0 | 1 | 11 | NpyOracle (NpyOracleTests) |
| Array flags oracle (`test/NumSharp.Tests/Backends/corpus/flags_oracle.jsonl`) | 1,245 | 225 | 0 | 1 | 2 | FlagsOracleTests |
| Result-layout parity oracle (`test/NumSharp.Tests/Backends/corpus/layout_parity_oracle.jsonl`) | 568 | 0 | 0 | 1 | 69 | LayoutParityOracleTests |
| String oracle (numpy.strings and numpy.char calls) (`test/NumSharp.Tests.Oracle/Strings/corpus`) | 80,623 | 19,187 | 0 | 18 | 146 | StringOracleTests |

Host-pinned contracts are hard-gated only on the pinned host (win-amd64 CRT libm, or the content-hash-pinned OpenBLAS) and Inconclusive elsewhere.

Per surface (expanded scope; surfaces with nothing available or gated are omitted). *Verified* shares the coverage percentage's denominator: every API row of the surface.

| Surface | Total | Available | Oracle-verified | Direct | Alias | Unverified available | Verified |
|---|---:|---:|---:|---:|---:|---:|---:|
| broadcast | 10 | 9 | 8 | 8 | 0 | 1 | 80.0% |
| char | 78 | 52 | 50 | 15 | 35 | 2 | 64.1% |
| char.chararray | 191 | 129 | 81 | 81 | 0 | 48 | 42.4% |
| dtype | 40 | 34 | 22 | 22 | 0 | 12 | 55.0% |
| dtypes.StringDType | 8 | 7 | 0 | 0 | 0 | 7 | 0.0% |
| emath | 9 | 9 | 9 | 9 | 0 | 0 | 100.0% |
| np.fft.* | 18 | 18 | 18 | 18 | 0 | 0 | 100.0% |
| finfo | 21 | 19 | 13 | 13 | 0 | 6 | 61.9% |
| flatiter | 16 | 8 | 4 | 4 | 0 | 4 | 25.0% |
| generic | 922 | 700 | 0 | 0 | 0 | 700 | 0.0% |
| iinfo | 9 | 6 | 5 | 5 | 0 | 1 | 55.6% |
| np.linalg.* | 31 | 31 | 31 | 28 | 3 | 0 | 100.0% |
| np.ma.* | 466 | 318 | 215 | 203 | 12 | 103 | 46.1% |
| matrix | 163 | 122 | 85 | 85 | 0 | 37 | 52.1% |
| ndarray.* | 153 | 107 | 83 | 83 | 0 | 25 | 54.2% |
| ndarray.flags | 20 | 20 | 14 | 14 | 0 | 6 | 70.0% |
| ndenumerate | 3 | 1 | 0 | 0 | 0 | 1 | 0.0% |
| ndindex | 2 | 1 | 0 | 0 | 0 | 1 | 0.0% |
| nditer | 33 | 30 | 24 | 24 | 0 | 6 | 72.7% |
| np.* | 379 | 362 | 362 | 354 | 8 | 0 | 95.5% |
| np.polynomial.* | 531 | 523 | 510 | 510 | 0 | 13 | 96.0% |
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
| strings | 108 | 46 | 46 | 46 | 0 | 0 | 42.6% |
| vectorize | 6 | 2 | 0 | 0 | 0 | 2 | 0.0% |

### Available headline APIs without oracle contracts

Parity for these is declared, not oracle-proven (some carry dedicated unit suites instead; see OracleSurfaceCoverageTests' sibling-owned and known-gap classifications).

_None: every available headline API has oracle contracts._

### Oracle-gated APIs the catalog does not credit

The corpus proves NumSharp reproduces these NumPy behaviours, yet no NumSharp member is credited under the NumPy name (add the member, or record a reviewed alias in overrides.json).

_None._

## Signature parity

Availability credits a NumPy API by NAME. Here every credited API's NumPy signature is compared with every C# spelling of it (module functions, members, constructors, extension methods and the typed `NDArray<T>`), from the API diff gap map of `tools/api-parameter-parity` (docs/plans/API_DIFF_GAP.md), which the generator runs on the same tree: a reflection dump, NumPy's runtime signatures completed from its `.pyi` stubs and numpydoc sections, and a compile probe of NumPy-style calls at C# 12 and C# 14. The rules are docs/plans/api-parameter-parity.md's: a positional-only NumPy parameter is matched by slot, a positional-or-keyword one by name and slot, a keyword-only one by name; `*args` needs a `params` array, and keywords NumPy takes through `**kwargs` are not differences. A call is **identical** when it carries no mark. Marks: **M** a NumPy parameter no overload offers (a rename counts: NumPy's keyword does not compile; M ◦, only parameters the plan excludes such as `like`, `subok` or a ufunc's `casting` / `order` / `signature`, is not a difference), **X** a parameter NumPy does not have, **O** more than one overload (NumPy has one signature), **T** a dtype taken as `NPTypeCode` / `Type` instead of `DType` (T ◇: picked by a generic type argument, `np.zeros<T>(shape)`), **B** a NumPy-style call that does not compile (`np.linspace(0, 1)`, `a.transpose(1, 0)`; ◐ at C# 12 only), **L** `null` refused where NumPy's default is None (◐ C# 12 only; ◦ only a `Shape` refuses it and `default` binds), **Q** an integer sequence refused where NumPy's stubs allow one (`axis=(0, 1)`), **N** credited through another name (NumPy's spelling does not compile). The closest overload (the fewest edits from NumPy's) adds positional-order, required-parameter, default-value and `*args` differences. A credited row the gap map does not compare keeps the inventory's own reflection comparison and is not compile-probed.

Headline scope: **368** of 559 APIs have NumPy's exact signature (**65.8%**); of the 523 credited calls compared, **155** differ; 0 credited calls were not compared; 0 compared calls were not compile-probed.

Expanded scope: **1138** of 3908 APIs have NumPy's exact signature (**29.1%**); of the 1532 credited calls compared, **394** differ; 2 credited calls were not compared; 173 compared calls were not compile-probed.

| Mark | Meaning | Headline rows | Expanded rows |
|---|---|---:|---:|
| M | a NumPy parameter no overload offers | 10 | 51 |
| X | a parameter NumPy does not have | 18 | 49 |
| O | more than one overload | 139 | 315 |
| T ● | a dtype taken as NPTypeCode / Type | 5 | 14 |
| T ◇ | a generic type argument picks the dtype | 4 | 7 |
| B | a NumPy-style call does not compile | 3 | 6 |
| L | `null` refused for NumPy's None | 2 | 3 |
| Q | an integer sequence refused | 0 | 4 |
| N | credited through another name | 0 | 0 |
| order | a positional parameter at another slot (closest overload) | 7 | 18 |
| required | C# requires what NumPy defaults (closest overload) | 10 | 26 |
| default | another default value (closest overload) | 2 | 2 |
| varargs | `*args` without a `params` array (closest overload) | 6 | 8 |
| M ◦ | only plan-excluded parameters missing (not a difference) | 127 | 219 |
| unprobed | compared by the inventory's own reflection only (no compile probe) | 0 | 173 |

Per surface (expanded scope; surfaces without a credited call are omitted):

| Surface | APIs | Identical | Differs | Not compared | M | X | O | T | B | L | Q | N | Identical % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| broadcast | 10 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10.0% |
| char | 78 | 52 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 66.7% |
| char.chararray | 191 | 75 | 16 | 0 | 3 | 4 | 10 | 2 | 0 | 0 | 0 | 0 | 39.3% |
| dtype | 40 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7.5% |
| emath | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| np.fft.* | 18 | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 100.0% |
| flatiter | 16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6.2% |
| generic | 922 | 35 | 13 | 2 | 0 | 2 | 10 | 1 | 0 | 0 | 0 | 0 | 3.8% |
| np.linalg.* | 31 | 30 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 96.8% |
| np.ma.* | 466 | 222 | 50 | 0 | 28 | 24 | 15 | 0 | 3 | 1 | 3 | 0 | 47.6% |
| matrix | 163 | 36 | 30 | 0 | 10 | 1 | 17 | 2 | 0 | 0 | 1 | 0 | 22.1% |
| ndarray.* | 153 | 38 | 18 | 0 | 3 | 3 | 12 | 3 | 0 | 0 | 0 | 0 | 24.8% |
| nditer | 33 | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24.2% |
| np.* | 379 | 273 | 89 | 0 | 7 | 15 | 81 | 4 | 1 | 0 | 0 | 0 | 72.0% |
| np.polynomial.* | 531 | 256 | 47 | 0 | 0 | 0 | 41 | 6 | 0 | 0 | 0 | 0 | 48.2% |
| np.random.* | 51 | 6 | 45 | 0 | 0 | 0 | 43 | 1 | 2 | 2 | 0 | 0 | 11.8% |
| random.BitGenerator | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18.2% |
| random.Generator | 50 | 8 | 36 | 0 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 0 | 16.0% |
| random.MT19937 | 12 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25.0% |
| random.PCG64 | 13 | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 23.1% |
| random.PCG64DXSM | 13 | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 23.1% |
| random.Philox | 13 | 3 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 23.1% |
| random.RandomState | 54 | 6 | 43 | 0 | 0 | 0 | 43 | 1 | 0 | 0 | 0 | 0 | 11.1% |
| random.SFC64 | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18.2% |
| random.SeedSequence | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18.2% |
| strings | 108 | 43 | 3 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 39.8% |

### The API diff gap map in this artifact

The gap map (`tools/api-parameter-parity`, docs/plans/API_DIFF_GAP.md) counts C# spellings; the rows above count NumPy APIs. Every spelling with a NumPy counterpart joins the row of the API it spells (a constructor joins the class row, `NDArray<T>`'s members join `ndarray`'s, an alias or an inherited member adopts the spellings of the row with the same C# member and the same NumPy object); the columns count the same keys over the gap map and over the spellings the emitted rows carry. Compile probe: 9,118 calls, 150 fail at C# 12, 124 at C# 14.

| Count | Gap map | Carried by the artifact |
|---|---:|---:|
| C# spellings with a NumPy counterpart | 1341 | 1317 |
| M ● a NumPy parameter missing | 57 | 53 |
| M ◦ only plan-excluded parameters missing | 221 | 220 |
| X a parameter NumPy lacks | 50 | 50 |
| O more than one overload | 249 | 247 |
| T ● NPTypeCode / Type | 9 | 9 |
| T ◇ a generic dtype selector | 7 | 7 |
| B a NumPy-style call does not compile | 14 | 14 |
| B ◐ only at C# 12 | 0 | 0 |
| L ● null refused | 1 | 1 |
| L ◐ null refused at C# 12 only | 15 | 15 |
| L ◦ only Shape refuses null (default binds) | 0 | 0 |
| Q an integer sequence refused | 7 | 6 |
| no mark at all | 837 | 819 |

Spellings with a NumPy counterpart that no NumPy row carries:

| Spelling | Where | Reason |
|---|---|---|
| `np.random` / `get_bit_generator` | `numsharp.random.get_bit_generator` | NumPy has the name outside its __all__ and documentation; the catalog lists the NumSharp member as NumSharp-only |
| `np.random` / `set_bit_generator` | `numsharp.random.set_bit_generator` | NumPy has the name outside its __all__ and documentation; the catalog lists the NumSharp member as NumSharp-only |
| `np.lib.array_utils` / `byte_bounds` | not shown (`numpy.lib.array_utils.byte_bounds`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.array_utils` / `normalize_axis_index` | not shown (`numpy.lib.array_utils.normalize_axis_index`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.array_utils` / `normalize_axis_tuple` | not shown (`numpy.lib.array_utils.normalize_axis_tuple`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `descr_to_dtype` | not shown (`numpy.lib.format.descr_to_dtype`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `drop_metadata` | not shown (`numpy.lib.format.drop_metadata`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `dtype_to_descr` | not shown (`numpy.lib.format.dtype_to_descr`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `header_data_from_array_1_0` | not shown (`numpy.lib.format.header_data_from_array_1_0`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `magic` | not shown (`numpy.lib.format.magic`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `read_array` | not shown (`numpy.lib.format.read_array`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `read_array_header_1_0` | not shown (`numpy.lib.format.read_array_header_1_0`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `read_array_header_2_0` | not shown (`numpy.lib.format.read_array_header_2_0`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `read_magic` | not shown (`numpy.lib.format.read_magic`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `write_array` | not shown (`numpy.lib.format.write_array`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `write_array_header_1_0` | not shown (`numpy.lib.format.write_array_header_1_0`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.format` / `write_array_header_2_0` | not shown (`numpy.lib.format.write_array_header_2_0`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.stride_tricks` / `as_strided` | not shown (`numpy.lib.stride_tricks.as_strided`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.stride_tricks` / `sliding_window_view` | not shown (`numpy.lib.stride_tricks.sliding_window_view`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.npyio.NpzFile` / `close` | not shown (`numpy.lib.npyio.NpzFile.close`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.npyio.NpzFile` / `get` | not shown (`numpy.lib.npyio.NpzFile.get`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.npyio.NpzFile` / `items` | not shown (`numpy.lib.npyio.NpzFile.items`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.npyio.NpzFile` / `keys` | not shown (`numpy.lib.npyio.NpzFile.keys`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.npyio.NpzFile` / `values` | not shown (`numpy.lib.npyio.NpzFile.values`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.lib.npyio.NpzFile` / `(constructor)` | not shown (`numpy.lib.npyio.NpzFile`) | the dashboard projection leaves the row out (numpy.lib and numpy.testing are not shown, by the owner's rule) |
| `np.random.MT19937` / `_legacy_seeding` | not shown (`numpy.random.MT19937._legacy_seeding`) | NumPy's name is private (a leading underscore): the catalog lists public NumPy APIs only, while NumSharp exposes the member publicly |

### Headline APIs whose signature differs (155)

| API | Marks | Missing | NumSharp-only parameters | Overloads | NPTypeCode / Type / generic | Closest overload | Compile probe |
|---|---|---|---|---:|---|---|---|
| [`np.linalg.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html#numpy.linalg.tensordot) | O |  |  | 3 |  |  |  |
| [`ndarray.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmax.html#numpy.ndarray.argmax) | O |  |  | 2 |  |  |  |
| [`ndarray.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argmin.html#numpy.ndarray.argmin) | O |  |  | 2 |  |  |  |
| [`ndarray.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.argpartition.html#numpy.ndarray.argpartition) | O |  |  | 3 |  |  |  |
| [`ndarray.astype`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html#numpy.ndarray.astype) |  |  |  | 1 |  | `copy: NumPy #4, C# #3` |  |
| [`ndarray.choose`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.choose.html#numpy.ndarray.choose) | O |  |  | 3 |  |  |  |
| [`ndarray.conj`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conj.html#numpy.ndarray.conj) | X |  | `out` | 1 |  |  |  |
| [`ndarray.conjugate`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conjugate.html#numpy.ndarray.conjugate) | X |  | `out` | 1 |  |  |  |
| [`ndarray.getfield`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.getfield.html#numpy.ndarray.getfield) | OT |  |  | 2 | `<T>(int offset)` |  |  |
| [`ndarray.item`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html#numpy.ndarray.item) | OT |  |  | 10 | `<T>()`, `<T>(long index)`, `<T>(long i, long j)`, `<T>(long i, long j, long k)`, `<T>(long[] indices)` |  |  |
| [`ndarray.partition`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.partition.html#numpy.ndarray.partition) | O |  |  | 3 |  |  |  |
| [`ndarray.reshape`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.reshape.html#numpy.ndarray.reshape) | O |  |  | 3 |  |  |  |
| [`ndarray.resize`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.resize.html#numpy.ndarray.resize) | O |  |  | 3 |  |  |  |
| [`ndarray.searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.searchsorted.html#numpy.ndarray.searchsorted) | O |  |  | 3 |  |  |  |
| [`ndarray.setfield`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.setfield.html#numpy.ndarray.setfield) | MX | `val (C#: value)` | `value` | 1 |  |  |  |
| [`ndarray.std`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.std.html#numpy.ndarray.std) | M | `correction` |  | 1 |  |  |  |
| [`ndarray.tofile`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.tofile.html#numpy.ndarray.tofile) | O |  |  | 2 |  |  |  |
| [`ndarray.var`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.var.html#numpy.ndarray.var) | M | `correction` |  | 1 |  |  |  |
| [`ndarray.view`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.view.html#numpy.ndarray.view) | OT |  |  | 4 | `Type type`, `<T>()` | `C# requires dtype`, `C# requires type` |  |
| [`np.all`](https://numpy.org/doc/stable/reference/generated/numpy.all.html#numpy.all) | O |  |  | 2 |  |  |  |
| [`np.any`](https://numpy.org/doc/stable/reference/generated/numpy.any.html#numpy.any) | O |  |  | 2 |  |  |  |
| [`np.apply_along_axis`](https://numpy.org/doc/stable/reference/generated/numpy.apply_along_axis.html#numpy.apply_along_axis) | O |  |  | 2 |  |  |  |
| [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html#numpy.arange) | O |  |  | 14 |  |  |  |
| [`np.argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html#numpy.argmax) | O |  |  | 2 |  |  |  |
| [`np.argmin`](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html#numpy.argmin) | O |  |  | 2 |  |  |  |
| [`np.argpartition`](https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html#numpy.argpartition) | O |  |  | 3 |  |  |  |
| [`np.array`](https://numpy.org/doc/stable/reference/generated/numpy.array.html#numpy.array) | MXO | `object (C#: array, buffer, data, nd, scalar, strArray, value)`, `ndmax` | `array`, `buffer`, `data`, `nd`, `scalar`, `size`, `strArray`, `value` | 33 |  | `order: 'K' vs 'C'` |  |
| [`np.array2string`](https://numpy.org/doc/stable/reference/generated/numpy.array2string.html#numpy.array2string) | M | `formatter`, `legacy` |  | 1 |  |  |  |
| [`np.array_split`](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html#numpy.array_split) | XO |  | `indices` | 3 |  |  |  |
| [`np.asanyarray`](https://numpy.org/doc/stable/reference/generated/numpy.asanyarray.html#numpy.asanyarray) | O |  |  | 2 |  |  |  |
| [`np.asarray`](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html#numpy.asarray) | XO |  | `ndim` | 7 |  |  |  |
| [`np.ascontiguousarray`](https://numpy.org/doc/stable/reference/generated/numpy.ascontiguousarray.html#numpy.ascontiguousarray) | O |  |  | 2 |  |  |  |
| [`np.asfortranarray`](https://numpy.org/doc/stable/reference/generated/numpy.asfortranarray.html#numpy.asfortranarray) | O |  |  | 2 |  |  |  |
| [`np.asmatrix`](https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html#numpy.asmatrix) | O |  |  | 4 |  |  |  |
| [`np.atleast_1d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_1d.html#numpy.atleast_1d) | O |  |  | 4 |  |  |  |
| [`np.atleast_2d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html#numpy.atleast_2d) | O |  |  | 4 |  |  |  |
| [`np.atleast_3d`](https://numpy.org/doc/stable/reference/generated/numpy.atleast_3d.html#numpy.atleast_3d) | O |  |  | 4 |  |  |  |
| [`np.bmat`](https://numpy.org/doc/stable/reference/generated/numpy.bmat.html#numpy.bmat) | OB |  |  | 5 |  | `C# requires ldict` | B: `gdict` alone: needs `ldict` |
| [`np.broadcast_arrays`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_arrays.html#numpy.broadcast_arrays) | O |  |  | 2 |  |  |  |
| [`np.broadcast_to`](https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html#numpy.broadcast_to) | MXO | `array (C#: from)`, `shape (C#: against)` | `against`, `from` | 9 |  |  |  |
| [`np.can_cast`](https://numpy.org/doc/stable/reference/generated/numpy.can_cast.html#numpy.can_cast) | MXOT | `from_ (C#: from, value)` | `from`, `value` | 8 | `NPTypeCode from`, `NPTypeCode to`, `Type from`, `Type to` |  |  |
| [`np.choose`](https://numpy.org/doc/stable/reference/generated/numpy.choose.html#numpy.choose) | O |  |  | 3 |  |  |  |
| [`np.concat`](https://numpy.org/doc/stable/reference/generated/numpy.concat.html#numpy.concat) | O |  |  | 9 |  |  |  |
| [`np.concatenate`](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html#numpy.concatenate) | O |  |  | 9 |  |  |  |
| [`np.copyto`](https://numpy.org/doc/stable/reference/generated/numpy.copyto.html#numpy.copyto) | O |  |  | 4 |  |  |  |
| [`np.count_nonzero`](https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html#numpy.count_nonzero) | O |  |  | 2 |  |  |  |
| [`np.delete`](https://numpy.org/doc/stable/reference/generated/numpy.delete.html#numpy.delete) | O |  |  | 3 |  |  |  |
| [`np.dsplit`](https://numpy.org/doc/stable/reference/generated/numpy.dsplit.html#numpy.dsplit) | XO |  | `indices` | 2 |  |  |  |
| [`np.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html#numpy.einsum) | O |  |  | 3 |  | `order: ... vs 'K'`, `casting: ... vs "safe"`, `*operands as string subscripts, NDArray[] operands (no params array)` |  |
| [`np.einsum_path`](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html#numpy.einsum_path) | O |  |  | 3 |  | `C# requires optimize`, `*operands as string subscripts, NDArray[] operands (no params array)` |  |
| [`np.empty_like`](https://numpy.org/doc/stable/reference/generated/numpy.empty_like.html#numpy.empty_like) |  |  |  | 1 |  | `shape: NumPy #4, C# #3` |  |
| [`np.equal`](https://numpy.org/doc/stable/reference/generated/numpy.equal.html#numpy.equal) | O |  |  | 3 |  |  |  |
| [`np.float_power`](https://numpy.org/doc/stable/reference/generated/numpy.float_power.html#numpy.float_power) | O |  |  | 2 |  |  |  |
| [`np.floor_divide`](https://numpy.org/doc/stable/reference/generated/numpy.floor_divide.html#numpy.floor_divide) | O |  |  | 2 |  |  |  |
| [`np.fmod`](https://numpy.org/doc/stable/reference/generated/numpy.fmod.html#numpy.fmod) | O |  |  | 2 |  |  |  |
| [`np.frombuffer`](https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html#numpy.frombuffer) | XOT |  | `address`, `array`, `byteLength`, `dispose`, `memory`, `segment` | 18 | `NPTypeCode dtype` |  |  |
| [`np.fromfile`](https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html#numpy.fromfile) | XO |  | `stream` | 2 |  |  |  |
| [`np.frompyfunc`](https://numpy.org/doc/stable/reference/generated/numpy.frompyfunc.html#numpy.frompyfunc) | M | `identity` |  | 1 |  |  |  |
| [`np.fromstring`](https://numpy.org/doc/stable/reference/generated/numpy.fromstring.html#numpy.fromstring) | O |  |  | 2 |  |  |  |
| [`np.full_like`](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy.full_like) |  |  |  | 1 |  | `shape: NumPy #5, C# #4` |  |
| [`np.gradient`](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html#numpy.gradient) | O |  |  | 4 |  | `*varargs as object spacing (no params array)` |  |
| [`np.greater`](https://numpy.org/doc/stable/reference/generated/numpy.greater.html#numpy.greater) | O |  |  | 3 |  |  |  |
| [`np.greater_equal`](https://numpy.org/doc/stable/reference/generated/numpy.greater_equal.html#numpy.greater_equal) | O |  |  | 3 |  |  |  |
| [`np.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html#numpy.histogram) | O |  |  | 4 |  |  |  |
| [`np.histogram2d`](https://numpy.org/doc/stable/reference/generated/numpy.histogram2d.html#numpy.histogram2d) | O |  |  | 6 |  |  |  |
| [`np.histogram_bin_edges`](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges) | O |  |  | 4 |  |  |  |
| [`np.histogramdd`](https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html#numpy.histogramdd) | O |  |  | 8 |  |  |  |
| [`np.hsplit`](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html#numpy.hsplit) | XO |  | `indices` | 2 |  |  |  |
| [`np.hstack`](https://numpy.org/doc/stable/reference/generated/numpy.hstack.html#numpy.hstack) | O |  |  | 2 |  |  |  |
| [`np.insert`](https://numpy.org/doc/stable/reference/generated/numpy.insert.html#numpy.insert) | O |  |  | 3 |  |  |  |
| [`np.isdtype`](https://numpy.org/doc/stable/reference/generated/numpy.isdtype.html#numpy.isdtype) | XO |  | `kinds` | 13 |  |  |  |
| [`np.issubdtype`](https://numpy.org/doc/stable/reference/generated/numpy.issubdtype.html#numpy.issubdtype) | OT |  |  | 10 | `NPTypeCode arg1`, `NPTypeCode arg2`, `Type arg1`, `Type arg2` |  |  |
| [`np.less`](https://numpy.org/doc/stable/reference/generated/numpy.less.html#numpy.less) | O |  |  | 3 |  |  |  |
| [`np.less_equal`](https://numpy.org/doc/stable/reference/generated/numpy.less_equal.html#numpy.less_equal) | O |  |  | 3 |  |  |  |
| [`np.lexsort`](https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html#numpy.lexsort) | O |  |  | 2 |  |  |  |
| [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace) | O |  |  | 2 |  |  |  |
| [`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html#numpy.load) | XO |  | `bytes` | 3 |  |  |  |
| [`np.loadtxt`](https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html#numpy.loadtxt) | XO |  | `lines`, `reader`, `stream` | 4 |  |  |  |
| [`np.meshgrid`](https://numpy.org/doc/stable/reference/generated/numpy.meshgrid.html#numpy.meshgrid) | O |  |  | 3 |  | `*xi as NDArray x1, NDArray x2 (no params array)` |  |
| [`np.mod`](https://numpy.org/doc/stable/reference/generated/numpy.mod.html#numpy.mod) | O |  |  | 2 |  |  |  |
| [`np.nanargmax`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmax.html#numpy.nanargmax) | O |  |  | 2 |  |  |  |
| [`np.nanargmin`](https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html#numpy.nanargmin) | O |  |  | 2 |  |  |  |
| [`np.nested_iters`](https://numpy.org/doc/stable/reference/generated/numpy.nested_iters.html#numpy.nested_iters) | O |  |  | 2 |  |  |  |
| [`np.not_equal`](https://numpy.org/doc/stable/reference/generated/numpy.not_equal.html#numpy.not_equal) | O |  |  | 3 |  |  |  |
| [`np.ones_like`](https://numpy.org/doc/stable/reference/generated/numpy.ones_like.html#numpy.ones_like) |  |  |  | 1 |  | `shape: NumPy #4, C# #3` |  |
| [`np.pad`](https://numpy.org/doc/stable/reference/generated/numpy.pad.html#numpy.pad) | O |  |  | 10 |  |  |  |
| [`np.partition`](https://numpy.org/doc/stable/reference/generated/numpy.partition.html#numpy.partition) | O |  |  | 3 |  |  |  |
| [`np.piecewise`](https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html#numpy.piecewise) | O |  |  | 3 |  |  |  |
| [`np.place`](https://numpy.org/doc/stable/reference/generated/numpy.place.html#numpy.place) | O |  |  | 2 |  |  |  |
| [`np.printoptions`](https://numpy.org/doc/stable/reference/generated/numpy.printoptions.html#numpy.printoptions) | M | `formatter`, `legacy`, `override_repr` |  | 1 |  | `sign: NumPy #8, C# #7`, `floatmode: NumPy #9, C# #8` |  |
| [`np.putmask`](https://numpy.org/doc/stable/reference/generated/numpy.putmask.html#numpy.putmask) | O |  |  | 2 |  |  |  |
| [`np.ravel_multi_index`](https://numpy.org/doc/stable/reference/generated/numpy.ravel_multi_index.html#numpy.ravel_multi_index) | O |  |  | 3 |  |  |  |
| [`np.remainder`](https://numpy.org/doc/stable/reference/generated/numpy.remainder.html#numpy.remainder) | O |  |  | 2 |  |  |  |
| [`np.repeat`](https://numpy.org/doc/stable/reference/generated/numpy.repeat.html#numpy.repeat) | O |  |  | 2 |  |  |  |
| [`np.result_type`](https://numpy.org/doc/stable/reference/generated/numpy.result_type.html#numpy.result_type) | OT |  |  | 8 | `NPTypeCode[] types`, `NPTypeCode type1`, `NPTypeCode type2`, `Type type1`, `Type type2` |  |  |
| [`np.save`](https://numpy.org/doc/stable/reference/generated/numpy.save.html#numpy.save) | O |  |  | 4 |  |  |  |
| [`np.savetxt`](https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html#numpy.savetxt) | XO |  | `stream`, `writer` | 6 |  |  |  |
| [`np.savez`](https://numpy.org/doc/stable/reference/generated/numpy.savez.html#numpy.savez) | O |  |  | 8 |  | `*args as NDArray[] args (no params array)` |  |
| [`np.savez_compressed`](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html#numpy.savez_compressed) | O |  |  | 8 |  | `*args as NDArray[] args (no params array)` |  |
| [`np.searchsorted`](https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html#numpy.searchsorted) | O |  |  | 3 |  |  |  |
| [`np.set_printoptions`](https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html#numpy.set_printoptions) | M | `formatter`, `legacy`, `override_repr` |  | 1 |  | `sign: NumPy #8, C# #7`, `floatmode: NumPy #9, C# #8` |  |
| [`np.split`](https://numpy.org/doc/stable/reference/generated/numpy.split.html#numpy.split) | XO |  | `indices` | 3 |  |  |  |
| [`np.tensordot`](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html#numpy.tensordot) | O |  |  | 3 |  |  |  |
| [`np.unravel_index`](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html#numpy.unravel_index) | O |  |  | 2 |  |  |  |
| [`np.vsplit`](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html#numpy.vsplit) | XO |  | `indices` | 2 |  |  |  |
| [`np.vstack`](https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy.vstack) | O |  |  | 3 |  |  |  |
| [`np.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html#numpy.where) | O |  |  | 5 |  | `C# requires x`, `C# requires y` |  |
| [`np.zeros_like`](https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html#numpy.zeros_like) |  |  |  | 1 |  | `shape: NumPy #4, C# #3` |  |
| [`np.polyint`](https://numpy.org/doc/stable/reference/generated/numpy.polyint.html#numpy.polyint) | O |  |  | 2 |  |  |  |
| [`np.polyval`](https://numpy.org/doc/stable/reference/generated/numpy.polyval.html#numpy.polyval) | O |  |  | 2 |  |  |  |
| [`np.random.beta`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.beta.html#numpy.random.beta) | O |  |  | 3 |  |  |  |
| [`np.random.binomial`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html#numpy.random.binomial) | O |  |  | 3 |  |  |  |
| [`np.random.chisquare`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.chisquare.html#numpy.random.chisquare) | O |  |  | 3 |  |  |  |
| [`np.random.choice`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html#numpy.random.choice) | O |  |  | 3 |  |  |  |
| [`np.random.default_rng`](https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng) | OL |  |  | 13 |  | `C# requires seed` | L: `seed`: ambiguous: `NumPyRandom.default_rng(int[])` / `NumPyRandom.default_rng(long[])` (C# 12 only) |
| [`np.random.dirichlet`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.dirichlet.html#numpy.random.dirichlet) | O |  |  | 2 |  |  |  |
| [`np.random.exponential`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.exponential.html#numpy.random.exponential) | O |  |  | 3 |  |  |  |
| [`np.random.f`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.f.html#numpy.random.f) | O |  |  | 3 |  |  |  |
| [`np.random.gamma`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html#numpy.random.gamma) | O |  |  | 3 |  |  |  |
| [`np.random.geometric`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.geometric.html#numpy.random.geometric) | O |  |  | 3 |  |  |  |
| [`np.random.get_state`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html#numpy.random.get_state) | O |  |  | 2 |  | `C# requires legacy` |  |
| [`np.random.gumbel`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.gumbel.html#numpy.random.gumbel) | O |  |  | 3 |  |  |  |
| [`np.random.hypergeometric`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.hypergeometric.html#numpy.random.hypergeometric) | O |  |  | 3 |  |  |  |
| [`np.random.laplace`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.laplace.html#numpy.random.laplace) | O |  |  | 3 |  |  |  |
| [`np.random.logistic`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.logistic.html#numpy.random.logistic) | O |  |  | 3 |  |  |  |
| [`np.random.lognormal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.lognormal.html#numpy.random.lognormal) | O |  |  | 3 |  |  |  |
| [`np.random.logseries`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.logseries.html#numpy.random.logseries) | O |  |  | 2 |  |  |  |
| [`np.random.multinomial`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multinomial.html#numpy.random.multinomial) | O |  |  | 2 |  |  |  |
| [`np.random.multivariate_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.multivariate_normal.html#numpy.random.multivariate_normal) | O |  |  | 2 |  |  |  |
| [`np.random.negative_binomial`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.negative_binomial.html#numpy.random.negative_binomial) | O |  |  | 3 |  |  |  |
| [`np.random.noncentral_chisquare`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_chisquare.html#numpy.random.noncentral_chisquare) | O |  |  | 3 |  |  |  |
| [`np.random.noncentral_f`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.noncentral_f.html#numpy.random.noncentral_f) | O |  |  | 3 |  |  |  |
| [`np.random.normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html#numpy.random.normal) | O |  |  | 3 |  |  |  |
| [`np.random.pareto`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.pareto.html#numpy.random.pareto) | O |  |  | 3 |  |  |  |
| [`np.random.permutation`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.permutation.html#numpy.random.permutation) | O |  |  | 3 |  |  |  |
| [`np.random.poisson`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.poisson.html#numpy.random.poisson) | O |  |  | 3 |  |  |  |
| [`np.random.power`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.power.html#numpy.random.power) | O |  |  | 3 |  |  |  |
| [`np.random.randint`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html#numpy.random.randint) | O |  |  | 4 |  |  |  |
| [`np.random.randn`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html#numpy.random.randn) | OT |  |  | 2 | `<T>()` |  |  |
| [`np.random.ranf`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.ranf.html#numpy.random.ranf) | B |  |  | 1 |  |  | B: several `*args` values: no overload takes 2 arguments — `np.random.ranf(new Shape(2), new Shape(2))` |
| [`np.random.rayleigh`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.rayleigh.html#numpy.random.rayleigh) | O |  |  | 3 |  |  |  |
| [`np.random.sample`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.sample.html#numpy.random.sample) | B |  |  | 1 |  |  | B: several `*args` values: no overload takes 2 arguments — `np.random.sample(new Shape(2), new Shape(2))` |
| [`np.random.seed`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html#numpy.random.seed) | OL |  |  | 8 |  | `C# requires seed` | L: `seed`: ambiguous: `NumPyRandom.seed(uint[])` / `NumPyRandom.seed(long[])` (C# 12 only) |
| [`np.random.set_state`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.set_state.html#numpy.random.set_state) | O |  |  | 3 |  |  |  |
| [`np.random.standard_cauchy`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_cauchy.html#numpy.random.standard_cauchy) | O |  |  | 2 |  | `C# requires size` |  |
| [`np.random.standard_exponential`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_exponential.html#numpy.random.standard_exponential) | O |  |  | 2 |  | `C# requires size` |  |
| [`np.random.standard_gamma`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_gamma.html#numpy.random.standard_gamma) | O |  |  | 3 |  |  |  |
| [`np.random.standard_normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_normal.html#numpy.random.standard_normal) | O |  |  | 2 |  | `C# requires size` |  |
| [`np.random.standard_t`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.standard_t.html#numpy.random.standard_t) | O |  |  | 3 |  |  |  |
| [`np.random.triangular`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.triangular.html#numpy.random.triangular) | O |  |  | 3 |  |  |  |
| [`np.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html#numpy.random.uniform) | O |  |  | 3 |  |  |  |
| [`np.random.vonmises`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.vonmises.html#numpy.random.vonmises) | O |  |  | 3 |  |  |  |
| [`np.random.wald`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.wald.html#numpy.random.wald) | O |  |  | 3 |  |  |  |
| [`np.random.weibull`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html#numpy.random.weibull) | O |  |  | 3 |  |  |  |
| [`np.random.zipf`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.zipf.html#numpy.random.zipf) | O |  |  | 3 |  |  |  |

### Credited headline calls not compared (0)

_None._

## Highest-priority gaps

| API | Surface | Status | Category |
|---|---|---|---|
| [`ndarray.data`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html#numpy.ndarray.data) | ndarray | partial | Array attributes |
| [`ndarray.ctypes`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.ctypes.html#numpy.ndarray.ctypes) | ndarray | missing | Array attributes |
| [`ndarray.dump`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dump.html#numpy.ndarray.dump) | ndarray | missing | Array methods |
| [`ndarray.dumps`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.dumps.html#numpy.ndarray.dumps) | ndarray | missing | Array methods |
| [`np.busday_count`](https://numpy.org/doc/stable/reference/generated/numpy.busday_count.html#numpy.busday_count) | np | missing | Date & time |
| [`np.busday_offset`](https://numpy.org/doc/stable/reference/generated/numpy.busday_offset.html#numpy.busday_offset) | np | missing | Date & time |
| [`np.datetime_as_string`](https://numpy.org/doc/stable/reference/generated/numpy.datetime_as_string.html#numpy.datetime_as_string) | np | missing | Date & time |
| [`np.from_dlpack`](https://numpy.org/doc/stable/reference/generated/numpy.from_dlpack.html#numpy.from_dlpack) | np | missing | Array creation |
| [`np.fromregex`](https://numpy.org/doc/stable/reference/generated/numpy.fromregex.html#numpy.fromregex) | np | missing | Input & output |
| [`np.genfromtxt`](https://numpy.org/doc/stable/reference/generated/numpy.genfromtxt.html#numpy.genfromtxt) | np | missing | Input & output |
| [`np.get_include`](https://numpy.org/doc/stable/reference/generated/numpy.get_include.html#numpy.get_include) | np | missing | Runtime & diagnostics |
| [`np.geterr`](https://numpy.org/doc/stable/reference/generated/numpy.geterr.html#numpy.geterr) | np | missing | Floating-point handling |
| [`np.geterrcall`](https://numpy.org/doc/stable/reference/generated/numpy.geterrcall.html#numpy.geterrcall) | np | missing | Floating-point handling |
| [`np.info`](https://numpy.org/doc/stable/reference/generated/numpy.info.html#numpy.info) | np | missing | Runtime & diagnostics |
| [`np.is_busday`](https://numpy.org/doc/stable/reference/generated/numpy.is_busday.html#numpy.is_busday) | np | missing | Date & time |
| [`np.isnat`](https://numpy.org/doc/stable/reference/generated/numpy.isnat.html#numpy.isnat) | np | missing | Date & time |
| [`np.seterr`](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html#numpy.seterr) | np | missing | Floating-point handling |
| [`np.seterrcall`](https://numpy.org/doc/stable/reference/generated/numpy.seterrcall.html#numpy.seterrcall) | np | missing | Floating-point handling |
| [`np.show_config`](https://numpy.org/doc/stable/reference/generated/numpy.show_config.html#numpy.show_config) | np | missing | Runtime & diagnostics |
| [`np.show_runtime`](https://numpy.org/doc/stable/reference/generated/numpy.show_runtime.html#numpy.show_runtime) | np | missing | Runtime & diagnostics |
| [`np.test`](https://numpy.org/doc/stable/reference/routines.testing.html#module-numpy.testing) | np | missing | Runtime & diagnostics |

## Case-insensitive near-misses

NumPy's public API is case-sensitive, so a NumSharp member is credited only when the spelling matches exactly. The generator additionally folds case to surface near-misses — NumPy APIs left *missing* for which NumSharp exposes a member differing only by case on the same surface (a module facade, or the owner type of an object, class or scalar member). These are **not** counted as available; rename to the exact NumPy spelling (or record a reviewed alias) to close them.

| NumPy API | Surface | Differs only by case from |
|---|---|---|
| [`np.exceptions.AxisError.axis`](https://numpy.org/doc/stable/reference/generated/numpy.exceptions.AxisError.html#numpy.exceptions.AxisError) | exceptions.AxisError | `NumSharp.AxisError.Axis` |
| [`np.exceptions.AxisError.ndim`](https://numpy.org/doc/stable/reference/generated/numpy.exceptions.AxisError.html#numpy.exceptions.AxisError) | exceptions.AxisError | `NumSharp.AxisError.NDim` |

## Extended NumPy submodules and object contracts

These surfaces appear in the dashboard by default. `candidate` denotes numeric APIs and object contracts; `subsystem` denotes a separate object model such as masked/string/record arrays, ufuncs or NumPy's scalar types; `tooling` denotes Python tooling. Class existence is catalogued separately from member availability. The historical headline remains unchanged by these additions.

| Surface | Disposition | Available | Missing | Total | Notable missing |
|---|---|---:|---:|---:|---|
| numpy.__array_namespace_info__ | candidate | 0 | 5 | 5 | `capabilities`, `default_device`, `default_dtypes`, `devices`, `dtypes` |
| numpy.broadcast | candidate | 9 | 1 | 10 | `__next__` |
| numpy.char.chararray | candidate | 129 | 63 | 192 | `__abs__`, `__and__`, `__array__`, `__array_finalize__`, `__array_function__`, `__array_interface__` … |
| numpy.dtype | candidate | 34 | 6 | 40 | `__bool__`, `__class_getitem__`, `__mul__`, `__reduce__`, `__rmul__`, `__setstate__` |
| numpy.dtypes | candidate | 30 | 4 | 34 | `Complex64DType`, `ObjectDType`, `register_dlpack_dtype`, `VoidDType` |
| numpy.dtypes.StringDType | candidate | 7 | 5 | 12 | `__ge__`, `__gt__`, `__le__`, `__lt__`, `__reduce__` |
| numpy.emath | candidate | 9 | 0 | 9 |  |
| numpy.errstate | candidate | 0 | 3 | 3 | `__call__`, `__enter__`, `__exit__` |
| numpy.exceptions | candidate | 3 | 4 | 7 | `ComplexWarning`, `ModuleDeprecationWarning`, `RankWarning`, `VisibleDeprecationWarning` |
| numpy.exceptions.AxisError | candidate | 0 | 3 | 3 | `__str__`, `axis`, `ndim` |
| numpy.finfo | candidate | 19 | 2 | 21 | `__class_getitem__`, `__str__` |
| numpy.flatiter | candidate | 8 | 9 | 17 | `__array__`, `__delitem__`, `__eq__`, `__ge__`, `__gt__`, `__le__` … |
| numpy.generic | candidate | 700 | 223 | 923 | `bool.__buffer__`, `bool.__class_getitem__`, `byte.__abs__`, `byte.__buffer__`, `byte.__class_getitem__`, `byte.__divmod__` … |
| numpy.iinfo | candidate | 6 | 3 | 9 | `__class_getitem__`, `__str__`, `key` |
| numpy.matrix | candidate | 122 | 42 | 164 | `__abs__`, `__array__`, `__array_finalize__`, `__array_function__`, `__array_interface__`, `__array_namespace__` … |
| numpy.ndarray | candidate | 41 | 43 | 84 | `__abs__`, `__array__`, `__array_finalize__`, `__array_function__`, `__array_interface__`, `__array_namespace__` … |
| numpy.ndarray.flags | candidate | 20 | 5 | 25 | `__delitem__`, `__ge__`, `__gt__`, `__le__`, `__lt__` |
| numpy.ndenumerate | candidate | 1 | 2 | 3 | `__next__`, `iter` |
| numpy.ndindex | candidate | 1 | 1 | 2 | `__next__` |
| numpy.nditer | candidate | 30 | 4 | 34 | `__copy__`, `__delitem__`, `__len__`, `__next__` |
| numpy.polynomial | candidate | 556 | 8 | 564 | `__array__`, `__iter__`, `__len__`, `__pow__`, `__radd__`, `__rsub__` … |
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
| numpy.ma | subsystem | 329 | 321 | 650 | `add.accumulate`, `add.outer`, `add.reduce`, `arctan2.accumulate`, `arctan2.outer`, `arctan2.reduce` … |
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
| Array creation | 31 | 0 | 1 | 32 |
| Array flags | 20 | 0 | 0 | 20 |
| Array iterators | 49 | 0 | 15 | 64 |
| Array metadata & memory | 5 | 0 | 0 | 5 |
| Array methods | 32 | 0 | 2 | 34 |
| Array protocols | 41 | 0 | 42 | 83 |
| Bitwise operations | 13 | 0 | 0 | 13 |
| Complex numbers | 6 | 0 | 0 | 6 |
| Complex-domain math | 9 | 0 | 0 | 9 |
| Date & time | 1 | 0 | 7 | 8 |
| Differences & integration | 6 | 0 | 0 | 6 |
| Dtype & promotion | 9 | 0 | 0 | 9 |
| Dtype classes | 7 | 0 | 2 | 9 |
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
| Machine limits | 25 | 0 | 5 | 30 |
| Masked arrays | 318 | 0 | 148 | 466 |
| Math | 30 | 0 | 0 | 30 |
| Matrix objects | 122 | 0 | 41 | 163 |
| Memory-mapped arrays | 0 | 0 | 5 | 5 |
| Polynomials | 523 | 0 | 8 | 531 |
| Random | 51 | 0 | 0 | 51 |
| Random bit generators | 37 | 0 | 36 | 73 |
| Random generators | 46 | 0 | 4 | 50 |
| Random seeding | 9 | 0 | 2 | 11 |
| Record arrays | 0 | 0 | 15 | 15 |
| Reductions | 50 | 0 | 0 | 50 |
| Rounding | 7 | 0 | 0 | 7 |
| Runtime & diagnostics | 0 | 0 | 19 | 19 |
| Scalar types | 700 | 0 | 222 | 922 |
| Set operations | 10 | 0 | 0 | 10 |
| Shape manipulation | 48 | 0 | 0 | 48 |
| Sorting & searching | 16 | 0 | 0 | 16 |
| Statistics & histograms | 9 | 0 | 0 | 9 |
| String array objects | 129 | 0 | 62 | 191 |
| String operations | 46 | 0 | 0 | 46 |
| Text & formatting | 11 | 0 | 0 | 11 |
| Trigonometric functions | 18 | 0 | 0 | 18 |
| Universal function objects | 0 | 0 | 485 | 485 |
| Window functions | 5 | 0 | 0 | 5 |
| ctypes interop | 0 | 0 | 5 | 5 |

## Counting rules

The historical headline is NumPy top-level callables, ndarray public methods/properties, and callables in numpy.random, numpy.linalg, and numpy.fft. The expanded API scope adds public extended-submodule callables and object methods/properties/protocols: the Python protocols (dunder members) NumPy defines, the members of NumPy's scalar types, and the members of every other public NumPy class. Types, constants, modules, and NumSharp-only APIs remain searchable in the full catalog but do not enter either API denominator. Aliases are separate exported names; inherited members are separate contracts on each object type, while NumPy's scalar types and the classes no object surface owns list each member once, at the class that defines it. A ufunc method row records the exposed protocol even when that ufunc rejects the operation (for example, unary reductions), and a protocol row records a slot NumPy fills only to refuse it. Matching a namespace function never credits a ufunc or a different object type's member. APIs NumPy keeps only for NumPy 1.x code and has itself deprecated (numpy.matlib, numpy.typing.NBitBase) are left out; numpy.lib and numpy.testing are catalogued but not shown, except numpy.lib.NumpyVersion.
