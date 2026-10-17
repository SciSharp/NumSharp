# API diff gap: NumSharp's `np.*` signatures against NumPy 2.4.2

Living worklist. Branch `strings`. Created 2026-10-17.

**Request (verbatim):** "Kindly create in docs/plans/API_DIFF_GAP.md covering all those with explanations and
instructions". "All those" are the findings of the two signature passes of 2026-10-17 (`964986b1` / `d2973c00`, then
the second pass `596b9a83`): every NumSharp name compared with NumPy 2.4.2, the marks it carries, the NumPy names
NumSharp lacks and the NumPy modules without a facade.

**This document** explains each kind of gap, gives the procedure that fixes one, and lists every affected name in a
work package (`AD1` to `AD10`, §5). §6 and §7 classify what is absent. §8 collects the decisions only the owner can
make, §9 suggests an order, §10 says how to regenerate the data and prove a fix. The appendices list every name with
no mark and every NumSharp-only name, so nothing in the inventory is left out.

**Relation to the other plans.** [`api-parameter-parity.md`](api-parameter-parity.md) is the living plan with
numbered items (A, B, C, ... X) and the state log; where a work package here overlaps one of its items, the item is
named (`B1`, `D3`, `G4`, ...). Fix work is recorded there, item by item. This file is the complete map of the gap;
it is updated when the data is regenerated (§10), not per fix.

**Data snapshot.** Every count and list below comes from
[`numpy-signature-diff.csv`](numpy-signature-diff.csv) and [`numpy-signature-marks.md`](numpy-signature-marks.md) as
regenerated on 2026-10-17 on branch `strings` after `ffa74606` (the latest change to a public signature then: the
`np.char` padding and partition functions). When the tree changes, the generated files are the truth; regenerate them
(§10) before acting on a list here.

---

## Contents

1. [What is measured](#1-what-is-measured)
2. [The marks, explained](#2-the-marks-explained)
3. [Rules every fix follows](#3-rules-every-fix-follows)
4. [Procedure for one name or one family](#4-procedure-for-one-name-or-one-family)
5. [Work packages](#5-work-packages) (AD1 to AD10)
6. [NumPy names NumSharp lacks](#6-numpy-names-numsharp-lacks)
7. [NumPy modules without a facade](#7-numpy-modules-without-a-facade)
8. [Decisions for the owner](#8-decisions-for-the-owner)
9. [Suggested order of work](#9-suggested-order-of-work)
10. [Regenerating the data and proving a fix](#10-regenerating-the-data-and-proving-a-fix)
11. [Limits of the marks](#11-limits-of-the-marks)
- [Appendix A: counts per group](#appendix-a-counts-per-group)
- [Appendix B: names with no mark](#appendix-b-names-with-no-mark)
- [Appendix C: NumSharp-only names](#appendix-c-numsharp-only-names)

---

## 1. What is measured

Three tools in `tools/api-parameter-parity/` produce the data:

| Tool | What it does |
|---|---|
| `dump_modules.cs` | Builds NumSharp.Core (Debug) and reflects every public overload of every `[ModuleName]` facade (`np`, `ndarray`, `np.linalg`, `np.fft`, `np.random`, `np.ma`, `np.emath`, `np.char`, `np.strings`, `np.dtypes`, `np.typing`, `np.polynomial` and its 7 submodules), the object API `np.random.Generator`, and (second output) every exported type: constructors, extension methods, the typed `NDArray<T>`. |
| `signature_diff.py` | Compares every name with NumPy 2.4.2's signature under the matching rules of [`api-parameter-parity.md`](api-parameter-parity.md) §1. NumPy's side is its runtime signature, completed from NumPy's `.pyi` type stubs and numpydoc `Parameters` sections where the runtime signature forwards `(*args, **kwargs)`. NumPy's other classes are compared through the C# type that implements each (`np.matrix` -> `NDMatrix`, `np.char.chararray` -> `NDCharArray`, `np.ma.MaskedArray` -> `NDMaskedArray`, `np.dtype` -> `DType`, `np.nditer`, `np.broadcast`, `np.flatiter`, `np.ndindex`, `np.ndenumerate`, `np.poly1d`, `np.vectorize`, `np.finfo`, `np.iinfo`, `NpzFile`, `SeedSequence`, `BitGenerator` and the five engines). |
| `binding_probe.py` | Writes NumPy-style C# calls of every compared name (7,137 calls) and compiles them against the NumSharp.dll the dump came from, at C# 12 (the default of a `net8.0` consumer, which ignores `[OverloadResolutionPriority]`) and C# 14. Reflection can say that a parameter exists in some overload; only the compiler says whether a NumPy call, written the way a port writes it, binds. |

Outputs (all generated, never edited by hand):

- [`numpy-signature-marks.md`](numpy-signature-marks.md): every name once, its marks, NumPy's signature, the missing and
  foreign parameters, the dtype parameters, and the compile probe's findings with the compiler's reason. **Look a name
  up here.**
- [`numpy-signature-diff.md`](numpy-signature-diff.md): per group, the closest overload's differences, every name with
  more than one overload with ALL its overloads, the identical names.
- [`numpy-signature-diff.csv`](numpy-signature-diff.csv): one row per name with every column (marks, missing, foreign,
  probe texts, all overloads).

Totals of this snapshot:

| | count |
|---|---|
| C# names / overloads | 1,359 / 2,414 |
| names with a NumPy counterpart | 1,191 |
| names NumPy does not have (NumSharp-only) | 168 |
| NumPy names a group lacks | 192 |
| NumPy public modules without a facade | 15 (140 names) |
| compile probes | 7,137 (455 fail at C# 14, 488 at C# 12) |

Marks over the 1,191 names with a NumPy counterpart:

| mark | names | work package |
|---|---|---|
| **M ●** a NumPy parameter is missing | 234 | AD2 (renames), AD3 (missing) |
| **M ◦** only plan-excluded parameters are missing | 132 | AD4 |
| **X** a parameter NumPy does not have | 90 | AD5 |
| **O** more than one overload | 360 (+63 NumSharp-only names) | AD1 |
| **T ●** a dtype as `NPTypeCode` / `Type` | 10 (+8 NumSharp-only) | AD6 |
| **T ◇** a generic type argument picks the dtype | 17 (+18 NumSharp-only) | AD7 |
| **B** a NumPy-style call does not compile | 45 | AD8 |
| **L ●** `null` refused for NumPy's None | 31 | AD9 |
| **L ◐** the same, at C# 12 only | 19 | AD9 |
| **L ◦** only `Shape` refuses `null`; `default` binds | 73 | AD9 |
| **Q** an integer sequence refused | 77 | AD10 |
| no mark at all | 508 | Appendix B |

## 2. The marks, explained

Each mark looks at ALL overloads of a name together (extension methods included). Why each matters for a port of
NumPy code, and what fixing it means:

**M ● (missing parameter).** NumPy has a parameter that no C# overload offers, so a NumPy call that passes it does not
compile. A **rename** counts: `np.transpose(a, axes=(1, 0))` ported as `np.transpose(a, axes: new[] { 1, 0 })` fails
because the C# parameter is called `premute`. The marks file shows a rename as `axes (C#: premute)`. A rename is
detected by slot: a NumPy parameter missing by name with a C# parameter of another name at its position. Most are
true renames (`x` for `a`), some are a different parameter at NumPy's slot (`fill_value` where `np.ma.amax` has
`out`); AD2 marks those.

**M ◦ (excluded only).** The only missing parameters are ones the plan excludes (§3 of the parity plan: `like=`,
`subok=`, a ufunc's `casting=` / `order=` / `signature=`, the sorts' field `order=`, `einsum_path`'s `einsum_call=`).
Not a gap under the current scope.

**X (foreign parameter).** A C# overload has a parameter NumPy does not have. Under the owner's rule (one NumPy
signature, remove the rest) such a parameter goes, or it is the C# half of a rename (fixed with AD2). Names at a
positional-only slot, C# spellings of `*args`, and keywords NumPy accepts through `**kwargs` are not counted.

**O n (overloads).** The name has n public C# overloads where NumPy has one signature. The owner's rule (2026-10-17):
"We must have one whole api per function as numpy provides. b. remove others." Extra overloads also only coexist
through `[OverloadResolutionPriority]`, which C# 12 ignores, so they turn into CS0121 errors for every default
`net8.0` consumer.

**T ● (NPTypeCode / Type).** An overload takes a dtype as `NPTypeCode` or `Type` (nullable and array forms included)
instead of `DType`. Since the dtype system's Stage B, `DType` is the one dtype spelling and converts implicitly from
`Type`, `NPTypeCode` and a dtype string, so a single `DType` parameter accepts every spelling.

**T ◇ (generic selector).** A generic overload whose type argument picks the dtype and is used by no parameter
(`np.zeros<T>(shape)`, `np.iinfo<T>()`): the caller names a CLR type instead of passing a `DType`. It is one more
overload (O) of a name NumPy spells with `dtype=`.

**B ● (does not bind).** A NumPy-style call does not compile, although every parameter may exist somewhere. The probe
tries, per name: the required parameters only; the optional positional ones positionally in NumPy's order; each
optional parameter alone by name; the dtype as a string; several values at NumPy's `*args` (`a.transpose(1, 0)`,
`np.meshgrid(x, y, z)`); the call without a parameter NumPy marks "not passed" (`np._NoValue`) where every C# overload
requires it (`np.ma.where(condition)`; `np.vectorize()`, NumPy's decorator form, is exempt); all NumPy parameters
together. Typical causes: a parameter lives only in an overload that requires another one (`np.empty(shape, device:
...)` needs `dtype`), C# requires a parameter NumPy defaults (`np.linspace(start, stop)` needs `num`), a C# overload
orders the parameters differently from NumPy (`np.std(a, 1, np.float64)` lands `np.float64` on `keepdims`), NumPy's
`*args` is a C# array without `params` (`a.transpose(1, 0)` has no overload; only `a.transpose(new[] { 1, 0 })`
binds). **B ◐** would be a call that compiles only at C# 14 (none in this snapshot).

**L (`null` refused).** NumPy's default is None, and the C# call with `null` does not compile:
- **L ●** at both language versions: the C# parameter is a value type (`int axis`, `char order`, `bool density`), or
  two overloads take `null` equally well (`int? axis` and `int[] axis`: CS0121).
- **L ◐** only at C# 12: the overloads are ranked with `[OverloadResolutionPriority]`, which C# 12 ignores.
- **L ◦** only a `Shape` parameter refuses `null`, and `default` binds (`size: default`, NumSharp's documented
  spelling of NumPy's `size=None`).

**Q (sequence refused).** NumPy's stubs let the parameter take a sequence of integers (`axis=(0, 1)`, `shift=(1, 2)`),
and neither `new[] { 0, 1 }` nor the C# tuple `(0, 1)` binds. Almost always a reduction that supports one axis only.

**★ in reports and the `+ name (stub)` / `(doc)` notes**: parameters NumPy's runtime signature hides, read from the
stubs or the docstring; they count like any other NumPy parameter.

## 3. Rules every fix follows

1. **One NumPy signature per name** (owner, 2026-10-17): one public overload with NumPy's whole signature; every other
   form is removed. Applied so far to the max/min family (`795560d5`): `np.amin` / `amax` / `min` / `max` /
   `nanmin` / `nanmax`, `ndarray.min` / `max`.
2. **NumPy is the source of truth**; breaking changes that match NumPy are acceptable (`.claude/CLAUDE.md`).
3. **Names and positions**: NumPy's parameter names verbatim (snake_case; a C# keyword gets `@`: `@out`, `@object`;
   NumPy's `from_` stays `from_`). Positional-or-keyword parameters at NumPy's positions. Keyword-only parameters
   after them in NumPy's order (C# cannot make a parameter keyword-only; a port names them anyway).
4. **Defaults**: NumPy's None is C#'s `null` on a nullable type (`int? axis = null`, `DType dtype = null`,
   `NDArray @out = null`). NumPy's `np._NoValue` ("not passed") is a C# default that the implementation reads as "not
   given". A value default is the same value.
5. **Types**: a dtype is `DType`; a shape is `Shape` (`Shape?` where None is allowed); an array_like is `NDArray`
   (C# arrays and scalars convert implicitly) or `object` where a Python sequence must be accepted (decision OD2); an
   axis follows decision OD1; a file follows OD3. (The `OD` decisions are listed in §8; plain `B1`, `D3`, `G4` ... are
   items of `api-parameter-parity.md`.)
6. **Every changed member is documented in full**: XML `<summary>` that says what the parameter does and its NumPy
   semantics, one `<param>` per parameter, `<returns>`, every `<exception>`; `<remarks>` where behavior has
   consequences (house rule for all code).
7. **C# 12 must bind**: no rule may depend on `[OverloadResolutionPriority]` (an external `net8.0` consumer compiles
   at C# 12). The one-signature rule removes the need for it.

## 4. Procedure for one name or one family

Follow this for every item of §5. It is the procedure that worked for the max/min family (`795560d5`).

1. **Read the row.** In [`numpy-signature-marks.md`](numpy-signature-marks.md): the marks, NumPy's signature, the
   missing / foreign parameters, the probe column. In [`numpy-signature-diff.md`](numpy-signature-diff.md): every
   overload of the name and the closest one.
2. **Probe NumPy 2.4.2** with the calls the port must accept: positional, by keyword, with None, with a tuple where
   the stubs allow one. Record the outputs. A behavior change (not only a signature change) follows the parity plan's
   cycle: prove the divergence first, then fix (`api-parameter-parity.md` §2).
3. **Write down the one C# signature** under §3. Check it against the probe kinds: required only, positional in
   NumPy's order, each parameter alone by name, `null` for every None default, several values at `*args`, the call
   without a "not passed" parameter, all parameters together.
4. **Census the callers before deleting anything.** Mark every overload that will go
   `[System.Obsolete("AD<n>", false)]` (a warning, so Core still builds and dependents get built), build a scratch
   solution holding every csproj that references NumSharp.Core (33 on `strings`, the examples included), and collect
   the warnings with that tag. File-based `dotnet run` scripts (`#:project`) are compiled by no project: grep them.
5. **Watch for silent rebinds.** After a collapse, an old positional call can bind a DIFFERENT parameter through an
   implicit conversion: `np.max(a, 1, true)` put `true` into `out` (a bool converts to `NDArray`). Make such a call
   NumPy's error instead of a value: `ReductionArguments.ValidateParsedArguments` raises "return arrays must be of
   ArrayType" for `NDArray.WasScalarArgument`. Check every new signature for `bool` / number / string slots that an
   old positional argument can now reach.
6. **Implement and delete.** The surviving overload calls the engine; the other overloads are deleted; every
   `[OverloadResolutionPriority]` of the name is removed (60 uses in 18 files on `strings`, mostly `np.random`).
7. **Migrate callers**: `src/`, tests, benchmarks (`benchmark/NumSharp.Benchmark.CSharp` builds for `net8.0` too),
   examples, the oracle registry (`test/NumSharp.Tests.Oracle/Fuzz/OpRegistry*.cs`), docs snippets.
8. **Update the gates that pin signatures and names**: the reflection gate for the family (one overload, no ORPA:
   `test/NumSharp.Tests/Selection/np.max.Keywords.Test.cs` is the model), `LinAlgSignatureParityTests` for
   `np.linalg`, the random-API gate (`test/oracle/random_numpy_signatures.json`, `RandomApiCoverageTests` G3) for
   `np.random`, and the coverage alias tables when a name changes
   (`OracleSurfaceCoverageTests.EquivalentAliases` / `NdarrayAliases` and `coverage/oracle_map.json`, together).
9. **Tests from NumPy output**: unit tests for each new parameter and call form; oracle cases where a tier exists
   (`reduce_kwargs` for reductions, the random-API oracle, the ma tiers).
10. **Prove the binding**: regenerate the three outputs (§10). The name's row must lose its marks; the probe must show
    no failure at C# 12 and C# 14. Build the unit suite and the oracle on `net8.0` and `net10.0` (Release,
    `TestCategory!=OpenBugs`).
11. **Record**: the plan item in `api-parameter-parity.md` with the commit; the breaking change in the release notes
    (`docs/releases/RELEASE_<version>.md`); the catalog changelog.

## 5. Work packages

### AD1: one overload per name (O)

**What.** 360 names with a NumPy counterpart have 2 to 33 overloads. **Why.** The owner's rule (§3.1); every extra
overload is a CS0121 risk at C# 12 and a second code path where bugs hide (the `dtype=` min path returned NaN for
int64 -> float64). **Target.** One overload per name with NumPy's signature.

**How.** Most overload sets fall into a few patterns; each has a standard collapse:

| pattern | example | collapse to |
|---|---|---|
| shape spelled as `int`, `int[]`, `long[]`, `params`, `Shape` | `np.empty` 8, `np.zeros` 8, `np.ones` 8, `reshape` 4–6 | one `Shape` parameter: `Shape` converts implicitly from `int`, `long`, `int[]`, `long[]` and 2- to 6-element tuples |
| one overload per C# scalar type | `np.arange` 11, `np.linspace` 4, `np.insert` 12, `np.pad` 10 | one overload over `NDArray` / `object` / the widest type; a value's C# type still decides the dtype where NumPy's Python type does (decision OD2) |
| one overload per C# input array type | `np.array` 33, `np.asarray` 7, `np.frombuffer` 18 | one array_like parameter (decision OD2) |
| dtype spelled as `NPTypeCode`, `Type`, `DType`, generic `<T>` | `np.can_cast` 20, `np.isdtype` 15, `np.issubdtype` 11, `np.result_type` 8 | one `DType` (AD6, AD7) |
| axis spelled as `int`, `int?`, `int[]` | `np.all` 6, `np.any` 6, `np.sum` 7, `np.mean` 5 | one axis type (decision OD1; AD9, AD10) |
| seeds and sizes ranked with `[OverloadResolutionPriority]` | `np.random.default_rng` 13, the engines 8–9, the samplers 3–7 | one seed parameter, one `size` parameter (decisions OD4, OD8) |
| file spelled as path, `Stream`, `TextReader`, `TextWriter`, `byte[]` | `np.savez` 8, `np.savetxt` 6, `np.loadtxt` 4, `np.load` 3 | one file parameter (decision OD3) |
| positional-dtype convenience forms | `np.exp(x, NPTypeCode)` beside `np.exp(x, out, where, dtype)` | delete the convenience form (it is not a NumPy call form) |
| typed iterators | `np.nditer` 3, `np.ndenumerate` 2, `np.ndindex` 2 | decision OD7 |
| polynomial forms for allocation-free binding | `mapdomain` 10, `mapparms` 4, `{p}pow` 4 | decision OD2: the generic tuple overloads exist so a C# tuple binds without boxing (`mapparms<T0..T3>`), and the `object` pow forms port NumPy's `int(pow)` |

Then the procedure of §4 per name or per family (a family shares its census and its gate).

**Names (360, overload count after each name):**
- **`np`** (150): `abs` 2 · `absolute` 2 · `all` 6 · `any` 6 · `append` 2 · `apply_along_axis` 2 · `apply_over_axes` 2 · `arange` 11 · `argmax` 3 · `argmin` 3 · `argpartition` 3 · `argsort` 2 · `array` 33 · `array_split` 3 · `asanyarray` 3 · `asarray` 7 · `ascontiguousarray` 2 · `asfortranarray` 2 · `asmatrix` 4 · `atleast_1d` 4 · `atleast_2d` 4 · `atleast_3d` 4 · `average` 2 · `base_repr` 2 · `binary_repr` 2 · `bmat` 5 · `broadcast` 2 · `broadcast_arrays` 2 · `broadcast_to` 9 · `can_cast` 20 · `choose` 3 · `concat` 9 · `concatenate` 9 · `copyto` 4 · `count_nonzero` 2 · `delete` 7 · `divmod` 2 · `dsplit` 2 · `dtype` 5 · `einsum` 3 · `einsum_path` 3 · `empty` 8 · `empty_like` 2 · `equal` 3 · `exp` 2 · `exp2` 2 · `expand_dims` 3 · `expm1` 2 · `finfo` 5 · `flip` 2 · `float_power` 2 · `floor_divide` 2 · `fmod` 2 · `frombuffer` 18 · `fromfile` 2 · `fromstring` 2 · `full` 5 · `full_like` 2 · `geomspace` 4 · `gradient` 5 · `greater` 3 · `greater_equal` 3 · `histogram` 4 · `histogram2d` 6 · `histogram_bin_edges` 4 · `histogramdd` 8 · `hsplit` 2 · `iinfo` 5 · `indices` 3 · `insert` 12 · `interp` 2 · `intersect1d` 3 · `isdtype` 15 · `issubdtype` 11 · `left_shift` 2 · `less` 3 · `less_equal` 3 · `lexsort` 2 · `linspace` 4 · `load` 3 · `loadtxt` 4 · `log` 2 · `log10` 2 · `log1p` 2 · `log2` 2 · `logspace` 2 · `matrix` 4 · `mean` 5 · `median` 2 · `meshgrid` 3 · `mintypecode` 2 · `mod` 2 · `moveaxis` 4 · `nanargmax` 2 · `nanargmin` 2 · `nanmedian` 2 · `nanpercentile` 6 · `nanquantile` 8 · `ndarray` 2 · `ndenumerate` 2 · `ndindex` 2 · `nditer` 3 · `nested_iters` 2 · `not_equal` 3 · `ones` 8 · `ones_like` 2 · `pad` 10 · `partition` 3 · `percentile` 6 · `piecewise` 3 · `place` 2 · `polyint` 2 · `polyval` 2 · `power` 4 · `promote_types` 4 · `ptp` 2 · `put` 2 · `putmask` 2 · `quantile` 8 · `ravel` 2 · `ravel_multi_index` 3 · `remainder` 2 · `repeat` 5 · `require` 2 · `reshape` 4 · `result_type` 8 · `right_shift` 2 · `roll` 2 · `save` 4 · `savetxt` 6 · `savez` 8 · `savez_compressed` 8 · `searchsorted` 3 · `split` 3 · `square` 2 · `squeeze` 3 · `std` 2 · `sum` 7 · `take` 2 · `tensordot` 3 · `tile` 2 · `trim_zeros` 2 · `unravel_index` 2 · `unwrap` 2 · `var` 2 · `vectorize` 12 · `vsplit` 2 · `where` 5 · `zeros` 8 · `zeros_like` 2
- **`ndarray`** (24): `__getitem__` 4 · `__setitem__` 3 · `argmax` 3 · `argmin` 3 · `argpartition` 3 · `argsort` 2 · `astype` 2 · `choose` 3 · `getfield` 2 · `item` 10 · `mean` 4 · `partition` 3 · `ravel` 2 · `repeat` 3 · `reshape` 6 · `resize` 3 · `searchsorted` 3 · `std` 2 · `sum` 3 · `take` 2 · `tofile` 2 · `var` 2 · `view` 4 · `(constructor)` 11
- **`np.char`** (2): `compare_chararrays` 2 · `join` 2
- **`np.dtypes`** (1): `StringDType` 2
- **`np.fft`** (4): `fftfreq` 3 · `fftshift` 2 · `ifftshift` 2 · `rfftfreq` 3
- **`np.linalg`** (4): `multi_dot` 2 · `norm` 2 · `tensordot` 2 · `vector_norm` 2
- **`np.ma`** (10): `apply_over_axes` 2 · `arange` 3 · `argsort` 2 · `around` 2 · `default_fill_value` 2 · `fromfunction` 2 · `hsplit` 2 · `indices` 3 · `reshape` 2 · `unique` 2
- **`np.polynomial.*`** (6 × 3): `chebpow` 4 · `chebtrim` 2 · `chebvander` 2 · `hermpow` 4 · `hermtrim` 2 · `hermvander` 2 · `hermepow` 4 · `hermetrim` 2 · `hermevander` 2 · `lagpow` 4 · `lagtrim` 2 · `lagvander` 2 · `legpow` 4 · `legtrim` 2 · `legvander` 2 · `polypow` 4 · `polytrim` 2 · `polyvander` 2
- **`np.polynomial.polyutils`** (4): `mapdomain` 10 · `mapparms` 4 · `trimcoef` 2 · `trimseq` 3
- **`np.random`** (55): `MT19937` 8 · `PCG64` 8 · `PCG64DXSM` 8 · `Philox` 9 · `RandomState` 8 · `SFC64` 8 · `SeedSequence` 8 · `beta` 3 · `binomial` 3 · `chisquare` 3 · `choice` 3 · `default_rng` 13 · `dirichlet` 5 · `exponential` 3 · `f` 3 · `gamma` 3 · `geometric` 3 · `get_state` 2 · `gumbel` 3 · `hypergeometric` 7 · `laplace` 3 · `logistic` 3 · `lognormal` 3 · `logseries` 6 · `multinomial` 4 · `multivariate_normal` 5 · `negative_binomial` 3 · `noncentral_chisquare` 3 · `noncentral_f` 3 · `normal` 3 · `pareto` 6 · `permutation` 3 · `poisson` 3 · `power` 6 · `rand` 2 · `randint` 4 · `randn` 2 · `random` 2 · `random_sample` 2 · `ranf` 2 · `rayleigh` 3 · `sample` 2 · `seed` 8 · `set_state` 3 · `standard_cauchy` 2 · `standard_exponential` 2 · `standard_gamma` 3 · `standard_normal` 2 · `standard_t` 3 · `triangular` 3 · `uniform` 3 · `vonmises` 3 · `wald` 3 · `weibull` 3 · `zipf` 3
- **`np.random.Generator`** (36): `beta` 2 · `binomial` 2 · `chisquare` 2 · `choice` 2 · `dirichlet` 2 · `exponential` 2 · `f` 2 · `gamma` 2 · `geometric` 2 · `gumbel` 2 · `hypergeometric` 2 · `integers` 4 · `laplace` 2 · `logistic` 2 · `lognormal` 2 · `logseries` 2 · `multinomial` 3 · `multivariate_hypergeometric` 2 · `multivariate_normal` 2 · `negative_binomial` 2 · `noncentral_chisquare` 2 · `noncentral_f` 2 · `normal` 2 · `pareto` 2 · `permutation` 2 · `poisson` 2 · `power` 2 · `rayleigh` 2 · `standard_gamma` 2 · `standard_t` 2 · `triangular` 2 · `uniform` 2 · `vonmises` 2 · `wald` 2 · `weibull` 2 · `zipf` 2
- **`np.strings`** (3): `add` 4 · `multiply` 2 · `slice` 4
- **`NDArray<T>`** (2): `reshape` 5 · `(constructor)` 9
- **`np.matrix`** (19): `__getitem__` 4 · `__pow__` 2 · `__setitem__` 4 · `all` 2 · `any` 2 · `astype` 2 · `max` 2 · `mean` 2 · `min` 2 · `prod` 2 · `ptp` 2 · `ravel` 2 · `repeat` 3 · `reshape` 4 · `round` 2 · `std` 2 · `sum` 2 · `var` 2 · `view` 3
- **`np.char.chararray`** (12): `__getitem__` 4 · `argmax` 3 · `argmin` 3 · `argpartition` 3 · `astype` 2 · `join` 2 · `partition` 4 · `ravel` 2 · `repeat` 3 · `reshape` 6 · `take` 2 · `view` 3
- **`np.ma.MaskedArray`** (2): `argsort` 2 · `partition` 2
- **`np.dtype`** (2): `newbyteorder` 2 · `(constructor)` 2
- **`np.poly1d`**, **`NpzFile`** (1 each): `(constructor)` 2 · `(constructor)` 2
- **`SeedSequence`** and the engines: `SeedSequence` `(constructor)` 8 · `MT19937` `(constructor)` 8 and `_legacy_seeding` 5 · `PCG64` `(constructor)` 8 and `jumped` 2 · `PCG64DXSM` `(constructor)` 8 and `jumped` 2 · `Philox` `(constructor)` 9 and `jumped` 2 · `SFC64` `(constructor)` 8

**Notes.**
- The `__getitem__` / `__setitem__` rows are the Python-protocol methods (`Backends/NDArray.Container.cs`,
  `NDMatrix`, `NDCharArray`) that forward to the C# indexers; their overloads stand for Python's index forms (an int,
  a slice string, coordinates, an index object). NumPy has one `__getitem__(key)`; collapse them only if decision OD2
  gives one index type that covers strings, slices, integers, masks and index arrays.
- 63 NumSharp-only names also have several overloads (Appendix C). The owner's rule speaks of NumPy functions; those
  names wait for decision OD5.

### AD2: renamed parameters (M ● renames)

**What.** 65 parameters in 55 names are spelled differently from NumPy (or a different parameter sits at NumPy's
slot). **Why.** A named port (`np.isreal(x: a)`) does not compile. **Target.** NumPy's name.

**How.** Rename the C# parameter (and its `<param>` doc). A rename breaks named-argument callers only, so the census of
§4 step 4 searches for `oldName:` in callers. Fix the matching X entries (AD5) in the same change. Where the "rename"
is really a different parameter at NumPy's slot (marked **slot** below), the fix is the full NumPy signature, not a
rename.

- **`np`** (23 in 18 names): `around`: `a` ← `x` · `array`: `object` ← `array, buffer, data, nd, scalar, strArray, value` (one parameter replaces 33 overloads' first parameters; AD1) · `array_equal`: `a1` ← `a`, `a2` ← `b` · `array_repr`: `arr` ← `a` · `broadcast_to`: `array` ← `from`, `shape` ← `against` · `can_cast`: `from_` ← `from, value` · `cumprod`: `a` ← `arr`, `dtype` ← `typeCode` (plan B9) · `cumsum`: `a` ← `arr`, `dtype` ← `typeCode` (B9) · `frexp`: `out` ← `out1` (**slot**: NumPy's `out=(o1, o2)` tuple, plan F1) · `iinfo`: `int_type` ← `arr, dtype, dtypeName` · `iscomplex`, `iscomplexobj`, `isreal`, `isrealobj`: `x` ← `a` · `isscalar`: `element` ← `obj` · `modf`: `out` ← `outFrac` (**slot**, F1) · `put`: `ind` ← `index, indices`, `v` ← `value, values` · `transpose`: `axes` ← `premute` (plan E4). Plan G1 covers most of these.
- **`ndarray`** (2): `setfield`: `val` ← `value` · `(constructor)`: `buffer` ← `fillZeros` (**slot**; decision OD6)
- **`np.ma`** (30 in 29 names): `amax`, `amin`: `out` ← `fill_value` (**slot**: NumPy's `np.ma.amax` IS `np.amax`, with `out` / `initial` / `where` and no `fill_value`) · `compressed`: `x` ← `a` · `count_masked`: `arr` ← `a` · `getmaskarray`: `arr` ← `a` · `hsplit`: `ary` ← `a`, `indices_or_sections` ← `indices, sections` · `is_mask`: `m` ← `a` · `left_shift`, `right_shift`: `n` ← `b` · `make_mask_descr`: `ndtype` ← `dtype` · `masked_all_like`: `arr` ← `a` · `masked_equal`, `masked_greater`, `masked_greater_equal`, `masked_inside`, `masked_less`, `masked_less_equal`, `masked_not_equal`, `masked_object`, `masked_outside`, `masked_values`: `x` ← `a` · `max`, `min`, `ptp`, `maximum_fill_value`, `minimum_fill_value`, `ndim`, `shape`, `size`: `obj` ← `a`
- **`NDArray<T>`** (1): `(constructor)`: `dtype` ← `fillZeros` (**slot**; OD6)
- **`np.ma.MaskedArray`** (3 in 2 names): `clip`: `min` ← `a_min`, `max` ← `a_max` (NumPy 2.x names) · `resize`: `newshape` ← `new_shape`
- **`np.dtype`** (1): `(constructor)`: `dtype` ← `type, typecode` (OD6)
- **`np.iinfo`** (1): `(constructor)`: `int_type` ← `dtype`
- **`NpzFile`** (4 in 1 name): `(constructor)`: `fid` ← `path, stream` · `own_fid` ← `allowPickle, ownStream` · `allow_pickle` ← `allowPickle, maxHeaderSize` · `pickle_kwargs` ← `maxHeaderSize` (**slot**: the C# order differs too; NumPy: `NpzFile(fid, own_fid=False, allow_pickle=False, pickle_kwargs=None, *, max_header_size=10000)`)

### AD3: missing parameters (M ● that are not renames)

**What.** 393 (name, parameter) pairs in which NumPy's parameter does not exist in any overload. **Why.** A port that
passes it does not compile, and the feature behind it is usually missing too (`out=`, `where=`, `initial=`).
**Target.** The parameter exists, at NumPy's position, and works (values, dtype, the returned object, NumPy's errors).

**How.** Each family below names the engine that already does the work, or what is missing. In every case: probe
NumPy, implement, test from NumPy output, then §4.

#### AD3a: the numpy.ma ufunc wrappers lack `out=` / `where=` / `dtype=`

48 names in `np.ma`: `abs`, `absolute`, `add`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctan2`,
`arctanh`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `ceil`, `conjugate`, `cos`, `cosh`, `divide`, `equal`, `exp`,
`fabs`, `floor`, `floor_divide`, `fmod`, `greater`, `greater_equal`, `hypot`, `less`, `less_equal`, `log`, `log10`,
`log2`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `mod`, `multiply`, `negative`, `not_equal`,
`remainder`, `sin`, `sinh`, `sqrt`, `subtract`, `tan`, `tanh`, `true_divide`.

NumPy: `_MaskedUnaryOperation` / `_MaskedBinaryOperation` / `_DomainedBinaryOperation` forward `*args, **kwargs` to
the plain ufunc (so `out=`, `where=`, `dtype=` reach it), then apply the domain and the mask. NumSharp's plain ufuncs
already take all three (`np.add(x, y, out, where, dtype)`), so the wrapper passes them on and keeps its masking.
Prove first what NumPy does with a masked `out` and with `where=` on masked slots (the result's mask and data).
Oracle: the ma tiers.

#### AD3b: reduction keywords (`out`, `initial`, `where`, `keepdims`, `mean`, `correction`)

- **`np`**: `mean` (out, where) · `nanmean` (out, where) · `nanprod` (out, initial, where; and dtype) · `nanstd` (out, where, mean, correction) · `nansum` (out, initial, where; and dtype) · `nanvar` (out, where, mean, correction) · `prod` (out, initial, where) · `std` (out, where, mean, correction) · `sum` (out, initial, where) · `var` (out, where, mean, correction)
- **`ndarray`**: `mean` (out, where) · `prod` (out, initial, where) · `std` (out, where, mean, correction) · `sum` (out, initial, where) · `var` (out, where, mean, correction)
- **`np.ma`**: `all` (out) · `amax` (initial, where) · `amin` (initial, where) · `any` (out) · `argmax` (out, keepdims) · `argmin` (out, keepdims) · `cumprod` (out) · `cumsum` (out) · `max` (out) · `mean` (out) · `median` (out; and overwrite_input) · `min` (out) · `prod` (out) · `product` (out) · `ptp` (out) · `std` (out) · `sum` (out) · `trace` (out) · `var` (out)
- **`np.matrix`**: `cumprod` (out) · `cumsum` (out) · `trace` (out)
- **`np.ma.MaskedArray`**: `all` (out) · `any` (out) · `argmax` (out, keepdims) · `argmin` (out, keepdims) · `cumprod` (out) · `cumsum` (out) · `max` (out) · `mean` (out) · `min` (out) · `prod` (out) · `product` (out) · `ptp` (out) · `std` (out) · `sum` (out) · `trace` (out) · `var` (out)

Plan items B1–B8 cover the `np` / `ndarray` ones. The model is A4 (the max/min family): NumPy's reduce prologue
(`ReductionArguments.Validate`, error order and texts), `initial=` packed by `ReductionInitial`, `where=` and a cast
`out=` through `NumPyReduceIterator`. The float sums must keep their exact NumPy schedules (np.sum's add.reduce, the
var/std ports): `out=` / `where=` change the iteration NumPy runs, so each route is proved against the `reduce_kwargs`
oracle tier (`gen_oracle.py reduce_kwargs`; add the function to `REDUCE_KWARGS_FNS`). numpy.ma's reductions add the
mask rules of `ma.mean` / `ma.var` (plan A6) to `out=`.

#### AD3c: `out=` on functions that are not reductions

- **`np`**: `stack` (plan E1, with `dtype=` and `casting=`)
- **`ndarray`**: `dot` (plan G3)
- **`np.ma`**: `around`, `choose`, `clip`, `compress`, `dot`, `round`, `round_`, `stack`, `take`
- **`np.matrix`**: `choose`, `clip`, `compress`, `dot`, `take`
- **`np.ma.MaskedArray`**: `choose`, `clip`, `compress`, `dot`, `round`, `take`
- **`np.char.chararray`**: `compress`, `take`

NumPy's `out=` contract everywhere: the result is written into `out` and `out` itself is returned; `out` must have the
result's shape (it is never broadcast); the result is cast into `out` under NumPy's rule for that function (ufuncs:
`same_kind`; `take`: unsafe into a matching shape); `out` may overlap an input. Check the existing implementations
(`np.take` has `out=`) and reuse them.

#### AD3d: `dtype=` and `casting=` outside the ufunc wrappers

- **`np`**: `frexp` (dtype; plan F1) · `ldexp` (dtype; F2) · `nanprod`, `nansum` (dtype; B5) · `stack` (dtype, casting; E1) · `hstack`, `vstack` (dtype, casting; E2)
- **`np.ma`**: `arange` (dtype) · `clip` (dtype) · `hstack`, `row_stack`, `stack`, `vstack` (dtype, casting) · `make_mask`, `make_mask_none` (dtype)
- **`np.ma.MaskedArray`**: `astype` (order, casting)

`np.concatenate` already takes `dtype=` / `casting=`: the stacking functions pass them to it, as NumPy's do.

#### AD3e: creation and layout keywords

- `order`: `np.full`, `np.reshape` (E3), `NDArray<T>.reshape`; numpy.ma: `array`, `asanyarray`, `asarray`, `copy`, `empty`, `empty_like`, `masked_array`, `ones`, `ones_like`, `reshape`, `zeros`, `zeros_like`; `MaskedArray.copy`, `.reshape`, `.resize`
- `shape`: `np.full_like`, `np.ones_like`, `np.zeros_like` (plan D2); `np.ma.empty_like`, `ones_like`, `zeros_like`; `MaskedArray.reshape`
- `copy`: `np.asanyarray` (plan D5), `np.reshape` (plan E3), `ndarray.reshape`, `NDArray<T>.reshape`, `NDMatrix.reshape`, `NDCharArray.reshape`, `MaskedArray.reshape`; `np.ma.make_mask`, `np.ma.mask_or`; `np.dtype` constructor
- `device`: `np.ma.arange`, `empty`, `empty_like`, `ones`, `ones_like`, `zeros`, `zeros_like`
- `fill_value`, `hardmask`: `np.ma.arange`, `clip`, `empty`, `identity`, `indices`, `ones`, `squeeze`, `zeros` (numpy.ma's `_convert2ma` wrappers add both keywords to the function they wrap)
- `ndmin`: `np.ma.array`, `np.ma.masked_array` · `ndmax`: `np.array` (plan D6)
- `offset`, `strides`: the `ndarray` and `NDArray<T>` constructors; `buffer` too for `NDArray<T>` (decision OD6; `np.ndarray(...)` already has all of them since `f17c3b2d`)

#### AD3f: sorting keywords

`kind`, `stable`: `np.ma.argsort`, `np.ma.sort`, `NDMatrix.argsort`, `MaskedArray.argsort`, `MaskedArray.sort` (the
`np` / `ndarray` forms are plan C1 / C2). On `strings`, `np.sort` honours `kind` since `9771aa9c`.

#### AD3g: one-offs

| name | missing | note |
|---|---|---|
| `np.linspace` | `retstep`, `axis` | plan D3: one result type for NumPy's array-or-tuple return |
| `np.average`, `np.ma.average` | `returned` | B10: `np.average_returned` exists as a separate NumSharp name; NumPy's form is one function |
| `np.indices`, `np.ma.indices` | `sparse` | plan D4: `np.indices_sparse` exists separately |
| `np.vectorize` | `doc`, `excluded`, `cache` | H2 |
| `np.array2string`, `np.printoptions`, `np.set_printoptions` | `formatter`, `legacy` (+ `override_repr` for the last two) | H1 |
| `np.frompyfunc` | `identity` | read from NumPy's stub |
| `np.ma.angle` | `deg` | |
| `np.ma.apply_along_axis`, `MaskedArray.item` | `*args` | a `params object[]` tail |
| `np.ma.clip` | `min`, `max` | NumPy 2.x aliases of `a_min` / `a_max` (the `np.clip` rule of `.claude/CLAUDE.md` applies) |
| `np.ma.mask_cols`, `np.ma.mask_rows` | `axis` | deprecated in NumPy (it ignores it); accept and ignore, as NumPy does |
| `np.ma.ndenumerate` | `compressed` | |
| `np.ma.size` | `axis` | |
| `MaskedArray.view` | `type` | NumPy's array class (`Type`) |
| `np.broadcast` (constructor) | `*arrays` | a `params NDArray[]` constructor |
| `np.dtype` (constructor) | `align`, `copy`, `metadata` | H3; `np.dtype(...)` (the function) already has them |
| `NpzFile` (constructor) | `max_header_size` | |

### AD4: excluded parameters only (M ◦)

**What.** 132 names miss nothing but parameters the plan excludes (and some names with M ● miss them too). **Action.**
None under the current scope. These lists are the worklist if the owner widens it; note that `subok=` has a meaning
again now that `NDMatrix`, `NDCharArray` and `NDMaskedArray` exist (decision OD11).

- **ufunc `casting`, `order`, `subok`, `signature`** (162 names): `np`: `abs`, `absolute`, `acosh`, `add`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctan2`, `arctanh`, `asinh`, `atanh`, `bitwise_and`, `bitwise_count`, `bitwise_not`, `bitwise_or`, `bitwise_xor`, `cbrt`, `ceil`, `conj`, `conjugate`, `copysign`, `cos`, `cosh`, `deg2rad`, `degrees`, `divide`, `divmod`, `equal`, `exp`, `exp2`, `expm1`, `fabs`, `float_power`, `floor`, `floor_divide`, `fmax`, `fmin`, `fmod`, `frexp`, `gcd`, `greater`, `greater_equal`, `heaviside`, `hypot`, `invert`, `isfinite`, `isinf`, `isnan`, `lcm`, `ldexp`, `left_shift`, `less`, `less_equal`, `log`, `log10`, `log1p`, `log2`, `logaddexp`, `logaddexp2`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `matvec`, `maximum`, `minimum`, `mod`, `modf`, `multiply`, `negative`, `nextafter`, `not_equal`, `positive`, `power`, `rad2deg`, `radians`, `reciprocal`, `remainder`, `right_shift`, `rint`, `sign`, `signbit`, `sin`, `sinh`, `spacing`, `sqrt`, `square`, `subtract`, `tan`, `tanh`, `true_divide`, `trunc`, `vecdot`, `vecmat`; `np.char`: `add`; `np.ma`: the 48 ufunc wrappers of AD3a; `np.strings`: `add`, `equal`, `greater`, `greater_equal`, `isalnum`, `isalpha`, `isdecimal`, `isdigit`, `islower`, `isnumeric`, `isspace`, `istitle`, `isupper`, `less`, `less_equal`, `not_equal`, `str_len`
- **`like`** (20): `np`: `arange`, `asanyarray`, `ascontiguousarray`, `asfortranarray`, `empty`, `eye`, `frombuffer`, `full`, `identity`, `loadtxt`, `ones`, `tri`, `zeros`; `np.ma`: `arange`, `empty`, `frombuffer`, `fromfunction`, `identity`, `ones`, `zeros`
- **`subok`** (16): `np`: `broadcast_arrays`, `broadcast_to`, `empty_like`, `full_like`, `ones_like`, `zeros_like`; `ndarray`: `astype`; `np.ma`: `array`, `empty_like`, `getdata`, `masked_array`, `ones_like`, `zeros_like`; `NDMatrix.astype`; `NDCharArray.astype`; `MaskedArray.astype`
- **sort field `order`** (5): `np.ma.argsort`, `np.ma.sort`, `NDMatrix.argsort`, `MaskedArray.argsort`, `MaskedArray.sort`
- **`subok`, `like`** (1): `np.array` · **`einsum_call`** (1): `np.einsum_path` · **`subok`, `signature`** (1): `np.matmul`

### AD5: foreign parameters (X)

**What.** 90 names have a parameter NumPy does not have. **Why.** Under the one-signature rule an extra parameter goes,
or it is spelled the NumPy way. **How.** Three kinds; the first is fixed with AD2, the second needs the owner's
decisions OD2 / OD3 / OD5, the third is a straight removal or rename.

1. **The C# half of a rename (AD2).** `np`: `around` (x), `array_equal` (a, b), `array_repr` (a), `broadcast_to`
   (against, from), `can_cast` (from, value), `cumprod` / `cumsum` (arr, typeCode), `iscomplex` / `iscomplexobj` /
   `isreal` / `isrealobj` (a), `isscalar` (obj), `put` (index, indices, value, values), `transpose` (premute), `frexp`
   (out1, out2), `modf` (outFrac, outIntegral), `iinfo` (arr, dtype, dtypeName); `ndarray.setfield` (value); every
   `np.ma` entry: `compressed`, `count_masked`, `getmaskarray`, `hsplit`, `is_mask`, `left_shift`, `make_mask_descr`,
   `masked_all_like`, the ten `masked_*` predicates, `max`, `maximum_fill_value`, `min`, `minimum_fill_value`, `ndim`,
   `ptp`, `right_shift`, `shape`, `size` (a / b / dtype); `MaskedArray.resize` (new_shape); the `np.iinfo` and
   `NpzFile` constructors.
2. **NumSharp extension inputs** (decision OD2 / OD3 / OD5):
   - `np.array` (array, buffer, data, nd, scalar, size, strArray, value): one overload per input kind (AD1);
   - `np.asarray` (buffer, data, ndim), `np.asanyarray`, `np.ascontiguousarray`, `np.asfortranarray` (buffer): the
     `np.MemoryView` and raw-buffer forms;
   - `np.frombuffer` (address, array, byteLength, dispose, memory, segment): raw pointer and `Memory<T>` forms (NumPy:
     `frombuffer(buffer, dtype, count, offset)`);
   - IO: `np.fromfile` (stream), `np.load` (bytes), `np.loadtxt` (lines, reader, stream), `np.savetxt` (stream, writer);
   - `np.nditer` (writeable): the typed iterator `np.nditer<T>` (OD7);
   - `np.squeeze` (shape): a `Shape`-only helper overload; `np.finfo` (arr, dtypeName), `np.issubdtype` (arr),
     `np.isdtype` (arr, kinds): convenience overloads that take an array or a dtype name;
   - the `ndarray` constructor (fillZeros, size, storage, values), the `NDArray<T>` constructor (fillZeros, size,
     values), `np.random.RandomState` (state), `np.poly1d` constructor (source) (OD6).
3. **Not NumPy's spelling of NumPy's parameter**:
   - `np.all`, `np.any`, `np.argsort` (nd): NumPy's first parameter is `a`;
   - `np.split`, `np.array_split`, `np.hsplit`, `np.vsplit`, `np.dsplit` (indices): NumPy has one parameter
     `indices_or_sections` (an int or a sequence);
   - `np.take`, `ndarray.take`, `NDCharArray.take` (index), `np.unravel_index` (index): NumPy's `indices`;
   - `np.insert` (value): NumPy's `values`; `np.ravel_multi_index` (coords, modes): NumPy's `multi_index`, `mode`;
   - `np.tensordot`, `np.linalg.tensordot` (axesA, axesB): NumPy has one `axes` (an int or a pair of sequences);
   - `ndarray.conj`, `ndarray.conjugate` (out): NumPy's methods take no `out`;
   - `np.ma.amax`, `np.ma.amin` (fill_value): NumPy's are `np.amax` / `np.amin` (AD2 slot); `np.ma.apply_over_axes`
     (axis): NumPy's `axes`; `np.ma.default_fill_value` (dtype): NumPy's `obj`;
   - `NDCharArray.partition` (axis, kind, kth, order): `NDCharArray` shows `NDArray.partition`, but NumPy's
     `chararray.partition(sep)` is the string method; hide the inherited overload (`new`) so only `partition(sep)`
     remains.

### AD6: dtypes as `NPTypeCode` / `Type` (T ●)

**What.** 10 NumPy names (8 NumSharp-only). **How.** Replace the `NPTypeCode` / `Type` overloads with the one `DType`
overload: positional callers keep compiling, because `NPTypeCode`, `Type` and a dtype string convert to `DType`
implicitly. Named callers change only if the parameter name changes. A return type `NPTypeCode` becomes `DType`.

- **`np`**: `can_cast` (NPTypeCode / Type `from`, `to`) · `dtype` (Type type, NPTypeCode typecode) · `frombuffer` (NPTypeCode dtype) · `issubdtype` (NPTypeCode / Type arg1, arg2) · `promote_types` (NPTypeCode / Type type1, type2) · `result_type` (NPTypeCode[] types, NPTypeCode / Type type1, type2)
- **`ndarray.view`, `NDMatrix.view`, `NDCharArray.view`** (Type type): **not a defect.** NumPy's `view(dtype, type)`
  takes `type` as an array CLASS, and `Type` is the right C# analog. The mark counts it by type name. These names
  still carry O (3–4 overloads).
- **`np.dtype` constructor** (Type type, NPTypeCode typecode): decision OD6 (`np.dtype(...)`, the function, takes
  `object`).
- NumSharp-only: `np.common_type_code` (returns NPTypeCode), `np.find_common_type`, `np.issubsctype`,
  `np.maximum_sctype`, `np.sctype2char`, `ndarray.ReplaceData`, `DType.Equals`, `DType.From`. Four of them
  (`find_common_type`, `issubsctype`, `maximum_sctype`, `sctype2char`) are names NumPy no longer has (removed by
  2.0): delete them (breaking, NumPy-aligned), with `np.issctype`, `np.asscalar`, `np.round_` and `ndarray.itemset`
  (§6, Appendix C).

### AD7: generic dtype selectors (T ◇)

**What.** 17 NumPy names have a generic overload whose type argument picks the dtype; 18 NumSharp-only names are
typed accessors. **How.** For NumPy names, remove the generic form under the one-signature rule (the max/min family
did: `np.amin<T>` is gone since `795560d5`); the caller writes `dtype: np.float64` (or `typeof(double)`, which
converts). For the typed iterators and accessors, decision OD7 / OD5.

- **`np`** (11): `argsort` (`<T>(NDArray nd, int axis)`) · `can_cast` (`<TFrom, TTo>(string casting)`) · `empty` (`<T>(int[] shape)`, `<T>(long[] shape)`) · `finfo` (`<T>()`) · `full` (`<T>(int[] shape, object fill_value)`, `<T>(long[] shape, object fill_value)`) · `iinfo` (`<T>()`) · `ndenumerate` (`<T>(NDArray arr)`) · `nditer` (`<T>(NDArray op, bool writeable, char order)`) · `ones` (`<T>(int[] shape)`) · `promote_types` (`<T1, T2>()`) · `zeros` (`<T>(int[] shape)`, `<T>(long[] shape)`)
- **`ndarray`** (4): `argsort` (`<T>(int axis)`) · `getfield` (`<T>(int offset)`) · `item` (`<T>()`, `<T>(long index)`, `<T>(long i, long j)`, `<T>(long i, long j, long k)`, `<T>(long[] indices)`) · `view` (`<T>()`)
- **`np.random`** (1): `randn` (`<T>()`)
- **`NDArray<T>`** (1): its constructors (`<TDType>`): the typed class itself; keep.
- NumSharp-only (18): `np.asscalar`, `np.flat`, `np.nditer_chunks`; `ndarray.AsGeneric`, `AsOrMakeGeneric`,
  `CloneData`, `Data`, `FromMultiDimArray`, `GetAtIndex`, `GetData`, `GetValue`, `MakeGeneric`, `Scalar`, `ToArray`,
  `ToJaggedArray`, `ToMuliDimArray`, `AsEnumerable`; `np.flatiter`'s `AsTyped`. These are C# interop API (typed views,
  conversions), not dtype requests for a computation.

### AD8: NumPy-style calls that do not compile (B)

**What.** 45 names. Each line: the failing call form, the compiler's reason, the fix. Most fixes are the one NumPy
signature of AD1 for that name. 13 of the names come from two probe kinds added after the second pass (§2 B): several
values at NumPy's `*args` (`a.transpose(1, 0)`) and the call without a parameter NumPy leaves unset
(`np.ma.where(condition)`).

**`np`** (18):
- `arange`: `step` alone and `device` alone do not bind (no 2-argument overload). NumPy:
  `arange(start_or_stop, /, stop=None, step=1, *, dtype=None, device=None, like=None)`. One overload in which `stop`,
  `step`, `dtype`, `device` are optional; the value types follow decision OD2 (NumPy infers int64 / float64 from the
  Python types of the bounds).
- `asanyarray`: `order` alone binds only the `np.MemoryView` overload. NumPy:
  `asanyarray(a, dtype=None, order=None, *, device=None, copy=None, like=None)`; add `copy` (plan D5).
- `bmat`: `gdict` needs `ldict`. NumPy: `bmat(obj, ldict=None, gdict=None)`: make `ldict` optional.
- `count_nonzero`: `keepdims` needs `axis`. NumPy: `count_nonzero(a, axis=None, *, keepdims=False)`.
- `empty`, `ones`, `zeros`: `device` needs `dtype`; NumPy's positional `(shape, dtype, order)` fails because C# has
  `(shape, order, dtype)`. NumPy: `zeros(shape, dtype=None, order='C', *, device=None, like=None)`: one overload
  `(Shape shape, DType dtype = null, char order = 'C', string device = null)` (plan D1).
- `empty_like`, `full_like`, `ones_like`, `zeros_like`: `order` needs `dtype`. NumPy:
  `empty_like(prototype, /, dtype=None, order='K', subok=True, shape=None, *, device=None)` (plan D2; `subok`
  excluded).
- `linspace`: the required-only call `np.linspace(start, stop)` does not compile (C# requires `num`). NumPy:
  `linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0, *, device=None)` (plan D3).
- `mean`: `dtype` needs `axis` (B3).
- `nanmean`, `nanstd`, `nanvar`, `std`, `var`: NumPy's positional `(a, axis, dtype)` fails (C# puts `keepdims` at
  slot 2 or 3). NumPy: `std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>,
  mean=<no value>, correction=<no value>)` (B4, B6, B7).

**`ndarray`** (7):
- `astype`: `order` and `casting` need `copy`; NumPy's positional `(dtype, order)` fails. NumPy:
  `a.astype(dtype, order='K', casting='unsafe', subok=True, copy=True)` (plan G2). The same failure on
  `NDMatrix.astype` and `NDCharArray.astype` (their `new` overloads inherit the order).
- `mean`: `dtype` needs `axis`; `keepdims` alone has no overload. `sum`: `dtype` and `keepdims` need `axis`; positional
  `(axis, dtype)` fails. `std`, `var`: positional `(axis, dtype)` fails. NumPy:
  `a.sum(axis=None, dtype=None, out=None, *, keepdims=<no value>, initial=<no value>, where=<no value>)` (B1–B4).
- `transpose`: `a.transpose(1, 0)` does not compile (CS1501): the only overload takes an `int[] premute`, not a
  `params` array, so only `a.transpose(new[] { 1, 0 })` binds. NumPy: `a.transpose(*axes)` takes the axes one by one
  or as one sequence and no keyword (`a.transpose(axes=...)` is NumPy's TypeError). One overload
  `transpose(params int[] axes)` binds all three spellings (none, several, one array); an empty array then stands for
  NumPy's `a.transpose()` (reverse the axes), so NumPy's explicit empty tuple `a.transpose(())`, an error for an array
  of one or more axes, has no spelling of its own: document it. The same on `NDMatrix`, `NDCharArray` and
  `NDMaskedArray`, which declare their own `transpose`.
- `(constructor)`: `new NDArray(new Shape(2))` has no overload. NumPy's class call `ndarray(shape, dtype=None,
  buffer=None, offset=0, strides=None, order=None)` (decision OD6; `np.ndarray(...)` has this signature).

**`np.ma`** (8):
- `apply_along_axis`: `np.ma.apply_along_axis(f, 1, m, 2)` does not compile: the only overload takes
  `Func<NDMaskedArray, NDMaskedArray>` and no further arguments. NumPy: `apply_along_axis(func1d, axis, arr, *args,
  **kwargs)` hands `*args` and `**kwargs` to `func1d`. Take a `params object[] args` and a `func1d` that receives them,
  as `np.apply_along_axis` does (`Func<NDArray, object[], NDArray>`); a port passes keyword arguments through a lambda.
- `arange`: `step` alone does not bind (as `np.arange`).
- `atleast_1d`, `atleast_2d`, `atleast_3d`: `np.ma.atleast_1d(x, y)` does not compile: the only overload takes one
  `object a`. NumPy: `atleast_1d(*arys)` returns one masked array for one input and a tuple for several, the same
  contract as `np.atleast_1d` (which takes `params object[] arys` and is O 4): give `np.ma` the single signature AD1
  settles for `np.atleast_1d`.
- `clip`: `np.ma.clip(m)` and `np.ma.clip(m, min: 1, max: 2)` do not compile (CS7036, `a_min` and `a_max` required).
  NumPy's `np.ma.clip` is `np.clip` over masked arrays: `clip(a, a_min=<no value>, a_max=<no value>, out=None, *,
  min=<no value>, max=<no value>, **kwargs)`, with no bound a positive copy. Take `np.clip`'s signature and keyword
  rule (`a_min` / `a_max` optional, `min` / `max`, `out`, `dtype`, `where`; AD3c, AD3d, AD3e and AD3g list the
  missing ones).
- `masked_array`: the call with no argument does not compile (C# requires `data`; NumPy's `data=None`).
- `where`: `np.ma.where(m)` does not compile (CS7036, `x` and `y` required). NumPy: `where(condition, x=<no value>,
  y=<no value>)` returns `condition.nonzero()` when neither `x` nor `y` is given (masked entries count as False) and
  raises when only one is. `np.where` has that one-argument form as a separate overload returning `NDArray<long>[]`;
  give `np.ma.where` the same form AD1 settles for `np.where`.

**Object classes** (7):
- `np.matrix`, `np.char.chararray`: `astype` (see `ndarray.astype`) and `transpose` (see `ndarray.transpose`).
- `np.ma.MaskedArray.transpose`: as `ndarray.transpose`.
- `np.ma.MaskedArray.clip`: `m.clip()` does not compile (CS7036, `a_min` required). NumPy: `MaskedArray.clip(min=<no
  value>, max=<no value>, out=None, **kwargs)`; no bound is a copy. Take `ndarray.clip`'s signature (`min`, `max`,
  `out`, `dtype`, `where`); the C# names `a_min` / `a_max` are also AD2 renames.
- `np.ma.MaskedArray.item`: `m.item(4)` and `m.item(0, 1)` do not compile: the only overload is `item()`. NumPy:
  `MaskedArray.item(*args)` IS `ndarray.item` (no argument for a one-element array, one flat index, or one index per
  axis), so it returns the stored value of a masked element too (`np.ma.array([1, 2], mask=[0, 1]).item(1)` is 2).
  Take `ndarray.item`'s `params long[]` form.

**Others** (5):
- `np.random.SeedSequence` and the `SeedSequence` constructor: `spawn_key`, `pool_size`, `n_children_spawned` need
  `entropy` (NumPy: `SeedSequence(entropy=None, *, spawn_key=(), pool_size=4, n_children_spawned=0)`; None draws fresh
  OS entropy).
- `NDArray<T>` constructor: `order` alone binds no overload (OD6).
- `np.dtype` constructor: a dtype string does not bind (it takes `Type`) (OD6). This is the only place where NumPy's
  `dtype='float64'` spelling fails: every T ● name also has a `DType` overload.
- `np.broadcast` constructor: `new np.Broadcast(a, b)` does not compile; the class has only a parameterless
  constructor (NumPy's `np.broadcast()`). `np.broadcast(a, b)`, the NumPy spelling, compiles through the factory
  (`params NDArray[]`); decision OD6 decides whether constructors mirror the class call.

### AD9: `null` refused where NumPy's default is None (L)

**AD9a: value-type parameters (L ●).** Make the parameter nullable and read `null` as NumPy's None.
- `axis` (an `int`): `np.count_nonzero`, `np.mean`, `np.squeeze`, `np.std`, `np.sum`, `np.var`; `ndarray.mean`,
  `ndarray.std`, `ndarray.sum`, `ndarray.var`;
- `order` (a `char`): `np.asarray`, `np.asarray_chkfinite`, `np.ndarray` (NumPy: `order=None`, which each function
  reads as its own default: `'K'` for `asarray`, C for `ndarray`);
- `density` (a `bool`): `np.histogram`, `np.histogram2d`, `np.histogramdd`;
- `stop` (a `double` / `int`): `np.arange`, `np.ma.arange` (with AD8's single overload).

**AD9b: `int?` against `int[]` (L ●, plan G4).** Two overloads take `null` equally well (CS0121): `np.average`,
`np.flip`, `np.median`, `np.nanmedian`, `np.percentile`, `np.nanpercentile`, `np.quantile`, `np.nanquantile`,
`np.ptp`; together with other None values: `np.trim_zeros(a, trim, axis: null)`, `np.linalg.norm(a, ord: null, axis:
null)`. The fix is one axis parameter type that takes an int, a sequence and `null` (decision OD1), which also closes
AD10 and the axis overloads of AD1.

**AD9c: only at C# 12 (L ◐).** The overloads compile at C# 14 because `[OverloadResolutionPriority]` picks one; a
`net8.0` consumer gets CS0121:
- `np.all`, `np.any` (`axis: null`: `int[]` against `int?`);
- the seeds: `np.random.seed`, `np.random.RandomState` (`uint[]` / `long[]`), `np.random.default_rng`,
  `np.random.MT19937`, `PCG64`, `PCG64DXSM`, `Philox`, `SFC64` (factories and constructors), `np.random.SeedSequence`
  and its constructor (`entropy`) (`int[]` / `long[]`);
- `np.random.dirichlet`, `np.random.multivariate_normal` (`size: null`: `int[]` / `long[]`).

Fix: the AD1 collapse (one seed parameter, decision OD8; one `size`, decision OD4), which removes the attribute.

**AD9d: `size` / `shape` as `Shape` (L ◦, 73 names).** `null` does not convert to the struct, and `size: default`
binds (NumSharp's documented spelling of `size=None`). Names: `np.empty_like` (`shape`); `np.random`: `beta`,
`binomial`, `chisquare`, `choice`, `exponential`, `f`, `gamma`, `geometric`, `gumbel`, `laplace`, `logistic`,
`lognormal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `poisson`, `power`,
`randint`, `random_integers`, `rayleigh`, `standard_cauchy`, `standard_exponential`, `standard_gamma`,
`standard_normal`, `standard_t`, `triangular`, `uniform`, `vonmises`, `wald`, `weibull`, `zipf`; all 38 samplers of
`np.random.Generator` (`beta` … `zipf`, `integers`, `random`, `multinomial`, `multivariate_hypergeometric`,
`multivariate_normal`, `standard_*`); `BitGenerator.random_raw`. Decision OD4: keep the idiom, or make every such
parameter `Shape? size = null` (then `null`, `default` and every shape spelling bind).

**AD9e: a `Shape?` overload that `null` cannot reach (L ●, a defect).** `np.random.hypergeometric` and
`np.random.logseries` have `Shape? size = null`, but also a `Shape size` overload, which C#'s betterness rules
prefer for `null` (and then refuse). Removing the `Shape` overload (AD1) fixes it.

### AD10: integer sequences refused (Q)

**What.** 77 names take only a scalar where NumPy's stubs take an integer sequence. **Why.** `np.sum(a, axis=(0, 1))`
is common NumPy code; tuple axes were "deferred, library-wide" (`.claude/CLAUDE.md`). **Engine state.** The reduce
prologue already validates tuples (`ReductionArguments.Validate(NDArray, int[] axes, ...)`), the reduction iterator
already reduces several axes (`NumPyReduceIterator.Build(Shape, bool[] axisFlags, ...)`), and `np.all` / `np.any`
already take `int[] axis` with `out=` / `where=`.

**How.**
1. Decide the axis type (OD1), so that one parameter takes an int, a sequence and None.
2. Reductions: reduce ALL requested axes in one NumPy iterator pass, never as repeated single-axis reductions: the
   float sums' bits depend on NumPy's call schedule (the add.reduce and var/std ports), and a repeated reduction
   changes it. Order-free reductions (max / min / all / any / count / integer sums) can use the existing kernels per
   axis, but prove that the result layout is NumPy's.
3. Oracle: tuple-axis cases in `reduce_kwargs`, `sum_identity`, `mean_loops`, `varstd`, `nanvarstd`, `nanmean`,
   `minmax_kwargs`, and the ma tiers.
4. Non-reductions: `np.squeeze` (removes the listed axes), `np.ma.expand_dims` (inserts several), `np.size` (the
   product of the listed dimensions), `np.gradient` (a list of gradients), `np.trim_zeros` (trims the listed axes),
   `np.linalg.norm` (a 2-tuple means a matrix norm), `np.roll` (`shift` and `axis` as sequences: one roll per pair).

**Names (77, parameter in parentheses):**
- **`np`** (22): `amax`, `amin`, `count_nonzero`, `gradient`, `max`, `mean`, `min`, `nanmax`, `nanmean`, `nanmin`, `nanprod`, `nanstd`, `nansum`, `nanvar`, `prod`, `size`, `squeeze`, `std`, `sum`, `trim_zeros`, `var` (axis); `roll` (axis, shift)
- **`ndarray`** (10): `all`, `any`, `max`, `mean`, `min`, `prod`, `squeeze`, `std`, `sum`, `var` (axis)
- **`np.linalg`** (1): `norm` (axis)
- **`np.ma`** (17): `all`, `amax`, `amin`, `any`, `compress`, `count`, `expand_dims`, `max`, `mean`, `min`, `prod`, `product`, `ptp`, `squeeze`, `std`, `sum`, `var` (axis)
- **`np.matrix`** (13): `all`, `any`, `argmax`, `argmin`, `max`, `mean`, `min`, `prod`, `ptp`, `squeeze`, `std`, `sum`, `var` (axis; `NDMatrix`'s two-axis rule keeps the result 2-D)
- **`np.char.chararray`** (1): `squeeze` (axis)
- **`np.ma.MaskedArray`** (13): `all`, `any`, `compress`, `count`, `max`, `mean`, `min`, `prod`, `ptp`, `squeeze`, `std`, `sum`, `var` (axis)

## 6. NumPy names NumSharp lacks

192 NumPy names have no member of that name in their NumSharp group. "(in `X`)" means NumSharp has the name in
another group. Classes and what to do:

| class | meaning | action |
|---|---|---|
| **Alias** | NumPy 2.x spells an existing function another way (the same ufunc object) | add the name with the existing function's one signature; register it in the coverage alias tables |
| **Expose** | NumSharp implements it under another group, type or casing | add the NumPy name in this group, forwarding to the implementation |
| **Implement** | a function NumSharp does not have | probe, port, test (§4) |
| **Plan** | an existing plan covers it | follow that plan |
| **Datetime** | needs `datetime64` / `timedelta64` storage | [`datetime64.md`](datetime64.md) (dtype-system Stage C) |
| **Structured** | needs structured (record) dtypes | none planned; out of scope |
| **N/A** | CPython / NumPy-build specific, no .NET meaning | document, do not implement |
| **Decide** | the owner decides (§8) | |

**`np`** (47):
- **Alias**: `acos`, `asin`, `atan`, `atan2`, `pow`, `bitwise_invert`, `bitwise_left_shift`, `bitwise_right_shift`
  (NumPy 2.0's Array-API names of `arccos`, `arcsin`, `arctan`, `arctan2`, `power`, `invert`, `left_shift`,
  `right_shift`; the coverage artifact credits them through the old names, but `np.acos(x)` does not compile);
  `round` (NumPy 2.0 keeps `round` and removed `round_`: add `np.round`, delete the NumSharp-only `np.round_`).
- **Expose**: `astype` (NumPy 2.1's `np.astype(x, dtype, /, *, copy=True, device=None)`; NumSharp has the methods);
  `ndim`, `shape` (`np.ndim(a)`, `np.shape(a)`; NumSharp has `np.ma.ndim` / `np.ma.shape` and the properties);
  `fromfunction` (in `np.ma`); `poly1d` (the class exists; add the `np.poly1d(...)` factory the way
  `np.random.PCG64(...)` exists); `flatiter` (NumSharp's `np.FlatIterator`, decision OD9).
- **Implement**: `cumulative_sum`, `cumulative_prod` (NumPy 2.1: `cumulative_sum(x, /, *, axis=None, dtype=None,
  out=None, include_initial=False)`, on the cumsum engine); `fromiter` (fix on the way the `np.array<T>(IEnumerable<T>,
  size)` defect that returns uninitialized memory for a short sequence, where NumPy's `fromiter(count=)` raises
  "iterator too short"); `memmap` (the class; `np.load(mmap_mode=)` and `NpyFormat.OpenMemmap` already map files);
  `little_endian` (`BitConverter.IsLittleEndian`).
- **Datetime**: `datetime64`, `timedelta64`, `datetime_as_string`, `busday_count`, `busday_offset`, `busdaycalendar`,
  `is_busday`, `isnat`.
- **Structured**: `recarray`, `record`, `fromregex`, `genfromtxt`.
- **Decide**: `ufunc` (ufunc objects with `.reduce` / `.accumulate` / `.at`; a delegate-typed field plus extension
  methods was shown to work for `np.maximum.reduce` syntax); `errstate`, `seterr`, `geterr`, `seterrcall`,
  `geterrcall` (NumSharp does not model floating-point error states or warnings); `from_dlpack` (DLPack interop);
  `__version__` (NumSharp's version or the NumPy version it emulates).
- **N/A**: `__array_namespace_info__`, `get_include`, `show_config`, `show_runtime`, `info`, `test`. `row_stack` is
  deprecated in NumPy 2.0 (in `np.ma` too); the coverage policy leaves deprecated APIs out.

**`ndarray`** (3) and **`NDArray<T>`** (3): `ctypes` (N/A: `nd.Unsafe.Pointer<T>()` is the .NET form), `dump`,
`dumps` (pickle: N/A).

**`np.char`** (30) and **`np.char.chararray`** (28): **Plan** [`np-char.md`](np-char.md) (C5: the `chararray` string
methods land in batches, one family each, with the `np.@char` re-exports of the same functions; `ffa74606`, C4 and the
first batch, gave `np.@char` `partition`, `rpartition` and the padding functions `center`, `ljust`, `rjust`, `zfill`,
`expandtabs`, and `chararray` the same methods but `partition(sep)`, which it had). 26 of the `np.char` names already
compute in `np.strings`: `capitalize`, `count`, `endswith`, `find`, `index`, `isalnum`, `isalpha`, `isdecimal`,
`isdigit`, `islower`, `isnumeric`, `isspace`, `istitle`, `isupper`, `lower`, `lstrip`, `replace`, `rfind`, `rindex`,
`rstrip`, `startswith`, `str_len`, `strip`, `swapcase`, `title`, `upper`. Not yet anywhere as text functions:
`decode`, `encode`, `translate`, `mod` (string `%` formatting): [`strings-phase2.md`](strings-phase2.md). `chararray`
lacks the same methods except `str_len` and `mod`, which NumPy's `chararray` does not have either (its `%` is the
`__mod__` operator).

**`np.strings`** (4): `decode`, `encode`, `mod`, `translate`: **Plan** [`strings-phase2.md`](strings-phase2.md).

**`np.dtypes`** (3): `Complex64DType`, `ObjectDType`, `VoidDType`: N/A today (NumSharp has no complex64, object or
void dtype; [`dtype-system.md`](dtype-system.md)).

**`np.linalg`** (1): `LinAlgError`: **Expose** (the exception `NumSharp.LinAlgError` exists; decision OD9).

**`np.ma`** (6): `MAError`, `MaskError` (**Expose**: `Exceptions/MAError.cs`; OD9), `MaskedArray` (**Expose**:
`NDMaskedArray`; OD9), `flatten_structured_array`, `fromflex`, `mvoid` (**Structured**).

**`np.polynomial`** and its bases: **Plan**
[`numpy-polynomial-remaining-compute.md`](numpy-polynomial-remaining-compute.md) (routes measured and decided on
2026-10-17):
- `np.polynomial` (7): `Chebyshev`, `Hermite`, `HermiteE`, `Laguerre`, `Legendre`, `Polynomial` (the classes),
  `set_default_printstyle`;
- `chebyshev` (7): `Chebyshev`, `chebfit`, `chebgauss`, `chebinterpolate`, `chebpts1`, `chebpts2`, `chebweight`;
- `hermite` (4): `Hermite`, `hermfit`, `hermgauss`, `hermweight`; `hermite_e` (4): `HermiteE`, `hermefit`,
  `hermegauss`, `hermeweight`; `laguerre` (4): `Laguerre`, `lagfit`, `laggauss`, `lagweight`; `legendre` (4):
  `Legendre`, `legfit`, `leggauss`, `legweight`;
- `polynomial` (3): `Polynomial`, `polyfit` (`np.polyfit` / `np.ma.polyfit` exist with the legacy coefficient order,
  not this one), `polyvalfromroots`;
- `polyutils` (1): `format_float` (the printing unit).

**`np.random`** (1): `BitGenerator`: **Expose** (the base class exists; NumPy cannot instantiate it either; OD9).

**`np.ma.MaskedArray`** (17):
- **Expose** (members `NDArray` has and `NDMaskedArray` lacks): `base`, `device`, `flags`, `getfield`, `setfield`,
  `setflags`, `to_device`;
- **Implement**: `get_fill_value`, `set_fill_value` (methods beside the `fill_value` property; `np.ma.set_fill_value`
  exists), `get_imag`, `get_real`, `sharedmask`;
- **Structured**: `toflex`, `torecords`; **N/A**: `ctypes`, `dump`, `dumps`.

**`np.flatiter`** (1): `base`: **Expose** (`np.FlatIterator.Base` is PascalCase; rename to `base`).

**`np.finfo`** (4): `iexp`, `machep`, `negep`, `nexp`: **Implement** (attributes NumPy computes from the format).

**`NpzFile`** (7): `close`, `fid`, `get`, `items`, `keys`, `values`, `zip`: **Expose** (`Close`, `Keys`, `Values`,
`Zip` exist in PascalCase; `get` and `items` are the dict API; keep `Dispose` for C#).

**`BitGenerator`** (3): `capsule`, `cffi`, `ctypes`: **N/A** (CPython C-API handles).

## 7. NumPy modules without a facade

15 public modules (the coverage tool's list, which follows NumPy's `test_public_api.py`) have no `[ModuleName]` type:

| module | names | action |
|---|---|---|
| `np.lib.stride_tricks` | `as_strided`, `sliding_window_view` | **Implement**: both are views (`Shape` with given strides; a windowed view with a stride-0-free layout); much used |
| `np.lib.array_utils` | `byte_bounds`, `normalize_axis_index`, `normalize_axis_tuple` | **Expose**: the axis normalization exists internally (the `normalize_axis` port, plan X36); `byte_bounds` from `Shape` |
| `np.lib.format` | `ARRAY_ALIGN`, `BUFFER_SIZE`, `EXPECTED_KEYS`, `GROWTH_AXIS_MAX_DIGITS`, `MAGIC_LEN`, `MAGIC_PREFIX`, `descr_to_dtype`, `drop_metadata`, `dtype_to_descr`, `header_data_from_array_1_0`, `isfileobj`, `magic`, `open_memmap`, `read_array`, `read_array_header_1_0`, `read_array_header_2_0`, `read_magic`, `write_array`, `write_array_header_1_0`, `write_array_header_2_0` | **Expose**: `IO.NpyFormat` implements them in PascalCase (`ReadArray`, `WriteArray`, `OpenMemmap`, ...; the NumPy-name audit lists them); NumPy's `__all__` is empty, but the module's documented API is these names |
| `np.exceptions` | `AxisError`, `ComplexWarning`, `DTypePromotionError`, `ModuleDeprecationWarning`, `TooHardError`, `VisibleDeprecationWarning` | **Expose** the exceptions that exist (`AxisError`, `TooHardError`, the dtype promotion error); the three warnings: N/A (no warnings model); decision OD9 |
| `np.version` | `full_version`, `git_revision`, `release`, `short_version`, `version` | **Decide** with `np.__version__` (OD9) |
| `np.rec` | `array`, `find_duplicate`, `format_parser`, `fromarrays`, `fromfile`, `fromrecords`, `fromstring`, `recarray`, `record` | **Structured** |
| `np.lib.recfunctions` | 21 names (`append_fields` … `unstructured_to_structured`) | **Structured** |
| `np.ma.mrecords` | `MaskedRecords`, `addfield`, `fromarrays`, `fromrecords`, `fromtextfile`, `mrecarray` | **Structured** |
| `np.ctypeslib` | `as_array`, `as_ctypes`, `as_ctypes_type`, `c_intp`, `load_library`, `ndpointer` | **N/A** (Python `ctypes`; the .NET analog is `nd.Unsafe` and `np.frombuffer` over a pointer) |
| `np.lib.introspect` | `opt_func_info` | **Decide** (could report NumSharp's SIMD paths) |
| `np.lib` | `Arrayterator`, `NumpyVersion`, `add_docstring`, `add_newdoc`, `tracemalloc_domain` | **N/A** except `NumpyVersion` (a version-compare helper; the coverage tool shows it anyway) |
| `np.lib.mixins` | `NDArrayOperatorsMixin` | **N/A** (Python's operator protocol) |
| `np.f2py` | `get_include`, `run_main` | **N/A** |
| `np.testing` | 49 names (`assert_allclose`, `assert_array_equal`, ...) | **N/A by the owner's coverage rule** (the inventory leaves `numpy.testing` out); a C# test helper would be separate work |
| `np.testing.overrides` | `allows_array_function_override`, `allows_array_ufunc_override`, `get_overridable_numpy_array_functions`, `get_overridable_numpy_ufuncs` | **N/A** |

## 8. Decisions for the owner

| id | question | options | affects |
|---|---|---|---|
| **OD1** | The axis parameter type (NumPy: `int`, a tuple of ints, or None) | (a) an axis type with implicit conversions from `int`, `int[]`, C# tuples and `null` (a class so `null` binds; or a `struct?`, probe that `axis: 0` and `axis: null` both bind); (b) `object axis`; (c) keep `int?` and add a second parameter (breaks NumPy's one signature) | AD1 (all axis overloads), AD9b, AD10, plan G4, B1–B8 |
| **OD2** | array_like / scalar parameters (NumPy takes lists, tuples, scalars, arrays) | (a) `NDArray` (C# arrays and scalars convert implicitly; Python-style nested lists do not); (b) `object` + NumPy's coercion (the polynomial package's `PolySequence` model: nested `object[]`, `ITuple`, typed collections, NEP 50 kinds); per parameter | AD1 (`np.array` 33, `np.asarray`, `np.arange`, `insert`, `pad`, ...), AD5 kind 2 |
| **OD3** | file parameters (NumPy takes a path or a file-like object) | one `object file` parameter that dispatches on `string` / `Stream` / `TextReader` / `TextWriter` / `byte[]`; or keep typed overloads as a documented exception | `np.load`, `np.loadtxt`, `np.savetxt`, `np.save`, `np.savez`, `np.savez_compressed`, `np.fromfile`, `ndarray.tofile` |
| **OD4** | `size=None` of the samplers | (a) keep `Shape size = default` (documented idiom; `null` refused); (b) `Shape? size = null` everywhere (every spelling binds) | AD9d (73 names), AD9e, AD1 for `np.random` |
| **OD5** | NumSharp-only API (typed accessors, `GetData<T>`, `CopyTo`, `Get*/Set*`, interop overloads, `np.evaluate`, `np.multithreading`, ...) | keep as NumSharp extensions (outside the one-signature rule, which speaks of NumPy functions), rename to make the extension visible, or remove | Appendix C, AD5 kind 2, AD7 |
| **OD6** | constructors (`new NDArray(...)`, `new NDArray<T>(...)`, `new DType(...)`, engine and `SeedSequence` constructors) | (a) must mirror NumPy's class call (`ndarray(shape, dtype, buffer, offset, strides, order)`, `dtype(dtype, align, copy, metadata)`); (b) the factories (`np.ndarray`, `np.dtype`, `np.random.PCG64`, ...) are the NumPy spelling and the constructors stay C# API | AD2 slots, AD3e, AD8, AD6 |
| **OD7** | generic typed forms of NumPy names (`np.zeros<T>`, `ndarray.item<T>`, `np.nditer<T>`, `np.ndenumerate<T>`) | remove (the max/min family's precedent) or keep the iterator / accessor forms as NumSharp extensions | AD7, AD1 |
| **OD8** | the seed parameter (NumPy: None, an int, a sequence of ints, `SeedSequence`, `BitGenerator`) | one `object seed = null` with NumPy's conversion rules (the random-API oracle pins them), or a seed type with implicit conversions | AD9c, AD1 (`np.random`, the engines) |
| **OD9** | exposing types and constants under NumPy's module paths (`np.exceptions`, `np.linalg.LinAlgError`, `np.ma.MaskedArray` / `MaskError`, `np.random.BitGenerator`, `np.flatiter`, `np.version`, `np.__version__`) | nested types / static members on the facades, or documentation only | §6, §7 |
| **OD10** | which absent names to implement (§6 Implement / Decide rows: `ufunc` objects, FP error state, DLPack, `memmap`, `fromiter`, stride tricks) | per row | §6, §7 |
| **OD11** | `subok=` now that `NDMatrix`, `NDCharArray`, `NDMaskedArray` exist | keep the plan's exclusion, or model it where a subclass exists | AD4 |

## 9. Suggested order of work

1. **Cheap and NumPy-exact, no decision needed**: the AD9a nullability fixes; the §6 Alias names (`acos` … `round`)
   and the removal of the five names NumPy 2.0 removed (AD6); the AD2 true renames; `np.ndim` / `np.shape` /
   `np.fromfunction` / `np.poly1d` exposure; `np.ma.masked_array()` and `SeedSequence` optional first parameters
   (AD8).
2. **Creation family**, after OD2 / OD6: `np.empty` / `zeros` / `ones` / `full` and the `*_like` functions (AD1,
   AD3e, AD8: one signature each, plan D1 / D2), `np.linspace` (plan D3), `np.arange`, `ndarray.astype` (plan G2).
3. **Reductions**, after OD1: one signature with NumPy's keywords and tuple axes for `sum`, `prod`, `mean`, `std`,
   `var`, the nan family, `count_nonzero`, the percentile family, `average`, `ptp`, `flip`, `squeeze`, `roll`
   (AD1, AD3b, AD8, AD9b, AD10; plan B1–B11, G4). The largest package; do it as one design with its oracle tiers.
4. **numpy.ma**: AD3a (`out=` / `where=` / `dtype=` on the 48 wrappers), the ma creation keywords (AD3e), the ma
   reductions (AD3b), the renames (AD2), `MaskedArray`'s members (§6).
5. **np.random**, after OD4 / OD8: collapse the samplers, seeds and engines (AD1, AD9c–e), keeping the random-API
   oracle green.
6. **Object APIs**, after OD6: constructors, `NDMatrix` / `NDCharArray` `astype`, `NDCharArray.partition`, `NpzFile`'s
   NumPy names, `finfo` attributes.
7. **IO** after OD3, then §6 / §7 implementations by decision (OD9, OD10).

Each step ends with §10: the marks of every touched name must be gone.

## 10. Regenerating the data and proving a fix

From `tools/api-parameter-parity/` (NumPy 2.4.2 importable; `<s>` is a scratch directory):

```bash
dotnet run --no-cache dump_modules.cs <s>/modules_dump.json <s>/types_dump.json
python -I binding_probe.py <s>/modules_dump.json <s>/types_dump.json \
    ../../src/NumSharp.Core/bin/Debug/net10.0/NumSharp.dll <s>/probe <s>/probe_results.json
python -I signature_diff.py <s>/modules_dump.json ../../docs/plans/numpy-signature-diff.md \
    ../../docs/plans/numpy-signature-diff.csv ../../docs/plans/numpy-signature-marks.md \
    --types <s>/types_dump.json --probe <s>/probe_results.json
```

- `dump_modules.cs` compiles the working tree (Debug), and the probe references that DLL: run them in this order after
  every change to a public signature. `--no-cache` matters: a file-based app otherwise reuses its cached build.
- The last command prints the totals (the `numpy`, `M`, `X`, `O`, `T`, `B`, `L`, `Q`, `no_mark` counts). A fix must
  lower the counts it targets and raise none.
- A fixed name: its row in `numpy-signature-marks.md` has no mark and an empty *compile probe* cell.
- The probe's whole surface compiles at C# 12, which is what an external `net8.0` consumer sees; a C# 12-only failure
  is a `◐` mark.
- Plan item S1 (a signature-parity gate in CI) can be built on these outputs: commit NumPy's signatures and the marks,
  and fail when a clean name gains a mark. The coverage pipeline already reports the M / X / O / T marks per credited
  API since `f949fa17` (its N mark means "credited through another name"; this document's L is the `null` mark).

## 11. Limits of the marks

- **Signatures, not behavior.** "No mark" means a NumPy call binds; it does not mean each parameter works. Behavior is
  the job of the oracle tiers and of plan phase X.
- **A call can bind the wrong overload.** The probe proves that a call compiles. When two parameters share a type, a
  positional call can bind a different NumPy parameter without an error; the S probe catches it only when the types
  differ. The diff report's *slot* column shows where the closest overload orders parameters differently.
- **Renames are found by slot.** AD2 marks the entries where a different parameter, not a renamed one, sits at NumPy's
  slot.
- **T ● counts by type name**: `view(type=...)` is a correct use of `Type` (§AD6).
- **Q covers integer sequences only**, from the stubs' annotations. Python lists of values for array_like parameters
  are decision OD2, not probed.
- **A `null` default is not a mark.** 55 names give 68 parameters the default `null` (or `default`) where NumPy's
  default is a value: `np.strings.find`'s `start=0`, `np.fft.fft2`'s `axes=(-2, -1)`, the polynomial package's
  `scl=1`, `lbnd=0` and `k=[]`, `np.char.center`'s `fillchar=' '`, `np.divmod`'s `out=(None, None)`, ... 63 of them
  are `object`, array, `NDArray` or tuple parameters, because NumPy accepts an array_like there, and C# allows no
  other default for those types: a NumPy call binds, and `null` stands for NumPy's value. The other 5 are nullable
  value types: `np.matmul`'s `bool? keepdims = null` means "not passed" (NumPy raises for any explicit value), the
  `ddof` of `np.std`, `np.var`, `ndarray.std`, `ndarray.var` (`int? ddof = null` for NumPy's 0) could be `int ddof = 0`
  in their AD1 signature. What a fix keeps: the `<param>` documentation names the NumPy value `null` stands for. The
  diff report's *null default* column lists every such parameter; its *only null defaults* count (54) is the names
  whose closest overload differs in nothing else. The reverse, NumPy's None against a C# value (6 names: `order` of
  `np.asarray`, `np.asarray_chkfinite`, `np.ndarray`, `density` of the three histograms), is the L mark (AD9a).
- **NumPy's side** is the documented signature for ndarray methods (their C forwarding accepts some keywords
  positionally that the documentation makes keyword-only), the stubs where the runtime signature forwards
  `(*args, **kwargs)`, and the wrapped callable for numpy.ma's ufunc wrappers.
- **`numpy.testing` and deprecated NumPy 1.x APIs** are out of scope by the coverage policy (§7).

## Appendix A: counts per group

| group | NumPy names | M ● | M ◦ only | X | O | T ● | T ◇ | B | L ● | L ◐ | L ◦ | Q | no mark | NumSharp-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `np` | 366 | 46 | 108 | 45 | 150 | 6 | 11 | 18 | 23 | 2 | 1 | 22 | 109 | 18 |
| `ndarray` | 57 | 9 | 1 | 5 | 24 | 1 | 4 | 7 | 4 | 0 | 0 | 10 | 22 | 83 |
| `np.char` | 23 | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 1 |
| `np.dtypes` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `np.emath` | 9 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 0 |
| `np.fft` | 18 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 0 |
| `np.linalg` | 31 | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 27 | 0 |
| `np.ma` | 211 | 130 | 3 | 31 | 10 | 0 | 0 | 8 | 1 | 0 | 0 | 17 | 69 | 4 |
| `np.polynomial.chebyshev` | 23 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 1 |
| `np.polynomial.hermite` | 23 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 1 |
| `np.polynomial.hermite_e` | 23 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 1 |
| `np.polynomial.laguerre` | 23 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 1 |
| `np.polynomial.legendre` | 23 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 20 | 1 |
| `np.polynomial.polynomial` | 21 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 18 | 1 |
| `np.polynomial.polyutils` | 6 | 0 | 0 | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| `np.random` | 61 | 0 | 0 | 1 | 55 | 0 | 1 | 1 | 2 | 11 | 33 | 0 | 5 | 3 |
| `np.random.Generator` | 45 | 0 | 0 | 0 | 36 | 0 | 0 | 0 | 0 | 0 | 38 | 0 | 5 | 1 |
| `np.strings` | 42 | 0 | 17 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 23 | 0 |
| `NDArray<T>` | 2 | 2 | 0 | 1 | 2 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 3 |
| `np.matrix` | 53 | 10 | 1 | 0 | 19 | 1 | 0 | 2 | 0 | 0 | 0 | 13 | 21 | 4 |
| `np.char.chararray` | 32 | 3 | 1 | 2 | 12 | 1 | 0 | 2 | 0 | 0 | 0 | 1 | 17 | 4 |
| `np.ma.MaskedArray` | 59 | 30 | 0 | 1 | 2 | 0 | 0 | 3 | 0 | 0 | 0 | 13 | 25 | 2 |
| `np.dtype` | 4 | 1 | 0 | 0 | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 2 | 5 |
| `np.nditer` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 3 |
| `np.broadcast` | 2 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 3 |
| `np.flatiter` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| `np.ndindex` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 |
| `np.ndenumerate` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |
| `np.poly1d` | 3 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 5 |
| `np.vectorize` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| `np.finfo` | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| `np.iinfo` | 1 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| `NpzFile` | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| `SeedSequence` | 3 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 2 | 1 |
| `BitGenerator` | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 0 |
| `MT19937` | 3 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `PCG64` | 3 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `PCG64DXSM` | 3 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `Philox` | 3 | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| `SFC64` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| **total** | **1,191** | **234** | **132** | **90** | **360** | **10** | **17** | **45** | **31** | **19** | **73** | **77** | **508** | **168** |

## Appendix B: names with no mark

508 names match NumPy in every respect the marks check (a NumPy call of each binds at C# 12 and C# 14):

- **`np`** (109): `allclose`, `angle`, `argwhere`, `array_equiv`, `array_str`, `bartlett`, `bincount`, `blackman`, `block`, `broadcast_shapes`, `clip`, `column_stack`, `common_type`, `compress`, `convolve`, `copy`, `corrcoef`, `correlate`, `cov`, `cross`, `datetime_data`, `diag`, `diag_indices`, `diag_indices_from`, `diagflat`, `diagonal`, `diff`, `digitize`, `dot`, `dstack`, `ediff1d`, `extract`, `fill_diagonal`, `fix`, `flatnonzero`, `fliplr`, `flipud`, `format_float_positional`, `format_float_scientific`, `get_printoptions`, `getbufsize`, `hamming`, `hanning`, `i0`, `imag`, `inner`, `isclose`, `isfortran`, `isin`, `isneginf`, `isposinf`, `iterable`, `ix_`, `kaiser`, `kron`, `mask_indices`, `matrix_transpose`, `may_share_memory`, `min_scalar_type`, `nan_to_num`, `nancumprod`, `nancumsum`, `nonzero`, `outer`, `packbits`, `permute_dims`, `poly`, `polyadd`, `polyder`, `polydiv`, `polyfit`, `polymul`, `polysub`, `put_along_axis`, `real`, `real_if_close`, `resize`, `rollaxis`, `roots`, `rot90`, `select`, `setbufsize`, `setdiff1d`, `setxor1d`, `shares_memory`, `sinc`, `sort`, `sort_complex`, `swapaxes`, `take_along_axis`, `trace`, `trapezoid`, `tril`, `tril_indices`, `tril_indices_from`, `triu`, `triu_indices`, `triu_indices_from`, `typename`, `union1d`, `unique`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values`, `unpackbits`, `unstack`, `vander`, `vdot`
- **`ndarray`** (22): `__contains__`, `__iter__`, `__len__`, `byteswap`, `clip`, `compress`, `copy`, `cumprod`, `cumsum`, `diagonal`, `fill`, `flatten`, `nonzero`, `put`, `round`, `setflags`, `sort`, `swapaxes`, `to_device`, `tobytes`, `tolist`, `trace`
- **`np.char`** (20): `array`, `asarray`, `center`, `chararray`, `equal`, `expandtabs`, `greater`, `greater_equal`, `less`, `less_equal`, `ljust`, `multiply`, `not_equal`, `partition`, `rjust`, `rpartition`, `rsplit`, `split`, `splitlines`, `zfill`
- **`np.emath`** (9): `arccos`, `arcsin`, `arctanh`, `log`, `log10`, `log2`, `logn`, `power`, `sqrt`
- **`np.fft`** (14): `fft`, `fft2`, `fftn`, `hfft`, `ifft`, `ifft2`, `ifftn`, `ihfft`, `irfft`, `irfft2`, `irfftn`, `rfft`, `rfft2`, `rfftn`
- **`np.linalg`** (27): `cholesky`, `cond`, `cross`, `det`, `diagonal`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `inv`, `lstsq`, `matmul`, `matrix_norm`, `matrix_power`, `matrix_rank`, `matrix_transpose`, `outer`, `pinv`, `qr`, `slogdet`, `solve`, `svd`, `svdvals`, `tensorinv`, `tensorsolve`, `trace`, `vecdot`
- **`np.ma`** (69): `allclose`, `allequal`, `alltrue`, `anom`, `anomalies`, `append`, `clump_masked`, `clump_unmasked`, `column_stack`, `common_fill_value`, `compress_cols`, `compress_nd`, `compress_rowcols`, `compress_rows`, `concatenate`, `convolve`, `corrcoef`, `correlate`, `cov`, `diag`, `diagflat`, `diagonal`, `diff`, `dstack`, `ediff1d`, `filled`, `fix_invalid`, `flatnotmasked_contiguous`, `flatnotmasked_edges`, `flatten_mask`, `getmask`, `harden_mask`, `ids`, `in1d`, `inner`, `innerproduct`, `intersect1d`, `isMA`, `isMaskedArray`, `is_masked`, `isarray`, `isin`, `mask_rowcols`, `masked_all`, `masked_invalid`, `masked_where`, `maximum`, `minimum`, `nonzero`, `notmasked_contiguous`, `notmasked_edges`, `outer`, `outerproduct`, `polyfit`, `power`, `put`, `putmask`, `ravel`, `repeat`, `resize`, `set_fill_value`, `setdiff1d`, `setxor1d`, `soften_mask`, `sometrue`, `swapaxes`, `transpose`, `union1d`, `vander`
- **`np.polynomial.chebyshev`** (20): `cheb2poly`, `chebadd`, `chebcompanion`, `chebder`, `chebdiv`, `chebfromroots`, `chebgrid2d`, `chebgrid3d`, `chebint`, `chebline`, `chebmul`, `chebmulx`, `chebroots`, `chebsub`, `chebval`, `chebval2d`, `chebval3d`, `chebvander2d`, `chebvander3d`, `poly2cheb`
- **`np.polynomial.hermite`** (20): `herm2poly`, `hermadd`, `hermcompanion`, `hermder`, `hermdiv`, `hermfromroots`, `hermgrid2d`, `hermgrid3d`, `hermint`, `hermline`, `hermmul`, `hermmulx`, `hermroots`, `hermsub`, `hermval`, `hermval2d`, `hermval3d`, `hermvander2d`, `hermvander3d`, `poly2herm`
- **`np.polynomial.hermite_e`** (20): `herme2poly`, `hermeadd`, `hermecompanion`, `hermeder`, `hermediv`, `hermefromroots`, `hermegrid2d`, `hermegrid3d`, `hermeint`, `hermeline`, `hermemul`, `hermemulx`, `hermeroots`, `hermesub`, `hermeval`, `hermeval2d`, `hermeval3d`, `hermevander2d`, `hermevander3d`, `poly2herme`
- **`np.polynomial.laguerre`** (20): `lag2poly`, `lagadd`, `lagcompanion`, `lagder`, `lagdiv`, `lagfromroots`, `laggrid2d`, `laggrid3d`, `lagint`, `lagline`, `lagmul`, `lagmulx`, `lagroots`, `lagsub`, `lagval`, `lagval2d`, `lagval3d`, `lagvander2d`, `lagvander3d`, `poly2lag`
- **`np.polynomial.legendre`** (20): `leg2poly`, `legadd`, `legcompanion`, `legder`, `legdiv`, `legfromroots`, `leggrid2d`, `leggrid3d`, `legint`, `legline`, `legmul`, `legmulx`, `legroots`, `legsub`, `legval`, `legval2d`, `legval3d`, `legvander2d`, `legvander3d`, `poly2leg`
- **`np.polynomial.polynomial`** (18): `polyadd`, `polycompanion`, `polyder`, `polydiv`, `polyfromroots`, `polygrid2d`, `polygrid3d`, `polyint`, `polyline`, `polymul`, `polymulx`, `polyroots`, `polysub`, `polyval`, `polyval2d`, `polyval3d`, `polyvander2d`, `polyvander3d`
- **`np.polynomial.polyutils`** (2): `as_series`, `getdomain`
- **`np.random`** (5): `Generator`, `bytes`, `get_bit_generator`, `set_bit_generator`, `shuffle`
- **`np.random.Generator`** (5): `bytes`, `permuted`, `shuffle`, `spawn`, `(constructor)`
- **`np.strings`** (23): `capitalize`, `center`, `count`, `endswith`, `expandtabs`, `find`, `index`, `ljust`, `lower`, `lstrip`, `partition`, `replace`, `rfind`, `rindex`, `rjust`, `rpartition`, `rstrip`, `startswith`, `strip`, `swapcase`, `title`, `upper`, `zfill`
- **`np.matrix`** (21): `__imul__`, `__ipow__`, `__iter__`, `__mul__`, `__rmul__`, `__rpow__`, `argpartition`, `byteswap`, `conj`, `conjugate`, `copy`, `diagonal`, `flatten`, `getA`, `getA1`, `getH`, `getI`, `getT`, `getfield`, `swapaxes`, `to_device`
- **`np.char.chararray`** (17): `__iter__`, `byteswap`, `center`, `copy`, `diagonal`, `expandtabs`, `flatten`, `getfield`, `ljust`, `rjust`, `rpartition`, `rsplit`, `split`, `splitlines`, `swapaxes`, `to_device`, `zfill`
- **`np.ma.MaskedArray`** (25): `anom`, `argpartition`, `byteswap`, `compressed`, `conj`, `conjugate`, `diagonal`, `fill`, `filled`, `flatten`, `harden_mask`, `ids`, `iscontiguous`, `nonzero`, `put`, `ravel`, `repeat`, `searchsorted`, `shrink_mask`, `soften_mask`, `swapaxes`, `tobytes`, `tofile`, `tolist`, `unshare_mask`
- **`np.dtype`** (2): `__getitem__`, `__len__`
- **`np.nditer`** (8): `close`, `copy`, `debug_print`, `enable_external_loop`, `iternext`, `remove_axis`, `remove_multi_index`, `reset`
- **Others** (12): `np.broadcast.reset`, `np.flatiter.copy`, `np.poly1d.deriv`, `np.poly1d.integ`, `np.finfo` (constructor), `SeedSequence.generate_state`, `SeedSequence.spawn`, `BitGenerator.spawn`, `MT19937.jumped`, and `advance` of `PCG64`, `PCG64DXSM`, `Philox`

## Appendix C: NumSharp-only names

168 names NumPy does not have (decision OD5 decides their future). Eight are names NumPy no longer has (removed by
2.0: `np.find_common_type`, `np.issubsctype`, `np.maximum_sctype`, `np.sctype2char`, `np.issctype`, `np.asscalar`,
`np.round_`, `ndarray.itemset`): delete them (AD6, §6).

- **with several overloads (63)**: `np`: `are_broadcastable` 5, `asscalar` 6 (removed in NumPy 2.0), `average_returned` 2 (NumPy: `average(returned=True)`), `common_type_code` 2, `evaluate` 2, `find_common_type` 9 (removed in NumPy 2.0), `flat` 2, `indices_sparse` 3 (NumPy: `indices(sparse=True)`), `load_npy` 3, `load_npz` 3; `ndarray`: `CloneData` 2, `CopyTo` 6, `GetAtIndex` 2, `GetBoolean` … `GetUInt64` (15 typed getters, 2 each), `GetData` 4, `GetValue` 4, `ReplaceData` 6, `Scalar` 4, `SetAtIndex` 2, `SetBoolean` … `SetUInt64` (15 typed setters, 2 each), `SetData` 5, `SetValue` 4, `ToString` 2, `itemset` 4 (removed in NumPy 2.0), `reshape_unsafe` 5, `roll` 4, `unique` 2; `np.random`: `bernoulli` 2, `tomaxint` 2; `NDArray<T>`: `reshape_unsafe` 5; `np.ma.MaskedArray`: `ToString` 2; `np.dtype`: `Equals` 5, `From` 4, `ToString` 2; `np.poly1d`: `Call` 2
- **with a T mark and one overload**: `np.issubsctype` (removed in NumPy 2.0), `np.maximum_sctype` (removed), `np.sctype2char` (removed); `ndarray`: `AsGeneric`, `AsOrMakeGeneric`, `Data`, `FromMultiDimArray`, `MakeGeneric`, `ToArray`, `ToJaggedArray`, `ToMuliDimArray`, `AsEnumerable`; `np.nditer_chunks`; `np.flatiter.AsTyped`
- **no mark (91)**: `np`: `issctype` (removed in NumPy 2.0), `multithreading`, `round_` (removed in NumPy 2.0; §6 Alias), `save_version`; `ndarray`: `AsString`, `AsStringArray`, `Clone`, `Contains`, `Dispose`, `Equals`, `FromString`, `GetEnumerator`, `GetHashCode`, `GetIndices`, `GetNDArrays`, `GetString`, `GetStringAt`, `Normalize`, `SetIndices`, `SetString`, `SetStringAt`, `__hash__`, `array_equal`, `convolve`, `correlate`, `delete`, `dstack`, `hstack`, `matrix_power`, `negate`, `negative`, `positive`, `vstack`; `np.char`: `ToString`; `np.ma`: `average_returned`, `flatten`, `moveaxis`, `shrink_mask`; `np.polynomial.*`: `chebvalnd`, `hermvalnd`, `hermevalnd`, `lagvalnd`, `legvalnd`, `polyvalnd`; `np.random`: `ToString`; `np.random.Generator`: `ToString`; `NDArray<T>`: `Clone`, `GetAtIndex`; `np.matrix`, `np.char.chararray`: `Clone`, `Equals`, `GetEnumerator`, `GetHashCode`; `np.ma.MaskedArray`: `putmask`; `np.dtype`: `GetHashCode`, `GetTypeCode`; `np.nditer`, `np.broadcast`, `np.ndenumerate`: `Dispose`, `GetEnumerator`, `MoveNext`; `np.ndindex`: the same and `AsSpans`; `np.flatiter`: `GetEnumerator`, `next`; `np.poly1d`: `Dispose`, `Equals`, `GetHashCode`, `ToString`; `np.vectorize`: `Call`, `CallMany`; `np.finfo`, `np.iinfo`, `SeedSequence`: `ToString`; `NpzFile`: `Close`, `ContainsKey`, `Dispose`, `GetEnumerator`, `GetRawBytes`, `IsArray`, `ToString`, `TryGetValue`

Most of the no-mark NumSharp-only names are .NET protocol members (`Dispose`, `GetEnumerator`, `MoveNext`,
`Equals`, `GetHashCode`, `ToString`, `Clone`): they stay.
