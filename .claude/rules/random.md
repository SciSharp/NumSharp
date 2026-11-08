---
paths:
  - "src/NumSharp.Core/RandomSampling/**"
  - "src/NumSharp.Core/Indexing/np.take.FancyAxis.cs"
  - "test/NumSharp.Tests/{Random,RandomSampling}/**"
  - "test/oracle/gen_random_oracle.py"
---

# `np.random`

The legacy API and `Generator` match NumPy 1-to-1: MT19937 (NumPy's Mersenne Twister, `RandomSampling/MT19937.cs`) with
the Marsaglia-polar Gaussian and its cached-Gaussian carry, NumPy's state-tuple `get_state`/`set_state` format, and
PCG64/`default_rng`, so a seed produces NumPy 2.4.2's byte-identical sequences. Oracles: the `random_parity` tiers and
the random-API oracle (`oracle.md`).

## Integer types

Gate `RandomSampling/RandomTypeParity.Test.cs`.
- The legacy integers are C `long` in NumPy, and NumSharp models one width for all of them, the LP64 (Linux/macOS) int64
  (`NumPyRandom.LegacyLong`): `randint`'s default dtype, `random_integers`, `permutation(n)`, `choice`'s indices,
  `multinomial`'s counts and the discrete samplers, with `long` counts in. Windows NumPy returns the same values as int32
  wherever it accepts the input (it rejects bounds past 2**31), which is why the Windows-authored `random_parity` /
  `generator_parity` corpora record every legacy int widened to int64 (`gen_oracle.py` `_RND_INT64_CAST`, the
  `random_integers` branch). Where NumPy's messages differ by platform (`Python int too large to convert to C long`
  against `int too big to convert`, `value too large to convert to uint32_t` against `…C unsigned long`), the Linux text
  is pinned.
- `SeedSequence` carries NumPy's member types (`spawn_key` `BigInteger[]`, `pool_size` `long`, `n_children_spawned`
  `uint`, `pool`/`generate_state(long n_words, …)` NDArrays) and NumPy's string rule (`"0x…"` hex, a leading digit
  DECIMAL, no octal: `"012"` is 12). `Seed` is a `uint`.
- After `set_bit_generator` swaps the singleton to a non-MT19937 engine, `np.random.seed(x)` re-seeds it as NumPy's MODULE
  function does (`engine.state = type(engine)(x).state`, the cached Gaussian kept), only for `np.random` itself; another
  `RandomState` over such an engine refuses (the method rule).

## Call spellings port verbatim

The random-API oracle binds overloads by reflection, so call shapes are pinned at compile time instead
(`RandomSampling/RandomApiWholeness.Test.cs`):
1. The nine all-default legacy samplers (`normal`, `uniform`, `exponential`, `poisson`, `gumbel`, `laplace`, `logistic`,
   `lognormal`, `rayleigh`) take `size:` alone or any keyword subset: the size overload carries NumPy's defaults and the
   one-draw overload's parameters are REQUIRED, so the two never compete.
2. numpy.random's classes through the module (`np.random.PCG64(42)`, `np.random.SeedSequence(...)`,
   `np.random.Generator(bit_generator)`): `NumPyRandom` factories mirroring every constructor overload of the five
   engines, `SeedSequence` and `Generator` (`np.random.classes.cs`; no `BitGenerator`, which NumPy refuses to
   instantiate). Trap: inside `NumPyRandom` the factories SHADOW the class names in expressions and crefs; spell
   `global::NumSharp.X.Member` / `cref="NumSharp.X"`.
3. A bare `null` seed binds (`[OverloadResolutionPriority(1)]` on the overload that takes it, polyfilled for net8.0 in
   `Assembly/OverloadResolutionPriorityAttribute.cs` and honored across assemblies), and a typed null seed ARRAY is
   NumPy's `None` too: fresh entropy.
4. `size: default` is NumPy's `size=None` (`Shape` is a struct, so `size: null` cannot compile) and binds the `Shape`
   overload; the `int[]`/`long[]`/`long` size shims carry priority -1 (otherwise
   `multivariate_normal(mean, cov, size: default)` would bind the `long` shim: size 0, an empty result).
5. `integers` and `randint` take NumPy's ARRAY bounds (`BoundedIntegers.Broadcast.cs`, the `_rand_<dtype>` broadcast path:
   `can_cast`-skipped bound checks in NumPy's order and words, buffered words carried across positions, the
   size-smaller-than-broadcast and non-C-layout-float-bounds quirks) and `BigInteger` (Python-int) bounds:
   `g.integers(0, BigInteger.Pow(2, 64), dtype: np.uint64)` is the full-range idiom. One per-position loop serves every
   width (per-width chunk loops, NumPy's own per-dtype `_rand_<dtype>_broadcast` structure, would be faster for the narrow
   widths; the np-function rule forbids per-dtype functions). The uint8 Lemire threshold `(255 - rng) % (rng + 1)` is a
   256-entry table (`BoundedIntegers.LemireThreshold8`).

## The kind of array returned

Gate `RandomSampling/RandomResultOwnership.Test.cs`.
- Every sampler of both APIs returns a fresh writeable C-ordered array that owns its data, `choice` without replacement
  and `multivariate_normal` included: NumPy ends them with an in-place `idx.shape = shape` / `x.shape = final_shape`, so
  they allocate the final shape (`shape.AsClean()`) or assign `result.shape`, never `reshape` (a view).
- NumPy's two exceptions are kept: the legacy `RandomState.choice(int, size, replace=False)` without `p` is a slice of
  `permutation(pop_size)` (a view), and `Generator.permutation(x, axis > 0)` is `arr[tuple(slices)]`, NumPy's fancy-index
  layout: a VIEW of a fresh buffer whose first axis is the permuted one (MapIter's buffer transposed back: F-contiguous
  for 2-D, permuted strides beyond). `np.FancyIndexAlongAxis` (`Indexing/np.take.FancyAxis.cs`) builds it with one tiled
  IL gather (`DirectILKernelGenerator.TakeLeading.cs`: tiles of about 1 MB of source rows, each index writing one
  contiguous run, typed moves 4× unrolled or `cpblk`). For a non-C source of three or more dims NumPy also keeps the
  source's memory order for the other axes, which `np.FancyIndexAlongAxis` (it copies the source to C first) does not;
  NumSharp's general fancy index does not follow NumPy's layout rules either (`a[:, idx]` is C-ordered and owning). Both
  are pinned in `OpenBugs.FancyIndex.cs`.
