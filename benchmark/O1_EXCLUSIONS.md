# O(1)-in-N benchmark exclusions

This ledger defines operation-matrix scenarios whose work is independent of the input element
count `N`. They remain benchmarked and their raw NumPy/NumSharp timings remain visible, but they are
always `negligible` and never contribute to a geomean, win/loss count, performance band, ranking,
heatmap score, scenario score, backend score, or headline.

`O(ndim)` is `O(1)` for this purpose because the benchmark rank is fixed. This is a claim about the
exact benchmark scenarios, not every possible call to the API: for example, the measured
`reshape`/`ravel` inputs are contiguous, `conj` uses a real dtype, `ix_` uses integer inputs, and
split/unstack use fixed output counts.

## Excluded scenarios

| Class | Benchmarked operations | Why independent of `N` |
|---|---|---|
| Identity/view conversion | `np.real`, `asarray`, `asanyarray`, `asmatrix`, `frombuffer`, `atleast_1d/2d/3d` | Returns the input or a wrapper/view over the same storage. |
| Shape/stride views | `broadcast_arrays`, `broadcast_to`, 2-D `diag`, `diagonal`, `expand_dims`, `flip`, `fliplr`, `flipud`, `matrix_transpose`, `moveaxis`, `permute_dims`, contiguous `ravel`/`reshape`, `rollaxis`, `rot90`, `squeeze`, `swapaxes`, `transpose`, `linalg.diagonal`, `linalg.matrix_transpose` | Rewrites a fixed-size shape/stride/offset descriptor; no element loop. |
| Index/view construction | `ix_`; the five slice-creation cases; `ndarray.T`, `diagonal`, `getfield`, real `conj`/`conjugate`, `ravel`, `reshape`, `squeeze`, `swapaxes`, `transpose`, `view`, and `to_device(cpu)` | Creates one/few aliases or reads one scalar (`item`); storage is shared. |
| Fixed-count view collections | `array_split(..., 7)`, `split/hsplit/vsplit/dsplit(..., 10)`, `unstack` with axis length 10 | O(k) view creation with `k` fixed by the benchmark, independent of source element count. |
| Metadata/scalar dispatch | `size`, `iscomplexobj`, `isrealobj`, `isscalar`, `isfortran`, `can_cast`, `common_type`, `isdtype`, `issubdtype`, `min_scalar_type`, `mintypecode`, `promote_types`, `result_type`, `iterable`, `nested_iters`, scalar float formatting, and print-option get/set/context construction | Reads fixed dtype/shape/global metadata or constructs fixed iterator/context state. |
| Shape-only planner | `einsum_path` for the fixed two-operand expression | Reads shapes/dtypes and plans one contraction; it does not execute the contraction or traverse values. |

The executable policy is [`scripts/credibility.py`](scripts/credibility.py). Both merge stages and
both dashboard JavaScript scopes are regression-tested against that policy.

## Structural proof from NumPy

The vendored NumPy 2.x source is authoritative:

- [`np.real`](../src/numpy/numpy/lib/_type_check_impl.py) returns `val.real` directly.
- [`flip`](../src/numpy/numpy/lib/_function_base_impl.py) explicitly documents a view performed in
  constant time; `rot90` composes flips and transpose views.
- [`expand_dims` and split`](../src/numpy/numpy/lib/_shape_base_impl.py) return reshaped/sliced views;
  split loops over the requested section count, not array elements.
- [`unstack`](../src/numpy/numpy/_core/shape_base.py) is `tuple(moveaxis(...))`; the benchmark fixes
  the leading axis at 10.
- [`broadcast_to`/`broadcast_arrays`](../src/numpy/numpy/lib/_stride_tricks_impl.py) return views,
  including stride-zero dimensions.
- [`moveaxis`/`rollaxis`](../src/numpy/numpy/_core/numeric.py) and
  [`transpose`/`swapaxes`/`squeeze`](../src/numpy/numpy/_core/fromnumeric.py) only permute shape and
  stride metadata for ndarray inputs.
- [`linalg.matrix_transpose`](../src/numpy/numpy/linalg/_linalg.py) delegates to `swapaxes`; diagonal
  extraction is a strided view.
- [`einsum_path`](../src/numpy/numpy/_core/einsumfunc.py) states its complexity in the number of
  contraction terms. The benchmark fixes that count at two.

## Empirical cross-check

On NumPy 2.4.2, a Bash `python <<'EOF'` sweep timed 75 representative calls at `N=1,000`, `100,000`,
and `10,000,000` (a 10,000x element-count increase):

| Evidence | Result |
|---|---:|
| Storage-sharing/fixed-count assertions | 303 / 303 passed |
| Median timing max/min across the three sizes | 1.44x |
| 95th percentile timing max/min | 2.85x |
| Worst timing max/min | 3.16x (`broadcast_arrays`) |

These small factors are wrapper allocation, dispatch, cache, and timer noise—not growth with `N`.
For comparison, an O(N) operation should approach 10,000x over the same size sweep.

The captured NumSharp side of snapshot `2026-08-29_9b200075` independently shows 78 three-tier
O(1) groups: 16 were below BenchmarkDotNet's resolution, while the other 62 had median 1.29x,
95th-percentile 1.87x, and maximum 2.40x timing spread over the same 10,000x `N` increase.

As an exhaustiveness check, the same snapshot was scanned in the other direction: require all
three tiers, exclude this ledger, then flag cases whose NumPy **and** NumSharp timings both stayed
within 3.5x over the 10,000x `N` increase. Only four groups remained: `np.trace`,
`np.linalg.trace`, `ndarray.trace`, and `diag_indices_from`. None is O(1): for the benchmark's square
matrices they consume or construct a diagonal of length `Theta(sqrt(N))`, so they correctly remain
in the performance dataset.

## Deliberate non-members

- `empty`, `zeros`, and `zeros_like` are allocation scenarios, not view/metadata operations.
- `all` and `any` can short-circuit on the current random data, but their worst case is O(N); they
  are not asserted O(1).
- `imag`, `iscomplex`, and `isreal` on real arrays allocate output arrays and are not asserted O(1).
- Copying `ravel`/`reshape` variants would be O(N); only the explicitly contiguous view scenarios
  are excluded.
