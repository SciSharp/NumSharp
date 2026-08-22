# Journey3 touched-function oracle receipt

Baseline: repository mainline `master` at `61506de1`; journey branch snapshot `aaa41ef2`, plus
the current oracle-completeness worktree. This repository names its mainline `master` (there is no
local `main` ref).

The inventory rule is intentionally conservative: every public NumPy callable owned by a
`NumSharp.Core` source file changed in `master...journey3` is considered touched, even when the
diff changed only a neighbouring overload or shared helper in that file. This produces **186
callables**, and `Journey3TouchedOracleCoverageTests` now requires a committed direct corpus key
(or, for a random distribution, its direct stream key) for every one: **186/186 direct**.

“Added-file” and “modified-file” below describe the owning source file relative to the baseline;
they do not claim every method in an added file was written in one commit.

## Added-file surfaces (124)

- `np` (75): `acosh`, `angle`, `arccosh`, `arcsinh`, `arctanh`, `argpartition`, `asinh`,
  `atanh`, `bincount`, `choose`, `conj`, `conjugate`, `corrcoef`, `correlate`, `cov`, `cross`,
  `diag`, `diag_indices`, `diag_indices_from`, `diagflat`, `digitize`, `einsum`, `einsum_path`,
  `fill_diagonal`, `fromstring`, `imag`, `inner`, `intersect1d`, `isfortran`, `isin`, `ix_`,
  `kron`, `lexsort`, `loadtxt`, `mask_indices`, `matvec`, `nanargmax`, `nanargmin`, `nditer`,
  `nested_iters`, `partition`, `poly`, `polyadd`, `polyder`, `polydiv`, `polyfit`, `polyint`,
  `polymul`, `polysub`, `polyval`, `real`, `roots`, `savetxt`, `select`, `setdiff1d`,
  `setxor1d`, `sort_complex`, `take_along_axis`, `tensordot`, `tri`, `tril`, `tril_indices`,
  `tril_indices_from`, `triu`, `triu_indices`, `triu_indices_from`, `union1d`, `unique_all`,
  `unique_counts`, `unique_inverse`, `unique_values`, `vander`, `vdot`, `vecdot`, `vecmat`.
- `np.linalg` (31): `cholesky`, `cond`, `cross`, `det`, `diagonal`, `eig`, `eigh`, `eigvals`,
  `eigvalsh`, `inv`, `lstsq`, `matmul`, `matrix_norm`, `matrix_power`, `matrix_rank`,
  `matrix_transpose`, `multi_dot`, `norm`, `outer`, `pinv`, `qr`, `slogdet`, `solve`, `svd`,
  `svdvals`, `tensordot`, `tensorinv`, `tensorsolve`, `trace`, `vecdot`, `vector_norm`.
- `np.fft` (18): `fft`, `fft2`, `fftfreq`, `fftn`, `fftshift`, `hfft`, `ifft`, `ifft2`,
  `ifftn`, `ifftshift`, `ihfft`, `irfft`, `irfft2`, `irfftn`, `rfft`, `rfft2`, `rfftfreq`,
  `rfftn`.

## Modified-file surfaces (62)

- `np` (58): `all`, `any`, `arange`, `array`, `array_split`, `asanyarray`, `asarray`,
  `ascontiguousarray`, `asfortranarray`, `block`, `broadcast_arrays`, `broadcast_to`, `clip`,
  `concatenate`, `copyto`, `cumprod`, `cumsum`, `delete`, `dot`, `empty`, `empty_like`,
  `expand_dims`, `eye`, `frombuffer`, `fromfile`, `full`, `full_like`, `identity`, `insert`,
  `isclose`, `isfinite`, `isinf`, `isnan`, `isscalar`, `iterable`, `linspace`, `matmul`,
  `meshgrid`, `mintypecode`, `nanmean`, `nanstd`, `nanvar`, `ones`, `ones_like`, `outer`,
  `place`, `ptp`, `put`, `ravel_multi_index`, `require`, `searchsorted`, `split`, `squeeze`,
  `take`, `trace`, `unique`, `zeros`, `zeros_like`.
- `np.random` (4): `get_state`, `seed`, `set_state`, `shuffle`.

## The 15 direct gaps that this pass closed

| Function(s) | New direct cases | Oracle technique |
|---|---:|---|
| `acosh`, `asinh`, `atanh`, `conj` | 364 each | Replay the actual alias entry point across the full dtype × layout unary matrix. |
| `block` | 4 | Record the real public op name for nested 2×2 block assembly. |
| `empty` | 14 | Allocate with `empty`, immediately fill both NumPy and NumSharp results with zero, then compare dtype/shape/writeability/bytes; flags remain independently gated. |
| `empty_like` | 42 | Same post-initialization technique across C/F/strided prototypes. |
| `fromfile` | 14 | NumPy reads a real temporary binary file; replay feeds the exact serialized operand bytes through NumSharp's public Stream overload. |
| `loadtxt` | 14 | Parse a committed text artifact and compare dtype/shape/bytes. |
| `savetxt` | 14 | Serialize the same array and compare emitted text verbatim. |
| `nditer` | 792 | Rename the materialized value-stream cases to the public operation while retaining index/chunk companion traces. |
| `nested_iters` | 6 | Materialize two- and three-level recursive iteration streams across real, integer, and complex dtypes. |
| `seed` | 4 | Seed with boundary values and compare the following portable MT19937 draw bytes. |
| `get_state` | 4 | Compare a canonical full trace: algorithm, position, Gaussian-cache bits, and all 624 state words. |
| `set_state` | 4 | Restore captured 624-word NumPy states at four stream positions and compare the following draw bytes. |

## Shared-kernel changes

Journey3 also changed shared cast, binary, unary, reduction, search, selection, and matmul kernels.
Those implementation files do not own one public method, so they are not inflated into the
186-name source-owner receipt. They are covered by the whole-corpus matrices: all registered ops
replay managed C#; BLAS-eligible products additionally replay backend-on with semantic outcome
deduplication; dtype/layout/special/error/out-where/precision and metamorphic tiers probe the
shared execution paths orthogonally.
