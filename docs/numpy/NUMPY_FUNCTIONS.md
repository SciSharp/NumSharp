# NumPy 2.4.2 vs NumSharp: Function Comparison

This document provides a comprehensive comparison of `np.*` functions between NumPy 2.4.2 and NumSharp, identifying missing functions and prioritizing them by popularity.

## Summary

| Category | NumPy 2.4.2 | NumSharp | Missing |
|----------|-------------|----------|---------|
| Callable np.* functions | 404 | 177 | 239 |
| Coverage | 100% | 44% | 56% |

**Note:** NumPy count excludes type aliases, constants, modules, and dtype classes. NumSharp count includes static `np.*` functions only (instance methods listed separately below).

---

## NumSharp Supported Functions (177 static functions)

### Array Creation (31)
`arange`, `array`, `asanyarray`, `asarray`, `broadcast`, `broadcast_arrays`, `broadcast_to`, `concatenate`, `copy`, `dstack`, `empty`, `empty_like`, `eye`, `frombuffer`, `FromMultiDimArray`, `FromString`, `full`, `full_like`, `hstack`, `identity`, `linspace`, `meshgrid`, `mgrid`, `ndarray`, `ones`, `ones_like`, `Scalar`, `stack`, `vstack`, `zeros`, `zeros_like`

### Math - Arithmetic (55)
`abs`, `absolute`, `add`, `arccos`, `arcsin`, `arctan`, `arctan2`, `around`, `cbrt`, `ceil`, `clip`, `convolve`, `cos`, `cosh`, `deg2rad`, `degrees`, `divide`, `exp`, `exp2`, `expm1`, `floor`, `floor_divide`, `fmax`, `fmin`, `invert`, `bitwise_not`, `left_shift`, `log`, `log10`, `log1p`, `log2`, `maximum`, `minimum`, `mod`, `modf`, `multiply`, `negative`, `positive`, `power`, `rad2deg`, `radians`, `reciprocal`, `right_shift`, `round`, `round_`, `sign`, `sin`, `sinh`, `sqrt`, `square`, `subtract`, `tan`, `tanh`, `true_divide`, `trunc`

### Reductions (25)
`all`, `amax`, `amin`, `any`, `argmax`, `argmin`, `count_nonzero`, `cumprod`, `cumsum`, `max`, `mean`, `min`, `nanmax`, `nanmean`, `nanmin`, `nanprod`, `nanstd`, `nansum`, `nanvar`, `prod`, `size`, `std`, `sum`, `var`

### Logic & Comparison (27)
`allclose`, `are_broadcastable`, `array_equal`, `can_cast`, `equal`, `greater`, `greater_equal`, `isclose`, `iscomplex`, `iscomplexobj`, `isfinite`, `isinf`, `isnan`, `isreal`, `isrealobj`, `isscalar`, `issctype`, `isdtype`, `issubdtype`, `issubsctype`, `less`, `less_equal`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `not_equal`

### Shape Manipulation (17)
`asscalar`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `copyto`, `expand_dims`, `moveaxis`, `ravel`, `repeat`, `reshape`, `roll`, `rollaxis`, `squeeze`, `swapaxes`, `transpose`, `unique`

### Sorting & Searching (6)
`argmax`, `argmin`, `argsort`, `nonzero`, `searchsorted`

### Splitting (5)
`array_split`, `dsplit`, `hsplit`, `split`, `vsplit`

### Linear Algebra (3)
`dot`, `matmul`, `outer`

### Random (np.random.*) (46)
`bernoulli`, `beta`, `binomial`, `chisquare`, `choice`, `dirichlet`, `exponential`, `f`, `gamma`, `geometric`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `logseries`, `multinomial`, `multivariate_normal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `permutation`, `poisson`, `power`, `rand`, `randint`, `randn`, `random`, `random_sample`, `rayleigh`, `seed`, `shuffle`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`, `uniform`, `vonmises`, `wald`, `weibull`, `zipf`

### File I/O (4)
`fromfile`, `load`, `save`, `tofile`

### Type Info (13)
`can_cast`, `common_type`, `common_type_code`, `dtype`, `finfo`, `find_common_type`, `iinfo`, `maximum_sctype`, `min_scalar_type`, `mintypecode`, `promote_types`, `result_type`, `sctype2char`

---

## NDArray Instance Methods

These methods are available on NDArray objects (equivalent to `ndarray.method()` in NumPy):

| Method | Description |
|--------|-------------|
| `amax()` / `max()` | Maximum value |
| `amin()` / `min()` | Minimum value |
| `argmax(axis)` | Indices of maximum |
| `argmin(axis)` | Indices of minimum |
| `argsort(axis)` | Indices that would sort |
| `astype(dtype)` | Cast to specified type |
| `Clone()` | Deep copy |
| `convolve(v)` | 1D convolution |
| `copy()` | Return a copy |
| `cumsum(axis)` | Cumulative sum |
| `delete(indices)` | Delete elements |
| `dot(b)` | Dot product |
| `flatten(order)` | Return flattened copy |
| `inv()` | Matrix inverse |
| `item(*args)` | Copy element to Python scalar |
| `itemset(*args)` | Set a single item |
| `lstqr(b)` | Least-squares solution |
| `matrix_power(n)` | Raise matrix to power n |
| `mean(axis)` | Mean value |
| `multi_dot(*arrays)` | Chained dot products |
| `negative()` / `negate()` | Numerical negative |
| `positive()` | Numerical positive |
| `prod(axis)` | Product of elements |
| `ravel()` | Return flattened view |
| `reshape(shape)` | Return reshaped array |
| `roll(shift)` | Roll elements |
| `std(axis)` | Standard deviation |
| `sum(axis)` | Sum of elements |
| `swapaxes(a1, a2)` | Swap two axes |
| `tofile(path)` | Write to file |
| `tolist()` | Return as nested Python list |
| `transpose()` | Transpose |
| `unique()` | Unique elements |
| `var(axis)` | Variance |
| `view(dtype)` | New view with different dtype |

### NDArray Properties

| Property | Description |
|----------|-------------|
| `T` | Transpose |
| `base` | Base object if view |
| `dtype` | Data type |
| `dtypesize` | Size of dtype in bytes |
| `flat` | 1D iterator |
| `ndim` | Number of dimensions |
| `shape` | Tuple of dimensions |
| `size` | Number of elements |
| `strides` | Strides of data |
| `typecode` | NPTypeCode of dtype |

---

## Missing Functions (239 total)

### Complete Alphabetical List

```
acos                    acosh                   angle                   append
apply_along_axis        apply_over_axes         arccosh                 arcsinh
arctanh                 argpartition            argwhere                array2string
array_equiv             array_repr              array_str               asarray_chkfinite
ascontiguousarray       asfortranarray          asin                    asinh
asmatrix                astype                  atan                    atan2
atanh                   average                 bartlett                base_repr
binary_repr             bincount                bitwise_and             bitwise_count
bitwise_invert          bitwise_left_shift      bitwise_or              bitwise_right_shift
bitwise_xor             blackman                block                   bmat
broadcast_shapes        busday_count            busday_offset           c_
choose                  column_stack            compress                concat
conj                    conjugate               copysign                corrcoef
correlate               cov                     cross                   cumulative_prod
cumulative_sum          datetime_as_string      datetime_data           delete
diag                    diag_indices            diag_indices_from       diagflat
diagonal                diff                    digitize                divmod
ediff1d                 einsum                  einsum_path             emath
errstate                extract                 fabs                    fill_diagonal
fix                     flatnonzero             flip                    fliplr
flipud                  float_power             fmod                    format_float_positional
format_float_scientific frexp                   from_dlpack             fromfunction
fromiter                frompyfunc              fromregex               fromstring
gcd                     genfromtxt              geomspace               get_include
get_printoptions        getbufsize              geterr                  geterrcall
gradient                hamming                 hanning                 heaviside
histogram               histogram2d             histogram_bin_edges     histogramdd
hypot                   i0                      imag                    index_exp
indices                 info                    inner                   insert
interp                  intersect1d             is_busday               isfortran
isin                    isnat                   isneginf                isposinf
iterable                ix_                     kaiser                  kron
lcm                     ldexp                   lexsort                 loadtxt
logaddexp               logaddexp2              logspace                mask_indices
matrix                  matrix_transpose        matvec                  may_share_memory
median                  nan_to_num              nanargmax               nanargmin
nancumprod              nancumsum               nanmedian               nanpercentile
nanquantile             ndenumerate             ndim                    ndindex
nextafter               ogrid                   packbits                pad
partition               percentile              permute_dims            piecewise
place                   poly                    poly1d                  polyadd
polyder                 polydiv                 polyfit                 polyint
polymul                 polysub                 polyval                 pow
printoptions            ptp                     put                     put_along_axis
putmask                 quantile                r_                      ravel_multi_index
real                    real_if_close           remainder               require
resize                  rint                    roots                   rot90
row_stack               s_                      savetxt                 savez
savez_compressed        select                  set_printoptions        setbufsize
setdiff1d               seterr                  seterrcall              setxor1d
shape                   shares_memory           show_config             show_runtime
signbit                 sinc                    sort                    sort_complex
spacing                 take                    take_along_axis         tensordot
tile                    trace                   trapezoid               tri
tril                    tril_indices            tril_indices_from       trim_zeros
triu                    triu_indices            triu_indices_from       typename
union1d                 unique_all              unique_counts           unique_inverse
unique_values           unpackbits              unravel_index           unstack
unwrap                  vander                  vdot                    vecdot
vecmat                  vectorize               where
```

---

## Missing Functions by Priority

### Tier 1: Critical (Most Popular) - 25 functions

| # | Function | Category | Why Popular |
|---|----------|----------|-------------|
| 1 | `where` | Selection | Conditional element selection - used constantly |
| 2 | `sort` | Sorting | In-place/out-of-place sorting |
| 3 | `flip` | Manipulation | Reverse arrays along axis |
| 4 | `diag` | Linear Algebra | Extract/create diagonals |
| 5 | `diagonal` | Linear Algebra | Return specified diagonals |
| 6 | `trace` | Linear Algebra | Sum of diagonal elements |
| 7 | `diff` | Math | Discrete difference (derivatives) |
| 8 | `histogram` | Statistics | Compute histogram |
| 9 | `percentile` | Statistics | Percentile computation |
| 10 | `median` | Statistics | Median value |
| 11 | `pad` | Manipulation | Array padding (CNNs, signal processing) |
| 12 | `tile` | Manipulation | Repeat array (data augmentation) |
| 13 | `bitwise_and` | Bitwise | Bitwise AND |
| 14 | `bitwise_or` | Bitwise | Bitwise OR |
| 15 | `bitwise_xor` | Bitwise | Bitwise XOR |
| 16 | `argwhere` | Indexing | Find indices where True |
| 17 | `flatnonzero` | Indexing | Flat indices of non-zero |
| 18 | `nan_to_num` | NaN handling | Replace NaN/inf with numbers |
| 19 | `rot90` | Manipulation | Rotate 90 degrees |
| 20 | `average` | Statistics | Weighted average |
| 21 | `take` | Indexing | Take elements along axis |
| 22 | `put` | Indexing | Put values at indices |
| 23 | `insert` | Manipulation | Insert elements |
| 24 | `append` | Manipulation | Append to array |
| 25 | `delete` | Manipulation | Delete elements (static version) |

### Tier 2: High Priority - 30 functions

| # | Function | Category | Description |
|---|----------|----------|-------------|
| 26 | `quantile` | Statistics | Quantile values |
| 27 | `cov` | Statistics | Covariance matrix |
| 28 | `corrcoef` | Statistics | Correlation coefficients |
| 29 | `inner` | Linear Algebra | Inner product |
| 30 | `vdot` | Linear Algebra | Vector dot product |
| 31 | `cross` | Linear Algebra | Cross product |
| 32 | `tensordot` | Linear Algebra | Tensor contraction |
| 33 | `kron` | Linear Algebra | Kronecker product |
| 34 | `nanargmax` | NaN-aware | Argmax ignoring NaN |
| 35 | `nanargmin` | NaN-aware | Argmin ignoring NaN |
| 36 | `nanmedian` | NaN-aware | Median ignoring NaN |
| 37 | `nancumsum` | NaN-aware | Cumsum ignoring NaN |
| 38 | `nancumprod` | NaN-aware | Cumprod ignoring NaN |
| 39 | `nanpercentile` | NaN-aware | Percentile ignoring NaN |
| 40 | `nanquantile` | NaN-aware | Quantile ignoring NaN |
| 41 | `fliplr` | Manipulation | Flip left-right |
| 42 | `flipud` | Manipulation | Flip up-down |
| 43 | `histogram2d` | Statistics | 2D histogram |
| 44 | `histogramdd` | Statistics | N-D histogram |
| 45 | `gradient` | Math | Numerical gradient |
| 46 | `arccosh` | Trig | Inverse hyperbolic cosine |
| 47 | `arcsinh` | Trig | Inverse hyperbolic sine |
| 48 | `arctanh` | Trig | Inverse hyperbolic tangent |
| 49 | `correlate` | Signal | Cross-correlation |
| 50 | `lexsort` | Sorting | Indirect lexicographic sort |
| 51 | `partition` | Sorting | Partial sort |
| 52 | `argpartition` | Sorting | Indices for partial sort |
| 53 | `logspace` | Creation | Log-spaced array |
| 54 | `geomspace` | Creation | Geometrically spaced |
| 55 | `resize` | Manipulation | Resize array |

### Tier 3: Medium Priority - 40 functions

| Function | Category |
|----------|----------|
| `take_along_axis` | Indexing |
| `put_along_axis` | Indexing |
| `putmask` | Indexing |
| `compress` | Indexing |
| `choose` | Indexing |
| `select` | Indexing |
| `extract` | Indexing |
| `place` | Indexing |
| `ptp` | Statistics |
| `hypot` | Math |
| `sinc` | Math |
| `rint` | Math |
| `fix` | Math |
| `real` | Complex |
| `imag` | Complex |
| `conj` / `conjugate` | Complex |
| `angle` | Complex |
| `triu` | Matrix |
| `tril` | Matrix |
| `tri` | Matrix |
| `diagflat` | Matrix |
| `vander` | Matrix |
| `piecewise` | Math |
| `ediff1d` | Math |
| `heaviside` | Math |
| `unwrap` | Signal |
| `indices` | Creation |
| `ogrid` | Creation |
| `fromfunction` | Creation |
| `fromiter` | Creation |
| `fromstring` | Creation |
| `block` | Creation |
| `column_stack` | Manipulation |
| `row_stack` | Manipulation |
| `unstack` | Manipulation |
| `mask_indices` | Indexing |
| `fill_diagonal` | Indexing |
| `ravel_multi_index` | Indexing |
| `unravel_index` | Indexing |
| `ix_` | Indexing |

### Tier 4: Lower Priority - 50 functions

| Category | Functions |
|----------|-----------|
| Triangle indices | `triu_indices`, `triu_indices_from`, `tril_indices`, `tril_indices_from`, `diag_indices`, `diag_indices_from` |
| Iteration | `ndenumerate`, `ndindex` |
| Functional | `apply_along_axis`, `apply_over_axes`, `vectorize`, `frompyfunc` |
| Interpolation | `interp` |
| Discretization | `digitize`, `bincount` |
| Set operations | `intersect1d`, `union1d`, `setdiff1d`, `setxor1d`, `isin`, `unique_all`, `unique_counts`, `unique_inverse`, `unique_values` |
| Bit manipulation | `packbits`, `unpackbits`, `bitwise_count` |
| Memory | `shares_memory`, `may_share_memory`, `ascontiguousarray`, `asfortranarray`, `require` |
| Special functions | `i0`, `frexp`, `ldexp`, `fmod`, `copysign`, `signbit`, `nextafter`, `spacing`, `logaddexp`, `logaddexp2`, `gcd`, `lcm`, `fabs`, `float_power`, `divmod`, `remainder` |
| Window functions | `hamming`, `hanning`, `bartlett`, `blackman`, `kaiser` |
| Integration | `trapezoid` |

### Tier 5: Specialized/Rarely Used - 94 functions

| Category | Functions |
|----------|-----------|
| Polynomial | `poly`, `roots`, `polyfit`, `polyval`, `polyadd`, `polysub`, `polymul`, `polydiv`, `polyint`, `polyder`, `poly1d` |
| I/O | `loadtxt`, `savetxt`, `genfromtxt`, `fromregex`, `savez`, `savez_compressed` |
| Error handling | `seterr`, `geterr`, `errstate`, `seterrcall`, `geterrcall` |
| Print options | `set_printoptions`, `get_printoptions`, `printoptions`, `array2string`, `format_float_scientific`, `format_float_positional`, `array_repr`, `array_str` |
| NumPy 2.x additions | `cumulative_sum`, `cumulative_prod`, `matrix_transpose`, `vecdot`, `vecmat`, `matvec`, `permute_dims`, `concat` |
| Date/time | `busday_count`, `busday_offset`, `is_busday`, `datetime_as_string`, `datetime_data`, `isnat` |
| Index expressions | `r_`, `c_`, `s_`, `index_exp` |
| Matrix class (legacy) | `matrix`, `asmatrix`, `bmat` |
| Misc | `typename`, `iterable`, `info`, `get_include`, `show_config`, `show_runtime`, `from_dlpack`, `broadcast_shapes`, `setbufsize`, `getbufsize`, `array_equiv`, `asarray_chkfinite`, `isfortran`, `isneginf`, `isposinf`, `acos`, `acosh`, `asin`, `asinh`, `atan`, `atan2`, `atanh`, `astype`, `base_repr`, `binary_repr`, `emath`, `ndim`, `shape`, `sort_complex`, `trim_zeros`, `real_if_close`, `pow` |

---

## Implementation Recommendations

### Quick Wins (Low effort, high impact)

| Function | Implementation Approach |
|----------|------------------------|
| `flip` | Index manipulation with `[::-1]` slicing |
| `fliplr` | `flip(a, axis=1)` |
| `flipud` | `flip(a, axis=0)` |
| `rot90` | Transpose + flip combination |
| `diag` | Index extraction/creation |
| `diagonal` | Stride-based view |
| `trace` | `sum(diagonal(a))` |
| `bitwise_and/or/xor` | ILKernel already has infrastructure |
| `arccosh/arcsinh/arctanh` | Follow existing trig pattern |
| `nan_to_num` | Simple masking operation |
| `ptp` | `max(a) - min(a)` |
| `flatnonzero` | `nonzero(ravel(a))[0]` |
| `triu/tril` | Index masking |
| `correlate` | `convolve(a, v[::-1])` |
| `logspace` | `power(base, linspace(...))` |
| `geomspace` | `exp(linspace(log(start), log(stop)))` |

### Medium Effort

| Function | Implementation Approach |
|----------|------------------------|
| `where` | Conditional indexing with broadcasting |
| `sort` | Sorting algorithm with dtype support |
| `histogram` | Bin counting with edge handling |
| `median` | Sort + middle element(s) |
| `percentile/quantile` | Sort + interpolation |
| `pad` | Memory allocation + copy patterns |
| `tile` | Repeat + reshape |
| `diff` | Subtraction with slicing |
| `average` | Weighted sum / weight sum |
| `argwhere` | Coordinate extraction from mask |
| `take/put` | Fancy indexing generalization |

### Higher Effort

| Function | Implementation Approach |
|----------|------------------------|
| `einsum` | Einstein summation parser + optimizer |
| `tensordot` | Generalized tensor contraction |
| `kron` | Kronecker product expansion |
| `cov/corrcoef` | Covariance calculation |
| `gradient` | Finite difference schemes |
| `histogram2d/histogramdd` | N-D binning |

---

## Reference

- NumPy 2.4.2 source: `src/numpy/`
- NumPy API reference: https://numpy.org/doc/stable/reference/
- NumSharp source: `src/NumSharp.Core/`

Last updated: 2026-04-12
