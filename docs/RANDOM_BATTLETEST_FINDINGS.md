# NumPy Random Battletest Findings

Generated from comprehensive testing of `np.random` methods against NumPy 2.x behavior.

## Key Findings Summary

### 1. Seed Behavior
| Input | NumPy Behavior |
|-------|---------------|
| `seed(0)` to `seed(2**32-1)` | Valid range |
| `seed(-1)` | `ValueError: Seed must be between 0 and 2**32 - 1` |
| `seed(2**32)` | `ValueError: Seed must be between 0 and 2**32 - 1` |
| `seed(42.0)` | `TypeError: Cannot cast scalar from dtype('float64') to dtype('int64')` |
| `seed(None)` | **Valid!** Uses system entropy - returns None |
| `seed([])` | `ValueError: Seed must be non-empty` |
| `seed([[1,2],[3,4]])` | `ValueError: Seed array must be 1-d` |
| `seed([1,2,3,4])` | Valid array seeding |

### 2. Size Parameter Behavior
| Input | Result |
|-------|--------|
| `size=None` | Returns Python scalar (float/int) |
| `size=()` | Returns 0-d ndarray (shape=(), ndim=0) |
| `size=5` | Returns 1-d ndarray (shape=(5,)) |
| `size=(2,3)` | Returns 2-d ndarray (shape=(2,3)) |
| `size=0` | Returns empty 1-d ndarray (shape=(0,)) |
| `size=(5,0)` | Returns empty 2-d ndarray (shape=(5,0)) |
| `size=-1` | `ValueError: negative dimensions are not allowed` |

### 3. randint Specifics
| Test | NumPy Behavior |
|------|---------------|
| `randint(10)` | Returns Python int (not ndarray) |
| `randint(10, size=())` | Returns 0-d ndarray with dtype |
| `randint(0)` | `ValueError: high <= 0` |
| `randint(10, 5)` | `ValueError: low >= high` |
| `randint(5, 5)` | `ValueError: low >= high` |
| `randint(256, dtype=np.int8)` | `ValueError: high is out of bounds for int8` |
| `randint(-1, 10, dtype=np.uint8)` | `ValueError: low is out of bounds for uint8` |
| Default dtype | `int32` on most systems (not int64!) |

### 4. Surprising "No Error" Cases
These inputs do NOT throw errors in NumPy (they produce nan/inf or degenerate outputs):

| Function | Input | NumPy Output |
|----------|-------|--------------|
| `normal` | `normal(nan, 1)` | Array of nan |
| `normal` | `normal(0, nan)` | Array of nan |
| `normal` | `normal(0, inf)` | Array of inf |
| `gamma` | `gamma(0, 1)` | Array of 0.0 |
| `gamma` | `gamma(1, 0)` | Array of 0.0 |
| `gamma` | `gamma(nan, 1)` | Array of nan |
| `gamma` | `gamma(inf, 1)` | Array of inf |
| `standard_gamma` | `standard_gamma(0)` | Array of 0.0 |
| `standard_gamma` | `standard_gamma(nan)` | Array of nan |
| `exponential` | `exponential(0)` | Array of 0.0 |
| `exponential` | `exponential(nan)` | Array of nan |
| `exponential` | `exponential(inf)` | Array of inf |
| `beta` | `beta(nan, 1)` | Array of nan |
| `beta` | `beta(inf, 1)` | Array of nan (inf/inf) |
| `negative_binomial` | `negative_binomial(1, 0)` | Array of inf (large ints) |
| `negative_binomial` | `negative_binomial(1, 1)` | Array of 0 |
| `chisquare` | `chisquare(nan)` | Array of nan |
| `standard_t` | `standard_t(nan)` | Array of nan |
| `laplace` | `laplace(0, 0)` | Array of 0.0 |
| `laplace` | `laplace(nan, 1)` | Array of nan |
| `logistic` | `logistic(0, 0)` | Array of 0.0 |
| `gumbel` | `gumbel(0, 0)` | Array of 0.0 |
| `lognormal` | `lognormal(0, 0)` | Array of 1.0 |
| `logseries` | `logseries(0)` | Array of 1 |
| `rayleigh` | `rayleigh(0)` | Array of 0.0 |

### 5. Error Cases That DO Throw
| Function | Input | Error |
|----------|-------|-------|
| `beta(0, 1)` | a <= 0 | `ValueError: a <= 0` |
| `beta(-1, 1)` | negative a | `ValueError: a <= 0` |
| `gamma(-1, 1)` | negative shape | `ValueError: shape < 0` |
| `gamma(1, -1)` | negative scale | `ValueError: scale < 0` |
| `exponential(-1)` | negative scale | `ValueError: scale < 0` |
| `poisson(-1)` | negative lam | `ValueError: lam < 0` |
| `poisson(inf)` | inf lam | `ValueError: lam value too large` |
| `poisson(1e10)` | very large lam | `ValueError: lam value too large` |
| `binomial(-1, 0.5)` | negative n | `ValueError: n < 0` |
| `binomial(10, -0.1)` | p < 0 | `ValueError: p < 0` |
| `binomial(10, 1.1)` | p > 1 | `ValueError: p > 1` |
| `geometric(0)` | p = 0 | `ValueError: p <= 0` |
| `geometric(1.1)` | p > 1 | `ValueError: p > 1` |
| `chisquare(0)` | df = 0 | `ValueError: df <= 0` |
| `chisquare(-1)` | negative df | `ValueError: df <= 0` |
| `uniform(inf, inf)` | both inf | `OverflowError: Range exceeds valid bounds` |
| `uniform(-inf, inf)` | infinite range | `OverflowError: Range exceeds valid bounds` |
| `hypergeometric(10, 5, 0)` | nsample = 0 | `ValueError: nsample < 1 or nsample is NaN` |
| `triangular(0, 0, 0)` | degenerate | `ValueError: left == right` |
| `triangular(1, 0, 2)` | mode < left | `ValueError: left > mode` |
| `logseries(1)` | p = 1 | `ValueError: p >= 1` |
| `zipf(1)` | a <= 1 | `ValueError: a <= 1` |
| `pareto(0)` | a = 0 | `ValueError: a <= 0` |
| `power(0)` | a = 0 | `ValueError: a <= 0` |
| `rayleigh(-1)` | scale < 0 | `ValueError: scale < 0` |
| `vonmises(0, -1)` | kappa < 0 | `ValueError: kappa < 0` |

### 6. Default dtypes
| Function | Default dtype |
|----------|--------------|
| `rand()` | float64 |
| `randn()` | float64 |
| `uniform()` | float64 |
| `normal()` | float64 |
| `randint()` | **int32** (not int64!) |
| `binomial()` | int32 |
| `poisson()` | int64 |
| `choice(int)` | int32 |
| `geometric()` | int64 |
| `hypergeometric()` | int64 |
| `negative_binomial()` | int64 |

### 7. Seeded Reference Values (seed=42)

```python
# randint(100, size=5)
[51, 92, 14, 71, 60]

# rand(5) - first 5 uniform values
[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]

# randn(10) - first 10 normal values
[ 0.49671415, -0.1382643,   0.64768854,  1.52302986, -0.23415337,
 -0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004]

# uniform(0, 100, size=5)
[37.4540119, 95.0714306, 73.1993942, 59.8658484, 15.6018640]

# normal(0, 1, size=5) - same as randn
[0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337]

# choice(10, size=5)
[6, 3, 7, 4, 6]

# permutation(10)
[8, 1, 5, 0, 7, 2, 9, 4, 3, 6]

# shuffle(arange(10)) - same result as permutation!
[8, 1, 5, 0, 7, 2, 9, 4, 3, 6]

# beta(2, 5, size=5)
[0.18626021, 0.34556073, 0.39676747, 0.53881673, 0.41919451]

# gamma(2, 1, size=5)
[2.77527951, 0.93700099, 1.40881563, 1.23399074, 1.98883678]

# poisson(5, size=5)
[8, 7, 2, 3, 8]

# binomial(10, 0.5, size=5)
[4, 4, 5, 3, 5]

# exponential(1, size=5)
[0.98229985, 0.05052044, 0.31223139, 0.51526898, 1.85810637]

# dirichlet([1,1,1], size=3)
[[0.09784297, 0.62761396, 0.27454307],
 [0.72909200, 0.13546541, 0.13544259],
 [0.02001195, 0.67261832, 0.30736973]]

# multinomial(10, [0.2,0.3,0.5], size=3)
[[1, 6, 3],
 [3, 3, 4],
 [1, 2, 7]]

# multivariate_normal([0,0], [[1,0],[0,1]], size=3)
[[ 0.49671415, -0.1382643 ],
 [ 0.64768854,  1.52302986],
 [-0.23415337, -0.23413696]]

# weibull(2, size=5)
[0.68503145, 1.73497015, 1.14749540, 0.95548027, 0.41185540]

# wald(1, 1, size=5)
[1.63516639, 1.14815282, 0.79166122, 1.26314598, 0.23479012]

# zipf(2, size=5)
[1, 3, 1, 1, 2]

# vonmises(0, 1, size=5)
[0.62690657, -1.17478453, 0.08884717, 1.55489819, -2.12889830]

# triangular(0, 0.5, 1, size=5)
[0.43274711, 0.84301960, 0.63393576, 0.55203710, 0.27930149]

# chisquare(5, size=5)
[4.41509069, 3.15095986, 2.58780440, 4.21266247, 5.57149053]

# f(5, 10, size=5)
[0.77077920, 0.48855703, 2.03697116, 0.69105959, 0.37853674]

# standard_t(5, size=5)
[0.41849820, -1.02185215, 0.74854279, 1.65033893, -0.20238273]

# pareto(2, size=5)
[0.26444595, 3.50442711, 0.93164669, 0.57849408, 0.08851288]

# power(2, size=5)
[0.61199683, 0.97504580, 0.85556645, 0.77373024, 0.39499195]

# rayleigh(1, size=5)
[0.96878077, 2.45361832, 1.62280356, 1.35125316, 0.58245149]

# laplace(0, 1, size=5)
[-0.25946279, 2.96452754, 1.30743155, 0.92028717, -0.36478629]

# logistic(0, 1, size=5)
[-0.51348793, 2.95060543, 0.99082996, 0.39640659, -0.82862263]

# gumbel(0, 1, size=5)
[-0.18473606, 2.96085813, 1.23393979, 0.87423905, -0.41556001]

# lognormal(0, 1, size=5)
[1.64345205, 0.87073553, 1.91122818, 4.58587488, 0.79119989]
```

### 8. State Structure
```python
state = np.random.get_state()
# Returns tuple:
# ('MT19937',
#  array([624 uint32 values], dtype=uint32),  # key array
#  624,                                        # position (0-624)
#  0,                                          # has_gauss (0 or 1)
#  0.0)                                        # cached_gaussian
```

### 9. NumSharp Implementation Gaps

Based on this battletest, NumSharp should:

1. **seed(None)** - Should use system entropy, not throw
2. **seed([])** - Should throw "Seed must be non-empty"
3. **Edge case handling** - Many distributions accept nan/inf and return nan/inf without errors
4. **randint default dtype** - Should be int32, not int64
5. **size=() behavior** - Should return 0-d ndarray, not scalar
6. **Poisson large lambda** - Should throw for lam >= ~1e10
7. **hypergeometric nsample=0** - Should throw "nsample < 1"
8. **triangular degenerate** - Should throw "left == right" when left==mode==right

## Full Battletest Script

See `battletest_random.py` for the complete test script.

## Output File

See `battletest_random_output.txt` for full NumPy output (2227 lines).
