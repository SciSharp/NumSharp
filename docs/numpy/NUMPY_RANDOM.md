# NumPy Random Module - Complete Reference

> **NumPy Version:** 2.4.2
> **Purpose:** Authoritative reference for implementing NumSharp's `np.random` module with 100% NumPy compatibility.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Reference](#quick-reference)
3. [Architecture](#architecture)
4. [Legacy API (np.random.*)](#legacy-api)
5. [Modern Generator API](#modern-generator-api)
6. [BitGenerators](#bitgenerators)
7. [SeedSequence](#seedsequence)
8. [Distribution Reference](#distribution-reference)
9. [Seeded Reference Values](#seeded-reference-values)
10. [Validation Rules](#validation-rules)
11. [Edge Cases](#edge-cases)
12. [Implicit Behaviors](#implicit-behaviors)
13. [NumSharp Implementation Status](#numsharp-implementation-status)
14. [MT19937 Implementation Details](#mt19937-implementation-details)

---

## Overview

NumPy's random module provides two APIs:

| API | Introduced | Default RNG | Recommended |
|-----|------------|-------------|-------------|
| **Legacy** (`np.random.*`) | NumPy 1.0 | MT19937 | No (deprecated patterns) |
| **Modern** (`np.random.Generator`) | NumPy 1.17 | PCG64 | Yes (SPEC 7 compliant) |

### Key Differences

| Feature | Legacy | Modern |
|---------|--------|--------|
| Global state | Yes (`np.random.seed()`) | No (explicit Generator) |
| Default algorithm | MT19937 | PCG64 |
| Thread safety | No | Yes (per-Generator) |
| Reproducibility | Global seed | Explicit seeding |
| Parallel streams | Manual | `spawn()` / `SeedSequence` |
| `axis` parameter | No | Yes (shuffle, permutation, choice) |

---

## Quick Reference

### All Functions by Category

#### Uniform Distributions
| Function | Legacy | Generator | NumSharp | Description |
|----------|--------|-----------|----------|-------------|
| `random()` | `random(size)` | `random(size, dtype, out)` | ✅ | Uniform [0, 1) |
| `rand()` | `rand(*shape)` | — | ✅ | Uniform [0, 1), shape as args |
| `random_sample()` | `random_sample(size)` | — | ✅ | Alias for `random()` |
| `ranf()` | `ranf(size)` | — | ❌ | Alias for `random()` |
| `sample()` | `sample(size)` | — | ❌ | Alias for `random()` |
| `uniform()` | `uniform(low, high, size)` | `uniform(low, high, size)` | ✅ | Uniform [low, high) |

#### Normal Distributions
| Function | Legacy | Generator | NumSharp | Description |
|----------|--------|-----------|----------|-------------|
| `randn()` | `randn(*shape)` | — | ✅ | Standard normal, shape as args |
| `standard_normal()` | `standard_normal(size)` | `standard_normal(size, dtype, out)` | ✅ | Standard normal N(0,1) |
| `normal()` | `normal(loc, scale, size)` | `normal(loc, scale, size)` | ✅ | Normal N(loc, scale²) |
| `lognormal()` | `lognormal(mean, sigma, size)` | `lognormal(mean, sigma, size)` | ✅ | Log-normal |

#### Integer Distributions
| Function | Legacy | Generator | NumSharp | Description |
|----------|--------|-----------|----------|-------------|
| `randint()` | `randint(low, high, size, dtype)` | — | ✅ | Random integers [low, high) |
| `integers()` | — | `integers(low, high, size, dtype, endpoint)` | ❌ | Random integers with endpoint option |
| `random_integers()` | `random_integers(low, high, size)` | — | ❌ | **DEPRECATED** - integers [low, high] |

#### Sequences
| Function | Legacy | Generator | NumSharp | Description |
|----------|--------|-----------|----------|-------------|
| `choice()` | `choice(a, size, replace, p)` | `choice(a, size, replace, p, axis, shuffle)` | ✅ | Random selection |
| `permutation()` | `permutation(x)` | `permutation(x, axis)` | ✅ | Random permutation |
| `permuted()` | — | `permuted(x, axis, out)` | ❌ | Independent axis permutation |
| `shuffle()` | `shuffle(x)` | `shuffle(x, axis)` | ✅ | In-place shuffle |

#### Continuous Distributions
| Function | Parameters | NumSharp | Description |
|----------|------------|----------|-------------|
| `beta()` | `a, b, size` | ✅ | Beta distribution |
| `chisquare()` | `df, size` | ✅ | Chi-square distribution |
| `exponential()` | `scale, size` | ✅ | Exponential distribution |
| `f()` | `dfnum, dfden, size` | ✅ | F distribution |
| `gamma()` | `shape, scale, size` | ✅ | Gamma distribution |
| `gumbel()` | `loc, scale, size` | ✅ | Gumbel (extreme value type I) |
| `laplace()` | `loc, scale, size` | ✅ | Laplace (double exponential) |
| `logistic()` | `loc, scale, size` | ✅ | Logistic distribution |
| `noncentral_chisquare()` | `df, nonc, size` | ✅ | Non-central chi-square |
| `noncentral_f()` | `dfnum, dfden, nonc, size` | ✅ | Non-central F |
| `pareto()` | `a, size` | ✅ | Pareto (Lomax) |
| `power()` | `a, size` | ✅ | Power distribution |
| `rayleigh()` | `scale, size` | ✅ | Rayleigh distribution |
| `standard_cauchy()` | `size` | ✅ | Standard Cauchy |
| `standard_exponential()` | `size` | ✅ | Standard exponential |
| `standard_gamma()` | `shape, size` | ✅ | Standard gamma |
| `standard_t()` | `df, size` | ✅ | Student's t |
| `triangular()` | `left, mode, right, size` | ✅ | Triangular distribution |
| `vonmises()` | `mu, kappa, size` | ✅ | Von Mises (circular) |
| `wald()` | `mean, scale, size` | ✅ | Wald (inverse Gaussian) |
| `weibull()` | `a, size` | ✅ | Weibull distribution |

#### Discrete Distributions
| Function | Parameters | NumSharp | Description |
|----------|------------|----------|-------------|
| `bernoulli()` | `p, size` | ✅* | Bernoulli (NumSharp extra) |
| `binomial()` | `n, p, size` | ✅ | Binomial distribution |
| `geometric()` | `p, size` | ✅ | Geometric distribution |
| `hypergeometric()` | `ngood, nbad, nsample, size` | ✅ | Hypergeometric |
| `logseries()` | `p, size` | ✅ | Logarithmic series |
| `negative_binomial()` | `n, p, size` | ✅ | Negative binomial |
| `poisson()` | `lam, size` | ✅ | Poisson distribution |
| `zipf()` | `a, size` | ✅ | Zipf distribution |

#### Multivariate Distributions
| Function | Parameters | NumSharp | Description |
|----------|------------|----------|-------------|
| `dirichlet()` | `alpha, size` | ✅ | Dirichlet distribution |
| `multinomial()` | `n, pvals, size` | ✅ | Multinomial distribution |
| `multivariate_normal()` | `mean, cov, size, check_valid, tol` | ✅ | Multivariate normal |
| `multivariate_hypergeometric()` | `colors, nsample, size, method` | ❌ | Multivariate hypergeometric |

#### State Management
| Function | Legacy | Generator | NumSharp | Description |
|----------|--------|-----------|----------|-------------|
| `seed()` | `seed(seed)` | — | ✅ | Set global seed |
| `get_state()` | `get_state(legacy)` | — | ✅ | Get RNG state |
| `set_state()` | `set_state(state)` | — | ✅ | Set RNG state |
| `get_bit_generator()` | `get_bit_generator()` | — | ❌ | Get underlying BitGenerator |
| `set_bit_generator()` | `set_bit_generator(bg)` | — | ❌ | Set underlying BitGenerator |

#### Utilities
| Function | Legacy | Generator | NumSharp | Description |
|----------|--------|-----------|----------|-------------|
| `bytes()` | `bytes(length)` | `bytes(length)` | ❌ | Random bytes |
| `spawn()` | — | `spawn(n)` | ❌ | Create child generators |

---

## Architecture

```
np.random (module)
├── Module-level functions (legacy API)
│   ├── seed(), get_state(), set_state()
│   ├── rand(), randn(), randint(), random()
│   └── All distribution functions
│
├── default_rng(seed) → Generator
│
├── RandomState(seed) → Legacy container
│   └── Same methods as module-level
│
├── Generator(bit_generator) → Modern container
│   ├── All distribution methods
│   ├── integers() (replaces randint)
│   ├── permuted() (axis-aware)
│   └── spawn() (parallel streams)
│
├── BitGenerators (abstract base)
│   ├── MT19937 (Mersenne Twister)
│   ├── PCG64 (default, recommended)
│   ├── PCG64DXSM (PCG variant)
│   ├── Philox (counter-based)
│   └── SFC64 (Small Fast Chaotic)
│
└── SeedSequence
    ├── Proper seed mixing
    ├── spawn() for parallel streams
    └── generate_state()
```

---

## Legacy API

### Module-Level Functions

The legacy API uses global state. All functions operate on a shared `RandomState` instance.

```python
import numpy as np

# Set global seed
np.random.seed(42)

# Generate random values
x = np.random.rand(5)           # Uniform [0, 1)
y = np.random.randn(5)          # Standard normal
z = np.random.randint(0, 100, 5)  # Integers [0, 100)
```

### RandomState Class

Encapsulates MT19937 state for reproducible sequences:

```python
rs = np.random.RandomState(42)
x = rs.rand(5)
y = rs.randn(5)
```

### RandomState Methods (Complete List)

```
beta(a, b, size=None)
binomial(n, p, size=None)
bytes(length)
chisquare(df, size=None)
choice(a, size=None, replace=True, p=None)
dirichlet(alpha, size=None)
exponential(scale=1.0, size=None)
f(dfnum, dfden, size=None)
gamma(shape, scale=1.0, size=None)
geometric(p, size=None)
get_state(legacy=True)
gumbel(loc=0.0, scale=1.0, size=None)
hypergeometric(ngood, nbad, nsample, size=None)
laplace(loc=0.0, scale=1.0, size=None)
logistic(loc=0.0, scale=1.0, size=None)
lognormal(mean=0.0, sigma=1.0, size=None)
logseries(p, size=None)
multinomial(n, pvals, size=None)
multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-08)
negative_binomial(n, p, size=None)
noncentral_chisquare(df, nonc, size=None)
noncentral_f(dfnum, dfden, nonc, size=None)
normal(loc=0.0, scale=1.0, size=None)
pareto(a, size=None)
permutation(x)
poisson(lam=1.0, size=None)
power(a, size=None)
rand(*args)
randint(low, high=None, size=None, dtype=int)
randn(*args)
random(size=None)
random_integers(low, high=None, size=None)  # DEPRECATED
random_sample(size=None)
rayleigh(scale=1.0, size=None)
seed(seed=None)
set_state(state)
shuffle(x)
standard_cauchy(size=None)
standard_exponential(size=None)
standard_gamma(shape, size=None)
standard_normal(size=None)
standard_t(df, size=None)
tomaxint(size=None)
triangular(left, mode, right, size=None)
uniform(low=0.0, high=1.0, size=None)
vonmises(mu, kappa, size=None)
wald(mean, scale, size=None)
weibull(a, size=None)
zipf(a, size=None)
```

### State Format

```python
state = np.random.get_state()
# Returns tuple:
# ('MT19937',
#  array([624 uint32 values], dtype=uint32),  # key array
#  624,                                        # position (0-624)
#  0,                                          # has_gauss (0 or 1)
#  0.0)                                        # cached_gaussian
```

---

## Modern Generator API

### Creating Generators

```python
# Recommended: use default_rng()
rng = np.random.default_rng(42)

# Or construct with specific BitGenerator
rng = np.random.Generator(np.random.PCG64(42))
rng = np.random.Generator(np.random.MT19937(42))
```

### Generator Methods (Complete List)

```
beta(a, b, size=None)
binomial(n, p, size=None)
bytes(length)
chisquare(df, size=None)
choice(a, size=None, replace=True, p=None, axis=0, shuffle=True)
dirichlet(alpha, size=None)
exponential(scale=1.0, size=None)
f(dfnum, dfden, size=None)
gamma(shape, scale=1.0, size=None)
geometric(p, size=None)
gumbel(loc=0.0, scale=1.0, size=None)
hypergeometric(ngood, nbad, nsample, size=None)
integers(low, high=None, size=None, dtype=np.int64, endpoint=False)
laplace(loc=0.0, scale=1.0, size=None)
logistic(loc=0.0, scale=1.0, size=None)
lognormal(mean=0.0, sigma=1.0, size=None)
logseries(p, size=None)
multinomial(n, pvals, size=None)
multivariate_hypergeometric(colors, nsample, size=None, method='marginals')
multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-08, *, method='svd')
negative_binomial(n, p, size=None)
noncentral_chisquare(df, nonc, size=None)
noncentral_f(dfnum, dfden, nonc, size=None)
normal(loc=0.0, scale=1.0, size=None)
pareto(a, size=None)
permutation(x, axis=0)
permuted(x, *, axis=None, out=None)
poisson(lam=1.0, size=None)
power(a, size=None)
random(size=None, dtype=np.float64, out=None)
rayleigh(scale=1.0, size=None)
shuffle(x, axis=0)
spawn(n_children)
standard_cauchy(size=None)
standard_exponential(size=None, dtype=np.float64, method='zig', out=None)
standard_gamma(shape, size=None, dtype=np.float64, out=None)
standard_normal(size=None, dtype=np.float64, out=None)
standard_t(df, size=None)
triangular(left, mode, right, size=None)
uniform(low=0.0, high=1.0, size=None)
vonmises(mu, kappa, size=None)
wald(mean, scale, size=None)
weibull(a, size=None)
zipf(a, size=None)
```

### Generator Properties

| Property | Description |
|----------|-------------|
| `bit_generator` | The underlying BitGenerator instance |

### Generator-Specific Features

#### `integers()` vs `randint()`

```python
rng = np.random.default_rng(42)

# endpoint=False (default): [low, high)
rng.integers(10, size=5)  # [0, 7, 6, 4, 4]

# endpoint=True: [low, high]
rng.integers(10, size=5, endpoint=True)  # [0, 8, 7, 4, 4]
```

#### `permuted()` - Independent Axis Shuffle

Unlike `shuffle()`, `permuted()` shuffles each slice independently:

```python
rng = np.random.default_rng(42)
arr = np.arange(12).reshape(3, 4)
# [[ 0,  1,  2,  3],
#  [ 4,  5,  6,  7],
#  [ 8,  9, 10, 11]]

# Shuffle along axis=1 (each row shuffled independently)
rng.permuted(arr, axis=1)
# [[ 3,  2,  1,  0],
#  [ 7,  6,  4,  5],
#  [ 8, 11, 10,  9]]
```

#### `shuffle()` with Axis

```python
rng = np.random.default_rng(42)
arr = np.arange(12).reshape(3, 4)

# Shuffle along axis=1 (columns reordered, same for all rows)
rng.shuffle(arr, axis=1)
# [[ 3,  2,  1,  0],
#  [ 7,  6,  5,  4],
#  [11, 10,  9,  8]]
```

#### `choice()` with Axis

```python
rng = np.random.default_rng(42)
arr = np.arange(12).reshape(3, 4)

# Select 2 items along axis=0 (rows)
rng.choice(arr, 2, axis=0)
# [[ 0,  1,  2,  3],
#  [ 8,  9, 10, 11]]

# Select 2 items along axis=1 (columns)
rng.choice(arr, 2, axis=1)
# [[ 0,  3],
#  [ 4,  7],
#  [ 8, 11]]
```

#### `spawn()` for Parallel Streams

```python
rng = np.random.default_rng(42)
children = rng.spawn(3)

# Each child is independent
for i, child in enumerate(children):
    print(f"child[{i}]: {child.random()}")
# child[0]: 0.9167441575549085
# child[1]: 0.4674907799518424
# child[2]: 0.07123920291270869
```

---

## BitGenerators

### Overview

| BitGenerator | Period | Speed | Use Case |
|--------------|--------|-------|----------|
| `PCG64` | 2^128 | Fast | **Default** - general use |
| `PCG64DXSM` | 2^128 | Fast | Better statistical properties |
| `MT19937` | 2^19937-1 | Medium | Legacy compatibility |
| `Philox` | 2^256 | Medium | Parallelization, reproducibility |
| `SFC64` | ~2^256 | Fastest | Speed-critical applications |

### PCG64 (Default)

Permuted Congruential Generator - NumPy's recommended default.

```python
bg = np.random.PCG64(42)
rng = np.random.Generator(bg)

# State
bg.state
# {'bit_generator': 'PCG64',
#  'state': {'state': 229482374823..., 'inc': 283847592...},
#  'has_uint32': 0,
#  'uinteger': 0}

# Advance state by n steps
bg.advance(1000)

# Jump ahead by 2^64 steps
bg.jumped()
```

### MT19937 (Mersenne Twister)

Classic algorithm, used by legacy API.

```python
bg = np.random.MT19937(42)
rng = np.random.Generator(bg)

# State
bg.state
# {'bit_generator': 'MT19937',
#  'state': {'key': array([...624 uint32...]), 'pos': 624}}

# Jump ahead by 2^128 steps
bg.jumped()
```

### Philox (Counter-Based)

Deterministic, parallelizable, cryptographically-derived.

```python
bg = np.random.Philox(42)
rng = np.random.Generator(bg)

# Key feature: advance to any position
bg.advance(1000000)  # Skip 1M values instantly
```

### SFC64 (Small Fast Chaotic)

Fastest option for non-cryptographic use.

```python
bg = np.random.SFC64(42)
rng = np.random.Generator(bg)
```

### BitGenerator Common Methods

| Method | Description |
|--------|-------------|
| `random_raw(size, output)` | Raw uint64 values |
| `spawn(n)` | Create n child BitGenerators |
| `jumped(jumps)` | Return jumped copy |
| `state` | Get/set state dict |
| `seed_seq` | Associated SeedSequence |
| `lock` | Threading lock (property) |
| `capsule` | PyCapsule for C API (property) |
| `cffi` | CFFI interface (property) |
| `ctypes` | ctypes interface (property) |

### Seeded Output Comparison (seed=42)

| BitGenerator | First 5 `random()` values |
|--------------|---------------------------|
| MT19937 | 0.5420, 0.6197, 0.0574, 0.8119, 0.8601 |
| PCG64 | 0.7740, 0.4389, 0.8586, 0.6974, 0.0942 |
| PCG64DXSM | 0.6684, 0.0068, 0.6580, 0.3713, 0.2067 |
| Philox | 0.0861, 0.1416, 0.2701, 0.8740, 0.1702 |
| SFC64 | 0.5299, 0.3782, 0.9454, 0.4211, 0.6412 |

---

## SeedSequence

Proper seed mixing for parallel streams. Avoids correlation issues with naive seeding.

### Basic Usage

```python
ss = np.random.SeedSequence(42)

# Properties
ss.entropy       # 42
ss.pool_size     # 4
ss.pool          # Mixed seed array
ss.spawn_key     # ()
ss.state         # Internal state dict
ss.n_children_spawned  # Number of children spawned
```

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `spawn` | `spawn(n_children)` | Create n child SeedSequences |
| `generate_state` | `generate_state(n_words, dtype=np.uint32)` | Generate seed material |

### Spawning Independent Streams

```python
# BAD: Sequential seeds can be correlated
rng1 = np.random.default_rng(0)
rng2 = np.random.default_rng(1)  # Potentially correlated!

# GOOD: Use SeedSequence.spawn()
ss = np.random.SeedSequence(42)
children = ss.spawn(10)
rngs = [np.random.default_rng(child) for child in children]
# All statistically independent
```

### Generate State Material

```python
ss = np.random.SeedSequence(42)
state = ss.generate_state(4)  # [3444837047, 2669555309, 2046530742, 3581440988]
```

### Hierarchical Spawning

```python
ss = np.random.SeedSequence(42)
level1 = ss.spawn(3)

# Each child can spawn further
level2 = level1[0].spawn(5)
```

---

## Distribution Reference

### Uniform Distributions

#### `random(size=None)` / `rand(*shape)`

Uniform distribution over [0, 1).

```python
np.random.seed(42)
np.random.random(5)      # [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]
np.random.rand(2, 3)     # Shape as separate args
```

**Returns:** float64

#### `uniform(low=0.0, high=1.0, size=None)`

Uniform distribution over [low, high).

```python
np.random.seed(42)
np.random.uniform(0, 10, 5)  # [3.74540119, 9.50714306, 7.31993942, 5.98658484, 1.5601864]
```

**Validation:**
- `low` and `high` must be finite
- `OverflowError` if range exceeds float64 bounds

#### `randint(low, high=None, size=None, dtype=int)`

Random integers from [low, high).

```python
np.random.seed(42)
np.random.randint(100, size=5)     # [51, 92, 14, 71, 60]
np.random.randint(0, 100, size=5)  # Same as above
```

**Default dtype:** `int32` (NOT int64!)

**Validation:**
- `ValueError: high <= 0` if only low provided and low <= 0
- `ValueError: low >= high` if low >= high

#### `integers(low, high=None, size=None, dtype=np.int64, endpoint=False)` [Generator only]

Modern replacement for `randint()`.

```python
rng = np.random.default_rng(42)
rng.integers(10, size=5)                    # [0, 7, 6, 4, 4] - exclusive
rng.integers(10, size=5, endpoint=True)     # [0, 8, 7, 4, 4] - inclusive
```

---

### Normal Distributions

#### `randn(*shape)` / `standard_normal(size=None)`

Standard normal distribution N(0, 1).

```python
np.random.seed(42)
np.random.randn(5)  # [0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337]
```

**Returns:** float64

#### `normal(loc=0.0, scale=1.0, size=None)`

Normal distribution N(loc, scale²).

```python
np.random.seed(42)
np.random.normal(0, 1, 5)    # Same as randn(5)
np.random.normal(100, 15, 5) # Mean=100, std=15
```

**Edge cases:**
- `normal(nan, 1)` → array of nan
- `normal(0, inf)` → array of inf
- `normal(0, 0)` → array of loc (constant)

#### `lognormal(mean=0.0, sigma=1.0, size=None)`

Log-normal distribution.

```python
np.random.seed(42)
np.random.lognormal(0, 1, 5)  # [1.64331272, 0.87086849, 1.91111824, 4.58609939, 0.79124045]
```

**Note:** `mean` and `sigma` are of the underlying normal distribution, not the log-normal.

---

### Discrete Distributions

#### `binomial(n, p, size=None)`

Binomial distribution.

```python
np.random.seed(42)
np.random.binomial(10, 0.5, 5)  # [4, 8, 6, 5, 3]
```

**Returns:** int32

**Validation:**
- `ValueError: n < 0`
- `ValueError: p < 0, p > 1 or p is NaN`

#### `poisson(lam=1.0, size=None)`

Poisson distribution.

```python
np.random.seed(42)
np.random.poisson(5, 5)  # [5, 4, 4, 5, 5]
```

**Returns:** int32

**Validation:**
- `ValueError: lam < 0 or lam is NaN`
- `ValueError: lam value too large` (lam >= ~1e10)

#### `geometric(p, size=None)`

Geometric distribution (number of trials until first success).

```python
np.random.seed(42)
np.random.geometric(0.5, 5)  # [1, 5, 2, 2, 1]
```

**Returns:** int32

**Validation:**
- `ValueError: p <= 0, p > 1 or p contains NaNs`

#### `negative_binomial(n, p, size=None)`

Negative binomial distribution.

```python
np.random.seed(42)
np.random.negative_binomial(5, 0.5, 5)  # [3, 2, 3, 2, 5]
```

**Validation:**
- `ValueError: n <= 0`
- `ValueError: p < 0, p > 1 or p is NaN`

**Edge cases:**
- `negative_binomial(1, 0)` → large integers (no error!)
- `negative_binomial(1, 1)` → array of 0

#### `hypergeometric(ngood, nbad, nsample, size=None)`

Hypergeometric distribution.

```python
np.random.seed(42)
np.random.hypergeometric(10, 5, 7, 5)  # [5, 3, 6, 6, 5]
```

**Validation:**
- `ValueError: nsample < 1 or nsample is NaN`

#### `logseries(p, size=None)`

Logarithmic series distribution.

```python
np.random.seed(42)
np.random.logseries(0.9, 5)  # [9, 2, 2, 20, 3]
```

**Validation:**
- `ValueError: p < 0, p >= 1 or p is NaN`

**Edge case:** `logseries(0)` → array of 1 (no error!)

#### `zipf(a, size=None)`

Zipf distribution.

```python
np.random.seed(42)
np.random.zipf(2, 5)  # [1, 3, 1, 1, 2]
```

**Validation:**
- `ValueError: a <= 1 or a is NaN`

---

### Continuous Distributions

#### `beta(a, b, size=None)`

Beta distribution.

```python
np.random.seed(42)
np.random.beta(2, 5, 5)  # [0.35367666, 0.24855807, 0.41595909, 0.15996758, 0.55028308]
```

**Validation:**
- `ValueError: a <= 0`
- `ValueError: b <= 0`

#### `gamma(shape, scale=1.0, size=None)`

Gamma distribution.

```python
np.random.seed(42)
np.random.gamma(2, 1, 5)  # [2.39367939, 1.49446473, 1.38228358, 1.38230229, 4.64971441]
```

**Validation:**
- `ValueError: shape < 0`
- `ValueError: scale < 0`

**Edge cases:**
- `gamma(0, 1)` → array of 0.0 (no error!)
- `gamma(nan, 1)` → array of nan
- `gamma(inf, 1)` → array of inf

#### `exponential(scale=1.0, size=None)`

Exponential distribution.

```python
np.random.seed(42)
np.random.exponential(1, 5)  # [0.46926809, 3.01012143, 1.31674569, 0.91294255, 0.16962487]
```

**Validation:**
- `ValueError: scale < 0`

**Edge cases:**
- `exponential(0)` → array of 0.0
- `exponential(nan)` → array of nan
- `exponential(inf)` → array of inf

#### `chisquare(df, size=None)`

Chi-square distribution.

```python
np.random.seed(42)
np.random.chisquare(5, 5)  # [5.96627073, 3.93890591, 3.67991027, 3.67995362, 10.84321348]
```

**Validation:**
- `ValueError: df <= 0`

**Edge case:** `chisquare(nan)` → array of nan

#### `f(dfnum, dfden, size=None)`

F distribution.

```python
np.random.seed(42)
np.random.f(5, 10, 5)  # [1.36393455, 0.88058749, 1.66088313, 0.52073749, 3.11291601]
```

#### `standard_t(df, size=None)`

Student's t distribution.

```python
np.random.seed(42)
np.random.standard_t(5, 5)  # [0.55963354, -1.07574122, 1.33391804, -0.75446925, 0.60920065]
```

**Edge case:** `standard_t(nan)` → array of nan

#### `vonmises(mu, kappa, size=None)`

Von Mises distribution (circular normal).

```python
np.random.seed(42)
np.random.vonmises(0, 1, 5)  # [0.62690657, -1.17478453, 0.08884717, 1.55489819, -2.1288983]
```

**Validation:**
- `ValueError: kappa < 0`

#### `wald(mean, scale, size=None)`

Wald (inverse Gaussian) distribution.

```python
np.random.seed(42)
np.random.wald(1, 1, 5)  # [1.63516639, 1.14815282, 0.79166122, 1.26314598, 0.23479012]
```

**Validation:**
- `ValueError: mean <= 0`
- `ValueError: scale <= 0`

#### `weibull(a, size=None)`

Weibull distribution.

```python
np.random.seed(42)
np.random.weibull(2, 5)  # [0.68503145, 1.73497015, 1.1474954, 0.95548027, 0.4118554]
```

**Edge case:** `weibull(0)` → no error (returns values)

#### `pareto(a, size=None)`

Pareto distribution.

```python
np.random.seed(42)
np.random.pareto(2, 5)  # [0.26444595, 3.50442711, 0.93164669, 0.57849408, 0.08851288]
```

**Validation:**
- `ValueError: a <= 0`

#### `power(a, size=None)`

Power distribution.

```python
np.random.seed(42)
np.random.power(2, 5)  # [0.61199683, 0.9750458, 0.85556645, 0.77373024, 0.39499195]
```

**Validation:**
- `ValueError: a <= 0`

#### `rayleigh(scale=1.0, size=None)`

Rayleigh distribution.

```python
np.random.seed(42)
np.random.rayleigh(1, 5)  # [0.96878077, 2.45361832, 1.62280356, 1.35125316, 0.58245149]
```

**Validation:**
- `ValueError: scale < 0`

**Edge case:** `rayleigh(0)` → array of 0.0

#### `laplace(loc=0.0, scale=1.0, size=None)`

Laplace (double exponential) distribution.

```python
np.random.seed(42)
np.random.laplace(0, 1, 5)  # [-0.28890917, 2.31697425, 0.62359851, 0.21979537, -1.16463261]
```

**Edge cases:**
- `laplace(0, 0)` → array of 0.0
- `laplace(nan, 1)` → array of nan

#### `logistic(loc=0.0, scale=1.0, size=None)`

Logistic distribution.

```python
np.random.seed(42)
np.random.logistic(0, 1, 5)  # [-0.51278827, 2.95957976, 1.00476265, 0.39987857, -1.68815492]
```

**Edge case:** `logistic(0, 0)` → array of 0.0

#### `gumbel(loc=0.0, scale=1.0, size=None)`

Gumbel (extreme value type I) distribution.

```python
np.random.seed(42)
np.random.gumbel(0, 1, 5)  # [0.75658105, -1.10198042, -0.27516331, 0.09108232, 1.77416592]
```

**Edge case:** `gumbel(0, 0)` → array of 0.0

#### `triangular(left, mode, right, size=None)`

Triangular distribution.

```python
np.random.seed(42)
np.random.triangular(0, 0.5, 1, 5)  # [0.43274711, 0.8430196, 0.63393576, 0.5520371, 0.27930149]
```

**Validation:**
- `ValueError: left > mode`
- `ValueError: mode > right`
- `ValueError: left == right` (when left == mode == right)

#### `standard_cauchy(size=None)`

Standard Cauchy distribution.

```python
np.random.seed(42)
np.random.standard_cauchy(5)  # [-3.59249748, 0.42526319, 1.00007012, 2.05778128, -0.8652948]
```

#### `standard_exponential(size=None)`

Standard exponential distribution (scale=1).

```python
np.random.seed(42)
np.random.standard_exponential(5)  # [0.46926809, 3.01012143, 1.31674569, 0.91294255, 0.16962487]
```

#### `standard_gamma(shape, size=None)`

Standard gamma distribution (scale=1).

```python
np.random.seed(42)
np.random.standard_gamma(2, 5)  # [2.39367939, 1.49446473, 1.38228358, 1.38230229, 4.64971441]
```

#### `noncentral_chisquare(df, nonc, size=None)`

Non-central chi-square distribution.

```python
np.random.seed(42)
np.random.noncentral_chisquare(5, 1, 5)  # [5.52994719, 2.94728841, 12.42325435, 2.27267645, 4.83200133]
```

#### `noncentral_f(dfnum, dfden, nonc, size=None)`

Non-central F distribution.

```python
np.random.seed(42)
np.random.noncentral_f(5, 10, 1, 5)  # [2.08421137, 0.81334421, 1.3119517, 2.28476301, 0.48492134]
```

---

### Multivariate Distributions

#### `dirichlet(alpha, size=None)`

Dirichlet distribution.

```python
np.random.seed(42)
np.random.dirichlet([1, 1, 1], 3)
# [[0.09784297, 0.62761396, 0.27454307],
#  [0.72909200, 0.13546541, 0.13544259],
#  [0.02001195, 0.67261832, 0.30736973]]
```

**Validation:**
- `ValueError: alpha <= 0`

#### `multinomial(n, pvals, size=None)`

Multinomial distribution.

```python
np.random.seed(42)
np.random.multinomial(10, [0.2, 0.3, 0.5], 3)
# [[1, 6, 3],
#  [3, 3, 4],
#  [1, 2, 7]]
```

**Note:** `pvals` should sum to 1, but NumPy doesn't strictly enforce this.

#### `multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-08)`

Multivariate normal distribution.

```python
np.random.seed(42)
np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 3)
# [[ 0.49671415, -0.1382643 ],
#  [ 0.64768854,  1.52302986],
#  [-0.23415337, -0.23413696]]
```

**Parameters:**
- `check_valid`: 'warn', 'raise', or 'ignore' for covariance matrix validity
- `tol`: Tolerance for covariance matrix symmetry check

#### `multivariate_hypergeometric(colors, nsample, size=None, method='marginals')` [Generator only]

Multivariate hypergeometric distribution.

```python
rng = np.random.default_rng(42)
rng.multivariate_hypergeometric([10, 5, 3], 8)  # [5, 2, 1]
```

**Parameters:**
- `colors`: Number of items of each type
- `nsample`: Number of items to sample
- `method`: 'marginals' or 'count'

---

### Sequence Operations

#### `choice(a, size=None, replace=True, p=None)`

Random selection from array or range.

```python
np.random.seed(42)
np.random.choice(10, 5)              # [6, 3, 7, 4, 6]
np.random.choice([1,2,3,4,5], 3)     # Selection from array
np.random.choice(10, 5, replace=False)  # Without replacement
np.random.choice(10, 5, p=[0.1]*10)    # With probabilities
```

**Generator version** adds `axis` and `shuffle` parameters:
```python
rng = np.random.default_rng(42)
arr = np.arange(12).reshape(3, 4)
rng.choice(arr, 2, axis=0)  # Select rows
rng.choice(arr, 2, axis=1)  # Select columns
```

**Validation:**
- `ValueError: 'a' cannot be empty unless no samples are taken`

#### `permutation(x)`

Return a randomly permuted copy.

```python
np.random.seed(42)
np.random.permutation(10)  # [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]
np.random.permutation([1, 2, 3, 4, 5])  # Permute array
```

**Generator version** adds `axis` parameter:
```python
rng = np.random.default_rng(42)
rng.permutation(arr, axis=1)  # Permute along axis
```

#### `shuffle(x)`

Shuffle array in-place.

```python
arr = np.arange(10)
np.random.seed(42)
np.random.shuffle(arr)  # arr is now [8, 1, 5, 0, 7, 2, 9, 4, 3, 6]
```

**Note:** For multi-dimensional arrays, only shuffles along first axis.

**Generator version** adds `axis` parameter:
```python
rng = np.random.default_rng(42)
rng.shuffle(arr, axis=1)  # Shuffle along specific axis
```

#### `permuted(x, axis=None, out=None)` [Generator only]

Shuffle along axis with independent permutations per slice.

```python
rng = np.random.default_rng(42)
arr = np.arange(12).reshape(3, 4)

# Each row is shuffled independently
rng.permuted(arr, axis=1)
# [[ 3,  2,  1,  0],
#  [ 7,  6,  4,  5],
#  [ 8, 11, 10,  9]]
```

**Key difference from `shuffle(axis=...)`:**
- `shuffle(axis=1)`: Reorders columns, same order for all rows
- `permuted(axis=1)`: Each row gets its own independent shuffle

#### `bytes(length)`

Generate random bytes.

```python
np.random.seed(42)
np.random.bytes(10)  # b'\xc9\xa5...'
```

---

## Seeded Reference Values

All values generated with `seed(42)`, `size=5` unless noted.

### Core Functions

| Function | Output |
|----------|--------|
| `rand(5)` | `[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]` |
| `randn(5)` | `[0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337]` |
| `randint(100, size=5)` | `[51, 92, 14, 71, 60]` |
| `random(5)` | `[0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]` |
| `uniform(0, 10, 5)` | `[3.74540119, 9.50714306, 7.31993942, 5.98658484, 1.5601864]` |
| `normal(0, 1, 5)` | `[0.49671415, -0.1382643, 0.64768854, 1.52302986, -0.23415337]` |
| `choice(10, 5)` | `[6, 3, 7, 4, 6]` |
| `permutation(10)` | `[8, 1, 5, 0, 7, 2, 9, 4, 3, 6]` |

### Distributions

| Distribution | Output |
|--------------|--------|
| `beta(2, 5, 5)` | `[0.35367666, 0.24855807, 0.41595909, 0.15996758, 0.55028308]` |
| `binomial(10, 0.5, 5)` | `[4, 8, 6, 5, 3]` |
| `chisquare(5, 5)` | `[5.96627073, 3.93890591, 3.67991027, 3.67995362, 10.84321348]` |
| `exponential(1, 5)` | `[0.46926809, 3.01012143, 1.31674569, 0.91294255, 0.16962487]` |
| `f(5, 10, 5)` | `[1.36393455, 0.88058749, 1.66088313, 0.52073749, 3.11291601]` |
| `gamma(2, 1, 5)` | `[2.39367939, 1.49446473, 1.38228358, 1.38230229, 4.64971441]` |
| `geometric(0.5, 5)` | `[1, 5, 2, 2, 1]` |
| `gumbel(0, 1, 5)` | `[0.75658105, -1.10198042, -0.27516331, 0.09108232, 1.77416592]` |
| `hypergeometric(10, 5, 7, 5)` | `[5, 3, 6, 6, 5]` |
| `laplace(0, 1, 5)` | `[-0.28890917, 2.31697425, 0.62359851, 0.21979537, -1.16463261]` |
| `logistic(0, 1, 5)` | `[-0.51278827, 2.95957976, 1.00476265, 0.39987857, -1.68815492]` |
| `lognormal(0, 1, 5)` | `[1.64331272, 0.87086849, 1.91111824, 4.58609939, 0.79124045]` |
| `logseries(0.9, 5)` | `[9, 2, 2, 20, 3]` |
| `negative_binomial(5, 0.5, 5)` | `[3, 2, 3, 2, 5]` |
| `noncentral_chisquare(5, 1, 5)` | `[5.52994719, 2.94728841, 12.42325435, 2.27267645, 4.83200133]` |
| `noncentral_f(5, 10, 1, 5)` | `[2.08421137, 0.81334421, 1.3119517, 2.28476301, 0.48492134]` |
| `pareto(2, 5)` | `[0.26444595, 3.50442711, 0.93164669, 0.57849408, 0.08851288]` |
| `poisson(5, 5)` | `[5, 4, 4, 5, 5]` |
| `power(2, 5)` | `[0.61199683, 0.9750458, 0.85556645, 0.77373024, 0.39499195]` |
| `rayleigh(1, 5)` | `[0.96878077, 2.45361832, 1.62280356, 1.35125316, 0.58245149]` |
| `standard_cauchy(5)` | `[-3.59249748, 0.42526319, 1.00007012, 2.05778128, -0.8652948]` |
| `standard_exponential(5)` | `[0.46926809, 3.01012143, 1.31674569, 0.91294255, 0.16962487]` |
| `standard_gamma(2, 5)` | `[2.39367939, 1.49446473, 1.38228358, 1.38230229, 4.64971441]` |
| `standard_t(5, 5)` | `[0.55963354, -1.07574122, 1.33391804, -0.75446925, 0.60920065]` |
| `triangular(0, 0.5, 1, 5)` | `[0.43274711, 0.8430196, 0.63393576, 0.5520371, 0.27930149]` |
| `vonmises(0, 1, 5)` | `[0.62690657, -1.17478453, 0.08884717, 1.55489819, -2.1288983]` |
| `wald(1, 1, 5)` | `[1.63516639, 1.14815282, 0.79166122, 1.26314598, 0.23479012]` |
| `weibull(2, 5)` | `[0.68503145, 1.73497015, 1.1474954, 0.95548027, 0.4118554]` |
| `zipf(2, 5)` | `[1, 3, 1, 1, 2]` |

### Multivariate (size=3)

| Distribution | Output |
|--------------|--------|
| `dirichlet([1,1,1], 3)` | `[[0.098, 0.628, 0.275], [0.729, 0.135, 0.135], [0.020, 0.673, 0.307]]` |
| `multinomial(10, [.2,.3,.5], 3)` | `[[1, 6, 3], [3, 3, 4], [1, 2, 7]]` |
| `multivariate_normal([0,0], [[1,0],[0,1]], 3)` | `[[0.497, -0.138], [0.648, 1.523], [-0.234, -0.234]]` |

---

## Validation Rules

### Seed Validation

| Input | Result |
|-------|--------|
| `seed(0)` to `seed(2**32-1)` | Valid |
| `seed(-1)` | `ValueError: Seed must be between 0 and 2**32 - 1` |
| `seed(2**32)` | `ValueError: Seed must be between 0 and 2**32 - 1` |
| `seed(42.0)` | `TypeError: Cannot cast scalar from dtype('float64') to dtype('int64')` |
| `seed(None)` | Valid - uses system entropy |
| `seed([])` | `ValueError: Seed must be non-empty` |
| `seed([1,2,3,4])` | Valid - array seeding |
| `seed([[1,2],[3,4]])` | `ValueError: Seed array must be 1-d` |

### Size Parameter

| Input | Result |
|-------|--------|
| `size=None` | Returns Python scalar (float/int) |
| `size=()` | Returns 0-d ndarray (shape=(), ndim=0) |
| `size=5` | Returns 1-d ndarray (shape=(5,)) |
| `size=(2,3)` | Returns 2-d ndarray (shape=(2,3)) |
| `size=0` | Returns empty 1-d ndarray (shape=(0,)) |
| `size=(5,0)` | Returns empty 2-d ndarray (shape=(5,0)) |
| `size=-1` | `ValueError: negative dimensions are not allowed` |

### Distribution Validation Summary

| Distribution | Parameter | Constraint | Error |
|--------------|-----------|------------|-------|
| `beta` | `a`, `b` | > 0 | `ValueError: a <= 0` / `b <= 0` |
| `binomial` | `n` | >= 0 | `ValueError: n < 0` |
| `binomial` | `p` | [0, 1] | `ValueError: p < 0, p > 1 or p is NaN` |
| `chisquare` | `df` | > 0 | `ValueError: df <= 0` |
| `exponential` | `scale` | >= 0 | `ValueError: scale < 0` |
| `gamma` | `shape` | >= 0 | `ValueError: shape < 0` |
| `gamma` | `scale` | >= 0 | `ValueError: scale < 0` |
| `geometric` | `p` | (0, 1] | `ValueError: p <= 0, p > 1 or p contains NaNs` |
| `hypergeometric` | `nsample` | >= 1 | `ValueError: nsample < 1 or nsample is NaN` |
| `hypergeometric` | `nsample` | <= ngood + nbad | `ValueError: ngood + nbad < nsample` |
| `logseries` | `p` | [0, 1) | `ValueError: p < 0, p >= 1 or p is NaN` |
| `negative_binomial` | `n` | > 0 | `ValueError: n <= 0` |
| `pareto` | `a` | > 0 | `ValueError: a <= 0` |
| `poisson` | `lam` | >= 0, < ~1e10 | `ValueError: lam < 0 or lam is NaN` |
| `power` | `a` | > 0 | `ValueError: a <= 0` |
| `randint` | `low`, `high` | low < high | `ValueError: low >= high` |
| `rayleigh` | `scale` | >= 0 | `ValueError: scale < 0` |
| `triangular` | `left`, `mode`, `right` | left <= mode <= right | `ValueError: left > mode` |
| `uniform` | `low`, `high` | finite | `OverflowError: Range exceeds valid bounds` |
| `vonmises` | `kappa` | >= 0 | `ValueError: kappa < 0` |
| `wald` | `mean`, `scale` | > 0 | `ValueError: mean <= 0` / `scale <= 0` |
| `zipf` | `a` | > 1 | `ValueError: a <= 1 or a is NaN` |
| `dirichlet` | `alpha` | all > 0 | `ValueError: alpha <= 0` |
| `multinomial` | `n` | >= 0 | `ValueError: n < 0` |
| `choice` | `a` | non-empty | `ValueError: 'a' cannot be empty unless no samples are taken` |

---

## Edge Cases

### No-Error Edge Cases

These inputs do NOT throw errors in NumPy - they produce nan/inf or degenerate outputs:

| Function | Input | Output |
|----------|-------|--------|
| `normal(nan, 1)` | nan loc | Array of nan |
| `normal(0, inf)` | inf scale | Array of inf |
| `normal(0, 0)` | zero scale | Array of loc (constant) |
| `gamma(0, 1)` | zero shape | Array of 0.0 |
| `gamma(nan, 1)` | nan shape | Array of nan |
| `gamma(inf, 1)` | inf shape | Array of inf |
| `exponential(0)` | zero scale | Array of 0.0 |
| `exponential(nan)` | nan scale | Array of nan |
| `exponential(inf)` | inf scale | Array of inf |
| `beta(nan, 1)` | nan a | Array of nan |
| `beta(inf, 1)` | inf a | Array of nan (inf/inf) |
| `standard_gamma(0)` | zero shape | Array of 0.0 |
| `standard_gamma(nan)` | nan shape | Array of nan |
| `chisquare(nan)` | nan df | Array of nan |
| `standard_t(nan)` | nan df | Array of nan |
| `laplace(0, 0)` | zero scale | Array of 0.0 |
| `laplace(nan, 1)` | nan loc | Array of nan |
| `logistic(0, 0)` | zero scale | Array of 0.0 |
| `gumbel(0, 0)` | zero scale | Array of 0.0 |
| `lognormal(0, 0)` | zero sigma | Array of 1.0 |
| `logseries(0)` | zero p | Array of 1 |
| `rayleigh(0)` | zero scale | Array of 0.0 |
| `negative_binomial(1, 0)` | zero p | Large integers |
| `negative_binomial(1, 1)` | one p | Array of 0 |
| `weibull(0)` | zero a | Values (no error) |
| `multinomial(10, [0.5, 0.6])` | pvals > 1 | Works (no validation!) |

---

## Implicit Behaviors

### Return Type: Scalar vs Array

The `size` parameter controls whether a scalar or array is returned:

| `size` value | Return type | Shape | Notes |
|--------------|-------------|-------|-------|
| `None` (default) | Python scalar (`float`/`int`) | N/A | NOT an ndarray |
| `()` | 0-d ndarray | `()` | `ndim=0`, `size=1` |
| `5` | 1-d ndarray | `(5,)` | |
| `(2, 3)` | 2-d ndarray | `(2, 3)` | |
| `0` | Empty 1-d ndarray | `(0,)` | |
| `(5, 0)` | Empty 2-d ndarray | `(5, 0)` | |

**0-d array properties:**
```python
arr0d = np.random.random(size=())
arr0d.shape   # ()
arr0d.ndim    # 0
arr0d.size    # 1
arr0d.item()  # Extract as Python scalar
arr0d[()]     # Also extracts scalar
```

### Default Return Dtypes

| Category | Functions | Default dtype |
|----------|-----------|---------------|
| Floating | `rand`, `randn`, `random`, `uniform`, `normal`, `standard_normal`, `beta`, `gamma`, `exponential`, `chisquare`, `f`, `standard_t`, `vonmises`, `wald`, `weibull`, `pareto`, `power`, `rayleigh`, `laplace`, `logistic`, `gumbel`, `triangular`, `lognormal`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `noncentral_chisquare`, `noncentral_f`, `dirichlet` | `float64` |
| Integer | `randint`, `choice`, `permutation`, `binomial`, `poisson`, `geometric`, `negative_binomial`, `hypergeometric`, `zipf`, `logseries`, `multinomial` | `int32` |

**Note:** `randint` default is `int32`, not `int64`!

### Broadcasting in Distribution Parameters

Most distribution parameters accept arrays and broadcast:

```python
# Array parameters broadcast with size
np.random.seed(42)
np.random.uniform(low=[0, 1, 2], high=10, size=3)
# [3.74540119, 9.55642876, 7.85595153]

np.random.normal(loc=[0, 10, 100], scale=1, size=3)
# [0.49671415, 9.8617357, 100.64768854]

np.random.poisson(lam=[1, 5, 10], size=3)
# [1, 4, 14]
```

**Shape inference when `size=None`:**

When `size` is not provided, output shape is inferred from parameter broadcasting:

```python
np.random.uniform([0, 1], [10, 20])  # shape=(2,)
np.random.normal([[0], [1]], [[1, 2], [3, 4]])  # shape=(2, 2)
```

### Multivariate Distribution Shapes

Multivariate distributions append their output dimension to the size:

| Distribution | Parameters | size=None | size=3 | size=(2,3) |
|--------------|------------|-----------|--------|------------|
| `multivariate_normal` | mean=(k,) | (k,) | (3, k) | (2, 3, k) |
| `dirichlet` | alpha=(k,) | (k,) | (3, k) | (2, 3, k) |
| `multinomial` | pvals=(k,) | (k,) | (3, k) | (2, 3, k) |
| `multivariate_hypergeometric` | colors=(k,) | (k,) | (3, k) | (2, 3, k) |

### Sequence Operation Details

#### `shuffle(x)` Behavior

- **Returns:** `None` (in-place modification)
- **Multi-dimensional:** Only shuffles along axis 0 (rows)
- **Generator version:** Accepts `axis` parameter

```python
arr = np.arange(10)
result = np.random.shuffle(arr)  # result is None
# arr is modified in-place

arr2d = np.arange(12).reshape(3, 4)
np.random.shuffle(arr2d)  # Only shuffles rows, not elements
```

#### `permutation(x)` Behavior

- **Returns:** New array (copy, original unchanged)
- **Multi-dimensional:** Only permutes along axis 0 (rows)
- **Generator version:** Accepts `axis` parameter

```python
arr = np.arange(10)
result = np.random.permutation(arr)  # New array
# arr is unchanged
```

#### `choice()` Validation

| Condition | Error |
|-----------|-------|
| Empty `a` with size > 0 | `ValueError: 'a' cannot be empty unless no samples are taken` |
| `replace=False` and size > len(a) | `ValueError: Cannot take a larger sample than population when 'replace=False'` |
| Negative probability | `ValueError: probabilities are not non-negative` |
| Probabilities don't sum to 1 | `ValueError: probabilities do not sum to 1` |

### Generator-Specific Parameters

#### `out` Parameter

Many Generator methods accept an `out` parameter for in-place output:

```python
rng = np.random.default_rng(42)
out = np.zeros(5)
result = rng.random(size=5, out=out)
# result is out (same object)
# out is filled with random values
```

**Validation:**
- Wrong shape: `ValueError`
- Wrong dtype: `TypeError`

#### `dtype` Parameter

Generator methods support `dtype` for output type:

```python
rng = np.random.default_rng(42)
rng.random(5, dtype=np.float32)  # Returns float32 array
rng.standard_normal(5, dtype=np.float32)  # Returns float32 array
```

**Supported:** `np.float32`, `np.float64` (default)

#### `method` Parameter

Some Generator methods have algorithm selection:

```python
# standard_exponential: 'zig' (default, Ziggurat) or 'inv' (inverse transform)
rng.standard_exponential(3, method='zig')  # Faster
rng.standard_exponential(3, method='inv')  # Different values

# multivariate_normal: 'svd' (default), 'cholesky', 'eigh'
rng.multivariate_normal(mean, cov, method='cholesky')
```

### `default_rng()` Input Types

| Input | Behavior |
|-------|----------|
| `None` | Create new Generator with system entropy |
| `int` | Create Generator seeded with integer |
| `SeedSequence` | Create Generator from SeedSequence |
| `BitGenerator` | Wrap in Generator |
| `Generator` | **Return same object** (no copy) |

```python
rng1 = np.random.default_rng(42)
rng2 = np.random.default_rng(rng1)
rng1 is rng2  # True! Same object
```

### State Format Details

#### Legacy Format (`get_state(legacy=True)`)

Returns a tuple:

```python
('MT19937',           # [0] Algorithm name
 array([...624...]),  # [1] uint32[624] state array
 624,                 # [2] Position (0-624)
 0,                   # [3] has_gauss (0 or 1)
 0.0)                 # [4] cached_gaussian
```

**Gaussian caching:** After calling `randn()`, the state includes the cached second Gaussian value from Box-Muller:

```python
np.random.seed(42)
np.random.randn()
state = np.random.get_state()
state[3]  # 1 (has_gauss = True)
state[4]  # -0.138264... (cached value)
```

#### Dict Format (`get_state(legacy=False)`)

Returns a dict:

```python
{
    'bit_generator': 'MT19937',
    'state': {'key': array([...624...]), 'pos': 624},
    'has_gauss': 0,
    'gauss': 0.0
}
```

### BitGenerator Properties and Methods

| Property/Method | Description |
|-----------------|-------------|
| `state` | Get/set state dict |
| `seed_seq` | Associated SeedSequence |
| `lock` | Threading `RLock` for thread safety |
| `capsule` | PyCapsule for C API |
| `cffi` | CFFI interface |
| `ctypes` | ctypes interface |
| `random_raw(size, output)` | Raw uint64 values |
| `spawn(n)` | Create n child BitGenerators |
| `jumped(jumps)` | Return jumped copy (advance by 2^128 steps) |
| `advance(delta)` | Advance state by delta steps (PCG64/Philox only) |

### `randint` Dtype Bounds Checking

When specifying dtype, values must fit in the dtype range:

```python
# Valid
np.random.randint(0, 127, dtype=np.int8)

# Invalid - high out of bounds
np.random.randint(0, 256, dtype=np.int8)
# ValueError: high is out of bounds for int8

# Invalid - low out of bounds for unsigned
np.random.randint(-1, 10, dtype=np.uint8)
# ValueError: low is out of bounds for uint8
```

### Negative Axis Support (Generator)

Generator methods with `axis` parameter support negative indices:

```python
rng = np.random.default_rng(42)
arr = np.arange(12).reshape(3, 4)

rng.permuted(arr, axis=-1)  # Same as axis=1
rng.permuted(arr, axis=-2)  # Same as axis=0
```

### Special Float Value Behavior

| Distribution | nan input | inf input |
|--------------|-----------|-----------|
| `normal(loc, scale)` | nan → array of nan | inf → array of inf |
| `gamma(shape, scale)` | nan → array of nan | inf → array of inf |
| `exponential(scale)` | nan → array of nan | inf → array of inf |
| `uniform(low, high)` | — | `OverflowError` |
| `beta(a, b)` | nan → array of nan | inf → array of nan |

### Validation Error Messages

NumPy uses specific error message formats. Key patterns:

| Pattern | Example |
|---------|---------|
| `X <= 0` | `ValueError: a <= 0` |
| `X < 0` | `ValueError: scale < 0` |
| `X < 0, X > 1 or X is NaN` | `ValueError: p < 0, p > 1 or p is NaN` |
| `low >= high` | `ValueError: low >= high` |
| `high is out of bounds for X` | `ValueError: high is out of bounds for int8` |

---

## NumSharp Implementation Status

### Fully Implemented (40)

| Category | Functions |
|----------|-----------|
| Uniform | `rand`, `random`, `random_sample`, `uniform` |
| Normal | `randn`, `standard_normal`, `normal`, `lognormal` |
| Integer | `randint` |
| Sequence | `choice`, `permutation`, `shuffle` |
| Discrete | `bernoulli`*, `binomial`, `geometric`, `hypergeometric`, `logseries`, `negative_binomial`, `poisson`, `zipf` |
| Continuous | `beta`, `chisquare`, `exponential`, `f`, `gamma`, `gumbel`, `laplace`, `logistic`, `noncentral_chisquare`, `noncentral_f`, `pareto`, `power`, `rayleigh`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_t`, `triangular`, `vonmises`, `wald`, `weibull` |
| Multivariate | `dirichlet`, `multinomial`, `multivariate_normal` |
| State | `seed`, `get_state`, `set_state` |

\* `bernoulli` is NumSharp extra, not in NumPy

### Missing - Legacy API (4)

| Function | Priority | Notes |
|----------|----------|-------|
| `bytes(length)` | Low | Generate random bytes |
| `ranf(size)` | Low | Alias for `random_sample` |
| `sample(size)` | Low | Alias for `random_sample` |
| `random_integers` | Skip | Deprecated in NumPy |

### Missing - Generator API (7)

| Function | Priority | Notes |
|----------|----------|-------|
| `Generator` class | High | Modern RNG container |
| `default_rng(seed)` | High | Factory for Generator |
| `integers(low, high, endpoint)` | High | Modern randint replacement |
| `permuted(x, axis)` | High | Independent axis shuffle |
| `shuffle(x, axis)` | Medium | Axis-aware in-place shuffle |
| `multivariate_hypergeometric` | Medium | Multivariate hypergeometric |
| `spawn(n)` | High | Create child generators |

### Missing - BitGenerators (5)

| Class | Priority | Notes |
|-------|----------|-------|
| `PCG64` | High | NumPy's new default |
| `PCG64DXSM` | Medium | PCG variant |
| `Philox` | Medium | Counter-based, parallelizable |
| `SFC64` | Low | Fastest option |
| `SeedSequence` | High | Proper parallel seeding |

### Implementation Roadmap

#### Phase 1: Complete Legacy API (Low effort)
- Add `ranf()` and `sample()` aliases
- Add `bytes()` function

#### Phase 2: Generator Infrastructure (High effort)
- Implement `Generator` class
- Implement `default_rng()` factory
- Port all distribution methods with `out` parameter support

#### Phase 3: BitGenerators (Medium effort)
- Implement `PCG64` (highest priority)
- Implement `SeedSequence`
- Add `Philox`, `SFC64`, `PCG64DXSM`

#### Phase 4: Generator-Only Features (Medium effort)
- Add `integers()` with endpoint
- Add `permuted()` with axis
- Add `shuffle()` / `permutation()` axis support
- Add `multivariate_hypergeometric()`
- Add `spawn()` for parallel streams

---

## MT19937 Implementation Details

This section provides the exact algorithm details needed to implement NumPy-compatible MT19937.

### Constants

```c
#define RK_STATE_LEN 624       // State array length (N)
#define _MT19937_N 624         // Period parameter N
#define _MT19937_M 397         // Period parameter M
#define MATRIX_A 0x9908b0dfUL  // Constant vector A
#define UPPER_MASK 0x80000000UL // Most significant w-r bits
#define LOWER_MASK 0x7fffffffUL // Least significant r bits
```

### State Structure

```c
typedef struct {
    uint32_t key[624];  // State array
    int pos;            // Current position in state array
} mt19937_state;
```

### Seeding (Single Integer)

NumPy uses Knuth's PRNG for seeding (different from the generator itself):

```c
void mt19937_seed(mt19937_state *state, uint32_t seed) {
    seed &= 0xffffffffUL;
    for (int pos = 0; pos < 624; pos++) {
        state->key[pos] = seed;
        seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
    }
    state->pos = 624;  // Force regeneration on first use
}
```

**Key constant:** `1812433253` (Knuth's multiplier)

### Array Seeding

For seeding with an array of integers:

```c
void mt19937_init_by_array(mt19937_state *state, uint32_t *init_key, int key_length) {
    // First, initialize with fixed seed
    init_genrand(state, 19650218UL);

    // Then mix in the key array
    int i = 1, j = 0;
    int k = (624 > key_length) ? 624 : key_length;

    for (; k; k--) {
        state->key[i] = (state->key[i] ^
            ((state->key[i-1] ^ (state->key[i-1] >> 30)) * 1664525UL))
            + init_key[j] + j;
        state->key[i] &= 0xffffffffUL;
        i++; j++;
        if (i >= 624) { state->key[0] = state->key[623]; i = 1; }
        if (j >= key_length) { j = 0; }
    }

    for (k = 623; k; k--) {
        state->key[i] = (state->key[i] ^
            ((state->key[i-1] ^ (state->key[i-1] >> 30)) * 1566083941UL)) - i;
        state->key[i] &= 0xffffffffUL;
        i++;
        if (i >= 624) { state->key[0] = state->key[623]; i = 1; }
    }

    state->key[0] = 0x80000000UL;  // MSB=1 ensures non-zero initial state
}
```

**Key constants:**
- `19650218` - Initial seed for array seeding
- `1664525` - First mixing multiplier
- `1566083941` - Second mixing multiplier

### State Generation (Twist)

When all 624 values have been used, generate 624 new ones:

```c
void mt19937_gen(mt19937_state *state) {
    uint32_t y;

    // First 227 values (N - M = 624 - 397 = 227)
    for (int i = 0; i < 227; i++) {
        y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
        state->key[i] = state->key[i + 397] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
    }

    // Next 396 values (up to N - 1)
    for (int i = 227; i < 623; i++) {
        y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
        state->key[i] = state->key[i - 227] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);
    }

    // Last value wraps around
    y = (state->key[623] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
    state->key[623] = state->key[396] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

    state->pos = 0;
}
```

### Extracting Random Values

Raw 32-bit integer with tempering:

```c
uint32_t mt19937_next(mt19937_state *state) {
    if (state->pos == 624) {
        mt19937_gen(state);  // Generate new batch
    }

    uint32_t y = state->key[state->pos++];

    // Tempering transformation
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}
```

**Tempering constants:**
- `0x9d2c5680` - Tempering mask B
- `0xefc60000` - Tempering mask C

### Converting to Double [0, 1)

NumPy uses a 53-bit precision method:

```c
double mt19937_next_double(mt19937_state *state) {
    // Use 27 bits from one call, 26 bits from another
    int32_t a = mt19937_next(state) >> 5;   // 27 bits
    int32_t b = mt19937_next(state) >> 6;   // 26 bits

    // Combine for 53-bit mantissa (IEEE 754 double precision)
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
```

**Key values:**
- `67108864 = 2^26` (shift factor for combining)
- `9007199254740992 = 2^53` (normalization factor)

### Gaussian Caching

The legacy `RandomState` caches one Gaussian value for Box-Muller:

```c
typedef struct {
    int has_gauss;      // 0 or 1
    double gauss;       // Cached value
} augmented_state;
```

This means `get_state()` / `set_state()` must save/restore this cached value for reproducibility.

### C# Implementation Template

```csharp
public sealed class MT19937
{
    private const int N = 624;
    private const int M = 397;
    private const uint MATRIX_A = 0x9908b0dfU;
    private const uint UPPER_MASK = 0x80000000U;
    private const uint LOWER_MASK = 0x7fffffffU;

    private uint[] key = new uint[N];
    private int pos;

    public void Seed(uint seed)
    {
        seed &= 0xffffffffU;
        for (int i = 0; i < N; i++)
        {
            key[i] = seed;
            seed = (1812433253U * (seed ^ (seed >> 30)) + (uint)(i + 1)) & 0xffffffffU;
        }
        pos = N;
    }

    private void Generate()
    {
        uint y;
        for (int i = 0; i < N - M; i++)
        {
            y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
            key[i] = key[i + M] ^ (y >> 1) ^ ((y & 1) == 0 ? 0 : MATRIX_A);
        }
        for (int i = N - M; i < N - 1; i++)
        {
            y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
            key[i] = key[i + (M - N)] ^ (y >> 1) ^ ((y & 1) == 0 ? 0 : MATRIX_A);
        }
        y = (key[N - 1] & UPPER_MASK) | (key[0] & LOWER_MASK);
        key[N - 1] = key[M - 1] ^ (y >> 1) ^ ((y & 1) == 0 ? 0 : MATRIX_A);

        pos = 0;
    }

    public uint NextUInt32()
    {
        if (pos == N) Generate();
        uint y = key[pos++];

        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);

        return y;
    }

    public double NextDouble()
    {
        int a = (int)(NextUInt32() >> 5);  // 27 bits
        int b = (int)(NextUInt32() >> 6);  // 26 bits
        return (a * 67108864.0 + b) / 9007199254740992.0;
    }
}
```

---

## Related Documentation

- [NumPy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html)
- [SPEC 7: Seeding Pseudo-Random Number Generation](https://scientific-python.org/specs/spec-0007/)
- [NEP 19: Random Number Generator Policy](https://numpy.org/neps/nep-0019-rng-policy.html)
- [Mersenne Twister Home Page](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html)
- [MT19937 Paper](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/ARTICLES/mt.pdf)
- NumPy Source: `numpy/random/src/mt19937/mt19937.c`
- NumPy Source: `numpy/random/_mt19937.pyx`
- NumPy Source: `numpy/random/mtrand.pyx`
- NumSharp Issue #559: [SPEC7] Seeding Pseudo-Random Number Generation
- NumSharp Issue #553: [NEP01 NEP19] File Format and RNG Interoperability
