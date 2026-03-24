# NumPy Random Number Generation - Implementation Reference

This document describes NumPy's random number generation implementation to guide NumSharp's alignment with NumPy 2.x behavior.

## Current Problem

NumSharp's `Randomizer` class uses **.NET's Subtractive Generator** (Knuth's algorithm from "Numerical Recipes in C"), while NumPy uses the **Mersenne Twister (MT19937)** algorithm. This causes:

- Same seed produces completely different sequences
- No seed compatibility between NumPy and NumSharp
- Statistical properties differ slightly

**Example of the mismatch:**
```
NumPy seed=42:    rand() = 0.3745401188473625
NumSharp seed=42: rand() = 0.668106465911542
```

## NumPy's Architecture

NumPy 1.17+ introduced a new random API with pluggable BitGenerators:

```
┌─────────────────────────────────────────────────┐
│                  User API                        │
│  np.random.rand(), np.random.normal(), etc.     │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│              Generator / RandomState             │
│  High-level distribution sampling               │
│  (Box-Muller, inverse transform, etc.)          │
└─────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│               BitGenerator                       │
│  MT19937, PCG64, Philox, SFC64                  │
│  Produces raw random bits (uint32, uint64)      │
└─────────────────────────────────────────────────┘
```

### Legacy API (np.random.*)

The global `np.random.*` functions use a singleton `RandomState` backed by MT19937:

```python
# These all use the legacy singleton RandomState
np.random.seed(42)
np.random.rand()
np.random.randn()
```

### Modern API (Generator)

```python
from numpy.random import Generator, MT19937
rng = Generator(MT19937(seed=42))
rng.random()
```

## MT19937 Algorithm Details

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

## Gaussian (Normal) Distribution

NumPy uses the **Ziggurat algorithm** for normal distribution, not Box-Muller:

```
Location: numpy/random/src/distributions/distributions.c
Function: random_standard_normal()
```

However, the legacy `RandomState` caches one Gaussian value:

```c
typedef struct {
    int has_gauss;      // 0 or 1
    double gauss;       // Cached value
} augmented_state;
```

This means `get_state()` / `set_state()` must save/restore this cached value for reproducibility.

## State Serialization

NumPy's `get_state()` returns:
```python
('MT19937',           # Algorithm name
 array([...624...]),  # uint32 state array
 pos,                 # int: position in state (0-624)
 has_gauss,           # int: 0 or 1
 cached_gaussian)     # float: cached normal value
```

## Implementation Checklist for NumSharp

To achieve NumPy compatibility, NumSharp must:

### 1. Replace Randomizer with MT19937

```csharp
public class MT19937
{
    private const int N = 624;
    private const int M = 397;
    private const uint MATRIX_A = 0x9908b0dfU;
    private const uint UPPER_MASK = 0x80000000U;
    private const uint LOWER_MASK = 0x7fffffffU;

    private uint[] key = new uint[N];
    private int pos;
}
```

### 2. Implement Exact Seeding

```csharp
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
```

### 3. Implement Generation (Twist)

```csharp
private void Generate()
{
    uint y;
    for (int i = 0; i < N - M; i++)
    {
        y = (key[i] & UPPER_MASK) | (key[i + 1] & LOWER_MASK);
        key[i] = key[i + M] ^ (y >> 1) ^ ((y & 1) == 0 ? 0 : MATRIX_A);
    }
    // ... rest of generation
    pos = 0;
}
```

### 4. Implement Tempering

```csharp
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
```

### 5. Implement Double Conversion

```csharp
public double NextDouble()
{
    int a = (int)(NextUInt32() >> 5);  // 27 bits
    int b = (int)(NextUInt32() >> 6);  // 26 bits
    return (a * 67108864.0 + b) / 9007199254740992.0;
}
```

### 6. Cache Gaussian Value

```csharp
private bool _hasGauss;
private double _gaussCache;

// Include in state serialization
```

## Verification

After implementation, verify with:

```python
# Python
import numpy as np
np.random.seed(42)
print([np.random.rand() for _ in range(5)])
# [0.3745401188473625, 0.9507143064099162, 0.7319939418114051,
#  0.5986584841970366, 0.15601864044243652]
```

```csharp
// C#
np.random.seed(42);
var result = np.random.rand(5);
// Should produce identical values
```

## References

1. [Mersenne Twister Home Page](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/emt.html)
2. [NumPy Random Documentation](https://numpy.org/doc/stable/reference/random/index.html)
3. [MT19937 Paper](http://www.math.sci.hiroshima-u.ac.jp/m-mat/MT/ARTICLES/mt.pdf)
4. NumPy Source: `numpy/random/src/mt19937/mt19937.c`
5. NumPy Source: `numpy/random/_mt19937.pyx`
6. NumPy Source: `numpy/random/mtrand.pyx`
