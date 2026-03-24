# NEP 19 - Random Number Generator Policy

**Status:** Final
**NumSharp Impact:** HIGH - NumSharp claims 1-to-1 seed/state matching with NumPy

## Summary

Defines the new `Generator` API and establishes policy for RNG stream compatibility.

## Key Policy Change

**NumPy no longer guarantees stream compatibility across versions.**

> "The standard practice now for bit-for-bit reproducible research is to pin all of the versions of code of your software stack."

## Two RNG APIs

### Legacy: `RandomState`
```python
np.random.seed(42)
np.random.random()
np.random.normal()
```
- Fixed behavior for backwards compatibility
- Stream compatibility maintained within major versions
- Use for: unit testing, legacy code

### New: `Generator`
```python
from numpy.random import Generator, MT19937
rng = Generator(MT19937(42))
rng.random()
rng.standard_normal()
```
- No stream compatibility guarantee
- Allows algorithm improvements
- Use for: new code, best practices

## BitGenerator Infrastructure

BitGenerators are the core PRNG algorithms. Examples:
- `MT19937` - Mersenne Twister (legacy default)
- `PCG64` - Permuted Congruential Generator (new default)
- `Philox` - Counter-based RNG
- `SFC64` - Small Fast Chaotic

### Stream-Compatibility Guarantees (within version)

Only these methods are guaranteed stable:
```python
bg.bytes()      # Raw byte output
bg.integers()   # Integer generation
bg.random()     # Uniform [0,1) floats
```

## NumSharp Implementation Notes

NumSharp's `np.random` module in `RandomSampling/` claims 1-to-1 matching.

### What This Means

1. **Match legacy `RandomState` behavior** - for compatibility with existing NumPy code
2. **Same seed → same sequence** - critical for reproducibility
3. **Algorithm-specific** - must use same PRNG algorithm (likely MT19937)

### Verification Approach

```python
# Python
np.random.seed(42)
expected = [np.random.random() for _ in range(10)]
```

```csharp
// C#
np.random.seed(42);
var actual = Enumerable.Range(0, 10).Select(_ => np.random.random()).ToArray();
// actual should match expected exactly
```

### Implemented Distributions

- `rand`, `randn` - uniform, normal
- `randint` - integers
- `uniform`, `choice`, `shuffle`, `permutation`
- `beta`, `binomial`, `gamma`, `poisson`
- `exponential`, `geometric`, `lognormal`
- `chisquare`, `bernoulli`

## References

- [NEP 19 Full Text](https://numpy.org/neps/nep-0019-rng-policy.html)
- `src/NumSharp.Core/RandomSampling/`
