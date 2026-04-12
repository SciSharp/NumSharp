# Random Number Generator Migration Plan

## Objective

Replace NumSharp's current `Randomizer` (based on .NET's Subtractive Generator) with a NumPy-compatible **MT19937 (Mersenne Twister)** implementation to achieve 100% seed compatibility with NumPy 2.x.

## Current State Analysis

### Existing Files

| File | Purpose | Changes Needed |
|------|---------|----------------|
| `Randomizer.cs` | Core RNG (Subtractive Generator) | **Replace entirely** with MT19937 |
| `NativeRandomState.cs` | State serialization | Update for MT19937 state format |
| `np.random.cs` | NumPyRandom base class | Add Gaussian caching |
| `np.random.*.cs` (40 files) | Distribution implementations | Verify algorithms match NumPy |

### Current Architecture

```
NumPyRandom
├── randomizer: Randomizer (Subtractive Generator)
├── NextGaussian() - Box-Muller (no caching)
└── seed(), get_state(), set_state()

Randomizer
├── SeedArray[56] - int32 state
├── inext, inextp - position indices
├── NextDouble() → double [0,1)
├── Next(max) → int [0,max)
└── Serialize/Deserialize
```

### Target Architecture (NumPy-compatible)

```
NumPyRandom
├── bitGenerator: MT19937
├── hasGauss: bool (cached Gaussian flag)
├── gaussCache: double (cached Gaussian value)
├── NextGaussian() - with caching for state reproducibility
└── seed(), get_state(), set_state()

MT19937
├── key[624] - uint32 state array
├── pos - position (0-624)
├── NextUInt32() → uint32
├── NextDouble() → double [0,1) using 53-bit precision
└── Serialize/Deserialize (NumPy-compatible format)
```

## Migration Phases

### Phase 1: Implement MT19937 Core

**Goal:** Create new `MT19937.cs` with NumPy-identical algorithm

**Files to create:**
- `src/NumSharp.Core/RandomSampling/MT19937.cs`

**Implementation:**

```csharp
public sealed class MT19937 : ICloneable
{
    // Constants (must match NumPy exactly)
    private const int N = 624;
    private const int M = 397;
    private const uint MATRIX_A = 0x9908b0dfU;
    private const uint UPPER_MASK = 0x80000000U;
    private const uint LOWER_MASK = 0x7fffffffU;

    // State
    private uint[] key = new uint[N];
    private int pos;

    // Methods
    public void Seed(uint seed) { ... }
    public void SeedByArray(uint[] initKey) { ... }
    private void Generate() { ... }  // Twist operation
    public uint NextUInt32() { ... } // With tempering
    public double NextDouble() { ... } // 53-bit precision
}
```

**Verification tests:**
```csharp
[Test]
public void MT19937_Seed42_MatchesNumPy()
{
    var mt = new MT19937();
    mt.Seed(42);

    // First 5 raw uint32 values from NumPy's MT19937
    Assert.That(mt.NextUInt32(), Is.EqualTo(0x...));
    // ...
}
```

**Estimated effort:** 4-6 hours

---

### Phase 2: Update State Serialization

**Goal:** Make `get_state()` / `set_state()` NumPy-compatible

**Files to modify:**
- `src/NumSharp.Core/RandomSampling/NativeRandomState.cs`
- `src/NumSharp.Core/RandomSampling/MT19937.cs`

**NumPy state format:**
```python
('MT19937',           # Algorithm identifier
 array([...624...]),  # uint32[624] state array
 pos,                 # int: position (0-624)
 has_gauss,           # int: 0 or 1
 cached_gaussian)     # float: cached value
```

**Implementation:**

```csharp
public struct NativeRandomState
{
    public string Algorithm;      // "MT19937"
    public uint[] Key;            // uint32[624]
    public int Pos;               // 0-624
    public int HasGauss;          // 0 or 1
    public double CachedGaussian; // Cached normal value
}
```

**Backward compatibility:**
- Detect old format (byte[] with 56 ints) and throw informative exception
- Or provide migration utility

**Estimated effort:** 2-3 hours

---

### Phase 3: Update NumPyRandom

**Goal:** Integrate MT19937 and add Gaussian caching

**Files to modify:**
- `src/NumSharp.Core/RandomSampling/np.random.cs`

**Changes:**

1. Replace `Randomizer` with `MT19937`:
```csharp
public partial class NumPyRandom
{
    protected internal MT19937 bitGenerator;  // Was: Randomizer randomizer

    // Gaussian caching (required for state reproducibility)
    private bool _hasGauss;
    private double _gaussCache;
```

2. Update `NextGaussian()` with caching:
```csharp
protected internal double NextGaussian()
{
    if (_hasGauss)
    {
        _hasGauss = false;
        return _gaussCache;
    }

    // Box-Muller generates two values
    double u1, u2;
    do { u1 = bitGenerator.NextDouble(); } while (u1 == 0);
    u2 = bitGenerator.NextDouble();

    double r = Math.Sqrt(-2.0 * Math.Log(u1));
    double theta = 2.0 * Math.PI * u2;

    _gaussCache = r * Math.Sin(theta);
    _hasGauss = true;

    return r * Math.Cos(theta);
}
```

3. Update `get_state()` / `set_state()`:
```csharp
public NativeRandomState get_state()
{
    return new NativeRandomState
    {
        Algorithm = "MT19937",
        Key = (uint[])bitGenerator.Key.Clone(),
        Pos = bitGenerator.Pos,
        HasGauss = _hasGauss ? 1 : 0,
        CachedGaussian = _gaussCache
    };
}
```

**Estimated effort:** 2-3 hours

---

### Phase 4: Update Distribution Implementations

**Goal:** Verify all distributions use correct algorithms and produce NumPy-identical output

**Files to audit (40 files):**

| Distribution | File | Algorithm | Priority |
|-------------|------|-----------|----------|
| rand | np.random.rand.cs | Direct NextDouble | High |
| randn | np.random.randn.cs | NextGaussian | High |
| randint | np.random.randint.cs | Bounded integer | High |
| uniform | np.random.uniform.cs | Linear transform | High |
| normal | np.random.randn.cs | loc + scale * NextGaussian | High |
| choice | np.random.choice.cs | Index selection | High |
| permutation | np.random.permutation.cs | Fisher-Yates | High |
| shuffle | np.random.shuffle.cs | Fisher-Yates | High |
| beta | np.random.beta.cs | Gamma ratio | Medium |
| gamma | np.random.gamma.cs | Marsaglia | Medium |
| exponential | np.random.exponential.cs | -log(1-U) | Medium |
| poisson | np.random.poisson.cs | Multiple methods | Medium |
| binomial | np.random.binomial.cs | BTPE/Inversion | Medium |
| ... | ... | ... | Low |

**Key changes needed:**

1. **Replace `randomizer.NextDouble()` with `bitGenerator.NextDouble()`**
2. **Replace `randomizer.Next(n)` with proper bounded integer generation**
3. **Verify algorithm implementations match NumPy**

**NumPy's bounded integer algorithm:**
```csharp
// NumPy uses rejection sampling for unbiased integers
public int NextInt(int low, int high)
{
    uint range = (uint)(high - low);
    uint mask = NextPowerOf2(range) - 1;
    uint result;
    do {
        result = NextUInt32() & mask;
    } while (result >= range);
    return (int)result + low;
}
```

**Estimated effort:** 8-12 hours (including verification)

---

### Phase 5: Deprecate/Remove Randomizer

**Goal:** Clean up old implementation

**Files to modify/remove:**
- `src/NumSharp.Core/RandomSampling/Randomizer.cs` → **Delete** or mark `[Obsolete]`

**Breaking changes:**
- `Randomizer` class removed from public API
- State format incompatible with previous versions
- Same seed produces different sequences (intentional)

**Migration guide for users:**
```csharp
// Old (NumSharp < 0.42)
np.random.seed(42);
var x = np.random.rand();  // Returns 0.668...

// New (NumSharp >= 0.42)
np.random.seed(42);
var x = np.random.rand();  // Returns 0.374... (matches NumPy!)
```

**Estimated effort:** 1-2 hours

---

### Phase 6: Comprehensive Testing

**Goal:** Verify 100% NumPy compatibility

**Test categories:**

1. **Seed compatibility tests** (OpenBugs.Random.cs → regular tests)
   - All 15 existing tests should pass

2. **State round-trip tests**
   ```csharp
   [Test]
   public void GetSetState_Roundtrip()
   {
       np.random.seed(42);
       np.random.rand(100);
       var state = np.random.get_state();
       var x1 = np.random.rand();

       np.random.set_state(state);
       var x2 = np.random.rand();

       Assert.That(x1, Is.EqualTo(x2));
   }
   ```

3. **Cross-language verification**
   ```python
   # Generate reference values in Python
   import numpy as np
   np.random.seed(42)
   for _ in range(1000):
       print(np.random.rand())
   ```

4. **Statistical tests**
   - Mean/variance of large samples
   - Chi-squared uniformity test
   - Correlation tests

**Estimated effort:** 4-6 hours

---

## Implementation Order

```
Week 1:
├── Phase 1: MT19937 Core (4-6h)
├── Phase 2: State Serialization (2-3h)
└── Phase 3: NumPyRandom Integration (2-3h)

Week 2:
├── Phase 4: Distribution Updates (8-12h)
├── Phase 5: Cleanup (1-2h)
└── Phase 6: Testing (4-6h)
```

**Total estimated effort: 21-32 hours**

---

## Risk Mitigation

### Breaking Change Communication

1. **Version bump:** 0.41.x → 0.42.0 (minor version for breaking change)
2. **Release notes:** Clearly document the change
3. **Migration guide:** Provide examples

### Backward Compatibility Options

**Option A: Clean break (Recommended)**
- Remove old Randomizer entirely
- Document as intentional breaking change for NumPy alignment

**Option B: Parallel support**
- Keep Randomizer as `LegacyRandomizer`
- Add `np.random.use_legacy(true)` flag
- More maintenance burden

### Fallback Plan

If MT19937 implementation has issues:
1. Keep old Randomizer as fallback
2. Add feature flag to switch implementations
3. Fix issues incrementally

---

## Verification Checklist

### Phase 1 Complete When:
- [ ] `MT19937.Seed(42)` produces NumPy-identical uint32 sequence
- [ ] `MT19937.NextDouble()` matches NumPy's 53-bit conversion
- [ ] Unit tests pass for seed values: 0, 1, 42, 12345, 2^32-1

### Phase 2 Complete When:
- [ ] `get_state()` returns NumPy-compatible tuple format
- [ ] `set_state()` restores state correctly
- [ ] State round-trip produces identical sequences

### Phase 3 Complete When:
- [ ] `NextGaussian()` caching works correctly
- [ ] Gaussian cache included in state serialization
- [ ] All existing tests still pass

### Phase 4 Complete When:
- [ ] `rand()`, `randn()`, `randint()` match NumPy exactly
- [ ] `choice()`, `permutation()`, `shuffle()` match NumPy
- [ ] All 40 distribution files updated

### Phase 5 Complete When:
- [ ] Old Randomizer removed or deprecated
- [ ] No compilation warnings
- [ ] Documentation updated

### Phase 6 Complete When:
- [ ] All OpenBugs.Random tests pass (moved to regular tests)
- [ ] 1000-value sequences match NumPy exactly
- [ ] Statistical tests pass

---

## Files Changed Summary

| Action | File |
|--------|------|
| **Create** | `MT19937.cs` |
| **Modify** | `NativeRandomState.cs` |
| **Modify** | `np.random.cs` |
| **Modify** | `np.random.rand.cs` |
| **Modify** | `np.random.randn.cs` |
| **Modify** | `np.random.randint.cs` |
| **Modify** | `np.random.choice.cs` |
| **Modify** | `np.random.permutation.cs` |
| **Modify** | `np.random.shuffle.cs` |
| **Modify** | 30+ other distribution files |
| **Delete** | `Randomizer.cs` (or deprecate) |
| **Promote** | `OpenBugs.Random.cs` → regular tests |

---

## Success Criteria

The migration is complete when:

```csharp
// This produces IDENTICAL output to:
// >>> import numpy as np
// >>> np.random.seed(42)
// >>> np.random.rand(5)
// array([0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864])

np.random.seed(42);
var result = np.random.rand(5);
// result = [0.37454012, 0.95071431, 0.73199394, 0.59865848, 0.15601864]
```

And all 15 tests in `OpenBugs.Random.cs` pass.
