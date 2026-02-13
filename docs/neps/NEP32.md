# NEP 32 - Remove Financial Functions

**Status:** Final
**NumSharp Impact:** LOW - NumSharp should NOT implement these functions

## Summary

Removed 10 financial functions from NumPy as they were too specialized and not part of NumPy's core mission.

## Removed Functions

| Function | Description |
|----------|-------------|
| `fv` | Future value |
| `ipmt` | Interest payment |
| `irr` | Internal rate of return |
| `mirr` | Modified internal rate of return |
| `nper` | Number of periods |
| `npv` | Net present value |
| `pmt` | Payment |
| `ppmt` | Principal payment |
| `pv` | Present value |
| `rate` | Rate of return |

## Timeline

- **NumPy 1.18:** Functions deprecated with warnings
- **NumPy 1.20:** Functions removed from NumPy

## Replacement

Functions moved to separate package: **`numpy-financial`**

```bash
pip install numpy-financial
```

```python
# Old
from numpy import npv, irr, pmt

# New
from numpy_financial import npv, irr, pmt
```

## NumSharp Implications

**DO NOT IMPLEMENT** these functions in NumSharp.

If users need financial functions, they should use a dedicated C# financial library instead of expecting NumSharp to provide them.

### Rationale

1. Outside NumSharp's scope (array operations, not domain-specific calculations)
2. Real financial calculations need proper date/calendar handling
3. Low usage in Python NumPy (only 8 GitHub repos found using them)
4. Maintenance burden without core team interest

## References

- [NEP 32 Full Text](https://numpy.org/neps/nep-0032-remove-financial-functions.html)
- [numpy-financial package](https://pypi.org/project/numpy-financial/)
