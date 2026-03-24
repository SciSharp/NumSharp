# NEP 07 - DateTime and Timedelta Types

**Status:** Final
**NumSharp Impact:** HIGH - NumSharp does NOT currently support datetime64/timedelta64

## Summary

Defines `datetime64` (absolute time) and `timedelta64` (relative time) types for NumPy arrays.

## Type Definitions

### datetime64
- 64-bit signed integer
- Measures units from POSIX epoch (January 1, 1970, 12:00 AM)
- NaT (Not a Time) = `-2^63`

### timedelta64
- 64-bit signed integer
- Represents duration/interval
- NaT = `-2^63`

## Time Units

| Code | Meaning | Range |
|------|---------|-------|
| Y | year | ± 9.2e18 years |
| M | month | ± 7.6e17 years |
| W | week | ± 1.7e17 years |
| D | day | ± 2.5e16 years |
| h | hour | ± 1.0e15 years |
| m | minute | ± 1.7e13 years |
| s | second | ± 2.9e12 years |
| ms | millisecond | ± 2.9e9 years |
| us | microsecond (default) | ± 2.9e6 years |
| ns | nanosecond | ± 292 years |
| ps | picosecond | ± 106 days |
| fs | femtosecond | ± 2.6 hours |
| as | attosecond | ± 9.2 seconds |

## Operations

### datetime64 vs datetime64
- **Subtraction:** Returns timedelta64
- **Comparison:** ==, !=, <, >, <=, >=
- **NOT allowed:** Addition, multiplication, division

### datetime64 vs timedelta64
- **Addition/Subtraction:** Returns datetime64
- **NOT allowed:** Multiplication, division

### timedelta64 vs timedelta64
- **All arithmetic:** +, -, *, /
- **Result unit:** Shorter (more precise) of the two units

## String Notation

```python
dtype('datetime64[us]')   # Long form
dtype('M8[us]')           # Short form (M8 = datetime64)
dtype('timedelta64[ms]')  # Long form
dtype('m8[ms]')           # Short form (m8 = timedelta64)

# Multiples
dtype('M8[100ns]')        # 100 nanoseconds
dtype('M8[3M]')           # 3 months
```

## NumSharp Implementation Status

**NOT IMPLEMENTED** - datetime64 and timedelta64 are not supported.

### Implementation Requirements

1. Add `NPTypeCode.DateTime64` and `NPTypeCode.TimeDelta64`
2. Store as Int64 internally with unit metadata
3. Implement unit conversion
4. Support ISO 8601 string parsing
5. Handle NaT values
6. Implement arithmetic operations with unit rules

### Potential C# Mapping

```csharp
// datetime64 could map to:
DateTimeOffset  // for absolute time with timezone
DateTime        // for naive datetime

// timedelta64 could map to:
TimeSpan        // for durations
```

## References

- [NEP 7 Full Text](https://numpy.org/neps/nep-0007-datetime-proposal.html)
- [NumPy datetime64 docs](https://numpy.org/doc/stable/reference/arrays.datetime.html)
