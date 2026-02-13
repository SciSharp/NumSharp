# NEP 42 - New and Extensible DTypes

**Status:** Accepted (Implementation in Progress)
**NumSharp Impact:** INFORMATIONAL - Informs future dtype architecture

## Summary

Proposes a modular, class-based architecture for NumPy dtypes, enabling user-defined custom dtypes with full functionality.

## Key Changes

### From Monolithic to Modular
- Each dtype becomes an instance of a DType subclass
- `np.dtype("float64")` returns instance of `Float64` class
- User-extensible through subclassing

### DType Hierarchy

```
numpy.dtype (base)
├── Abstract dtypes (cannot be instantiated)
│   ├── Integer
│   ├── Floating
│   ├── Complex
│   └── User abstract (e.g., Unit, Categorical)
│
└── Concrete dtypes (can be instantiated, cannot be subclassed)
    ├── Float64
    ├── Int32
    ├── String
    └── User concrete (e.g., Float64Unit)
```

### Class Getter Syntax
```python
np.dtype[np.int64]      # Get DType class for int64
np.dtype[UserScalar]    # Works with user-defined scalars
```

## User-Defined DType API

### Basic Structure
```python
class DType(np.dtype):
    type : type        # Python scalar type
    parametric : bool  # Whether dtype has parameters

    @property
    def canonical(self) -> bool: ...

    def ensure_canonical(self) -> DType: ...
```

### Casting Methods
```python
@classmethod
def common_dtype(cls, other) -> DTypeMeta: ...

def common_instance(self, other) -> DType: ...
```

## NumSharp Relevance

### Current Architecture
NumSharp uses `NPTypeCode` enum for dtype identification:
```csharp
enum NPTypeCode {
    Boolean, Byte, Int16, UInt16, Int32, UInt32,
    Int64, UInt64, Char, Single, Double, Decimal
}
```

### Potential Future Alignment

If NumSharp wanted to match NumPy's new dtype system:

1. **DType Classes:** Create class hierarchy for dtypes
   ```csharp
   abstract class DType { }
   class Float64 : DType { }
   class Int32 : DType { }
   ```

2. **Type Resolution:** Implement `common_dtype` for type promotion
   ```csharp
   DType CommonDType(DType other);
   ```

3. **Extensibility:** Allow user-defined dtypes
   ```csharp
   class QuantityDType : DType {
       public Unit Unit { get; }
   }
   ```

### Practical Implications

For now, NumSharp can continue with `NPTypeCode` enum. The NEP 42 architecture is primarily useful for:
- Understanding NumPy's internal evolution
- Planning future extensibility (e.g., custom dtypes for units, categoricals)
- Ensuring type promotion logic matches NumPy

## References

- [NEP 42 Full Text](https://numpy.org/neps/nep-0042-new-dtypes.html)
- [NEP 40 - Legacy dtypes](https://numpy.org/neps/nep-0040-legacy-datatype-impl.html)
- [NEP 41 - First step towards new dtypes](https://numpy.org/neps/nep-0041-improved-dtype.html)
