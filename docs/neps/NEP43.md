# NEP 43 - Enhancing the Extensibility of UFuncs

**Status:** Draft
**NumSharp Impact:** LOW - Future extensibility patterns

## Summary

Proposes `ArrayMethod` objects that encapsulate dtype-specific ufunc implementations, enabling user-defined dtypes to have custom ufunc behavior.

## Relationship to Other NEPs

Part of the dtype modernization series:
- **NEP 40:** Documents current dtype shortcomings
- **NEP 41:** Overview of replacement proposal
- **NEP 42:** New dtype APIs (see [NEP42.md](NEP42.md))
- **NEP 43:** Ufunc APIs for new dtypes (this NEP)

## Key Concept: ArrayMethod

Encapsulates dtype-specific ufunc functionality:

```python
class ArrayMethod:
    name: str
    casting: str = "no"

    def resolve_descriptors(self, DTypes, given_descrs):
        """Determine output dtype parameters"""
        # For parametric types like strings: S5 + S4 -> S9
        return (resolved_descrs, casting_safety)

    def strided_inner_loop(context, data, dims, strides):
        """The actual computation kernel"""
        pass
```

## Why This Matters

### Parametric Types Need Runtime Resolution

```python
# String concatenation: output length = input1 + input2
np.add(np.array(["abc"], dtype="S3"),
       np.array(["xy"], dtype="S2"))
# Result dtype must be S5, determined at runtime
```

### Custom DTypes Need Custom UFuncs

```python
# Unit type example
class UnitDType:
    def __init__(self, unit):
        self.unit = unit

# Adding meters + kilometers requires unit conversion
np.add(UnitDType("m")(1.0), UnitDType("km")(1.0))
```

## NumSharp Relevance

### Current Architecture

NumSharp's `TensorEngine` serves a similar purpose:

```csharp
public abstract class TensorEngine {
    public abstract NDArray Add(NDArray a, NDArray b);
    public abstract NDArray Multiply(NDArray a, NDArray b);
}
```

### Future Considerations

If NumSharp adds custom dtypes:

```csharp
// Similar to ArrayMethod concept
public interface ITypeOperation<T> {
    NPTypeCode ResolveOutputType(NPTypeCode[] inputs);
    void Execute(Span<T> a, Span<T> b, Span<T> output);
}

// Registration
TensorEngine.RegisterOperation<float>(
    OperationType.Add,
    new FloatAddOperation()
);
```

### Registration Pattern

```csharp
// Register custom operation for custom dtype
public static void RegisterUFunc(
    string ufuncName,
    NPTypeCode[] inputTypes,
    NPTypeCode outputType,
    Delegate implementation)
{
    _ufuncRegistry[(ufuncName, inputTypes)] = (outputType, implementation);
}
```

## Key Design Elements

### 1. Context Object

Passes dtype metadata to inner loops:

```csharp
public class UFuncContext {
    public NPTypeCode[] InputTypes { get; }
    public NPTypeCode OutputType { get; }
    public object[] DTypeMetadata { get; }  // e.g., string lengths, units
}
```

### 2. Promoter Functions

Handle type promotion when exact match not found:

```csharp
// Promote int32 to int64 for timedelta operations
public static NPTypeCode PromoteTimedeltaInteger(NPTypeCode[] inputs) {
    if (inputs[0] == NPTypeCode.TimeDelta && IsInteger(inputs[1])) {
        return NPTypeCode.Int64;
    }
    return NPTypeCode.NotDefined;
}
```

### 3. Existing Loop Wrapping

Reuse implementations with type adapters:

```csharp
// Wrap existing float64 implementation for unit types
public class UnitAddOperation : ITypeOperation<UnitFloat64> {
    private readonly FloatAddOperation _inner = new();

    public void Execute(Span<UnitFloat64> a, Span<UnitFloat64> b, Span<UnitFloat64> output) {
        // Convert units, call inner, wrap result
        var normalized = ConvertUnits(a, b);
        _inner.Execute(normalized.a, normalized.b, output.AsFloats());
        ApplyOutputUnit(output);
    }
}
```

## References

- [NEP 43 Full Text](https://numpy.org/neps/nep-0043-extensible-ufuncs.html)
- [NEP 42 - New DTypes](NEP42.md)
