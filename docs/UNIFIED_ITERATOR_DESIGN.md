# NDIterator Design (v4)

## Design Principles

1. **No backwards compatibility** - All existing iterators/incrementors will be deleted
2. **Direct IL control** - Users can inject their own IL generation
3. **Zero allocation** - Struct-based state, no closures
4. **Three tiers** - Interface kernels (fast), IL injection (full control), Func delegates (simple)

---

## Architecture Overview

```
+---------------------------------------------------------------------+
|                         NDIterator                                  |
|  +----------------+  +----------------+  +------------------------+ |
|  | IteratorState  |  | LayoutDetector |  | KernelInjectionSystem  | |
|  |   (struct)     |  |    (static)    |  |                        | |
|  +----------------+  +----------------+  |  +--------------------+ | |
|                                          |  | Tier 1: IKernel    | | |
|  Iteration Modes:                        |  | (static abstract)  | | |
|  +- Contiguous (SIMD)                    |  +--------------------+ | |
|  +- Strided (1D)                         |  | Tier 2: ILEmit     | | |
|  +- General (N-D)                        |  | (raw IL inject)    | | |
|  +- Axis (reduction/cumulative)          |  +--------------------+ | |
|  +- Broadcast (paired)                   |  | Tier 3: Func<>     | | |
|                                          |  | (delegate)         | | |
|                                          |  +--------------------+ | |
+---------------------------------------------------------------------+
```

---

## Kernel Interfaces (Complete)

### Tier 1: Static Abstract Interfaces (JIT-Inlinable)

```csharp
// =============================================================================
// UNARY: TIn -> TOut
// =============================================================================

/// <summary>
/// Unary kernel with static abstract for JIT inlining.
/// The Apply method should be simple enough for the JIT to inline.
/// </summary>
public interface IUnaryKernel<TIn, TOut>
    where TIn : unmanaged
    where TOut : unmanaged
{
    /// <summary>Transform a single element.</summary>
    static abstract TOut Apply(TIn value);

    /// <summary>
    /// Optional: Provide SIMD implementation.
    /// Return number of elements processed, or 0 to use scalar fallback.
    /// </summary>
    static virtual int ApplyVector(ReadOnlySpan<TIn> input, Span<TOut> output) => 0;
}

// =============================================================================
// BINARY: (TLeft, TRight) -> TOut
// =============================================================================

/// <summary>Binary kernel for element-wise operations.</summary>
public interface IBinaryKernel<TLeft, TRight, TOut>
    where TLeft : unmanaged
    where TRight : unmanaged
    where TOut : unmanaged
{
    static abstract TOut Apply(TLeft left, TRight right);

    static virtual int ApplyVector(
        ReadOnlySpan<TLeft> left,
        ReadOnlySpan<TRight> right,
        Span<TOut> output) => 0;
}

// =============================================================================
// REDUCTION: (TAccum, TIn) -> TAccum
// =============================================================================

/// <summary>Reduction kernel with early-exit support.</summary>
public interface IReductionKernel<TIn, TAccum>
    where TIn : unmanaged
    where TAccum : unmanaged
{
    /// <summary>Identity value (0 for sum, 1 for prod, etc.).</summary>
    static abstract TAccum Identity { get; }

    /// <summary>Combine accumulator with next value.</summary>
    static abstract TAccum Combine(TAccum accumulator, TIn value);

    /// <summary>
    /// Return false to exit reduction early.
    /// Default: always continue (no early exit).
    /// Used by All (exit on false) and Any (exit on true).
    /// </summary>
    static virtual bool ShouldContinue(TAccum accumulator) => true;

    /// <summary>
    /// Optional: SIMD reduction over span.
    /// Default implementation uses scalar Combine with early-exit check.
    /// </summary>
    static virtual TAccum CombineVector(TAccum accumulator, ReadOnlySpan<TIn> values)
    {
        foreach (var v in values)
        {
            accumulator = Combine(accumulator, v);
            if (!ShouldContinue(accumulator))
                break;
        }
        return accumulator;
    }
}

// =============================================================================
// INDEXED REDUCTION: (TAccum, TIn, index) -> TAccum
// =============================================================================

/// <summary>
/// Indexed reduction for ArgMax/ArgMin where index tracking is required.
/// </summary>
public interface IIndexedReductionKernel<TIn, TAccum>
    where TIn : unmanaged
    where TAccum : unmanaged
{
    static abstract TAccum Identity { get; }
    static abstract TAccum Combine(TAccum accumulator, TIn value, int index);
    static virtual bool ShouldContinue(TAccum accumulator) => true;
}

// =============================================================================
// AXIS: Process entire axis slice (cumsum, cumprod, etc.)
// =============================================================================

/// <summary>
/// Kernel for axis-wise operations with stride support.
/// Handles non-contiguous axis slices via pointer+stride.
/// </summary>
public interface IAxisKernel<TIn, TOut>
    where TIn : unmanaged
    where TOut : unmanaged
{
    /// <summary>
    /// Process an axis slice. Input/output may be non-contiguous.
    /// </summary>
    /// <param name="input">Pointer to first element of input axis</param>
    /// <param name="output">Pointer to first element of output axis</param>
    /// <param name="inputStride">Stride between input elements (in elements, not bytes)</param>
    /// <param name="outputStride">Stride between output elements</param>
    /// <param name="length">Number of elements along axis</param>
    static abstract unsafe void ProcessAxis(
        TIn* input,
        TOut* output,
        int inputStride,
        int outputStride,
        int length);
}

// =============================================================================
// TERNARY: (bool, T, T) -> T (np.where)
// =============================================================================

/// <summary>Ternary select kernel for np.where-style operations.</summary>
public interface ITernaryKernel<T>
    where T : unmanaged
{
    static abstract T Apply(bool condition, T ifTrue, T ifFalse);

    static virtual int ApplyVector(
        ReadOnlySpan<bool> condition,
        ReadOnlySpan<T> ifTrue,
        ReadOnlySpan<T> ifFalse,
        Span<T> output) => 0;
}

// =============================================================================
// PREDICATE: T -> bool (masking)
// =============================================================================

/// <summary>Predicate kernel for creating boolean masks.</summary>
public interface IPredicateKernel<T>
    where T : unmanaged
{
    static abstract bool Apply(T value);
    static virtual int ApplyVector(ReadOnlySpan<T> input, Span<bool> output) => 0;
}
```

---

### Tier 1: Example Implementations

```csharp
// ============ UNARY ============

public readonly struct SquareKernel : IUnaryKernel<double, double>
{
    public static double Apply(double value) => value * value;

    public static int ApplyVector(ReadOnlySpan<double> input, Span<double> output)
    {
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            for (; i <= input.Length - Vector256<double>.Count; i += Vector256<double>.Count)
            {
                var v = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input), (nuint)i);
                (v * v).StoreUnsafe(ref MemoryMarshal.GetReference(output), (nuint)i);
            }
        }
        return i; // Return how many we processed; iterator handles remainder
    }
}

public readonly struct NegateKernel<T> : IUnaryKernel<T, T>
    where T : unmanaged, IUnaryNegationOperators<T, T>
{
    public static T Apply(T value) => -value;
}

public readonly struct AbsKernel : IUnaryKernel<double, double>
{
    public static double Apply(double value) => Math.Abs(value);
}

// ============ BINARY ============

public readonly struct AddKernel<T> : IBinaryKernel<T, T, T>
    where T : unmanaged, IAdditionOperators<T, T, T>
{
    public static T Apply(T left, T right) => left + right;
}

public readonly struct SubtractKernel<T> : IBinaryKernel<T, T, T>
    where T : unmanaged, ISubtractionOperators<T, T, T>
{
    public static T Apply(T left, T right) => left - right;
}

public readonly struct MultiplyKernel<T> : IBinaryKernel<T, T, T>
    where T : unmanaged, IMultiplyOperators<T, T, T>
{
    public static T Apply(T left, T right) => left * right;
}

public readonly struct MaxKernel<T> : IBinaryKernel<T, T, T>
    where T : unmanaged, IComparisonOperators<T, T, bool>
{
    public static T Apply(T left, T right) => left > right ? left : right;
}

// ============ REDUCTION ============

public readonly struct SumKernel<T> : IReductionKernel<T, T>
    where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
{
    public static T Identity => T.AdditiveIdentity;
    public static T Combine(T acc, T val) => acc + val;
}

public readonly struct ProdKernel<T> : IReductionKernel<T, T>
    where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
{
    public static T Identity => T.MultiplicativeIdentity;
    public static T Combine(T acc, T val) => acc * val;
}

public readonly struct AllKernel : IReductionKernel<bool, bool>
{
    public static bool Identity => true;
    public static bool Combine(bool acc, bool val) => acc && val;
    public static bool ShouldContinue(bool acc) => acc; // Exit when false
}

public readonly struct AnyKernel : IReductionKernel<bool, bool>
{
    public static bool Identity => false;
    public static bool Combine(bool acc, bool val) => acc || val;
    public static bool ShouldContinue(bool acc) => !acc; // Exit when true
}

// ============ INDEXED REDUCTION ============

public readonly struct ArgMaxKernel : IIndexedReductionKernel<double, (double Value, int Index)>
{
    public static (double, int) Identity => (double.NegativeInfinity, -1);

    public static (double, int) Combine((double Value, int Index) acc, double value, int index)
        => value > acc.Value ? (value, index) : acc;
}

public readonly struct ArgMinKernel : IIndexedReductionKernel<double, (double Value, int Index)>
{
    public static (double, int) Identity => (double.PositiveInfinity, -1);

    public static (double, int) Combine((double Value, int Index) acc, double value, int index)
        => value < acc.Value ? (value, index) : acc;
}

// ============ AXIS ============

public readonly struct CumSumAxisKernel<T> : IAxisKernel<T, T>
    where T : unmanaged, IAdditionOperators<T, T, T>, IAdditiveIdentity<T, T>
{
    public static unsafe void ProcessAxis(
        T* input, T* output,
        int inputStride, int outputStride, int length)
    {
        T sum = T.AdditiveIdentity;
        for (int i = 0; i < length; i++)
        {
            sum += input[i * inputStride];
            output[i * outputStride] = sum;
        }
    }
}

public readonly struct CumProdAxisKernel<T> : IAxisKernel<T, T>
    where T : unmanaged, IMultiplyOperators<T, T, T>, IMultiplicativeIdentity<T, T>
{
    public static unsafe void ProcessAxis(
        T* input, T* output,
        int inputStride, int outputStride, int length)
    {
        T prod = T.MultiplicativeIdentity;
        for (int i = 0; i < length; i++)
        {
            prod *= input[i * inputStride];
            output[i * outputStride] = prod;
        }
    }
}

// ============ TERNARY ============

public readonly struct SelectKernel<T> : ITernaryKernel<T>
    where T : unmanaged
{
    public static T Apply(bool condition, T ifTrue, T ifFalse)
        => condition ? ifTrue : ifFalse;
}

// ============ PREDICATE ============

public readonly struct IsPositiveKernel<T> : IPredicateKernel<T>
    where T : unmanaged, IComparisonOperators<T, T, bool>, IAdditiveIdentity<T, T>
{
    public static bool Apply(T value) => value > T.AdditiveIdentity;
}

public readonly struct IsNaNKernel : IPredicateKernel<double>
{
    public static bool Apply(double value) => double.IsNaN(value);
}

public readonly struct IsFiniteKernel : IPredicateKernel<double>
{
    public static bool Apply(double value) => double.IsFinite(value);
}
```

---

### Tier 2: Direct IL Injection (Full Control)

For users who need complete control over the generated IL.

```csharp
/// <summary>
/// IL emitter interface for direct kernel code generation.
/// Implementers have full control over the generated IL.
/// </summary>
public interface IKernelEmitter
{
    /// <summary>
    /// Emit IL for the kernel operation.
    /// Stack state on entry depends on kernel kind (see Stack Contract).
    /// Must leave result on stack.
    /// </summary>
    void Emit(ILGenerator il, KernelEmitContext context);
}

/// <summary>Context provided to IL emitters.</summary>
public readonly struct KernelEmitContext
{
    public NPTypeCode InputType { get; init; }
    public NPTypeCode OutputType { get; init; }
    public KernelKind Kind { get; init; }

    // Locals that the iterator has already declared (for reuse)
    public LocalBuilder? LocalTemp1 { get; init; }
    public LocalBuilder? LocalTemp2 { get; init; }

    // Labels for control flow (e.g., early exit in reductions)
    public Label? EarlyExitLabel { get; init; }
}

public enum KernelKind
{
    Unary,           // T -> TOut
    Binary,          // (TLeft, TRight) -> TOut
    Comparison,      // (TLeft, TRight) -> bool
    Reduction,       // (TAccum, TIn) -> TAccum
    IndexedReduction,// (TAccum, TIn, int) -> TAccum
    Axis,            // Process entire axis slice
    Ternary,         // (bool, T, T) -> T
    Predicate,       // T -> bool
}

/// <summary>
/// Delegate-based IL emitter for inline definition.
/// </summary>
public delegate void KernelEmitDelegate(ILGenerator il, KernelEmitContext context);
```

**Stack Contract for IL Emitters:**

| Kernel Kind | Stack on Entry | Stack on Exit |
|-------------|----------------|---------------|
| Unary | `[value]` | `[result]` |
| Binary | `[left, right]` | `[result]` |
| Comparison | `[left, right]` | `[bool]` |
| Reduction | `[accumulator, value]` | `[new_accumulator]` |
| IndexedReduction | `[accumulator, value, index]` | `[new_accumulator]` |
| Ternary | `[condition, ifTrue, ifFalse]` | `[result]` |
| Predicate | `[value]` | `[bool]` |

---

### Tier 2: IKernelEmitter Examples

```csharp
// =============================================================================
// EXAMPLE 1: Power operation (Math.Pow)
// =============================================================================

public class PowerEmitter : IKernelEmitter
{
    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack on entry: [base, exponent] (both as input type)
        // Need to call Math.Pow(double, double) -> double

        // Store exponent to local
        var locExp = il.DeclareLocal(typeof(double));
        EmitConvertToDouble(il, context.InputType);
        il.Emit(OpCodes.Stloc, locExp);

        // Convert base to double (it's now on top)
        EmitConvertToDouble(il, context.InputType);

        // Load exponent
        il.Emit(OpCodes.Ldloc, locExp);

        // Call Math.Pow
        var powMethod = typeof(Math).GetMethod(nameof(Math.Pow),
            new[] { typeof(double), typeof(double) });
        il.EmitCall(OpCodes.Call, powMethod!, null);

        // Convert result back to output type
        EmitConvertFromDouble(il, context.OutputType);
        // Stack on exit: [result]
    }

    private static void EmitConvertToDouble(ILGenerator il, NPTypeCode type)
    {
        if (type == NPTypeCode.Double) return;
        if (type == NPTypeCode.Single) { il.Emit(OpCodes.Conv_R8); return; }
        if (type == NPTypeCode.Byte || type == NPTypeCode.UInt16 ||
            type == NPTypeCode.UInt32 || type == NPTypeCode.UInt64)
            il.Emit(OpCodes.Conv_R_Un);
        il.Emit(OpCodes.Conv_R8);
    }

    private static void EmitConvertFromDouble(ILGenerator il, NPTypeCode type)
    {
        switch (type)
        {
            case NPTypeCode.Double: break;
            case NPTypeCode.Single: il.Emit(OpCodes.Conv_R4); break;
            case NPTypeCode.Byte: il.Emit(OpCodes.Conv_U1); break;
            case NPTypeCode.Int16: il.Emit(OpCodes.Conv_I2); break;
            case NPTypeCode.Int32: il.Emit(OpCodes.Conv_I4); break;
            case NPTypeCode.Int64: il.Emit(OpCodes.Conv_I8); break;
            // ... etc
        }
    }
}

// Usage:
NDIterator.TransformBinary<double, double, double>(
    bases, exponents, result, new PowerEmitter());

// =============================================================================
// EXAMPLE 2: Fused Multiply-Add (a * b + c) as Binary on pre-added arrays
// =============================================================================

public class FusedMultiplyAddEmitter : IKernelEmitter
{
    private readonly double _addend;

    public FusedMultiplyAddEmitter(double addend) => _addend = addend;

    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [left, right]
        il.Emit(OpCodes.Mul);           // [left * right]
        il.Emit(OpCodes.Ldc_R8, _addend); // [left * right, addend]
        il.Emit(OpCodes.Add);           // [left * right + addend]
    }
}

// =============================================================================
// EXAMPLE 3: Clip (clamp to range)
// =============================================================================

public class ClipEmitter : IKernelEmitter
{
    private readonly double _min;
    private readonly double _max;

    public ClipEmitter(double min, double max)
    {
        _min = min;
        _max = max;
    }

    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [value]
        var lblCheckMax = il.DefineLabel();
        var lblEnd = il.DefineLabel();

        // if (value < min) return min
        il.Emit(OpCodes.Dup);                    // [value, value]
        il.Emit(OpCodes.Ldc_R8, _min);           // [value, value, min]
        il.Emit(OpCodes.Bge_Un, lblCheckMax);    // [value] (branch if value >= min)
        il.Emit(OpCodes.Pop);                    // []
        il.Emit(OpCodes.Ldc_R8, _min);           // [min]
        il.Emit(OpCodes.Br, lblEnd);

        // if (value > max) return max
        il.MarkLabel(lblCheckMax);
        il.Emit(OpCodes.Dup);                    // [value, value]
        il.Emit(OpCodes.Ldc_R8, _max);           // [value, value, max]
        il.Emit(OpCodes.Ble_Un, lblEnd);         // [value] (branch if value <= max)
        il.Emit(OpCodes.Pop);                    // []
        il.Emit(OpCodes.Ldc_R8, _max);           // [max]

        il.MarkLabel(lblEnd);
        // Stack: [clipped_value]
    }
}

// Usage:
NDIterator.Transform<double, double>(arr, result, new ClipEmitter(0.0, 1.0));

// =============================================================================
// EXAMPLE 4: Sigmoid activation function: 1 / (1 + exp(-x))
// =============================================================================

public class SigmoidEmitter : IKernelEmitter
{
    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [x]
        il.Emit(OpCodes.Neg);                    // [-x]

        var expMethod = typeof(Math).GetMethod(nameof(Math.Exp), new[] { typeof(double) });
        il.EmitCall(OpCodes.Call, expMethod!, null);  // [exp(-x)]

        il.Emit(OpCodes.Ldc_R8, 1.0);            // [exp(-x), 1.0]
        il.Emit(OpCodes.Add);                    // [1 + exp(-x)]

        il.Emit(OpCodes.Ldc_R8, 1.0);            // [1 + exp(-x), 1.0]
        il.Emit(OpCodes.Div);                    // [1 / (1 + exp(-x))]
        // Stack: [sigmoid(x)]
    }
}

// =============================================================================
// EXAMPLE 5: ReLU with inline delegate
// =============================================================================

var reluEmitter = new KernelEmitDelegate((il, ctx) =>
{
    // Stack: [x]
    il.Emit(OpCodes.Dup);                    // [x, x]
    il.Emit(OpCodes.Ldc_R8, 0.0);            // [x, x, 0]
    var lblPositive = il.DefineLabel();
    il.Emit(OpCodes.Bge, lblPositive);       // [x] (branch if x >= 0)
    il.Emit(OpCodes.Pop);                    // []
    il.Emit(OpCodes.Ldc_R8, 0.0);            // [0]
    il.MarkLabel(lblPositive);
    // Stack: [max(0, x)]
});

NDIterator.Transform<double, double>(arr, result, reluEmitter);

// =============================================================================
// EXAMPLE 6: Leaky ReLU with configurable alpha
// =============================================================================

public class LeakyReLUEmitter : IKernelEmitter
{
    private readonly double _alpha;

    public LeakyReLUEmitter(double alpha = 0.01) => _alpha = alpha;

    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [x]
        // return x >= 0 ? x : alpha * x

        var lblNegative = il.DefineLabel();
        var lblEnd = il.DefineLabel();

        il.Emit(OpCodes.Dup);                    // [x, x]
        il.Emit(OpCodes.Ldc_R8, 0.0);            // [x, x, 0]
        il.Emit(OpCodes.Blt, lblNegative);       // [x] (branch if x < 0)
        il.Emit(OpCodes.Br, lblEnd);             // x >= 0, keep x

        il.MarkLabel(lblNegative);
        il.Emit(OpCodes.Ldc_R8, _alpha);         // [x, alpha]
        il.Emit(OpCodes.Mul);                    // [alpha * x]

        il.MarkLabel(lblEnd);
        // Stack: [result]
    }
}

// =============================================================================
// EXAMPLE 7: Softplus: log(1 + exp(x))
// =============================================================================

public class SoftplusEmitter : IKernelEmitter
{
    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [x]
        var expMethod = typeof(Math).GetMethod(nameof(Math.Exp), new[] { typeof(double) });
        var logMethod = typeof(Math).GetMethod(nameof(Math.Log), new[] { typeof(double) });

        il.EmitCall(OpCodes.Call, expMethod!, null);  // [exp(x)]
        il.Emit(OpCodes.Ldc_R8, 1.0);                 // [exp(x), 1]
        il.Emit(OpCodes.Add);                         // [1 + exp(x)]
        il.EmitCall(OpCodes.Call, logMethod!, null);  // [log(1 + exp(x))]
    }
}

// =============================================================================
// EXAMPLE 8: Custom reduction - LogSumExp (numerically stable)
// =============================================================================

public class LogSumExpReductionEmitter : IKernelEmitter
{
    private readonly double _maxValue; // Pre-computed max for stability

    public LogSumExpReductionEmitter(double maxValue) => _maxValue = maxValue;

    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [accumulator, value]
        // Compute: acc + exp(value - max)

        var expMethod = typeof(Math).GetMethod(nameof(Math.Exp), new[] { typeof(double) });

        // Store accumulator
        var locAcc = il.DeclareLocal(typeof(double));
        il.Emit(OpCodes.Stloc, locAcc);      // Stack: [value]

        // Compute exp(value - max)
        il.Emit(OpCodes.Ldc_R8, _maxValue);  // [value, max]
        il.Emit(OpCodes.Sub);                // [value - max]
        il.EmitCall(OpCodes.Call, expMethod!, null); // [exp(value - max)]

        // Add to accumulator
        il.Emit(OpCodes.Ldloc, locAcc);      // [exp(value - max), acc]
        il.Emit(OpCodes.Add);                // [acc + exp(value - max)]
    }
}

// =============================================================================
// EXAMPLE 9: Euclidean distance accumulation (for norm)
// =============================================================================

public class SumSquaresEmitter : IKernelEmitter
{
    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [accumulator, value]
        var locAcc = il.DeclareLocal(typeof(double));
        il.Emit(OpCodes.Stloc, locAcc);      // [value]

        il.Emit(OpCodes.Dup);                // [value, value]
        il.Emit(OpCodes.Mul);                // [value^2]
        il.Emit(OpCodes.Ldloc, locAcc);      // [value^2, acc]
        il.Emit(OpCodes.Add);                // [acc + value^2]
    }
}

// Usage: Compute L2 norm
double sumSq = NDIterator.Reduce<double, double>(arr, new SumSquaresEmitter(), 0.0);
double norm = Math.Sqrt(sumSq);

// =============================================================================
// EXAMPLE 10: Polynomial evaluation (Horner's method for ax^2 + bx + c)
// =============================================================================

public class QuadraticEmitter : IKernelEmitter
{
    private readonly double _a, _b, _c;

    public QuadraticEmitter(double a, double b, double c)
    {
        _a = a; _b = b; _c = c;
    }

    public void Emit(ILGenerator il, KernelEmitContext context)
    {
        // Stack: [x]
        // Horner: ((a * x) + b) * x + c

        il.Emit(OpCodes.Dup);                // [x, x]
        il.Emit(OpCodes.Ldc_R8, _a);         // [x, x, a]
        il.Emit(OpCodes.Mul);                // [x, a*x]
        il.Emit(OpCodes.Ldc_R8, _b);         // [x, a*x, b]
        il.Emit(OpCodes.Add);                // [x, a*x + b]
        il.Emit(OpCodes.Mul);                // [(a*x + b) * x]
        il.Emit(OpCodes.Ldc_R8, _c);         // [(a*x + b) * x, c]
        il.Emit(OpCodes.Add);                // [a*x^2 + b*x + c]
    }
}
```

---

### Tier 3: Func/Action Delegates (Simple)

For quick prototyping and cold paths. Has delegate invocation overhead (~10-15 cycles per element).

```csharp
// Usage examples with Func<> delegates

// Unary transform
NDIterator.Transform<double, double>(arr, result, x => x * x);
NDIterator.Transform<double, double>(arr, result, Math.Sin);
NDIterator.Transform<int, double>(arr, result, x => Math.Sqrt(x));

// Binary transform
NDIterator.TransformBinary<double, double, double>(a, b, result, (x, y) => x + y);
NDIterator.TransformBinary<double, double, double>(a, b, result, Math.Max);

// Reduction
double sum = NDIterator.Reduce<double, double>(arr, (acc, x) => acc + x, 0.0);
double prod = NDIterator.Reduce<double, double>(arr, (acc, x) => acc * x, 1.0);
double sumSq = NDIterator.Reduce<double, double>(arr, (acc, x) => acc + x * x, 0.0);

// Indexed reduction
var (maxVal, maxIdx) = NDIterator.ReduceIndexed<double, (double, int)>(
    arr,
    (acc, val, idx) => val > acc.Item1 ? (val, idx) : acc,
    (double.NegativeInfinity, -1));

// Axis reduction
NDArray rowSums = NDIterator.ReduceAxis<double, double>(
    arr, axis: 1, (acc, x) => acc + x, 0.0);

// Masking
NDArray mask = NDIterator.Mask<double>(arr, x => x > 0.5);
NDArray nanMask = NDIterator.Mask<double>(arr, double.IsNaN);

// np.where
NDIterator.Where<double>(condition, ifTrue, ifFalse, dest,
    (c, t, f) => c ? t : f);
```

---

## Iterator State (Unified)

```csharp
/// <summary>
/// Unified iterator state. Stack-allocated, no managed references.
/// Replaces all existing incrementors and iterator closures.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct IteratorState
{
    // Core pointers
    public void* InputAddress;
    public void* OutputAddress;

    // Position tracking
    public int LinearIndex;      // Current flat index (0..Size-1)
    public int Size;             // Total elements

    // Shape info (inline for common case <= 16 dims)
    public fixed int Shape[16];
    public fixed int InputStrides[16];
    public fixed int OutputStrides[16];
    public fixed int Coords[16];  // Current N-D coordinates
    public int NDim;

    // Overflow pointers for >16 dims (rare, managed by NDArray)
    public int* ShapeOverflow;
    public int* InputStridesOverflow;
    public int* OutputStridesOverflow;
    public int* CoordsOverflow;

    // Axis iteration
    public int Axis;
    public int AxisSize;
    public int AxisStride;

    // Behavior flags
    public IteratorFlags Flags;

    // Type info for IL generation
    public NPTypeCode InputType;
    public NPTypeCode OutputType;

    // Accessors
    public readonly Span<int> GetShape() => NDim <= 16
        ? new Span<int>(Unsafe.AsPointer(ref Unsafe.AsRef(in Shape[0])), NDim)
        : new Span<int>(ShapeOverflow, NDim);

    public readonly Span<int> GetInputStrides() => NDim <= 16
        ? new Span<int>(Unsafe.AsPointer(ref Unsafe.AsRef(in InputStrides[0])), NDim)
        : new Span<int>(InputStridesOverflow, NDim);

    public readonly Span<int> GetOutputStrides() => NDim <= 16
        ? new Span<int>(Unsafe.AsPointer(ref Unsafe.AsRef(in OutputStrides[0])), NDim)
        : new Span<int>(OutputStridesOverflow, NDim);

    public readonly Span<int> GetCoords() => NDim <= 16
        ? new Span<int>(Unsafe.AsPointer(ref Unsafe.AsRef(in Coords[0])), NDim)
        : new Span<int>(CoordsOverflow, NDim);
}

[Flags]
public enum IteratorFlags : ushort
{
    None = 0,

    // Layout flags
    InputContiguous = 1 << 0,
    OutputContiguous = 1 << 1,
    BothContiguous = InputContiguous | OutputContiguous,

    // Behavior flags
    AutoReset = 1 << 2,         // Cycle back to start (for broadcasting)
    Broadcast = 1 << 3,         // Handle stride=0 dimensions

    // Iteration mode
    AxisMode = 1 << 4,          // Iterating along axis
    PairedMode = 1 << 5,        // Two-input iteration (binary ops)

    // SIMD hints
    SimdEligible = 1 << 6,      // Inner dim is SIMD-friendly
}
```

---

## NDIterator API (Complete)

```csharp
public static class NDIterator
{
    // =========================================================================
    // TIER 1: Static Interface Kernels (Zero overhead, SIMD support)
    // =========================================================================

    /// <summary>Apply unary kernel to all elements.</summary>
    public static void Transform<TIn, TOut, TKernel>(NDArray source, NDArray dest)
        where TIn : unmanaged
        where TOut : unmanaged
        where TKernel : struct, IUnaryKernel<TIn, TOut>;

    /// <summary>Apply binary kernel element-wise.</summary>
    public static void TransformBinary<TL, TR, TOut, TKernel>(
        NDArray left, NDArray right, NDArray dest)
        where TL : unmanaged
        where TR : unmanaged
        where TOut : unmanaged
        where TKernel : struct, IBinaryKernel<TL, TR, TOut>;

    /// <summary>Reduce all elements.</summary>
    public static TAccum Reduce<TIn, TAccum, TKernel>(NDArray source)
        where TIn : unmanaged
        where TAccum : unmanaged
        where TKernel : struct, IReductionKernel<TIn, TAccum>;

    /// <summary>Indexed reduction (ArgMax, ArgMin).</summary>
    public static TAccum ReduceIndexed<TIn, TAccum, TKernel>(NDArray source)
        where TIn : unmanaged
        where TAccum : unmanaged
        where TKernel : struct, IIndexedReductionKernel<TIn, TAccum>;

    /// <summary>Reduce along axis.</summary>
    public static NDArray ReduceAxis<TIn, TAccum, TKernel>(NDArray source, int axis)
        where TIn : unmanaged
        where TAccum : unmanaged
        where TKernel : struct, IReductionKernel<TIn, TAccum>;

    /// <summary>Cumulative axis operation (cumsum, cumprod).</summary>
    public static void IterateAxisTransform<TIn, TOut, TKernel>(
        NDArray source, NDArray dest, int axis)
        where TIn : unmanaged
        where TOut : unmanaged
        where TKernel : struct, IAxisKernel<TIn, TOut>;

    /// <summary>np.where-style ternary select.</summary>
    public static void Where<T, TKernel>(
        NDArray condition, NDArray ifTrue, NDArray ifFalse, NDArray dest)
        where T : unmanaged
        where TKernel : struct, ITernaryKernel<T>;

    /// <summary>Create boolean mask.</summary>
    public static NDArray Mask<T, TKernel>(NDArray source)
        where T : unmanaged
        where TKernel : struct, IPredicateKernel<T>;

    // =========================================================================
    // TIER 2: Direct IL Injection (Full control)
    // =========================================================================

    /// <summary>Unary transform with custom IL emitter.</summary>
    public static void Transform<TIn, TOut>(
        NDArray source, NDArray dest, IKernelEmitter emitter)
        where TIn : unmanaged
        where TOut : unmanaged;

    /// <summary>Unary transform with inline IL delegate.</summary>
    public static void Transform<TIn, TOut>(
        NDArray source, NDArray dest, KernelEmitDelegate emitKernel)
        where TIn : unmanaged
        where TOut : unmanaged;

    /// <summary>Binary transform with custom IL emitter.</summary>
    public static void TransformBinary<TL, TR, TOut>(
        NDArray left, NDArray right, NDArray dest, IKernelEmitter emitter)
        where TL : unmanaged
        where TR : unmanaged
        where TOut : unmanaged;

    /// <summary>Binary transform with inline IL delegate.</summary>
    public static void TransformBinary<TL, TR, TOut>(
        NDArray left, NDArray right, NDArray dest, KernelEmitDelegate emitKernel)
        where TL : unmanaged
        where TR : unmanaged
        where TOut : unmanaged;

    /// <summary>Reduction with custom IL emitter.</summary>
    public static TAccum Reduce<TIn, TAccum>(
        NDArray source, IKernelEmitter emitter, TAccum identity)
        where TIn : unmanaged
        where TAccum : unmanaged;

    /// <summary>Axis reduction with custom IL emitter.</summary>
    public static NDArray ReduceAxis<TIn, TAccum>(
        NDArray source, int axis, IKernelEmitter emitter, TAccum identity)
        where TIn : unmanaged
        where TAccum : unmanaged;

    // =========================================================================
    // TIER 3: Func/Action Delegates (Simple, ~10-15 cycle overhead)
    // =========================================================================

    /// <summary>Unary transform with delegate.</summary>
    public static void Transform<TIn, TOut>(
        NDArray source, NDArray dest, Func<TIn, TOut> transform)
        where TIn : unmanaged
        where TOut : unmanaged;

    /// <summary>Binary transform with delegate.</summary>
    public static void TransformBinary<TL, TR, TOut>(
        NDArray left, NDArray right, NDArray dest, Func<TL, TR, TOut> transform)
        where TL : unmanaged
        where TR : unmanaged
        where TOut : unmanaged;

    /// <summary>Reduction with delegate.</summary>
    public static TAccum Reduce<TIn, TAccum>(
        NDArray source, Func<TAccum, TIn, TAccum> combine, TAccum identity)
        where TIn : unmanaged
        where TAccum : unmanaged;

    /// <summary>Indexed reduction with delegate.</summary>
    public static TAccum ReduceIndexed<TIn, TAccum>(
        NDArray source, Func<TAccum, TIn, int, TAccum> combine, TAccum identity)
        where TIn : unmanaged
        where TAccum : unmanaged;

    /// <summary>Axis reduction with delegate.</summary>
    public static NDArray ReduceAxis<TIn, TAccum>(
        NDArray source, int axis, Func<TAccum, TIn, TAccum> combine, TAccum identity)
        where TIn : unmanaged
        where TAccum : unmanaged;

    /// <summary>Create mask with predicate delegate.</summary>
    public static NDArray Mask<T>(NDArray source, Func<T, bool> predicate)
        where T : unmanaged;

    /// <summary>np.where with delegate.</summary>
    public static void Where<T>(
        NDArray condition, NDArray ifTrue, NDArray ifFalse, NDArray dest,
        Func<bool, T, T, T> select)
        where T : unmanaged;

    // =========================================================================
    // AXIS ITERATION (Direct pointer access)
    // =========================================================================

    /// <summary>
    /// Iterate along an axis, providing pointer + stride for each slice.
    /// Replaces the old Slice[]-based pattern entirely.
    /// </summary>
    public static unsafe void IterateAxis<T>(
        NDArray source,
        int axis,
        AxisIterationDelegate<T> callback)
        where T : unmanaged;

    // =========================================================================
    // LOW-LEVEL ACCESS
    // =========================================================================

    /// <summary>
    /// Create an iterator state for manual iteration.
    /// For advanced use cases requiring custom control flow.
    /// </summary>
    public static IteratorState CreateState(NDArray source, NDArray? dest = null);

    /// <summary>Advance to next element, returning offsets.</summary>
    public static bool MoveNext(ref IteratorState state,
        out int inputOffset, out int outputOffset);

    /// <summary>Advance to next axis slice.</summary>
    public static bool MoveNextAxis(ref IteratorState state,
        out ReadOnlySpan<int> axisOffsets);
}

/// <summary>Callback for axis iteration.</summary>
public unsafe delegate void AxisIterationDelegate<T>(
    T* axisData,           // Pointer to start of axis slice
    int axisStride,        // Stride between elements along axis
    int axisLength,        // Number of elements along axis
    int sliceIndex)        // Which slice (0..numSlices-1)
    where T : unmanaged;
```

---

## SIMD Strategy

| Tier | Loop Generation | Kernel Responsibility |
|------|-----------------|----------------------|
| Tier 1 (Interface) | NDIterator generates SIMD loop | Kernel provides `ApplyVector` for SIMD, `Apply` for scalar tail |
| Tier 2 (IL Inject) | NDIterator generates scalar loop | User handles SIMD manually if desired |
| Tier 3 (Func<>) | NDIterator generates scalar loop | No SIMD (delegate overhead dominates anyway) |

**Rationale**:
- Tier 1 kernels can opt-in to SIMD via `ApplyVector`. If it returns 0, scalar `Apply` is used.
- Tier 2 gives full IL control; user can emit their own SIMD if needed.
- Tier 3 is for simplicity; the delegate call overhead makes SIMD gains negligible.

---

## Migration: Old Patterns to New

### Pattern 1: NDIterator Element Iteration

**Old:**
```csharp
var iter = new NDIterator<double>(source, false);
while (iter.HasNext())
{
    var val = iter.MoveNext();
    sum += val * val;
}
```

**New (Tier 1):**
```csharp
sum = NDIterator.Reduce<double, double, SumOfSquaresKernel>(source);

public readonly struct SumOfSquaresKernel : IReductionKernel<double, double>
{
    public static double Identity => 0.0;
    public static double Combine(double acc, double val) => acc + val * val;
}
```

**New (Tier 3):**
```csharp
sum = NDIterator.Reduce<double, double>(source, (acc, val) => acc + val * val, 0.0);
```

### Pattern 2: NDCoordinatesAxisIncrementor with Slices

**Old:**
```csharp
var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
var slices = iterAxis.Slices;
do
{
    var slice = arr[slices];  // Creates view
    var result = ProcessSlice(slice);
    ret[slices] = result;
} while (iterAxis.Next() != null);
```

**New:**
```csharp
// Direct pointer-based axis iteration
NDIterator.IterateAxis<double>(arr, axis,
    (double* axisData, int stride, int length, int sliceIdx) =>
    {
        // Process axis data directly via pointer
        double sum = 0;
        for (int i = 0; i < length; i++)
            sum += axisData[i * stride];
        output[sliceIdx] = sum;
    });
```

**New (with kernel):**
```csharp
// For cumsum along axis
NDIterator.IterateAxisTransform<double, double, CumSumAxisKernel<double>>(
    source, dest, axis);
```

### Pattern 3: MultiIterator Broadcast Assignment

**Old:**
```csharp
var (lIter, rIter) = MultiIterator.GetIterators(lhs, rhs, broadcast: true);
while (lIter.HasNext())
    lIter.MoveNextReference() = rIter.MoveNext();
```

**New:**
```csharp
NDIterator.TransformBinary<T, T, T, AssignKernel<T>>(rhs, lhs, lhs);
// Or more directly:
NDIterator.Copy(rhs, lhs);  // Handles broadcasting internally
```

### Pattern 4: ValueCoordinatesIncrementor for Coordinate Access

**Old:**
```csharp
var incr = new ValueCoordinatesIncrementor(ref shape);
int[] coords = incr.Index;
do
{
    var offset = shape.GetOffset(coords);
    Process(data + offset);
} while (incr.Next() != null);
```

**New:**
```csharp
var state = NDIterator.CreateState(arr);
while (NDIterator.MoveNext(ref state, out int offset, out _))
{
    Process(data + offset);
}
```

---

## Files to Delete (Post-Migration)

```
src/NumSharp.Core/Backends/Iterators/
|-- INDIterator.cs                    [DELETE]
|-- IteratorType.cs                   [DELETE]
|-- MultiIterator.cs                  [DELETE]
|-- NDIterator.cs                     [DELETE]
|-- NDIterator.template.cs            [DELETE]
|-- NDIteratorExtensions.cs           [DELETE]
+-- NDIteratorCasts/
    +-- NDIterator.Cast.*.cs (x12)    [DELETE]

src/NumSharp.Core/Utilities/Incrementors/
|-- NDCoordinatesAxisIncrementor.cs   [DELETE]
|-- NDCoordinatesIncrementor.cs       [DELETE]
|-- NDCoordinatesLeftToAxisIncrementor.cs  [DELETE - already dead code]
|-- NDExtendedCoordinatesIncrementor.cs    [DELETE - already dead code]
|-- NDOffsetIncrementor.cs            [DELETE - already dead code]
|-- ValueCoordinatesIncrementor.cs    [DELETE]
+-- ValueOffsetIncrementor.cs         [DELETE]
```

**Total: 23 files to delete**

---

## New Files to Create

```
src/NumSharp.Core/Backends/Iterators/
|-- NDIterator.cs                   # Main static API class
|-- NDIterator.State.cs             # IteratorState struct, IteratorFlags
|-- NDIterator.Kernels.cs           # All kernel interfaces
|-- NDIterator.Kernels.Builtin.cs   # Built-in kernel implementations
|-- NDIterator.IL.cs                # IL generation for iteration loops
|-- NDIterator.Axis.cs              # Axis-specific iteration
+-- NDIterator.Emitters.cs          # IKernelEmitter, KernelEmitContext, helpers
```

**Total: 7 new files (~3,000 lines estimated)**

---

## Performance Expectations

| Tier | Kernel Overhead | JIT Inlining | SIMD | Use Case |
|------|-----------------|--------------|------|----------|
| 1 (Interface) | ~0 cycles | Yes (small methods) | Via ApplyVector | Built-in ops, hot paths |
| 2 (IL Inject) | ~0 cycles | Full control | Manual | Custom complex ops |
| 3 (Func<>) | ~10-15 cycles | No | No | Prototyping, cold paths |
| Old (NDIterator) | ~10-15 cycles | No | No | **Eliminated** |

**Key insight**: Tier 1 and Tier 2 emit direct IL with no delegate indirection. Tier 3 has delegate overhead but is much simpler to use. Choose based on performance requirements.

---

## Scope Limitations

The following are **out of scope** for NDIterator:

1. **Multi-output operations** (e.g., `modf` returning two arrays) - Use ILKernelGenerator directly
2. **Type promotion** - Caller's responsibility using existing NumSharp utilities
3. **Broadcasting** - Caller provides already-broadcast NDArrays
4. **Memory allocation** - Caller provides output NDArray

NDIterator focuses on the common mathematical iteration patterns. Specialized operations should use ILKernelGenerator or custom implementations.
