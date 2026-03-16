## Problem

NumSharp has **14+ iterator/incrementor types** with significant issues:
- **3 types are dead code** (0 usages)
- **~23 files, ~4,600 lines** of duplicated iteration logic
- **Delegate overhead**: 4-5 closures per `NDIterator`
- **No kernel fusion**: Operations applied outside iteration loop
- **No SIMD integration**: Sequential element-by-element only

---

## Solution: Unified NDIterator with 3-Tier Kernel System

### Tier 1: Static Interfaces (Zero overhead, SIMD)

```csharp
public interface IUnaryKernel<TIn, TOut>
{
    static abstract TOut Apply(TIn value);
    static virtual int ApplyVector(ReadOnlySpan<TIn> input, Span<TOut> output) => 0;
}

public interface IReductionKernel<TIn, TAccum>
{
    static abstract TAccum Identity { get; }
    static abstract TAccum Combine(TAccum accumulator, TIn value);
    static virtual bool ShouldContinue(TAccum accumulator) => true; // Early-exit
}

// Usage
NDIterator.Transform<double, double, SquareKernel>(source, dest);
NDIterator.Reduce<double, double, SumKernel>(source);
```

### Tier 2: IL Injection (Full control)

```csharp
// Sigmoid: 1 / (1 + exp(-x))
NDIterator.Transform<double, double>(arr, result, (il, ctx) =>
{
    il.Emit(OpCodes.Neg);
    il.EmitCall(OpCodes.Call, typeof(Math).GetMethod("Exp"), null);
    il.Emit(OpCodes.Ldc_R8, 1.0);
    il.Emit(OpCodes.Add);
    il.Emit(OpCodes.Ldc_R8, 1.0);
    il.Emit(OpCodes.Div);
});
```

### Tier 3: Func Delegates (Simple, ~10-15 cycle overhead)

```csharp
NDIterator.Transform(source, dest, x => x * x);
NDIterator.Reduce(source, (acc, x) => acc + x, identity: 0.0);
NDIterator.Mask(source, x => x > 0.5);
```

---

## Kernel Interfaces (7 total)

| Interface | Purpose | SIMD |
|-----------|---------|------|
| `IUnaryKernel<TIn, TOut>` | Element transform | `ApplyVector` |
| `IBinaryKernel<TL, TR, TOut>` | Binary element-wise | `ApplyVector` |
| `IReductionKernel<TIn, TAccum>` | Sum, Prod, Min, Max + early-exit | `CombineVector` |
| `IIndexedReductionKernel<TIn, TAccum>` | ArgMax, ArgMin | - |
| `IAxisKernel<TIn, TOut>` | CumSum, CumProd (stride support) | Manual |
| `ITernaryKernel<T>` | np.where | `ApplyVector` |
| `IPredicateKernel<T>` | Boolean masking | `ApplyVector` |

---

## Performance

| Tier | Overhead | JIT Inline | SIMD | Use Case |
|------|----------|------------|------|----------|
| 1 (Interface) | ~0 | Yes | Via ApplyVector | Hot paths |
| 2 (IL Inject) | ~0 | Full control | Manual | Custom ops |
| 3 (Func<>) | ~10-15 cycles | No | No | Prototyping |
| Old NDIterator | ~10-15 cycles | No | No | **Eliminated** |

---

## Migration

**Old:**
```csharp
var iter = new NDIterator<double>(source, false);
while (iter.HasNext()) sum += iter.MoveNext() * iter.MoveNext();
```

**New:**
```csharp
sum = NDIterator.Reduce<double, double, SumOfSquaresKernel>(source);
// Or simple:
sum = NDIterator.Reduce(source, (acc, x) => acc + x * x, 0.0);
```

---

## Scope

- **23 files deleted** (iterators + incrementors)
- **7 new files** (~3,000 lines)
- **Out of scope**: Type promotion (caller's job), broadcasting (caller provides), multi-output ops

---

## Full Design

See [`docs/UNIFIED_ITERATOR_DESIGN.md`](docs/UNIFIED_ITERATOR_DESIGN.md) for:
- Complete interface definitions with examples
- 10+ IKernelEmitter examples (Power, Clip, Sigmoid, ReLU, etc.)
- IteratorState struct specification
- Stack contracts for IL emitters
- Built-in kernel implementations
