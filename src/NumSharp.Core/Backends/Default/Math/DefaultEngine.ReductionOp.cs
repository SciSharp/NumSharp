using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    /// <summary>
    /// Reduction operation dispatch using IL-generated kernels.
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute an element-wise reduction operation (axis=null) using IL-generated kernels.
        /// Reduces all elements to a single scalar value.
        /// </summary>
        /// <typeparam name="TResult">Result type</typeparam>
        /// <param name="arr">Input array</param>
        /// <param name="op">Reduction operation</param>
        /// <param name="accumulatorType">Optional accumulator type (defaults to input type)</param>
        /// <returns>Scalar result</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal unsafe TResult ExecuteElementReduction<TResult>(NDArray arr, ReductionOp op, NPTypeCode? accumulatorType = null)
            where TResult : unmanaged
        {
            if (arr.size == 0)
            {
                // Return identity for empty arrays
                return (TResult)op.GetIdentity(typeof(TResult).GetTypeCode());
            }

            var inputType = arr.GetTypeCode;
            var accumType = accumulatorType ?? inputType.GetAccumulatingType();

            // Handle scalar case - just return the value (possibly converted)
            if (arr.Shape.IsScalar)
            {
                return ExecuteScalarReduction<TResult>(arr, op, accumType);
            }

            // Broadcast views (a stride-0 axis with dim>1) are the worst flat-reduction
            // case: the coordinate-walk kernel visits every one of the D×N logical elements
            // computing coordinates per element, though only N are unique — ~50× NumPy on the
            // bcast-reduce canary (np.sum(broadcast_to(a,(1024,8192)))). NumPy never copies a
            // broadcast to reduce it; it folds the stride-0 axis in place, cache-hot. Match
            // that: fold each broadcast axis with the fast per-chunk axis kernel — O(unique)
            // memory (NOT the O(D×N) a full materialize would allocate, which OOMs on a large
            // broadcast the coordinate walk merely handled slowly), and cache-hot like NumPy.
            // Folding a stride-0 axis with the SAME op collapses its D identical copies
            // (sum→×D, prod→^D, min/max→identity, exact); accumType matches the op (widened
            // for sum/prod, preserved for min/max), so the fold dtype is correct. The
            // remainder is a small contiguous array → fast flat finish. (Residual vs NumPy is
            // ULP-level summation-order, which the codebase's pairwise reductions already
            // accept; min/max are order-independent → exact.) ArgMin/ArgMax opt out — their
            // result index must address the full broadcast, which folding would destroy.
            if (arr.Shape.IsBroadcasted && op != ReductionOp.ArgMax && op != ReductionOp.ArgMin)
            {
                while (arr.Shape.IsBroadcasted)
                {
                    int ax = -1;
                    for (int d = arr.ndim - 1; d >= 0; d--)
                        if (arr.Shape.strides[d] == 0 && arr.Shape.dimensions[d] > 1) { ax = d; break; }
                    if (ax < 0) break;
                    arr = ExecuteAxisReduction(arr, ax, false, accumType, null, op);
                    if (arr.Shape.IsScalar || arr.size == 1)
                        return ExecuteScalarReduction<TResult>(arr, op, accumType);
                }
                inputType = arr.GetTypeCode; // collapse output dtype == accumType (may have widened)
            }

            // Determine if array is contiguous
            bool isContiguous = arr.Shape.IsContiguous;

            // ─── NpyIter routing for non-contig flat reduction ─────────────
            // The direct ElementReductionKernel walks non-contig inputs via
            // coordinate math per element, which made strided/transposed
            // reductions 20-54× slower than NumPy. NpyIter coalesces dims,
            // permutes axes by stride magnitude (NPY_KEEPORDER-style), and
            // normalizes negative strides — so the kernel is called with a
            // layout it can handle as contig in the inner loop.
            //
            // Contig stays on the direct path: zero dispatch overhead, and
            // the existing kernel is already at parity / faster than NumPy
            // there.
            //
            // ArgMax/ArgMin opt out: the returned index must be the C-order
            // flat position of the extreme element, but NpyIter permutes axes
            // by stride magnitude which can re-order the traversal and break
            // that contract. (e.g. argmax(arr.T) on a 2D F-contig view: C-order
            // expects [1,9,2,4]→idx 1; NpyIter's F-walk gives [1,2,9,4]→idx 2.)
            // Direct path's coordinate walk preserves the C-order contract.
            if (!isContiguous && op != ReductionOp.ArgMax && op != ReductionOp.ArgMin)
            {
                var routed = TryExecuteElementReductionViaNpyIter<TResult>(arr, op, inputType, accumType);
                if (routed.HasValue) return routed.Value;
            }

            // Get kernel key
            var key = new ElementReductionKernelKey(inputType, accumType, op, isContiguous);

            // Get or generate kernel
            var kernel = DirectILKernelGenerator.TryGetTypedElementReductionKernel<TResult>(key);

            if (kernel != null)
            {
                return ExecuteTypedReductionKernel<TResult>(kernel, arr);
            }
            else
            {
                // Fallback - should not happen for implemented operations
                throw new NotSupportedException(
                    $"IL kernel not available for {op}({inputType}) -> {accumType}. " +
                    "Please report this as a bug.");
            }
        }

        /// <summary>
        ///     NpyIter routing for non-contig flat reductions.
        ///
        ///     The iterator does the heavy lifting before the kernel runs:
        ///     dimension coalescing, axis permutation by stride magnitude,
        ///     negative-stride normalization. After that, the existing
        ///     ElementReductionKernel handles the per-element loop.
        ///
        ///     Returns the reduction result on success, null when the iterator
        ///     can't be set up (e.g. dim > int.MaxValue) so the caller falls
        ///     back to the direct path.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private unsafe TResult? TryExecuteElementReductionViaNpyIter<TResult>(
            NDArray arr, ReductionOp op, NPTypeCode inputType, NPTypeCode accumType)
            where TResult : unmanaged
        {
            var shape = arr.Shape;
            if (shape.size < 0) return null;
            for (int i = 0; i < shape.NDim; i++)
                if (shape.dimensions[i] > int.MaxValue) return null;

            try
            {
                using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.None);
                return iter.ExecuteReduction<TResult>(op);
            }
            catch (Exception)
            {
                // Catch broadly: iterator setup or kernel resolution may fail
                // for combos that the direct path still handles via fallback.
                return null;
            }
        }

        /// <summary>
        /// Execute scalar reduction - just return the value, possibly converted.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static TResult ExecuteScalarReduction<TResult>(NDArray arr, ReductionOp op, NPTypeCode accumType)
            where TResult : unmanaged
        {
            // For ArgMax/ArgMin of scalar, index is 0
            if (op == ReductionOp.ArgMax || op == ReductionOp.ArgMin)
            {
                return (TResult)(object)0;
            }

            // For other ops, return the scalar value converted to result type
            var value = arr.GetAtIndex(0);
            return (TResult)Converts.ChangeType(value, typeof(TResult).GetTypeCode());
        }

        /// <summary>
        /// Execute the IL-generated typed reduction kernel.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        private static unsafe TResult ExecuteTypedReductionKernel<TResult>(
            TypedElementReductionKernel<TResult> kernel,
            NDArray input)
            where TResult : unmanaged
        {
            int inputElemSize = input.dtypesize;
            var inputShape = input.Shape;

            // Calculate base address accounting for shape offset (for sliced views)
            byte* inputAddr = (byte*)input.Address + inputShape.offset * inputElemSize;

            fixed (long* strides = inputShape.strides)
            fixed (long* shape = inputShape.dimensions)
            {
                return kernel(
                    (void*)inputAddr,
                    strides,
                    shape,
                    input.ndim,
                    input.size
                );
            }
        }

        #region Type-Specific Element Reduction Wrappers

        /// <summary>
        /// Execute element-wise sum reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected object sum_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Sum, retType),
                NPTypeCode.SByte => ExecuteElementReduction<sbyte>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Sum, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Sum, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Sum, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Sum, retType),
                NPTypeCode.Half => SumElementwiseHalfFallback(arr),
                NPTypeCode.Complex => SumElementwiseComplexFallback(arr),
                _ => throw new NotSupportedException($"Sum not supported for type {retType}")
            };
        }

        /// <summary>
        /// Execute element-wise product reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected object prod_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Prod, retType),
                NPTypeCode.SByte => ExecuteElementReduction<sbyte>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Prod, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Prod, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Prod, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Prod, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Prod, retType),
                // B4: Half and Complex fallbacks (IL kernel doesn't cover them).
                NPTypeCode.Half => ProdElementwiseHalfFallback(arr),
                NPTypeCode.Complex => ProdElementwiseComplexFallback(arr),
                _ => throw new NotSupportedException($"Prod not supported for type {retType}")
            };
        }

        /// <summary>
        /// Fallback product for Half using iterator (double accumulator for precision).
        /// Matches NumPy: product of empty array is 1.0.
        /// </summary>
        private object ProdElementwiseHalfFallback(NDArray arr)
        {
            double prod = 1.0;
            var iter = arr.AsIterator<Half>();
            while (iter.HasNext())
                prod *= (double)iter.MoveNext();
            return (Half)prod;
        }

        /// <summary>
        /// Fallback product for Complex using iterator.
        /// </summary>
        private object ProdElementwiseComplexFallback(NDArray arr)
        {
            var prod = System.Numerics.Complex.One;
            var iter = arr.AsIterator<System.Numerics.Complex>();
            while (iter.HasNext())
                prod *= iter.MoveNext();
            return prod;
        }

        /// <summary>
        /// Execute element-wise max reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected object max_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode;

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Max, retType),
                NPTypeCode.SByte => ExecuteElementReduction<sbyte>(arr, ReductionOp.Max, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Max, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Max, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Max, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Max, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Max, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Max, retType),
                // B1: Half IL kernel uses OpCodes.Bgt/Blt which don't work on Half struct; use fallback.
                NPTypeCode.Half => MaxElementwiseHalfFallback(arr),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Max, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Max, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Max, retType),
                // B8: Complex has no total ordering; NumPy uses lexicographic (real then imag) compare.
                NPTypeCode.Complex => MaxElementwiseComplexFallback(arr),
                // Boolean: max == "any true" (logical OR). NumPy: np.max([T,F,T]) → True.
                NPTypeCode.Boolean => MaxElementwiseBooleanFallback(arr),
                // Char: scalar comparison via char's natural ordering.
                NPTypeCode.Char => MaxElementwiseCharFallback(arr),
                _ => throw new NotSupportedException($"Max not supported for type {retType}")
            };
        }

        /// <summary>
        /// Execute element-wise min reduction using IL kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected object min_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Converts.ChangeType(arr.GetAtIndex(0), typeCode.Value) : arr.GetAtIndex(0);

            var retType = typeCode ?? arr.GetTypeCode;

            return retType switch
            {
                NPTypeCode.Byte => ExecuteElementReduction<byte>(arr, ReductionOp.Min, retType),
                NPTypeCode.SByte => ExecuteElementReduction<sbyte>(arr, ReductionOp.Min, retType),
                NPTypeCode.Int16 => ExecuteElementReduction<short>(arr, ReductionOp.Min, retType),
                NPTypeCode.UInt16 => ExecuteElementReduction<ushort>(arr, ReductionOp.Min, retType),
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Min, retType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Min, retType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Min, retType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Min, retType),
                // B1: Half IL kernel uses OpCodes.Bgt/Blt which don't work on Half struct; use fallback.
                NPTypeCode.Half => MinElementwiseHalfFallback(arr),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Min, retType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Min, retType),
                NPTypeCode.Decimal => ExecuteElementReduction<decimal>(arr, ReductionOp.Min, retType),
                // B8: Complex has no total ordering; NumPy uses lexicographic (real then imag) compare.
                NPTypeCode.Complex => MinElementwiseComplexFallback(arr),
                // Boolean: min == "all true" (logical AND). NumPy: np.min([T,F,T]) → False.
                NPTypeCode.Boolean => MinElementwiseBooleanFallback(arr),
                NPTypeCode.Char => MinElementwiseCharFallback(arr),
                _ => throw new NotSupportedException($"Min not supported for type {retType}")
            };
        }

        /// <summary>
        /// Max/min for Half. The IL reduction kernel can't drive Half (OpCodes.Bgt/Blt don't
        /// apply to the struct), so this stays out-of-IL — but Half DOES expose a hardware-backed
        /// comparison order, so the contiguous buffer is scanned with Half's own operators rather
        /// than bridging every element through (double). That boxing-free, no-round-trip scan is
        /// ~9× the old iterator+double path. NaN propagates per NumPy (max/min with NaN → NaN):
        /// once the accumulator is NaN, <c>x &gt; acc</c> is false and only another NaN re-sets it,
        /// so the first NaN sticks. Non-contiguous / empty inputs keep the iterator fallback.
        /// </summary>
        private unsafe object MaxElementwiseHalfFallback(NDArray arr)
        {
            long n = arr.size;
            if (arr.Shape.IsContiguous && n > 0)
            {
                Half* p = (Half*)((byte*)arr.Address + arr.Shape.offset * 2);
                Half acc = p[0];
                for (long i = 1; i < n; i++) { Half x = p[i]; if (x > acc || Half.IsNaN(x)) acc = x; }
                return acc;
            }

            var iter = arr.AsIterator<Half>();
            double best = double.NegativeInfinity;
            bool seenAny = false;
            while (iter.HasNext())
            {
                double v = (double)iter.MoveNext();
                if (double.IsNaN(v)) return Half.NaN;
                if (!seenAny || v > best) { best = v; seenAny = true; }
            }
            return (Half)best;
        }

        private unsafe object MinElementwiseHalfFallback(NDArray arr)
        {
            long n = arr.size;
            if (arr.Shape.IsContiguous && n > 0)
            {
                Half* p = (Half*)((byte*)arr.Address + arr.Shape.offset * 2);
                Half acc = p[0];
                for (long i = 1; i < n; i++) { Half x = p[i]; if (x < acc || Half.IsNaN(x)) acc = x; }
                return acc;
            }

            var iter = arr.AsIterator<Half>();
            double best = double.PositiveInfinity;
            bool seenAny = false;
            while (iter.HasNext())
            {
                double v = (double)iter.MoveNext();
                if (double.IsNaN(v)) return Half.NaN;
                if (!seenAny || v < best) { best = v; seenAny = true; }
            }
            return (Half)best;
        }

        /// <summary>
        /// Fallback max/min for Complex: NumPy uses lexicographic comparison (real first, imag as tie-break).
        /// A NaN in either component propagates: the first NaN-bearing element is returned VERBATIM
        /// (matching NumPy's minimum/maximum, which return the NaN operand as-is — not (nan,nan)).
        /// </summary>
        private object MaxElementwiseComplexFallback(NDArray arr)
        {
            var iter = arr.AsIterator<System.Numerics.Complex>();
            var best = System.Numerics.Complex.Zero;
            bool seenAny = false;
            while (iter.HasNext())
            {
                var v = iter.MoveNext();
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary))
                    return v; // NumPy parity: the NaN-bearing operand is returned VERBATIM (not a
                              // synthesized (nan,nan)); in a left-fold the FIRST NaN element wins and
                              // stays, so returning v on the first NaN matches np.min/np.max exactly
                              // (e.g. min([1+1j, nan+0j, 2+2j]) -> (nan, 0), not (nan, nan)).
                if (!seenAny
                    || v.Real > best.Real
                    || (v.Real == best.Real && v.Imaginary > best.Imaginary))
                {
                    best = v;
                    seenAny = true;
                }
            }
            return best;
        }

        private object MinElementwiseComplexFallback(NDArray arr)
        {
            var iter = arr.AsIterator<System.Numerics.Complex>();
            var best = System.Numerics.Complex.Zero;
            bool seenAny = false;
            while (iter.HasNext())
            {
                var v = iter.MoveNext();
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary))
                    return v; // NumPy parity: the NaN-bearing operand is returned VERBATIM (not a
                              // synthesized (nan,nan)); in a left-fold the FIRST NaN element wins and
                              // stays, so returning v on the first NaN matches np.min/np.max exactly
                              // (e.g. min([1+1j, nan+0j, 2+2j]) -> (nan, 0), not (nan, nan)).
                if (!seenAny
                    || v.Real < best.Real
                    || (v.Real == best.Real && v.Imaginary < best.Imaginary))
                {
                    best = v;
                    seenAny = true;
                }
            }
            return best;
        }

        /// <summary>
        ///     Boolean max == "any true" (logical OR). NumPy parity:
        ///     <c>np.max([T,F,T])</c> → <c>True</c>. Short-circuits on first true.
        /// </summary>
        private object MaxElementwiseBooleanFallback(NDArray arr)
        {
            var iter = arr.AsIterator<bool>();
            while (iter.HasNext())
                if (iter.MoveNext()) return true;
            return false;
        }

        /// <summary>
        ///     Boolean min == "all true" (logical AND). NumPy parity:
        ///     <c>np.min([T,F,T])</c> → <c>False</c>. Short-circuits on first false.
        /// </summary>
        private object MinElementwiseBooleanFallback(NDArray arr)
        {
            var iter = arr.AsIterator<bool>();
            while (iter.HasNext())
                if (!iter.MoveNext()) return false;
            return true;
        }

        /// <summary>
        /// Char max/min via uint16 min/max. char is unsigned 16-bit with a total order
        /// bit-identical to its UTF-16 code unit, so the char buffer reduces bit-for-bit
        /// through the ushort SIMD reducer (vpminuw/vpmaxuw) — ~100× the scalar char
        /// iterator this used to run, and it reuses ExecuteElementReduction's full routing
        /// (contig SIMD, broadcast-fold, NpyIter-strided). <see cref="view"/>(ushort) is a
        /// zero-copy byte reinterpret (char and ushort are both 2 bytes, so shape/strides/
        /// offset are preserved across every layout). The same trick that fixed bool/char
        /// amin/amax along an axis.
        /// </summary>
        private object MaxElementwiseCharFallback(NDArray arr)
            => (char)ExecuteElementReduction<ushort>(arr.view(typeof(ushort)), ReductionOp.Max, NPTypeCode.UInt16);

        private object MinElementwiseCharFallback(NDArray arr)
            => (char)ExecuteElementReduction<ushort>(arr.view(typeof(ushort)), ReductionOp.Min, NPTypeCode.UInt16);

        /// <summary>
        /// Execute element-wise argmax reduction using IL kernels.
        /// Returns the index of the maximum value.
        /// All types including Boolean, Single, Double now use unified IL kernel path.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected long argmax_elementwise_il(NDArray arr)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return 0;

            var inputType = arr.GetTypeCode;

            // ArgMax returns long (int64) to match NumPy 2.x behavior
            // IL kernel tracks index as long internally, supports arrays >2GB elements
            // All types use IL kernels - NaN-aware helpers for float/double, bool-aware for boolean
            return inputType switch
            {
                NPTypeCode.Boolean => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Boolean),
                NPTypeCode.Byte => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Byte),
                NPTypeCode.SByte => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.SByte),
                NPTypeCode.Int16 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Int16),
                NPTypeCode.UInt16 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.UInt16),
                NPTypeCode.Int32 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Int32),
                NPTypeCode.UInt32 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.UInt32),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Int64),
                NPTypeCode.UInt64 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.UInt64),
                // B1/B7: IL OpCodes.Bgt don't work on Half struct; use C# fallback.
                NPTypeCode.Half => ArgMaxHalfFallback(arr),
                NPTypeCode.Single => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Single),
                NPTypeCode.Double => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Double),
                NPTypeCode.Decimal => ExecuteElementReduction<long>(arr, ReductionOp.ArgMax, NPTypeCode.Decimal),
                // B12: Complex IL kernel tiebreak is wrong; fallback uses lexicographic compare.
                NPTypeCode.Complex => ArgMaxComplexFallback(arr),
                _ => throw new NotSupportedException($"ArgMax not supported for type {inputType}")
            };
        }

        /// <summary>
        /// Fallback argmax for Half (IL kernel uses Bgt which doesn't work on Half struct).
        /// NumPy: first occurrence of max; NaN propagates (argmax of array with NaN returns index of first NaN).
        /// </summary>
        private long ArgMaxHalfFallback(NDArray arr)
        {
            var iter = arr.AsIterator<Half>();
            long bestIdx = 0;
            long idx = 0;
            double best = (double)iter.MoveNext();
            if (double.IsNaN(best)) return 0;
            idx = 1;
            while (iter.HasNext())
            {
                double v = (double)iter.MoveNext();
                if (double.IsNaN(v)) return idx;
                if (v > best) { best = v; bestIdx = idx; }
                idx++;
            }
            return bestIdx;
        }

        private long ArgMinHalfFallback(NDArray arr)
        {
            var iter = arr.AsIterator<Half>();
            long bestIdx = 0;
            long idx = 0;
            double best = (double)iter.MoveNext();
            if (double.IsNaN(best)) return 0;
            idx = 1;
            while (iter.HasNext())
            {
                double v = (double)iter.MoveNext();
                if (double.IsNaN(v)) return idx;
                if (v < best) { best = v; bestIdx = idx; }
                idx++;
            }
            return bestIdx;
        }

        /// <summary>
        /// Fallback argmax for Complex using lexicographic comparison (real, then imag).
        /// Returns index of first occurrence of the maximum (NumPy tiebreak semantics).
        /// NaN propagates: a Complex value with NaN in either component "wins" argmax at its first occurrence.
        /// </summary>
        private long ArgMaxComplexFallback(NDArray arr)
        {
            var iter = arr.AsIterator<System.Numerics.Complex>();
            long bestIdx = 0;
            long idx = 0;
            var best = iter.MoveNext();
            if (double.IsNaN(best.Real) || double.IsNaN(best.Imaginary)) return 0;
            idx = 1;
            while (iter.HasNext())
            {
                var v = iter.MoveNext();
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary)) return idx;
                if (v.Real > best.Real || (v.Real == best.Real && v.Imaginary > best.Imaginary))
                {
                    best = v;
                    bestIdx = idx;
                }
                idx++;
            }
            return bestIdx;
        }

        /// <summary>
        /// Fallback argmin for Complex using lexicographic comparison (real, then imag).
        /// NaN propagates: a Complex value with NaN in either component "wins" argmin at its first occurrence.
        /// </summary>
        private long ArgMinComplexFallback(NDArray arr)
        {
            var iter = arr.AsIterator<System.Numerics.Complex>();
            long bestIdx = 0;
            long idx = 0;
            var best = iter.MoveNext();
            if (double.IsNaN(best.Real) || double.IsNaN(best.Imaginary)) return 0;
            idx = 1;
            while (iter.HasNext())
            {
                var v = iter.MoveNext();
                if (double.IsNaN(v.Real) || double.IsNaN(v.Imaginary)) return idx;
                if (v.Real < best.Real || (v.Real == best.Real && v.Imaginary < best.Imaginary))
                {
                    best = v;
                    bestIdx = idx;
                }
                idx++;
            }
            return bestIdx;
        }

        /// <summary>
        /// Execute element-wise argmin reduction using IL kernels.
        /// Returns the index of the minimum value.
        /// All types including Boolean, Single, Double now use unified IL kernel path.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected long argmin_elementwise_il(NDArray arr)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return 0;

            var inputType = arr.GetTypeCode;

            // ArgMin returns long (int64) to match NumPy 2.x behavior
            // IL kernel tracks index as long internally, supports arrays >2GB elements
            // All types use IL kernels - NaN-aware helpers for float/double, bool-aware for boolean
            return inputType switch
            {
                NPTypeCode.Boolean => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Boolean),
                NPTypeCode.Byte => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Byte),
                NPTypeCode.SByte => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.SByte),
                NPTypeCode.Int16 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Int16),
                NPTypeCode.UInt16 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.UInt16),
                NPTypeCode.Int32 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Int32),
                NPTypeCode.UInt32 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.UInt32),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Int64),
                NPTypeCode.UInt64 => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.UInt64),
                // B1/B7: IL OpCodes.Blt don't work on Half struct; use C# fallback.
                NPTypeCode.Half => ArgMinHalfFallback(arr),
                NPTypeCode.Single => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Single),
                NPTypeCode.Double => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Double),
                NPTypeCode.Decimal => ExecuteElementReduction<long>(arr, ReductionOp.ArgMin, NPTypeCode.Decimal),
                // B12: Complex IL kernel tiebreak is wrong; fallback uses lexicographic compare.
                NPTypeCode.Complex => ArgMinComplexFallback(arr),
                _ => throw new NotSupportedException($"ArgMin not supported for type {inputType}")
            };
        }

        /// <summary>
        /// Execute element-wise mean using IL kernels for sum.
        /// Mean = Sum / count
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        protected object mean_elementwise_il(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
            {
                var val = arr.GetAtIndex(0);
                if (arr.GetTypeCode == NPTypeCode.Complex)
                    return val; // Complex mean of single element is the element itself
                // Converts.ToDouble handles all 15 dtypes including Half/Complex (System.Convert throws on those).
                return typeCode.HasValue ? Converts.ChangeType(val, typeCode.Value) : Converts.ToDouble(val);
            }

            long count = arr.size;
            var sumType = arr.GetTypeCode.GetAccumulatingType();

            // Handle Complex separately - mean is Complex, not double
            if (sumType == NPTypeCode.Complex)
            {
                var sum = ExecuteElementReduction<System.Numerics.Complex>(arr, ReductionOp.Sum, sumType);
                return sum / count;
            }

            // Handle Half separately - NumPy 2.x preserves float16 dtype for mean
            if (sumType == NPTypeCode.Half)
            {
                var sum = ExecuteElementReduction<Half>(arr, ReductionOp.Sum, sumType);
                return (Half)((double)sum / count);
            }

            // NumPy 2.x: mean preserves float types, promotes int to float64
            var retType = typeCode ?? (arr.GetTypeCode switch
            {
                NPTypeCode.Single => NPTypeCode.Single,
                NPTypeCode.Double => NPTypeCode.Double,
                _ => NPTypeCode.Double
            });

            // Sum in accumulating type, then divide
            double sum2 = sumType switch
            {
                NPTypeCode.Int32 => ExecuteElementReduction<int>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.UInt32 => ExecuteElementReduction<uint>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Int64 => ExecuteElementReduction<long>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.UInt64 => ExecuteElementReduction<ulong>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Single => ExecuteElementReduction<float>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Double => ExecuteElementReduction<double>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Decimal => (double)ExecuteElementReduction<decimal>(arr, ReductionOp.Sum, sumType),
                NPTypeCode.Half => (double)ExecuteElementReduction<Half>(arr, ReductionOp.Sum, sumType),
                _ => throw new NotSupportedException($"Mean not supported for accumulator type {sumType}")
            };

            double mean = sum2 / count;
            return Converts.ChangeType(mean, retType);
        }

        #endregion

        #region Half/Complex Fallback Methods

        /// <summary>
        /// Fallback sum for Half type using iterator.
        /// </summary>
        private object SumElementwiseHalfFallback(NDArray arr)
        {
            double sum = 0.0;
            var iter = arr.AsIterator<Half>();
            while (iter.HasNext())
                sum += (double)iter.MoveNext();
            return (Half)sum;
        }

        /// <summary>
        /// Fallback sum for Complex type using iterator.
        /// </summary>
        private object SumElementwiseComplexFallback(NDArray arr)
        {
            var sum = System.Numerics.Complex.Zero;
            var iter = arr.AsIterator<System.Numerics.Complex>();
            while (iter.HasNext())
                sum += iter.MoveNext();
            return sum;
        }

        #endregion
    }
}
