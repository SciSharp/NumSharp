using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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

            // Broadcast views (a stride-0 axis with dim>1) are the worst flat-reduction case for
            // the coordinate-walk kernel: it visits every one of the D×N logical elements though
            // only N are unique — ~50× NumPy on the bcast-reduce canary
            // (np.sum(broadcast_to(a,(1024,8192)))). A stride-0 axis repeats its data EXACTLY, so
            // reducing over it is CLOSED-FORM — there is nothing to iterate:
            //     sum → ×D,  prod → ^D,  min/max → identity   (mean = sum/count, in the caller).
            // Drop every broadcast axis to a non-broadcast view of the UNIQUE data (index 0 along
            // each stride-0 axis — an O(1) view), reduce THAT once via the fast contiguous/NpyIter
            // path, then apply the collapsed multiplicity to the scalar. O(unique), not the prior
            // O(D×unique) per-copy fold: ~960× NumPy here vs the fold's ~1.0× (measured, Release).
            // Integer and min/max are BIT-EXACT (modular ×/pow; min/max order-independent); float
            // sum/prod differ only ULP-level — the summation-order class the codebase already
            // accepts here, and one multiply is fewer roundings than D adds. ArgMin/ArgMax opt
            // out — their result index must address the full broadcast, which collapsing destroys.
            long bcastMult = 1;
            if (arr.Shape.IsBroadcasted && op != ReductionOp.ArgMax && op != ReductionOp.ArgMin)
            {
                for (int d = 0; d < arr.ndim; d++)
                    if (arr.Shape.strides[d] == 0 && arr.Shape.dimensions[d] > 1)
                        bcastMult *= arr.Shape.dimensions[d];

                arr = DropBroadcastAxes(arr);            // O(1) view of the unique data
                inputType = arr.GetTypeCode;

                if (arr.Shape.IsScalar || arr.size == 1)
                    return CombineWithCount(ExecuteScalarReduction<TResult>(arr, op, accumType), bcastMult, op);
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
                if (routed.HasValue) return CombineWithCount(routed.Value, bcastMult, op);
            }

            // Fast contiguous f64/f32 min/max: raw Avx.Min/Max + cheap finite-mask NaN tracking.
            // The IL kernel emits Vector256.Min/Max, which on net9+ the JIT lowers to vminp{s,d}
            // PLUS an IEEE NaN-propagation fixup (cmp + blend) — ~2× the cost of the raw
            // instruction. We don't need that fixup: we drop NaN in the hot loop (raw VMINPD/
            // VMAXPD) and track "any NaN seen" in a finite mask, taking a cold scalar scan for the
            // exact first-NaN bits ONLY when one is present. Measured (2000-elem rows excluded —
            // flat 10K/100K): f64 0.66×→1.42× @100K and 1.56×→4.08× @10K vs NumPy, bit-exact incl.
            // NaN/±inf/±0. Non-x86 (no Avx) falls through to the portable IL kernel below.
            if (isContiguous && (op == ReductionOp.Min || op == ReductionOp.Max)
                && inputType == accumType && Avx.IsSupported
                && (inputType == NPTypeCode.Double || inputType == NPTypeCode.Single))
            {
                byte* baseAddr = (byte*)arr.Address + arr.Shape.offset * arr.dtypesize;
                bool max = op == ReductionOp.Max;
                if (inputType == NPTypeCode.Double)
                {
                    double r = FlatMinMaxF64Avx((double*)baseAddr, arr.size, max);
                    return CombineWithCount(*(TResult*)&r, bcastMult, op);
                }
                else
                {
                    float r = FlatMinMaxF32Avx((float*)baseAddr, arr.size, max);
                    return CombineWithCount(*(TResult*)&r, bcastMult, op);
                }
            }

            // Get kernel key
            var key = new ElementReductionKernelKey(inputType, accumType, op, isContiguous);

            // Get or generate kernel
            var kernel = DirectILKernelGenerator.TryGetTypedElementReductionKernel<TResult>(key);

            if (kernel != null)
            {
                return CombineWithCount(ExecuteTypedReductionKernel<TResult>(kernel, arr), bcastMult, op);
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
        ///     Build a non-broadcast view of the UNIQUE data by indexing 0 along every stride-0
        ///     (dim&gt;1) axis. Those axes repeat their data exactly, so index 0 (offset unchanged,
        ///     axis dropped) yields the unique slice with zero copy. Size-1 and real (stride≠0)
        ///     axes are kept. The result is never broadcast, so the reduction below runs the fast
        ///     contiguous / NpyIter path over O(unique) elements.
        /// </summary>
        private static NDArray DropBroadcastAxes(NDArray arr)
        {
            var slices = new Slice[arr.ndim];
            for (int d = 0; d < arr.ndim; d++)
                slices[d] = (arr.Shape.strides[d] == 0 && arr.Shape.dimensions[d] > 1)
                    ? Slice.Index(0)
                    : Slice.All;

            // No copy: the unique slice may be strided (e.g. broadcast_to(a["1:-1,1:-1"], …)),
            // but the reduction below routes a non-contiguous input through NpyIter, which
            // handles strided/offset views correctly and fast (coalesce + axis-permute by
            // stride). The sub-word strided-prod overflow a copy used to dodge is now fixed at
            // its root (NpyIter.DetermineAccumulatorType delegates to GetAccumulatingType).
            return arr[slices];
        }

        /// <summary>
        ///     Apply a collapsed broadcast multiplicity to a flat-reduction scalar in closed form:
        ///     sum → v×count, prod → v^count, min/max/arg → v (identity). <paramref name="count"/>==1
        ///     is a no-op, so non-broadcast reductions pass straight through. Arithmetic runs in the
        ///     accumulator type so integer wraparound matches a materialized per-copy reduction
        ///     EXACTLY (modular × / pow compose under truncation); float is ULP-equivalent.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe TResult CombineWithCount<TResult>(TResult v, long count, ReductionOp op)
            where TResult : unmanaged
        {
            if (count <= 1 || op == ReductionOp.Min || op == ReductionOp.Max ||
                op == ReductionOp.ArgMax || op == ReductionOp.ArgMin)
                return v;

            bool isProd = op == ReductionOp.Prod;
            switch (InfoOf<TResult>.NPTypeCode)
            {
                case NPTypeCode.Byte:    { byte    x = *(byte*)&v;    byte    r = isProd ? (byte)   IPowU(x, count) : unchecked((byte)   (x * (ulong)count)); return *(TResult*)&r; }
                case NPTypeCode.SByte:   { sbyte   x = *(sbyte*)&v;   sbyte   r = isProd ? (sbyte)  IPowS(x, count) : unchecked((sbyte)  (x * count));         return *(TResult*)&r; }
                case NPTypeCode.Int16:   { short   x = *(short*)&v;   short   r = isProd ? (short)  IPowS(x, count) : unchecked((short)  (x * count));         return *(TResult*)&r; }
                case NPTypeCode.UInt16:  { ushort  x = *(ushort*)&v;  ushort  r = isProd ? (ushort) IPowU(x, count) : unchecked((ushort) (x * (ulong)count)); return *(TResult*)&r; }
                case NPTypeCode.Int32:   { int     x = *(int*)&v;     int     r = isProd ? (int)    IPowS(x, count) : unchecked((int)    (x * count));         return *(TResult*)&r; }
                case NPTypeCode.UInt32:  { uint    x = *(uint*)&v;    uint    r = isProd ? (uint)   IPowU(x, count) : unchecked((uint)   (x * (ulong)count)); return *(TResult*)&r; }
                case NPTypeCode.Int64:   { long    x = *(long*)&v;    long    r = isProd ? IPowS(x, count)          : unchecked(x * count);                    return *(TResult*)&r; }
                case NPTypeCode.UInt64:  { ulong   x = *(ulong*)&v;   ulong   r = isProd ? IPowU(x, count)          : unchecked(x * (ulong)count);             return *(TResult*)&r; }
                case NPTypeCode.Single:  { float   x = *(float*)&v;   float   r = isProd ? (float)Math.Pow(x, count) : (float)(x * (double)count);            return *(TResult*)&r; }
                case NPTypeCode.Double:  { double  x = *(double*)&v;  double  r = isProd ? Math.Pow(x, count)        : x * count;                              return *(TResult*)&r; }
                case NPTypeCode.Decimal: { decimal x = *(decimal*)&v; decimal r = isProd ? DPow(x, count)           : x * count;                              return *(TResult*)&r; }
                // Char/Boolean/Half/Complex only reach CombineWithCount via min/max (returned above) → identity.
                default: return v;
            }
        }

        /// <summary>Signed integer pow with two's-complement wraparound (square-and-multiply, unchecked).</summary>
        private static long IPowS(long b, long e)
        {
            long r = 1, bb = b; ulong ee = (ulong)e;
            unchecked { while (ee > 0) { if ((ee & 1) != 0) r *= bb; ee >>= 1; if (ee > 0) bb *= bb; } }
            return r;
        }

        /// <summary>Unsigned integer pow with modular wraparound (square-and-multiply, unchecked).</summary>
        private static ulong IPowU(ulong b, long e)
        {
            ulong r = 1, bb = b, ee = (ulong)e;
            unchecked { while (ee > 0) { if ((ee & 1) != 0) r *= bb; ee >>= 1; if (ee > 0) bb *= bb; } }
            return r;
        }

        /// <summary>Decimal pow (square-and-multiply); overflow throws OverflowException — decimal has no infinity, matching np.prod-on-decimal overflow behavior.</summary>
        private static decimal DPow(decimal b, long e)
        {
            decimal r = 1m, bb = b; ulong ee = (ulong)e;
            while (ee > 0) { if ((ee & 1) != 0) r *= bb; ee >>= 1; if (ee > 0) bb *= bb; }
            return r;
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

        // =====================================================================
        // Fast contiguous float min/max (Avx2): raw VMINPD/VMAXPD (NaN-dropping) hot loop +
        // per-lane finite mask; a cold scalar scan recovers the exact first-NaN bits only when a
        // NaN is actually present. Bypasses the IL kernel's Vector256.Min/Max, whose net9+ JIT
        // lowering bakes in an IEEE NaN-propagation fixup (~2× the raw instruction). Bit-exact
        // with np.min/np.max (NaN propagates with the input NaN's bits; ±inf/±0 ties match the
        // scalar fold). 8 explicit accumulators saturate the two min/max ALU ports.
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe double FlatMinMaxF64Avx(double* d, long n, bool max)
        {
            const int W = 4;                                  // Vector256<double>.Count
            if (n < W)
            {
                double s = max ? double.NegativeInfinity : double.PositiveInfinity;
                for (long k = 0; k < n; k++) { double x = d[k]; if (double.IsNaN(x)) return x; s = max ? (x > s ? x : s) : (x < s ? x : s); }
                return s;
            }
            var seed = Vector256.Create(max ? double.NegativeInfinity : double.PositiveInfinity);
            var a0 = seed; var a1 = seed; var a2 = seed; var a3 = seed; var a4 = seed; var a5 = seed; var a6 = seed; var a7 = seed;
            var fin = Vector256<double>.AllBitsSet;
            long i = 0, step = W * 8, lim = n - n % step;
            for (; i < lim; i += step)
            {
                var v0 = Vector256.Load(d + i); var v1 = Vector256.Load(d + i + W); var v2 = Vector256.Load(d + i + 2 * W); var v3 = Vector256.Load(d + i + 3 * W);
                var v4 = Vector256.Load(d + i + 4 * W); var v5 = Vector256.Load(d + i + 5 * W); var v6 = Vector256.Load(d + i + 6 * W); var v7 = Vector256.Load(d + i + 7 * W);
                if (max) { a0 = Avx.Max(a0, v0); a1 = Avx.Max(a1, v1); a2 = Avx.Max(a2, v2); a3 = Avx.Max(a3, v3); a4 = Avx.Max(a4, v4); a5 = Avx.Max(a5, v5); a6 = Avx.Max(a6, v6); a7 = Avx.Max(a7, v7); }
                else { a0 = Avx.Min(a0, v0); a1 = Avx.Min(a1, v1); a2 = Avx.Min(a2, v2); a3 = Avx.Min(a3, v3); a4 = Avx.Min(a4, v4); a5 = Avx.Min(a5, v5); a6 = Avx.Min(a6, v6); a7 = Avx.Min(a7, v7); }
                fin &= Vector256.Equals(v0, v0) & Vector256.Equals(v1, v1) & Vector256.Equals(v2, v2) & Vector256.Equals(v3, v3) & Vector256.Equals(v4, v4) & Vector256.Equals(v5, v5) & Vector256.Equals(v6, v6) & Vector256.Equals(v7, v7);
            }
            var va = max ? Avx.Max(Avx.Max(Avx.Max(a0, a1), Avx.Max(a2, a3)), Avx.Max(Avx.Max(a4, a5), Avx.Max(a6, a7)))
                         : Avx.Min(Avx.Min(Avx.Min(a0, a1), Avx.Min(a2, a3)), Avx.Min(Avx.Min(a4, a5), Avx.Min(a6, a7)));
            for (; i + W <= n; i += W) { var v = Vector256.Load(d + i); va = max ? Avx.Max(va, v) : Avx.Min(va, v); fin &= Vector256.Equals(v, v); }
            double acc = va.GetElement(0);
            for (int q = 1; q < W; q++) { double x = va.GetElement(q); acc = max ? (x > acc ? x : acc) : (x < acc ? x : acc); }
            bool anyNaN = Vector256.ExtractMostSignificantBits(fin) != (uint)((1 << W) - 1);
            for (; i < n; i++) { double x = d[i]; if (double.IsNaN(x)) anyNaN = true; acc = max ? (x > acc ? x : acc) : (x < acc ? x : acc); }
            if (anyNaN) for (long k = 0; k < n; k++) if (double.IsNaN(d[k])) return d[k];
            return acc;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float FlatMinMaxF32Avx(float* d, long n, bool max)
        {
            const int W = 8;                                  // Vector256<float>.Count
            if (n < W)
            {
                float s = max ? float.NegativeInfinity : float.PositiveInfinity;
                for (long k = 0; k < n; k++) { float x = d[k]; if (float.IsNaN(x)) return x; s = max ? (x > s ? x : s) : (x < s ? x : s); }
                return s;
            }
            var seed = Vector256.Create(max ? float.NegativeInfinity : float.PositiveInfinity);
            var a0 = seed; var a1 = seed; var a2 = seed; var a3 = seed; var a4 = seed; var a5 = seed; var a6 = seed; var a7 = seed;
            var fin = Vector256<float>.AllBitsSet;
            long i = 0, step = W * 8, lim = n - n % step;
            for (; i < lim; i += step)
            {
                var v0 = Vector256.Load(d + i); var v1 = Vector256.Load(d + i + W); var v2 = Vector256.Load(d + i + 2 * W); var v3 = Vector256.Load(d + i + 3 * W);
                var v4 = Vector256.Load(d + i + 4 * W); var v5 = Vector256.Load(d + i + 5 * W); var v6 = Vector256.Load(d + i + 6 * W); var v7 = Vector256.Load(d + i + 7 * W);
                if (max) { a0 = Avx.Max(a0, v0); a1 = Avx.Max(a1, v1); a2 = Avx.Max(a2, v2); a3 = Avx.Max(a3, v3); a4 = Avx.Max(a4, v4); a5 = Avx.Max(a5, v5); a6 = Avx.Max(a6, v6); a7 = Avx.Max(a7, v7); }
                else { a0 = Avx.Min(a0, v0); a1 = Avx.Min(a1, v1); a2 = Avx.Min(a2, v2); a3 = Avx.Min(a3, v3); a4 = Avx.Min(a4, v4); a5 = Avx.Min(a5, v5); a6 = Avx.Min(a6, v6); a7 = Avx.Min(a7, v7); }
                fin &= Vector256.Equals(v0, v0) & Vector256.Equals(v1, v1) & Vector256.Equals(v2, v2) & Vector256.Equals(v3, v3) & Vector256.Equals(v4, v4) & Vector256.Equals(v5, v5) & Vector256.Equals(v6, v6) & Vector256.Equals(v7, v7);
            }
            var va = max ? Avx.Max(Avx.Max(Avx.Max(a0, a1), Avx.Max(a2, a3)), Avx.Max(Avx.Max(a4, a5), Avx.Max(a6, a7)))
                         : Avx.Min(Avx.Min(Avx.Min(a0, a1), Avx.Min(a2, a3)), Avx.Min(Avx.Min(a4, a5), Avx.Min(a6, a7)));
            for (; i + W <= n; i += W) { var v = Vector256.Load(d + i); va = max ? Avx.Max(va, v) : Avx.Min(va, v); fin &= Vector256.Equals(v, v); }
            float acc = va.GetElement(0);
            for (int q = 1; q < W; q++) { float x = va.GetElement(q); acc = max ? (x > acc ? x : acc) : (x < acc ? x : acc); }
            bool anyNaN = Vector256.ExtractMostSignificantBits(fin) != (uint)((1 << W) - 1);
            for (; i < n; i++) { float x = d[i]; if (float.IsNaN(x)) anyNaN = true; acc = max ? (x > acc ? x : acc) : (x < acc ? x : acc); }
            if (anyNaN) for (long k = 0; k < n; k++) if (float.IsNaN(d[k])) return d[k];
            return acc;
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
        /// Fallback product for Half (double accumulator for precision). Contiguous buffers take
        /// the boxing-free pointer scan (no AsIterator dispatch, ~Nx); empty/non-contig keep the
        /// iterator. Matches NumPy: product of empty array is 1.0.
        /// </summary>
        private unsafe object ProdElementwiseHalfFallback(NDArray arr)
        {
            if (TryHalfAccumulateContiguous(arr, isProd: true, out double p))
                return (Half)p;

            // non-contiguous → NpyIter chunked path (no per-element AsIterator dispatch)
            return (Half)HalfReduceViaNpyIter(arr, isProd: true);
        }

        /// <summary>
        /// Fallback product for Complex via NpyIter's chunked EXTERNAL_LOOP (see
        /// <see cref="ComplexReduceViaNpyIter"/>).
        /// </summary>
        private object ProdElementwiseComplexFallback(NDArray arr)
            => ComplexReduceViaNpyIter(arr, isProd: true);

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

            // non-contiguous → NpyIter chunked path (no per-element AsIterator dispatch)
            return HalfMinMaxViaNpyIter(arr, isMin: false);
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

            // non-contiguous → NpyIter chunked path (no per-element AsIterator dispatch)
            return HalfMinMaxViaNpyIter(arr, isMin: true);
        }

        /// <summary>
        /// Fallback max/min for Complex: NumPy uses lexicographic comparison (real first, imag as
        /// tie-break). A NaN in either component propagates — the FIRST NaN-bearing element (in
        /// memory/iteration order) is returned VERBATIM, matching NumPy's minimum/maximum, which
        /// return the NaN operand as-is (e.g. min([1+1j, nan+0j, 2+2j]) -> (nan,0), not (nan,nan)).
        ///
        /// Routed through <see cref="ComplexMinMaxViaNpyIter"/> (struct-generic ExecuteReducing,
        /// NPY_KEEPORDER) instead of the old per-element AsIterator: ~3.6× contiguous / ~8.6×
        /// strided, AND a NumPy-parity fix — KEEPORDER visits elements in MEMORY order, so the NaN
        /// element returned for a non-C-contiguous (e.g. transposed) array now matches NumPy's
        /// reduce, which also propagates the memory-order-first NaN. The old AsIterator forced
        /// C-order and returned the wrong NaN element for transposed multi-NaN inputs
        /// (np.min(a.T) where a=[[1+1j,nan+5j,3+3j],[nan+7j,2+2j,4+4j]]: NumPy → (nan,5), old → (nan,7)).
        /// </summary>
        private unsafe object MaxElementwiseComplexFallback(NDArray arr)
            => ComplexMinMaxViaNpyIter(arr, isMin: false);

        private unsafe object MinElementwiseComplexFallback(NDArray arr)
            => ComplexMinMaxViaNpyIter(arr, isMin: true);

        /// <summary>
        /// Complex min/max via NpyIter's chunked EXTERNAL_LOOP in NPY_KEEPORDER. The accumulator's
        /// Best holds either the lexicographic extremum or — once a NaN-bearing element is seen —
        /// that element verbatim (the kernel aborts on the first NaN). Empty → (0,0), preserving the
        /// prior AsIterator fallback (np.max/np.min of an empty array is guarded upstream).
        /// </summary>
        private static unsafe System.Numerics.Complex ComplexMinMaxViaNpyIter(NDArray arr, bool isMin)
        {
            if (arr.size == 0) return System.Numerics.Complex.Zero;
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
            var acc = isMin
                ? iter.ExecuteReducing<ComplexMinKernel, ComplexMinMaxAccumulator>(default, default)
                : iter.ExecuteReducing<ComplexMaxKernel, ComplexMinMaxAccumulator>(default, default);
            return acc.Best;
        }

        /// <summary>
        ///     Boolean max == "any true" (logical OR). NumPy parity:
        ///     <c>np.max([T,F,T])</c> → <c>True</c>. Delegates to <see cref="Any(NDArray)"/>
        ///     (the SIMD ReduceBool path, contig + strided) instead of a per-element
        ///     AsIterator scan. Empty → False (np.max guards empty upstream).
        /// </summary>
        private object MaxElementwiseBooleanFallback(NDArray arr)
            => Any(arr);

        /// <summary>
        ///     Boolean min == "all true" (logical AND). NumPy parity:
        ///     <c>np.min([T,F,T])</c> → <c>False</c>. Delegates to <see cref="All(NDArray)"/>
        ///     (the SIMD ReduceBool path, contig + strided) instead of a per-element
        ///     AsIterator scan. Empty → True (np.min guards empty upstream).
        /// </summary>
        private object MinElementwiseBooleanFallback(NDArray arr)
            => All(arr);

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
        /// Fallback argmax/argmin for Half (the IL kernel uses OpCodes.Bgt/Blt which don't apply to
        /// the Half struct). NumPy: first occurrence of the extremum; NaN propagates (argmax/argmin
        /// of an array containing NaN returns the index of the first NaN).
        ///
        /// Routed through struct-generic ExecuteReducing with a running C-order index (~2× the old
        /// per-element AsIterator). NPY_CORDER (NOT the KEEPORDER default) is mandatory: the
        /// returned index must be the C-order flat position even for transposed/strided views.
        /// </summary>
        private unsafe long ArgMaxHalfFallback(NDArray arr)
        {
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
            var a = iter.ExecuteReducing<HalfArgMaxKernel, HalfArgAccumulator>(
                default, new HalfArgAccumulator { BestIdx = -1, Cur = 0, SawNaNIdx = -1 });
            return a.SawNaNIdx >= 0 ? a.SawNaNIdx : a.BestIdx;
        }

        private unsafe long ArgMinHalfFallback(NDArray arr)
        {
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
            var a = iter.ExecuteReducing<HalfArgMinKernel, HalfArgAccumulator>(
                default, new HalfArgAccumulator { BestIdx = -1, Cur = 0, SawNaNIdx = -1 });
            return a.SawNaNIdx >= 0 ? a.SawNaNIdx : a.BestIdx;
        }

        /// <summary>
        /// Fallback argmax/argmin for Complex using lexicographic comparison (real, then imag).
        /// Returns the C-order flat index of the first occurrence of the extremum (NumPy tiebreak
        /// semantics). NaN propagates: a Complex value with NaN in either component "wins" at its
        /// first occurrence.
        ///
        /// Routed through struct-generic ExecuteReducing with a running C-order index (~2× the old
        /// per-element AsIterator). The iterator is built with NPY_CORDER (NOT the KEEPORDER
        /// default): the returned index must be the C-order position, so traversal must follow
        /// C-order even on transposed/strided views — matching the contract the IL arg kernels
        /// already honor for the other dtypes, and NumPy (argmax(a.T) returns a C-order index).
        /// </summary>
        private unsafe long ArgMaxComplexFallback(NDArray arr)
        {
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
            var a = iter.ExecuteReducing<ComplexArgMaxKernel, ComplexArgAccumulator>(
                default, new ComplexArgAccumulator { BestIdx = -1, Cur = 0, SawNaNIdx = -1 });
            return a.SawNaNIdx >= 0 ? a.SawNaNIdx : a.BestIdx;
        }

        private unsafe long ArgMinComplexFallback(NDArray arr)
        {
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP,
                NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
            var a = iter.ExecuteReducing<ComplexArgMinKernel, ComplexArgAccumulator>(
                default, new ComplexArgAccumulator { BestIdx = -1, Cur = 0, SawNaNIdx = -1 });
            return a.SawNaNIdx >= 0 ? a.SawNaNIdx : a.BestIdx;
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

            // Handle Complex separately - mean is Complex, not double.
            // Reuse the one-pass SIMD Complex sum (ComplexReduceViaNpyIter, the
            // same path np.sum takes) then divide — unifies the Complex sum path
            // and picks up the Vector256 contiguous-chunk kernel. mean of a
            // single element is handled by the scalar guard above.
            if (sumType == NPTypeCode.Complex)
            {
                var sum = ComplexReduceViaNpyIter(arr, isProd: false);
                return sum / count;
            }

            // Handle Half separately - NumPy 2.x preserves float16 dtype for mean.
            // NumPy upcasts float16 for the mean accumulation. Accumulating the sum in Half
            // (the previous ExecuteElementReduction<Half> path) overflowed to garbage on large
            // arrays — mean([2.5]×100k) returned 0.08 instead of 2.5 — and was also the slowest
            // half reduction (~32 ms/4M). Accumulate in double (the precision NumPy uses for the
            // f16 mean), then narrow: correct AND faster.
            if (sumType == NPTypeCode.Half)
            {
                if (!TryHalfAccumulateContiguous(arr, isProd: false, out double hsum))
                    hsum = HalfReduceViaNpyIter(arr, isProd: false);
                return (Half)(hsum / count);
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
        /// Fallback sum for Half (double accumulator). Contiguous buffers take the boxing-free
        /// pointer scan (no AsIterator dispatch, ~Nx); empty/non-contig keep the iterator. The
        /// double accumulate then narrows to Half — so a large sum saturates to ±inf exactly like
        /// NumPy's float16 reduce (e.g. sum([2.5]×100k) → inf).
        /// </summary>
        private unsafe object SumElementwiseHalfFallback(NDArray arr)
        {
            if (TryHalfAccumulateContiguous(arr, isProd: false, out double s))
                return (Half)s;

            // non-contiguous → NpyIter chunked path (no per-element AsIterator dispatch)
            return (Half)HalfReduceViaNpyIter(arr, isProd: false);
        }

        /// <summary>
        /// Boxing-free contiguous Half sum/product accumulated in <see cref="double"/> — the
        /// precision NumSharp's f16 reductions already use (and NumPy uses for the f16 mean). Half
        /// has no Vector&lt;Half&gt; in the BCL, but reading the raw buffer and widening each element
        /// avoids the legacy NDIterator's virtual MoveNext/HasNext per element, which dominated
        /// these reductions (sum/prod ~14×, mean ~53× the double path). Returns false for empty or
        /// non-contiguous inputs so the caller keeps its iterator path. Offset-contiguous (sliced)
        /// views are honored via <see cref="Shape.offset"/>.
        /// </summary>
        private static unsafe bool TryHalfAccumulateContiguous(NDArray arr, bool isProd, out double result)
        {
            long n = arr.size;
            if (!arr.Shape.IsContiguous || n <= 0)
            {
                result = isProd ? 1.0 : 0.0;
                return false;
            }

            Half* p = (Half*)((byte*)arr.Address + arr.Shape.offset * 2);
            double acc = isProd ? 1.0 : 0.0;
            if (isProd)
                for (long i = 0; i < n; i++) acc *= (double)p[i];
            else
                for (long i = 0; i < n; i++) acc += (double)p[i];
            result = acc;
            return true;
        }

        /// <summary>
        /// Non-contiguous Half sum/product via NpyIter's chunked EXTERNAL_LOOP
        /// (<see cref="NpyIterRef.ForEach"/>), accumulated in double. Replaces the per-element
        /// AsIterator&lt;Half&gt; tail: NpyIter normalizes the strided/transposed layout to
        /// contiguous-inner chunks and amortizes iterator dispatch over each chunk — ~2.6× the
        /// per-element NDIterator on strided views. sum/prod are commutative, so the permuted
        /// traversal order is value-equivalent (modulo float rounding, which the f16 fuzz tolerates).
        /// Contiguous inputs take <see cref="TryHalfAccumulateContiguous"/> upstream.
        /// </summary>
        private static unsafe double HalfReduceViaNpyIter(NDArray arr, bool isProd)
        {
            double acc = isProd ? 1.0 : 0.0;
            if (arr.size == 0) return acc;

            // Struct-generic ExecuteReducing — the accumulator stays in a register
            // instead of the per-element *aux memory round-trip the ForEach delegate
            // forced (~1.6× on 4M, both layouts). Half has no Vector<Half> in the BCL
            // and the f16→f64 widen is the wall, so there is no SIMD/unroll win here;
            // the gain is purely devirtualization + register accumulation. sum/prod
            // are commutative ⇒ KEEPORDER's permuted traversal is value-equivalent
            // modulo ULP. Contiguous inputs take TryHalfAccumulateContiguous upstream.
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
            return isProd
                ? iter.ExecuteReducing<HalfProdKernel, double>(default, acc)
                : iter.ExecuteReducing<HalfSumKernel, double>(default, acc);
        }

        /// <summary>
        /// Non-contiguous Half min/max via NpyIter's chunked EXTERNAL_LOOP. The min/max VALUE is
        /// order-independent and NaN propagation (any NaN → NaN) is too, so NpyIter's permuted
        /// traversal is safe here — unlike argmin/argmax, whose returned index would shift under the
        /// permutation. Empty → ±inf (matching the prior iterator fallback); any NaN → NaN.
        /// Contiguous inputs take the direct-pointer scan upstream.
        ///
        /// Struct-generic ExecuteReducing (HalfMaxKernel/HalfMinKernel) replaces the ForEach
        /// delegate: a 4-accumulator unroll on the contiguous inner chunk breaks the per-element
        /// dependency chain (~1.3× on 4M) and the JIT inlines the kernel.
        /// </summary>
        private static unsafe object HalfMinMaxViaNpyIter(NDArray arr, bool isMin)
        {
            if (arr.size == 0)
                return isMin ? Half.PositiveInfinity : Half.NegativeInfinity;

            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
            var acc = isMin
                ? iter.ExecuteReducing<HalfMinKernel, HalfMinMaxAccumulator>(default, default)
                : iter.ExecuteReducing<HalfMaxKernel, HalfMinMaxAccumulator>(default, default);

            if (acc.SawNaN) return Half.NaN;
            if (!acc.Seen) return isMin ? Half.PositiveInfinity : Half.NegativeInfinity;
            return (Half)acc.Best;
        }

        /// <summary>
        /// Complex sum/product via NpyIter's chunked EXTERNAL_LOOP — replaces the per-element
        /// AsIterator&lt;Complex&gt; over EVERY layout (Complex had no contiguous fast path, so this
        /// wins ~2.2× contiguous and ~2.6× strided). Both ops are commutative, so the permuted
        /// traversal is value-equivalent modulo rounding. Empty: sum→0, prod→1 (NumPy parity).
        /// </summary>
        private static unsafe System.Numerics.Complex ComplexReduceViaNpyIter(NDArray arr, bool isProd)
        {
            var acc = isProd ? System.Numerics.Complex.One : System.Numerics.Complex.Zero;
            if (arr.size == 0) return acc;

            // Struct-generic ExecuteReducing (devirtualized + inlined; the
            // accumulator stays in a register instead of the per-element memory
            // round-trip the ForEach delegate forced, and sum adds a
            // Vector256<double> contiguous-chunk fast path) — measured ~2.4×
            // (sum) / ~1.5× (prod) the prior ForEach(delegate) fold on 4M
            // elements, both layouts. KEEPORDER (NpyIterRef.New default) keeps
            // the inner chunk contiguous for transposed views too. Both ops are
            // commutative, so the permuted traversal is value-equivalent modulo
            // ULP-level rounding (same class as the codebase's pairwise sums).
            using var iter = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP);
            return isProd
                ? iter.ExecuteReducing<ComplexProdKernel, System.Numerics.Complex>(default, acc)
                : iter.ExecuteReducing<ComplexSumKernel, System.Numerics.Complex>(default, acc);
        }

        /// <summary>
        /// Fallback sum for Complex type via NpyIter's chunked EXTERNAL_LOOP (see
        /// <see cref="ComplexReduceViaNpyIter"/>).
        /// </summary>
        private object SumElementwiseComplexFallback(NDArray arr)
            => ComplexReduceViaNpyIter(arr, isProd: false);

        #endregion
    }
}
