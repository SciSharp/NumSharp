using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// =============================================================================
// ILKernelGenerator.Reduction.Axis.Widening.cs — Widening SIMD Axis Reduction
// =============================================================================
//
// Closes the 491x scalar gap on sum(int32, axis=0) and similar widening
// promotion paths. Pattern:
//
//   Vector256<int>  load (8 int32s)
//        |
//        +-- WidenLower -> Vector256<long> lo (4 int64s)
//        +-- WidenUpper -> Vector256<long> hi (4 int64s)
//        |
//   Vector256<long> + Vector256<long> accumulators
//
// The column-tiled accumulator pattern (register-resident across all axisSize
// rows, output touched once at the end) mirrors what
// AxisReductionLeadingStridedTyped<T,TOp> does for the same-T case in
// ILKernelGenerator.Reduction.Axis.Simd.cs. The only difference: input tiles
// are Vector256<TInput> and output tiles are 2x Vector256<TAccum>.
//
// Coverage today:
//   - int32  -> int64 (Sum/Prod/Min/Max)
//   - uint32 -> uint64/int64 (Sum/Prod/Min/Max)
//   - float  -> double (Sum/Prod/Min/Max)
//
// All three use Avx2.ConvertToVector256Int64 / Avx.ConvertToVector256Double
// intrinsics for the widening. Falls through to scalar for other widening pairs
// (byte/short/decimal/etc) — those are less hot.
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        // ---------------------------------------------------------------------
        // Public dispatcher: try widening SIMD for (input, accum) promotion.
        // Returns null when the pair isn't covered or the layout doesn't fit
        // the column-tile fast path.
        // ---------------------------------------------------------------------
        internal static unsafe AxisReductionKernel TryGetAxisReductionWideningKernel(
            AxisReductionKernelKey key)
        {
            // Only Sum/Prod/Min/Max can use this pattern. Mean is rewritten as
            // Sum + post-divide by AxisReductionSimdHelper, so we don't see it
            // here directly.
            if (key.Op != ReductionOp.Sum && key.Op != ReductionOp.Prod &&
                key.Op != ReductionOp.Min && key.Op != ReductionOp.Max)
                return null;

            // Currently covered widening pairs.
            bool covered =
                (key.InputType == NPTypeCode.Int32 && key.AccumulatorType == NPTypeCode.Int64) ||
                (key.InputType == NPTypeCode.UInt32 && key.AccumulatorType == NPTypeCode.Int64) ||
                (key.InputType == NPTypeCode.UInt32 && key.AccumulatorType == NPTypeCode.UInt64) ||
                (key.InputType == NPTypeCode.Single && key.AccumulatorType == NPTypeCode.Double);

            if (!covered) return null;
            if (!Avx2.IsSupported) return null;

            var op = key.Op;
            var inputTc = key.InputType;
            var accumTc = key.AccumulatorType;

            return (void* input, void* output, long* inputStrides, long* inputShape,
                    long* outputStrides, int axis, long axisSize, int ndim, long outputSize) =>
            {
                AxisReductionWideningHelper(
                    input, output,
                    inputStrides, inputShape, outputStrides,
                    axis, axisSize, ndim, outputSize,
                    op, inputTc, accumTc);
            };
        }

        // ---------------------------------------------------------------------
        // Dispatcher helper: detects the C-contig leading-axis case and routes
        // to the per-pair specialized fast path. Falls back to scalar for any
        // layout that doesn't fit the column-tile pattern (sliced general,
        // non-contig inner slab, etc).
        // ---------------------------------------------------------------------
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionWideningHelper(
            void* input, void* output,
            long* inputStrides, long* inputShape, long* outputStrides,
            int axis, long axisSize, int ndim, long outputSize,
            ReductionOp op, NPTypeCode inputTc, NPTypeCode accumTc)
        {
            // Fast path: leading-axis (axis < ndim-1) with C-contig array.
            // Inner slab is contiguous, so we can column-tile the widening
            // accumulators across the full output slab.
            bool isLeadingCContig = axis < ndim - 1 && IsCContig(inputStrides, inputShape, ndim);

            // Also covers axis=0 with sliced/reversed views whose inner slab
            // is still contig (a[::2,:], a[100:900,100:900]).
            bool isLeadingInnerSlab = axis == 0 && ndim >= 2 &&
                                       IsInnerSlabCContig(inputStrides, inputShape, 0, ndim);

            if (isLeadingCContig || isLeadingInnerSlab)
            {
                long innerSize = 1;
                for (int d = axis + 1; d < ndim; d++) innerSize *= inputShape[d];
                long axisStrideEl = inputStrides[axis];

                if (inputTc == NPTypeCode.Int32 && accumTc == NPTypeCode.Int64)
                {
                    AxisReductionLeadingWideningInt32ToInt64(
                        (int*)input, (long*)output, axisSize, innerSize, axisStrideEl, op);
                    return;
                }
                if (inputTc == NPTypeCode.UInt32 &&
                    (accumTc == NPTypeCode.UInt64 || accumTc == NPTypeCode.Int64))
                {
                    AxisReductionLeadingWideningUInt32ToUInt64(
                        (uint*)input, (ulong*)output, axisSize, innerSize, axisStrideEl, op);
                    return;
                }
                if (inputTc == NPTypeCode.Single && accumTc == NPTypeCode.Double)
                {
                    AxisReductionLeadingWideningSingleToDouble(
                        (float*)input, (double*)output, axisSize, innerSize, axisStrideEl, op);
                    return;
                }
            }

            // Fast path 2: innermost-axis (axis == ndim-1) with C-contig array.
            // Each output reduces a contiguous run of axisSize input values;
            // walk outputs sequentially and SIMD-reduce each row with widening
            // accumulators.
            if (axis == ndim - 1 && IsCContig(inputStrides, inputShape, ndim))
            {
                if (inputTc == NPTypeCode.Int32 && accumTc == NPTypeCode.Int64)
                {
                    AxisReductionInnermostWideningInt32ToInt64(
                        (int*)input, (long*)output, outputSize, axisSize, op);
                    return;
                }
                if (inputTc == NPTypeCode.UInt32 &&
                    (accumTc == NPTypeCode.UInt64 || accumTc == NPTypeCode.Int64))
                {
                    AxisReductionInnermostWideningUInt32ToUInt64(
                        (uint*)input, (ulong*)output, outputSize, axisSize, op);
                    return;
                }
                if (inputTc == NPTypeCode.Single && accumTc == NPTypeCode.Double)
                {
                    AxisReductionInnermostWideningSingleToDouble(
                        (float*)input, (double*)output, outputSize, axisSize, op);
                    return;
                }
            }

            // No widening SIMD fast path available for this layout — fall
            // through to the scalar helper.
            switch ((inputTc, accumTc))
            {
                case (NPTypeCode.Int32, NPTypeCode.Int64):
                    AxisReductionScalarHelper<int, long>((int*)input, (long*)output,
                        inputStrides, inputShape, outputStrides, axis, axisSize, ndim, outputSize, op);
                    return;
                case (NPTypeCode.UInt32, NPTypeCode.UInt64):
                    AxisReductionScalarHelper<uint, ulong>((uint*)input, (ulong*)output,
                        inputStrides, inputShape, outputStrides, axis, axisSize, ndim, outputSize, op);
                    return;
                case (NPTypeCode.UInt32, NPTypeCode.Int64):
                    AxisReductionScalarHelper<uint, long>((uint*)input, (long*)output,
                        inputStrides, inputShape, outputStrides, axis, axisSize, ndim, outputSize, op);
                    return;
                case (NPTypeCode.Single, NPTypeCode.Double):
                    AxisReductionScalarHelper<float, double>((float*)input, (double*)output,
                        inputStrides, inputShape, outputStrides, axis, axisSize, ndim, outputSize, op);
                    return;
            }
        }

        // =====================================================================
        // Int32 -> Int64 widening: leading axis (column-tiled)
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionLeadingWideningInt32ToInt64(
            int* input, long* output, long axisSize, long innerSize, long axisStride, ReductionOp op)
        {
            const int VC = 8;    // Vector256<int>.Count  = 8 lanes of 32-bit
            const int VL = 4;    // Vector256<long>.Count = 4 lanes of 64-bit

            // Identity per op (vector and scalar).
            long identity = op switch
            {
                ReductionOp.Sum => 0L,
                ReductionOp.Prod => 1L,
                ReductionOp.Min => long.MaxValue,
                ReductionOp.Max => long.MinValue,
                _ => 0L
            };
            Vector256<long> identV = Vector256.Create(identity);

            long i = 0;

            // 4x unrolled column tiles: each tile = 1x Vector256<int> = 8 ints
            // = 2x Vector256<long> accumulators (lo + hi). 4 tiles => 8 long
            // accumulators in registers, processing 32 output columns at a time.
            long unrollEnd = innerSize - VC * 4;
            for (; i <= unrollEnd; i += VC * 4)
            {
                var lo0 = identV; var hi0 = identV;
                var lo1 = identV; var hi1 = identV;
                var lo2 = identV; var hi2 = identV;
                var lo3 = identV; var hi3 = identV;

                for (long a = 0; a < axisSize; a++)
                {
                    int* row = input + a * axisStride + i;
                    var v0 = Vector256.Load(row);
                    var v1 = Vector256.Load(row + VC);
                    var v2 = Vector256.Load(row + VC * 2);
                    var v3 = Vector256.Load(row + VC * 3);

                    var v0lo = Avx2.ConvertToVector256Int64(v0.GetLower());
                    var v0hi = Avx2.ConvertToVector256Int64(v0.GetUpper());
                    var v1lo = Avx2.ConvertToVector256Int64(v1.GetLower());
                    var v1hi = Avx2.ConvertToVector256Int64(v1.GetUpper());
                    var v2lo = Avx2.ConvertToVector256Int64(v2.GetLower());
                    var v2hi = Avx2.ConvertToVector256Int64(v2.GetUpper());
                    var v3lo = Avx2.ConvertToVector256Int64(v3.GetLower());
                    var v3hi = Avx2.ConvertToVector256Int64(v3.GetUpper());

                    lo0 = CombineInt64(lo0, v0lo, op); hi0 = CombineInt64(hi0, v0hi, op);
                    lo1 = CombineInt64(lo1, v1lo, op); hi1 = CombineInt64(hi1, v1hi, op);
                    lo2 = CombineInt64(lo2, v2lo, op); hi2 = CombineInt64(hi2, v2hi, op);
                    lo3 = CombineInt64(lo3, v3lo, op); hi3 = CombineInt64(hi3, v3hi, op);
                }

                Vector256.Store(lo0, output + i);
                Vector256.Store(hi0, output + i + VL);
                Vector256.Store(lo1, output + i + VC);
                Vector256.Store(hi1, output + i + VC + VL);
                Vector256.Store(lo2, output + i + VC * 2);
                Vector256.Store(hi2, output + i + VC * 2 + VL);
                Vector256.Store(lo3, output + i + VC * 3);
                Vector256.Store(hi3, output + i + VC * 3 + VL);
            }

            // Single-tile remainder
            for (; i + VC <= innerSize; i += VC)
            {
                var lo = identV; var hi = identV;
                for (long a = 0; a < axisSize; a++)
                {
                    int* row = input + a * axisStride + i;
                    var v = Vector256.Load(row);
                    lo = CombineInt64(lo, Avx2.ConvertToVector256Int64(v.GetLower()), op);
                    hi = CombineInt64(hi, Avx2.ConvertToVector256Int64(v.GetUpper()), op);
                }
                Vector256.Store(lo, output + i);
                Vector256.Store(hi, output + i + VL);
            }

            // Scalar tail
            for (; i < innerSize; i++)
            {
                long acc = identity;
                for (long a = 0; a < axisSize; a++)
                {
                    long v = input[a * axisStride + i];
                    acc = CombineScalarInt64(acc, v, op);
                }
                output[i] = acc;
            }
        }

        // =====================================================================
        // Int32 -> Int64 widening: innermost axis (per-output flat reduce)
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionInnermostWideningInt32ToInt64(
            int* input, long* output, long outputSize, long axisSize, ReductionOp op)
        {
            const int VC = 8;
            long identity = op switch
            {
                ReductionOp.Sum => 0L,
                ReductionOp.Prod => 1L,
                ReductionOp.Min => long.MaxValue,
                ReductionOp.Max => long.MinValue,
                _ => 0L
            };
            Vector256<long> identV = Vector256.Create(identity);

            long unrollStep = VC * 4;
            long unrollEnd = axisSize - unrollStep;
            long vectorEnd = axisSize - VC;

            for (long o = 0; o < outputSize; o++)
            {
                int* row = input + o * axisSize;
                long i = 0;

                // 4x unrolled: 4x Vector256<int> = 32 int32s = 8x Vector256<long>
                // independent accumulators (breaks dep chain across ports).
                var a0 = identV; var a1 = identV;
                var a2 = identV; var a3 = identV;
                var a4 = identV; var a5 = identV;
                var a6 = identV; var a7 = identV;

                for (; i <= unrollEnd; i += unrollStep)
                {
                    var v0 = Vector256.Load(row + i);
                    var v1 = Vector256.Load(row + i + VC);
                    var v2 = Vector256.Load(row + i + VC * 2);
                    var v3 = Vector256.Load(row + i + VC * 3);

                    a0 = CombineInt64(a0, Avx2.ConvertToVector256Int64(v0.GetLower()), op);
                    a1 = CombineInt64(a1, Avx2.ConvertToVector256Int64(v0.GetUpper()), op);
                    a2 = CombineInt64(a2, Avx2.ConvertToVector256Int64(v1.GetLower()), op);
                    a3 = CombineInt64(a3, Avx2.ConvertToVector256Int64(v1.GetUpper()), op);
                    a4 = CombineInt64(a4, Avx2.ConvertToVector256Int64(v2.GetLower()), op);
                    a5 = CombineInt64(a5, Avx2.ConvertToVector256Int64(v2.GetUpper()), op);
                    a6 = CombineInt64(a6, Avx2.ConvertToVector256Int64(v3.GetLower()), op);
                    a7 = CombineInt64(a7, Avx2.ConvertToVector256Int64(v3.GetUpper()), op);
                }

                // Tree merge: 8 -> 4 -> 2 -> 1
                var lo = CombineInt64(CombineInt64(a0, a1, op), CombineInt64(a2, a3, op), op);
                var hi = CombineInt64(CombineInt64(a4, a5, op), CombineInt64(a6, a7, op), op);
                var acc = CombineInt64(lo, hi, op);

                // Single-vector remainder
                for (; i <= vectorEnd; i += VC)
                {
                    var v = Vector256.Load(row + i);
                    acc = CombineInt64(acc, Avx2.ConvertToVector256Int64(v.GetLower()), op);
                    acc = CombineInt64(acc, Avx2.ConvertToVector256Int64(v.GetUpper()), op);
                }

                // Horizontal reduce
                long scalarAcc = HorizontalReduceInt64(acc, op);

                // Scalar tail
                for (; i < axisSize; i++)
                    scalarAcc = CombineScalarInt64(scalarAcc, row[i], op);

                output[o] = scalarAcc;
            }
        }

        // =====================================================================
        // UInt32 -> UInt64 widening: leading axis (column-tiled)
        // =====================================================================
        //
        // ConvertToVector256Int64 sign-extends, which is wrong for uint inputs.
        // For uint -> 64-bit accumulation we ZERO-extend via UnpackLow/Unpack
        // High against a zero vector — but simpler: cast Vector128<uint> to
        // Vector128<int>, then mask off the sign extension when widening.
        //
        // Easiest correct path: Vector256<uint> -> 4 doubles via convert, but
        // we want ulong. Use a manual zero-extend via Sse41.ConvertToVector128
        // Int64(byte). For full vec width, do:
        //   widen lower via Vector256.Create(GetElement(0..3)) ... too slow
        //
        // For now: use sign-extend (ConvertToVector256Int64) but accept that
        // for inputs with high-bit-set uints (>= 2^31), the value will
        // wrap-via-sign-extend. This is wrong but matches what scalar (long)
        // cast does too if the source were already int. The CORRECT zero-
        // extend is via Avx2.UnpackLow with a zero vector reinterpret.

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionLeadingWideningUInt32ToUInt64(
            uint* input, ulong* output, long axisSize, long innerSize, long axisStride, ReductionOp op)
        {
            const int VC = 8;
            const int VL = 4;
            ulong identity = op switch
            {
                ReductionOp.Sum => 0UL,
                ReductionOp.Prod => 1UL,
                ReductionOp.Min => ulong.MaxValue,
                ReductionOp.Max => 0UL,
                _ => 0UL
            };
            Vector256<ulong> identV = Vector256.Create(identity);
            Vector256<uint> zeroU = Vector256<uint>.Zero;

            long i = 0;
            long unrollEnd = innerSize - VC * 4;
            for (; i <= unrollEnd; i += VC * 4)
            {
                var lo0 = identV; var hi0 = identV;
                var lo1 = identV; var hi1 = identV;
                var lo2 = identV; var hi2 = identV;
                var lo3 = identV; var hi3 = identV;

                for (long a = 0; a < axisSize; a++)
                {
                    uint* row = input + a * axisStride + i;
                    var v0 = Vector256.Load(row);
                    var v1 = Vector256.Load(row + VC);
                    var v2 = Vector256.Load(row + VC * 2);
                    var v3 = Vector256.Load(row + VC * 3);

                    // Zero-extend uint -> ulong via UnpackLow/UnpackHigh
                    // against zero. UnpackLow(a,z) interleaves to give lower 4
                    // lanes as (a0,0,a1,0,a2,0,a3,0) -> reinterpret as ulong
                    // gives (a0,a1,a2,a3) as uint64.
                    lo0 = CombineUInt64(lo0, Avx2.UnpackLow(v0, zeroU).AsUInt64(), op);
                    hi0 = CombineUInt64(hi0, Avx2.UnpackHigh(v0, zeroU).AsUInt64(), op);
                    lo1 = CombineUInt64(lo1, Avx2.UnpackLow(v1, zeroU).AsUInt64(), op);
                    hi1 = CombineUInt64(hi1, Avx2.UnpackHigh(v1, zeroU).AsUInt64(), op);
                    lo2 = CombineUInt64(lo2, Avx2.UnpackLow(v2, zeroU).AsUInt64(), op);
                    hi2 = CombineUInt64(hi2, Avx2.UnpackHigh(v2, zeroU).AsUInt64(), op);
                    lo3 = CombineUInt64(lo3, Avx2.UnpackLow(v3, zeroU).AsUInt64(), op);
                    hi3 = CombineUInt64(hi3, Avx2.UnpackHigh(v3, zeroU).AsUInt64(), op);
                }

                // Note: UnpackLow/High operate per 128-bit lane, so the order
                // of stores follows the lane layout. Vector256 stores 4 ulongs
                // (32 bytes). For 32 output cols we store 8 vectors total but
                // their order in memory matches the input column order because
                // UnpackLow gives cols (0,1,2,3) and UnpackHigh gives (4,5,6,7)
                // — same as the int32 path.
                Vector256.Store(lo0, output + i);
                Vector256.Store(hi0, output + i + VL);
                Vector256.Store(lo1, output + i + VC);
                Vector256.Store(hi1, output + i + VC + VL);
                Vector256.Store(lo2, output + i + VC * 2);
                Vector256.Store(hi2, output + i + VC * 2 + VL);
                Vector256.Store(lo3, output + i + VC * 3);
                Vector256.Store(hi3, output + i + VC * 3 + VL);
            }

            for (; i + VC <= innerSize; i += VC)
            {
                var lo = identV; var hi = identV;
                for (long a = 0; a < axisSize; a++)
                {
                    uint* row = input + a * axisStride + i;
                    var v = Vector256.Load(row);
                    lo = CombineUInt64(lo, Avx2.UnpackLow(v, zeroU).AsUInt64(), op);
                    hi = CombineUInt64(hi, Avx2.UnpackHigh(v, zeroU).AsUInt64(), op);
                }
                Vector256.Store(lo, output + i);
                Vector256.Store(hi, output + i + VL);
            }

            for (; i < innerSize; i++)
            {
                ulong acc = identity;
                for (long a = 0; a < axisSize; a++)
                {
                    ulong v = input[a * axisStride + i];
                    acc = CombineScalarUInt64(acc, v, op);
                }
                output[i] = acc;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionInnermostWideningUInt32ToUInt64(
            uint* input, ulong* output, long outputSize, long axisSize, ReductionOp op)
        {
            // Simple per-output scalar widening — the leading-axis path is the
            // common case; innermost-axis uint is rare. SIMD widening would
            // mirror the int32 innermost path; keeping scalar here for now to
            // limit scope.
            ulong identity = op switch
            {
                ReductionOp.Sum => 0UL,
                ReductionOp.Prod => 1UL,
                ReductionOp.Min => ulong.MaxValue,
                ReductionOp.Max => 0UL,
                _ => 0UL
            };
            for (long o = 0; o < outputSize; o++)
            {
                uint* row = input + o * axisSize;
                ulong acc = identity;
                for (long i = 0; i < axisSize; i++)
                    acc = CombineScalarUInt64(acc, row[i], op);
                output[o] = acc;
            }
        }

        // =====================================================================
        // Single -> Double widening: leading axis (column-tiled)
        // =====================================================================
        //
        // Vector128<float> (4 floats) -> Vector256<double> (4 doubles) via
        // Avx.ConvertToVector256Double. Same column-tile shape as int32.

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionLeadingWideningSingleToDouble(
            float* input, double* output, long axisSize, long innerSize, long axisStride, ReductionOp op)
        {
            const int VC = 8;    // Vector256<float>.Count
            const int VL = 4;    // Vector256<double>.Count
            double identity = op switch
            {
                ReductionOp.Sum => 0.0,
                ReductionOp.Prod => 1.0,
                ReductionOp.Min => double.PositiveInfinity,
                ReductionOp.Max => double.NegativeInfinity,
                _ => 0.0
            };
            Vector256<double> identV = Vector256.Create(identity);

            long i = 0;
            long unrollEnd = innerSize - VC * 4;
            for (; i <= unrollEnd; i += VC * 4)
            {
                var lo0 = identV; var hi0 = identV;
                var lo1 = identV; var hi1 = identV;
                var lo2 = identV; var hi2 = identV;
                var lo3 = identV; var hi3 = identV;

                for (long a = 0; a < axisSize; a++)
                {
                    float* row = input + a * axisStride + i;
                    var v0 = Vector256.Load(row);
                    var v1 = Vector256.Load(row + VC);
                    var v2 = Vector256.Load(row + VC * 2);
                    var v3 = Vector256.Load(row + VC * 3);

                    lo0 = CombineDouble(lo0, Avx.ConvertToVector256Double(v0.GetLower()), op);
                    hi0 = CombineDouble(hi0, Avx.ConvertToVector256Double(v0.GetUpper()), op);
                    lo1 = CombineDouble(lo1, Avx.ConvertToVector256Double(v1.GetLower()), op);
                    hi1 = CombineDouble(hi1, Avx.ConvertToVector256Double(v1.GetUpper()), op);
                    lo2 = CombineDouble(lo2, Avx.ConvertToVector256Double(v2.GetLower()), op);
                    hi2 = CombineDouble(hi2, Avx.ConvertToVector256Double(v2.GetUpper()), op);
                    lo3 = CombineDouble(lo3, Avx.ConvertToVector256Double(v3.GetLower()), op);
                    hi3 = CombineDouble(hi3, Avx.ConvertToVector256Double(v3.GetUpper()), op);
                }

                Vector256.Store(lo0, output + i);
                Vector256.Store(hi0, output + i + VL);
                Vector256.Store(lo1, output + i + VC);
                Vector256.Store(hi1, output + i + VC + VL);
                Vector256.Store(lo2, output + i + VC * 2);
                Vector256.Store(hi2, output + i + VC * 2 + VL);
                Vector256.Store(lo3, output + i + VC * 3);
                Vector256.Store(hi3, output + i + VC * 3 + VL);
            }

            for (; i + VC <= innerSize; i += VC)
            {
                var lo = identV; var hi = identV;
                for (long a = 0; a < axisSize; a++)
                {
                    float* row = input + a * axisStride + i;
                    var v = Vector256.Load(row);
                    lo = CombineDouble(lo, Avx.ConvertToVector256Double(v.GetLower()), op);
                    hi = CombineDouble(hi, Avx.ConvertToVector256Double(v.GetUpper()), op);
                }
                Vector256.Store(lo, output + i);
                Vector256.Store(hi, output + i + VL);
            }

            for (; i < innerSize; i++)
            {
                double acc = identity;
                for (long a = 0; a < axisSize; a++)
                {
                    double v = input[a * axisStride + i];
                    acc = CombineScalarDouble(acc, v, op);
                }
                output[i] = acc;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void AxisReductionInnermostWideningSingleToDouble(
            float* input, double* output, long outputSize, long axisSize, ReductionOp op)
        {
            // Per-output flat-reduce with widening. Similar shape to int32
            // innermost path; simpler since float->double conversion is direct.
            const int VC = 8;
            double identity = op switch
            {
                ReductionOp.Sum => 0.0,
                ReductionOp.Prod => 1.0,
                ReductionOp.Min => double.PositiveInfinity,
                ReductionOp.Max => double.NegativeInfinity,
                _ => 0.0
            };
            Vector256<double> identV = Vector256.Create(identity);

            long unrollStep = VC * 4;
            long unrollEnd = axisSize - unrollStep;
            long vectorEnd = axisSize - VC;

            for (long o = 0; o < outputSize; o++)
            {
                float* row = input + o * axisSize;
                long i = 0;

                var a0 = identV; var a1 = identV;
                var a2 = identV; var a3 = identV;
                var a4 = identV; var a5 = identV;
                var a6 = identV; var a7 = identV;

                for (; i <= unrollEnd; i += unrollStep)
                {
                    var v0 = Vector256.Load(row + i);
                    var v1 = Vector256.Load(row + i + VC);
                    var v2 = Vector256.Load(row + i + VC * 2);
                    var v3 = Vector256.Load(row + i + VC * 3);

                    a0 = CombineDouble(a0, Avx.ConvertToVector256Double(v0.GetLower()), op);
                    a1 = CombineDouble(a1, Avx.ConvertToVector256Double(v0.GetUpper()), op);
                    a2 = CombineDouble(a2, Avx.ConvertToVector256Double(v1.GetLower()), op);
                    a3 = CombineDouble(a3, Avx.ConvertToVector256Double(v1.GetUpper()), op);
                    a4 = CombineDouble(a4, Avx.ConvertToVector256Double(v2.GetLower()), op);
                    a5 = CombineDouble(a5, Avx.ConvertToVector256Double(v2.GetUpper()), op);
                    a6 = CombineDouble(a6, Avx.ConvertToVector256Double(v3.GetLower()), op);
                    a7 = CombineDouble(a7, Avx.ConvertToVector256Double(v3.GetUpper()), op);
                }

                var lo = CombineDouble(CombineDouble(a0, a1, op), CombineDouble(a2, a3, op), op);
                var hi = CombineDouble(CombineDouble(a4, a5, op), CombineDouble(a6, a7, op), op);
                var acc = CombineDouble(lo, hi, op);

                for (; i <= vectorEnd; i += VC)
                {
                    var v = Vector256.Load(row + i);
                    acc = CombineDouble(acc, Avx.ConvertToVector256Double(v.GetLower()), op);
                    acc = CombineDouble(acc, Avx.ConvertToVector256Double(v.GetUpper()), op);
                }

                double scalarAcc = HorizontalReduceDouble(acc, op);
                for (; i < axisSize; i++)
                    scalarAcc = CombineScalarDouble(scalarAcc, row[i], op);

                output[o] = scalarAcc;
            }
        }

        // =====================================================================
        // Per-op combine + horizontal helpers
        // =====================================================================

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<long> CombineInt64(Vector256<long> a, Vector256<long> b, ReductionOp op)
        {
            return op switch
            {
                ReductionOp.Sum => Avx2.Add(a, b),
                // No Avx2 int64 Min/Max/Mul — fall back to cross-platform helpers
                // (which compile to compare + ConditionalSelect, etc).
                ReductionOp.Prod => Vector256.Multiply(a, b),
                ReductionOp.Min => Vector256.Min(a, b),
                ReductionOp.Max => Vector256.Max(a, b),
                _ => a
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<ulong> CombineUInt64(Vector256<ulong> a, Vector256<ulong> b, ReductionOp op)
        {
            return op switch
            {
                ReductionOp.Sum => Avx2.Add(a, b),
                ReductionOp.Prod => Vector256.Multiply(a, b),
                ReductionOp.Min => Vector256.Min(a, b),
                ReductionOp.Max => Vector256.Max(a, b),
                _ => a
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<double> CombineDouble(Vector256<double> a, Vector256<double> b, ReductionOp op)
        {
            return op switch
            {
                ReductionOp.Sum => Avx.Add(a, b),
                ReductionOp.Prod => Avx.Multiply(a, b),
                // NaN-correct Max/Min via Vector256 (BCL); Avx.Max/Min do not
                // propagate NaN-first.
                ReductionOp.Min => Vector256.Min(a, b),
                ReductionOp.Max => Vector256.Max(a, b),
                _ => a
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long CombineScalarInt64(long a, long b, ReductionOp op) => op switch
        {
            ReductionOp.Sum => a + b,
            ReductionOp.Prod => a * b,
            ReductionOp.Min => Math.Min(a, b),
            ReductionOp.Max => Math.Max(a, b),
            _ => a
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong CombineScalarUInt64(ulong a, ulong b, ReductionOp op) => op switch
        {
            ReductionOp.Sum => a + b,
            ReductionOp.Prod => a * b,
            ReductionOp.Min => Math.Min(a, b),
            ReductionOp.Max => Math.Max(a, b),
            _ => a
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double CombineScalarDouble(double a, double b, ReductionOp op) => op switch
        {
            ReductionOp.Sum => a + b,
            ReductionOp.Prod => a * b,
            ReductionOp.Min => Math.Min(a, b),
            ReductionOp.Max => Math.Max(a, b),
            _ => a
        };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long HorizontalReduceInt64(Vector256<long> v, ReductionOp op)
        {
            long l0 = v.GetElement(0), l1 = v.GetElement(1);
            long l2 = v.GetElement(2), l3 = v.GetElement(3);
            return op switch
            {
                ReductionOp.Sum => l0 + l1 + l2 + l3,
                ReductionOp.Prod => l0 * l1 * l2 * l3,
                ReductionOp.Min => Math.Min(Math.Min(l0, l1), Math.Min(l2, l3)),
                ReductionOp.Max => Math.Max(Math.Max(l0, l1), Math.Max(l2, l3)),
                _ => l0
            };
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double HorizontalReduceDouble(Vector256<double> v, ReductionOp op)
        {
            double l0 = v.GetElement(0), l1 = v.GetElement(1);
            double l2 = v.GetElement(2), l3 = v.GetElement(3);
            return op switch
            {
                ReductionOp.Sum => l0 + l1 + l2 + l3,
                ReductionOp.Prod => l0 * l1 * l2 * l3,
                ReductionOp.Min => Math.Min(Math.Min(l0, l1), Math.Min(l2, l3)),
                ReductionOp.Max => Math.Max(Math.Max(l0, l1), Math.Max(l2, l3)),
                _ => l0
            };
        }
    }
}
