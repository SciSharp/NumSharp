using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// DirectILKernelGenerator.Reduction.Arg.cs - ArgMax/ArgMin Reductions
// =============================================================================
//
// RESPONSIBILITY:
//   - ArgMax/ArgMin reduction with SIMD index tracking
//   - Two-pass algorithm: find extreme value with SIMD, then find index
//   - EmitArgMaxMinSimdLoop() - IL emission
//   - ArgMaxSimdHelper<T>(), ArgMinSimdHelper<T>() - SIMD helpers
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class DirectILKernelGenerator
    {
        #region ArgMax/ArgMin Reduction Helpers
        /// <summary>
        /// Emit ArgMax/ArgMin SIMD loop.
        /// Uses helper methods for clean implementation with SIMD index tracking.
        /// Dispatches to type-specific helpers for NaN-awareness (float/double) and Boolean.
        /// </summary>
        private static void EmitArgMaxMinSimdLoop(ILGenerator il, ElementReductionKernelKey key, int inputSize)
        {
            // Dispatch to specialized helpers for types needing special handling.
            bool isMax = key.Op == ReductionOp.ArgMax;
            MethodInfo helperMethod;
            bool isGeneric = true;

            if (key.InputType == NPTypeCode.Single)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxFloatNaNHelper) : nameof(ArgMinFloatNaNHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Double)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxDoubleNaNHelper) : nameof(ArgMinDoubleNaNHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Half)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxHalfNaNHelper) : nameof(ArgMinHalfNaNHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Boolean)
            {
                helperMethod = GetHelper(isMax ? nameof(ArgMaxBoolHelper) : nameof(ArgMinBoolHelper));
                isGeneric = false;
            }
            else if (key.InputType == NPTypeCode.Complex)
            {
                // Complex uses magnitude comparison.
                helperMethod = GetHelper(isMax ? nameof(ArgMaxComplexHelper) : nameof(ArgMinComplexHelper));
                isGeneric = false;
            }
            else
            {
                // Generic SIMD path for integer types.
                helperMethod = GetHelper(isMax ? nameof(ArgMaxSimdHelper) : nameof(ArgMinSimdHelper));
            }

            if (isGeneric)
                helperMethod = helperMethod.MakeGenericMethod(GetClrType(key.InputType));

            // Call helper: *Helper(input, totalSize)
            il.Emit(OpCodes.Ldarg_0); // input
            il.Emit(OpCodes.Ldarg_S, (byte)4); // totalSize
            il.EmitCall(OpCodes.Call, helperMethod, null);

            // Result (long) is already on stack
        }

        /// <summary>
        /// SIMD helper for ArgMax reduction.
        /// Returns the index of the maximum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe long ArgMaxSimdHelper<T>(void* input, long totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            long bestIndex = 0;

            int vectorCount = Vector256<T>.Count;
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                // First pass: find the maximum value using SIMD
                var maxVec = Vector256.Load(src);
                long i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    maxVec = Vector256.Max(maxVec, vec);
                }

                // Horizontal reduce the max vector to find the scalar max
                T maxValue = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = maxVec.GetElement(j);
                    if (elem.CompareTo(maxValue) > 0)
                        maxValue = elem;
                }

                // Process scalar tail for max value
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) > 0)
                        maxValue = src[i];
                }

                // Second pass: find the first index with the max value
                // Use SIMD to quickly scan for the max value
                var targetVec = Vector256.Create(maxValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, targetVec);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        // Found it! Return index of first match
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                // Check scalar tail
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) == 0)
                        return i;
                }

                return 0; // Should never reach here
            }
            vectorCount = Vector128<T>.Count;
            if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                var maxVec = Vector128.Load(src);
                long i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    maxVec = Vector128.Max(maxVec, vec);
                }

                T maxValue = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = maxVec.GetElement(j);
                    if (elem.CompareTo(maxValue) > 0)
                        maxValue = elem;
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) > 0)
                        maxValue = src[i];
                }

                var targetVec = Vector128.Create(maxValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, targetVec);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(maxValue) == 0)
                        return i;
                }

                return 0;
            }
            // Scalar fallback
            for (long i = 1; i < totalSize; i++)
            {
                if (src[i].CompareTo(bestValue) > 0)
                {
                    bestValue = src[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// SIMD helper for ArgMin reduction.
        /// Returns the index of the minimum element.
        /// Uses SIMD to find candidates then scalar to resolve exact index.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe long ArgMinSimdHelper<T>(void* input, long totalSize) where T : unmanaged, IComparable<T>
        {
            if (totalSize == 0)
                return -1;

            if (totalSize == 1)
                return 0;

            T* src = (T*)input;
            T bestValue = src[0];
            long bestIndex = 0;

            int vectorCount = Vector256<T>.Count;
            if (Vector256.IsHardwareAccelerated && Vector256<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                // First pass: find the minimum value using SIMD
                var minVec = Vector256.Load(src);
                long i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    minVec = Vector256.Min(minVec, vec);
                }

                // Horizontal reduce the min vector to find the scalar min
                T minValue = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = minVec.GetElement(j);
                    if (elem.CompareTo(minValue) < 0)
                        minValue = elem;
                }

                // Process scalar tail for min value
                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) < 0)
                        minValue = src[i];
                }

                // Second pass: find the first index with the min value
                var targetVec = Vector256.Create(minValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var mask = Vector256.Equals(vec, targetVec);
                    uint bits = Vector256.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) == 0)
                        return i;
                }

                return 0;
            }
            vectorCount = Vector128<T>.Count;
            if (Vector128.IsHardwareAccelerated && Vector128<T>.IsSupported && totalSize >= vectorCount * 2)
            {
                long vectorEnd = totalSize - vectorCount;

                var minVec = Vector128.Load(src);
                long i = vectorCount;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    minVec = Vector128.Min(minVec, vec);
                }

                T minValue = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    T elem = minVec.GetElement(j);
                    if (elem.CompareTo(minValue) < 0)
                        minValue = elem;
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) < 0)
                        minValue = src[i];
                }

                var targetVec = Vector128.Create(minValue);
                for (i = 0; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var mask = Vector128.Equals(vec, targetVec);
                    uint bits = Vector128.ExtractMostSignificantBits(mask);
                    if (bits != 0)
                    {
                        return i + System.Numerics.BitOperations.TrailingZeroCount(bits);
                    }
                }

                for (; i < totalSize; i++)
                {
                    if (src[i].CompareTo(minValue) == 0)
                        return i;
                }

                return 0;
            }
            // Scalar fallback
            for (long i = 1; i < totalSize; i++)
            {
                if (src[i].CompareTo(bestValue) < 0)
                {
                    bestValue = src[i];
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        #endregion

        #region NaN-Aware ArgMax/ArgMin Helpers (Float/Double)

        // -----------------------------------------------------------------
        // SIMD float/double ArgMax/ArgMin — the port of NumPy's
        // simd_argmax_@sfx@ / simd_argmin_@sfx@ (argfunc.dispatch.c.src): a
        // single-pass, 4×-unrolled tournament that tracks the winning lane's
        // ABSOLUTE index in a companion integer vector, with NaN handled the
        // NumPy way (first NaN in encounter order wins → early return). Strict
        // >/< keeps the LOWEST index on ties, and the acc is only overwritten
        // on a strict win so a later equal value never displaces an earlier
        // one — first-occurrence semantics across lanes AND blocks.
        //
        // This replaced a per-element scalar walk (the documented ~12× gap):
        // measured NPY/NS on float32 argmax — 1K 9.1×, 100K 1.11×, 10M 0.83×
        // (the 10M cell is memory-bandwidth bound, where NumPy's own kernel
        // sits at the same wall). Max and Min are separate methods on purpose:
        // an in-loop `isMax` branch is free at 100K but 2.26× slower at 10M
        // (it blocks the JIT from streaming), so — like NumPy's own repeat
        // macro — the compare direction is baked per method.
        //
        // Half stays scalar below (no Vector256<Half> arithmetic in the BCL,
        // and NumPy's f16 argmax is scalar too); integer types keep the
        // generic ArgMaxSimdHelper<T> two-pass path.

        /// <summary>
        /// ArgMax helper for float — SIMD tournament, first-NaN-wins (NumPy simd_argmax_f32).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe long ArgMaxFloatNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            float* ip = (float*)input;
            const int vstep = 8;              // Vector256<float>.Count
            const int wstep = vstep * 4;
            if (!Vector256.IsHardwareAccelerated || totalSize < wstep)
            {
                float sb = ip[0]; long si = 0;
                for (long q = 1; q < totalSize; q++)
                {
                    float v = ip[q];
                    if (v > sb || (float.IsNaN(v) && !float.IsNaN(sb))) { sb = v; si = q; }
                }
                return si;
            }

            // Cap the vectorized region so the int32 index vector cannot overflow on a
            // >2^31-element array (8 GB of float); the scalar tail finishes the remainder.
            long len0 = totalSize <= int.MaxValue ? totalSize : int.MaxValue;

            var vind0 = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            var vind1 = Vector256.Create(8, 9, 10, 11, 12, 13, 14, 15);
            var vind2 = Vector256.Create(16, 17, 18, 19, 20, 21, 22, 23);
            var vind3 = Vector256.Create(24, 25, 26, 27, 28, 29, 30, 31);
            var accIdx = Vector256<int>.Zero;
            var acc = Vector256.Create(ip[0]);
            long i = 0;

            for (long n = len0 & -wstep; i < n; i += wstep)
            {
                var vi = Vector256.Create((int)i);
                var a = Vector256.Load(ip + i);
                var b = Vector256.Load(ip + i + vstep);
                var c = Vector256.Load(ip + i + vstep * 2);
                var d = Vector256.Load(ip + i + vstep * 3);

                var mBA = Vector256.GreaterThan(b, a);
                var mDC = Vector256.GreaterThan(d, c);
                var xBA = Vector256.ConditionalSelect(mBA, b, a);
                var xDC = Vector256.ConditionalSelect(mDC, d, c);
                var mDCBA = Vector256.GreaterThan(xDC, xBA);
                var xDCBA = Vector256.ConditionalSelect(mDCBA, xDC, xBA);

                var idxBA = Vector256.ConditionalSelect(mBA.AsInt32(), vind1, vind0);
                var idxDC = Vector256.ConditionalSelect(mDC.AsInt32(), vind3, vind2);
                var idxDCBA = Vector256.ConditionalSelect(mDCBA.AsInt32(), idxDC, idxBA);

                var mAcc = Vector256.GreaterThan(xDCBA, acc);
                acc = Vector256.ConditionalSelect(mAcc, xDCBA, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt32(), vi + idxDCBA, accIdx);

                // notnan(x) = (x == x); any NaN in the block → first NaN lane wins.
                var nA = Vector256.Equals(a, a);
                var nB = Vector256.Equals(b, b);
                var nC = Vector256.Equals(c, c);
                var nD = Vector256.Equals(d, d);
                if (Vector256.ExtractMostSignificantBits((nA & nB) & (nC & nD)) != 0xFFu)
                {
                    uint mA = Vector256.ExtractMostSignificantBits(nA);
                    if (mA != 0xFFu) return i + BitOperations.TrailingZeroCount(~mA & 0xFFu);
                    uint mB = Vector256.ExtractMostSignificantBits(nB);
                    if (mB != 0xFFu) return i + vstep + BitOperations.TrailingZeroCount(~mB & 0xFFu);
                    uint mC = Vector256.ExtractMostSignificantBits(nC);
                    if (mC != 0xFFu) return i + vstep * 2 + BitOperations.TrailingZeroCount(~mC & 0xFFu);
                    uint mD = Vector256.ExtractMostSignificantBits(nD);
                    return i + vstep * 3 + BitOperations.TrailingZeroCount(~mD & 0xFFu);
                }
            }
            for (long n = len0 & -vstep; i < n; i += vstep)
            {
                var vi = Vector256.Create((int)i);
                var a = Vector256.Load(ip + i);
                var mAcc = Vector256.GreaterThan(a, acc);
                acc = Vector256.ConditionalSelect(mAcc, a, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt32(), vi + vind0, accIdx);
                uint bits = Vector256.ExtractMostSignificantBits(Vector256.Equals(a, a));
                if (bits != 0xFFu) return i + BitOperations.TrailingZeroCount(~bits & 0xFFu);
            }

            // horizontal reduce: max value, then lowest index among tied lanes.
            Span<float> dacc = stackalloc float[vstep];
            Span<int> didx = stackalloc int[vstep];
            acc.CopyTo(dacc); accIdx.CopyTo(didx);
            float best = dacc[0]; long bestIdx = didx[0];
            for (int vj = 1; vj < vstep; vj++) if (dacc[vj] > best) { best = dacc[vj]; bestIdx = didx[vj]; }
            for (int vj = 0; vj < vstep; vj++) if (best == dacc[vj] && bestIdx > didx[vj]) bestIdx = didx[vj];

            for (; i < totalSize; i++)
            {
                float v = ip[i];
                if (v > best || (float.IsNaN(v) && !float.IsNaN(best))) { best = v; bestIdx = i; }
            }
            return bestIdx;
        }

        /// <summary>
        /// ArgMin helper for float — SIMD tournament, first-NaN-wins (NumPy simd_argmin_f32).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe long ArgMinFloatNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            float* ip = (float*)input;
            const int vstep = 8;
            const int wstep = vstep * 4;
            if (!Vector256.IsHardwareAccelerated || totalSize < wstep)
            {
                float sb = ip[0]; long si = 0;
                for (long q = 1; q < totalSize; q++)
                {
                    float v = ip[q];
                    if (v < sb || (float.IsNaN(v) && !float.IsNaN(sb))) { sb = v; si = q; }
                }
                return si;
            }

            long len0 = totalSize <= int.MaxValue ? totalSize : int.MaxValue;

            var vind0 = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            var vind1 = Vector256.Create(8, 9, 10, 11, 12, 13, 14, 15);
            var vind2 = Vector256.Create(16, 17, 18, 19, 20, 21, 22, 23);
            var vind3 = Vector256.Create(24, 25, 26, 27, 28, 29, 30, 31);
            var accIdx = Vector256<int>.Zero;
            var acc = Vector256.Create(ip[0]);
            long i = 0;

            for (long n = len0 & -wstep; i < n; i += wstep)
            {
                var vi = Vector256.Create((int)i);
                var a = Vector256.Load(ip + i);
                var b = Vector256.Load(ip + i + vstep);
                var c = Vector256.Load(ip + i + vstep * 2);
                var d = Vector256.Load(ip + i + vstep * 3);

                var mBA = Vector256.LessThan(b, a);
                var mDC = Vector256.LessThan(d, c);
                var xBA = Vector256.ConditionalSelect(mBA, b, a);
                var xDC = Vector256.ConditionalSelect(mDC, d, c);
                var mDCBA = Vector256.LessThan(xDC, xBA);
                var xDCBA = Vector256.ConditionalSelect(mDCBA, xDC, xBA);

                var idxBA = Vector256.ConditionalSelect(mBA.AsInt32(), vind1, vind0);
                var idxDC = Vector256.ConditionalSelect(mDC.AsInt32(), vind3, vind2);
                var idxDCBA = Vector256.ConditionalSelect(mDCBA.AsInt32(), idxDC, idxBA);

                var mAcc = Vector256.LessThan(xDCBA, acc);
                acc = Vector256.ConditionalSelect(mAcc, xDCBA, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt32(), vi + idxDCBA, accIdx);

                var nA = Vector256.Equals(a, a);
                var nB = Vector256.Equals(b, b);
                var nC = Vector256.Equals(c, c);
                var nD = Vector256.Equals(d, d);
                if (Vector256.ExtractMostSignificantBits((nA & nB) & (nC & nD)) != 0xFFu)
                {
                    uint mA = Vector256.ExtractMostSignificantBits(nA);
                    if (mA != 0xFFu) return i + BitOperations.TrailingZeroCount(~mA & 0xFFu);
                    uint mB = Vector256.ExtractMostSignificantBits(nB);
                    if (mB != 0xFFu) return i + vstep + BitOperations.TrailingZeroCount(~mB & 0xFFu);
                    uint mC = Vector256.ExtractMostSignificantBits(nC);
                    if (mC != 0xFFu) return i + vstep * 2 + BitOperations.TrailingZeroCount(~mC & 0xFFu);
                    uint mD = Vector256.ExtractMostSignificantBits(nD);
                    return i + vstep * 3 + BitOperations.TrailingZeroCount(~mD & 0xFFu);
                }
            }
            for (long n = len0 & -vstep; i < n; i += vstep)
            {
                var vi = Vector256.Create((int)i);
                var a = Vector256.Load(ip + i);
                var mAcc = Vector256.LessThan(a, acc);
                acc = Vector256.ConditionalSelect(mAcc, a, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt32(), vi + vind0, accIdx);
                uint bits = Vector256.ExtractMostSignificantBits(Vector256.Equals(a, a));
                if (bits != 0xFFu) return i + BitOperations.TrailingZeroCount(~bits & 0xFFu);
            }

            Span<float> dacc = stackalloc float[vstep];
            Span<int> didx = stackalloc int[vstep];
            acc.CopyTo(dacc); accIdx.CopyTo(didx);
            float best = dacc[0]; long bestIdx = didx[0];
            for (int vj = 1; vj < vstep; vj++) if (dacc[vj] < best) { best = dacc[vj]; bestIdx = didx[vj]; }
            for (int vj = 0; vj < vstep; vj++) if (best == dacc[vj] && bestIdx > didx[vj]) bestIdx = didx[vj];

            for (; i < totalSize; i++)
            {
                float v = ip[i];
                if (v < best || (float.IsNaN(v) && !float.IsNaN(best))) { best = v; bestIdx = i; }
            }
            return bestIdx;
        }

        /// <summary>
        /// ArgMax helper for double — SIMD tournament, first-NaN-wins (NumPy simd_argmax_f64).
        /// 4 lanes / int64 index vector (no overflow cap needed — intp indices cover any array).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe long ArgMaxDoubleNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            double* ip = (double*)input;
            const int vstep = 4;              // Vector256<double>.Count
            const int wstep = vstep * 4;
            if (!Vector256.IsHardwareAccelerated || totalSize < wstep)
            {
                double sb = ip[0]; long si = 0;
                for (long q = 1; q < totalSize; q++)
                {
                    double v = ip[q];
                    if (v > sb || (double.IsNaN(v) && !double.IsNaN(sb))) { sb = v; si = q; }
                }
                return si;
            }

            var vind0 = Vector256.Create(0L, 1L, 2L, 3L);
            var vind1 = Vector256.Create(4L, 5L, 6L, 7L);
            var vind2 = Vector256.Create(8L, 9L, 10L, 11L);
            var vind3 = Vector256.Create(12L, 13L, 14L, 15L);
            var accIdx = Vector256<long>.Zero;
            var acc = Vector256.Create(ip[0]);
            long i = 0;

            for (long n = totalSize & -wstep; i < n; i += wstep)
            {
                var vi = Vector256.Create((long)i);
                var a = Vector256.Load(ip + i);
                var b = Vector256.Load(ip + i + vstep);
                var c = Vector256.Load(ip + i + vstep * 2);
                var d = Vector256.Load(ip + i + vstep * 3);

                var mBA = Vector256.GreaterThan(b, a);
                var mDC = Vector256.GreaterThan(d, c);
                var xBA = Vector256.ConditionalSelect(mBA, b, a);
                var xDC = Vector256.ConditionalSelect(mDC, d, c);
                var mDCBA = Vector256.GreaterThan(xDC, xBA);
                var xDCBA = Vector256.ConditionalSelect(mDCBA, xDC, xBA);

                var idxBA = Vector256.ConditionalSelect(mBA.AsInt64(), vind1, vind0);
                var idxDC = Vector256.ConditionalSelect(mDC.AsInt64(), vind3, vind2);
                var idxDCBA = Vector256.ConditionalSelect(mDCBA.AsInt64(), idxDC, idxBA);

                var mAcc = Vector256.GreaterThan(xDCBA, acc);
                acc = Vector256.ConditionalSelect(mAcc, xDCBA, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt64(), vi + idxDCBA, accIdx);

                var nA = Vector256.Equals(a, a);
                var nB = Vector256.Equals(b, b);
                var nC = Vector256.Equals(c, c);
                var nD = Vector256.Equals(d, d);
                if (Vector256.ExtractMostSignificantBits((nA & nB) & (nC & nD)) != 0xFu)
                {
                    uint mA = Vector256.ExtractMostSignificantBits(nA);
                    if (mA != 0xFu) return i + BitOperations.TrailingZeroCount(~mA & 0xFu);
                    uint mB = Vector256.ExtractMostSignificantBits(nB);
                    if (mB != 0xFu) return i + vstep + BitOperations.TrailingZeroCount(~mB & 0xFu);
                    uint mC = Vector256.ExtractMostSignificantBits(nC);
                    if (mC != 0xFu) return i + vstep * 2 + BitOperations.TrailingZeroCount(~mC & 0xFu);
                    uint mD = Vector256.ExtractMostSignificantBits(nD);
                    return i + vstep * 3 + BitOperations.TrailingZeroCount(~mD & 0xFu);
                }
            }
            for (long n = totalSize & -vstep; i < n; i += vstep)
            {
                var vi = Vector256.Create((long)i);
                var a = Vector256.Load(ip + i);
                var mAcc = Vector256.GreaterThan(a, acc);
                acc = Vector256.ConditionalSelect(mAcc, a, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt64(), vi + vind0, accIdx);
                uint bits = Vector256.ExtractMostSignificantBits(Vector256.Equals(a, a));
                if (bits != 0xFu) return i + BitOperations.TrailingZeroCount(~bits & 0xFu);
            }

            Span<double> dacc = stackalloc double[vstep];
            Span<long> didx = stackalloc long[vstep];
            acc.CopyTo(dacc); accIdx.CopyTo(didx);
            double best = dacc[0]; long bestIdx = didx[0];
            for (int vj = 1; vj < vstep; vj++) if (dacc[vj] > best) { best = dacc[vj]; bestIdx = didx[vj]; }
            for (int vj = 0; vj < vstep; vj++) if (best == dacc[vj] && bestIdx > didx[vj]) bestIdx = didx[vj];

            for (; i < totalSize; i++)
            {
                double v = ip[i];
                if (v > best || (double.IsNaN(v) && !double.IsNaN(best))) { best = v; bestIdx = i; }
            }
            return bestIdx;
        }

        /// <summary>
        /// ArgMin helper for double — SIMD tournament, first-NaN-wins (NumPy simd_argmin_f64).
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(OptimizeAndInline)]
        internal static unsafe long ArgMinDoubleNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            double* ip = (double*)input;
            const int vstep = 4;
            const int wstep = vstep * 4;
            if (!Vector256.IsHardwareAccelerated || totalSize < wstep)
            {
                double sb = ip[0]; long si = 0;
                for (long q = 1; q < totalSize; q++)
                {
                    double v = ip[q];
                    if (v < sb || (double.IsNaN(v) && !double.IsNaN(sb))) { sb = v; si = q; }
                }
                return si;
            }

            var vind0 = Vector256.Create(0L, 1L, 2L, 3L);
            var vind1 = Vector256.Create(4L, 5L, 6L, 7L);
            var vind2 = Vector256.Create(8L, 9L, 10L, 11L);
            var vind3 = Vector256.Create(12L, 13L, 14L, 15L);
            var accIdx = Vector256<long>.Zero;
            var acc = Vector256.Create(ip[0]);
            long i = 0;

            for (long n = totalSize & -wstep; i < n; i += wstep)
            {
                var vi = Vector256.Create((long)i);
                var a = Vector256.Load(ip + i);
                var b = Vector256.Load(ip + i + vstep);
                var c = Vector256.Load(ip + i + vstep * 2);
                var d = Vector256.Load(ip + i + vstep * 3);

                var mBA = Vector256.LessThan(b, a);
                var mDC = Vector256.LessThan(d, c);
                var xBA = Vector256.ConditionalSelect(mBA, b, a);
                var xDC = Vector256.ConditionalSelect(mDC, d, c);
                var mDCBA = Vector256.LessThan(xDC, xBA);
                var xDCBA = Vector256.ConditionalSelect(mDCBA, xDC, xBA);

                var idxBA = Vector256.ConditionalSelect(mBA.AsInt64(), vind1, vind0);
                var idxDC = Vector256.ConditionalSelect(mDC.AsInt64(), vind3, vind2);
                var idxDCBA = Vector256.ConditionalSelect(mDCBA.AsInt64(), idxDC, idxBA);

                var mAcc = Vector256.LessThan(xDCBA, acc);
                acc = Vector256.ConditionalSelect(mAcc, xDCBA, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt64(), vi + idxDCBA, accIdx);

                var nA = Vector256.Equals(a, a);
                var nB = Vector256.Equals(b, b);
                var nC = Vector256.Equals(c, c);
                var nD = Vector256.Equals(d, d);
                if (Vector256.ExtractMostSignificantBits((nA & nB) & (nC & nD)) != 0xFu)
                {
                    uint mA = Vector256.ExtractMostSignificantBits(nA);
                    if (mA != 0xFu) return i + BitOperations.TrailingZeroCount(~mA & 0xFu);
                    uint mB = Vector256.ExtractMostSignificantBits(nB);
                    if (mB != 0xFu) return i + vstep + BitOperations.TrailingZeroCount(~mB & 0xFu);
                    uint mC = Vector256.ExtractMostSignificantBits(nC);
                    if (mC != 0xFu) return i + vstep * 2 + BitOperations.TrailingZeroCount(~mC & 0xFu);
                    uint mD = Vector256.ExtractMostSignificantBits(nD);
                    return i + vstep * 3 + BitOperations.TrailingZeroCount(~mD & 0xFu);
                }
            }
            for (long n = totalSize & -vstep; i < n; i += vstep)
            {
                var vi = Vector256.Create((long)i);
                var a = Vector256.Load(ip + i);
                var mAcc = Vector256.LessThan(a, acc);
                acc = Vector256.ConditionalSelect(mAcc, a, acc);
                accIdx = Vector256.ConditionalSelect(mAcc.AsInt64(), vi + vind0, accIdx);
                uint bits = Vector256.ExtractMostSignificantBits(Vector256.Equals(a, a));
                if (bits != 0xFu) return i + BitOperations.TrailingZeroCount(~bits & 0xFu);
            }

            Span<double> dacc = stackalloc double[vstep];
            Span<long> didx = stackalloc long[vstep];
            acc.CopyTo(dacc); accIdx.CopyTo(didx);
            double best = dacc[0]; long bestIdx = didx[0];
            for (int vj = 1; vj < vstep; vj++) if (dacc[vj] < best) { best = dacc[vj]; bestIdx = didx[vj]; }
            for (int vj = 0; vj < vstep; vj++) if (best == dacc[vj] && bestIdx > didx[vj]) bestIdx = didx[vj];

            for (; i < totalSize; i++)
            {
                double v = ip[i];
                if (v < best || (double.IsNaN(v) && !double.IsNaN(best))) { best = v; bestIdx = i; }
            }
            return bestIdx;
        }

        /// <summary>
        /// ArgMax helper for Half with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "maximum").
        /// </summary>
        internal static unsafe long ArgMaxHalfNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Half* src = (Half*)input;
            Half bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                Half val = src[i];
                // NumPy: first NaN always wins
                if (val > bestValue || (Half.IsNaN(val) && !Half.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMin helper for Half with NaN awareness.
        /// NumPy behavior: first NaN always wins (considered "minimum").
        /// </summary>
        internal static unsafe long ArgMinHalfNaNHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Half* src = (Half*)input;
            Half bestValue = src[0];
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                Half val = src[i];
                // NumPy: first NaN always wins
                if (val < bestValue || (Half.IsNaN(val) && !Half.IsNaN(bestValue)))
                {
                    bestValue = val;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        #endregion

        #region Boolean ArgMax/ArgMin Helpers

        /// <summary>
        /// ArgMax helper for boolean arrays: the index of the FIRST True (any nonzero byte counts,
        /// matching NumPy — a frombuffer 0x80 is True), 0 when all False. Port of NumPy's
        /// <c>BOOL_argmax</c> (argfunc.dispatch.c.src): a 4×-vector block scan skips all-zero
        /// prefixes (OR the 4 loads — any nonzero byte breaks), then a scalar finish pins the
        /// exact index. Early exit makes the common case O(first-True), not O(n).
        /// </summary>
        internal static unsafe long ArgMaxBoolHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;

            byte* src = (byte*)input;
            long i = 0;

            if (Vector256.IsHardwareAccelerated)
            {
                const long wstep = 32 * 4;
                var zero = Vector256<byte>.Zero;
                for (long n = totalSize & -wstep; i < n; i += wstep)
                {
                    var a = Vector256.Load(src + i);
                    var b = Vector256.Load(src + i + 32);
                    var c = Vector256.Load(src + i + 64);
                    var d = Vector256.Load(src + i + 96);
                    if (((a | b) | (c | d)) != zero)
                        break;
                }
            }

            for (; i < totalSize; i++)
                if (src[i] != 0)
                    return i;
            return 0; // All False
        }

        /// <summary>
        /// ArgMin helper for boolean arrays: the index of the FIRST False (zero byte), 0 when all
        /// True. NumPy's <c>BOOL_argmin</c> is literally <c>memchr(ip, 0, n)</c>
        /// (arraytypes.c.src); the SIMD block scan here is the same find-first-zero — a block
        /// contains a zero byte iff the Min of its 4 vectors has a zero lane.
        /// </summary>
        internal static unsafe long ArgMinBoolHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;

            byte* src = (byte*)input;
            long i = 0;

            if (Vector256.IsHardwareAccelerated)
            {
                const long wstep = 32 * 4;
                var zero = Vector256<byte>.Zero;
                for (long n = totalSize & -wstep; i < n; i += wstep)
                {
                    var a = Vector256.Load(src + i);
                    var b = Vector256.Load(src + i + 32);
                    var c = Vector256.Load(src + i + 64);
                    var d = Vector256.Load(src + i + 96);
                    if (Vector256.EqualsAny(Vector256.Min(Vector256.Min(a, b), Vector256.Min(c, d)), zero))
                        break;
                }
            }

            for (; i < totalSize; i++)
                if (src[i] == 0)
                    return i;
            return 0; // All True
        }

        #endregion

        #region Complex ArgMax/ArgMin Helpers

        /// <summary>
        /// ArgMax helper for Complex arrays.
        /// NumPy: argmax uses magnitude |z| = sqrt(real² + imag²) for comparison.
        /// On tie (equal magnitudes), returns first occurrence.
        /// </summary>
        internal static unsafe long ArgMaxComplexHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Complex* src = (Complex*)input;
            double bestMagnitude = Complex.Abs(src[0]);
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                double mag = Complex.Abs(src[i]);
                if (mag > bestMagnitude)
                {
                    bestMagnitude = mag;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        /// <summary>
        /// ArgMin helper for Complex arrays.
        /// NumPy: argmin uses magnitude |z| = sqrt(real² + imag²) for comparison.
        /// On tie (equal magnitudes), returns first occurrence.
        /// </summary>
        internal static unsafe long ArgMinComplexHelper(void* input, long totalSize)
        {
            if (totalSize == 0) return -1;
            if (totalSize == 1) return 0;

            Complex* src = (Complex*)input;
            double bestMagnitude = Complex.Abs(src[0]);
            long bestIndex = 0;

            for (long i = 1; i < totalSize; i++)
            {
                double mag = Complex.Abs(src[i]);
                if (mag < bestMagnitude)
                {
                    bestMagnitude = mag;
                    bestIndex = i;
                }
            }
            return bestIndex;
        }

        #endregion
    }
}
