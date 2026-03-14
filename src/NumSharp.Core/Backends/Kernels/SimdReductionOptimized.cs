using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Optimized SIMD reduction kernels with loop unrolling and tree reduction.
    ///
    /// Key optimizations over the baseline:
    /// 1. 4x loop unrolling - process 4 vectors per iteration
    /// 2. Tree reduction pattern - breaks serial dependency chain
    /// 3. Multiple accumulator vectors for instruction-level parallelism
    /// </summary>
    public static class SimdReductionOptimized
    {
        #region Double Sum - Optimized

        /// <summary>
        /// Optimized sum for contiguous double array using 4x unrolling + tree reduction.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double SumDouble_Optimized(double* data, long size)
        {
            if (size == 0) return 0.0;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<double>.Count * 4)
            {
                return SumDouble_Scalar(data, size);
            }

            int vectorCount = Vector256<double>.Count;  // 4 doubles per vector
            int unrollFactor = 4;
            int unrollStep = vectorCount * unrollFactor;  // 16 doubles per unrolled iteration
            long unrollEnd = size - unrollStep;

            // Use 4 independent accumulator vectors to maximize ILP
            var acc0 = Vector256<double>.Zero;
            var acc1 = Vector256<double>.Zero;
            var acc2 = Vector256<double>.Zero;
            var acc3 = Vector256<double>.Zero;

            long i = 0;

            // Main unrolled loop - processes 16 doubles per iteration
            // No serial dependency between acc0, acc1, acc2, acc3 updates!
            for (; i <= unrollEnd; i += unrollStep)
            {
                // Load 4 vectors (16 doubles)
                var v0 = Vector256.Load(data + i);
                var v1 = Vector256.Load(data + i + vectorCount);
                var v2 = Vector256.Load(data + i + vectorCount * 2);
                var v3 = Vector256.Load(data + i + vectorCount * 3);

                // Accumulate into independent accumulators (can execute in parallel)
                acc0 += v0;
                acc1 += v1;
                acc2 += v2;
                acc3 += v3;
            }

            // Single vector loop for remaining full vectors
            long vectorEnd = size - vectorCount;
            for (; i <= vectorEnd; i += vectorCount)
            {
                acc0 += Vector256.Load(data + i);
            }

            // Tree reduction of accumulators: (acc0+acc1) + (acc2+acc3)
            var sum01 = acc0 + acc1;
            var sum23 = acc2 + acc3;
            var sumAll = sum01 + sum23;

            // Horizontal sum of final vector
            double result = Vector256.Sum(sumAll);

            // Scalar tail
            for (; i < size; i++)
            {
                result += data[i];
            }

            return result;
        }

        /// <summary>
        /// Baseline sum (current NumSharp approach) - single accumulator, no unrolling.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double SumDouble_Baseline(double* data, long size)
        {
            if (size == 0) return 0.0;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<double>.Count)
            {
                return SumDouble_Scalar(data, size);
            }

            int vectorCount = Vector256<double>.Count;
            long vectorEnd = size - vectorCount;

            var accumVec = Vector256<double>.Zero;

            long i = 0;
            // Serial dependency: each iteration depends on previous accumVec
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                accumVec += vec;  // Must wait for previous iteration!
            }

            double result = Vector256.Sum(accumVec);

            for (; i < size; i++)
            {
                result += data[i];
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe double SumDouble_Scalar(double* data, long size)
        {
            double sum = 0;
            for (long i = 0; i < size; i++)
                sum += data[i];
            return sum;
        }

        #endregion

        #region Float Sum - Optimized

        /// <summary>
        /// Optimized sum for contiguous float array using 4x unrolling + tree reduction.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe float SumFloat_Optimized(float* data, long size)
        {
            if (size == 0) return 0f;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<float>.Count * 4)
            {
                return SumFloat_Scalar(data, size);
            }

            int vectorCount = Vector256<float>.Count;  // 8 floats per vector
            int unrollFactor = 4;
            int unrollStep = vectorCount * unrollFactor;  // 32 floats per unrolled iteration
            long unrollEnd = size - unrollStep;

            var acc0 = Vector256<float>.Zero;
            var acc1 = Vector256<float>.Zero;
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;

            long i = 0;

            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector256.Load(data + i);
                var v1 = Vector256.Load(data + i + vectorCount);
                var v2 = Vector256.Load(data + i + vectorCount * 2);
                var v3 = Vector256.Load(data + i + vectorCount * 3);

                acc0 += v0;
                acc1 += v1;
                acc2 += v2;
                acc3 += v3;
            }

            long vectorEnd = size - vectorCount;
            for (; i <= vectorEnd; i += vectorCount)
            {
                acc0 += Vector256.Load(data + i);
            }

            var sum01 = acc0 + acc1;
            var sum23 = acc2 + acc3;
            var sumAll = sum01 + sum23;

            float result = Vector256.Sum(sumAll);

            for (; i < size; i++)
            {
                result += data[i];
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe float SumFloat_Baseline(float* data, long size)
        {
            if (size == 0) return 0f;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<float>.Count)
            {
                return SumFloat_Scalar(data, size);
            }

            int vectorCount = Vector256<float>.Count;
            long vectorEnd = size - vectorCount;

            var accumVec = Vector256<float>.Zero;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                accumVec += vec;
            }

            float result = Vector256.Sum(accumVec);

            for (; i < size; i++)
            {
                result += data[i];
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float SumFloat_Scalar(float* data, long size)
        {
            float sum = 0;
            for (long i = 0; i < size; i++)
                sum += data[i];
            return sum;
        }

        #endregion

        #region Int64 Sum - Optimized

        /// <summary>
        /// Optimized sum for contiguous long array using 4x unrolling.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe long SumInt64_Optimized(long* data, long size)
        {
            if (size == 0) return 0L;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<long>.Count * 4)
            {
                return SumInt64_Scalar(data, size);
            }

            int vectorCount = Vector256<long>.Count;  // 4 longs per vector
            int unrollFactor = 4;
            int unrollStep = vectorCount * unrollFactor;  // 16 longs per iteration
            long unrollEnd = size - unrollStep;

            var acc0 = Vector256<long>.Zero;
            var acc1 = Vector256<long>.Zero;
            var acc2 = Vector256<long>.Zero;
            var acc3 = Vector256<long>.Zero;

            long i = 0;

            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector256.Load(data + i);
                var v1 = Vector256.Load(data + i + vectorCount);
                var v2 = Vector256.Load(data + i + vectorCount * 2);
                var v3 = Vector256.Load(data + i + vectorCount * 3);

                acc0 += v0;
                acc1 += v1;
                acc2 += v2;
                acc3 += v3;
            }

            long vectorEnd = size - vectorCount;
            for (; i <= vectorEnd; i += vectorCount)
            {
                acc0 += Vector256.Load(data + i);
            }

            var sum01 = acc0 + acc1;
            var sum23 = acc2 + acc3;
            var sumAll = sum01 + sum23;

            // Manual horizontal sum for long (no Vector256.Sum for long)
            long result = sumAll[0] + sumAll[1] + sumAll[2] + sumAll[3];

            for (; i < size; i++)
            {
                result += data[i];
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe long SumInt64_Baseline(long* data, long size)
        {
            if (size == 0) return 0L;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<long>.Count)
            {
                return SumInt64_Scalar(data, size);
            }

            int vectorCount = Vector256<long>.Count;
            long vectorEnd = size - vectorCount;

            var accumVec = Vector256<long>.Zero;

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                accumVec += vec;
            }

            long result = accumVec[0] + accumVec[1] + accumVec[2] + accumVec[3];

            for (; i < size; i++)
            {
                result += data[i];
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe long SumInt64_Scalar(long* data, long size)
        {
            long sum = 0;
            for (long i = 0; i < size; i++)
                sum += data[i];
            return sum;
        }

        #endregion

        #region Double Sum - 8x Unrolled (NumPy style)

        /// <summary>
        /// 8x unrolled sum matching NumPy's approach - uses all 16 YMM registers.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double SumDouble_8x(double* data, long size)
        {
            if (size == 0) return 0.0;
            if (size == 1) return data[0];

            int vc = Vector256<double>.Count;  // 4 doubles per vector

            if (!Vector256.IsHardwareAccelerated || size < vc * 8)
            {
                return SumDouble_Scalar(data, size);
            }

            int unrollStep = vc * 8;  // 32 doubles per iteration
            long unrollEnd = size - unrollStep;

            // 8 independent accumulators - uses all 16 YMM registers
            var acc0 = Vector256<double>.Zero;
            var acc1 = Vector256<double>.Zero;
            var acc2 = Vector256<double>.Zero;
            var acc3 = Vector256<double>.Zero;
            var acc4 = Vector256<double>.Zero;
            var acc5 = Vector256<double>.Zero;
            var acc6 = Vector256<double>.Zero;
            var acc7 = Vector256<double>.Zero;

            long i = 0;

            // Main 8x unrolled loop
            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector256.Load(data + i);
                var v1 = Vector256.Load(data + i + vc);
                var v2 = Vector256.Load(data + i + vc * 2);
                var v3 = Vector256.Load(data + i + vc * 3);
                var v4 = Vector256.Load(data + i + vc * 4);
                var v5 = Vector256.Load(data + i + vc * 5);
                var v6 = Vector256.Load(data + i + vc * 6);
                var v7 = Vector256.Load(data + i + vc * 7);

                acc0 += v0; acc1 += v1; acc2 += v2; acc3 += v3;
                acc4 += v4; acc5 += v5; acc6 += v6; acc7 += v7;
            }

            // Single vector loop for remaining
            long vectorEnd = size - vc;
            for (; i <= vectorEnd; i += vc)
            {
                acc0 += Vector256.Load(data + i);
            }

            // Tree reduction: 8 → 4 → 2 → 1
            var r01 = acc0 + acc1;
            var r23 = acc2 + acc3;
            var r45 = acc4 + acc5;
            var r67 = acc6 + acc7;
            var r0123 = r01 + r23;
            var r4567 = r45 + r67;
            var sumAll = r0123 + r4567;

            double result = Vector256.Sum(sumAll);

            // Scalar tail
            for (; i < size; i++)
            {
                result += data[i];
            }

            return result;
        }

        #endregion

        #region Max Double - Optimized

        /// <summary>
        /// Optimized max for contiguous double array using 4x unrolling.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double MaxDouble_Optimized(double* data, long size)
        {
            if (size == 0) return double.NegativeInfinity;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<double>.Count * 4)
            {
                return MaxDouble_Scalar(data, size);
            }

            int vectorCount = Vector256<double>.Count;
            int unrollFactor = 4;
            int unrollStep = vectorCount * unrollFactor;
            long unrollEnd = size - unrollStep;

            var acc0 = Vector256.Create(double.NegativeInfinity);
            var acc1 = Vector256.Create(double.NegativeInfinity);
            var acc2 = Vector256.Create(double.NegativeInfinity);
            var acc3 = Vector256.Create(double.NegativeInfinity);

            long i = 0;

            for (; i <= unrollEnd; i += unrollStep)
            {
                var v0 = Vector256.Load(data + i);
                var v1 = Vector256.Load(data + i + vectorCount);
                var v2 = Vector256.Load(data + i + vectorCount * 2);
                var v3 = Vector256.Load(data + i + vectorCount * 3);

                acc0 = Vector256.Max(acc0, v0);
                acc1 = Vector256.Max(acc1, v1);
                acc2 = Vector256.Max(acc2, v2);
                acc3 = Vector256.Max(acc3, v3);
            }

            long vectorEnd = size - vectorCount;
            for (; i <= vectorEnd; i += vectorCount)
            {
                acc0 = Vector256.Max(acc0, Vector256.Load(data + i));
            }

            // Tree reduce accumulators
            var max01 = Vector256.Max(acc0, acc1);
            var max23 = Vector256.Max(acc2, acc3);
            var maxAll = Vector256.Max(max01, max23);

            // Horizontal max
            double result = maxAll[0];
            for (int j = 1; j < vectorCount; j++)
                result = Math.Max(result, maxAll[j]);

            for (; i < size; i++)
            {
                result = Math.Max(result, data[i]);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe double MaxDouble_Baseline(double* data, long size)
        {
            if (size == 0) return double.NegativeInfinity;
            if (size == 1) return data[0];

            if (!Vector256.IsHardwareAccelerated || size < Vector256<double>.Count)
            {
                return MaxDouble_Scalar(data, size);
            }

            int vectorCount = Vector256<double>.Count;
            long vectorEnd = size - vectorCount;

            var accumVec = Vector256.Create(double.NegativeInfinity);

            long i = 0;
            for (; i <= vectorEnd; i += vectorCount)
            {
                var vec = Vector256.Load(data + i);
                accumVec = Vector256.Max(accumVec, vec);
            }

            double result = accumVec[0];
            for (int j = 1; j < vectorCount; j++)
                result = Math.Max(result, accumVec[j]);

            for (; i < size; i++)
            {
                result = Math.Max(result, data[i]);
            }

            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe double MaxDouble_Scalar(double* data, long size)
        {
            double max = double.NegativeInfinity;
            for (long i = 0; i < size; i++)
                max = Math.Max(max, data[i]);
            return max;
        }

        #endregion
    }
}
