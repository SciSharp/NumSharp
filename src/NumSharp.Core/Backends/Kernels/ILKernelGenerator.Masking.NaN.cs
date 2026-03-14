using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Masking.NaN.cs - NaN-aware SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - NanSumSimdHelperFloat/Double - sum ignoring NaN values
//   - NanProdSimdHelperFloat/Double - product ignoring NaN values
//   - NanMinSimdHelperFloat/Double - min ignoring NaN values
//   - NanMaxSimdHelperFloat/Double - max ignoring NaN values
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region NaN-aware SIMD Helpers
        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        /// <param name="src">Pointer to contiguous float data</param>
        /// <param name="size">Number of elements</param>
        /// <returns>Sum of non-NaN elements</returns>
        internal static unsafe float NanSumSimdHelperFloat(float* src, long size)
        {
            if (size == 0)
                return 0f;

            float sum = 0f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector256<float>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Create mask where NaN becomes 0, valid values stay
                    // NaN comparison: x != x is true for NaN
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN, false for NaN
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector128<float>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsSingle());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware sum of a contiguous double array.
        /// NaN values are treated as 0 (ignored in the sum).
        /// </summary>
        internal static unsafe double NanSumSimdHelperDouble(double* src, long size)
        {
            if (size == 0)
                return 0.0;

            double sum = 0.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector256<double>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector256.Add(sumVec, cleaned);
                }

                sum = Vector256.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var sumVec = Vector128<double>.Zero;
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.BitwiseAnd(vec, nanMask.AsDouble());
                    sumVec = Vector128.Add(sumVec, cleaned);
                }

                sum = Vector128.Sum(sumVec);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        sum += src[i];
                }
            }

            return sum;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous float array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        internal static unsafe float NanProdSimdHelperFloat(float* src, long size)
        {
            if (size == 0)
                return 1f;

            float prod = 1f;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1f);
                var oneVec = Vector256.Create(1f);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with 1: if NaN, use 1; otherwise use original
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                // Horizontal product (no built-in, do manually)
                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1f);
                var oneVec = Vector128.Create(1f);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        /// <summary>
        /// SIMD helper for NaN-aware product of a contiguous double array.
        /// NaN values are treated as 1 (ignored in the product).
        /// </summary>
        internal static unsafe double NanProdSimdHelperDouble(double* src, long size)
        {
            if (size == 0)
                return 1.0;

            double prod = 1.0;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector256.Create(1.0);
                var oneVec = Vector256.Create(1.0);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector256.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var prodVec = Vector128.Create(1.0);
                var oneVec = Vector128.Create(1.0);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, oneVec);
                    prodVec = Vector128.Multiply(prodVec, cleaned);
                }

                prod = prodVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                    prod *= prodVec.GetElement(j);

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                        prod *= src[i];
                }
            }

            return prod;
        }

        /// <summary>
        /// SIMD helper for NaN-aware minimum of a contiguous float array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe float NanMinSimdHelperFloat(float* src, long size)
        {
            if (size == 0)
                return float.NaN;

            float minVal = float.PositiveInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var minVec = Vector256.Create(float.PositiveInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with +Inf so they don't affect minimum
                    var nanMask = Vector256.Equals(vec, vec); // true for non-NaN
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.PositiveInfinity));
                    minVec = Vector256.Min(minVec, cleaned);
                }

                // Horizontal min
                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !float.IsPositiveInfinity(minVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var minVec = Vector128.Create(float.PositiveInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.PositiveInfinity));
                    minVec = Vector128.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !float.IsPositiveInfinity(minVal);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        if (src[i] < minVal)
                            minVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? minVal : float.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware minimum of a contiguous double array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe double NanMinSimdHelperDouble(double* src, long size)
        {
            if (size == 0)
                return double.NaN;

            double minVal = double.PositiveInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var minVec = Vector256.Create(double.PositiveInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.PositiveInfinity));
                    minVec = Vector256.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !double.IsPositiveInfinity(minVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var minVec = Vector128.Create(double.PositiveInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.PositiveInfinity));
                    minVec = Vector128.Min(minVec, cleaned);
                }

                minVal = minVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = minVec.GetElement(j);
                    if (elem < minVal)
                        minVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] < minVal)
                        minVal = src[i];
                }

                foundNonNaN = !double.IsPositiveInfinity(minVal);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        if (src[i] < minVal)
                            minVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? minVal : double.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware maximum of a contiguous float array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe float NanMaxSimdHelperFloat(float* src, long size)
        {
            if (size == 0)
                return float.NaN;

            float maxVal = float.NegativeInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
            {
                int vectorCount = Vector256<float>.Count;
                long vectorEnd = size - vectorCount;
                var maxVec = Vector256.Create(float.NegativeInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    // Replace NaN with -Inf so they don't affect maximum
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(float.NegativeInfinity));
                    maxVec = Vector256.Max(maxVec, cleaned);
                }

                // Horizontal max
                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                // Scalar tail
                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !float.IsNegativeInfinity(maxVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
            {
                int vectorCount = Vector128<float>.Count;
                long vectorEnd = size - vectorCount;
                var maxVec = Vector128.Create(float.NegativeInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(float.NegativeInfinity));
                    maxVec = Vector128.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    float elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!float.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !float.IsNegativeInfinity(maxVal);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!float.IsNaN(src[i]))
                    {
                        if (src[i] > maxVal)
                            maxVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? maxVal : float.NaN;
        }

        /// <summary>
        /// SIMD helper for NaN-aware maximum of a contiguous double array.
        /// NaN values are ignored; returns NaN if all values are NaN.
        /// </summary>
        internal static unsafe double NanMaxSimdHelperDouble(double* src, long size)
        {
            if (size == 0)
                return double.NaN;

            double maxVal = double.NegativeInfinity;
            bool foundNonNaN = false;

            if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
            {
                int vectorCount = Vector256<double>.Count;
                long vectorEnd = size - vectorCount;
                var maxVec = Vector256.Create(double.NegativeInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector256.Load(src + i);
                    var nanMask = Vector256.Equals(vec, vec);
                    var cleaned = Vector256.ConditionalSelect(nanMask, vec, Vector256.Create(double.NegativeInfinity));
                    maxVec = Vector256.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !double.IsNegativeInfinity(maxVal);
            }
            else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
            {
                int vectorCount = Vector128<double>.Count;
                long vectorEnd = size - vectorCount;
                var maxVec = Vector128.Create(double.NegativeInfinity);
                long i = 0;

                for (; i <= vectorEnd; i += vectorCount)
                {
                    var vec = Vector128.Load(src + i);
                    var nanMask = Vector128.Equals(vec, vec);
                    var cleaned = Vector128.ConditionalSelect(nanMask, vec, Vector128.Create(double.NegativeInfinity));
                    maxVec = Vector128.Max(maxVec, cleaned);
                }

                maxVal = maxVec.GetElement(0);
                for (int j = 1; j < vectorCount; j++)
                {
                    double elem = maxVec.GetElement(j);
                    if (elem > maxVal)
                        maxVal = elem;
                }

                for (; i < size; i++)
                {
                    if (!double.IsNaN(src[i]) && src[i] > maxVal)
                        maxVal = src[i];
                }

                foundNonNaN = !double.IsNegativeInfinity(maxVal);
            }
            else
            {
                for (long i = 0; i < size; i++)
                {
                    if (!double.IsNaN(src[i]))
                    {
                        if (src[i] > maxVal)
                            maxVal = src[i];
                        foundNonNaN = true;
                    }
                }
            }

            return foundNonNaN ? maxVal : double.NaN;
        }

        #endregion
    }
}
