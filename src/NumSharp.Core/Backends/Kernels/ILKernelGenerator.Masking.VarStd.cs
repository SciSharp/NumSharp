using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Reflection.Emit;
using System.Runtime.Intrinsics;

// =============================================================================
// ILKernelGenerator.Masking.VarStd.cs - Variance/StdDev SIMD Helpers
// =============================================================================
//
// RESPONSIBILITY:
//   - VarSimdHelper<T> - variance of contiguous array (two-pass algorithm)
//   - StdSimdHelper<T> - standard deviation (sqrt of variance)
//
// NOTE: These are element-wise helpers for full-array Var/Std.
//       For axis reductions, see ILKernelGenerator.Reduction.Axis.VarStd.cs
//
// =============================================================================

namespace NumSharp.Backends.Kernels
{
    public static partial class ILKernelGenerator
    {
        #region Var/Std SIMD Helpers
        /// <summary>
        /// SIMD helper for computing variance of a contiguous array.
        /// Uses two-pass algorithm: compute mean, then sum of squared differences.
        /// </summary>
        /// <typeparam name="T">Element type (float or double)</typeparam>
        /// <param name="src">Pointer to contiguous data</param>
        /// <param name="size">Number of elements</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population variance, 1 for sample variance)</param>
        /// <returns>The variance as double</returns>
        internal static unsafe double VarSimdHelper<T>(T* src, int size, int ddof = 0)
            where T : unmanaged
        {
            if (size == 0)
                return double.NaN;

            if (size <= ddof)
                return double.NaN; // Division by zero or negative

            // Pass 1: Compute mean
            double sum = 0;

            if (typeof(T) == typeof(double))
            {
                double* p = (double*)(void*)src;

                // V512 -> V256 -> V128 -> scalar cascade
                if (Vector512.IsHardwareAccelerated && Vector512<double>.IsSupported && size >= Vector512<double>.Count)
                {
                    int vectorCount = Vector512<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector512<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector512.Add(sumVec, Vector512.Load(p + i));
                    }

                    // Horizontal sum
                    sum = Vector512.Sum(sumVec);

                    // Scalar tail
                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    }

                    // Horizontal sum
                    sum = Vector256.Sum(sumVec);

                    // Scalar tail
                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
                {
                    int vectorCount = Vector128<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector128<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector128.Add(sumVec, Vector128.Load(p + i));
                    }

                    sum = Vector128.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }

                double mean = sum / size;

                // Pass 2: Sum of squared differences
                double sqDiffSum = 0;

                if (Vector512.IsHardwareAccelerated && Vector512<double>.IsSupported && size >= Vector512<double>.Count)
                {
                    int vectorCount = Vector512<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector512.Create(mean);
                    var sqDiffVec = Vector512<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector512.Load(p + i);
                        var diff = Vector512.Subtract(vec, meanVec);
                        sqDiffVec = Vector512.Add(sqDiffVec, Vector512.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector512.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else if (Vector256.IsHardwareAccelerated && Vector256<double>.IsSupported && size >= Vector256<double>.Count)
                {
                    int vectorCount = Vector256<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create(mean);
                    var sqDiffVec = Vector256<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector256.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<double>.IsSupported && size >= Vector128<double>.Count)
                {
                    int vectorCount = Vector128<double>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector128.Create(mean);
                    var sqDiffVec = Vector128<double>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(p + i);
                        var diff = Vector128.Subtract(vec, meanVec);
                        sqDiffVec = Vector128.Add(sqDiffVec, Vector128.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector128.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                return sqDiffSum / (size - ddof);
            }
            else if (typeof(T) == typeof(float))
            {
                float* p = (float*)(void*)src;

                // V512 -> V256 -> V128 -> scalar cascade
                if (Vector512.IsHardwareAccelerated && Vector512<float>.IsSupported && size >= Vector512<float>.Count)
                {
                    int vectorCount = Vector512<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector512<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector512.Add(sumVec, Vector512.Load(p + i));
                    }

                    sum = Vector512.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector256<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector256.Add(sumVec, Vector256.Load(p + i));
                    }

                    sum = Vector256.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
                {
                    int vectorCount = Vector128<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var sumVec = Vector128<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        sumVec = Vector128.Add(sumVec, Vector128.Load(p + i));
                    }

                    sum = Vector128.Sum(sumVec);

                    for (; i < size; i++)
                        sum += p[i];
                }
                else
                {
                    for (int i = 0; i < size; i++)
                        sum += p[i];
                }

                double mean = sum / size;

                // Pass 2: Sum of squared differences (compute in double for precision)
                double sqDiffSum = 0;

                if (Vector512.IsHardwareAccelerated && Vector512<float>.IsSupported && size >= Vector512<float>.Count)
                {
                    int vectorCount = Vector512<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector512.Create((float)mean);
                    var sqDiffVec = Vector512<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector512.Load(p + i);
                        var diff = Vector512.Subtract(vec, meanVec);
                        sqDiffVec = Vector512.Add(sqDiffVec, Vector512.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector512.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else if (Vector256.IsHardwareAccelerated && Vector256<float>.IsSupported && size >= Vector256<float>.Count)
                {
                    int vectorCount = Vector256<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector256.Create((float)mean);
                    var sqDiffVec = Vector256<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector256.Load(p + i);
                        var diff = Vector256.Subtract(vec, meanVec);
                        sqDiffVec = Vector256.Add(sqDiffVec, Vector256.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector256.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else if (Vector128.IsHardwareAccelerated && Vector128<float>.IsSupported && size >= Vector128<float>.Count)
                {
                    int vectorCount = Vector128<float>.Count;
                    int vectorEnd = size - vectorCount;
                    var meanVec = Vector128.Create((float)mean);
                    var sqDiffVec = Vector128<float>.Zero;
                    int i = 0;

                    for (; i <= vectorEnd; i += vectorCount)
                    {
                        var vec = Vector128.Load(p + i);
                        var diff = Vector128.Subtract(vec, meanVec);
                        sqDiffVec = Vector128.Add(sqDiffVec, Vector128.Multiply(diff, diff));
                    }

                    sqDiffSum = Vector128.Sum(sqDiffVec);

                    for (; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }
                else
                {
                    for (int i = 0; i < size; i++)
                    {
                        double diff = p[i] - mean;
                        sqDiffSum += diff * diff;
                    }
                }

                return sqDiffSum / (size - ddof);
            }
            else
            {
                // For integer types, convert to double and compute
                double doubleSum = 0;
                for (int i = 0; i < size; i++)
                {
                    doubleSum += Convert.ToDouble(src[i]);
                }
                double mean = doubleSum / size;

                double sqDiffSum = 0;
                for (int i = 0; i < size; i++)
                {
                    double diff = Convert.ToDouble(src[i]) - mean;
                    sqDiffSum += diff * diff;
                }

                return sqDiffSum / (size - ddof);
            }
        }

        /// <summary>
        /// SIMD helper for computing standard deviation of a contiguous array.
        /// Returns sqrt(variance).
        /// </summary>
        /// <typeparam name="T">Element type (float or double)</typeparam>
        /// <param name="src">Pointer to contiguous data</param>
        /// <param name="size">Number of elements</param>
        /// <param name="ddof">Delta degrees of freedom (0 for population std, 1 for sample std)</param>
        /// <returns>The standard deviation as double</returns>
        internal static unsafe double StdSimdHelper<T>(T* src, int size, int ddof = 0)
            where T : unmanaged
        {
            double variance = VarSimdHelper(src, size, ddof);
            return Math.Sqrt(variance);
        }

        #endregion
    }
}
