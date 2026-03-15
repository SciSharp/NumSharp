using System;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        /// Returns the discrete, linear convolution of two one-dimensional sequences.
        ///
        /// The convolution operator is often seen in signal processing, where it models the effect of a linear time-invariant system on a signal[1]. In probability theory, the sum of two independent random variables is distributed according to the convolution of their individual distributions.
        ///
        /// If v is longer than a, the arrays are swapped before computation.
        /// </summary>
        /// <param name="v">The second one-dimensional input array.</param>
        /// <param name="mode">'full', 'same', or 'valid'. Default is 'full'.</param>
        /// <returns>Discrete, linear convolution of a and v.</returns>
        /// <remarks>
        /// NumPy Reference: https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
        ///
        /// The convolution product is only given for points where the signals overlap completely.
        /// Values outside the signal boundary have no effect.
        /// </remarks>
        public NDArray convolve(NDArray v, string mode = "full")
        {
            NDArray a = this;

            // Validate inputs are 1D
            if (a.ndim != 1)
                throw new IncorrectShapeException("First argument must be a 1-dimensional array.");
            if (v.ndim != 1)
                throw new IncorrectShapeException("Second argument must be a 1-dimensional array.");

            // Validate non-empty
            if (a.size == 0)
                throw new ArgumentException("a cannot be empty", nameof(a));
            if (v.size == 0)
                throw new ArgumentException("v cannot be empty", nameof(v));

            // NumPy swaps if v is longer than a
            if (v.size > a.size)
            {
                var temp = a;
                a = v;
                v = temp;
            }

            long na = a.size;
            long nv = v.size;

            // Determine output type using NumPy's type promotion rules
            var retType = np._FindCommonType(a, v);

            // Convert inputs to the result type for accurate computation
            // This ensures proper type promotion (e.g., int + int = int, int + float = float)
            NDArray aTyped = a.GetTypeCode == retType ? a : a.astype(retType);
            NDArray vTyped = v.GetTypeCode == retType ? v : v.astype(retType);

            // Compute convolution based on mode
            // Convolution formula: (a * v)[n] = sum_m(a[m] * v[n-m])
            // This is equivalent to correlation with v reversed: correlate(a, v[::-1])

            switch (mode.ToLowerInvariant())
            {
                case "full":
                    return ConvolveFull(aTyped, vTyped, retType);

                case "same":
                    return ConvolveSame(aTyped, vTyped, retType);

                case "valid":
                    return ConvolveValid(aTyped, vTyped, retType);

                default:
                    throw new ArgumentException($"mode must be 'full', 'same', or 'valid', got '{mode}'", nameof(mode));
            }
        }

        /// <summary>
        /// Full convolution: output length = na + nv - 1
        /// </summary>
        private static NDArray ConvolveFull(NDArray a, NDArray v, NPTypeCode retType)
        {
            long na = a.size;
            long nv = v.size;
            long outLen = na + nv - 1;

            var result = new NDArray(retType, Shape.Vector(outLen), true);

            // Convolution: result[k] = sum over j of a[j] * v[k-j]
            // where j ranges over valid indices
            // Note: ConvolveFullTyped uses int internally for loop counters, which is fine
            // since typical convolutions don't exceed 2B elements. The outer sizes are long for consistency.
            switch (retType)
            {
                case NPTypeCode.Double:
                    ConvolveFullTyped<double>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.Single:
                    ConvolveFullTyped<float>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.Int32:
                    ConvolveFullTyped<int>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.Int64:
                    ConvolveFullTyped<long>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.Int16:
                    ConvolveFullTyped<short>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.Byte:
                    ConvolveFullTyped<byte>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.UInt16:
                    ConvolveFullTyped<ushort>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.UInt32:
                    ConvolveFullTyped<uint>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.UInt64:
                    ConvolveFullTyped<ulong>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                case NPTypeCode.Decimal:
                    ConvolveFullTyped<decimal>(a, v, result, (int)na, (int)nv, (int)outLen);
                    break;
                default:
                    throw new NotSupportedException($"Type {retType} is not supported for convolution.");
            }

            return result;
        }

        private static unsafe void ConvolveFullTyped<T>(NDArray a, NDArray v, NDArray result, int na, int nv, int outLen)
            where T : unmanaged
        {
            T* aPtr = (T*)a.Address;
            T* vPtr = (T*)v.Address;
            T* rPtr = (T*)result.Address;

            for (int k = 0; k < outLen; k++)
            {
                // j ranges from max(0, k-nv+1) to min(na-1, k)
                int jMin = Math.Max(0, k - nv + 1);
                int jMax = Math.Min(na - 1, k);

                double sum = 0;
                for (int j = jMin; j <= jMax; j++)
                {
                    // v index is k - j, which is in range [0, nv-1] when j is in [jMin, jMax]
                    double aVal = Convert.ToDouble(aPtr[j]);
                    double vVal = Convert.ToDouble(vPtr[k - j]);
                    sum += aVal * vVal;
                }

                // Convert back to target type
                if (typeof(T) == typeof(double))
                    rPtr[k] = (T)(object)sum;
                else if (typeof(T) == typeof(float))
                    rPtr[k] = (T)(object)(float)sum;
                else if (typeof(T) == typeof(int))
                    rPtr[k] = (T)(object)(int)sum;
                else if (typeof(T) == typeof(long))
                    rPtr[k] = (T)(object)(long)sum;
                else if (typeof(T) == typeof(short))
                    rPtr[k] = (T)(object)(short)sum;
                else if (typeof(T) == typeof(byte))
                    rPtr[k] = (T)(object)(byte)sum;
                else if (typeof(T) == typeof(ushort))
                    rPtr[k] = (T)(object)(ushort)sum;
                else if (typeof(T) == typeof(uint))
                    rPtr[k] = (T)(object)(uint)sum;
                else if (typeof(T) == typeof(ulong))
                    rPtr[k] = (T)(object)(ulong)sum;
                else if (typeof(T) == typeof(decimal))
                    rPtr[k] = (T)(object)(decimal)sum;
            }
        }

        /// <summary>
        /// Same mode: output length = max(na, nv)
        /// Returns the central part of the full convolution
        /// </summary>
        private static NDArray ConvolveSame(NDArray a, NDArray v, NPTypeCode retType)
        {
            // Compute full convolution first
            var full = ConvolveFull(a, v, retType);

            long na = a.size;
            long nv = v.size;
            long outLen = Math.Max(na, nv);

            // For 'same' mode, we return the center portion of length max(na, nv)
            // Start index: (nv - 1) / 2  (integer division, floor)
            long startIdx = (nv - 1) / 2;

            // Slice from startIdx to startIdx + outLen
            return full[$"{startIdx}:{startIdx + outLen}"].copy();
        }

        /// <summary>
        /// Valid mode: output length = max(na, nv) - min(na, nv) + 1
        /// Only positions where signals fully overlap
        /// </summary>
        private static NDArray ConvolveValid(NDArray a, NDArray v, NPTypeCode retType)
        {
            // Compute full convolution first
            var full = ConvolveFull(a, v, retType);

            long na = a.size;
            long nv = v.size;
            long outLen = Math.Max(na, nv) - Math.Min(na, nv) + 1;

            // For 'valid' mode, we skip (min(na, nv) - 1) elements from start
            long startIdx = Math.Min(na, nv) - 1;

            // Slice from startIdx to startIdx + outLen
            return full[$"{startIdx}:{startIdx + outLen}"].copy();
        }
    }
}
