using System;
using NumSharp.Utilities;

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

            // The typed kernels below walk the raw buffer from Address, which does NOT include
            // Shape.offset — materialize sliced/strided/broadcast views first (astype of a
            // mismatched dtype already produced a fresh contiguous array).
            if (aTyped.Shape.IsSliced || aTyped.Shape.IsBroadcasted)
                aTyped = aTyped.copy();
            if (vTyped.Shape.IsSliced || vTyped.Shape.IsBroadcasted)
                vTyped = vTyped.copy();

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

            // Accumulator per dtype family mirrors NumPy's correlate inner loops (*_dot in
            // numpy/_core/src/multiarray/arraytypes.c.src): complex sums real/imag components in
            // double; every integer accumulates modularly (int64/uint64 accumulate + truncate is
            // bit-identical to any native-width two's-complement accumulate, so NumPy's
            // platform-dependent npy_long width is moot); Half accumulates float products in
            // float; bool is BOOL_dot's OR-of-ANDs; decimal (no NumPy analog) accumulates in
            // decimal to keep its full 28-digit precision.
            switch (retType)
            {
                case NPTypeCode.Double:
                    ConvolveFullDouble(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Single:
                    ConvolveFullSingle(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Half:
                    ConvolveFullHalf(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Boolean:
                    ConvolveFullBoolean(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.SByte:
                    ConvolveFullInteger<sbyte>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Byte:
                    ConvolveFullInteger<byte>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Int16:
                    ConvolveFullInteger<short>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.UInt16:
                    ConvolveFullInteger<ushort>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Int32:
                    ConvolveFullInteger<int>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.UInt32:
                    ConvolveFullInteger<uint>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Int64:
                    ConvolveFullInteger<long>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.UInt64:
                    ConvolveFullInteger<ulong>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Char:
                    ConvolveFullInteger<char>(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Decimal:
                    ConvolveFullDecimal(a, v, result, na, nv, outLen);
                    break;
                case NPTypeCode.Complex:
                    ConvolveFullComplex(a, v, result, na, nv, outLen);
                    break;
                default:
                    throw new NotSupportedException($"Type {retType} is not supported for convolution.");
            }

            return result;
        }

        private static unsafe void ConvolveFullDouble(NDArray a, NDArray v, NDArray result, long na, long nv, long outLen)
        {
            double* aPtr = (double*)a.Address;
            double* vPtr = (double*)v.Address;
            double* rPtr = (double*)result.Address;

            for (long k = 0; k < outLen; k++)
            {
                long jMin = Math.Max(0, k - nv + 1);
                long jMax = Math.Min(na - 1, k);

                double sum = 0;
                for (long j = jMin; j <= jMax; j++)
                    sum += aPtr[j] * vPtr[k - j];
                rPtr[k] = sum;
            }
        }

        private static unsafe void ConvolveFullSingle(NDArray a, NDArray v, NDArray result, long na, long nv, long outLen)
        {
            float* aPtr = (float*)a.Address;
            float* vPtr = (float*)v.Address;
            float* rPtr = (float*)result.Address;

            for (long k = 0; k < outLen; k++)
            {
                long jMin = Math.Max(0, k - nv + 1);
                long jMax = Math.Min(na - 1, k);

                // FLOAT_dot accumulates in float32 (the CBLAS sdot path also runs in float32).
                float sum = 0f;
                for (long j = jMin; j <= jMax; j++)
                    sum += aPtr[j] * vPtr[k - j];
                rPtr[k] = sum;
            }
        }

        private static unsafe void ConvolveFullHalf(NDArray a, NDArray v, NDArray result, long na, long nv, long outLen)
        {
            Half* aPtr = (Half*)a.Address;
            Half* vPtr = (Half*)v.Address;
            Half* rPtr = (Half*)result.Address;

            for (long k = 0; k < outLen; k++)
            {
                long jMin = Math.Max(0, k - nv + 1);
                long jMax = Math.Min(na - 1, k);

                // HALF_dot: products and accumulation in float32, one final round to half.
                float sum = 0f;
                for (long j = jMin; j <= jMax; j++)
                    sum += (float)aPtr[j] * (float)vPtr[k - j];
                rPtr[k] = (Half)sum;
            }
        }

        private static unsafe void ConvolveFullBoolean(NDArray a, NDArray v, NDArray result, long na, long nv, long outLen)
        {
            bool* aPtr = (bool*)a.Address;
            bool* vPtr = (bool*)v.Address;
            bool* rPtr = (bool*)result.Address;

            for (long k = 0; k < outLen; k++)
            {
                long jMin = Math.Max(0, k - nv + 1);
                long jMax = Math.Min(na - 1, k);

                // BOOL_dot: OR of ANDs with early exit.
                bool any = false;
                for (long j = jMin; j <= jMax && !any; j++)
                    any = aPtr[j] && vPtr[k - j];
                rPtr[k] = any;
            }
        }

        private static unsafe void ConvolveFullInteger<T>(NDArray a, NDArray v, NDArray result, long na, long nv, long outLen)
            where T : unmanaged, System.Numerics.IBinaryInteger<T>
        {
            T* aPtr = (T*)a.Address;
            T* vPtr = (T*)v.Address;
            T* rPtr = (T*)result.Address;

            for (long k = 0; k < outLen; k++)
            {
                long jMin = Math.Max(0, k - nv + 1);
                long jMax = Math.Min(na - 1, k);

                // Modular multiply-accumulate: CreateTruncating sign-extends signed sources, so
                // mod-2^64 products and sums truncated back to T equal NumPy's native-width wrap.
                ulong sum = 0;
                for (long j = jMin; j <= jMax; j++)
                    sum = unchecked(sum + ulong.CreateTruncating(aPtr[j]) * ulong.CreateTruncating(vPtr[k - j]));
                rPtr[k] = T.CreateTruncating(sum);
            }
        }

        private static unsafe void ConvolveFullDecimal(NDArray a, NDArray v, NDArray result, long na, long nv, long outLen)
        {
            decimal* aPtr = (decimal*)a.Address;
            decimal* vPtr = (decimal*)v.Address;
            decimal* rPtr = (decimal*)result.Address;

            for (long k = 0; k < outLen; k++)
            {
                long jMin = Math.Max(0, k - nv + 1);
                long jMax = Math.Min(na - 1, k);

                decimal sum = 0m;
                for (long j = jMin; j <= jMax; j++)
                    sum += aPtr[j] * vPtr[k - j];
                rPtr[k] = sum;
            }
        }

        private static unsafe void ConvolveFullComplex(NDArray a, NDArray v, NDArray result, long na, long nv, long outLen)
        {
            var aPtr = (System.Numerics.Complex*)a.Address;
            var vPtr = (System.Numerics.Complex*)v.Address;
            var rPtr = (System.Numerics.Complex*)result.Address;

            for (long k = 0; k < outLen; k++)
            {
                long jMin = Math.Max(0, k - nv + 1);
                long jMax = Math.Min(na - 1, k);

                // CDOUBLE_dot: component sums in double via the naive complex product, so NaN in
                // either component propagates into BOTH result components exactly like NumPy.
                double sumr = 0, sumi = 0;
                for (long j = jMin; j <= jMax; j++)
                {
                    var x = aPtr[j];
                    var y = vPtr[k - j];
                    sumr += x.Real * y.Real - x.Imaginary * y.Imaginary;
                    sumi += x.Real * y.Imaginary + x.Imaginary * y.Real;
                }

                rPtr[k] = new System.Numerics.Complex(sumr, sumi);
            }
        }

        /// <summary>
        /// Same mode: output length = max(na, nv)
        /// Returns the central part of the full convolution
        /// </summary>
        private static NDArray ConvolveSame(NDArray a, NDArray v, NPTypeCode retType)
        {
            // full is an owning intermediate — once we've sliced + materialized the centre
            // section into a fresh copy, the underlying na+nv-1 buffer is dead. Release it
            // atomically rather than waiting on the finalizer queue.
            using var full = ConvolveFull(a, v, retType);

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
            // full is an owning intermediate — see ConvolveSame for why.
            using var full = ConvolveFull(a, v, retType);

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
