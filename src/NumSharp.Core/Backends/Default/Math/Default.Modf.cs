using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, Type dtype) => ModF(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Return the fractional and integral parts of an array, element-wise.
        ///
        /// NumPy behavior (C standard modf):
        /// - modf(1.5) = (0.5, 1.0)
        /// - modf(-2.7) = (-0.7, -2.0)  -- sign of fractional matches input
        /// - modf(inf) = (0.0, inf)
        /// - modf(-inf) = (-0.0, -inf)
        /// - modf(nan) = (nan, nan)
        ///
        /// Decimal is a NumSharp extension (NumPy doesn't have decimal type).
        /// </summary>
        public override (NDArray Fractional, NDArray Intergral) ModF(in NDArray nd, NPTypeCode? typeCode = null)
        {
            var resolvedType = typeCode ?? nd.typecode;

            // Validate type - modf only makes sense for floating-point types
            if (resolvedType != NPTypeCode.Double &&
                resolvedType != NPTypeCode.Single &&
                resolvedType != NPTypeCode.Decimal)
            {
                throw new NotSupportedException(
                    $"modf only supports floating-point types (Single, Double, Decimal), got {resolvedType}");
            }

            // Cast to target type and materialize to contiguous memory
            // Cast with copy:true creates a contiguous copy, which is what we need
            // for both SIMD processing and correct output
            var fractional = Cast(nd, resolvedType, copy: true);
            var integral = Cast(nd, resolvedType, copy: true);
            var len = fractional.size;

            if (len == 0)
                return (fractional, integral);

            // All paths now use SIMD-optimized helpers (arrays are guaranteed contiguous after Cast)
            unsafe
            {
                switch (resolvedType)
                {
                    case NPTypeCode.Double:
                        ILKernelGenerator.ModfHelper((double*)fractional.Address, (double*)integral.Address, len);
                        return (fractional, integral);

                    case NPTypeCode.Single:
                        ILKernelGenerator.ModfHelper((float*)fractional.Address, (float*)integral.Address, len);
                        return (fractional, integral);

                    case NPTypeCode.Decimal:
                        ModfDecimal((decimal*)fractional.Address, (decimal*)integral.Address, len);
                        return (fractional, integral);

                    default:
                        throw new NotSupportedException($"Unexpected type: {resolvedType}");
                }
            }
        }

        /// <summary>
        /// Scalar modf for decimal type (no SIMD, decimal is 128-bit).
        /// </summary>
        private static unsafe void ModfDecimal(decimal* data, decimal* integral, int size)
        {
            for (int i = 0; i < size; i++)
            {
                var trunc = Math.Truncate(data[i]);
                integral[i] = trunc;
                data[i] = data[i] - trunc;
            }
        }
    }
}
