using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Internal helper: Clips array values to a specified range [min, max].
        /// NumPy behavior:
        /// - NaN in data propagates through (result is NaN)
        /// - NaN in scalar min/max: entire array becomes NaN (for floating-point)
        /// </summary>
        /// <remarks>
        /// Implementation uses IL kernels with:
        /// - SIMD for contiguous arrays (Vector256/Vector128)
        /// - Scalar iteration for strided arrays (via TransformOffset)
        ///
        /// The Cast(copy: true) call ensures we have a contiguous output array,
        /// so the SIMD path is always taken for supported types.
        /// </remarks>
        internal NDArray ClipScalar(NDArray lhs, object min, object max, NPTypeCode? typeCode = null)
        {
            if (lhs.size == 0)
                return lhs.Clone();

            var outTypeCode = typeCode ?? lhs.typecode;

            // NumPy behavior: NaN in scalar min/max causes entire result to be NaN
            // This must be handled before the kernel, as scalar fallback doesn't propagate NaN correctly
            if (outTypeCode == NPTypeCode.Double || outTypeCode == NPTypeCode.Single)
            {
                bool minIsNaN = min != null && (min is double dMin && double.IsNaN(dMin) || min is float fMin && float.IsNaN(fMin));
                bool maxIsNaN = max != null && (max is double dMax && double.IsNaN(dMax) || max is float fMax && float.IsNaN(fMax));

                if (minIsNaN || maxIsNaN)
                {
                    // Return array filled with NaN
                    if (outTypeCode == NPTypeCode.Double)
                        return np.full(lhs.Shape, double.NaN);
                    else
                        return np.full(lhs.Shape, float.NaN);
                }
            }

            var @out = Cast(lhs, outTypeCode, copy: true);
            var len = @out.size;

            // Unified dispatch through ClipCore - handles all dtype combinations
            // Cast(copy: true) guarantees contiguous output, so SIMD path is taken
            return ClipCore(@out, min, max);
        }

        private static unsafe void ClipBothDispatch<T>(nint addr, long len, object min, object max) where T : unmanaged, IComparable<T>
            => ILKernelGenerator.ClipHelper((T*)addr, len, Converts.ChangeType<T>(min), Converts.ChangeType<T>(max));

        private static unsafe void ClipMinDispatch<T>(nint addr, long len, object min) where T : unmanaged, IComparable<T>
            => ILKernelGenerator.ClipMinHelper((T*)addr, len, Converts.ChangeType<T>(min));

        private static unsafe void ClipMaxDispatch<T>(nint addr, long len, object max) where T : unmanaged, IComparable<T>
            => ILKernelGenerator.ClipMaxHelper((T*)addr, len, Converts.ChangeType<T>(max));

        /// <summary>
        /// Core clip implementation that dispatches to IL kernels based on dtype.
        /// Uses SIMD-optimized helpers for contiguous arrays (which is guaranteed
        /// by Cast(copy: true) in the calling method).
        /// </summary>
        private unsafe NDArray ClipCore(NDArray arr, object min, object max)
        {
            var len = arr.size;
            var tc = arr.GetTypeCode;

            if (tc == NPTypeCode.Complex)
                return ClipCoreComplex(arr, min, max);

            if (min != null && max != null)
                NpFunc.Invoke(tc, ClipBothDispatch<int>, (nint)arr.Address, len, min, max);
            else if (min != null)
                NpFunc.Invoke(tc, ClipMinDispatch<int>, (nint)arr.Address, len, min);
            else if (max != null)
                NpFunc.Invoke(tc, ClipMaxDispatch<int>, (nint)arr.Address, len, max);

            return arr;
        }

        private static unsafe NDArray ClipCoreComplex(NDArray arr, object min, object max)
        {
            var addr = (System.Numerics.Complex*)arr.Address;
            var len = arr.size;
            var minVal = min != null ? Converts.ChangeType<System.Numerics.Complex>(min) : default;
            var maxVal = max != null ? Converts.ChangeType<System.Numerics.Complex>(max) : default;
            bool hasMin = min != null, hasMax = max != null;

            for (long i = 0; i < len; i++)
            {
                var val = addr[i];
                if (hasMin)
                {
                    int cmp = val.Real.CompareTo(minVal.Real);
                    if (cmp == 0) cmp = val.Imaginary.CompareTo(minVal.Imaginary);
                    if (cmp < 0) val = minVal;
                }
                if (hasMax)
                {
                    int cmp = val.Real.CompareTo(maxVal.Real);
                    if (cmp == 0) cmp = val.Imaginary.CompareTo(maxVal.Imaginary);
                    if (cmp > 0) val = maxVal;
                }
                addr[i] = val;
            }
            return arr;
        }

        #region Scalar Fallbacks for Non-SIMD Types (Decimal, Char)

        private static unsafe void ClipDecimal(decimal* data, long size, decimal minVal, decimal maxVal)
        {
            for (long i = 0; i < size; i++)
            {
                var val = data[i];
                if (val > maxVal) val = maxVal;
                else if (val < minVal) val = minVal;
                data[i] = val;
            }
        }

        private static unsafe void ClipMinDecimal(decimal* data, long size, decimal minVal)
        {
            for (long i = 0; i < size; i++)
                if (data[i] < minVal) data[i] = minVal;
        }

        private static unsafe void ClipMaxDecimal(decimal* data, long size, decimal maxVal)
        {
            for (long i = 0; i < size; i++)
                if (data[i] > maxVal) data[i] = maxVal;
        }

        private static unsafe void ClipChar(char* data, long size, char minVal, char maxVal)
        {
            for (long i = 0; i < size; i++)
            {
                var val = data[i];
                if (val > maxVal) val = maxVal;
                else if (val < minVal) val = minVal;
                data[i] = val;
            }
        }

        private static unsafe void ClipMinChar(char* data, long size, char minVal)
        {
            for (long i = 0; i < size; i++)
                if (data[i] < minVal) data[i] = minVal;
        }

        private static unsafe void ClipMaxChar(char* data, long size, char maxVal)
        {
            for (long i = 0; i < size; i++)
                if (data[i] > maxVal) data[i] = maxVal;
        }

        #endregion
    }
}
