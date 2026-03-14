using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Clip(NDArray lhs, ValueType min, ValueType max, Type dtype) => Clip(lhs, min, max, dtype?.GetTypeCode());

        /// <summary>
        /// Clips array values to a specified range [min, max].
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
        public override NDArray Clip(NDArray lhs, ValueType min, ValueType max, NPTypeCode? typeCode = null)
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

        /// <summary>
        /// Core clip implementation that dispatches to IL kernels based on dtype.
        /// Uses SIMD-optimized helpers for contiguous arrays (which is guaranteed
        /// by Cast(copy: true) in the calling method).
        /// </summary>
        private unsafe NDArray ClipCore(NDArray arr, ValueType min, ValueType max)
        {
            var len = arr.size;

            if (min != null && max != null)
            {
                switch (arr.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipHelper((byte*)arr.Address, len, Converts.ToByte(min), Converts.ToByte(max));
                        return arr;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipHelper((short*)arr.Address, len, Converts.ToInt16(min), Converts.ToInt16(max));
                        return arr;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipHelper((ushort*)arr.Address, len, Converts.ToUInt16(min), Converts.ToUInt16(max));
                        return arr;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipHelper((int*)arr.Address, len, Converts.ToInt32(min), Converts.ToInt32(max));
                        return arr;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipHelper((uint*)arr.Address, len, Converts.ToUInt32(min), Converts.ToUInt32(max));
                        return arr;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipHelper((long*)arr.Address, len, Converts.ToInt64(min), Converts.ToInt64(max));
                        return arr;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipHelper((ulong*)arr.Address, len, Converts.ToUInt64(min), Converts.ToUInt64(max));
                        return arr;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipHelper((float*)arr.Address, len, Converts.ToSingle(min), Converts.ToSingle(max));
                        return arr;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipHelper((double*)arr.Address, len, Converts.ToDouble(min), Converts.ToDouble(max));
                        return arr;
                    case NPTypeCode.Decimal:
                        ClipDecimal((decimal*)arr.Address, len, Converts.ToDecimal(min), Converts.ToDecimal(max));
                        return arr;
                    case NPTypeCode.Char:
                        ClipChar((char*)arr.Address, len, Converts.ToChar(min), Converts.ToChar(max));
                        return arr;
                    default:
                        throw new NotSupportedException($"Clip not supported for dtype {arr.GetTypeCode}");
                }
            }
            else if (min != null)
            {
                switch (arr.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipMinHelper((byte*)arr.Address, len, Converts.ToByte(min));
                        return arr;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipMinHelper((short*)arr.Address, len, Converts.ToInt16(min));
                        return arr;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipMinHelper((ushort*)arr.Address, len, Converts.ToUInt16(min));
                        return arr;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipMinHelper((int*)arr.Address, len, Converts.ToInt32(min));
                        return arr;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipMinHelper((uint*)arr.Address, len, Converts.ToUInt32(min));
                        return arr;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipMinHelper((long*)arr.Address, len, Converts.ToInt64(min));
                        return arr;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipMinHelper((ulong*)arr.Address, len, Converts.ToUInt64(min));
                        return arr;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipMinHelper((float*)arr.Address, len, Converts.ToSingle(min));
                        return arr;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipMinHelper((double*)arr.Address, len, Converts.ToDouble(min));
                        return arr;
                    case NPTypeCode.Decimal:
                        ClipMinDecimal((decimal*)arr.Address, len, Converts.ToDecimal(min));
                        return arr;
                    case NPTypeCode.Char:
                        ClipMinChar((char*)arr.Address, len, Converts.ToChar(min));
                        return arr;
                    default:
                        throw new NotSupportedException($"Clip not supported for dtype {arr.GetTypeCode}");
                }
            }
            else if (max != null)
            {
                switch (arr.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipMaxHelper((byte*)arr.Address, len, Converts.ToByte(max));
                        return arr;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipMaxHelper((short*)arr.Address, len, Converts.ToInt16(max));
                        return arr;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipMaxHelper((ushort*)arr.Address, len, Converts.ToUInt16(max));
                        return arr;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipMaxHelper((int*)arr.Address, len, Converts.ToInt32(max));
                        return arr;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipMaxHelper((uint*)arr.Address, len, Converts.ToUInt32(max));
                        return arr;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipMaxHelper((long*)arr.Address, len, Converts.ToInt64(max));
                        return arr;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipMaxHelper((ulong*)arr.Address, len, Converts.ToUInt64(max));
                        return arr;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipMaxHelper((float*)arr.Address, len, Converts.ToSingle(max));
                        return arr;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipMaxHelper((double*)arr.Address, len, Converts.ToDouble(max));
                        return arr;
                    case NPTypeCode.Decimal:
                        ClipMaxDecimal((decimal*)arr.Address, len, Converts.ToDecimal(max));
                        return arr;
                    case NPTypeCode.Char:
                        ClipMaxChar((char*)arr.Address, len, Converts.ToChar(max));
                        return arr;
                    default:
                        throw new NotSupportedException($"Clip not supported for dtype {arr.GetTypeCode}");
                }
            }

            // Both min and max are null - return unchanged
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
