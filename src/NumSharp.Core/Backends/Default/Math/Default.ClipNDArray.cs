using System;
using System.Linq;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // ClipNDArray - Clip with array-valued min/max bounds
        // =============================================================================
        //
        // This implements np.clip(a, min_array, max_array) where min and/or max are
        // NDArrays (possibly broadcast) rather than scalar values.
        //
        // NumPy behavior:
        // - result[i] = min(max(a[i], min[i]), max[i])
        // - When min[i] > max[i] at any position, result is max[i]
        // - NaN in bounds array: propagates to result (IEEE comparison semantics)
        // - min/max arrays are broadcast to match input shape
        //
        // Implementation strategy:
        // 1. Broadcast min/max arrays to match input shape
        // 2. Create output array (copy of input with requested dtype)
        // 3. If all arrays are contiguous, use SIMD-optimized IL kernel path
        // 4. Otherwise, fall back to iterator-based element-wise processing
        //
        // =============================================================================

        public override NDArray ClipNDArray(in NDArray lhs, in NDArray min, in NDArray max, Type dtype, NDArray @out = null)
            => ClipNDArray(lhs, min, max, dtype?.GetTypeCode(), @out);

        public override NDArray ClipNDArray(in NDArray lhs, in NDArray min, in NDArray max, NPTypeCode? typeCode = null, NDArray @out = null)
        {
            if (lhs.size == 0)
                return lhs.Clone();

            // If both bounds are null, just return a copy (NumPy behavior)
            if (min is null && max is null)
                return Cast(lhs, typeCode ?? lhs.typecode, copy: true);

            // Broadcast bounds arrays to match input shape
            // np.broadcast_arrays ensures they are all broadcastable to each other
            var boundsToCheck = new NDArray[] { lhs, min, max }.Where(nd => !(nd is null)).ToArray();
            var broadcasted = np.broadcast_arrays(boundsToCheck);

            var _min = min is null ? null : np.broadcast_to(min, lhs.Shape);
            var _max = max is null ? null : np.broadcast_to(max, lhs.Shape);

            // Create or validate output array
            if (@out is null)
                @out = Cast(lhs, typeCode ?? lhs.typecode, copy: true);
            else
            {
                NumSharpException.ThrowIfNotWriteable(@out.Shape);
                if (@out.Shape != lhs.Shape)
                    throw new ArgumentException($"@out's shape ({@out.Shape}) must match lhs's shape ({lhs.Shape}).'");
            }

            var len = @out.size;

            // Check if we can use the fast contiguous SIMD path
            // All participating arrays must be contiguous with zero offset
            bool canUseFastPath = @out.Shape.IsContiguous && @out.Shape.Offset == 0;
            if (!(_min is null) && canUseFastPath)
                canUseFastPath = _min.Shape.IsContiguous && _min.Shape.Offset == 0;
            if (!(_max is null) && canUseFastPath)
                canUseFastPath = _max.Shape.IsContiguous && _max.Shape.Offset == 0;

            if (canUseFastPath)
                return ClipNDArrayContiguous(@out, _min, _max, len);
            else
                return ClipNDArrayGeneral(@out, _min, _max, len);
        }

        /// <summary>
        /// Fast path for contiguous arrays - uses IL kernel with SIMD support.
        /// </summary>
        private unsafe NDArray ClipNDArrayContiguous(NDArray @out, NDArray min, NDArray max, int len)
        {
            if (!(min is null) && !(max is null))
            {
                // Both bounds - use ClipArrayBounds
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipArrayBounds((byte*)@out.Address, (byte*)min.Address, (byte*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipArrayBounds((short*)@out.Address, (short*)min.Address, (short*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipArrayBounds((ushort*)@out.Address, (ushort*)min.Address, (ushort*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipArrayBounds((int*)@out.Address, (int*)min.Address, (int*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipArrayBounds((uint*)@out.Address, (uint*)min.Address, (uint*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipArrayBounds((long*)@out.Address, (long*)min.Address, (long*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipArrayBounds((ulong*)@out.Address, (ulong*)min.Address, (ulong*)max.Address, len);
                        return @out;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipArrayBounds((float*)@out.Address, (float*)min.Address, (float*)max.Address, len);
                        return @out;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipArrayBounds((double*)@out.Address, (double*)min.Address, (double*)max.Address, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipArrayBoundsDecimal((decimal*)@out.Address, (decimal*)min.Address, (decimal*)max.Address, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipArrayBoundsChar((char*)@out.Address, (char*)min.Address, (char*)max.Address, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else if (!(min is null))
            {
                // Min only - use ClipArrayMin
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipArrayMin((byte*)@out.Address, (byte*)min.Address, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipArrayMin((short*)@out.Address, (short*)min.Address, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipArrayMin((ushort*)@out.Address, (ushort*)min.Address, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipArrayMin((int*)@out.Address, (int*)min.Address, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipArrayMin((uint*)@out.Address, (uint*)min.Address, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipArrayMin((long*)@out.Address, (long*)min.Address, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipArrayMin((ulong*)@out.Address, (ulong*)min.Address, len);
                        return @out;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipArrayMin((float*)@out.Address, (float*)min.Address, len);
                        return @out;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipArrayMin((double*)@out.Address, (double*)min.Address, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipArrayMinDecimal((decimal*)@out.Address, (decimal*)min.Address, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipArrayMinChar((char*)@out.Address, (char*)min.Address, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else // max is not null
            {
                // Max only - use ClipArrayMax
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ILKernelGenerator.ClipArrayMax((byte*)@out.Address, (byte*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ILKernelGenerator.ClipArrayMax((short*)@out.Address, (short*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ILKernelGenerator.ClipArrayMax((ushort*)@out.Address, (ushort*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ILKernelGenerator.ClipArrayMax((int*)@out.Address, (int*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ILKernelGenerator.ClipArrayMax((uint*)@out.Address, (uint*)max.Address, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ILKernelGenerator.ClipArrayMax((long*)@out.Address, (long*)max.Address, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ILKernelGenerator.ClipArrayMax((ulong*)@out.Address, (ulong*)max.Address, len);
                        return @out;
                    case NPTypeCode.Single:
                        ILKernelGenerator.ClipArrayMax((float*)@out.Address, (float*)max.Address, len);
                        return @out;
                    case NPTypeCode.Double:
                        ILKernelGenerator.ClipArrayMax((double*)@out.Address, (double*)max.Address, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipArrayMaxDecimal((decimal*)@out.Address, (decimal*)max.Address, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipArrayMaxChar((char*)@out.Address, (char*)max.Address, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
        }

        /// <summary>
        /// General path for non-contiguous/broadcast arrays - uses GetAtIndex for element access.
        /// </summary>
        private unsafe NDArray ClipNDArrayGeneral(NDArray @out, NDArray min, NDArray max, int len)
        {
            if (!(min is null) && !(max is null))
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ClipNDArrayGeneralCore<byte>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ClipNDArrayGeneralCore<short>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ClipNDArrayGeneralCore<ushort>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ClipNDArrayGeneralCore<int>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ClipNDArrayGeneralCore<uint>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ClipNDArrayGeneralCore<long>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ClipNDArrayGeneralCore<ulong>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Single:
                        ClipNDArrayGeneralCore<float>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Double:
                        ClipNDArrayGeneralCore<double>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipNDArrayGeneralCore<decimal>(@out, min, max, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipNDArrayGeneralCore<char>(@out, min, max, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else if (!(min is null))
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ClipNDArrayMinGeneralCore<byte>(@out, min, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ClipNDArrayMinGeneralCore<short>(@out, min, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ClipNDArrayMinGeneralCore<ushort>(@out, min, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ClipNDArrayMinGeneralCore<int>(@out, min, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ClipNDArrayMinGeneralCore<uint>(@out, min, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ClipNDArrayMinGeneralCore<long>(@out, min, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ClipNDArrayMinGeneralCore<ulong>(@out, min, len);
                        return @out;
                    case NPTypeCode.Single:
                        ClipNDArrayMinGeneralCore<float>(@out, min, len);
                        return @out;
                    case NPTypeCode.Double:
                        ClipNDArrayMinGeneralCore<double>(@out, min, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipNDArrayMinGeneralCore<decimal>(@out, min, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipNDArrayMinGeneralCore<char>(@out, min, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
            else // max is not null
            {
                switch (@out.GetTypeCode)
                {
                    case NPTypeCode.Byte:
                        ClipNDArrayMaxGeneralCore<byte>(@out, max, len);
                        return @out;
                    case NPTypeCode.Int16:
                        ClipNDArrayMaxGeneralCore<short>(@out, max, len);
                        return @out;
                    case NPTypeCode.UInt16:
                        ClipNDArrayMaxGeneralCore<ushort>(@out, max, len);
                        return @out;
                    case NPTypeCode.Int32:
                        ClipNDArrayMaxGeneralCore<int>(@out, max, len);
                        return @out;
                    case NPTypeCode.UInt32:
                        ClipNDArrayMaxGeneralCore<uint>(@out, max, len);
                        return @out;
                    case NPTypeCode.Int64:
                        ClipNDArrayMaxGeneralCore<long>(@out, max, len);
                        return @out;
                    case NPTypeCode.UInt64:
                        ClipNDArrayMaxGeneralCore<ulong>(@out, max, len);
                        return @out;
                    case NPTypeCode.Single:
                        ClipNDArrayMaxGeneralCore<float>(@out, max, len);
                        return @out;
                    case NPTypeCode.Double:
                        ClipNDArrayMaxGeneralCore<double>(@out, max, len);
                        return @out;
                    case NPTypeCode.Decimal:
                        ClipNDArrayMaxGeneralCore<decimal>(@out, max, len);
                        return @out;
                    case NPTypeCode.Char:
                        ClipNDArrayMaxGeneralCore<char>(@out, max, len);
                        return @out;
                    default:
                        throw new NotSupportedException($"ClipNDArray not supported for dtype {@out.GetTypeCode}");
                }
            }
        }

        #region General Path Core Methods

        private static unsafe void ClipNDArrayGeneralCore<T>(NDArray @out, NDArray min, NDArray max, int len)
            where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipNDArrayGeneralCoreFloat(@out, min, max, len);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipNDArrayGeneralCoreDouble(@out, min, max, len);
                return;
            }

            var outAddr = (T*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ChangeType<T>(min.GetAtIndex(i));
                var maxVal = Converts.ChangeType<T>(max.GetAtIndex(i));

                // NumPy semantics: min(max(val, minVal), maxVal)
                if (val.CompareTo(minVal) < 0)
                    val = minVal;
                if (val.CompareTo(maxVal) > 0)
                    val = maxVal;
                outAddr[outOffset] = val;
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCore<T>(NDArray @out, NDArray min, int len)
            where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipNDArrayMinGeneralCoreFloat(@out, min, len);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipNDArrayMinGeneralCoreDouble(@out, min, len);
                return;
            }

            var outAddr = (T*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ChangeType<T>(min.GetAtIndex(i));

                if (val.CompareTo(minVal) < 0)
                    outAddr[outOffset] = minVal;
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCore<T>(NDArray @out, NDArray max, int len)
            where T : unmanaged, IComparable<T>
        {
            // Use specialized implementations for float/double to handle NaN correctly
            if (typeof(T) == typeof(float))
            {
                ClipNDArrayMaxGeneralCoreFloat(@out, max, len);
                return;
            }
            if (typeof(T) == typeof(double))
            {
                ClipNDArrayMaxGeneralCoreDouble(@out, max, len);
                return;
            }

            var outAddr = (T*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ChangeType<T>(max.GetAtIndex(i));

                if (val.CompareTo(maxVal) > 0)
                    outAddr[outOffset] = maxVal;
            }
        }

        #region Floating-Point General Path (NaN-aware)

        // These use Math.Max/Min which properly propagate NaN per IEEE semantics

        private static unsafe void ClipNDArrayGeneralCoreFloat(NDArray @out, NDArray min, NDArray max, int len)
        {
            var outAddr = (float*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToSingle(min.GetAtIndex(i));
                var maxVal = Converts.ToSingle(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(Math.Max(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayGeneralCoreDouble(NDArray @out, NDArray min, NDArray max, int len)
        {
            var outAddr = (double*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToDouble(min.GetAtIndex(i));
                var maxVal = Converts.ToDouble(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(Math.Max(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreFloat(NDArray @out, NDArray min, int len)
        {
            var outAddr = (float*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToSingle(min.GetAtIndex(i));
                outAddr[outOffset] = Math.Max(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreDouble(NDArray @out, NDArray min, int len)
        {
            var outAddr = (double*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToDouble(min.GetAtIndex(i));
                outAddr[outOffset] = Math.Max(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreFloat(NDArray @out, NDArray max, int len)
        {
            var outAddr = (float*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToSingle(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(val, maxVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreDouble(NDArray @out, NDArray max, int len)
        {
            var outAddr = (double*)@out.Address;
            for (int i = 0; i < len; i++)
            {
                int outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToDouble(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(val, maxVal);
            }
        }

        #endregion

        #endregion

        #region Scalar Fallbacks for Non-SIMD Types (Decimal, Char) - Array Bounds

        private static unsafe void ClipArrayBoundsDecimal(decimal* output, decimal* minArr, decimal* maxArr, int size)
        {
            for (int i = 0; i < size; i++)
            {
                var val = output[i];
                if (val < minArr[i]) val = minArr[i];
                if (val > maxArr[i]) val = maxArr[i];
                output[i] = val;
            }
        }

        private static unsafe void ClipArrayMinDecimal(decimal* output, decimal* minArr, int size)
        {
            for (int i = 0; i < size; i++)
                if (output[i] < minArr[i]) output[i] = minArr[i];
        }

        private static unsafe void ClipArrayMaxDecimal(decimal* output, decimal* maxArr, int size)
        {
            for (int i = 0; i < size; i++)
                if (output[i] > maxArr[i]) output[i] = maxArr[i];
        }

        private static unsafe void ClipArrayBoundsChar(char* output, char* minArr, char* maxArr, int size)
        {
            for (int i = 0; i < size; i++)
            {
                var val = output[i];
                if (val < minArr[i]) val = minArr[i];
                if (val > maxArr[i]) val = maxArr[i];
                output[i] = val;
            }
        }

        private static unsafe void ClipArrayMinChar(char* output, char* minArr, int size)
        {
            for (int i = 0; i < size; i++)
                if (output[i] < minArr[i]) output[i] = minArr[i];
        }

        private static unsafe void ClipArrayMaxChar(char* output, char* maxArr, int size)
        {
            for (int i = 0; i < size; i++)
                if (output[i] > maxArr[i]) output[i] = maxArr[i];
        }

        #endregion
    }
}
