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

        public override NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, Type dtype, NDArray @out = null)
            => ClipNDArray(lhs, min, max, dtype?.GetTypeCode(), @out);

        public override NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, NPTypeCode? typeCode = null, NDArray @out = null)
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

            // Determine output dtype
            var outType = typeCode ?? lhs.typecode;

            // Broadcast and cast min/max to output dtype to avoid mixed-type kernel bugs
            var _min = min is null ? null : np.broadcast_to(min, lhs.Shape).astype(outType);
            var _max = max is null ? null : np.broadcast_to(max, lhs.Shape).astype(outType);

            // Create or validate output array
            if (@out is null)
                @out = Cast(lhs, typeCode ?? lhs.typecode, copy: true);
            else
            {
                NumSharpException.ThrowIfNotWriteable(@out.Shape);
                if (@out.Shape != lhs.Shape)
                    throw new ArgumentException($"@out's shape ({@out.Shape}) must match lhs's shape ({lhs.Shape}).'");
                // Copy input data into @out - user-provided @out may contain garbage (e.g., np.empty)
                np.copyto(@out, lhs);
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

        private static unsafe void ClipArrayBoundsDispatch<T>(nint @out, nint min, nint max, long len) where T : unmanaged, IComparable<T>
            => ILKernelGenerator.ClipArrayBounds((T*)@out, (T*)min, (T*)max, len);

        private static unsafe void ClipArrayMinDispatch<T>(nint @out, nint min, long len) where T : unmanaged, IComparable<T>
            => ILKernelGenerator.ClipArrayMin((T*)@out, (T*)min, len);

        private static unsafe void ClipArrayMaxDispatch<T>(nint @out, nint max, long len) where T : unmanaged, IComparable<T>
            => ILKernelGenerator.ClipArrayMax((T*)@out, (T*)max, len);

        private static void ClipGeneralDispatch<T>(NDArray @out, NDArray min, NDArray max, long len) where T : unmanaged, IComparable<T>
            => ClipNDArrayGeneralCore<T>(@out, min, max, len);

        private static void ClipMinGeneralDispatch<T>(NDArray @out, NDArray min, long len) where T : unmanaged, IComparable<T>
            => ClipNDArrayMinGeneralCore<T>(@out, min, len);

        private static void ClipMaxGeneralDispatch<T>(NDArray @out, NDArray max, long len) where T : unmanaged, IComparable<T>
            => ClipNDArrayMaxGeneralCore<T>(@out, max, len);

        /// <summary>
        /// Fast path for contiguous arrays - uses IL kernel with SIMD support.
        /// </summary>
        private unsafe NDArray ClipNDArrayContiguous(NDArray @out, NDArray min, NDArray max, long len)
        {
            var tc = @out.GetTypeCode;

            if (tc == NPTypeCode.Complex)
                return ClipNDArrayContiguousComplex(@out, min, max, len);

            if (!(min is null) && !(max is null))
                NpFunc.Invoke(tc, ClipArrayBoundsDispatch<int>, (nint)@out.Address, (nint)min.Address, (nint)max.Address, len);
            else if (!(min is null))
                NpFunc.Invoke(tc, ClipArrayMinDispatch<int>, (nint)@out.Address, (nint)min.Address, len);
            else
                NpFunc.Invoke(tc, ClipArrayMaxDispatch<int>, (nint)@out.Address, (nint)max.Address, len);

            return @out;
        }

        /// <summary>
        /// General path for non-contiguous/broadcast arrays - uses GetAtIndex for element access.
        /// </summary>
        private unsafe NDArray ClipNDArrayGeneral(NDArray @out, NDArray min, NDArray max, long len)
        {
            var tc = @out.GetTypeCode;

            if (tc == NPTypeCode.Complex)
                return ClipNDArrayGeneralComplex(@out, min, max, len);

            if (!(min is null) && !(max is null))
                NpFunc.Invoke(tc, ClipGeneralDispatch<int>, @out, min, max, len);
            else if (!(min is null))
                NpFunc.Invoke(tc, ClipMinGeneralDispatch<int>, @out, min, len);
            else
                NpFunc.Invoke(tc, ClipMaxGeneralDispatch<int>, @out, max, len);

            return @out;
        }

        #region General Path Core Methods

        private static unsafe void ClipNDArrayGeneralCore<T>(NDArray @out, NDArray min, NDArray max, long len)
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
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
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

        private static unsafe void ClipNDArrayMinGeneralCore<T>(NDArray @out, NDArray min, long len)
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
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ChangeType<T>(min.GetAtIndex(i));

                if (val.CompareTo(minVal) < 0)
                    outAddr[outOffset] = minVal;
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCore<T>(NDArray @out, NDArray max, long len)
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
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ChangeType<T>(max.GetAtIndex(i));

                if (val.CompareTo(maxVal) > 0)
                    outAddr[outOffset] = maxVal;
            }
        }

        #region Floating-Point General Path (NaN-aware)

        // These use Math.Max/Min which properly propagate NaN per IEEE semantics

        private static unsafe void ClipNDArrayGeneralCoreFloat(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (float*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToSingle(min.GetAtIndex(i));
                var maxVal = Converts.ToSingle(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(Math.Max(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayGeneralCoreDouble(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (double*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToDouble(min.GetAtIndex(i));
                var maxVal = Converts.ToDouble(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(Math.Max(val, minVal), maxVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreFloat(NDArray @out, NDArray min, long len)
        {
            var outAddr = (float*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToSingle(min.GetAtIndex(i));
                outAddr[outOffset] = Math.Max(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMinGeneralCoreDouble(NDArray @out, NDArray min, long len)
        {
            var outAddr = (double*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var minVal = Converts.ToDouble(min.GetAtIndex(i));
                outAddr[outOffset] = Math.Max(val, minVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreFloat(NDArray @out, NDArray max, long len)
        {
            var outAddr = (float*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToSingle(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(val, maxVal);
            }
        }

        private static unsafe void ClipNDArrayMaxGeneralCoreDouble(NDArray @out, NDArray max, long len)
        {
            var outAddr = (double*)@out.Address;
            for (long i = 0; i < len; i++)
            {
                long outOffset = @out.Shape.TransformOffset(i);
                var val = outAddr[outOffset];
                var maxVal = Converts.ToDouble(max.GetAtIndex(i));
                outAddr[outOffset] = Math.Min(val, maxVal);
            }
        }

        #endregion

        #endregion

        #region Complex Clip (no IComparable — lexicographic comparison)

        private static int CompareLex(System.Numerics.Complex a, System.Numerics.Complex b)
        {
            int cmp = a.Real.CompareTo(b.Real);
            return cmp != 0 ? cmp : a.Imaginary.CompareTo(b.Imaginary);
        }

        private static bool HasNaN(System.Numerics.Complex c)
            => double.IsNaN(c.Real) || double.IsNaN(c.Imaginary);

        private static readonly System.Numerics.Complex ComplexNaN = new(double.NaN, double.NaN);

        private unsafe NDArray ClipNDArrayContiguousComplex(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (System.Numerics.Complex*)@out.Address;
            if (!(min is null) && !(max is null))
            {
                var minAddr = (System.Numerics.Complex*)min.Address;
                var maxAddr = (System.Numerics.Complex*)max.Address;
                for (long i = 0; i < len; i++)
                {
                    var val = outAddr[i];
                    if (HasNaN(val)) continue;
                    if (HasNaN(minAddr[i])) { outAddr[i] = minAddr[i]; continue; }
                    if (HasNaN(maxAddr[i])) { outAddr[i] = maxAddr[i]; continue; }
                    if (CompareLex(val, minAddr[i]) < 0) val = minAddr[i];
                    if (CompareLex(val, maxAddr[i]) > 0) val = maxAddr[i];
                    outAddr[i] = val;
                }
            }
            else if (!(min is null))
            {
                var minAddr = (System.Numerics.Complex*)min.Address;
                for (long i = 0; i < len; i++)
                {
                    var val = outAddr[i];
                    if (HasNaN(val)) continue;
                    if (HasNaN(minAddr[i])) { outAddr[i] = minAddr[i]; continue; }
                    if (CompareLex(val, minAddr[i]) < 0) outAddr[i] = minAddr[i];
                }
            }
            else
            {
                var maxAddr = (System.Numerics.Complex*)max.Address;
                for (long i = 0; i < len; i++)
                {
                    var val = outAddr[i];
                    if (HasNaN(val)) continue;
                    if (HasNaN(maxAddr[i])) { outAddr[i] = maxAddr[i]; continue; }
                    if (CompareLex(val, maxAddr[i]) > 0) outAddr[i] = maxAddr[i];
                }
            }
            return @out;
        }

        private unsafe NDArray ClipNDArrayGeneralComplex(NDArray @out, NDArray min, NDArray max, long len)
        {
            var outAddr = (System.Numerics.Complex*)@out.Address;
            if (!(min is null) && !(max is null))
            {
                for (long i = 0; i < len; i++)
                {
                    long off = @out.Shape.TransformOffset(i);
                    var val = outAddr[off];
                    if (HasNaN(val)) continue;
                    var minVal = (System.Numerics.Complex)min.GetAtIndex(i);
                    if (HasNaN(minVal)) { outAddr[off] = minVal; continue; }
                    var maxVal = (System.Numerics.Complex)max.GetAtIndex(i);
                    if (HasNaN(maxVal)) { outAddr[off] = maxVal; continue; }
                    if (CompareLex(val, minVal) < 0) val = minVal;
                    if (CompareLex(val, maxVal) > 0) val = maxVal;
                    outAddr[off] = val;
                }
            }
            else if (!(min is null))
            {
                for (long i = 0; i < len; i++)
                {
                    long off = @out.Shape.TransformOffset(i);
                    var val = outAddr[off];
                    if (HasNaN(val)) continue;
                    var minVal = (System.Numerics.Complex)min.GetAtIndex(i);
                    if (HasNaN(minVal)) { outAddr[off] = minVal; continue; }
                    if (CompareLex(val, minVal) < 0) outAddr[off] = minVal;
                }
            }
            else
            {
                for (long i = 0; i < len; i++)
                {
                    long off = @out.Shape.TransformOffset(i);
                    var val = outAddr[off];
                    if (HasNaN(val)) continue;
                    var maxVal = (System.Numerics.Complex)max.GetAtIndex(i);
                    if (HasNaN(maxVal)) { outAddr[off] = maxVal; continue; }
                    if (CompareLex(val, maxVal) > 0) outAddr[off] = maxVal;
                }
            }
            return @out;
        }

        #endregion

        #region Scalar Fallbacks for Non-SIMD Types (Decimal, Char) - Array Bounds

        private static unsafe void ClipArrayBoundsDecimal(decimal* output, decimal* minArr, decimal* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                if (val < minArr[i]) val = minArr[i];
                if (val > maxArr[i]) val = maxArr[i];
                output[i] = val;
            }
        }

        private static unsafe void ClipArrayMinDecimal(decimal* output, decimal* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] < minArr[i]) output[i] = minArr[i];
        }

        private static unsafe void ClipArrayMaxDecimal(decimal* output, decimal* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] > maxArr[i]) output[i] = maxArr[i];
        }

        private static unsafe void ClipArrayBoundsChar(char* output, char* minArr, char* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
            {
                var val = output[i];
                if (val < minArr[i]) val = minArr[i];
                if (val > maxArr[i]) val = maxArr[i];
                output[i] = val;
            }
        }

        private static unsafe void ClipArrayMinChar(char* output, char* minArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] < minArr[i]) output[i] = minArr[i];
        }

        private static unsafe void ClipArrayMaxChar(char* output, char* maxArr, long size)
        {
            for (long i = 0; i < size; i++)
                if (output[i] > maxArr[i]) output[i] = maxArr[i];
        }

        #endregion
    }
}
