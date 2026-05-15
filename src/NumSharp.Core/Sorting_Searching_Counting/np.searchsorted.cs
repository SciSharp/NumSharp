using System;
using NumSharp.Backends;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Value to insert into <paramref name="a"/>.</param>
        /// <param name="side">If "left" (default), index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Scalar index for insertion point.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static long searchsorted(NDArray a, int v, string side = "left", NDArray sorter = null)
        {
            ValidateSearchSorted(a, side, sorter);
            return SearchSortedScalar(a, Converts.ChangeType(v, a.typecode), side == "left", sorter);
        }

        /// <summary>
        /// Find index where a scalar should be inserted to maintain order.
        /// </summary>
        public static long searchsorted(NDArray a, double v, string side = "left", NDArray sorter = null)
        {
            ValidateSearchSorted(a, side, sorter);
            return SearchSortedScalar(a, Converts.ChangeType(v, a.typecode), side == "left", sorter);
        }

        /// <summary>
        /// Find indices where elements should be inserted to maintain order.
        ///
        /// Find the indices into a sorted array <paramref name="a"/> such that, if the corresponding elements
        /// in <paramref name="v"/> were inserted before the indices, the order of <paramref name="a"/> would be preserved.
        /// </summary>
        /// <param name="a">Input 1-D array. Must be sorted ascending unless <paramref name="sorter"/> is provided.</param>
        /// <param name="v">Values to insert into <paramref name="a"/>. May be a scalar or any shape.</param>
        /// <param name="side">If "left" (default), the index of the first suitable location is returned. If "right", the last such index.</param>
        /// <param name="sorter">Optional indices that sort <paramref name="a"/> into ascending order (typically <c>argsort(a)</c>).</param>
        /// <returns>Array of insertion points with the same shape as <paramref name="v"/>, or a scalar if <paramref name="v"/> is a scalar.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.searchsorted.html</remarks>
        public static NDArray searchsorted(NDArray a, NDArray v, string side = "left", NDArray sorter = null)
        {
            ValidateSearchSorted(a, side, sorter);
            bool leftSide = side == "left";

            if (v.Shape.IsScalar)
            {
                object scalar = Converts.ChangeType(v.Storage.GetAtIndex(0), a.typecode);
                long idx = SearchSortedScalar(a, scalar, leftSide, sorter);
                return NDArray.Scalar(idx);
            }

            Shape outShape = new Shape(v.shape);

            if (v.size == 0)
                return new NDArray(NPTypeCode.Int64, outShape, false);

            // Promote v to a's dtype and force contiguous (NumPy: PyArray_CARRAY_RO).
            NDArray vTyped = EnsureContiguousOfType(v, a.typecode);
            NDArray sorterTyped = (sorter is null) ? null : EnsureContiguousInt64(sorter);
            NDArray output = new NDArray(NPTypeCode.Int64, outShape, false);

            int elemSize = ILKernelGenerator.GetTypeSize(a.typecode);
            unsafe
            {
                var kernel = ILKernelGenerator.GetSearchSortedKernel(a.typecode, leftSide, sorter is not null);
                long arrStride = a.Shape.IsContiguous ? elemSize : a.Shape.strides[0] * elemSize;
                void* arrPtr = (void*)((byte*)a.Storage.Address + a.Shape.offset * elemSize);
                void* keyPtr = (void*)vTyped.Address;
                void* sorterPtr = sorterTyped is null ? null : (void*)sorterTyped.Address;
                long* retPtr = (long*)output.Address;
                kernel(arrPtr, a.size, arrStride, keyPtr, vTyped.size, sorterPtr, retPtr);
            }

            return output;
        }

        private static long SearchSortedScalar(NDArray a, object scalarValue, bool leftSide, NDArray sorter)
        {
            NDArray sorterTyped = (sorter is null) ? null : EnsureContiguousInt64(sorter);
            int elemSize = ILKernelGenerator.GetTypeSize(a.typecode);
            long result;
            unsafe
            {
                // 16-byte buffer fits all 15 dtypes (largest = decimal/complex at 16 bytes).
                byte* keyBuf = stackalloc byte[16];
                WriteScalar(keyBuf, scalarValue, a.typecode);
                long* retBuf = stackalloc long[1];
                var kernel = ILKernelGenerator.GetSearchSortedKernel(a.typecode, leftSide, sorter is not null);
                long arrStride = a.Shape.IsContiguous ? elemSize : a.Shape.strides[0] * elemSize;
                void* arrPtr = (void*)((byte*)a.Storage.Address + a.Shape.offset * elemSize);
                void* sorterPtr = sorterTyped is null ? null : (void*)sorterTyped.Address;
                kernel(arrPtr, a.size, arrStride, keyBuf, 1, sorterPtr, retBuf);
                result = retBuf[0];
            }
            return result;
        }

        private static unsafe void WriteScalar(byte* dest, object value, NPTypeCode typeCode)
        {
            switch (typeCode)
            {
                case NPTypeCode.Boolean: *(bool*)dest = (bool)value; break;
                case NPTypeCode.SByte:   *(sbyte*)dest = (sbyte)value; break;
                case NPTypeCode.Byte:    *(byte*)dest = (byte)value; break;
                case NPTypeCode.Int16:   *(short*)dest = (short)value; break;
                case NPTypeCode.UInt16:  *(ushort*)dest = (ushort)value; break;
                case NPTypeCode.Int32:   *(int*)dest = (int)value; break;
                case NPTypeCode.UInt32:  *(uint*)dest = (uint)value; break;
                case NPTypeCode.Int64:   *(long*)dest = (long)value; break;
                case NPTypeCode.UInt64:  *(ulong*)dest = (ulong)value; break;
                case NPTypeCode.Char:    *(char*)dest = (char)value; break;
                case NPTypeCode.Half:    *(Half*)dest = (Half)value; break;
                case NPTypeCode.Single:  *(float*)dest = (float)value; break;
                case NPTypeCode.Double:  *(double*)dest = (double)value; break;
                case NPTypeCode.Decimal: *(decimal*)dest = (decimal)value; break;
                case NPTypeCode.Complex: *(System.Numerics.Complex*)dest = (System.Numerics.Complex)value; break;
                default: throw new NotSupportedException($"WriteScalar: type {typeCode} not supported");
            }
        }

        private static NDArray EnsureContiguousOfType(NDArray src, NPTypeCode target)
        {
            if (src.typecode == target && src.Shape.IsContiguous)
                return src;
            var typed = src.typecode == target ? src : src.astype(target, copy: true);
            return typed.Shape.IsContiguous ? typed : typed.copy();
        }

        private static NDArray EnsureContiguousInt64(NDArray sorter)
        {
            if (sorter.typecode == NPTypeCode.Int64 && sorter.Shape.IsContiguous)
                return sorter;
            var typed = sorter.typecode == NPTypeCode.Int64 ? sorter : sorter.astype(NPTypeCode.Int64, copy: true);
            return typed.Shape.IsContiguous ? typed : typed.copy();
        }

        private static void ValidateSearchSorted(NDArray a, string side, NDArray sorter)
        {
            if (side != "left" && side != "right")
                throw new ArgumentException($"search side must be 'left' or 'right' (got '{side}')", nameof(side));

            if (a.ndim > 1)
                throw new ArgumentException("object too deep for desired array", nameof(a));

            if (sorter is not null)
            {
                if (sorter.ndim != 1)
                    throw new ArgumentException("sorter must be 1-D array", nameof(sorter));
                if (sorter.size != a.size)
                    throw new ArgumentException("sorter.size must equal a.size", nameof(sorter));
            }
        }
    }
}
