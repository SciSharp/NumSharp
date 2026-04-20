using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override unsafe NDArray ReduceCumAdd(NDArray arr, int? axis_, NPTypeCode? typeCode = null)
        {
            // NumPy: cumsum on boolean arrays treats True as 1 and False as 0, returning int64
            // Convert boolean input to int64 to match NumPy behavior
            if (arr.GetTypeCode == NPTypeCode.Boolean)
            {
                var int64Arr = arr.astype(NPTypeCode.Int64, copy: true);
                return ReduceCumAdd(int64Arr, axis_, typeCode ?? NPTypeCode.Int64);
            }

            //in order to iterate an axis:
            //consider arange shaped (1,2,3,4) when we want to summarize axis 1 (2nd dimension which its value is 2)
            //the size of the array is [1, 2, n, m] all shapes after 2nd multiplied gives size
            //the size of what we need to reduce is the size of the shape of the given axis (shape[axis])
            var shape = arr.Shape;
            if (shape.IsEmpty || shape.size == 0)
                return arr;

            if (shape.IsScalar || shape.size == 1 && shape.dimensions.Length == 1)
                return typeCode.HasValue ? Cast(arr, typeCode.Value, copy: true) : arr.Clone();

            if (axis_ == null)
            {
                var r = cumsum_elementwise(arr, typeCode);
                if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1)
                    r.Storage.Reshape(Shape.Scalar);
                return r;
            }

            var axis = axis_.Value;
            while (axis < 0)
                axis = arr.ndim + axis; //handle negative axis

            if (axis >= arr.ndim)
                throw new ArgumentOutOfRangeException(nameof(axis));

            if (shape[axis] == 1)
            {
                //if the given div axis is 1 - cumsum is just the value itself
                //Return a copy to avoid sharing memory with the original (NumPy behavior)
                return arr.copy();
            }

            // For broadcast arrays, we iterate over the input (which has stride=0 for broadcast dims)
            // but write to a contiguous output array. Key insight:
            // - Input: may have stride=0 (broadcast) - read same value multiple times
            // - Output: must be contiguous - write unique values to distinct memory locations
            NDArray inputArr = arr;

            // Create output with CONTIGUOUS strides even if input is broadcast.
            // Use dimensions only, not the input shape's strides.
            var outputShape = new Shape(shape.dimensions);  // Fresh contiguous shape
            var retTypeCode = typeCode ?? (inputArr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retTypeCode, outputShape, false);

            // Fast path: use IL-generated axis kernel when available
            // This avoids the overhead of iterator-based slicing and provides direct pointer access.
            // B6: Half and Complex aren't handled by the internal AxisCumSumSameType/General helpers
            // (they throw NotSupportedException at execution time, not creation time, so the kernel
            // cache returns a non-null delegate that then throws on first call). Skip the fast path
            // for these types and go straight to the iterator-based fallback.
            if (ILKernelGenerator.Enabled && !shape.IsBroadcasted
                && inputArr.GetTypeCode != NPTypeCode.Half
                && inputArr.GetTypeCode != NPTypeCode.Complex)
            {
                bool innerAxisContiguous = (axis == arr.ndim - 1) && (arr.strides[axis] == 1);
                var key = new CumulativeAxisKernelKey(inputArr.GetTypeCode, retTypeCode, ReductionOp.CumSum, innerAxisContiguous);
                var kernel = ILKernelGenerator.TryGetCumulativeAxisKernel(key);
                if (kernel != null)
                {
                    fixed (long* inputStrides = arr.strides)
                    fixed (long* shapePtr = arr.shape)
                    {
                        kernel((void*)arr.Address, (void*)ret.Address, inputStrides, shapePtr, axis, arr.ndim, arr.size);
                    }
                    return ret;
                }
            }

            // Fallback: iterator-based axis cumsum (handles broadcast, non-contiguous, edge cases)
            return ExecuteAxisCumSumFallback(inputArr, ret, shape, axis);
        }

        /// <summary>
        /// Fallback axis cumsum using iterators. Used when IL kernel not available.
        /// Handles broadcast arrays and type conversions safely.
        /// </summary>
        private unsafe NDArray ExecuteAxisCumSumFallback(NDArray inputArr, NDArray ret, Shape shape, int axis)
        {
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var slices = iterAxis.Slices;
            var retType = ret.GetTypeCode;

            // B6: Complex cumsum must preserve imaginary part (AsIterator<double> would drop it).
            if (retType == NPTypeCode.Complex)
            {
                do
                {
                    var inputSlice = inputArr[slices];
                    var outputSlice = ret[slices];
                    var inputIter = inputSlice.AsIterator<System.Numerics.Complex>();
                    var sum = System.Numerics.Complex.Zero;
                    long idx = 0;
                    while (inputIter.HasNext())
                    {
                        sum += inputIter.MoveNext();
                        outputSlice.SetAtIndex(sum, idx++);
                    }
                } while (iterAxis.Next() != null);
                return ret;
            }

            // Use type-specific iteration based on return type
            // This handles type promotion correctly (e.g., int32 input -> int64 output)
            do
            {
                var inputSlice = inputArr[slices];
                var outputSlice = ret[slices];

                // Get input as double for uniform accumulation
                var inputIter = inputSlice.AsIterator<double>();
                var moveNext = inputIter.MoveNext;
                var hasNext = inputIter.HasNext;

                // Write to output with proper type handling
                double sum = 0;
                long idx = 0;
                while (hasNext())
                {
                    sum += moveNext();
                    // Use SetAtIndex with coordinate calculation for proper slice handling
                    outputSlice.SetAtIndex(Converts.ChangeType(sum, retType), idx++);
                }
            } while (iterAxis.Next() != null);

            return ret;
        }

        public NDArray CumSumElementwise<T>(NDArray arr, NPTypeCode? typeCode) where T : unmanaged
        {
            var ret = cumsum_elementwise(arr, typeCode);
            return typeCode.HasValue && typeCode.Value != ret.typecode ? ret.astype(typeCode.Value, true) : ret;
        }

        protected unsafe NDArray cumsum_elementwise(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retType, Shape.Vector(arr.size));

            // Fast path: use IL-generated kernel for contiguous arrays
            if (arr.Shape.IsContiguous && ILKernelGenerator.Enabled)
            {
                var key = new CumulativeKernelKey(arr.GetTypeCode, retType, ReductionOp.CumSum, IsContiguous: true);
                var kernel = ILKernelGenerator.TryGetCumulativeKernel(key);
                if (kernel != null)
                {
                    fixed (long* strides = arr.strides)
                    fixed (long* shape = arr.shape)
                    {
                        kernel((void*)arr.Address, (void*)ret.Address, strides, shape, arr.ndim, arr.size);
                    }
                    return ret;
                }
            }

            // Fallback: iterator-based element-wise cumsum
            return cumsum_elementwise_fallback(arr, ret, retType);
        }

        /// <summary>
        /// Fallback element-wise cumsum using iterators.
        /// </summary>
        private unsafe NDArray cumsum_elementwise_fallback(NDArray arr, NDArray ret, NPTypeCode retType)
        {
            // Handle Decimal separately for precision
            if (arr.GetTypeCode == NPTypeCode.Decimal && retType == NPTypeCode.Decimal)
            {
                var iter = arr.AsIterator<decimal>();
                var addr = (decimal*)ret.Address;
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                int i = 0;
                decimal sum = 0;
                while (hasNext())
                {
                    sum += moveNext();
                    addr[i++] = sum;
                }
                return ret;
            }

            // Handle Complex separately - requires Complex accumulator
            if (arr.GetTypeCode == NPTypeCode.Complex && retType == NPTypeCode.Complex)
            {
                var iter = arr.AsIterator<System.Numerics.Complex>();
                var addr = (System.Numerics.Complex*)ret.Address;
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                int i = 0;
                var sum = System.Numerics.Complex.Zero;
                while (hasNext())
                {
                    sum += moveNext();
                    addr[i++] = sum;
                }
                return ret;
            }

            // All other types: use double for accumulation, convert at output
            {
                var iter = arr.AsIterator<double>();
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                double sum = 0;
                int i = 0;

                // Write to output based on return type
                switch (retType)
                {
                    case NPTypeCode.Byte:
                    {
                        var addr = (byte*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (byte)sum;
                        }
                        break;
                    }
                    case NPTypeCode.SByte:
                    {
                        var addr = (sbyte*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (sbyte)sum;
                        }
                        break;
                    }
                    case NPTypeCode.Int16:
                    {
                        var addr = (short*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (short)sum;
                        }
                        break;
                    }
                    case NPTypeCode.UInt16:
                    {
                        var addr = (ushort*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (ushort)sum;
                        }
                        break;
                    }
                    case NPTypeCode.Int32:
                    {
                        var addr = (int*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (int)sum;
                        }
                        break;
                    }
                    case NPTypeCode.UInt32:
                    {
                        var addr = (uint*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (uint)sum;
                        }
                        break;
                    }
                    case NPTypeCode.Int64:
                    {
                        var addr = (long*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (long)sum;
                        }
                        break;
                    }
                    case NPTypeCode.UInt64:
                    {
                        var addr = (ulong*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (ulong)sum;
                        }
                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        var addr = (float*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (float)sum;
                        }
                        break;
                    }
                    case NPTypeCode.Half:
                    {
                        var addr = (Half*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (Half)sum;
                        }
                        break;
                    }
                    case NPTypeCode.Double:
                    {
                        var addr = (double*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = sum;
                        }
                        break;
                    }
                    case NPTypeCode.Decimal:
                    {
                        var addr = (decimal*)ret.Address;
                        while (hasNext())
                        {
                            sum += moveNext();
                            addr[i++] = (decimal)sum;
                        }
                        break;
                    }
                    default:
                        throw new NotSupportedException($"CumSum output type {retType} not supported");
                }
                return ret;
            }
        }
    }
}
