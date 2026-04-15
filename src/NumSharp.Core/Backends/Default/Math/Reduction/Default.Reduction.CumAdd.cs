using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;

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
            // Note: We only use the IL kernel for contiguous arrays without offset, as it doesn't
            // handle negative strides or offset-based views correctly.
            if (ILKernelGenerator.Enabled && !shape.IsBroadcasted && shape.IsContiguous && shape.offset == 0)
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
            return ExecuteAxisCumSumFallback(inputArr, ret, axis);
        }

        /// <summary>
        /// Fallback axis cumsum on the new axis iterator path.
        /// </summary>
        private unsafe NDArray ExecuteAxisCumSumFallback(NDArray inputArr, NDArray ret, int axis)
        {
            var retType = ret.GetTypeCode;

            if (inputArr.GetTypeCode != retType)
                inputArr = Cast(inputArr, retType, copy: true);

            switch (retType)
            {
                case NPTypeCode.Byte:
                    NpyAxisIter.ExecuteSameType<byte, CumSumAxisKernel<byte>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.SByte:
                    NpyAxisIter.ExecuteSameType<sbyte, CumSumAxisKernel<sbyte>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Int16:
                    NpyAxisIter.ExecuteSameType<short, CumSumAxisKernel<short>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.UInt16:
                    NpyAxisIter.ExecuteSameType<ushort, CumSumAxisKernel<ushort>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Int32:
                    NpyAxisIter.ExecuteSameType<int, CumSumAxisKernel<int>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.UInt32:
                    NpyAxisIter.ExecuteSameType<uint, CumSumAxisKernel<uint>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Int64:
                    NpyAxisIter.ExecuteSameType<long, CumSumAxisKernel<long>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.UInt64:
                    NpyAxisIter.ExecuteSameType<ulong, CumSumAxisKernel<ulong>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Half:
                    NpyAxisIter.ExecuteSameType<Half, CumSumAxisKernel<Half>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Single:
                    NpyAxisIter.ExecuteSameType<float, CumSumAxisKernel<float>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Double:
                    NpyAxisIter.ExecuteSameType<double, CumSumAxisKernel<double>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Decimal:
                    NpyAxisIter.ExecuteSameType<decimal, CumSumAxisKernel<decimal>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Complex:
                    NpyAxisIter.ExecuteSameType<System.Numerics.Complex, CumSumAxisKernel<System.Numerics.Complex>>(inputArr.Storage, ret.Storage, axis);
                    break;
                default:
                    throw new NotSupportedException($"Axis cumsum output type {retType} not supported");
            }

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

            if (!arr.Shape.IsContiguous)
                return cumsum_elementwise(arr.copy(), typeCode);

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());

            // Fast path: use IL-generated kernel for contiguous arrays
            if (arr.Shape.IsContiguous && ILKernelGenerator.Enabled)
            {
                var ret = new NDArray(retType, Shape.Vector(arr.size));
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

            // Fallback: contiguous prefix-sum loop
            return cumsum_elementwise_fallback(arr, retType);
        }

        /// <summary>
        /// Fallback element-wise cumsum for contiguous input.
        /// </summary>
        private unsafe NDArray cumsum_elementwise_fallback(NDArray arr, NPTypeCode retType)
        {
            if (!arr.Shape.IsContiguous)
                throw new InvalidOperationException("cumsum_elementwise_fallback requires contiguous input.");

            var linearInput = arr.reshape(Shape.Vector(arr.size));
            var converted = linearInput.typecode == retType
                ? linearInput.Clone()
                : Cast(linearInput, retType, copy: true);

            switch (retType)
            {
                case NPTypeCode.Byte:
                {
                    var addr = (byte*)converted.Address;
                    byte sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.SByte:
                {
                    var addr = (sbyte*)converted.Address;
                    sbyte sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var addr = (short*)converted.Address;
                    short sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var addr = (ushort*)converted.Address;
                    ushort sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var addr = (int*)converted.Address;
                    int sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var addr = (uint*)converted.Address;
                    uint sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var addr = (long*)converted.Address;
                    long sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var addr = (ulong*)converted.Address;
                    ulong sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Single:
                {
                    var addr = (float*)converted.Address;
                    float sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Half:
                {
                    var addr = (Half*)converted.Address;
                    Half sum = Half.Zero;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Double:
                {
                    var addr = (double*)converted.Address;
                    double sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    var addr = (decimal*)converted.Address;
                    decimal sum = 0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    var addr = (System.Numerics.Complex*)converted.Address;
                    var sum = System.Numerics.Complex.Zero;
                    for (long i = 0; i < converted.size; i++)
                    {
                        sum += addr[i];
                        addr[i] = sum;
                    }
                    break;
                }
                default:
                    throw new NotSupportedException($"CumSum output type {retType} not supported");
            }

            return converted;
        }
    }
}
