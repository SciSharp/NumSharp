using System;
using NumSharp.Backends.Kernels;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override unsafe NDArray ReduceCumMul(NDArray arr, int? axis_, NPTypeCode? typeCode = null)
        {
            // NumPy: cumprod on boolean arrays treats True as 1 and False as 0, returning int64
            // Convert boolean input to int64 to match NumPy behavior
            if (arr.GetTypeCode == NPTypeCode.Boolean)
            {
                var int64Arr = arr.astype(NPTypeCode.Int64, copy: true);
                return ReduceCumMul(int64Arr, axis_, typeCode ?? NPTypeCode.Int64);
            }

            var shape = arr.Shape;
            if (shape.IsEmpty || shape.size == 0)
                return arr;

            if (shape.IsScalar || shape.size == 1 && shape.dimensions.Length == 1)
                return typeCode.HasValue ? Cast(arr, typeCode.Value, copy: true) : arr.Clone();

            if (axis_ == null)
            {
                var r = cumprod_elementwise(arr, typeCode);
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
                //if the given div axis is 1 - cumprod is just the value itself
                //Return a copy to avoid sharing memory with the original (NumPy behavior)
                return arr.copy();
            }

            // For broadcast arrays, we iterate over the input (which has stride=0 for broadcast dims)
            // but write to a contiguous output array.
            NDArray inputArr = arr;

            // Create output with CONTIGUOUS strides even if input is broadcast.
            var outputShape = new Shape(shape.dimensions);  // Fresh contiguous shape
            var retTypeCode = typeCode ?? (inputArr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retTypeCode, outputShape, false);

            // Fast path: use IL-generated axis kernel when available
            // Note: We only use the IL kernel for contiguous arrays without offset, as it doesn't
            // handle negative strides or offset-based views correctly.
            if (ILKernelGenerator.Enabled && !shape.IsBroadcasted && shape.IsContiguous && shape.offset == 0)
            {
                bool innerAxisContiguous = (axis == arr.ndim - 1) && (arr.strides[axis] == 1);
                var key = new CumulativeAxisKernelKey(inputArr.GetTypeCode, retTypeCode, ReductionOp.CumProd, innerAxisContiguous);
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

            // Fallback: iterator-based axis cumprod (handles broadcast, non-contiguous, edge cases)
            return ExecuteAxisCumProdFallback(inputArr, ret, axis);
        }

        /// <summary>
        /// Fallback axis cumprod on the new axis iterator path.
        /// </summary>
        private unsafe NDArray ExecuteAxisCumProdFallback(NDArray inputArr, NDArray ret, int axis)
        {
            var retType = ret.GetTypeCode;

            if (inputArr.GetTypeCode != retType)
                inputArr = Cast(inputArr, retType, copy: true);

            switch (retType)
            {
                case NPTypeCode.Byte:
                    NpyAxisIter.ExecuteSameType<byte, CumProdAxisKernel<byte>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.SByte:
                    NpyAxisIter.ExecuteSameType<sbyte, CumProdAxisKernel<sbyte>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Int16:
                    NpyAxisIter.ExecuteSameType<short, CumProdAxisKernel<short>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.UInt16:
                    NpyAxisIter.ExecuteSameType<ushort, CumProdAxisKernel<ushort>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Int32:
                    NpyAxisIter.ExecuteSameType<int, CumProdAxisKernel<int>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.UInt32:
                    NpyAxisIter.ExecuteSameType<uint, CumProdAxisKernel<uint>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Int64:
                    NpyAxisIter.ExecuteSameType<long, CumProdAxisKernel<long>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.UInt64:
                    NpyAxisIter.ExecuteSameType<ulong, CumProdAxisKernel<ulong>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Half:
                    NpyAxisIter.ExecuteSameType<Half, CumProdAxisKernel<Half>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Single:
                    NpyAxisIter.ExecuteSameType<float, CumProdAxisKernel<float>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Double:
                    NpyAxisIter.ExecuteSameType<double, CumProdAxisKernel<double>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Decimal:
                    NpyAxisIter.ExecuteSameType<decimal, CumProdAxisKernel<decimal>>(inputArr.Storage, ret.Storage, axis);
                    break;
                case NPTypeCode.Complex:
                    NpyAxisIter.ExecuteSameType<System.Numerics.Complex, CumProdAxisKernel<System.Numerics.Complex>>(inputArr.Storage, ret.Storage, axis);
                    break;
                default:
                    throw new NotSupportedException($"Axis cumprod output type {retType} not supported");
            }

            return ret;
        }

        protected unsafe NDArray cumprod_elementwise(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            if (!arr.Shape.IsContiguous)
                return cumprod_elementwise(arr.copy(), typeCode);

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());

            // Fast path: use IL-generated kernel for contiguous arrays
            if (arr.Shape.IsContiguous && ILKernelGenerator.Enabled)
            {
                var ret = new NDArray(retType, Shape.Vector(arr.size));
                var key = new CumulativeKernelKey(arr.GetTypeCode, retType, ReductionOp.CumProd, IsContiguous: true);
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

            // Fallback: contiguous prefix-product loop
            return cumprod_elementwise_fallback(arr, retType);
        }

        /// <summary>
        /// Fallback element-wise cumprod for contiguous input.
        /// </summary>
        private unsafe NDArray cumprod_elementwise_fallback(NDArray arr, NPTypeCode retType)
        {
            if (!arr.Shape.IsContiguous)
                throw new InvalidOperationException("cumprod_elementwise_fallback requires contiguous input.");

            var linearInput = arr.reshape(Shape.Vector(arr.size));
            var converted = linearInput.typecode == retType
                ? linearInput.Clone()
                : Cast(linearInput, retType, copy: true);

            switch (retType)
            {
                case NPTypeCode.Byte:
                {
                    var addr = (byte*)converted.Address;
                    byte product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.SByte:
                {
                    var addr = (sbyte*)converted.Address;
                    sbyte product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Int16:
                {
                    var addr = (short*)converted.Address;
                    short product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.UInt16:
                {
                    var addr = (ushort*)converted.Address;
                    ushort product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Int32:
                {
                    var addr = (int*)converted.Address;
                    int product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.UInt32:
                {
                    var addr = (uint*)converted.Address;
                    uint product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Int64:
                {
                    var addr = (long*)converted.Address;
                    long product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.UInt64:
                {
                    var addr = (ulong*)converted.Address;
                    ulong product = 1;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Single:
                {
                    var addr = (float*)converted.Address;
                    float product = 1f;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Half:
                {
                    var addr = (Half*)converted.Address;
                    Half product = (Half)1.0f;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Double:
                {
                    var addr = (double*)converted.Address;
                    double product = 1.0;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Decimal:
                {
                    var addr = (decimal*)converted.Address;
                    decimal product = 1m;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                case NPTypeCode.Complex:
                {
                    var addr = (System.Numerics.Complex*)converted.Address;
                    var product = System.Numerics.Complex.One;
                    for (long i = 0; i < converted.size; i++)
                    {
                        product *= addr[i];
                        addr[i] = product;
                    }
                    break;
                }
                default:
                    throw new NotSupportedException($"CumProd output type {retType} not supported");
            }

            return converted;
        }
    }
}
