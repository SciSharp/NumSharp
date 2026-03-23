using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

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
            if (ILKernelGenerator.Enabled && !shape.IsBroadcasted)
            {
                bool innerAxisContiguous = (axis == arr.ndim - 1) && (arr.strides[axis] == 1);
                var key = new CumulativeAxisKernelKey(inputArr.GetTypeCode, retTypeCode, ReductionOp.CumProd, innerAxisContiguous);
                var kernel = ILKernelGenerator.TryGetCumulativeAxisKernel(key);
                if (kernel != null)
                {
                    fixed (int* inputStrides = arr.strides)
                    fixed (int* shapePtr = arr.shape)
                    {
                        kernel((void*)arr.Address, (void*)ret.Address, inputStrides, shapePtr, axis, arr.ndim, arr.size);
                    }
                    return ret;
                }
            }

            // Fallback: iterator-based axis cumprod (handles broadcast, non-contiguous, edge cases)
            return ExecuteAxisCumProdFallback(inputArr, ret, shape, axis);
        }

        /// <summary>
        /// Fallback axis cumprod using iterators. Used when IL kernel not available.
        /// Handles broadcast arrays and type conversions safely.
        /// </summary>
        private unsafe NDArray ExecuteAxisCumProdFallback(NDArray inputArr, NDArray ret, Shape shape, int axis)
        {
            var iterAxis = new NDCoordinatesAxisIncrementor(ref shape, axis);
            var slices = iterAxis.Slices;
            var retType = ret.GetTypeCode;

            // Use type-specific iteration based on return type
            do
            {
                var inputSlice = inputArr[slices];
                var outputSlice = ret[slices];

                // Get input as double for uniform accumulation
                var inputIter = inputSlice.AsIterator<double>();
                var moveNext = inputIter.MoveNext;
                var hasNext = inputIter.HasNext;

                // Write to output with proper type handling
                double product = 1.0;
                int idx = 0;
                while (hasNext())
                {
                    product *= moveNext();
                    // Use SetAtIndex with coordinate calculation for proper slice handling
                    outputSlice.SetAtIndex(Converts.ChangeType(product, retType), idx++);
                }
            } while (iterAxis.Next() != null);

            return ret;
        }

        protected unsafe NDArray cumprod_elementwise(NDArray arr, NPTypeCode? typeCode)
        {
            if (arr.Shape.IsScalar || (arr.Shape.NDim == 1 && arr.Shape.size == 1))
                return typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();

            var retType = typeCode ?? (arr.GetTypeCode.GetAccumulatingType());
            var ret = new NDArray(retType, Shape.Vector(arr.size));

            // Fast path: use IL-generated kernel for contiguous arrays
            if (arr.Shape.IsContiguous && ILKernelGenerator.Enabled)
            {
                var key = new CumulativeKernelKey(arr.GetTypeCode, retType, ReductionOp.CumProd, IsContiguous: true);
                var kernel = ILKernelGenerator.TryGetCumulativeKernel(key);
                if (kernel != null)
                {
                    fixed (int* strides = arr.strides)
                    fixed (int* shape = arr.shape)
                    {
                        kernel((void*)arr.Address, (void*)ret.Address, strides, shape, arr.ndim, arr.size);
                    }
                    return ret;
                }
            }

            // Fallback: iterator-based element-wise cumprod
            return cumprod_elementwise_fallback(arr, ret, retType);
        }

        /// <summary>
        /// Fallback element-wise cumprod using iterators.
        /// </summary>
        private unsafe NDArray cumprod_elementwise_fallback(NDArray arr, NDArray ret, NPTypeCode retType)
        {
            // Handle Decimal separately for precision
            if (arr.GetTypeCode == NPTypeCode.Decimal && retType == NPTypeCode.Decimal)
            {
                var iter = arr.AsIterator<decimal>();
                var addr = (decimal*)ret.Address;
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                int i = 0;
                decimal product = 1m;
                while (hasNext())
                {
                    product *= moveNext();
                    addr[i++] = product;
                }
                return ret;
            }

            // All other types: use double for accumulation, convert at output
            {
                var iter = arr.AsIterator<double>();
                var moveNext = iter.MoveNext;
                var hasNext = iter.HasNext;
                double product = 1.0;
                int i = 0;

                // Write to output based on return type
                switch (retType)
                {
                    case NPTypeCode.Byte:
                    {
                        var addr = (byte*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (byte)product;
                        }
                        break;
                    }
                    case NPTypeCode.Int16:
                    {
                        var addr = (short*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (short)product;
                        }
                        break;
                    }
                    case NPTypeCode.UInt16:
                    {
                        var addr = (ushort*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (ushort)product;
                        }
                        break;
                    }
                    case NPTypeCode.Int32:
                    {
                        var addr = (int*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (int)product;
                        }
                        break;
                    }
                    case NPTypeCode.UInt32:
                    {
                        var addr = (uint*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (uint)product;
                        }
                        break;
                    }
                    case NPTypeCode.Int64:
                    {
                        var addr = (long*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (long)product;
                        }
                        break;
                    }
                    case NPTypeCode.UInt64:
                    {
                        var addr = (ulong*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (ulong)product;
                        }
                        break;
                    }
                    case NPTypeCode.Single:
                    {
                        var addr = (float*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (float)product;
                        }
                        break;
                    }
                    case NPTypeCode.Double:
                    {
                        var addr = (double*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = product;
                        }
                        break;
                    }
                    case NPTypeCode.Decimal:
                    {
                        var addr = (decimal*)ret.Address;
                        while (hasNext())
                        {
                            product *= moveNext();
                            addr[i++] = (decimal)product;
                        }
                        break;
                    }
                    default:
                        throw new NotSupportedException($"CumProd output type {retType} not supported");
                }
                return ret;
            }
        }
    }
}
