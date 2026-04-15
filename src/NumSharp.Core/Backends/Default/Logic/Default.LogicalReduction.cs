using System;
using NumSharp.Backends.Iteration;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public NDArray<bool> All(NDArray nd, int axis, bool keepdims)
            => ReduceLogicalAxis(nd, axis, keepdims, reduceAll: true);

        public NDArray<bool> Any(NDArray nd, int axis, bool keepdims)
            => ReduceLogicalAxis(nd, axis, keepdims, reduceAll: false);

        private NDArray<bool> ReduceLogicalAxis(NDArray nd, int axis, bool keepdims, bool reduceAll)
        {
            if (nd is null)
                throw new ArgumentNullException(nameof(nd));

            if (nd.ndim == 0)
            {
                if (axis == 0 || axis == -1)
                    return np.array(reduceAll ? All(nd) : Any(nd)).MakeGeneric<bool>();

                throw new AxisError(axis, 0);
            }

            axis = NormalizeAxis(axis, nd.ndim);

            var resultShape = CreateLogicalResultShape(nd.Shape, axis, keepdims);
            NDArray<bool> result = CreateLogicalResult(resultShape, reduceAll && nd.Shape.dimensions[axis] == 0);

            if (result.size == 0 || nd.Shape.dimensions[axis] == 0)
                return result;

            switch (nd.GetTypeCode)
            {
                case NPTypeCode.Boolean:
                    ExecuteLogicalAxis<bool>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Byte:
                    ExecuteLogicalAxis<byte>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Int16:
                    ExecuteLogicalAxis<short>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.UInt16:
                    ExecuteLogicalAxis<ushort>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Int32:
                    ExecuteLogicalAxis<int>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.UInt32:
                    ExecuteLogicalAxis<uint>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Int64:
                    ExecuteLogicalAxis<long>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.UInt64:
                    ExecuteLogicalAxis<ulong>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Char:
                    ExecuteLogicalAxis<char>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Single:
                    ExecuteLogicalAxis<float>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Double:
                    ExecuteLogicalAxis<double>(nd, result, axis, reduceAll);
                    break;
                case NPTypeCode.Decimal:
                    ExecuteLogicalAxis<decimal>(nd, result, axis, reduceAll);
                    break;
                default:
                    throw new NotSupportedException($"Type {nd.GetTypeCode} not supported for logical reduction.");
            }

            return result;
        }

        private static Shape CreateLogicalResultShape(Shape inputShape, int axis, bool keepdims)
        {
            if (keepdims)
            {
                var dims = (long[])inputShape.dimensions.Clone();
                dims[axis] = 1;
                return new Shape(dims);
            }

            var reducedDims = Shape.GetAxis(inputShape, axis);
            return reducedDims.Length == 0 ? Shape.Scalar : new Shape(reducedDims);
        }

        private static NDArray<bool> CreateLogicalResult(Shape resultShape, bool fillTrue)
        {
            var result = fillTrue
                ? np.ones(resultShape, NPTypeCode.Boolean)
                : np.zeros(resultShape, NPTypeCode.Boolean);

            return result.MakeGeneric<bool>();
        }

        private static void ExecuteLogicalAxis<T>(NDArray nd, NDArray<bool> result, int axis, bool reduceAll)
            where T : unmanaged
        {
            if (reduceAll)
                NpyAxisIter.ReduceBool<T, NpyAllKernel<T>>(nd.Storage, result.Storage, axis);
            else
                NpyAxisIter.ReduceBool<T, NpyAnyKernel<T>>(nd.Storage, result.Storage, axis);
        }
    }
}
