using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            var shape = arr.Shape;

            // Handle empty arrays - need to check axis-specific behavior
            if (shape.IsEmpty || shape.size == 0)
            {
                return HandleEmptyArrayMinMaxReduction(arr, axis_, keepdims, typeCode, "minimum");
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode, null);

            if (axis_ == null)
            {
                // The mirror of ReduceAMax's flat branch: NumPy's exact flat schedule for a same-dtype reduction, the IL
                // kernel for everything it declines.
                var r = typeCode == null || typeCode == arr.GetTypeCode
                    ? TryExactFlatMinMaxScalar(arr, MinMaxOp.Min)
                    : null;
                r ??= NDArray.Scalar(min_elementwise_il(arr, typeCode));
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r.MarkReductionScalar();
            }

            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = typeCode ?? arr.GetTypeCode;

            if (shape[axis] == 1)
                return HandleTrivialAxisReduction(arr, axis, keepdims, outputType, null);

            // NumPy's exact per-element schedule (ROW simd_reduce_c / strided 8-accumulator unroll / SLAB sequential
            // fold — see Default.Reduction.MinMax.Exact.cs): the float ±0-tie and NaN-payload bits equal NumPy's on
            // every non-broadcast layout. Only a same-dtype reduction qualifies (a dtype= request keeps the casting
            // kernel); null = declined (broadcast input, unsupported dtype) → the kernels below, as before.
            if (outputType == arr.GetTypeCode)
            {
                var exact = TryExactAxisMinMax(arr, axis, MinMaxOp.Min);
                if (exact is not null)
                {
                    if (keepdims)
                        exact.Storage.ExpandDimension(axis);
                    // Same PyArray_Return rule as ExecuteAxisReduction: a fresh 0-d result is a read-only scalar.
                    return exact.MarkReductionScalar();
                }
            }

            return ExecuteAxisReduction(arr, axis, keepdims, outputType, null, ReductionOp.Min);
        }
    }
}
