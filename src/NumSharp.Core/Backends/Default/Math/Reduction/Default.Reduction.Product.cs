using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.prod</c> (<c>np.multiply.reduce</c>): the product of <paramref name="arr"/>'s elements over
        ///     <paramref name="axis_"/>, or over every axis when it is null, in NEP 50's accumulator dtype.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The reduced axis (negative values count from the end), or null for every axis. A 0-d input
        ///     accepts 0 and -1 (NumPy's backward-compatibility rule) and nothing else.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the accumulator of the input.</param>
        /// <returns>The product (a read-only 0-d scalar for a full reduction).</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range — checked before any shortcut.</exception>
        /// <exception cref="StringComputationNotSupportedException">A string input or <paramref name="dtype"/> NumPy
        ///     computes on (the next phase of NumSharp's string support).</exception>
        public override NDArray ReduceProduct(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null)
        {
            // Text, or a string dtype= request: NumPy's prod checks its arguments first (the axis, and the dtype= class where
            // NumPy reads it), then refuses with its verbatim text or computes on text (the next phase).
            // A numeric dtype= over text: NumPy casts the text to that number type first (a parse), then reduces the numbers.
            if (Strings.NDStringUfuncs.ParseForNumericReduce("prod", arr, axis_, dtype) is { } parsed)
                return Strings.NDStringUfuncs.ReduceParsed(parsed, p => ReduceProduct(p, axis_, keepdims, dtype));
            Strings.NDStringUfuncs.ThrowIfReduce("prod", arr, axis_, dtype);
            // The ufunc reduce's axis check, before every shortcut below (empty, 0-d and one-element inputs included).
            ReductionArguments.CheckReduceAxis(arr.ndim, axis_);
            NPTypeCode? typeCode = UfuncLoopCode(dtype);
            var shape = arr.Shape;

            if (shape.IsEmpty)
            {
                // NumPy parity: prod of empty array uses accumulating type (int/bool -> int64/uint64, floats preserved).
                var emptyType = typeCode ?? arr.typecode.GetAccumulatingType();
                return NDArray.Scalar(emptyType.GetOneValue());
            }

            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    var emptyType = typeCode ?? arr.typecode.GetAccumulatingType();
                    var r = NDArray.Scalar(emptyType.GetOneValue());
                    if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
                    return r.MarkReductionScalar();
                }
                var axis = NormalizeAxis(axis_.Value, arr.ndim);
                var resultShape = Shape.GetAxis(shape, axis);
                var result = np.ones(Shape.Adopt(resultShape), typeCode ?? arr.GetTypeCode.GetAccumulatingType());
                if (keepdims)
                {
                    var ks = new long[arr.ndim];
                    for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : resultShape[sd++];
                    result.Storage.Reshape(Shape.Adopt(ks));
                }
                return result;
            }

            // Degenerate flat reduction (0-d scalar or single-element 1-D). NumPy still applies the
            // NEP50 prod accumulator here — prod(int32)->int64, prod(bool)->int64 — REGARDLESS of size,
            // so resolve the accumulating dtype at the call site (mirrors ReduceAdd and the n>=2 path).
            // HandleScalarReduction is shared with amin/amax, which must PRESERVE the input dtype, so the
            // widening is applied here rather than in the helper.
            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode ?? arr.GetTypeCode.GetAccumulatingType(), null);

            if (axis_ == null)
            {
                var result = prod_elementwise_il(arr, typeCode);
                var r = NDArray.Scalar(result);
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r.MarkReductionScalar();
            }

            var axis2 = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();

            if (shape[axis2] == 1)
                return HandleTrivialAxisReduction(arr, axis2, keepdims, outputType, null);

            return ExecuteAxisReduction(arr, axis2, keepdims, outputType, null, ReductionOp.Prod);
        }
    }
}
