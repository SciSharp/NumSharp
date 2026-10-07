using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray ReduceAMax(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null)
        {
            // Text, or a string dtype= request: NumPy's max checks its arguments first (the axis, and the dtype= class where
            // NumPy reads it), then refuses with its verbatim text or computes on text (the next phase).
            Strings.NDStringUfuncs.ThrowIfReduce("max", arr, axis_, dtype);
            NPTypeCode? typeCode = UfuncLoopCode(dtype);
            var shape = arr.Shape;

            // Handle empty arrays - need to check axis-specific behavior
            if (shape.IsEmpty || shape.size == 0)
            {
                return HandleEmptyArrayMinMaxReduction(arr, axis_, keepdims, typeCode, "maximum");
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode, null);

            // float16 without a dtype= cast: NumPy's sequential HALF_maximum fold in its visiting order, bit for bit on every
            // layout (first NaN verbatim, first ±0) — Default.Reduction.MinMax.Half.cs, flat and per axis. A reduced axis of
            // extent 1 copies its lone elements there too, exactly as HandleTrivialAxisReduction would.
            if (arr.GetTypeCode == NPTypeCode.Half && (typeCode == null || typeCode == NPTypeCode.Half))
                return HalfMinMaxReduce(arr, axis_, keepdims, isMax: true, nanIgnore: false);

            // complex128 without a dtype= cast: NumPy's sequential CDOUBLE_maximum loop in NumPy's visiting order (the first
            // NaN wins verbatim, a tie keeps the earlier element), through the reduction iterator of the keyword path
            // (Default.Reduction.MinMax.Keywords.cs). The IL kernels below walk memory order, so a reversed axis returned
            // another NaN's bits.
            if (arr.GetTypeCode == NPTypeCode.Complex && (typeCode == null || typeCode == NPTypeCode.Complex))
                return MaxMinNumPy(arr, axis_, null, keepdims, null, null, MinMaxOp.Max);

            // A broadcast input (a stride-0 axis) without a dtype= cast: the exact engines decline it (a zero stride changes
            // NpyIter's axis sort — its ambiguous-comparison rule), and the IL kernels below match only the value; the
            // reduction iterator ports that sort, so the ±0 tie and the NaN bits are NumPy's here too.
            if (shape.IsBroadcasted && (typeCode == null || typeCode == arr.GetTypeCode))
                return MaxMinNumPy(arr, axis_, null, keepdims, null, null, MinMaxOp.Max);

            // bool reduced along its innermost axis (the smallest stride: one NumPy reduce-form call per output): NumPy's
            // BOOL_logical_or stops at the first deciding byte of each row, and so does the keyword path's bool loop, while
            // the IL axis kernel below reads every byte (0.19x NumPy on rows of 10,000). Both answers are exact (the op is
            // order-free), so this is the faster route only.
            if (arr.GetTypeCode == NPTypeCode.Boolean && axis_ != null && (typeCode == null || typeCode == NPTypeCode.Boolean)
                && ReducedAxisIsInnermost(shape, axis_.Value))
                return MaxMinNumPy(arr, axis_, null, keepdims, null, null, MinMaxOp.Max);

            if (axis_ == null)
            {
                // NumPy's exact flat schedule (coalesced runs, NumPy's buffering decision, chained contiguous / strided
                // calls — Default.Reduction.MinMax.Exact.cs) for a same-dtype reduction, so the float ±0-tie and
                // NaN-payload bits equal NumPy's on every non-broadcast layout; a dtype= request, a broadcast input or an
                // unsupported dtype declines (null) to the IL kernel below, as before.
                var r = typeCode == null || typeCode == arr.GetTypeCode
                    ? TryExactFlatMinMaxScalar(arr, MinMaxOp.Max)
                    : null;
                r ??= NDArray.Scalar(max_elementwise_il(arr, typeCode));
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
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
                var exact = TryExactAxisMinMax(arr, axis, MinMaxOp.Max);
                if (exact is not null)
                {
                    if (keepdims)
                        exact.Storage.ExpandDimension(axis);
                    // Same PyArray_Return rule as ExecuteAxisReduction: a fresh 0-d result is a read-only scalar.
                    return exact.MarkReductionScalar();
                }
            }

            return ExecuteAxisReduction(arr, axis, keepdims, outputType, null, ReductionOp.Max);
        }
    }
}
