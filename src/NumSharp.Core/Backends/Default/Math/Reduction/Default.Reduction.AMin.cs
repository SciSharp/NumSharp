using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.min</c> / <c>np.amin</c> (<c>np.minimum.reduce</c>) over <paramref name="axis_"/>, or over
        ///     every axis when it is null.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The reduced axis (negative values count from the end), or null for every axis. A 0-d input
        ///     accepts 0 and -1 (NumPy's backward-compatibility rule) and nothing else.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the input's.</param>
        /// <param name="out">NumPy's <c>out=</c>, or null; any keyword takes <see cref="MaxMinNumPy"/>.</param>
        /// <param name="initial">NumPy's <c>initial=</c>, or null.</param>
        /// <param name="where">NumPy's <c>where=</c> bool mask, or null.</param>
        /// <returns>The minimum (a read-only 0-d scalar for a full reduction), or
        ///     <paramref name="out"/> itself when given.</returns>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> together with a keyword.</exception>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range — checked before any shortcut.</exception>
        /// <exception cref="ArgumentException">An empty reduction ("zero-size array to reduction operation minimum which has
        ///     no identity", NumPy's ValueError).</exception>
        /// <exception cref="TypeError">Fixed-width text, or a string <paramref name="dtype"/>: NumPy's refusal (its verbatim
        ///     text). A <c>StringDType</c> input computes.</exception>
        /// <exception cref="ValueError">A numeric <paramref name="dtype"/> over text that does not parse as that number
        ///     (NumPy casts the text first: <c>could not convert string to float</c>).</exception>
        /// <remarks><see cref="NDAnchoredAttribute"/>: the reduction engines read <paramref name="arr"/>'s raw address,
        ///     and every front door — <c>np.min</c>, <c>np.amin</c>, <c>ndarray.min</c>, <see cref="AMin"/> — reaches them
        ///     through this method, which opens no scope; the weaver roots <paramref name="arr"/> until it returns (see
        ///     <see cref="ReduceAMax"/>).</remarks>
        [NDAnchored]
        public override NDArray ReduceAMin(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null,
            NDArray @out = null, object initial = null, NDArray @where = null)
        {
            // NumPy's out= / initial= / where=: ufunc.reduce's keyword path, the reduction iterator and the seeded exact
            // engines (MaxMinNumPy), whose prologue checks the arguments in NumPy's order. A keyword-less call keeps the
            // route below. dtype= has no keyword route: the keyword path's loop is promote_types(out, arr).
            if (@out is not null || initial is not null || @where is not null)
            {
                if (dtype is not null)
                    throw new NotSupportedException(
                        "TensorEngine.ReduceAMin: dtype= together with out=, initial= or where= is not implemented " +
                        "(np.minimum.reduce takes them together; the keyword path computes in promote_types(out, arr))");
                return MaxMinNumPy(arr, axis_, @out, keepdims, initial, @where, MinMaxOp.Min);
            }
            // Text, or a string dtype= request: NumPy's min checks its arguments first (the axis, and the dtype= class where
            // NumPy reads it), then refuses with its verbatim text (fixed width) or computes on the text (StringDType).
            // A numeric dtype= over text: NumPy casts the text to that number type first (a parse), then reduces the numbers.
            if (Strings.NDStringUfuncs.ParseForNumericReduce("min", arr, axis_, dtype) is { } parsed)
                return Strings.NDStringUfuncs.ReduceParsed(parsed, p => ReduceAMin(p, axis_, keepdims, dtype));
            // StringDType: NumPy's minimum.reduce computes (strings plan D18) once its checks passed.
            if (Strings.NDStringUfuncs.VStringReduces("min", arr, axis_, dtype))
                return Strings.NDVStringReduction.MaxMin(false, arr, axis_, keepdims, null, null, null);
            Strings.NDStringUfuncs.ThrowIfReduce("min", arr, axis_, dtype);
            // The ufunc reduce's axis check, before every shortcut below (empty, 0-d and one-element inputs included).
            ReductionArguments.CheckReduceAxis(arr.ndim, axis_);
            NPTypeCode? typeCode = UfuncLoopCode(dtype);
            var shape = arr.Shape;

            // Handle empty arrays - need to check axis-specific behavior
            if (shape.IsEmpty || shape.size == 0)
            {
                return HandleEmptyArrayMinMaxReduction(arr, axis_, keepdims, typeCode, "minimum");
            }

            if (shape.IsScalar || (shape.size == 1 && shape.NDim == 1))
                return HandleScalarReduction(arr, keepdims, typeCode, null);

            // float16 without a dtype= cast: NumPy's sequential HALF_minimum fold in its visiting order, bit for bit on every
            // layout (first NaN verbatim, first ±0) — Default.Reduction.MinMax.Half.cs, flat and per axis. A reduced axis of
            // extent 1 copies its lone elements there too, exactly as HandleTrivialAxisReduction would.
            if (arr.GetTypeCode == NPTypeCode.Half && (typeCode == null || typeCode == NPTypeCode.Half))
                return HalfMinMaxReduce(arr, axis_, keepdims, isMax: false, nanIgnore: false);

            // complex128 without a dtype= cast: NumPy's sequential CDOUBLE_minimum loop in NumPy's visiting order (the first
            // NaN wins verbatim, a tie keeps the earlier element), through the reduction iterator of the keyword path
            // (Default.Reduction.MinMax.Keywords.cs). The IL kernels below walk memory order, so a reversed axis returned
            // another NaN's bits.
            if (arr.GetTypeCode == NPTypeCode.Complex && (typeCode == null || typeCode == NPTypeCode.Complex))
                return MaxMinNumPy(arr, axis_, null, keepdims, null, null, MinMaxOp.Min);

            // A broadcast input (a stride-0 axis) without a dtype= cast: the exact engines decline it (a zero stride changes
            // NpyIter's axis sort — its ambiguous-comparison rule), and the IL kernels below match only the value; the
            // reduction iterator ports that sort, so the ±0 tie and the NaN bits are NumPy's here too.
            if (shape.IsBroadcasted && (typeCode == null || typeCode == arr.GetTypeCode))
                return MaxMinNumPy(arr, axis_, null, keepdims, null, null, MinMaxOp.Min);

            // bool reduced along its innermost axis (the smallest stride: one NumPy reduce-form call per output): NumPy's
            // BOOL_logical_and stops at the first deciding byte of each row, and so does the keyword path's bool loop, while
            // the IL axis kernel below reads every byte (0.19x NumPy on rows of 10,000). Both answers are exact (the op is
            // order-free), so this is the faster route only.
            if (arr.GetTypeCode == NPTypeCode.Boolean && axis_ != null && (typeCode == null || typeCode == NPTypeCode.Boolean)
                && ReducedAxisIsInnermost(shape, axis_.Value))
                return MaxMinNumPy(arr, axis_, null, keepdims, null, null, MinMaxOp.Min);

            if (axis_ == null)
            {
                // The mirror of ReduceAMax's flat branch: NumPy's exact flat schedule for a same-dtype reduction, the IL
                // kernel for everything it declines.
                var r = typeCode == null || typeCode == arr.GetTypeCode
                    ? TryExactFlatMinMaxScalar(arr, MinMaxOp.Min)
                    : null;
                r ??= NDArray.Scalar(min_elementwise_il(arr, typeCode));
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
