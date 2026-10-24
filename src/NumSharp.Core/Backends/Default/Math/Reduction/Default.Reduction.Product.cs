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
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the accumulator of the input (for
        ///     <c>promote_types(out, arr)</c> when <paramref name="out"/> is given).</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape, or null for a fresh result: NumPy's
        ///     <c>multiply.reduce</c> with an out operand (<see cref="ReduceIntoOut"/>) — the text routes, the reduce
        ///     prologue, the out= loop rule, the unsafe store.</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the product (a read-only 0-d scalar for a full
        ///     reduction).</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range — checked before any shortcut.</exception>
        /// <exception cref="TypeError">Text, or a string <paramref name="dtype"/>: NumPy's refusal (its verbatim text; no
        ///     multiply loop reduces text into itself).</exception>
        /// <exception cref="ValueError">A numeric <paramref name="dtype"/> over text that does not parse as that number, or
        ///     a read-only / misshapen <paramref name="out"/> (the reduce prologue's texts).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not broadcast against the input.</exception>
        public override NDArray ReduceProduct(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null)
        {
            // out=: multiply.reduce with an out operand, whose checks and loop differ from the plain call's (the text routes
            // into out, the reduce prologue on (out, arr), promote_types(out, arr) as the loop). It computes the product
            // through this method again, without out.
            if (@out is not null)
                return ReduceIntoOut("prod", arr, axis_, keepdims, dtype, @out);

            // Text, or a string dtype= request: NumPy's prod checks its arguments first (the axis, and the dtype= class where
            // NumPy reads it), then refuses with its verbatim text.
            // A numeric dtype= over text: NumPy casts the text to that number type first (a parse), then reduces the numbers.
            if (Strings.NDStringUfuncs.ParseForNumericReduce("prod", arr, axis_, dtype) is { } parsed)
                return Strings.NDStringUfuncs.ReduceParsed(parsed, p => ReduceProduct(p, axis_, keepdims, dtype));
            Strings.NDStringUfuncs.ThrowIfReduce("prod", arr, axis_, dtype);
            // The ufunc reduce's axis check, before every shortcut below (empty, 0-d and one-element inputs included).
            ReductionArguments.CheckReduceAxis(arr.ndim, axis_);
            NPTypeCode? typeCode = UfuncLoopCode(dtype);
            // A bool LOOP (dtype=bool) is NumPy's BOOL_multiply, logical and, over the input's truth values
            // (Default.Reduction.BoolLoop.cs): the flat product kernels have no bool loop.
            if (typeCode == NPTypeCode.Boolean)
                return LogicalProduct(arr, axis_, keepdims);
            var shape = arr.Shape;

            // A char loop (dtype=char) is NumSharp's uint16 proxy, which the multiply kernels have no loop for (the flat
            // product refused it, the axis product saturated through a double): the product in the uint16 loop, handed back
            // as char (ReduceInCharLoop). The uninitialized shape keeps its identity answer below.
            if (typeCode == NPTypeCode.Char && !shape.IsEmpty)
                return ReduceInCharLoop(arr, axis_, keepdims, product: true);

            // A loop dtype that is neither the input's own nor its default accumulator: NumPy's buffered iterator casts every
            // element into the loop before multiply's loop reads it, so the product of the cast copy in its own dtype IS
            // NumPy's (ReduceProductInCastLoop). The routes below read the input as the loop dtype (the flat float16 /
            // complex128 fallbacks, past the end of a narrower input's buffer) or convert per element with their own rules
            // (a negative integer saturated to 0 in an unsigned loop, where NumPy wraps it), gap MX08 of the matrix tier.
            // The uninitialized shape and a zero-size input read nothing and keep their identity answers below.
            if (typeCode is { } loopCode && !shape.IsEmpty && shape.size > 0
                && loopCode != arr.GetTypeCode && loopCode != arr.GetTypeCode.GetAccumulatingType())
                return ReduceProductInCastLoop(arr, axis_, keepdims, loopCode);

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

        /// <summary>
        ///     <see cref="ReduceProduct"/> in a cast loop — a <c>dtype=</c> that is neither the input's own dtype nor its
        ///     accumulator: the input cast into the loop (NumPy's buffered cast, KEEPORDER like <c>astype</c>), then the
        ///     product of that copy in the loop dtype.
        /// </summary>
        /// <param name="arr">The input (not empty; its dtype differs from the loop's).</param>
        /// <param name="axis">The reduced axis (already checked by the caller), or null for every axis.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <param name="loopType">The loop dtype.</param>
        /// <returns>The product in <paramref name="loopType"/>: a fresh array, or NumPy's read-only scalar for a full
        ///     reduction.</returns>
        /// <remarks>
        ///     <para>The cast is NumPy's own unsafe element cast, so every input value reaches the loop as NumPy's iterator
        ///     hands it over: a negative integer wraps into an unsigned loop (<c>np.prod(int8 [-1, 3, 2], dtype=uint32)</c> is
        ///     4294967290), uint16 65535 rounds to inf in a float16 loop, a complex input keeps its real part in a real loop.
        ///     The routes this replaces read a narrower input as the loop dtype (the flat float16 / complex128 fallbacks:
        ///     garbage from past the end of the buffer) or converted per element with their own rules.</para>
        ///     <para>The copy is KEEPORDER, so a reversed or F-ordered input's elements keep their memory order in it; the
        ///     product of the copy then multiplies in the order the copy's own route walks.</para>
        ///     <para>The copy is this method's intermediate: disposed on every path, so a large cast loop returns its buffer
        ///     to the pool at once.</para>
        /// </remarks>
        private NDArray ReduceProductInCastLoop(NDArray arr, int? axis, bool keepdims, NPTypeCode loopType)
        {
            var cast = Cast(arr, loopType, copy: true);
            try
            {
                // The copy's dtype is the loop's own, so this call takes the input-dtype routes, never this branch again.
                return ReduceProduct(cast, axis, keepdims, loopType);
            }
            finally
            {
                cast.Dispose();
            }
        }
    }
}
