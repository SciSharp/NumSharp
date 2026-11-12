using System;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     NumPy's <c>np.prod</c> (<c>np.multiply.reduce</c>): the product of <paramref name="arr"/>'s elements over
        ///     <paramref name="axis"/>, or over every axis when it is null, in NEP 50's accumulator dtype.
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis">The reduced axis (negative values count from the end), a tuple of axes reduced together in one
        ///     pass of NumPy's reduction iterator (the empty tuple reduces none: every output is 1 times its element), or null
        ///     for every axis. A 0-d input accepts the INTEGERS 0 and -1 (NumPy's backward-compatibility rule) and nothing
        ///     else; a tuple item gets no such allowance.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the accumulator of the input (for
        ///     <c>promote_types(out, arr)</c> when <paramref name="out"/> is given).</param>
        /// <param name="out">NumPy's <c>out=</c>, an array of the result's shape, or null for a fresh result: NumPy's
        ///     <c>multiply.reduce</c> with an out operand (<see cref="ReduceIntoOut"/>) — the text routes, the reduce
        ///     prologue, the out= loop rule, the unsafe store.</param>
        /// <param name="initial">NumPy's <c>initial=</c>: the start value of every output instead of the identity 1, or null.
        ///     With it (or with <paramref name="where"/>) the call is PyUFunc_ReduceWrapper's keyword path
        ///     (<see cref="AddMulNumPy"/>): NumPy's sequential multiply loop, which can differ in the last bit of a float
        ///     product from this call's own flat loop (NL3).</param>
        /// <param name="where">NumPy's <c>where=</c> mask of the elements to multiply, or null (a 0-d true mask is no
        ///     mask).</param>
        /// <returns><paramref name="out"/> itself when given; otherwise the product (a read-only 0-d scalar for a full
        ///     reduction).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> (or an item of a tuple) is out of range — checked before any
        ///     shortcut.</exception>
        /// <exception cref="TypeError">Text, or a string <paramref name="dtype"/>: NumPy's refusal (its verbatim text; no
        ///     multiply loop reduces text into itself); a C# scalar as <paramref name="out"/>; a complex
        ///     <paramref name="initial"/> for a real loop.</exception>
        /// <exception cref="ValueError">A numeric <paramref name="dtype"/> over text that does not parse as that number, a
        ///     read-only / misshapen <paramref name="out"/> (the reduce prologue's texts), an <paramref name="initial"/>
        ///     the loop cannot hold, or a tuple that names an axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's TypeError text).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> or <paramref name="where"/> does not broadcast
        ///     against the input.</exception>
        /// <exception cref="OverflowError">An integer <paramref name="initial"/> past the loop's range.</exception>
        /// <remarks>
        ///     A complex128 or float16 loop takes the keyword path's reduction iterator even without a keyword (whenever more
        ///     than one element meets an output), because NumPy's result there depends on which inner-loop calls its iterator
        ///     makes: the reduce form (contracted <c>loop_scalar</c> for complex, a float32 accumulator per call for float16)
        ///     against the elementwise form (<c>simd_cmul</c>, a float16 rounding per product). The other loops keep the flat /
        ///     axis kernels.
        /// </remarks>
        public override NDArray ReduceProduct(NDArray arr, NDAxis? axis, bool keepdims = false, DType dtype = null, NDArray @out = null,
            object initial = null, NDArray @where = null)
        {
            // A tuple: one axis or every axis is the integer / None route (NumPy's same flags); any other tuple is
            // multiply.reduce's own statements over the whole set (AddMulNumPy from the identity 1), which raise in NumPy's
            // order (see ReduceAdd).
            if (axis is { IsTuple: true } tuple)
            {
                if (ReductionArguments.TryIntegerForm(arr.ndim, tuple, out int? single))
                    return ReduceProductAlong(arr, single, keepdims, dtype, @out, initial, @where);
                return AddMulNumPy(arr, tuple, keepdims, dtype, @out, initial, IsTrueScalarMask(@where) ? null : @where,
                    multiply: true);
            }
            return ReduceProductAlong(arr, ReductionArguments.IntegerAxis(axis), keepdims, dtype, @out, initial, @where);
        }

        /// <summary>
        ///     The integer-axis and <c>None</c> routes of <see cref="ReduceProduct"/>: NumPy's multiply.reduce over
        ///     <paramref name="axis_"/> (every axis when null), with the kernels and the shortcuts each route keeps (see
        ///     <see cref="ReduceProduct"/> for the whole contract).
        /// </summary>
        /// <param name="arr">The array.</param>
        /// <param name="axis_">The reduced axis (negative values count from the end), or null for every axis; a 0-d input
        ///     accepts 0 and -1.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <param name="dtype">The loop dtype, or null.</param>
        /// <param name="out">NumPy's <c>out=</c>, or null.</param>
        /// <param name="initial">NumPy's <c>initial=</c>, or null.</param>
        /// <param name="where">NumPy's <c>where=</c>, or null.</param>
        /// <returns>The product (see <see cref="ReduceProduct"/>).</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range.</exception>
        /// <exception cref="TypeError">See <see cref="ReduceProduct"/>.</exception>
        /// <exception cref="ValueError">See <see cref="ReduceProduct"/>.</exception>
        /// <exception cref="ArgumentException">See <see cref="ReduceProduct"/>.</exception>
        /// <exception cref="IncorrectShapeException">See <see cref="ReduceProduct"/>.</exception>
        /// <exception cref="OverflowError">See <see cref="ReduceProduct"/>.</exception>
        private NDArray ReduceProductAlong(NDArray arr, int? axis_, bool keepdims, DType dtype, NDArray @out, object initial,
            NDArray @where)
        {
            // initial= / where=: PyUFunc_ReduceWrapper's statements through NumPy's reduction iterator (AddMulNumPy). A 0-d
            // true mask is Python's where=True, no mask; multiply has an identity, so that call is the plain call below.
            if (IsTrueScalarMask(@where))
                @where = null;
            if (initial is not null || @where is not null)
                return AddMulNumPy(arr, axis_, keepdims, dtype, @out, initial, @where, multiply: true);

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

            // A complex128 or float16 LOOP is NumPy's own multiply.reduce, run by the reduction iterator the keyword path
            // uses (AddMulNumPy from the identity 1, the dtype= cast done in its buffer fills). Unlike the other loops these
            // two have no order-free product: each has a REDUCE form, for a call whose out stride is 0 (a flat product, or a
            // row whose reduced axis is innermost), and an ELEMENTWISE form (a slab: the out advances with the input), and
            // they round differently — CDOUBLE_multiply's reduce form is MSVC's contracted loop_scalar, its elementwise form
            // simd_cmul (an imaginary part one ULP apart: parity QA-10); HALF_multiply's reduce form multiplies one call in
            // float32 and narrows once, its elementwise form narrows every product (inf against a finite float16: QA-16). So
            // which calls NumPy makes, and where its buffer cuts a cast loop, is part of the answer, and the iterator ports
            // exactly that. The size-1 shortcuts below keep their copying routes: the complex identity they skip is a
            // separate pin (OpenBugsReductionTests.Prod_OneComplexElement_IsTheIdentityTimesTheElement), and for float16
            // the identity is exact.
            NPTypeCode productLoop = typeCode ?? arr.GetTypeCode.GetAccumulatingType();
            if ((productLoop == NPTypeCode.Complex || productLoop == NPTypeCode.Half)
                && !shape.IsEmpty && shape.size > 0
                && !shape.IsScalar && !(shape.size == 1 && shape.NDim == 1)
                && (axis_ is null || shape[NormalizeAxis(axis_.Value, arr.ndim)] != 1))
                return AddMulNumPy(arr, axis_, keepdims, dtype, null, null, null, multiply: true);

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
                return NewScalar(emptyType.GetOneValue());
            }

            if (shape.size == 0)
            {
                if (axis_ == null)
                {
                    var emptyType = typeCode ?? arr.typecode.GetAccumulatingType();
                    var r = NewScalar(emptyType.GetOneValue());
                    if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks, copy: false)); }
                    return r.PyArray_Return();
                }
                var axis = NormalizeAxis(axis_.Value, arr.ndim);
                var resultShape = Shape.GetAxis(shape, axis);
                var result = NDArray.ones(new Shape(resultShape, copy: false), typeCode ?? arr.GetTypeCode.GetAccumulatingType());
                // The kept axis at NumPy's keepdims place (all-zero strides for an empty result).
                if (keepdims)
                    ExpandReducedAxis(result, arr.Shape, axis);
                // A 1-D input along its only axis gives the 0-d identity, NumPy's scalar as for a non-empty input (this
                // exit returned it writeable); a no-op for an array result.
                return result.PyArray_Return();
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
                var r = NewScalar(result);
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(new Shape(ks, copy: false)); }
                else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
                return r.PyArray_Return();
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
