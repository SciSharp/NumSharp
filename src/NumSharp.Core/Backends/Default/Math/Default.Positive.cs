using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// NumPy 'positive' — identity at every numeric dtype (returns +x, a copy).
        /// Full ufunc surface: out=/where= ride the iterator Into-path with the
        /// identity kernel (<see cref="UnaryOp.Positive"/> emits nothing — the same
        /// masked-copy vehicle Default.Round uses), dtype= selects the loop.
        /// </summary>
        /// <param name="nd">The input (any numeric dtype but bool, any layout).</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the input's own.</param>
        /// <param name="out">NumPy's <c>out=</c>, or null for a fresh result.</param>
        /// <param name="where">NumPy's <c>where=</c> mask, or null.</param>
        /// <returns><paramref name="out"/> when given; else a fresh, writeable array in NumPy's ufunc output layout
        ///     (<see cref="IdentityLoopResult"/>; parity QA-36 and the view-flags tier's VF14).</returns>
        /// <exception cref="TypeError">A bool loop (positive has none), or text.</exception>
        /// <exception cref="TypeError">A <paramref name="dtype"/> the input cannot reach by a same_kind cast, or an out
        ///     that cannot take the loop dtype.</exception>
        /// <exception cref="ArgumentException">A mask the ufunc refuses (not bool), or an out / mask that does not
        ///     broadcast.</exception>
        public override NDArray Positive(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "positive", nd, null, @out, where);
            // positive has no string loop; the identity copy below would otherwise hand a string array back.
            Strings.NDStringUfuncs.ThrowIfUnary("positive", nd, typeCode);
            ValidatePositiveLoop(nd.GetTypeCode, typeCode);

            // out=/where=: iterator Into-path; the loop dtype is dtype ?? input
            // (explicit outputType — ExecuteUnaryOp's null-typeCode default
            // would float-promote, but positive preserves the input dtype).
            if (@out is not null || where is not null)
                return ExecuteUnaryOp(nd, UnaryOp.Positive, typeCode ?? nd.GetTypeCode, @out, where);

            // dtype-only: the identity loop at the requested dtype IS a cast-copy
            // (positive(i32, dtype=f64) ≡ astype(f64), probed); the operand is cast, so NumPy's
            // iterator, not its trivial loop, lays the result out.
            // Both exits below are fresh results: a 0-d one is NumPy's scalar (the ufunc's PyArray_Return).
            if (typeCode.HasValue && typeCode.Value != nd.GetTypeCode)
                return IdentityLoopResult(nd, typeCode.Value, inputCast: true).PyArray_Return();

            // Plain +x: the identity loop into a fresh array.
            return IdentityLoopResult(nd, nd.GetTypeCode, inputCast: false).PyArray_Return();
        }

        /// <summary>
        ///     The fresh result of positive's identity loop: the operand's values in the loop dtype, laid out as NumPy's ufunc
        ///     machinery allocates a fresh output (the trivial loop's plain C or F, else NpyIter's K order:
        ///     <see cref="TrivialLoopLaysOut(NDArray, bool, out bool)"/>, <see cref="TryKOrderPlan"/>) and always writeable.
        /// </summary>
        /// <param name="nd">The operand, any layout.</param>
        /// <param name="loop">The loop dtype (the operand's own, or <c>dtype=</c>'s).</param>
        /// <param name="inputCast">Whether the loop casts the operand (<paramref name="loop"/> is not its dtype): a cast
        ///     operand of two or more dimensions keeps NumPy off its trivial loop, which changes extent-1 strides.</param>
        /// <returns>A fresh array of <paramref name="loop"/> the caller owns.</returns>
        /// <exception cref="TypeError">Never for a validated loop: the copy is the loop's unsafe cast, and
        ///     <see cref="ValidatePositiveLoop"/> already refused every cast NumPy refuses.</exception>
        /// <remarks>
        ///     A storage clone is not this result: it keeps a read-only operand's flag (np.positive of a read-only owner or a
        ///     broadcast view is writeable in NumPy) and keeps a view's own strides. A K-order copy is not it either: it sorts
        ///     a broadcast axis innermost, where the ufunc's iterator lets a stride-0 axis abstain
        ///     (np.positive(np.broadcast_to(row, (3, 4))) is C). A zero-size operand gets NumPy's all-zero strides from the
        ///     allocator (<see cref="Shape.ForFreshAllocation"/>).
        /// </remarks>
        private NDArray IdentityLoopResult(NDArray nd, NPTypeCode loop, bool inputCast)
        {
            var dims = nd.Shape.dimensions;
            Shape layout;
            if (TrivialLoopLaysOut(nd, inputCast, out bool fortran))
                // The trivial loop's plain C / F; copy: false shares the operand's dims array, as every elementwise result
                // does (nothing writes into a Shape's arrays).
                layout = fortran ? new Shape(dims, 'F', copy: false) : new Shape(dims, copy: false);
            else if (!TryKOrderPlan(dims, new[] { nd.Shape }, out layout, out _))
                layout = new Shape(dims, copy: false);

            var result = NewArray(loop, layout, false);
            try
            {
                // The cast was validated same_kind by ValidatePositiveLoop; the copy itself is the loop's unsafe cast.
                UnmanagedStorage.Copyto(result, nd, NPY_CASTING.NPY_UNSAFE_CASTING);
            }
            catch
            {
                // The copy failed: the fresh array must not outlive the call.
                result.Dispose();
                throw;
            }
            return result;
        }

        /// <summary>
        /// NumPy's loop resolution for the <c>positive</c> ufunc, shared by <see cref="Positive"/> and by
        /// np.clip's no-bound route (NumPy's <c>_methods._clip</c> returns <c>um.positive(a, out=out, **kwargs)</c>
        /// when neither bound is left). Call it before any allocation. The input-cast check raises TypeError, the
        /// class of NumPy's UFuncTypeError, with NumPy's text.
        /// </summary>
        /// <param name="inputType">The dtype of the array positive is applied to.</param>
        /// <param name="dtype">The explicit <c>dtype=</c> loop request, or null for the input's own loop.</param>
        /// <exception cref="TypeError">The loop would be bool: positive has identity loops for every numeric dtype
        /// EXCEPT bool ('b->b'..'G->G' in ufunc.types, no '?->?'). A plain bool input reports "... BoolDType'> -> None",
        /// and a dtype=bool request names both sides. NumPy texts verbatim.</exception>
        /// <exception cref="TypeError">A <paramref name="dtype"/> request the input cannot reach by a same_kind cast:
        /// "Cannot cast ufunc 'positive' input from dtype('X') to dtype('Y') with casting rule 'same_kind'" (no input index,
        /// because positive is unary).</exception>
        internal static void ValidatePositiveLoop(NPTypeCode inputType, NPTypeCode? dtype)
        {
            // NumPy: positive has identity loops for every numeric dtype EXCEPT
            // bool ('b->b'..'G->G' in ufunc.types, no '?->?'). The guard keys
            // off the LOOP: positive(bool, dtype=f64) is legal ([1., 0.]) while
            // positive(f64, dtype=bool) raises naming both sides — probed 2.4.2,
            // texts verbatim.
            if (dtype == NPTypeCode.Boolean)
                throw new UFuncNoLoopError("positive", 1, ((DType)inputType).Meta, DType.Boolean.Meta);
            if (dtype is null && inputType == NPTypeCode.Boolean)
                throw new UFuncNoLoopError("positive", 1, DType.Boolean.Meta, null);

            // dtype= runs the identity loop in that dtype: the input must reach
            // it via a same_kind cast (positive(f64, dtype=i32) raises, probed).
            if (dtype.HasValue)
                ValidateUnaryInputCast(inputType, dtype.Value, "positive");
        }

    }
}
