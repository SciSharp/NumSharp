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
        /// <param name="nd">The operand.</param>
        /// <param name="dtype">The loop dtype (<c>dtype=</c>), or null for the operand's own loop.</param>
        /// <param name="out">The output array (<c>out=</c>), or null for a fresh result.</param>
        /// <param name="where">The bool mask (<c>where=</c>), or null for every element.</param>
        /// <returns><paramref name="out"/> when given; otherwise a fresh, writeable array of the loop dtype laid out as
        ///     NumPy's ufunc allocates it (<see cref="IdentityLoopResult"/>).</returns>
        /// <exception cref="TypeError">A bool loop (NumPy's positive has none), or a text operand.</exception>
        /// <exception cref="ArgumentException">A <paramref name="dtype"/> the operand cannot reach by a same_kind cast.</exception>
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
            if (typeCode.HasValue && typeCode.Value != nd.GetTypeCode)
                return IdentityLoopResult(nd, typeCode.Value, inputCast: true);

            // Plain +x: the identity loop into a fresh array.
            return IdentityLoopResult(nd, nd.GetTypeCode, inputCast: false);
        }

        /// <summary>
        ///     The fresh result of positive's identity loop: the operand's values in the loop dtype, laid out as NumPy's ufunc
        ///     machinery allocates a fresh output (<see cref="Shape.UfuncOutputShape"/>: C; the trivial loop's F for an F
        ///     operand; NpyIter's K order for a permuted, broadcast or cast operand) and always writeable.
        /// </summary>
        /// <param name="nd">The operand, any layout.</param>
        /// <param name="loop">The loop dtype (the operand's own, or <c>dtype=</c>'s).</param>
        /// <param name="inputCast">Whether the loop casts the operand (<paramref name="loop"/> is not its dtype): a cast
        ///     operand of two or more dimensions keeps NumPy off its trivial loop, which changes extent-1 strides.</param>
        /// <returns>A fresh array of <paramref name="loop"/> the caller owns.</returns>
        /// <remarks>
        ///     A storage clone is not this result: it keeps a read-only operand's flag (np.positive of a read-only owner or a
        ///     broadcast view is writeable in NumPy) and lays a permuted view out in C order (NumPy keeps the view's memory
        ///     order: np.positive(p) of <c>p = np.arange(24.).reshape(3, 2, 4).transpose(1, 0, 2)</c> has p's byte strides
        ///     (32, 64, 8)). A K-order copy is not it either: it sorts a broadcast axis innermost, where the ufunc's
        ///     iterator lets a stride-0 axis abstain (np.positive(np.broadcast_to(row, (3, 4))) is C).
        /// </remarks>
        private NDArray IdentityLoopResult(NDArray nd, NPTypeCode loop, bool inputCast)
        {
            if (TryUnaryNonCLayout(nd, inputCast, out var layout))
            {
                var laidOut = NewArray(loop, layout, false);
                // The cast was validated same_kind by ValidatePositiveLoop; the copy itself is the loop's unsafe cast.
                UnmanagedStorage.Copyto(laidOut, nd, NPY_CASTING.NPY_UNSAFE_CASTING);
                return laidOut;
            }

            // C order — forced, not 'K': a layout that resolves to C (a broadcast row, a C operand) must not take the K
            // copy's stride-0-innermost order. A zero-size operand gets NumPy's all-zero strides from the allocator.
            return NDIter.CopyAs(loop, nd, 'C', nd.TensorEngine);
        }

        /// <summary>
        /// NumPy's loop resolution for the <c>positive</c> ufunc, shared by <see cref="Positive"/> and by
        /// np.clip's no-bound route (NumPy's <c>_methods._clip</c> returns <c>um.positive(a, out=out, **kwargs)</c>
        /// when neither bound is left). Call it before any allocation. The input-cast check uses the house
        /// exception type, which the oracle maps to NumPy's UFuncTypeError.
        /// </summary>
        /// <param name="inputType">The dtype of the array positive is applied to.</param>
        /// <param name="dtype">The explicit <c>dtype=</c> loop request, or null for the input's own loop.</param>
        /// <exception cref="TypeError">The loop would be bool: positive has identity loops for every numeric dtype
        /// EXCEPT bool ('b->b'..'G->G' in ufunc.types, no '?->?'). A plain bool input reports "... BoolDType'> -> None",
        /// and a dtype=bool request names both sides. NumPy texts verbatim.</exception>
        /// <exception cref="ArgumentException">A <paramref name="dtype"/> request the input cannot reach by a same_kind cast:
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
                throw new TypeError(
                    "ufunc 'positive' did not contain a loop with signature matching types " +
                    $"<class 'numpy.dtypes.{NumPyDTypeClassName(inputType)}'> -> <class 'numpy.dtypes.BoolDType'>");
            if (dtype is null && inputType == NPTypeCode.Boolean)
                throw new TypeError(
                    "ufunc 'positive' did not contain a loop with signature matching types " +
                    "<class 'numpy.dtypes.BoolDType'> -> None");

            // dtype= runs the identity loop in that dtype: the input must reach
            // it via a same_kind cast (positive(f64, dtype=i32) raises, probed).
            if (dtype.HasValue)
                ValidateUnaryInputCast(inputType, dtype.Value, "positive");
        }

        /// <summary>
        /// NumPy dtype CLASS name as quoted in modern no-loop texts
        /// (numpy.dtypes.*DType — probed: Float64DType, BoolDType, Int32DType).
        /// Decimal/Char are NumSharp extensions without a NumPy counterpart;
        /// they synthesize the same pattern.
        /// </summary>
        private static string NumPyDTypeClassName(NPTypeCode tc) => (tc switch
        {
            NPTypeCode.Boolean => "Bool",
            NPTypeCode.Byte => "UInt8",
            NPTypeCode.SByte => "Int8",
            NPTypeCode.Int16 => "Int16",
            NPTypeCode.UInt16 => "UInt16",
            NPTypeCode.Int32 => "Int32",
            NPTypeCode.UInt32 => "UInt32",
            NPTypeCode.Int64 => "Int64",
            NPTypeCode.UInt64 => "UInt64",
            NPTypeCode.Half => "Float16",
            NPTypeCode.Single => "Float32",
            NPTypeCode.Double => "Float64",
            NPTypeCode.Complex => "Complex128",
            NPTypeCode.Decimal => "Decimal",
            NPTypeCode.Char => "Char",
            _ => tc.ToString(),
        }) + "DType";
    }
}
