using System;
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
        /// <returns><paramref name="out"/> when given; else a fresh copy in NumPy's ufunc output layout (the input's K
        ///     order; parity QA-36).</returns>
        /// <exception cref="TypeError">A bool loop (positive has none), or text.</exception>
        /// <exception cref="ArgumentException">A <paramref name="dtype"/> the input cannot reach by a same_kind cast, or an
        ///     out / mask the ufunc refuses.</exception>
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
            // (positive(i32, dtype=f64) ≡ astype(f64), probed).
            if (typeCode.HasValue && typeCode.Value != nd.GetTypeCode)
                return Cast(nd, typeCode.Value, copy: true);

            // Plain +x: identity copy in NumPy's ufunc output layout, the K order of the input (parity QA-36: a permuted
            // input keeps its memory order, a reversed axis is laid out forwards). The layout-preserving bulk clone
            // already is that layout for C, F and broadcast inputs; any other K order clones the input transposed into
            // it (a C-contiguous view for a dense permuted input, so the clone is one block copy) and relabels the copy.
            if (!IsKOrderCVoter(nd.Shape) && !nd.Shape.IsFContiguous
                && TryKOrderPlan(nd.Shape.dimensions, new[] { nd.Shape }, out var kLayout, out var kPerm))
            {
                NDArray t = IntoKOrder(nd, nd.ndim, kPerm);
                try
                {
                    return NDArray.AdoptLayout(t.Clone(), kLayout);
                }
                finally
                {
                    // The transposed view only fed the clone; the clone is a fresh array.
                    t.Dispose();
                }
            }
            return nd.Clone();
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
