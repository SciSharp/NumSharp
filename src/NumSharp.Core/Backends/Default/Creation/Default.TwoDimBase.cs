// NumPy's np.vander (numpy/lib/_twodim_base_impl.py) as DefaultEngine's member: its statements as the facade held them
// - the 1-D check, promote_types(x.dtype, int), the column of ones and multiply.accumulate's powers, with the text
// dtypes' refusals, through the engine's and the array object's members (the engine calls no np.* function).

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // the asarray conversion, the ones column, the x column and its broadcast, the powers and the increasing matrix are this call's own: only the matrix returned leaves
        public override NDArray Vander(NDArray x, int? N = null, bool increasing = false)
        {
            // np.asarray(x)'s statement for a plain array (np.vander's facade turns a matrix or chararray plain first).
            x = NDArray.asarray(x, null, 'K', null);
            if (x.ndim != 1)
                throw new ValueError("x must be a one-dimensional array or sequence.");

            long m = x.size;
            int n = N ?? checked((int)m);

            // StringDType: `promote_types(x.dtype, int)` finds no common DType, a DTypePromotionError raised before the
            // allocation (so for an empty x and for any N too). PromoteTypes raises NumPy's text itself.
            if (Backends.Strings.NDStringUfuncs.IsVString(x))
            {
                DTypePromotion.PromoteTypes(x.dtype, np.int64);
                // The promotion never succeeds for StringDType (no common DType with Int64DType); reaching this line would
                // be a promotion-table bug, not a NumPy answer.
                throw new InvalidOperationException("promote_types(StringDType, int64) unexpectedly found a common DType.");
            }

            // NumPy: dtype = promote_types(x.dtype, int) — `int` is intp (int64 on 64-bit). Text absorbs int64 as a
            // 21-character string, and the powers come from multiply.accumulate, which has no string loop.
            if (Backends.Strings.NDStringUfuncs.IsString(x))
            {
                var text = DTypePromotion.PromoteTypes(x.dtype, np.int64);
                // `if N > 1: ... multiply.accumulate(...)` — the dtype refusal, whatever the rows.
                if (n > 1)
                    throw Backends.Strings.NDStringUfuncs.NoLoop("multiply", text, text);
                // `if N > 0: tmp[:, 0] = 1` writes the number one as its str() — the text '1' — into every row of the single
                // column (np.ones' string route).
                if (n == 1)
                    return NDArray.ones(new Shape(m, n), text);
                // `empty((len(x), N), dtype)`: no column to write — the allocation is the matrix, and a negative N is its
                // own error. A U matrix is zero-filled, as NumPy's NEEDS_INIT allocation is.
                return new NDArray(text, new Shape(m, n), false);
            }
            NPTypeCode dt = DTypePromotion.PromoteTypes(x.typecode, NPTypeCode.Int64);

            // NumPy allocates `empty((len(x), N))`; a negative N surfaces the allocator's own error.
            if (n <= 0)
                return new NDArray(dt, new Shape(m, n), false);

            NDArray onesCol = NDArray.ones(new Shape(m, 1), dt);
            NDArray incr;
            if (n == 1)
            {
                incr = onesCol;
            }
            else
            {
                // Each of the trailing N-1 columns starts as x, then a cumulative product along the
                // columns turns column k into x^(k+1) — bit-identical to NumPy's `multiply.accumulate`.
                var xcol = NDArray.expand_dims(x.astype(dt), -1);                     // (M, 1)
                var repeated = NDArray.broadcast_to(xcol, new Shape(m, n - 1));       // (M, N-1), each column = x
                var powered = ReduceCumMul(repeated, 1, dt, null);                   // (M, N-1), column k = x^(k+1)
                incr = NDArray.concatenate(new[] { onesCol, powered }, 1, null, null, "same_kind"); // (M, N), increasing powers
            }

            if (increasing)
                return incr;

            // Reverse the columns so the first is x^(N-1); copy to a C-contiguous buffer like NumPy's.
            return incr[":, ::-1"].copy();
        }
    }
}
