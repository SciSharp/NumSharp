using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return the indices of the bins to which each value in input array <paramref name="x"/> belongs.
        ///
        ///     <list type="table">
        ///       <listheader><term>right</term><term>bins order</term><term>returned index i satisfies</term></listheader>
        ///       <item><term>false</term><term>increasing</term><term><c>bins[i-1] &lt;= x &lt; bins[i]</c></term></item>
        ///       <item><term>true</term><term>increasing</term><term><c>bins[i-1] &lt; x &lt;= bins[i]</c></term></item>
        ///       <item><term>false</term><term>decreasing</term><term><c>bins[i-1] &gt; x &gt;= bins[i]</c></term></item>
        ///       <item><term>true</term><term>decreasing</term><term><c>bins[i-1] &gt;= x &gt; bins[i]</c></term></item>
        ///     </list>
        ///
        ///     Values in <paramref name="x"/> beyond the bounds of <paramref name="bins"/> return 0 or
        ///     <c>len(bins)</c> as appropriate. Implemented in terms of <see cref="searchsorted(NDArray, NDArray, string, NDArray)"/>.
        /// </summary>
        /// <param name="x">Input array to be binned. May have any shape; the result has the same shape.</param>
        /// <param name="bins">1-D monotonic (increasing or decreasing) array of bin edges.</param>
        /// <param name="right">Whether the intervals include the right or the left bin edge. Default is left-closed (false).</param>
        /// <returns>Array of int64 indices, of the same shape as <paramref name="x"/>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="bins"/> is null.</exception>
        /// <exception cref="IncorrectTypeException"><paramref name="x"/> is complex (<c>x may not be complex</c>, checked first).</exception>
        /// <exception cref="ValueError"><paramref name="bins"/> is not 1-D (<c>object too deep for desired array</c> /
        ///     <c>object of too small depth for desired array</c>), or, after the cast below, not monotonic (<c>bins must be
        ///     monotonically increasing or decreasing</c>).</exception>
        /// <exception cref="InvalidCastException"><paramref name="bins"/> does not cast 'safely' to float64 (text or complex;
        ///     NumPy's TypeError text).</exception>
        /// <exception cref="StringComputationNotSupportedException">A text <paramref name="x"/> with elements to search among
        ///     non-empty bins: NumPy compares the keys as text against the bins converted to text (the computational
        ///     phase).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.digitize.html
        ///     <para>
        ///     <b>Check order (NumPy's).</b> A complex x first; then <c>_monotonicity(bins)</c>, whose
        ///     <c>PyArray_FROMANY(bins, NPY_DOUBLE, 1, 1)</c> checks the bins' rank before their 'safe' cast to float64; then the
        ///     monotonicity; then <c>searchsorted</c>. So a text x is only reached once the (numeric) bins passed — and when x has
        ///     no elements, or the bins are empty, NumPy's answer needs no comparison: an empty int64 array, or all zeros.
        ///     </para>
        /// </remarks>
        [NDScoped]
        public static NDArray digitize(NDArray x, NDArray bins, bool right = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(x, bins);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (bins is null) throw new ArgumentNullException(nameof(bins));

            // NumPy checks a complex `x` first, before any bins validation (TypeError).
            if (x.typecode == NPTypeCode.Complex)
                throw new IncorrectTypeException("x may not be complex");

            // NumPy: mono = _monotonicity(bins) — rank, then the 'safe' float64 cast, then the check; raise if not monotonic.
            int mono = BinsMonotonicity(bins);
            if (mono == 0)
                throw new ValueError("bins must be monotonically increasing or decreasing");

            // A text x is searched as text (searchsorted promotes the numeric bins to a string dtype). With no key, or no bin
            // edge, nothing is compared: NumPy returns int64 indices of x's shape — all zero for empty bins (every key's
            // insertion point), none for an empty x (a decreasing-bins `len(bins) - result` keeps both).
            if (Backends.Strings.NDStringUfuncs.IsString(x))
            {
                if (x.size == 0 || bins.size == 0)
                    return np.zeros(Shape.Adopt((long[])x.Shape.dimensions.Clone()), NPTypeCode.Int64);
                throw Backends.Strings.NDStringUfuncs.NextPhase("digitize", x);
            }

            // The side is reversed relative to `right` because searchsorted's operands are swapped
            // (numpy.digitize: side = 'left' if right else 'right').
            string side = right ? "left" : "right";

            // NumPy's searchsorted compares in result_type(bins, x). NumSharp's searchsorted casts the
            // keys to the sorted array's dtype, so promote `bins` to the common type first — that widens
            // `x` safely inside searchsorted rather than truncating it (e.g. float x into int bins).
            NPTypeCode common = np._FindCommonType(bins, x);
            NDArray binsC = bins.typecode == common ? bins : bins.astype(common);

            if (mono == -1)
            {
                // Decreasing bins: reverse to ascending, search, then invert the indices.
                //   result = len(bins) - searchsorted(bins[::-1], x, side)
                NDArray rev = binsC["::-1"].copy();
                NDArray ss = np.searchsorted(rev, x, side);
                return NDArray.Scalar((long)bins.Shape.dimensions[0]) - ss;
            }

            return np.searchsorted(binsC, x, side);
        }

        /// <summary>
        ///     Faithful port of NumPy's <c>check_array_monotonic</c> (numpy/_core/src/multiarray/compiled_base.c)
        ///     plus the <c>PyArray_FROMANY(NPY_DOUBLE, 1, 1)</c> shape rules that <c>_monotonicity</c> applies to
        ///     <paramref name="bins"/>. Returns +1 (monotonic increasing / all-equal / empty), -1 (monotonic
        ///     decreasing) or 0 (not monotonic).
        ///
        ///     The comparison runs in double (as NumPy forces) and is deliberately non-strict and NaN-quirky:
        ///     it skips leading repeats, fixes the direction from the first differing pair, then verifies the
        ///     rest with strict &lt;/&gt; — so a NaN pair (all comparisons false) never registers a violation,
        ///     exactly matching NumPy (e.g. <c>[1,2,nan,3]</c> is reported increasing).
        /// </summary>
        /// <param name="bins">The bin edges.</param>
        /// <returns>+1 (increasing / all equal / empty), -1 (decreasing) or 0 (not monotonic).</returns>
        /// <exception cref="ValueError">The bins are not 1-D (checked before the dtype, as <c>PyArray_FromAny</c> does).</exception>
        /// <exception cref="InvalidCastException">The bins are text or complex: NumPy's 'safe' cast to float64 refuses them
        ///     (<c>Cannot cast array data from dtype('&lt;U2') to dtype('float64') according to the rule 'safe'</c>). A complex
        ///     bins array used to lose its imaginary part silently.</exception>
        private static unsafe int BinsMonotonicity(NDArray bins)
        {
            // PyArray_FROMANY(..., 1, 1) rejects 0-D and >1-D with these verbatim messages.
            if (bins.ndim == 0)
                throw new ValueError("object of too small depth for desired array");
            if (bins.ndim > 1)
                throw new ValueError("object too deep for desired array");
            // ...and then casts under 'safe' (no FORCECAST in NPY_ARRAY_CARRAY_RO): text and complex refuse.
            if (Backends.Strings.NDStringUfuncs.IsString(bins) || bins.typecode == NPTypeCode.Complex)
                throw new InvalidCastException(
                    $"Cannot cast array data from {bins.dtype.ToString(true)} to dtype('float64') according to the rule 'safe'");

            // Forced to a contiguous, offset-0, C-order double buffer (as NumPy does for the check).
            NDArray d = bins.typecode == NPTypeCode.Double ? bins : bins.astype(NPTypeCode.Double);
            if (!d.Shape.IsContiguous || d.Shape.offset != 0)
                d = d.copy();

            long n = d.size;
            double* a = (double*)((byte*)d.Storage.Address + d.Shape.offset * sizeof(double));

            if (n == 0)
                return 1;                                   // "all bin edges hold the same value"

            double last = a[0];
            long i = 1;
            while (i < n && a[i] == last) i++;              // skip leading repeats
            if (i == n)
                return 1;                                   // all equal

            double next = a[i];
            if (last < next)
            {
                for (i += 1; i < n; i++)
                {
                    last = next;
                    next = a[i];
                    if (last > next) return 0;
                }
                return 1;
            }
            else
            {
                for (i += 1; i < n; i++)
                {
                    last = next;
                    next = a[i];
                    if (last < next) return 0;
                }
                return -1;
            }
        }
    }
}
