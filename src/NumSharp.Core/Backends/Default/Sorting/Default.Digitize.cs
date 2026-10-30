// NumPy's np.digitize (numpy/lib/_function_base_impl.py) as DefaultEngine's member: the checks, the monotonicity of
// the bins and searchsorted on them (the engine's SearchSorted, np.searchsorted's member). The one np.* call left is
// np._FindCommonType, a promotion helper whose move into the dtype layer waits (R2 pins it with its reason).

using System;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <inheritdoc />
        [NDScoped] // the cast bins, the reversed copy and the searchsorted indices are this call's own
        public override NDArray Digitize(NDArray x, NDArray bins, bool right = false)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            if (bins is null) throw new ArgumentNullException(nameof(bins));

            // NumPy checks a complex `x` first, before any bins validation (TypeError).
            if (x.typecode == NPTypeCode.Complex)
                throw new IncorrectTypeException("x may not be complex");

            // NumPy: mono = _monotonicity(bins) — rank, then the 'safe' float64 cast, then the check; raise if not monotonic.
            int mono = BinsMonotonicity(bins);
            if (mono == 0)
                throw new ValueError("bins must be monotonically increasing or decreasing");

            // A StringDType x (tested first: IsString refuses it): the numeric bins have no common descriptor with it, so
            // NumPy's searchsorted compares Python objects — a number against a str raises its TypeError, and the keys are
            // compared with each other even when the bins are empty (searchsorted answers every case, no shortcut here).
            if (Backends.Strings.NDStringUfuncs.IsVString(x))
            {
                string objectSide = right ? "left" : "right";
                if (mono == -1)
                    return NDArray.Scalar((long)bins.Shape.dimensions[0]) - SearchSorted(bins["::-1"], x, objectSide);
                return SearchSorted(bins, x, objectSide);
            }

            // A text x is searched as text: searchsorted converts the numeric bins to the common string dtype and runs NumPy's
            // binary search over the text (the bins were checked monotonic as NUMBERS, so their text need not be sorted: the
            // search's exact comparisons give NumPy's answer either way). With no key, or no bin edge, nothing is compared:
            // NumPy returns int64 indices of x's shape — all zero for empty bins (every key's insertion point), none for an
            // empty x (a decreasing-bins `len(bins) - result` keeps both).
            if (Backends.Strings.NDStringUfuncs.IsString(x))
            {
                if (x.size == 0 || bins.size == 0)
                    return new NDArray(NPTypeCode.Int64, Shape.Adopt((long[])x.Shape.dimensions.Clone()), true);
                string textSide = right ? "left" : "right";
                if (mono == -1)
                    return NDArray.Scalar((long)bins.Shape.dimensions[0]) - SearchSorted(bins["::-1"], x, textSide);
                return SearchSorted(bins, x, textSide);
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
                NDArray ss = SearchSorted(rev, x, side);
                return NDArray.Scalar((long)bins.Shape.dimensions[0]) - ss;
            }

            return SearchSorted(binsC, x, side);
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
            // ...and then casts under 'safe' (no FORCECAST in NPY_ARRAY_CARRAY_RO): text (StringDType included, tested first:
            // IsString refuses it) and complex refuse.
            if (Backends.Strings.NDStringUfuncs.IsVString(bins) || Backends.Strings.NDStringUfuncs.IsString(bins)
                || bins.typecode == NPTypeCode.Complex)
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
