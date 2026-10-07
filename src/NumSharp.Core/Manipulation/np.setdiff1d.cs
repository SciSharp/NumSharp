using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the set difference of two arrays.<br></br>
        ///     Return the unique values in <paramref name="ar1"/> that are not in <paramref name="ar2"/>.
        /// </summary>
        /// <param name="ar1">Input array.</param>
        /// <param name="ar2">Input comparison array.</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False.</param>
        /// <returns>1-D array of values in <paramref name="ar1"/> that are not in <paramref name="ar2"/>. Sorted
        ///     when <paramref name="assume_unique"/> is False.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.setdiff1d.html</remarks>
        /// <exception cref="StringComputationNotSupportedException">A string operand whose answer depends on the text: a string
        ///     <paramref name="ar1"/> of two or more elements (NumPy sorts it), or a membership test NumPy makes by comparing
        ///     text. Answered from sizes and dtypes: an empty <paramref name="ar1"/>, and every value of
        ///     <paramref name="ar1"/> kept where nothing can match (see <c>np.isin</c>).</exception>
        [NDScoped]
        public static NDArray setdiff1d(NDArray ar1, NDArray ar2, bool assume_unique = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(ar1, ar2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text: NumPy's statements below run on text through text-aware steps (np.unique, np.isin); where NumPy sorts
            // text whose order the answer does not need, NumSharp answers here instead of deferring.
            if (!assume_unique && (Backends.Strings.NDStringUfuncs.IsString(ar1) || Backends.Strings.NDStringUfuncs.IsString(ar2)))
            {
                // unique(ar1) sorts ar1's text, and the answer keeps it (the next phase of the string support).
                if (Backends.Strings.NDStringUfuncs.IsString(ar1) && ar1.size > 1)
                    throw Backends.Strings.NDStringUfuncs.NextPhase("setdiff1d", ar1);
                // An empty ar1 leaves nothing to remove from: NumPy's isin of it returns before any value of ar2 matters
                // (after a concatenation that decodes bytes into str, which can fail).
                if (ar1.size == 0)
                {
                    if (Backends.Strings.NDStringUfuncs.ConcatMayFailDecoding(ar1, ar2))
                        throw Backends.Strings.NDStringUfuncs.NextPhase("setdiff1d", ar2);
                    return new NDArray(ar1.dtype, Shape.Vector(0), true);
                }
                // unique(ar2) sorts ar2's text, but the answer only needs its size: isin takes its loop path whenever
                // ar2's own size is below the threshold, and there a pair equal has no loop for removes nothing.
                if (Backends.Strings.NDStringUfuncs.IsString(ar2) && ar2.size > 1)
                {
                    NDArray kept = np.unique(ar1);
                    // kept came from np.unique; NumPy canonicalises a surviving float32/float64 NaN (see np.setops.cs).
                    if (ar2.size < 10.0 * Math.Pow(kept.size, 0.145) && !Backends.Strings.NDStringUfuncs.ComparesText(kept.dtype, ar2.dtype))
                        return CanonicalizeSetOpNaN(kept);
                    throw Backends.Strings.NDStringUfuncs.NextPhase("setdiff1d", ar2);
                }
            }
            // Boundary scope: the unique/ravel temps and the isin membership mask are reclaimed
            // at exit; only the selected-values array is yielded.

            if (assume_unique)
            {
                ar1 = np.ravel(ar1);
            }
            else
            {
                ar1 = np.unique(ar1);
                ar2 = np.unique(ar2);
            }

            // ar1[isin(ar1, ar2, assume_unique=True, invert=True)] — keep only the values absent from ar2.
            NDArray mask = np.isin(ar1, ar2, assume_unique: true, invert: true);
            // ar1 came from np.unique; NumPy canonicalises a surviving float32/float64 NaN (see np.setops.cs).
            return CanonicalizeSetOpNaN(ar1[mask]);
        }
    }
}
