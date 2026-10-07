namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Find the union of two arrays.<br></br>
        ///     Return the unique, sorted array of values that are in either of the two input arrays.
        /// </summary>
        /// <param name="ar1">Input array (flattened if not already 1-D).</param>
        /// <param name="ar2">Input array (flattened if not already 1-D).</param>
        /// <returns>Unique, sorted union of the input arrays.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.union1d.html</remarks>
        /// <exception cref="UnicodeDecodeError">A <c>U</c> union over <c>S</c> text holding a non-ASCII byte (NumPy's
        ///     concatenation decodes it first).</exception>
        /// <exception cref="StringComputationNotSupportedException">A string union of two or more elements (NumPy sorts the
        ///     text), or numbers to format as text (the concatenation's text conversion).</exception>
        [NDScoped]
        public static NDArray union1d(NDArray ar1, NDArray ar2)
        {
            using var plain = NDMatrixBoundary.PlainFor(ar1, ar2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text: NumPy concatenates first (formatting numbers into text, decoding bytes into str — the concatenation
            // below does both as NumPy does, or defers the formatting), then sorts the union's text (the next phase of
            // NumSharp's string support) — except a union of at most one element, which orders nothing.
            if (Backends.Strings.NDStringUfuncs.IsString(ar1) || Backends.Strings.NDStringUfuncs.IsString(ar2))
            {
                NDArray all = np.concatenate((np.ravel(ar1), np.ravel(ar2)), 0);
                if (all.size > 1)
                    throw Backends.Strings.NDStringUfuncs.NextPhase("union1d", all);
                return np.unique(all);
            }
            // Boundary scope: the ravel views, the concatenate buffer and unique's internals are
            // reclaimed at exit; only the final (possibly NaN-canonicalised) array is yielded.
            // NumPy: unique(concatenate((ar1, ar2), axis=None)) — axis=None flattens both operands.
            // NumPy's unique-sort canonicalises a surviving float32/float64 NaN; match it (see np.setops.cs).
            return CanonicalizeSetOpNaN(np.unique(np.concatenate((np.ravel(ar1), np.ravel(ar2)), 0)));
        }
    }
}
