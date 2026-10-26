namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     True if two arrays have the same shape and all elements are equal, False otherwise.
        /// </summary>
        /// <param name="a">First input array.</param>
        /// <param name="b">Second input array.</param>
        /// <param name="equal_nan">
        ///     When <c>false</c> (the NumPy default) NaN never compares equal, so an array that holds a NaN
        ///     is NOT equal even to itself — <c>np.array_equal(a, a)</c> is <c>false</c> whenever <paramref name="a"/>
        ///     contains a NaN. When <c>true</c>, NaNs at matching positions are treated as equal; for a complex
        ///     dtype a value counts as NaN when EITHER its real or imaginary component is NaN.
        /// </param>
        /// <returns>True if the arrays have identical shape and every element is equal.</returns>
        /// <remarks>
        ///     The shapes must match EXACTLY — unlike <see cref="array_equiv(NDArray,NDArray)"/> this performs
        ///     NO broadcasting. Mirrors <c>numpy.array_equal(a1, a2, equal_nan=False)</c> (NumPy 2.4.2).
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array_equal.html
        /// </remarks>
        /// <exception cref="TypeError">A fixed-width string operand with <paramref name="equal_nan"/> and two different arrays
        ///     of one shape (NumPy's <c>isnan</c> has no string loop). Without it, two strings of one class compare their text
        ///     and other string pairs are answered from shape and dtype (a shape mismatch, or no <c>equal</c> loop, is False;
        ///     empty arrays are True). A <c>StringDType</c> operand compares through its own <c>==</c> (NumPy's NA rules: two
        ///     different NA objects raise its TypeError) and, with <paramref name="equal_nan"/>, its <c>isnan</c> (a NaN-like
        ///     NA is a NaN).</exception>
        public static bool array_equal(NDArray a, NDArray b, bool equal_nan = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, b);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // The body is NumPy's statements in ArrayEqualEngine (Logic/ArrayEqualEngine.cs), an operation class beside
            // this facade.
            return ArrayEqualEngine.ArrayEqual(a, b, equal_nan);
        }
    }
}
