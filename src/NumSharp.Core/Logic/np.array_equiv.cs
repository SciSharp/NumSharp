namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     True if the inputs are shape consistent and all elements are equal.
        /// </summary>
        /// <param name="a1">First input array.</param>
        /// <param name="a2">Second input array.</param>
        /// <returns>True if equivalent, False otherwise.</returns>
        /// <remarks>
        ///     "Shape consistent" means the two shapes are either equal or one can be BROADCAST to the
        ///     other — the key difference from <see cref="array_equal(NDArray,NDArray,bool)"/>, which
        ///     requires an exact shape match. NaN never compares equal (there is no <c>equal_nan</c>
        ///     option, matching NumPy), so <c>array_equiv(a, a)</c> is <c>false</c> whenever <paramref name="a1"/>
        ///     holds a NaN. Empty inputs of a broadcast-consistent shape are equivalent (all() over the
        ///     empty comparison is vacuously true). Mirrors <c>numpy.array_equiv(a1, a2)</c> (NumPy 2.4.2):
        ///     it converts both to arrays, checks they broadcast together, then returns
        ///     <c>bool((a1 == a2).all())</c>.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html
        /// </remarks>
        /// <remarks>Text: two strings of one class compare their text (zero-padded, unsigned code units); any other pair with
        ///     a string operand has no <c>equal</c> loop, so the <c>==</c> operator's all-False array makes it False unless the
        ///     broadcast is empty. A NEP 55 <c>StringDType</c> operand compares with <c>StringDType</c> or <c>U</c> text through
        ///     NumPy's <c>NpyString_cmp</c> (<c>'a\0c'</c> equals <c>'a\0b'</c>: the bytes after a shared NUL are not compared)
        ///     and its NA rules (a NaN-like NA is never equal; a None-like NA reads as <c>''</c>).</remarks>
        /// <exception cref="TypeError">Two <c>StringDType</c> operands whose NA objects differ (NumPy's comparison refuses the
        ///     pair).</exception>
        public static bool array_equiv(NDArray a1, NDArray a2)
        {
            using var plain = NDMatrixBoundary.PlainFor(a1, a2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // The body is NumPy's statements, TensorEngine.ArrayEquiv (DefaultEngine's in
            // Backends/Default/Logic/Default.ArrayEqual.cs), on the engine of the first operand that is not null. Two nulls
            // are equivalent (NumSharp's null convention) and have no engine to ask; one null reaches the engine, whose
            // statements answer False.
            var engine = (a1 ?? a2)?.TensorEngine;
            return engine is null || engine.ArrayEquiv(a1, a2);
        }
    }
}
