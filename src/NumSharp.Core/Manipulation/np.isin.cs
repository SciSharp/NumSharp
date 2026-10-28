using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Calculates <c>element in test_elements</c>, broadcasting over <paramref name="element"/> only.<br></br>
        ///     Returns a boolean array of the same shape as <paramref name="element"/> that is True where an
        ///     element of <paramref name="element"/> is in <paramref name="test_elements"/> and False otherwise.
        /// </summary>
        /// <param name="element">Input array.</param>
        /// <param name="test_elements">The values against which to test each value of <paramref name="element"/>.
        ///     Flattened before use.</param>
        /// <param name="assume_unique">If True, the input arrays are both assumed to be unique, which can speed up
        ///     the calculation. Default is False. Where NumPy takes its sort path, repeated values then get NumPy's answer
        ///     for them: an element is flagged when an equal value follows it in <paramref name="element"/> or lies in
        ///     <paramref name="test_elements"/>.</param>
        /// <param name="invert">If True, the values in the returned array are inverted, as if calculating
        ///     <c>element not in test_elements</c>. Default is False.</param>
        /// <param name="kind">The algorithm to use: <c>null</c> (auto), <c>"sort"</c>, or <c>"table"</c>. Every path gives
        ///     the same membership, except under <paramref name="assume_unique"/> with repeated values (which only the sort
        ///     path reads). <c>"table"</c> is only valid for boolean/integer arrays (char is NumPy's uint16).</param>
        /// <returns>A boolean array with the same shape as <paramref name="element"/>.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.isin.html</remarks>
        /// <exception cref="ValueError">An unknown <paramref name="kind"/> (<c>Invalid kind: 'x'. Please use None, 'sort' or
        ///     'table'.</c>), <c>kind='table'</c> for a non-integer array, or <c>kind='table'</c> over a set whose range is
        ///     <c>2**63 - 1</c> or more (<c>Maximum allowed dimension exceeded</c>, NumPy's table allocation).</exception>
        /// <exception cref="InvalidOperationException"><c>kind='table'</c> with a range wider than the dtype's maximum
        ///     (NumPy's RuntimeError text).</exception>
        /// <exception cref="UnicodeDecodeError">A string operand on NumPy's sort path whose concatenation decodes <c>S</c> text
        ///     holding a byte that is not ASCII into <c>U</c> (see <c>DefaultEngine.IsinText</c>).</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> set whose NA object is a sequence, held by an NA
        ///     cell (NumPy broadcasts the comparison against it; see <c>DefaultEngine.IsinVString</c>).</exception>
        // The woven boundary scope reclaims every intermediate this call tree creates — the ravels,
        // the promoted casts, the sort/table/hash machinery inside the engine's isin helpers — and the
        // yielded reshape view is the only survivor. The helpers stay scope-free on purpose.
        [NDScoped]
        public static NDArray isin(NDArray element, NDArray test_elements,
            bool assume_unique = false, bool invert = false, string kind = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(element, test_elements);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            return element.TensorEngine.Isin(element, test_elements, assume_unique, invert, kind);
        }
    }
}
