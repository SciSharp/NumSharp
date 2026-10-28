// The functions of numpy/lib/_arraysetops_impl.py, as TensorEngine members: NumPy writes them in Python, DefaultEngine
// runs those statements in NumPy's order, and an engine that derives from DefaultEngine can replace one for every array
// bound to it (docs/plans/np-layer-dtype-logic.md, section 1 "Which O", Appendix D).

using System;
using NumSharp.Backends;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.isin</c>: whether each element of <paramref name="element"/> is in
        ///     <paramref name="test_elements"/> (or, with <paramref name="invert"/>, is not), as a bool array of
        ///     <paramref name="element"/>'s shape. NaN is never a member; text compares as text, a <c>StringDType</c> NA by
        ///     its descriptor's rules.
        /// </summary>
        /// <param name="element">
        ///     The tested values, a plain array (<c>np.isin</c> computes a <c>numpy.matrix</c> operand as a plain one).
        /// </param>
        /// <param name="test_elements">The set, flattened before use.</param>
        /// <param name="assume_unique">
        ///     NumPy's speed hint. Where NumPy takes its sort path, repeated values then get NumPy's answer for them (an
        ///     element is flagged when an equal value follows it in <paramref name="element"/> or lies in the set); the
        ///     table method and the loop path ignore it.
        /// </param>
        /// <param name="invert">Answer <c>element not in test_elements</c> (NumPy's AND of <c>!=</c>).</param>
        /// <param name="kind">
        ///     The algorithm: null (automatic), <c>"sort"</c> or <c>"table"</c>. Every path gives the same membership, except
        ///     under <paramref name="assume_unique"/> with repeated values; <c>"table"</c> is only valid for bool and integer
        ///     arrays.
        /// </param>
        /// <returns>
        ///     A bool array of <paramref name="element"/>'s shape: a view of a fresh 1-D answer, as NumPy's reshape returns
        ///     it.
        /// </returns>
        /// <exception cref="ValueError">
        ///     An unknown <paramref name="kind"/> (<c>Invalid kind: 'x'. Please use None, 'sort' or 'table'.</c>),
        ///     <c>kind='table'</c> for an array that is not bool or integer, or <c>kind='table'</c> over a set whose range is
        ///     <c>2**63 - 1</c> or more (<c>Maximum allowed dimension exceeded</c>).
        /// </exception>
        /// <exception cref="InvalidOperationException">
        ///     <c>kind='table'</c> with a set whose range exceeds its dtype's maximum (NumPy's RuntimeError text).
        /// </exception>
        /// <exception cref="UnicodeDecodeError">
        ///     The text route's sort path decodes <c>S</c> text holding a byte that is not ASCII beside <c>U</c> text.
        /// </exception>
        /// <exception cref="NotSupportedException">
        ///     A <c>StringDType</c> set whose NA object is a sequence, held by an NA cell (NumPy broadcasts the comparison
        ///     against it).
        /// </exception>
        /// <remarks>
        ///     NumPy writes isin in Python (<c>_isin</c>); DefaultEngine's override runs NumPy's path decision (table, loop
        ///     or sort) and answers each path with NumPy's membership. An override replaces <c>np.isin</c> for every array
        ///     whose engine it is (the engine of <paramref name="element"/>). New in 0.71.0 (the np-layer dtype plan, phase
        ///     PE); it was the internal class <c>IsinEngine</c>.
        /// </remarks>
        public abstract NDArray Isin(NDArray element, NDArray test_elements, bool assume_unique = false,
            bool invert = false, string kind = null);
    }
}
