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

        /// <summary>
        ///     NumPy's <c>np.unique</c>: the sorted unique elements of <paramref name="ar"/> (flattened, or its sub-arrays
        ///     along <paramref name="axis"/>), with the optional first-occurrence indices, reconstruction indices and counts.
        /// </summary>
        /// <param name="ar">
        ///     The operand, a plain array (only read; <c>np.unique</c> computes a <c>numpy.matrix</c> operand as a plain one,
        ///     and a <c>numpy.char.chararray</c> operand after its own class boundary).
        /// </param>
        /// <param name="return_index">Also return the index of each value's first occurrence (along the axis, if given).</param>
        /// <param name="return_inverse">Also return the indices that rebuild <paramref name="ar"/> from the values.</param>
        /// <param name="return_counts">Also return how many times each value occurs.</param>
        /// <param name="axis">
        ///     The axis whose sub-arrays are compared, or null to flatten first. A 1-D operand's axis is checked, then the
        ///     flat path runs (NumPy's rule). Along an axis every NaN sub-array is distinct and <c>-0.0</c> equals
        ///     <c>+0.0</c> (NumPy compares rows as structured scalars).
        /// </param>
        /// <param name="equal_nan">
        ///     Collapse every NaN into one value on the flat numeric path (NumPy's default); the axis path and fixed-width
        ///     text ignore it.
        /// </param>
        /// <param name="sorted">
        ///     NumPy's <c>sorted</c> (NumPy 2.3), accepted for parity: the values always come back sorted. NumPy's
        ///     <c>sorted=False</c> returns its hash set's iteration order, which is platform-specific (identical as a set).
        /// </param>
        /// <returns>
        ///     The present outputs in NumPy's order, <c>[values, index?, inverse?, counts?]</c>: the values in the operand's
        ///     dtype, the index outputs int64. Without an axis the inverse has the operand's shape (a view); with one it is
        ///     1-D.
        /// </returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> operand with an axis over two or more dimensions.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <remarks>
        ///     <c>np.unique</c> wraps the result in <c>np.UniqueResult</c>; both <c>NDArray.unique</c> members call this
        ///     member too (the bare one takes element 0). An override replaces all three for every array whose engine it
        ///     is (the engine of <paramref name="ar"/>). New in 0.71.0 (the np-layer dtype plan, phase PE); it was the
        ///     internal class <c>UniqueEngine</c>.
        /// </remarks>
        public abstract NDArray[] Unique(NDArray ar, bool return_index = false, bool return_inverse = false,
            bool return_counts = false, int? axis = null, bool equal_nan = true, bool sorted = true);

        /// <summary>
        ///     NumPy's <c>np.unique_values</c> (the Array API): the unique values of the flattened operand, every NaN
        ///     distinct (<c>equal_nan=False</c>).
        /// </summary>
        /// <param name="x">The operand, a plain array (only read).</param>
        /// <returns>
        ///     The unique values, sorted. NumPy returns its hash set's order for integer, complex and text values (a
        ///     platform-specific order; the same set).
        /// </returns>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     An override replaces <c>np.unique_values</c> for every array whose engine it is. New in 0.71.0 (the np-layer
        ///     dtype plan, phase PE); it was the internal class <c>UniqueEngine</c>.
        /// </remarks>
        public abstract NDArray UniqueValues(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.unique_counts</c> (the Array API): the unique values of the flattened operand and how often
        ///     each occurs, every NaN distinct (<c>equal_nan=False</c>).
        /// </summary>
        /// <param name="x">The operand, a plain array (only read).</param>
        /// <returns><c>[values, counts]</c>: the values sorted, the counts int64.</returns>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     <c>np.unique_counts</c> wraps the result in <c>np.UniqueCountsResult</c>. An override replaces it for every
        ///     array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal class
        ///     <c>UniqueEngine</c>.
        /// </remarks>
        public abstract NDArray[] UniqueCounts(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.unique_inverse</c> (the Array API): the unique values of the flattened operand and the indices
        ///     that rebuild it, every NaN distinct (<c>equal_nan=False</c>).
        /// </summary>
        /// <param name="x">The operand, a plain array (only read).</param>
        /// <returns>
        ///     <c>[values, inverse_indices]</c>: the values sorted; the inverse int64 with <paramref name="x"/>'s shape (a
        ///     view; <c>()</c> for a 0-d operand), so <c>np.take(values, inverse_indices)</c> rebuilds the operand.
        /// </returns>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     <c>np.unique_inverse</c> wraps the result in <c>np.UniqueInverseResult</c>. An override replaces it for every
        ///     array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal class
        ///     <c>UniqueEngine</c>.
        /// </remarks>
        public abstract NDArray[] UniqueInverse(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.unique_all</c> (the Array API): the unique values of the flattened operand, the index of each
        ///     value's first occurrence, the indices that rebuild the operand and the counts, every NaN distinct
        ///     (<c>equal_nan=False</c>).
        /// </summary>
        /// <param name="x">The operand, a plain array (only read).</param>
        /// <returns>
        ///     <c>[values, indices, inverse_indices, counts]</c>: the values sorted, the other three int64; the inverse has
        ///     <paramref name="x"/>'s shape (a view; <c>()</c> for a 0-d operand).
        /// </returns>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>
        ///     <c>np.unique_all</c> wraps the result in <c>np.UniqueAllResult</c>. An override replaces it for every array
        ///     whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal class
        ///     <c>UniqueEngine</c>.
        /// </remarks>
        public abstract NDArray[] UniqueAll(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.ediff1d(ary, to_end=None, to_begin=None)</c>: the differences between consecutive elements
        ///     of the flattened array, with to_begin / to_end (each cast to <paramref name="ary"/>'s dtype under
        ///     <c>same_kind</c>) put before and after.
        /// </summary>
        /// <param name="ary">The input, flattened (not null).</param>
        /// <param name="to_end">Values to append, or null for none.</param>
        /// <param name="to_begin">Values to prepend, or null for none.</param>
        /// <returns>A new 1-D array in <paramref name="ary"/>'s dtype.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="ary"/> is null.</exception>
        /// <exception cref="ArgumentException">to_begin / to_end that cannot reach <paramref name="ary"/>'s dtype under
        ///     <c>same_kind</c> (NumPy's TypeError text).</exception>
        /// <exception cref="TypeError">Text or bool, which subtract has no loop for (NumPy's texts).</exception>
        public abstract NDArray Ediff1d(NDArray ary, object to_end = null, object to_begin = null);
    }
}
