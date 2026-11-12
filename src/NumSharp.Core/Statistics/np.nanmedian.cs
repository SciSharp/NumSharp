using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.nanmedian(a, axis=None, out=None, overwrite_input=False, keepdims=False)</c>: the median along
        ///     the given axes, ignoring NaNs. Like <c>np.median</c>, an even-sized (NaN-free) slice reduces to the
        ///     <b>mean</b> of the two central values — NumPy's mean-of-middle rule, NOT <c>np.nanquantile(a, 0.5)</c> (they
        ///     disagree by up to 1 ULP), so <c>medianMean: true</c> is REQUIRED. A slice that is entirely NaN (or empty)
        ///     yields NaN, matching NumPy's "All-NaN slice" behaviour.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text: see the exceptions).</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) reduces the flattened data; one axis
        ///     (negative counts from the end) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) reduces those axes, merged
        ///     into one; the empty tuple reduces nothing.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The NaN-ignoring median; <paramref name="out"/> itself when one is given. Over a NEP 55
        ///     <c>StringDType</c> array a median exists only for an all-NaN reduction: its NA object (a read-only 0-d
        ///     <c>StringDType</c> array, NumPy's Python object), or <c>np.float64(nan)</c> where NumPy's mean of NaN-like NAs
        ///     gives it.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/> (every axis is range-checked before
        ///     the repeat check).</exception>
        /// <exception cref="ValueError">A tuple names an axis twice (<c>repeated axis</c>); the store <c>out[...] =
        ///     result</c> (np.nanmedian's, after the computation): an <paramref name="out"/> the result does not broadcast to
        ///     (<c>could not broadcast input array from shape (4,) into shape (3,)</c>) or a read-only one (<c>assignment
        ///     destination is read-only</c>); for empty data, np.nanmean's add.reduce prologue texts (parity
        ///     QA-31).</exception>
        /// <exception cref="TypeError">Fixed-width text data that NumPy's NaN test (or, for empty data, its <c>nanmean</c>)
        ///     refuses (its verbatim text). A <c>StringDType</c> array: the median's mean refuses the concatenated text;
        ///     along an axis shorter than 600 numpy.ma's median refuses first (<c>Unsuitable type StringDType() for
        ///     calculating minimum.</c>); <paramref name="keepdims"/> over NumPy's Python-object result (<c>'float' object is
        ///     not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array: a None-like NA in a sorted line of two or more elements
        ///     (<c>Cannot compare null that is not a nan-like value</c>), an empty input's mean (no identity), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> reduction along a long axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html
        ///     <para>NumPy's <c>_ureduce</c> turns an integer axis into a one-element tuple, so the engine member takes
        ///     the axes as an array; a tuple is merged into one axis before the partition, and the empty tuple reduces
        ///     nothing.</para>
        ///     <para>A <c>numpy.matrix</c> operand: an integer axis gives the matrix of NumPy's <c>_collapse</c> rule, the
        ///     class rule of a TUPLE sees no axis (the rule NumSharp's tuple overloads had) - NumPy's own tuple answers come
        ///     from matrix semantics inside its Python code (NumPy's nanmedian of a matrix calls <c>matrix.sum</c> with <c>keepdims</c> and raises TypeError for every axis), which NumSharp does not reproduce.</para>
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the <c>int?</c> and <c>int[]</c>
        ///     overloads, which made <c>np.nanmedian(a, null)</c> ambiguous (CS0121) and had no tuple literal.</para>
        /// </remarks>
        public static NDArray nanmedian(NDArray a,
            NDAxis? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's np.nanmedian body (the text routes, then the NaN-ignoring median) is TensorEngine.NanMedian's: the
            // facade holds only the class rule. NumPy's _ureduce turns an int axis into a 1-tuple: the engine member takes
            // the axes as an array (null for None).
            return a.TensorEngine.NanMedian(a, axis?.ToArray(), @out, overwrite_input, keepdims);
        }

        /// <summary>
        ///     <see cref="nanmedian(NDArray, NDAxis?, NDArray, bool, bool)"/> with the axis as one <c>int</c>: the
        ///     overload a C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the
        ///     NDAxis form, so every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="nanmedian(NDArray, NDAxis?, NDArray, bool, bool)"/>
        public static NDArray nanmedian(NDArray a, int axis, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
            => nanmedian(a, (NDAxis?)axis, @out, overwrite_input, keepdims);

        /// <summary>
        ///     <see cref="nanmedian(NDArray, NDAxis?, NDArray, bool, bool)"/> with the axes as an <c>int[]</c>: the
        ///     overload an <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to
        ///     NDAxis), forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple
        ///     (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="nanmedian(NDArray, NDAxis?, NDArray, bool, bool)"/>
        public static NDArray nanmedian(NDArray a, int[] axis, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
            => nanmedian(a, (NDAxis?)axis, @out, overwrite_input, keepdims);
    }
}
