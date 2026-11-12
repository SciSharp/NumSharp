using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)</c>: the median along the
        ///     given axes (of the flattened data by default). For an even-sized slice the median is the <b>mean</b> of the two
        ///     central values, <c>(a+b)/2</c>; for an odd-sized slice it is the single central value. This is NOT the same
        ///     code path as <c>np.quantile(a, 0.5)</c> — NumPy computes <c>np.median</c> via <c>mean(part[middle])</c>, which
        ///     disagrees with the quantile <c>_lerp</c> at q=0.5 by up to 1 ULP on ~10 % of inputs — so
        ///     <c>medianMean: true</c> is REQUIRED for bit-exact parity.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text: see the exceptions).</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) reduces the flattened data; one axis
        ///     (negative counts from the end) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) reduces those axes, merged
        ///     into one; the empty tuple reduces nothing.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The median (NaN for an empty slice); <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/> (every axis is range-checked before
        ///     the repeat check).</exception>
        /// <exception cref="ValueError">A tuple names an axis twice (<c>repeated axis</c>); an <paramref name="out"/> that
        ///     np.median's closing <c>mean(part[indexer], out=out)</c> refuses in its add.reduce prologue, before anything is
        ///     computed: read-only (<c>output array is read-only</c>), of the wrong rank (<c>output parameter for reduction
        ///     operation add has the wrong number of dimensions: Found 2 but expected 1</c>) or with a length-1 axis where
        ///     the result is longer (parity QA-31).</exception>
        /// <exception cref="IncorrectShapeException">An <paramref name="out"/> whose axes do not broadcast with the
        ///     result's: the prologue's iterator text over the partition's middle slice (<c>operands could not be broadcast
        ///     together with remapped shapes [original-&gt;remapped]: (3,)-&gt;(3) (1,4) </c>), NumPy's ValueError; several
        ///     reduced axes beside a kept axis of length 0, any dtype: <c>_ureduce</c>'s merge cannot resolve its -1
        ///     (NumPy's ValueError <c>cannot reshape array of size 0 into shape (0,newaxis)</c>), raised before the text
        ///     refusals.</exception>
        /// <exception cref="TypeError">Text data or a text <paramref name="out"/>: NumPy's closing <c>mean</c> refuses (its
        ///     verbatim text). A NEP 55 <c>StringDType</c> array is sorted first (NumPy's partition of a <c>StringDType</c>
        ///     line), then its mean's divide refuses the concatenated text.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array: a None-like NA in a line of two or more elements (the sort's
        ///     <c>Cannot compare null that is not a nan-like value</c>), an empty reduced axis (the mean's add.reduce has no
        ///     identity) or a read-only <paramref name="out"/>.</exception>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the axes and keepdims' out are checked. With <c>overwrite_input</c> and no axis
        ///     NumPy partitions the text itself instead: its broadcast error for an even length other than 2
        ///     (<see cref="IncorrectShapeException"/>), else the closing mean's refusal of text.</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before the text refusals.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.median.html
        ///     <para>A <c>StringDType</c> median exists only when the middle elements are NaN-like NAs: NumPy's
        ///     <c>np.float64(nan)</c>, a read-only 0-d float64 here.</para>
        ///     <para>NumPy's <c>_ureduce</c> turns an integer axis into a one-element tuple, so the engine member takes
        ///     the axes as an array; a tuple is merged into one axis before the partition, and the empty tuple reduces
        ///     nothing.</para>
        ///     <para>A <c>numpy.matrix</c> operand: an integer axis gives the matrix of NumPy's <c>_collapse</c> rule, the
        ///     class rule of a TUPLE sees no axis (the rule NumSharp's tuple overloads had) - NumPy's own tuple answers come
        ///     from matrix semantics inside its Python code (<c>np.median(m, axis=(0, 1))</c> is a 1 x 1 matrix, <c>axis=()</c> reduces axis 1), which NumSharp does not reproduce.</para>
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the <c>int?</c> and <c>int[]</c>
        ///     overloads, which made <c>np.median(a, null)</c> ambiguous (CS0121) and had no tuple literal.</para>
        /// </remarks>
        public static NDArray median(NDArray a,
            NDAxis? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's np.median body (the text routes, then the numeric median) is TensorEngine.Median's: the facade
            // holds only the class rules. NumPy's _ureduce turns an int axis into a 1-tuple: the engine member takes the
            // axes as an array (null for None).
            return a.TensorEngine.Median(a, axis?.ToArray(), @out, overwrite_input, keepdims);
        }

        /// <summary>
        ///     <see cref="median(NDArray, NDAxis?, NDArray, bool, bool)"/> with the axis as one <c>int</c>: the
        ///     overload a C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the
        ///     NDAxis form, so every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="median(NDArray, NDAxis?, NDArray, bool, bool)"/>
        public static NDArray median(NDArray a, int axis, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
            => median(a, (NDAxis?)axis, @out, overwrite_input, keepdims);

        /// <summary>
        ///     <see cref="median(NDArray, NDAxis?, NDArray, bool, bool)"/> with the axes as an <c>int[]</c>: the
        ///     overload an <c>int[]</c> binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to
        ///     NDAxis), forwarding to the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple
        ///     (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="median(NDArray, NDAxis?, NDArray, bool, bool)"/>
        public static NDArray median(NDArray a, int[] axis, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
            => median(a, (NDAxis?)axis, @out, overwrite_input, keepdims);
    }
}
