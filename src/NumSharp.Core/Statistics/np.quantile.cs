using System;
using NumSharp.Backends;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *,
        ///     weights=None)</c>: the q-th quantile(s) of the data along the given axes (of the flattened data by
        ///     default).
        /// </summary>
        /// <param name="a">The data (any real dtype; text — <c>S</c>, <c>U</c>, NEP 55 <c>StringDType</c> — takes NumPy's
        ///     take path or its refusal, see the exceptions).</param>
        /// <param name="q">NumPy's array_like <c>q</c> as one array, each value in [0, 1]: a C# number converts to a 0-d q
        ///     (a scalar result; <c>0.5</c> or <c>1</c> is NumPy's Python number, weak for the result dtype, so float32 data
        ///     keeps float32 under interpolation), a <c>double[]</c> to a 1-d q (one result per element, q's axis first), and
        ///     an <see cref="NDArray"/> is taken as it is (0-d or 1-d; up to 2-d over text data), a STRONG q for the
        ///     result dtype.
        ///     An integer q - a C# integer literal is NumPy's Python int, a 0-d int64 - makes method <c>linear</c>'s
        ///     virtual index <c>(n - 1) * q</c> an integer that NumPy TAKES instead of interpolating (the data's own dtype
        ///     and element: <c>np.quantile(int_array, 1)</c> is an int64).</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) reduces the flattened data; one axis
        ///     (negative counts from the end) or a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) reduces those axes, merged
        ///     into one before the partition; the empty tuple reduces nothing.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (keyword-only there), only with method <c>inverted_cdf</c>: one
        ///     weight per element of <paramref name="a"/> (its shape, or the reduced axes' shape); null (the default)
        ///     weighs every element equally. A weighted call takes NumPy's weighted branch: the data sorted, the weights'
        ///     cumulative sum normalized, and each q searched in it; the result has the data's dtype.</param>
        /// <returns>The quantiles: one per element of q (q's axes first; a scalar result for a 0-d q), with the
        ///     axes reduced; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError">A text <paramref name="q"/> (NumPy's [0, 1] check: Python's <c>'&lt;=' not supported
        ///     between instances of 'float' and 'numpy.str_'</c> for a 1-d q of fewer than 10 elements, <c>q.min()</c>'s
        ///     <c>minimum</c> refusal otherwise; an empty one reaches the string repeat's refusal).</exception>
        /// <exception cref="ArgumentException">A numeric quantile's q of rank above 1, or a quantile outside [0, 1] (NumPy's
        ///     ValueError texts).</exception>
        /// <exception cref="ValueError">Over text data: a q of rank above 2 (<c>q must be a scalar or 1d</c>, after the axis),
        ///     or an empty q of two dimensions (its <c>q.min()</c> range check).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        /// <exception cref="TypeError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: an interpolating method (NumPy's
        ///     <c>_lerp</c> subtracts the two neighbours: <c>unsupported operand type(s) for -: 'str' and 'str'</c>, or no
        ///     <c>subtract</c> loop for a non-scalar result), an <paramref name="out"/> that does not cast safely to the data
        ///     (the take path) or receive <c>_lerp</c>'s float64, or keepdims over NumPy's Python-object result
        ///     (<c>string indices must be integers, not 'tuple'</c>, <c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A NEP 55 <c>StringDType</c> <paramref name="a"/>: a line of two or more elements
        ///     holding a None-like NA (the sort's <c>Cannot compare null that is not a nan-like value</c>), or the unsafe
        ///     write-back into a numeric <paramref name="out"/> that cannot parse a string (<c>could not convert string to
        ///     float: 'a'</c>).</exception>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the checks that come first (q, the axes, the method, keepdims' out).</exception>
        /// <exception cref="ValueError">With <paramref name="weights"/>: another method (<c>Only method 'inverted_cdf'
        ///     supports weights. Got: linear.</c>), weights of a wrong shape, a negative weight, weights that sum to NaN,
        ///     inf or zero, or a repeated axis (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="TypeError">With <paramref name="weights"/>: weights of another shape without an axis, or
        ///     text weights (the comparison's refusal).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before q's rank, the method and the data's refusals.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex ("a must be an array of real numbers", NumPy's first
        ///     statement); a text <paramref name="q"/> (its [0, 1] check refuses); over text data an interpolating method
        ///     (NumPy subtracts the text), an <paramref name="out"/> that does not cast safely to the data, or keepdims over
        ///     a 0-d result.</exception>
        /// <exception cref="ArgumentException">A numeric quantile's q of rank above 1, a quantile outside [0, 1], an
        ///     unknown method, or an <paramref name="out"/> of another shape (numeric data; NumPy's ValueError
        ///     texts).</exception>
        /// <exception cref="ValueError">An axis repeats; over text data a q of rank above 2, an empty q of two
        ///     dimensions, or an <paramref name="out"/> of another shape.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take, or the interpolating methods'
        ///     neighbour lookup).</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0, any dtype:
        ///     <c>_ureduce</c>'s merge cannot resolve its -1 (NumPy's ValueError <c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>), raised before q's rank, the method and the data's refusals.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.quantile.html. NumPy's body is the engine's
        ///     (<see cref="TensorEngine.Quantile"/>): the complex check, q's checks, the weighted branch, the text routes
        ///     and the numeric quantile, in NumPy's order; <c>_ureduce</c> turns an integer axis into a one-element tuple, so
        ///     the engine member takes the axes as an array.
        ///     <para>A <c>numpy.matrix</c> operand: an integer axis gives the class rule's matrix, the class rule of a TUPLE
        ///     sees no axis (the rule NumSharp's tuple overloads had: the scalar of a full reduction, a row otherwise) -
        ///     NumPy's own answers come from matrix semantics inside its Python code, which NumSharp does not reproduce.</para>
        ///     <para>One signature since API-parity batch 4 (2026-11-12): it replaces the 8 overloads (q as
        ///     <c>double</c>, <c>double[]</c>, <see cref="NDArray"/> or <c>long</c>, by an <c>int?</c> or
        ///     <c>int[]</c> axis), which made <c>np.quantile(a, q, null)</c> ambiguous (CS0121) and had no tuple literal. Every
        ///     former q spelling converts, and the engine reads a converted C# number as the number it was (its
        ///     <c>QuantileQ</c>): each call keeps the answer it had.</para>
        /// </remarks>
        public static NDArray quantile(NDArray a, NDArray q,
            NDAxis? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false, NDArray weights = null)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // NumPy's np.quantile body is TensorEngine.Quantile's: q as NumPy reads it (a converted C# number is NumPy's
            // Python number again there), its checks, the weighted branch, the text routes and the numeric quantile, all
            // in NumPy's order. The axes go as an array (null for None): _ureduce's one-element tuple of an int axis.
            return a.TensorEngine.Quantile(a, q, axis?.ToArray(), @out, overwrite_input, method, keepdims, weights);
        }
    }
}
