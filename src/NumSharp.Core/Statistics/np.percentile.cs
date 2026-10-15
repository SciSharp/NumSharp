using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the q-th percentile of the data along the specified axis.
        ///     <paramref name="q"/> must be in [0, 100]. Equivalent to <c>np.quantile(a, q/100)</c>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.percentile.html</remarks>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray percentile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidatePercentile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("percentile", a, new[] { q / 100.0 }, System.Array.Empty<long>(), false, axisArr, @out, method, keepdims);
            return QuantileEngine.Compute(a, new[] { q / 100.0 }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true);
        }

        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray percentile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                ValidatePercentile(q[i]);
                qFracs[i] = q[i] / 100.0;
            }
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("percentile", a, qFracs, new long[] { qFracs.Length }, false, axisArr, @out, method, keepdims);
            return QuantileEngine.Compute(a, qFracs, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false);
        }

        /// <summary>
        ///     The q-th percentile over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the reduced axes are merged into
        ///     one before the partition, so the result keeps only the other axes (0-d when every axis is reduced), and an
        ///     EMPTY axis list reduces nothing (each element is its own percentile).
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text takes NumPy's take path or its refusal, see the exceptions).</param>
        /// <param name="q">The percentile, in [0, 100].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The percentile over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 100] ("Percentiles must be in the range
        ///     [0, 100]"), or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> or <paramref name="method"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="IndexOutOfRangeException">The reduced axes hold no element (NumPy's IndexError text).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (the median / percentile / quantile family's gap class).</remarks>
        public static NDArray percentile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidatePercentile(q);
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("percentile", a, new[] { q / 100.0 }, System.Array.Empty<long>(), false, axis, @out, method, keepdims);
            return QuantileEngine.Compute(a, new[] { q / 100.0 }, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true);
        }

        /// <summary>
        ///     The q-th percentiles over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the result's first axis is q,
        ///     the reduced axes are merged into one before the partition, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text takes NumPy's take path or its refusal, see the exceptions).</param>
        /// <param name="q">The percentiles, each in [0, 100].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>One percentile per element of <paramref name="q"/> along the first axis, over the reduced axes;
        ///     <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="q"/>, <paramref name="a"/> or
        ///     <paramref name="method"/> is null.</exception>
        /// <exception cref="ValueError">A percentile is outside [0, 100] ("Percentiles must be in the range [0, 100]"),
        ///     or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="IndexOutOfRangeException">The reduced axes hold no element (NumPy's IndexError text).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (with a list of percentiles NumPy even raises a matrix-product error).</remarks>
        public static NDArray percentile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            double[] qFracs = new double[q.Length];
            for (int i = 0; i < q.Length; i++)
            {
                ValidatePercentile(q[i]);
                qFracs[i] = q[i] / 100.0;
            }
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("percentile", a, qFracs, new long[] { qFracs.Length }, false, axis, @out, method, keepdims);
            return QuantileEngine.Compute(a, qFracs, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false);
        }

        /// <summary>
        ///     The q-th percentiles (q in [0, 100]) of the data, q given as a 0-d or 1-d array — NumPy's
        ///     <c>np.percentile(a, q_array)</c>.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of percentiles in [0, 100] (up to 2-d over text data).</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The percentiles: one per element of q (a scalar result for a 0-d q), with the axis reduced.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError">A text <paramref name="q"/>: NumPy divides it by 100 first, and <c>divide</c> has no
        ///     text loop (<c>ufunc 'divide' not supported for the input types, …</c>).</exception>
        /// <exception cref="ArgumentException">A numeric percentile's q of rank above 1 (NumPy's ValueError text).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100]; over text data a q of rank above 2 (<c>q must be a
        ///     scalar or 1d</c>, after the axis) or an empty q of two dimensions (its <c>q.min()</c> range
        ///     check).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray percentile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => percentile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100, or the [0, 1] check), before the
            // data or q's rank is looked at.
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("percentile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
            // Text data: NumPy accepts a q of any rank up to 2 (_quantile_ureduce_func refuses rank 3 and above, after the
            // range and axis checks), so the text route runs before the numeric path's rank check.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                // np.true_divide(q, 100) is float64 whatever q held: percentile never takes the integer index.
                return QuantileOfText("percentile", a, PercentileFractions(q), (long[])q.Shape.dimensions.Clone(), false,
                    axis.HasValue ? new[] { axis.Value } : null, @out, method, keepdims);
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            bool qIsScalar = q.ndim == 0;
            double[] qArr;
            if (q.ndim == 0)
            {
                double v = QuantileValueAt(q, 0);
                ValidatePercentile(v);
                qArr = new[] { v / 100.0 };
            }
            else
            {
                qArr = new double[q.size];
                for (long i = 0; i < q.size; i++)
                {
                    double v = QuantileValueAt(q, i);
                    ValidatePercentile(v);
                    qArr[i] = v / 100.0;
                }
            }
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // weak_q = type(q) in (int, float): only a C# number (its 0-d array marked WasPythonLiteral) keeps float16 /
            // float32 data's dtype under interpolation; an array q, 0-d included, is a strong float64.
            return QuantileEngine.Compute(a, qArr, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar, weakQ: q.WasPythonLiteral);
        }

        /// <summary>
        ///     The q-th percentiles over SEVERAL axes at once, q given as an array — NumPy's
        ///     <c>np.percentile(a, q_array, axis=(i, j))</c>: the reduced axes are merged into one before the partition, the
        ///     result's leading axes are q's, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of percentiles in [0, 100] (up to 2-d over text data).</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The percentiles over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex ("a must be an array of real numbers", NumPy's first
        ///     statement); a text <paramref name="q"/> (NumPy divides it by 100 first, and <c>divide</c> has no text loop);
        ///     over text data an interpolating method, an <paramref name="out"/> that does not cast safely to the data, or
        ///     keepdims over a 0-d result.</exception>
        /// <exception cref="ArgumentException">A numeric percentile's q of rank above 1, an unknown method, or an
        ///     <paramref name="out"/> of another shape (numeric data; NumPy's ValueError texts).</exception>
        /// <exception cref="ValueError">A percentile outside [0, 100]; an axis repeats; over text data a q of rank above 2,
        ///     an empty q of two dimensions, or an <paramref name="out"/> of another shape.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take, or the interpolating methods'
        ///     neighbour lookup).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain, as for every
        ///     tuple-axis overload.</remarks>
        public static NDArray percentile(NDArray a, NDArray q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100), before the data is looked at.
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("percentile", a, q, method, axis);
            double[] fractions = PercentileFractions(q);
            // Text data: q's rank is NumPy's check after the axes (rank 3 and above refused), so it runs in the text route;
            // np.true_divide(q, 100) is float64 whatever q held, so percentile never takes the integer index.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("percentile", a, fractions, (long[])q.Shape.dimensions.Clone(), false, axis, @out,
                    method, keepdims);
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            // weak_q: only a C# number (WasPythonLiteral) is weak; an array q, 0-d included, is a strong float64.
            return QuantileEngine.Compute(a, fractions, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: q.ndim == 0, weakQ: q.WasPythonLiteral);
        }

        /// <summary>
        ///     NumPy's <c>q = np.true_divide(q, 100)</c> followed by <c>_quantile_is_valid(q)</c>, over a q array: the
        ///     fractions in C order, every percentile checked to lie in [0, 100], and an empty q of rank other than 1
        ///     refused by the <c>q.min()</c> that NumPy's check takes for it.
        /// </summary>
        /// <param name="q">The percentiles (any numeric dtype).</param>
        /// <returns>The quantiles <c>q / 100</c> in C order.</returns>
        /// <exception cref="ValueError">A percentile outside [0, 100] (or NaN): <c>Percentiles must be in the range [0,
        ///     100]</c>; an empty q of rank other than 1: <c>zero-size array to reduction operation minimum which has no
        ///     identity</c>.</exception>
        private static double[] PercentileFractions(NDArray q)
        {
            var fractions = new double[q.size];
            if (fractions.Length == 0 && q.ndim != 1)
                throw new ValueError("zero-size array to reduction operation minimum which has no identity");
            for (long i = 0; i < fractions.Length; i++)
            {
                double v = QuantileValueAt(q, i);
                ValidatePercentile(v);
                fractions[i] = v / 100.0;
            }
            return fractions;
        }

        /// <summary>NumPy's <c>_quantile_is_valid</c> for one percentile: inside <c>[0, 100]</c> (NaN is not).</summary>
        /// <param name="q">The percentile.</param>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 100] ("Percentiles must be in the range [0,
        ///     100]").</exception>
        private static void ValidatePercentile(double q)
        {
            if (!(q >= 0.0 && q <= 100.0))
                throw new ValueError("Percentiles must be in the range [0, 100]");
        }
    }
}
