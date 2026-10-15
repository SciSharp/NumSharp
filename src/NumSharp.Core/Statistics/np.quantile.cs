using System;
using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the q-th quantile of the data along the specified axis.
        ///     <paramref name="q"/> must be in the range [0, 1].
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.quantile.html</remarks>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray quantile(NDArray a, double q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidateQuantile(q);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("quantile", a, new[] { q }, System.Array.Empty<long>(), false, axisArr, @out, method, keepdims);
            return QuantileEngine.Compute(a, new[] { q }, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true);
        }

        /// <summary>
        ///     Compute the q-th quantiles of the data along the specified axis.
        ///     Each value in <paramref name="q"/> must be in [0, 1]. Result's first axis is q.
        /// </summary>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read).</exception>
        public static NDArray quantile(NDArray a, double[] q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("quantile", a, q, new long[] { q.Length }, false, axisArr, @out, method, keepdims);
            return QuantileEngine.Compute(a, q, axisArr, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false);
        }

        /// <summary>
        ///     The q-th quantile over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the reduced axes are merged into
        ///     one before the partition, so the result keeps only the other axes (0-d when every axis is reduced), and an
        ///     EMPTY axis list reduces nothing (each element is its own quantile).
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text takes NumPy's take path or its refusal, see the exceptions).</param>
        /// <param name="q">The quantile, in [0, 1].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The quantile over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1] ("Quantiles must be in the range [0, 1]"),
        ///     or an axis repeats.</exception>
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
        public static NDArray quantile(NDArray a, double q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            ValidateQuantile(q);
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("quantile", a, new[] { q }, System.Array.Empty<long>(), false, axis, @out, method, keepdims);
            return QuantileEngine.Compute(a, new[] { q }, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: true);
        }

        /// <summary>
        ///     The q-th quantiles over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the result's first axis is q, the
        ///     reduced axes are merged into one before the partition, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (any real dtype; fixed-width text takes NumPy's take path or its refusal, see the exceptions).</param>
        /// <param name="q">The quantiles, each in [0, 1].</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's thirteen names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>One quantile per element of <paramref name="q"/> along the first axis, over the reduced axes;
        ///     <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers", NumPy's
        ///     first statement (before q is read); or bool data with an interpolating method (NumPy's boolean-subtract
        ///     text).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="q"/>, <paramref name="a"/> or
        ///     <paramref name="method"/> is null.</exception>
        /// <exception cref="ValueError">A quantile is outside [0, 1] ("Quantiles must be in the range [0, 1]"), or an
        ///     axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="method"/> is not one of NumPy's names, or
        ///     <paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="IndexOutOfRangeException">The reduced axes hold no element (NumPy's IndexError text).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take: <c>cannot do a non-empty take from
        ///     an empty axes.</c>, or an index out of bounds).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (with a list of quantiles NumPy even raises a matrix-product error).</remarks>
        public static NDArray quantile(NDArray a, double[] q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            for (int i = 0; i < q.Length; i++) ValidateQuantile(q[i]);
            // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
            // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return QuantileOfText("quantile", a, q, new long[] { q.Length }, false, axis, @out, method, keepdims);
            return QuantileEngine.Compute(a, q, axis, @out, overwrite_input,
                QuantileEngine.ParseMethod(method), keepdims, qIsScalar: false);
        }

        /// <summary>
        ///     NDArray-q overload — accepts a 0-D or 1-D NDArray of quantile values (and, over text data, NumPy's 2-D q).
        ///     A numeric quantile of higher rank is rejected (NumPy raises "q must be a scalar or 1d" from rank 3).
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of quantiles in [0, 1] (up to 2-d over text data). An integer or bool q makes
        ///     NumPy's 'linear' virtual index <c>(n - 1) * q</c> an integer, which it TAKES instead of interpolating: the
        ///     'lower' answer, in the data's own dtype.</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The quantiles: one per element of q (a scalar result for a 0-d q), with the axis reduced.</returns>
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
        public static NDArray quantile(NDArray a, NDArray q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, q, axis, @out, overwrite_input, method, keepdims), static s => quantile(s.a, s.q, s.axis, s.@out, s.overwrite_input, s.method, s.keepdims), axis, keepdims, null);
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100, or the [0, 1] check), before the
            // data or q's rank is looked at.
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("quantile", a, q, method, axis.HasValue ? new[] { axis.Value } : null);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // Text data: NumPy accepts a q of any rank up to 2 (_quantile_ureduce_func refuses rank 3 and above, after the
            // range and axis checks), so the text route runs before the numeric path's rank check.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                double[] qText = QArrayFromNDArray(q);
                ValidateQuantileArray(q, qText);
                return QuantileOfText("quantile", a, qText, (long[])q.Shape.dimensions.Clone(), IsIntegerQuantile(q),
                    axisArr, @out, method, keepdims);
            }
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            bool qIsScalar = q.ndim == 0;
            double[] qArr = QArrayFromNDArray(q);
            for (int i = 0; i < qArr.Length; i++) ValidateQuantile(qArr[i]);
            var parsed = QuantileEngine.ParseMethod(method);
            // NumPy 2.4: `q = np.asanyarray(q)` whatever q is, so an integer (or bool) q — a Python int or bool among
            // them, over any data — makes 'linear's virtual index (n - 1) * q an integer, and _quantile then takes the
            // element there (its "supports_integers" path) instead of interpolating: exactly the 'lower' selection, with
            // the data's own dtype (np.quantile(int_array, np.array([0, 1])) is an int64 array, not float64, and
            // np.quantile([1., inf], 1) is inf where the interpolation would make inf - inf).
            if (parsed == QuantileMethod.Linear && IsIntegerQuantile(q))
                parsed = QuantileMethod.Lower;
            // weak_q = type(q) in (int, float): only a C# number (its 0-d array marked WasPythonLiteral) keeps float16 /
            // float32 data's dtype under interpolation; an array q, 0-d included, is a strong float64.
            return QuantileEngine.Compute(a, qArr, axisArr, @out, overwrite_input, parsed, keepdims, qIsScalar,
                weakQ: q.WasPythonLiteral);
        }

        /// <summary>
        ///     NDArray-q overload over SEVERAL axes at once — NumPy's <c>np.quantile(a, q_array, axis=(i, j))</c>: the
        ///     reduced axes are merged into one before the partition, the result's leading axes are q's, and an EMPTY axis
        ///     list reduces nothing.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">A 0-d or 1-d array of quantiles in [0, 1] (up to 2-d over text data). An integer or bool q makes
        ///     NumPy's 'linear' virtual index <c>(n - 1) * q</c> an integer, which it TAKES instead of interpolating.</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The quantiles over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="q"/> is null.</exception>
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
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain, as for every
        ///     tuple-axis overload (see the <see cref="double"/>-q one).</remarks>
        public static NDArray quantile(NDArray a, NDArray q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            if (q is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its [0, 1] check, before the data or q's rank is looked at.
            if (Backends.Strings.NDStringUfuncs.IsString(q))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer("quantile", a, q, method, axis);
            // Text data: q's rank is NumPy's check after the axes (rank 3 and above refused), so it runs in the text route.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                double[] qText = QArrayFromNDArray(q);
                ValidateQuantileArray(q, qText);
                return QuantileOfText("quantile", a, qText, (long[])q.Shape.dimensions.Clone(), IsIntegerQuantile(q),
                    axis, @out, method, keepdims);
            }
            if (q.ndim > 1) throw new ArgumentException("q must be a scalar or 1d");
            double[] qArr = QArrayFromNDArray(q);
            for (int i = 0; i < qArr.Length; i++) ValidateQuantile(qArr[i]);
            var parsed = QuantileEngine.ParseMethod(method);
            // An integer (or bool) q: 'linear' takes the element at the integer virtual index (see the one-axis overload).
            if (parsed == QuantileMethod.Linear && IsIntegerQuantile(q))
                parsed = QuantileMethod.Lower;
            // weak_q: only a C# number (WasPythonLiteral) is weak; an array q, 0-d included, is a strong float64.
            return QuantileEngine.Compute(a, qArr, axis, @out, overwrite_input, parsed, keepdims, qIsScalar: q.ndim == 0,
                weakQ: q.WasPythonLiteral);
        }

        /// <summary>
        ///     The q-th quantile for a Python INT q (a C# integer literal), NumPy 2.4's rule for it: <c>q =
        ///     np.asanyarray(q)</c> makes it an int64 array, whose 'linear' virtual index <c>(n - 1) * q</c> is an integer
        ///     that NumPy TAKES instead of interpolating — the data's own dtype and element, over float data too
        ///     (<c>np.quantile([1., inf], 1)</c> is inf), and over text the string itself.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">The quantile: 0 or 1 (anything else is outside [0, 1]).</param>
        /// <param name="axis">The axis to reduce, or null for the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method (<c>linear</c> by default; NumPy's names). Every selection method
        ///     takes the same element for q = 0 or 1; the other interpolating methods still interpolate.</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The quantile with the axis reduced (a scalar result over every axis).</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex: "a must be an array of real numbers".</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1] ("Quantiles must be in the range [0, 1]").</exception>
        /// <remarks>
        ///     C# binds an integer literal here rather than to the <see cref="double"/> overload (a conversion to
        ///     <see cref="long"/> is the better one), which is what makes <c>np.quantile(a, 0)</c> NumPy's call: over int
        ///     data it returns an int, over text it computes (NumPy's take path) where a float q would subtract text.
        ///     NumPy before 2.4 converted a Python int q to the data's dtype over float data; 2.4 no longer does.
        /// </remarks>
        public static NDArray quantile(NDArray a, long q,
            int? axis = null, NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // `q = np.asanyarray(q)` of a Python int: a 0-d int64 array, weak for the result dtype (weak_q). It is read
            // into the quantile list and never returned, so it is released here.
            using NDArray qArray = NDArray.PythonLiteral(q);
            return quantile(a, qArray, axis, @out, overwrite_input, method, keepdims);
        }

        /// <summary>
        ///     The q-th quantile over SEVERAL axes for a Python INT q (see the <see cref="long"/> overload with one axis for
        ///     NumPy's rule): the integer index is taken, whatever the data.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">The quantile: 0 or 1.</param>
        /// <param name="axis">The axes to reduce; null reduces the flattened data.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The estimation method.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The quantile over the reduced axes.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex.</exception>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1], or an axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        public static NDArray quantile(NDArray a, long q, int[] axis,
            NDArray @out = null, bool overwrite_input = false,
            string method = "linear", bool keepdims = false)
        {
            // `q = np.asanyarray(q)` of a Python int: a 0-d int64 array, weak for the result dtype (weak_q). It is read
            // into the quantile list and never returned, so it is released here.
            using NDArray qArray = NDArray.PythonLiteral(q);
            return quantile(a, qArray, axis, @out, overwrite_input, method, keepdims);
        }

        /// <summary>
        ///     NumPy's <c>_quantile_is_valid(q)</c> over a q array: a 1-d q of fewer than 10 elements element by element (an
        ///     empty one passes), any other through <c>q.min() &gt;= 0 and q.max() &lt;= 1</c> — which raises for an empty q of
        ///     another rank (the reduction has no identity).
        /// </summary>
        /// <param name="q">The q array (its rank and size).</param>
        /// <param name="values">Its values in C order.</param>
        /// <exception cref="ValueError">A quantile outside [0, 1] (or NaN): <c>Quantiles must be in the range [0, 1]</c>;
        ///     an empty q of rank other than 1: <c>zero-size array to reduction operation minimum which has no
        ///     identity</c>.</exception>
        private static void ValidateQuantileArray(NDArray q, double[] values)
        {
            if (values.Length == 0 && q.ndim != 1)
                throw new ValueError("zero-size array to reduction operation minimum which has no identity");
            for (int i = 0; i < values.Length; i++)
                ValidateQuantile(values[i]);
        }

        /// <summary>NumPy's <c>_quantile_is_valid</c> for one quantile: inside <c>[0, 1]</c> (NaN is not).</summary>
        /// <param name="q">The quantile.</param>
        /// <exception cref="ValueError"><paramref name="q"/> is outside [0, 1] ("Quantiles must be in the range [0, 1]").</exception>
        private static void ValidateQuantile(double q)
        {
            if (!(q >= 0.0 && q <= 1.0))
                throw new ValueError("Quantiles must be in the range [0, 1]");
        }

        /// <summary>
        ///     The first statement of np.quantile / np.percentile / np.nanquantile / np.nanpercentile:
        ///     <c>if a.dtype.kind == "c": raise TypeError("a must be an array of real numbers")</c>. np.median and
        ///     np.nanmedian do not have it (a complex median sorts lexicographically).
        /// </summary>
        /// <param name="a">The data operand (null passes; the caller's own null check reports it).</param>
        /// <exception cref="TypeError"><paramref name="a"/> is complex.</exception>
        internal static void RejectComplexQuantileData(NDArray a)
        {
            if (a is not null && a.typecode == NPTypeCode.Complex)
                throw new TypeError("a must be an array of real numbers");
        }

        /// <summary>A q array's values as doubles, in C order (one value for a 0-d q).</summary>
        /// <param name="q">The q array (a real dtype).</param>
        /// <returns>The values.</returns>
        /// <exception cref="InvalidCastException">A dtype without a real value (complex).</exception>
        private static double[] QArrayFromNDArray(NDArray q)
        {
            if (q.ndim == 0)
                return new[] { QuantileValueAt(q, 0) };
            double[] outQ = new double[q.size];
            for (long i = 0; i < q.size; i++) outQ[i] = QuantileValueAt(q, i);
            return outQ;
        }

        /// <summary>
        ///     One element of a q array as a double, the value NumPy's checks and virtual indexes read (every real dtype
        ///     converts exactly; <see cref="Half"/> is read by its own conversion, as it has no <see cref="IConvertible"/>).
        /// </summary>
        /// <param name="q">The q array.</param>
        /// <param name="i">The flat index, in C order.</param>
        /// <returns>The value.</returns>
        /// <exception cref="InvalidCastException">A dtype without a real value (complex).</exception>
        private static double QuantileValueAt(NDArray q, long i)
            => q.GetAtIndex(i) switch
            {
                Half h => (double)h,
                var v => Convert.ToDouble(v),
            };
    }
}
