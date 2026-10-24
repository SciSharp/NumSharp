using System;

namespace NumSharp.Statistics
{
    /// <summary>
    ///     The C# form in which a public quantile-family overload received <c>q</c>. The form decides which of NumPy's
    ///     checks on q apply (only an array can be text, have a rank, or carry an integer dtype) and how q enters the
    ///     weighted branch.
    /// </summary>
    internal enum QuantileForm
    {
        /// <summary>A C# <see cref="double"/>: NumPy's Python float, a scalar quantile.</summary>
        Number,

        /// <summary>A C# <see cref="double"/> array: NumPy's Python list of floats, one quantile per element.</summary>
        List,

        /// <summary>
        ///     An <see cref="NDArray"/> of any rank and dtype, including the weak 0-d int64 array a C# integer literal
        ///     arrives as (<see cref="NDArray.WasPythonLiteral"/>).
        /// </summary>
        Array,
    }

    /// <summary>
    ///     <c>q</c> exactly as a public quantile-family overload received it, NOT yet checked. The facade only wraps its C#
    ///     argument, so every check NumPy makes on q (a null list, a text array, the [0, 1] range, percentile's division
    ///     by 100) runs in the engine, after NumPy's first statement (the complex check on the data). Validating while
    ///     wrapping would reorder those errors.
    /// </summary>
    internal readonly struct QuantileArgument
    {
        /// <summary>Wraps a C# number (<see cref="QuantileForm.Number"/>).</summary>
        /// <param name="number">The quantile (or percentile) as given, any value: the range check runs later.</param>
        internal QuantileArgument(double number)
        {
            Form = QuantileForm.Number;
            Number = number;
            List = null;
            Array = null;
        }

        /// <summary>Wraps a C# list of quantiles (<see cref="QuantileForm.List"/>).</summary>
        /// <param name="list">The quantiles as given; null is kept, and the engine reports it after the complex
        ///     check (NumPy's order).</param>
        internal QuantileArgument(double[] list)
        {
            Form = QuantileForm.List;
            Number = 0;
            List = list;
            Array = null;
        }

        /// <summary>Wraps a q array (<see cref="QuantileForm.Array"/>).</summary>
        /// <param name="array">The q array as given; null is kept, and the engine reports it after the complex
        ///     check.</param>
        internal QuantileArgument(NDArray array)
        {
            Form = QuantileForm.Array;
            Number = 0;
            List = null;
            Array = array;
        }

        /// <summary>Which constructor made this value, and so which member holds q.</summary>
        internal QuantileForm Form { get; }

        /// <summary>The number of a <see cref="QuantileForm.Number"/> q (0 for the other forms).</summary>
        internal double Number { get; }

        /// <summary>The list of a <see cref="QuantileForm.List"/> q, possibly null (null for the other forms).</summary>
        internal double[] List { get; }

        /// <summary>The array of a <see cref="QuantileForm.Array"/> q, possibly null (null for the other forms).</summary>
        internal NDArray Array { get; }
    }

    internal static partial class QuantileEngine
    {
        /// <summary>
        ///     NumPy's <c>np.quantile</c> / <c>np.percentile</c> body after the class rules: the complex check on the data,
        ///     q's checks (and percentile's division by 100), the weighted branch, the text routes and the numeric quantile,
        ///     in the order every public overload runs them. The facade applies the <c>numpy.matrix</c> rule and, with
        ///     weights, the <c>numpy.char.chararray</c> rule, then calls this.
        /// </summary>
        /// <param name="op">The NumPy function, <c>quantile</c> or <c>percentile</c>: percentile divides q by 100 and checks
        ///     the quotient against [0, 1] with its own text, and every refusal names the function.</param>
        /// <param name="a">The data (any real dtype; text takes NumPy's take path or its refusal).</param>
        /// <param name="q">q as the overload received it (<see cref="QuantileArgument"/>).</param>
        /// <param name="axes">The reduced axes as the caller gave them (a one-element array for an <c>int?</c> overload),
        ///     or null for the flattened data. An EMPTY array reduces nothing.</param>
        /// <param name="out">An optional output array, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method name (NumPy's thirteen names; checked where NumPy looks it up).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c>, or null; a weighted call takes NumPy's weighted branch
        ///     (<see cref="QuantileWeighted"/>) for every dtype.</param>
        /// <param name="chararray">True when <paramref name="a"/> is a <c>numpy.char.chararray</c> operand of the user's call,
        ///     read by the facade before any region hid the class: NumPy's text route then meets the STRING partition
        ///     (<see cref="QuantileText.Compute"/>), whatever the operand's dtype. A chararray with weights never arrives
        ///     here with the flag set: the facade's <c>NDCharArrayBoundary.Kept</c> rule runs the plain call (this one,
        ///     with the flag false) and keeps the class.</param>
        /// <returns>The quantiles (q's axes first); <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, a list or array <paramref name="q"/>, or
        ///     <paramref name="method"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex (<c>a must be an array of real numbers</c>, NumPy's
        ///     first statement); a text q (the [0, 1] check's or the division's refusal); text data with an interpolating
        ///     method, an out that does not cast safely, or keepdims over a 0-d result; bool data with an interpolating
        ///     method; a chararray operand (<c>chararray.partition() got an unexpected keyword argument 'axis'</c>, after the
        ///     checks that come first); weights of another shape without an axis, or text weights.</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1] (<c>Quantiles must be in the range [0, 1]</c>) or a
        ///     percentile outside [0, 100] (<c>Percentiles must be in the range [0, 100]</c>); an empty q of rank other than
        ///     1 (its <c>q.min()</c>); a repeated axis; a q of rank above 2; a <c>StringDType</c> line holding a None-like NA,
        ///     or an unsafe write-back that cannot parse a string; the weights' checks (another method, a wrong shape, a
        ///     negative weight, a sum that is NaN, inf or zero).</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException">An unknown method name, or an <paramref name="out"/> of another shape than
        ///     the result (numeric data).</exception>
        /// <exception cref="IndexError">Text data with an empty reduced axis (NumPy's take).</exception>
        /// <exception cref="IndexOutOfRangeException">Numeric data whose reduced axes hold no element (NumPy's IndexError
        ///     texts).</exception>
        /// <exception cref="InvalidCastException">A q array of a dtype without a real value (complex).</exception>
        /// <exception cref="OverflowException">An integer q array whose virtual indexes leave its dtype (see
        ///     <see cref="Compute"/>).</exception>
        /// <remarks>
        ///     Moved from the sixteen public overloads of <c>np.quantile</c> and <c>np.percentile</c> by the np-layer dtype
        ///     plan's P1a (step 1), statement for statement in each overload's order: the facade's outcome is unchanged
        ///     (the snapshot tool's diff of the move is empty). Where an overload validated q again in a later branch, the
        ///     check now runs once before the branches; nothing observable happens between the two places. The plan's step
        ///     2 replaces this prologue with NumPy's shared one (<c>_ureduce</c>'s order).
        /// </remarks>
        internal static NDArray Quantile(string op, NDArray a, QuantileArgument q, int[] axes, NDArray @out,
            bool overwrite_input, string method, bool keepdims, NDArray weights, bool chararray)
        {
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            bool percent = op == "percentile";
            switch (q.Form)
            {
                case QuantileForm.Number:
                {
                    // np.quantile: `_quantile_is_valid(q)`. np.percentile: `q = np.true_divide(q, 100, out=...)` (float64 for
                    // a Python number), then `_quantile_is_valid` on the QUOTIENT: -5e-324 divides to -0.0 and passes.
                    double value;
                    if (percent)
                        value = PercentileFraction(q.Number);
                    else
                    {
                        ValidateQuantile(q.Number);
                        value = q.Number;
                    }
                    // weights=: NumPy checks them after q (the method, the axis, their shape and sign) and runs its weighted
                    // branch for every dtype (QuantileWeighted).
                    if (weights is not null)
                    {
                        using var qArray = NDArray.Scalar(value);
                        return QuantileWeighted.Compute(a, qArray, axes, @out, keepdims, method, weights);
                    }
                    // Text: NumPy checks q (above), then the axis and the method; the selection methods (and 'linear' with an
                    // integer q) then partition and pick the text, the interpolating ones refuse it (QuantileOfText; a
                    // StringDType array reads the two neighbours before _lerp refuses them).
                    if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                        return QuantileOfText(op, a, new[] { value }, System.Array.Empty<long>(), false, axes, @out, method,
                            keepdims, chararray);
                    return Compute(a, new[] { value }, axes, @out, overwrite_input,
                        default, keepdims, qIsScalar: true, methodName: method);
                }
                case QuantileForm.List:
                {
                    if (q.List is null) throw new ArgumentNullException(nameof(q));
                    // Every element checked (percentile: each divided by 100 first); the list itself is q for np.quantile.
                    double[] values;
                    if (percent)
                    {
                        values = new double[q.List.Length];
                        for (int i = 0; i < values.Length; i++)
                            values[i] = PercentileFraction(q.List[i]);
                    }
                    else
                    {
                        for (int i = 0; i < q.List.Length; i++)
                            ValidateQuantile(q.List[i]);
                        values = q.List;
                    }
                    // weights=: see the number form; q becomes NumPy's float64 array of the list.
                    if (weights is not null)
                    {
                        using var qArray = np.array(values);
                        return QuantileWeighted.Compute(a, qArray, axes, @out, keepdims, method, weights);
                    }
                    if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                        return QuantileOfText(op, a, values, new long[] { values.Length }, false, axes, @out, method,
                            keepdims, chararray);
                    return Compute(a, values, axes, @out, overwrite_input,
                        default, keepdims, qIsScalar: false, methodName: method);
                }
                default:
                {
                    NDArray qArray = q.Array;
                    if (qArray is null) throw new ArgumentNullException(nameof(q));
                    // Text q: NumPy's answer comes from its first statement on q (divide by 100, or the [0, 1] check),
                    // before the data or q's rank is looked at.
                    if (Backends.Strings.NDStringUfuncs.IsVString(qArray))
                        throw Backends.Strings.NDVStringStatistics.QuantileVStringQ(op, a, qArray, method, axes);
                    if (Backends.Strings.NDStringUfuncs.IsString(qArray))
                        throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer(op, a, qArray, method, axes);
                    return percent
                        ? PercentileOfArray(op, a, qArray, axes, @out, overwrite_input, method, keepdims, weights, chararray)
                        : QuantileOfArray(op, a, qArray, axes, @out, overwrite_input, method, keepdims, weights, chararray);
                }
            }
        }

        /// <summary>
        ///     <see cref="Quantile"/> for an <c>np.quantile</c> q array (after the complex check and the text-q answers):
        ///     the range check at any rank, then the weighted branch, the text route and the numeric quantile. An integer or
        ///     bool q makes 'linear' take the element at its integer virtual index.
        /// </summary>
        /// <param name="op">The NumPy function (<c>quantile</c>), for the refusals' texts.</param>
        /// <param name="a">The data.</param>
        /// <param name="q">The q array (not null, not text).</param>
        /// <param name="axes">The reduced axes, or null for every axis.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c>, or null.</param>
        /// <param name="chararray">The chararray flag (see <see cref="Quantile"/>).</param>
        /// <returns>The quantiles; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ValueError">A quantile outside [0, 1], an empty q of rank other than 1, and the other
        ///     errors of <see cref="Quantile"/>.</exception>
        /// <exception cref="InvalidCastException">A q dtype without a real value (complex).</exception>
        /// <exception cref="Exception">The errors of the weighted branch, the text route and <see cref="Compute"/>.</exception>
        private static NDArray QuantileOfArray(string op, NDArray a, NDArray q, int[] axes, NDArray @out,
            bool overwrite_input, string method, bool keepdims, NDArray weights, bool chararray)
        {
            // `_quantile_is_valid(q)` at any rank (an empty q of rank other than 1 fails its q.min()). Every branch below
            // read and checked q first, so the check runs once here.
            double[] values = QArrayFromNDArray(q);
            ValidateQuantileArray(q, values);
            // weights=: NumPy checks them after q and runs its weighted branch with the q array as it is.
            if (weights is not null)
                return QuantileWeighted.Compute(a, q, axes, @out, keepdims, method, weights);
            // Text data: NumPy accepts a q of any rank up to 2 (_quantile_ureduce_func refuses rank 3 and above, after the
            // range and axis checks), so the text route runs before the numeric path's rank check.
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                return QuantileOfText(op, a, values, (long[])q.Shape.dimensions.Clone(), IsIntegerQuantile(q), axes, @out,
                    method, keepdims, chararray, QuantileIndexes.IndexDtypeOf(q));
            bool qIsScalar = q.ndim == 0;
            // NumPy 2.4: `q = np.asanyarray(q)` whatever q is, so an integer (or bool) q — a Python int or bool among them,
            // over any data — makes 'linear's virtual index (n - 1) * q an integer, and _quantile then takes the element
            // there (its "supports_integers" path) instead of interpolating: exactly the 'lower' selection, with the data's
            // own dtype (np.quantile(int_array, np.array([0, 1])) is an int64 array, not float64, and
            // np.quantile([1., inf], 1) is inf where the interpolation would make inf - inf) — integerQ. weak_q =
            // type(q) in (int, float): only a C# number (its 0-d array marked WasPythonLiteral) keeps float16 / float32
            // data's dtype under interpolation; an array q, 0-d included, is a strong float64. The method name and q's
            // rank (a 2-d q keeps both axes) are checked inside, in NumPy's order.
            return Compute(a, values, axes, @out, overwrite_input, default, keepdims, qIsScalar,
                weakQ: q.WasPythonLiteral, qShape: qIsScalar ? null : (long[])q.Shape.dimensions.Clone(),
                methodName: method, integerQ: IsIntegerQuantile(q), qType: QuantileIndexes.IndexDtypeOf(q));
        }

        /// <summary>
        ///     <see cref="Quantile"/> for an <c>np.percentile</c> q array (after the complex check and the text-q answers):
        ///     NumPy's <c>np.true_divide(q, 100)</c> and range check, then the weighted branch, the text route and the
        ///     numeric quantile. A percentile never takes the integer index (the quotient is a float).
        /// </summary>
        /// <param name="op">The NumPy function (<c>percentile</c>), for the refusals' texts.</param>
        /// <param name="a">The data.</param>
        /// <param name="q">The q array (not null, not text).</param>
        /// <param name="axes">The reduced axes, or null for every axis.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c>, or null.</param>
        /// <param name="chararray">The chararray flag (see <see cref="Quantile"/>).</param>
        /// <returns>The percentiles; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ValueError">A percentile outside [0, 100], an empty q of rank other than 1, and the other
        ///     errors of <see cref="Quantile"/>.</exception>
        /// <exception cref="InvalidCastException">A q dtype without a real value (complex).</exception>
        /// <exception cref="Exception">The errors of the weighted branch, the text route and <see cref="Compute"/>.</exception>
        private static NDArray PercentileOfArray(string op, NDArray a, NDArray q, int[] axes, NDArray @out,
            bool overwrite_input, string method, bool keepdims, NDArray weights, bool chararray)
        {
            // `np.true_divide(q, 100)` and `_quantile_is_valid` at any rank; q's rank is `_quantile_ureduce_func`'s check,
            // after the axes, inside the engine (a 2-d q keeps both axes).
            double[] fractions = PercentileFractions(q);
            if (weights is not null)
            {
                // `q = np.true_divide(q, 100)` in q's own dtype (a float32 q stays float32), after the range check above.
                using var hundred = NDArray.PythonLiteral(100L);
                using var qArray = np.true_divide(q, hundred);
                return QuantileWeighted.Compute(a, qArray, axes, @out, keepdims, method, weights);
            }
            // Text data: q's rank is NumPy's check after the axes (rank 3 and above refused), so it runs in the text route;
            // np.true_divide(q, 100) is float64 whatever q held, so percentile never takes the integer index.
            if (chararray || Backends.Strings.NDStringUfuncs.IsText(a))
                return QuantileOfText(op, a, fractions, (long[])q.Shape.dimensions.Clone(), false, axes, @out, method,
                    keepdims, chararray, QuantileIndexes.PrecisionOf(q));
            // weak_q: only a C# number (WasPythonLiteral) is weak; an array q, 0-d included, is a strong float64.
            return Compute(a, fractions, axes, @out, overwrite_input,
                default, keepdims, qIsScalar: q.ndim == 0, weakQ: q.WasPythonLiteral, methodName: method,
                qShape: q.ndim == 0 ? null : (long[])q.Shape.dimensions.Clone(), qType: QuantileIndexes.PrecisionOf(q));
        }

        /// <summary>
        ///     NumPy's <c>np.nanquantile</c> / <c>np.nanpercentile</c> body after the class rule: the complex check on the
        ///     data, q's checks (and percentile's division by 100), the weighted branch, the text routes (a
        ///     <c>StringDType</c> array computes the quantile of what its NaN-like NAs leave, fixed-width text meets
        ///     <c>isnan</c>'s refusal) and the NaN-ignoring numeric quantile, in the order every public overload runs them.
        /// </summary>
        /// <param name="op">The NumPy function, <c>nanquantile</c> or <c>nanpercentile</c>: the percentile divides q by 100
        ///     and checks the quotient with its own text, and every refusal names the function.</param>
        /// <param name="a">The data.</param>
        /// <param name="q">q as the overload received it (<see cref="QuantileArgument"/>).</param>
        /// <param name="axes">The reduced axes as the caller gave them, or null for the flattened data.</param>
        /// <param name="out">An optional output array, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="method">The estimation method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c>, or null. With weights, empty data answers <c>np.nanmean</c>
        ///     (this call without weights), any other data runs the weighted branch over each line's non-NaN values.</param>
        /// <returns>The quantiles of the non-NaN values (NaN for an all-NaN slice); <paramref name="out"/> itself when one
        ///     is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, a list or array <paramref name="q"/>, or
        ///     <paramref name="method"/> is null.</exception>
        /// <exception cref="TypeError"><paramref name="a"/> is complex; a text q; fixed-width text data (<c>isnan</c> has no
        ///     text loop, or <c>np.nanmean</c>'s add.reduce refuses empty text); bool data with an interpolating method;
        ///     weights of another shape without an axis, or text weights.</exception>
        /// <exception cref="ValueError">A quantile or percentile out of range; an empty q of rank other than 1; a repeated
        ///     axis; a q of rank above 2 (except over data that is NaN everywhere); the weights' checks.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException">An unknown method name (except over data that is NaN everywhere), or an
        ///     <paramref name="out"/> of another shape.</exception>
        /// <exception cref="InvalidCastException">A q array of a dtype without a real value (complex).</exception>
        /// <exception cref="OverflowException">An integer q array whose virtual indexes leave its dtype.</exception>
        /// <remarks>
        ///     Moved from the public overloads of <c>np.nanquantile</c> and <c>np.nanpercentile</c> (and their private
        ///     <c>NanQuantileOfArrayQ</c> / <c>NanPercentileOfArrayQ</c>) by the np-layer dtype plan's P1a (step 1),
        ///     statement for statement in each overload's order. The weighted branch's empty-data answer used to re-enter
        ///     the public overload; it now re-enters this body, which is the same call: the facade's only rule, the
        ///     <c>numpy.matrix</c> one, cannot apply inside the weighted branch's plain-array region.
        /// </remarks>
        internal static NDArray NanQuantile(string op, NDArray a, QuantileArgument q, int[] axes, NDArray @out,
            bool overwrite_input, string method, bool keepdims, NDArray weights)
        {
            // `if a.dtype.kind == "c": raise TypeError(...)` — NumPy's first statement, before q is read.
            RejectComplexQuantileData(a);
            bool percent = op == "nanpercentile";
            switch (q.Form)
            {
                case QuantileForm.Number:
                {
                    // nanquantile: `_quantile_is_valid(q)`. nanpercentile: `q = np.true_divide(q, 100, out=...)` (float64 for
                    // a Python number), then `_quantile_is_valid` on the QUOTIENT: -5e-324 divides to -0.0 and passes.
                    double value;
                    if (percent)
                        value = PercentileFraction(q.Number);
                    else
                    {
                        ValidateQuantile(q.Number);
                        value = q.Number;
                    }
                    // weights=: NumPy checks them after q (the method, the axis, their shape and sign); empty data then
                    // answers np.nanmean (this call without weights), any other data runs the weighted branch over each
                    // line's non-NaN values (QuantileWeighted).
                    if (weights is not null)
                    {
                        using var qArray = NDArray.Scalar(value);
                        return QuantileWeighted.Compute(a, qArray, axes, @out, keepdims, method, weights,
                            (op, a, q, axes, @out, overwrite_input, method, keepdims), NanQuantileWithoutWeights);
                    }
                    // Text: NumPy checks q (above), then the axis, the NaN test or the method, and only then meets the
                    // text itself (NDStringUfuncs.QuantileAnswer). StringDType: np.isnan finds its NaN-like NAs, and NumPy
                    // computes the quantile of the rest.
                    if (Backends.Strings.NDStringUfuncs.IsVString(a))
                        return Backends.Strings.NDVStringStatistics.NanQuantile(op, a, new[] { value },
                            System.Array.Empty<long>(), false, axes, @out, method, keepdims);
                    if (Backends.Strings.NDStringUfuncs.IsString(a))
                        throw Backends.Strings.NDStringUfuncs.QuantileAnswer(op, a, method, axes, true);
                    return Compute(a, new[] { value }, axes, @out, overwrite_input,
                        default, keepdims, qIsScalar: true,
                        emptyReturnsNaN: true, ignoreNaN: true, methodName: method);
                }
                case QuantileForm.List:
                {
                    if (q.List is null) throw new ArgumentNullException(nameof(q));
                    double[] values;
                    if (percent)
                    {
                        values = new double[q.List.Length];
                        for (int i = 0; i < values.Length; i++)
                            values[i] = PercentileFraction(q.List[i]);
                    }
                    else
                    {
                        for (int i = 0; i < q.List.Length; i++)
                            ValidateQuantile(q.List[i]);
                        values = q.List;
                    }
                    // weights=: see the number form; q becomes NumPy's float64 array of the list.
                    if (weights is not null)
                    {
                        using var qArray = np.array(values);
                        return QuantileWeighted.Compute(a, qArray, axes, @out, keepdims, method, weights,
                            (op, a, q, axes, @out, overwrite_input, method, keepdims), NanQuantileWithoutWeights);
                    }
                    // Text: see the number form.
                    if (Backends.Strings.NDStringUfuncs.IsVString(a))
                        return Backends.Strings.NDVStringStatistics.NanQuantile(op, a, values, new long[] { values.Length },
                            false, axes, @out, method, keepdims);
                    if (Backends.Strings.NDStringUfuncs.IsString(a))
                        throw Backends.Strings.NDStringUfuncs.QuantileAnswer(op, a, method, axes, false);
                    return Compute(a, values, axes, @out, overwrite_input,
                        default, keepdims, qIsScalar: false,
                        emptyReturnsNaN: true, ignoreNaN: true, methodName: method);
                }
                default:
                    return percent
                        ? NanPercentileOfArray(op, a, q, axes, @out, overwrite_input, method, keepdims, weights)
                        : NanQuantileOfArray(op, a, q, axes, @out, overwrite_input, method, keepdims, weights);
            }
        }

        /// <summary>
        ///     NumPy's <c>np.nanquantile</c> for a q ARRAY, statement for statement (after <see cref="NanQuantile"/>'s complex
        ///     refusal): a null q; a text q answered by its [0, 1] check (an empty text q passes it, and only empty numeric
        ///     data goes on, to <c>np.nanmean</c>); <c>_quantile_is_valid(q)</c> at any rank; the weighted branch; then the
        ///     data — <c>StringDType</c> through its NaN-like NAs, fixed-width text refused by <c>isnan</c> after the axis
        ///     checks, numbers through <see cref="Compute"/>, which checks q's rank, the method name and the axes in NumPy's
        ///     order.
        /// </summary>
        /// <param name="op">The NumPy function (<c>nanquantile</c>).</param>
        /// <param name="a">The data.</param>
        /// <param name="q">q as received (<see cref="QuantileForm.Array"/>); the weighted branch's empty-data answer
        ///     re-enters <see cref="NanQuantile"/> with it.</param>
        /// <param name="axes">The reduced axes, or null.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c>, or null.</param>
        /// <returns>The quantiles of the non-NaN values; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException">The q array is null.</exception>
        /// <exception cref="TypeError">A text q (NumPy's [0, 1] check or the reduction after it).</exception>
        /// <exception cref="Exception">The other errors of <see cref="NanQuantile"/>.</exception>
        private static NDArray NanQuantileOfArray(string op, NDArray a, QuantileArgument q, int[] axes, NDArray @out,
            bool overwrite_input, string method, bool keepdims, NDArray weights)
        {
            NDArray qArray = q.Array;
            if (qArray is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its [0, 1] check on q, before the data or q's rank is looked at; an empty q
            // passes it, and only with empty numeric data does NumPy go on (the helper's null), to np.nanmean(a, axis) —
            // the engine's empty-input answer, which runs before it reads q.
            if (Backends.Strings.NDStringUfuncs.IsVString(qArray) || Backends.Strings.NDStringUfuncs.IsString(qArray))
            {
                var textQ = Backends.Strings.NDStringUfuncs.IsVString(qArray)
                    ? Backends.Strings.NDVStringStatistics.QuantileVStringQ(op, a, qArray, method, axes)
                    : Backends.Strings.NDStringUfuncs.QuantileTextQAnswer(op, a, qArray, method, axes);
                if (textQ is not null)
                    throw textQ;
                return Compute(a, System.Array.Empty<double>(), axes, @out, overwrite_input, QuantileMethod.Linear,
                    keepdims, qIsScalar: true, emptyReturnsNaN: true, ignoreNaN: true);
            }
            double[] values = QArrayFromNDArray(qArray);
            // `_quantile_is_valid(q)` at any rank (an empty q of rank other than 1 fails its q.min()).
            ValidateQuantileArray(qArray, values);
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign); empty data then answers
            // np.nanmean (this call without weights), any other data runs the weighted branch over each line's non-NaN
            // values (QuantileWeighted).
            if (weights is not null)
                return QuantileWeighted.Compute(a, qArray, axes, @out, keepdims, method, weights,
                    (op, a, q, axes, @out, overwrite_input, method, keepdims), NanQuantileWithoutWeights);
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                // StringDType data: NumPy validates q (above), then reads the data; q's rank is checked per line, by the
                // quantile of what is left after the NaN-like NAs are removed.
                return Backends.Strings.NDVStringStatistics.NanQuantile(op, a, values,
                    (long[])qArray.Shape.dimensions.Clone(), IsIntegerQuantile(qArray), axes, @out, method, keepdims,
                    QuantileIndexes.IndexDtypeOf(qArray));
            bool qIsScalar = qArray.ndim == 0;
            // Fixed-width text: NumPy checks the axes and then meets isnan, which has no text loop (q's rank is never
            // reached), or np.nanmean's add.reduce refusal for empty text (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer(op, a, method, axes, qIsScalar);
            // weak_q: only a C# number (WasPythonLiteral) is weak; an array q, 0-d included, is a strong float64. An integer
            // (or bool) q takes the element at the integer virtual index (integerQ). q's rank (a 2-d q keeps both axes) and
            // the method name are checked inside, in NumPy's order.
            return Compute(a, values, axes, @out, overwrite_input, default, keepdims, qIsScalar,
                emptyReturnsNaN: true, ignoreNaN: true, weakQ: qArray.WasPythonLiteral,
                qShape: qIsScalar ? null : (long[])qArray.Shape.dimensions.Clone(), methodName: method,
                integerQ: IsIntegerQuantile(qArray), qType: QuantileIndexes.IndexDtypeOf(qArray));
        }

        /// <summary>
        ///     NumPy's <c>np.nanpercentile</c> for a q ARRAY, statement for statement (after <see cref="NanQuantile"/>'s
        ///     complex refusal): a null q; a text q refused by <c>np.true_divide(q, 100)</c>; the division and
        ///     <c>_quantile_is_valid(q)</c> at any rank; the weighted branch (q divided by 100 in its own dtype); then the
        ///     data — <c>StringDType</c> through its NaN-like NAs, fixed-width text refused by <c>isnan</c> after the axis
        ///     checks, numbers through <see cref="Compute"/>, which checks q's rank, the method name and the axes in NumPy's
        ///     order.
        /// </summary>
        /// <param name="op">The NumPy function (<c>nanpercentile</c>).</param>
        /// <param name="a">The data.</param>
        /// <param name="q">q as received (<see cref="QuantileForm.Array"/>); the weighted branch's empty-data answer
        ///     re-enters <see cref="NanQuantile"/> with it.</param>
        /// <param name="axes">The reduced axes, or null.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">The method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c>, or null.</param>
        /// <returns>The percentiles of the non-NaN values; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException">The q array is null.</exception>
        /// <exception cref="TypeError">A text q (the division's refusal).</exception>
        /// <exception cref="Exception">The other errors of <see cref="NanQuantile"/>.</exception>
        private static NDArray NanPercentileOfArray(string op, NDArray a, QuantileArgument q, int[] axes, NDArray @out,
            bool overwrite_input, string method, bool keepdims, NDArray weights)
        {
            NDArray qArray = q.Array;
            if (qArray is null) throw new ArgumentNullException(nameof(q));
            // Text q: NumPy's answer comes from its first statement on q (divide by 100), before the data or q's rank is
            // looked at.
            if (Backends.Strings.NDStringUfuncs.IsVString(qArray))
                throw Backends.Strings.NDVStringStatistics.QuantileVStringQ(op, a, qArray, method, axes);
            if (Backends.Strings.NDStringUfuncs.IsString(qArray))
                throw Backends.Strings.NDStringUfuncs.QuantileTextQAnswer(op, a, qArray, method, axes);
            // `q = np.true_divide(q, 100)` and `_quantile_is_valid(q)` at any rank.
            double[] fractions = PercentileFractions(qArray);
            // weights=: NumPy checks them after q (the method, the axis, their shape and sign); empty data then answers
            // np.nanmean (this call without weights), any other data runs the weighted branch over each line's non-NaN
            // values (QuantileWeighted), q divided by 100 in its own dtype (a float32 q stays float32).
            if (weights is not null)
            {
                using var hundred = NDArray.PythonLiteral(100L);
                using var divided = np.true_divide(qArray, hundred);
                return QuantileWeighted.Compute(a, divided, axes, @out, keepdims, method, weights,
                    (op, a, q, axes, @out, overwrite_input, method, keepdims), NanQuantileWithoutWeights);
            }
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                // StringDType data: q divided by 100 and checked (any rank), then the data; q's rank is checked per line.
                return Backends.Strings.NDVStringStatistics.NanQuantile(op, a, fractions,
                    (long[])qArray.Shape.dimensions.Clone(), false, axes, @out, method, keepdims,
                    QuantileIndexes.PrecisionOf(qArray));
            bool qIsScalar = qArray.ndim == 0;
            // Fixed-width text: NumPy checks the axes and then meets isnan, which has no text loop (q's rank is never
            // reached), or np.nanmean's add.reduce refusal for empty text (NDStringUfuncs.QuantileAnswer).
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                throw Backends.Strings.NDStringUfuncs.QuantileAnswer(op, a, method, axes, qIsScalar);
            // weak_q: only a C# number (WasPythonLiteral) is weak; an array q, 0-d included, is a strong float64. q's rank
            // (a 2-d q keeps both axes) and the method name are checked inside, in NumPy's order.
            return Compute(a, fractions, axes, @out, overwrite_input, default, keepdims, qIsScalar,
                emptyReturnsNaN: true, ignoreNaN: true, weakQ: qArray.WasPythonLiteral,
                qShape: qIsScalar ? null : (long[])qArray.Shape.dimensions.Clone(), methodName: method,
                qType: QuantileIndexes.PrecisionOf(qArray));
        }

        /// <summary>
        ///     The empty-data answer of the nan forms' weighted branch: <c>_nanquantile_unchecked</c>'s
        ///     <c>np.nanmean(a, axis, out=out, keepdims=keepdims)</c>, given by this call without weights (which answers an
        ///     empty input before it reads q or the method). <see cref="QuantileWeighted"/> calls it inside its plain-array
        ///     region.
        /// </summary>
        /// <param name="s">The call's arguments: the function, the data, q as received, the axes, out, overwrite_input,
        ///     the method and keepdims.</param>
        /// <returns><see cref="NanQuantile"/>'s answer without weights.</returns>
        /// <exception cref="Exception">The errors of <see cref="NanQuantile"/> (NumPy's checks run again, and pass, before
        ///     the empty input is answered; text data refuses as <c>np.nanmean</c> does).</exception>
        private static NDArray NanQuantileWithoutWeights(
            (string op, NDArray a, QuantileArgument q, int[] axes, NDArray @out, bool overwrite_input, string method,
                bool keepdims) s)
            => NanQuantile(s.op, s.a, s.q, s.axes, s.@out, s.overwrite_input, s.method, s.keepdims, null);

        /// <summary>
        ///     The text branch shared by every <c>np.quantile</c> / <c>np.percentile</c> overload: NumPy's checks after q's
        ///     (the axes, the method, the interpolating methods' refusals — <see cref="Backends.Strings.NDStringUfuncs.QuantileAnswer"/>),
        ///     then its take path over the text (<see cref="QuantileText.Compute"/>).
        /// </summary>
        /// <param name="op">The NumPy function name (<c>quantile</c> or <c>percentile</c>), for the refusals' texts.</param>
        /// <param name="a">The text data (<c>S</c> / <c>U</c>, or <c>StringDType</c>), or a chararray operand of any dtype
        ///     (see <paramref name="chararray"/>).</param>
        /// <param name="q">The quantiles in [0, 1] (a percentile already divided by 100), in C order.</param>
        /// <param name="qShape">q's shape: empty for a scalar q.</param>
        /// <param name="qIsInteger">True when q is an integer or bool array: NumPy's 'linear' then takes the integer index
        ///     <c>(n - 1) * q</c> instead of interpolating.</param>
        /// <param name="axes">The reduced axes, or null for every axis.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="method">The method name.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="chararray">True when <paramref name="a"/> is a <c>numpy.char.chararray</c> operand of the user's call
        ///     (<see cref="NDCharArrayBoundary.Sees(NDArray, string)"/>, read by the public overload BEFORE it opens any
        ///     <see cref="NDMatrixBoundary.PlainFor(NDArray, NDArray)"/> region, which would hide the chararray). Such an
        ///     operand comes here whatever its dtype (an <c>int8</c> chararray too): NumPy's statements up to the
        ///     <c>partition</c> call do not depend on it.</param>
        /// <param name="qType">q's precision (<see cref="QuantileIndexes.PrecisionOf"/>): a float16 / float32 q array
        ///     computes the virtual indexes in its own dtype, and the partition then validates them (NumPy's <c>kth(=k) out
        ///     of bounds (n)</c> for a long line).</param>
        /// <returns>The selected strings (or <paramref name="out"/> itself).</returns>
        /// <exception cref="AxisError">An axis is out of bounds.</exception>
        /// <exception cref="ValueError">An axis repeats, or <paramref name="out"/>'s shape differs from the result's.</exception>
        /// <exception cref="ValueError">An unknown method (NumPy's text; a ValueError is an <see cref="System.ArgumentException"/>).</exception>
        /// <exception cref="TypeError">An interpolating method (NumPy subtracts the text), an <paramref name="out"/> whose dtype
        ///     does not cast safely to the data's, or keepdims over a 0-d result (NumPy indexes its scalar).</exception>
        /// <exception cref="IndexError">An empty reduced axis (see <see cref="QuantileText.Compute"/> for the take path's
        ///     texts).</exception>
        /// <remarks>A <c>numpy.char.chararray</c> operand of the user's call meets NumPy's
        ///     <c>arr.partition(..., axis=0)</c>, which is <c>chararray.partition(sep)</c>: every check before it, then
        ///     <see cref="TypeError"/> <c>chararray.partition() got an unexpected keyword argument 'axis'</c> for every method
        ///     (see <see cref="QuantileText.Compute"/>). Moved from <c>np.quantile.Text.cs</c> by the np-layer dtype plan's
        ///     P1a (step 1).</remarks>
        private static NDArray QuantileOfText(string op, NDArray a, double[] q, long[] qShape, bool qIsInteger, int[] axes,
            NDArray @out, string method, bool keepdims, bool chararray, NPTypeCode qType = NPTypeCode.Double)
        {
            // _ureduce normalizes the axes before anything else reads them (AxisError, or a repeated axis's ValueError);
            // QuantileText.Compute then runs the rest in NumPy's order (keepdims' out index, q's rank, the method) — and for
            // a chararray operand stops at NumPy's arr.partition(..., axis=0), the STRING partition's TypeError.
            Backends.Strings.NDStringUfuncs.ValidateReduceAxes(op, a.ndim, axes);
            return QuantileText.Compute(op, a, q, qShape, qIsInteger, axes, @out, method, keepdims, chararray, qType);
        }

        /// <summary>
        ///     The first statement of np.quantile / np.percentile / np.nanquantile / np.nanpercentile:
        ///     <c>if a.dtype.kind == "c": raise TypeError("a must be an array of real numbers")</c>. np.median and
        ///     np.nanmedian do not have it (a complex median sorts lexicographically).
        /// </summary>
        /// <param name="a">The data operand (null passes; the null check of <see cref="Compute"/> reports it).</param>
        /// <exception cref="TypeError"><paramref name="a"/> is complex.</exception>
        internal static void RejectComplexQuantileData(NDArray a)
        {
            if (a is not null && a.typecode == NPTypeCode.Complex)
                throw new TypeError("a must be an array of real numbers");
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

        /// <summary>
        ///     NumPy's <c>q = np.true_divide(q, 100, out=...)</c> followed by <c>_quantile_is_valid(q)</c>, over a q array: the
        ///     quotients in C order, computed in q's own dtype (a float16 / float32 q divides in float16 / float32 and its
        ///     quotient is what NumPy's index arithmetic reads; every other q divides in float64), each checked to lie in
        ///     [0, 1], and an empty q of rank other than 1 refused by the <c>q.min()</c> that NumPy's check takes for it.
        /// </summary>
        /// <param name="q">The percentiles (any numeric dtype).</param>
        /// <returns>The quantiles <c>q / 100</c> in C order (values of q's precision, see
        ///     <see cref="QuantileIndexes.PrecisionOf"/>).</returns>
        /// <exception cref="ValueError">A quotient outside [0, 1] (or NaN): <c>Percentiles must be in the range [0,
        ///     100]</c> — a float16 <c>100.0625</c> divides to <c>1.0009765625</c> and is refused, while a negative value that
        ///     divides to <c>-0.0</c> passes; an empty q of rank other than 1: <c>zero-size array to reduction operation
        ///     minimum which has no identity</c>.</exception>
        /// <exception cref="InvalidCastException">A dtype without a real value (complex).</exception>
        private static double[] PercentileFractions(NDArray q)
        {
            var fractions = new double[q.size];
            if (fractions.Length == 0 && q.ndim != 1)
                throw new ValueError("zero-size array to reduction operation minimum which has no identity");
            NPTypeCode precision = QuantileIndexes.PrecisionOf(q);
            for (long i = 0; i < fractions.Length; i++)
            {
                // One true_divide in q's dtype: the double quotient rounded once to float16 / float32 is NumPy's (a float16
                // or float32 division computed in double and rounded is correctly rounded).
                double fraction = QuantileIndexes.Round(precision, QuantileValueAt(q, i) / 100.0);
                ValidatePercentileFraction(fraction);
                fractions[i] = fraction;
            }
            return fractions;
        }

        /// <summary>
        ///     NumPy's <c>q = np.true_divide(q, 100, out=...)</c> and <c>_quantile_is_valid(q)</c> for one float64 percentile (a
        ///     Python number, or an element of a list): the quotient, checked to lie in [0, 1].
        /// </summary>
        /// <param name="q">The percentile.</param>
        /// <returns><c>q / 100</c>.</returns>
        /// <exception cref="ValueError">The quotient is outside [0, 1] (or NaN): "Percentiles must be in the range [0,
        ///     100]". The check reads the quotient, so a percentile of <c>-5e-324</c> (whose quotient is <c>-0.0</c>)
        ///     passes, as in NumPy.</exception>
        private static double PercentileFraction(double q)
        {
            double fraction = q / 100.0;
            ValidatePercentileFraction(fraction);
            return fraction;
        }

        /// <summary>NumPy's <c>_quantile_is_valid</c> of one quotient <c>q / 100</c>: inside <c>[0, 1]</c> (NaN is not).</summary>
        /// <param name="fraction">The quotient.</param>
        /// <exception cref="ValueError"><paramref name="fraction"/> is outside [0, 1] ("Percentiles must be in the range [0,
        ///     100]").</exception>
        private static void ValidatePercentileFraction(double fraction)
        {
            if (!(fraction >= 0.0 && fraction <= 1.0))
                throw new ValueError("Percentiles must be in the range [0, 100]");
        }

        /// <summary>Whether a q array holds integers or bools: NumPy's <c>(n - 1) * q</c> is then an integer index.</summary>
        /// <param name="q">The q array.</param>
        /// <returns>True for a bool or integer dtype.</returns>
        private static bool IsIntegerQuantile(NDArray q)
            => q.typecode is NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16
                or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;
    }
}
