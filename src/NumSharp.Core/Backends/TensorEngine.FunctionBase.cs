// The functions of numpy/lib/_function_base_impl.py that compute, as TensorEngine members: NumPy writes them in
// Python, DefaultEngine runs those statements in NumPy's order, and an engine that derives from DefaultEngine can
// replace one for every array bound to it. The array-object functions of the same NumPy file (copy, place, insert,
// delete, flip, ...) are not engine members (docs/plans/np-layer-dtype-logic.md, section 1 "Which O", Appendix D).

using System;
using NumSharp.Backends;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.angle</c>: the counterclockwise angle of each element from the positive real axis, in
        ///     radians in (-pi, pi], or in degrees when <paramref name="deg"/> is set. A complex element gives
        ///     <c>arctan2(z.imag, z.real)</c>; a real one gives NumPy's <c>arctan2(0, z)</c>: 0 for a positive value or
        ///     +0, pi for a negative value or -0, NaN for NaN.
        /// </summary>
        /// <param name="z">
        ///     The input, a plain array: <c>np.angle</c> decides the <c>numpy.matrix</c> class before it calls this
        ///     member, so an override never sees a matrix operand.
        /// </param>
        /// <param name="deg">Return degrees (the radians multiplied by <c>180 / pi</c> in the result's own dtype).</param>
        /// <returns>
        ///     A new array of <paramref name="z"/>'s shape: float64 for complex128; for a real input the float tier of
        ///     NumPy's <c>arctan2(0, z)</c> with a weak 0 (float16 for int8 / uint8 / float16, float32 for int16 /
        ///     uint16 / char / float32, float64 for bool, the wider integers, float64 and decimal).
        /// </returns>
        /// <exception cref="TypeError">
        ///     <paramref name="z"/> holds text (<c>S</c>, <c>U</c> or <c>StringDType</c>): NumPy's <c>arctan2</c> has no
        ///     string loop, and the text names <c>arctan2</c> as NumPy's does.
        /// </exception>
        /// <remarks>
        ///     <para>NumPy writes <c>angle</c> in Python (<c>numpy/lib/_function_base_impl.py</c>); DefaultEngine's
        ///     override runs those statements. An override replaces <c>np.angle</c> for every array whose engine it is,
        ///     so it must keep NumPy's dtype tiers and the real-input values above.</para>
        /// </remarks>
        public abstract NDArray Angle(NDArray z, bool deg = false);

        /// <summary>
        ///     NumPy's <c>np.quantile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *,
        ///     weights=None)</c>: the q-th quantiles of the data, flattened or along the given axes.
        /// </summary>
        /// <param name="a">The data, a plain array (any real dtype; text takes NumPy's take path or its refusal).</param>
        /// <param name="q">
        ///     The quantiles in [0, 1], NumPy's array_like: a <see cref="double"/> or <see cref="float"/> is a Python float
        ///     (the result keeps a float data's dtype), a C# integer a Python int (an integer virtual index is TAKEN, not
        ///     interpolated), a <see cref="double"/> array a list, an <see cref="NDArray"/> an ndarray (q's axes lead the
        ///     result); anything else is converted with <c>np.asanyarray</c>.
        /// </param>
        /// <param name="axis">The reduced axes (negative counts from the end), or null for the flattened data; an EMPTY array
        ///     reduces nothing. A one-axis call passes a one-element array.</param>
        /// <param name="out">An optional output array, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (DefaultEngine stages a copy anyway).</param>
        /// <param name="method">One of NumPy's thirteen estimation methods (<c>linear</c>, <c>lower</c>, <c>higher</c>,
        ///     <c>nearest</c>, <c>midpoint</c>, <c>inverted_cdf</c>, ...), matched exactly.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (only with <c>inverted_cdf</c>), or null for equal weights.</param>
        /// <returns>The quantiles (q's axes first); <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="TypeError"><paramref name="a"/> is complex (<c>a must be an array of real numbers</c>, NumPy's
        ///     first statement); a text q; text data with an interpolating method, an out that does not cast safely, or
        ///     keepdims over a 0-d result; bool data with an interpolating method; weights of another shape without an
        ///     axis, or text weights.</exception>
        /// <exception cref="ValueError">A quantile outside [0, 1] (<c>Quantiles must be in the range [0, 1]</c>); an empty q
        ///     of rank other than 1; a repeated axis; a q of rank above 2; a <c>StringDType</c> line holding a None-like NA;
        ///     the weights' checks (another method, a wrong shape, a negative weight, a sum that is NaN, inf or
        ///     zero).</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, a null q, or <paramref name="method"/> is
        ///     null.</exception>
        /// <exception cref="ArgumentException">An unknown method name, or an <paramref name="out"/> of another shape than
        ///     the result.</exception>
        /// <exception cref="IndexError">keepdims with an <paramref name="out"/> NumPy cannot index (<c>_ureduce</c>'s
        ///     prologue); text data with an empty reduced axis.</exception>
        /// <exception cref="IndexOutOfRangeException">Numeric data whose reduced axes hold no element (NumPy's IndexError
        ///     texts).</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0 (NumPy's ValueError
        ///     <c>cannot reshape array of size 0 into shape (0,newaxis)</c>).</exception>
        /// <exception cref="InvalidCastException">A q array of a complex dtype.</exception>
        /// <exception cref="OverflowException">An integer q array whose virtual indexes leave its dtype.</exception>
        /// <remarks>
        ///     NumPy writes quantile in Python; DefaultEngine's override runs its statements in NumPy's order (the complex
        ///     check, q's checks, the weighted branch, the text routes, then <c>_ureduce</c> and the numeric quantile).
        ///     <c>np.quantile</c> calls it after its class rules (a <c>numpy.char.chararray</c> operand takes DefaultEngine's
        ///     internal form, which keeps the class boundary's flag); an override replaces it for every array whose engine
        ///     it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal class <c>QuantileEngine</c>.
        /// </remarks>
        public abstract NDArray Quantile(NDArray a, object q, int[] axis = null, NDArray @out = null,
            bool overwrite_input = false, string method = "linear", bool keepdims = false, NDArray weights = null);

        /// <summary>
        ///     NumPy's <c>np.percentile(a, q, axis=None, out=None, overwrite_input=False, method='linear', keepdims=False, *,
        ///     weights=None)</c>: <see cref="Quantile"/> with q given in percent, <c>q = np.true_divide(q, 100)</c> first.
        /// </summary>
        /// <param name="a">The data, a plain array.</param>
        /// <param name="q">The percentiles in [0, 100], NumPy's array_like (the forms of <see cref="Quantile"/>'s q; the
        ///     division by 100 is float64 for a number, q's own dtype for an array, so an integer index is never
        ///     taken).</param>
        /// <param name="axis">The reduced axes, or null for the flattened data; an EMPTY array reduces nothing.</param>
        /// <param name="out">An optional output array, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="method">One of NumPy's thirteen estimation methods, matched exactly.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="weights">NumPy's <c>weights=</c> (only with <c>inverted_cdf</c>), or null.</param>
        /// <returns>The percentiles (q's axes first); <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ValueError">A percentile outside [0, 100] (<c>Percentiles must be in the range [0, 100]</c>, on
        ///     the quotient), and the other ValueErrors of <see cref="Quantile"/>.</exception>
        /// <exception cref="TypeError">The TypeErrors of <see cref="Quantile"/> (a text q is refused by the division).</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/>, a null q, or <paramref name="method"/> is
        ///     null.</exception>
        /// <exception cref="ArgumentException">An unknown method name, or an out of another shape.</exception>
        /// <exception cref="IndexError">keepdims with an out NumPy cannot index; text data with an empty reduced
        ///     axis.</exception>
        /// <exception cref="IndexOutOfRangeException">Numeric data whose reduced axes hold no element.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0.</exception>
        /// <exception cref="InvalidCastException">A q array of a complex dtype.</exception>
        /// <remarks>
        ///     NumPy writes percentile in Python; <c>np.percentile</c> calls this after its class rules. An override
        ///     replaces it for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the
        ///     internal class <c>QuantileEngine</c>.
        /// </remarks>
        public abstract NDArray Percentile(NDArray a, object q, int[] axis = null, NDArray @out = null,
            bool overwrite_input = false, string method = "linear", bool keepdims = false, NDArray weights = null);

        /// <summary>
        ///     NumPy's <c>np.median(a, axis=None, out=None, overwrite_input=False, keepdims=False)</c>: the median of the data,
        ///     flattened or along the given axes; an even-sized slice gives the MEAN of its two central values.
        /// </summary>
        /// <param name="a">The data, a plain array (bool computes as its 0 / 1 values; text is refused as NumPy refuses it,
        ///     a <c>StringDType</c> array is sorted first).</param>
        /// <param name="axis">The reduced axes (negative counts from the end), or null for the flattened data; an EMPTY array
        ///     reduces nothing. A one-axis call passes a one-element array (NumPy turns an int axis into a 1-tuple).</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The median (NaN for an empty slice); <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An axis repeats; or, for a <c>StringDType</c> array, a None-like NA in a sorted line,
        ///     an empty reduced axis or a read-only out.</exception>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data or a text out that NumPy's closing mean refuses (its verbatim
        ///     text).</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0.</exception>
        /// <exception cref="IndexError">keepdims with an out NumPy cannot index, before the text refusals.</exception>
        /// <remarks>
        ///     NumPy writes median in Python (<c>_ureduce</c> over <c>_median</c>); <c>np.median</c> calls this after its
        ///     class rules. An override replaces it for every array whose engine it is. New in 0.71.0 (the np-layer dtype
        ///     plan, phase PE); it was the internal class <c>QuantileEngine</c>.
        /// </remarks>
        public abstract NDArray Median(NDArray a, int[] axis = null, NDArray @out = null, bool overwrite_input = false,
            bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.average(a, axis=None, weights=None, returned=False, *, keepdims=False)</c>: the (weighted) average
        ///     over the given axes, and with <paramref name="returned"/> the sum of the weights in the average's shape.
        /// </summary>
        /// <param name="a">The operand, a plain array (not null): any dtype and layout; text takes NumPy's text
        ///     statements.</param>
        /// <param name="axis">The reduced axes (a negative axis counts from the end), or null for every axis. A one-axis
        ///     call passes a one-element array.</param>
        /// <param name="weights">The weights, or null for the plain mean: of <paramref name="a"/>'s shape, or of the reduced
        ///     axes' extents in the order of <paramref name="axis"/>.</param>
        /// <param name="returned">Also compute the sum of weights (NumPy's <c>returned=True</c>); false leaves the pair's
        ///     second element null.</param>
        /// <param name="keepdims">Keep the reduced axes with length 1.</param>
        /// <returns>
        ///     The average and the sum of weights (null unless <paramref name="returned"/>): arrays the caller owns, 0-d for a
        ///     reduction over every axis without <paramref name="keepdims"/>. The pair stands for NumPy's return, which is
        ///     the average alone or the tuple.
        /// </returns>
        /// <exception cref="AxisError">An axis out of bounds (<c>axis: axis 9 is out of bounds for array of dimension 2</c>:
        ///     NumPy names the argument).</exception>
        /// <exception cref="ValueError">An axis that appears twice (<c>repeated axis in `axis` argument</c>); weights not of
        ///     the reduced axes' shape.</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero (<c>Weights sum to zero, can't be
        ///     normalized</c>), or a mean without weights whose result is empty (<c>division by zero</c>).</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis (<c>Axis must be specified when shapes of a
        ///     and weights differ.</c>); the weight sum's refusal of a text result dtype; a text operand without weights (the
        ///     mean's add.reduce refusal).</exception>
        /// <exception cref="DTypePromotionError">Weights and an operand with no common DType (StringDType beside a number or
        ///     beside S).</exception>
        /// <exception cref="InvalidOperationException">Not reached: a weight sum in a text result dtype that NumPy refuses but
        ///     that computed (a wiring error inside NumSharp).</exception>
        /// <remarks>
        ///     NumPy writes average in Python; DefaultEngine's override runs its statements in NumPy's order.
        ///     <c>np.average</c> and <c>np.average_returned</c> call it after the numpy.matrix rule and the null check; an
        ///     override replaces both for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE);
        ///     it was the internal class <c>AverageEngine</c>.
        /// </remarks>
        public abstract (NDArray avg, NDArray sumOfWeights) Average(NDArray a, int[] axis = null, NDArray weights = null,
            bool returned = false, bool keepdims = false);

        /// <summary>
        ///     NumPy's <c>np.cov(m, y=None, rowvar=True, bias=False, ddof=None, fweights=None, aweights=None, *,
        ///     dtype=None)</c>: the covariance matrix of the variables (rows of <paramref name="m"/> unless
        ///     <paramref name="rowvar"/> is false, plus those of <paramref name="y"/>).
        /// </summary>
        /// <param name="m">The variables, a plain array (not null): 1-D or 2-D.</param>
        /// <param name="y">An additional set of variables, or null.</param>
        /// <param name="rowvar">Whether rows are variables (columns observations).</param>
        /// <param name="bias">Normalize by N instead of N - 1 (overridden by <paramref name="ddof"/>).</param>
        /// <param name="ddof">The delta degrees of freedom, or null for <paramref name="bias"/>'s default.</param>
        /// <param name="fweights">The integer frequency weights, or null.</param>
        /// <param name="aweights">The analytic weights, or null.</param>
        /// <param name="dtype">The result dtype, or null for <c>result_type(m, y, float64)</c>.</param>
        /// <returns>
        ///     The covariance matrix (squeezed: a 0-d array for one variable), a fresh array the caller owns; an empty
        ///     <c>(0, 0)</c> float64 array for no variables.
        /// </returns>
        /// <exception cref="ValueError">More than 2 dimensions; text variables whose <paramref name="m"/> and
        ///     <paramref name="y"/> have different observation counts; a negative weight; text that does not parse into a
        ///     numeric <paramref name="dtype"/>; a <c>StringDType</c> request over no observations.</exception>
        /// <exception cref="IncorrectShapeException">Numeric variables with different observation counts (concatenate).</exception>
        /// <exception cref="TypeError">A non-integral frequency weight; text the mean refuses.</exception>
        /// <exception cref="RuntimeError">A malformed weight array.</exception>
        /// <exception cref="DTypePromotionError"><c>StringDType</c> variables without <paramref name="dtype"/>, or a
        ///     <c>StringDType</c> request with weights.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/> (the in-place centring's cast
        ///     refusal).</exception>
        /// <exception cref="DivideByZeroException">Weights that sum to zero.</exception>
        /// <exception cref="OverflowException">An integer text outside an integer <paramref name="dtype"/>, or decimal
        ///     arithmetic out of range.</exception>
        /// <exception cref="UnicodeEncodeError">A U → S conversion of non-ASCII text.</exception>
        /// <exception cref="UnicodeDecodeError">An S → U conversion of non-ASCII bytes.</exception>
        /// <remarks>
        ///     NumPy writes cov in Python; DefaultEngine's override runs its statements (its product goes through
        ///     <c>np.dot</c>, so the OpenBLAS backend computes it when installed). <c>np.cov</c> (and <c>np.corrcoef</c>
        ///     through it) calls this after the matrix boundary's plain region and the null check; an override replaces it
        ///     for every array whose engine it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal
        ///     class <c>CovEngine</c>.
        /// </remarks>
        public abstract NDArray Cov(NDArray m, NDArray y = null, bool rowvar = true, bool bias = false, int? ddof = null,
            NDArray fweights = null, NDArray aweights = null, DType dtype = null);

        /// <summary>
        ///     NumPy's <c>np.corrcoef(x, y=None, rowvar=True, *, dtype=None)</c>: the Pearson correlation coefficients
        ///     <c>R_ij = C_ij / sqrt(C_ii * C_jj)</c> of <c>C = cov(x, y, rowvar, dtype=dtype)</c>, the two divisions made in
        ///     place, then the result clipped to [-1, 1] in place (the real and the imaginary lanes of a complex result), which
        ///     rounding can push a coefficient past.
        /// </summary>
        /// <param name="x">The variables, a plain array (not null): 1-D or 2-D, a variable per row unless
        ///     <paramref name="rowvar"/> is false.</param>
        /// <param name="y">An additional set of variables, or null.</param>
        /// <param name="rowvar">Whether rows are variables (columns observations).</param>
        /// <param name="dtype">The result dtype, or null for <see cref="Cov"/>'s <c>result_type(x, y, float64)</c>.</param>
        /// <returns>
        ///     The correlation matrix, a fresh array the caller owns (cov's own matrix, divided and clipped in place). For one
        ///     variable cov's result is 0-d, which has no diagonal: NumPy's <c>c / c</c> then, 1 or NaN (a NaN, infinite or
        ///     zero variance). No variables give cov's empty <c>(0, 0)</c> float64 array.
        /// </returns>
        /// <exception cref="ValueError">More than 2 dimensions; text variables whose <paramref name="x"/> and
        ///     <paramref name="y"/> have different observation counts; text that does not parse into a numeric
        ///     <paramref name="dtype"/>; a <c>StringDType</c> request over no observations (<see cref="Cov"/>'s).</exception>
        /// <exception cref="IncorrectShapeException">Numeric variables with different observation counts (cov's
        ///     concatenate).</exception>
        /// <exception cref="TypeError">Text that cov's mean refuses.</exception>
        /// <exception cref="DTypePromotionError"><c>StringDType</c> variables without <paramref name="dtype"/>.</exception>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="dtype"/> (cov's in-place centring refuses
        ///     the cast).</exception>
        /// <exception cref="OverflowException">An integer text outside an integer <paramref name="dtype"/>, or decimal
        ///     arithmetic out of range.</exception>
        /// <exception cref="UnicodeEncodeError">A U → S conversion of non-ASCII text.</exception>
        /// <exception cref="UnicodeDecodeError">An S → U conversion of non-ASCII bytes.</exception>
        /// <remarks>
        ///     NumPy writes corrcoef in Python over cov (NumPy 2.x dropped its <c>bias</c> / <c>ddof</c>); DefaultEngine runs
        ///     its statements over the operand engine's <see cref="Cov"/>, <see cref="Sqrt"/>, <see cref="Divide"/> and
        ///     <see cref="ClipNDArray"/> and the array object's diagonal and real / imaginary lanes. <c>np.corrcoef</c> calls
        ///     it after the matrix boundary's plain region and the null check; an override replaces it for every array whose
        ///     engine it is.
        /// </remarks>
        public abstract NDArray Corrcoef(NDArray x, NDArray y = null, bool rowvar = true, DType dtype = null);

        /// <summary>
        ///     NumPy's <c>np.gradient(f, *varargs, axis=None, edge_order=1)</c>: one fresh array per requested axis, with
        ///     second-order central differences inside and first- or second-order one-sided differences at both ends; each
        ///     component has <paramref name="f"/>'s shape.
        /// </summary>
        /// <param name="f">The sampled values, a plain array (null is refused by the body's first statement).</param>
        /// <param name="varargs">NumPy's <c>*varargs</c>: no entry (unit steps), one scalar for every axis, or one entry per
        ///     requested axis, each a scalar step (a C# number is a Python number, weak) or a 1-D coordinate array as long
        ///     as its axis.</param>
        /// <param name="axis">The axes to differentiate along (negative values count from the end), or null for every axis.
        ///     A one-axis call passes a one-element array. (DefaultEngine's override names it <c>axisArg</c>: its loop
        ///     rebinds NumPy's <c>axis</c>.)</param>
        /// <param name="edge_order">The boundary order: 1 gives first-order ends; any other value up to 2 (0 and negative
        ///     values included, as NumPy reads them) gives second-order ends.</param>
        /// <returns>One component per requested axis, a single array when exactly one axis was requested (NumPy's bare
        ///     ndarray), none for a 0-d <paramref name="f"/> without an axis.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="f"/> is null.</exception>
        /// <exception cref="AxisError">An axis outside <c>[-f.ndim, f.ndim)</c>.</exception>
        /// <exception cref="ValueError">A repeated axis; a ragged list spacing; a coordinate array of another rank than 1 or
        ///     another length than its axis; <paramref name="edge_order"/> above 2; an axis shorter than
        ///     <c>edge_order + 1</c>.</exception>
        /// <exception cref="NotSupportedException">A list spacing holding None or another object, or a Python int past
        ///     uint64 (NumPy's object array).</exception>
        /// <exception cref="TypeError">A spacing count other than 0, 1 or the number of axes; a bool
        ///     <paramref name="f"/> with a scalar step or first-order edges (NumPy's boolean subtract); a text
        ///     <paramref name="f"/>; bool coordinates; a str or None step.</exception>
        /// <exception cref="IndexError">The coordinates of an empty axis.</exception>
        /// <exception cref="DivideByZeroException">A zero step that is a Python number with second-order edges (CPython's
        ///     ZeroDivisionError); a decimal input or step divided by zero.</exception>
        /// <exception cref="OverflowException">A Python int step past the float range; a decimal result past decimal's
        ///     range.</exception>
        /// <remarks>
        ///     NumPy writes gradient in Python; DefaultEngine's override runs its statements, each in NumPy's dtype, so the
        ///     components are bit-identical to NumPy's. <c>np.gradient</c>'s overloads spread their spacings into one list,
        ///     apply the matrix boundary's plain region and call this; an override replaces it for every array whose engine
        ///     it is. New in 0.71.0 (the np-layer dtype plan, phase PE); it was the internal class <c>GradientEngine</c>.
        /// </remarks>
        public abstract np.GradientResult Gradient(NDArray f, object[] varargs, int[] axis = null, int edge_order = 1);

        /// <summary>
        ///     NumPy's <c>np.trapezoid(y, x=None, dx=1.0, axis=-1)</c>: the integral of <paramref name="y"/> along the axis
        ///     by the composite trapezoidal rule, <c>sum(d * (y[1:] + y[:-1]) / 2.0, axis)</c> with <c>d</c> the spacing.
        /// </summary>
        /// <param name="y">The integrand, a plain array (not null).</param>
        /// <param name="x">The sample points (1-D along the axis, or of <paramref name="y"/>'s rank), or null for evenly
        ///     spaced samples.</param>
        /// <param name="dx">The spacing when <paramref name="x"/> is null.</param>
        /// <param name="axis">The integration axis, read against each statement's array as NumPy reads it.</param>
        /// <returns>The integral: a fresh array the caller owns, of <paramref name="y"/>'s shape without the axis (0-d for a
        ///     1-D <paramref name="y"/>), its dtype NumPy's arithmetic result.</returns>
        /// <exception cref="IndexError">The axis is outside <paramref name="y"/>'s rank.</exception>
        /// <exception cref="AxisError">An N-D <paramref name="x"/> has no axis <paramref name="axis"/> (np.diff's
        ///     <c>normalize_axis_index</c>).</exception>
        /// <exception cref="ValueError">A 0-D <paramref name="x"/> (np.diff's "at least one dimensional"), or a StringDType
        ///     <paramref name="y"/> whose concatenation meets an NA it cannot add.</exception>
        /// <exception cref="IncorrectShapeException">The spacing does not broadcast against <c>y[1:] + y[:-1]</c>.</exception>
        /// <exception cref="TypeError">A text <paramref name="y"/> or <paramref name="x"/>: the refusal of the first NumPy
        ///     statement that has no loop for it.</exception>
        /// <remarks>
        ///     NumPy writes trapezoid in Python; DefaultEngine's override runs its statements, the half and its sum fused
        ///     into one pass where that gives NumPy's bits. <c>np.trapezoid</c> calls this after the matrix boundary and the
        ///     null check; an override replaces it for every array whose engine it is. New in 0.71.0 (the np-layer dtype
        ///     plan, phase PE); it was the internal class <c>TrapezoidEngine</c>.
        /// </remarks>
        public abstract NDArray Trapezoid(NDArray y, NDArray x = null, double dx = 1.0, int axis = -1);

        /// <summary>
        ///     NumPy's <c>np.sinc(x)</c>, the normalized sinc <c>sin(pi*x) / (pi*x)</c>: <c>y = pi * x</c>, every exact
        ///     zero of <c>y</c> replaced by the machine epsilon of <c>y</c>'s dtype (1e-20 for a dtype that is not float),
        ///     then <c>sin(y) / y</c>, so the removable singularity gives its limit 1.
        /// </summary>
        /// <param name="x">The input, any dtype and layout (not null).</param>
        /// <returns>The values, of <c>pi * x</c>'s dtype: float64 for bool, integers and char; float16 / float32 /
        ///     float64 kept; complex128 for complex; decimal for decimal (the double bridge).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="TypeError">A text operand: <c>pi * x</c> has no loop (NumPy's no-loop texts).</exception>
        /// <exception cref="OverflowException">A decimal <paramref name="x"/> whose <c>pi * x</c> leaves the decimal
        ///     range.</exception>
        /// <remarks>DefaultEngine runs the four statements as one fused <c>np.evaluate</c> pass.</remarks>
        public abstract NDArray Sinc(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.i0(x)</c>, the modified Bessel function of the first kind, order 0: <c>astype(float)</c> for a
        ///     dtype that is not float, <c>abs</c>, then the cephes Chebyshev series (<c>|x| &lt;= 8</c> and above), each
        ///     float width computed in its own precision as NumPy's ufunc chain computes it.
        /// </summary>
        /// <param name="x">The input, any dtype but complex, any layout (not null).</param>
        /// <returns>The values: float16 / float32 / float64 kept, float64 for bool, integers, char and parsed text,
        ///     decimal for decimal (the double bridge).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="IncorrectTypeException">A complex operand (NumPy's <c>i0 not supported for complex
        ///     values</c>).</exception>
        /// <exception cref="ValueError">Text that does not parse as a float (NumPy's <c>astype(float)</c>).</exception>
        /// <exception cref="OverflowException">A decimal result past the decimal range (<c>|x|</c> above about 72).</exception>
        /// <remarks>DefaultEngine runs the series per element in one fused <c>np.evaluate</c> pass.</remarks>
        public abstract NDArray I0(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.bartlett(M)</c>: the triangular window of <paramref name="M"/> points,
        ///     <c>where(n &lt;= 0, 1 + n/(M-1), 1 - n/(M-1))</c> over <c>n = arange(1-M, M, 2)</c> - zero at both ends, one at
        ///     the centre only when the count is odd.
        /// </summary>
        /// <param name="M">
        ///     The number of points, NumPy's <c>_FloatLike_co</c>: a fractional count gives NumPy's length
        ///     <c>len(arange(1 - M, M, 2))</c>; below one gives an empty array, exactly one gives <c>ones(1)</c>.
        /// </param>
        /// <returns>A new 1-D float64 array.</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN (arange's <c>arange: cannot compute length</c>), +infinity or
        ///     too large for arange to size (<c>Maximum allowed size exceeded</c>).</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     NumPy writes the window in Python; DefaultEngine fuses its ufunc chain into one <c>np.evaluate</c> pass in
        ///     NumPy's operation order, so the values are NumPy's bit for bit (float64 add, subtract and divide are
        ///     IEEE-exact). The window has no operand to choose an engine by: <c>np.bartlett</c> calls the default engine's
        ///     member, so an override is reached through that engine (or called directly).
        /// </remarks>
        public abstract NDArray Bartlett(double M);

        /// <summary>
        ///     NumPy's <c>np.blackman(M)</c>: the Blackman window of <paramref name="M"/> points, the first three terms of a
        ///     cosine sum, <c>0.42 + 0.5*cos(pi*n/(M-1)) + 0.08*cos(2.0*pi*n/(M-1))</c> over <c>n = arange(1-M, M, 2)</c>.
        /// </summary>
        /// <param name="M">The number of points, NumPy's <c>_FloatLike_co</c> (the length and the guards of
        ///     <see cref="Bartlett"/>).</param>
        /// <returns>A new 1-D float64 array (its end points are a tiny negative number, -1.3877787807814457e-17, as in
        ///     NumPy).</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, +infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     DefaultEngine's fused pass keeps NumPy's evaluation order (<c>(pi * n) / (M - 1)</c> and
        ///     <c>((2.0 * pi) * n) / (M - 1)</c> round in two steps each), and float64 <c>cos</c> is the host CRT's, NumPy's
        ///     own on win-amd64; elsewhere a libm can differ in the last bit. <c>np.blackman</c> calls the default engine's
        ///     member.
        /// </remarks>
        public abstract NDArray Blackman(double M);

        /// <summary>
        ///     NumPy's <c>np.hamming(M)</c>: the Hamming window of <paramref name="M"/> points, the raised cosine
        ///     <c>0.54 + 0.46*cos(pi*n/(M-1))</c> over <c>n = arange(1-M, M, 2)</c> - its ends stay at 0.08, not zero.
        /// </summary>
        /// <param name="M">The number of points, NumPy's <c>_FloatLike_co</c> (the length and the guards of
        ///     <see cref="Bartlett"/>).</param>
        /// <returns>A new 1-D float64 array.</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, +infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>DefaultEngine's fused pass keeps NumPy's evaluation order; <c>np.hamming</c> calls the default engine's
        ///     member.</remarks>
        public abstract NDArray Hamming(double M);

        /// <summary>
        ///     NumPy's <c>np.hanning(M)</c>: the Hann window of <paramref name="M"/> points, the raised cosine
        ///     <c>0.5 + 0.5*cos(pi*n/(M-1))</c> over <c>n = arange(1-M, M, 2)</c> - zero at both ends.
        /// </summary>
        /// <param name="M">The number of points, NumPy's <c>_FloatLike_co</c> (the length and the guards of
        ///     <see cref="Bartlett"/>).</param>
        /// <returns>A new 1-D float64 array.</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, +infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>DefaultEngine's fused pass keeps NumPy's evaluation order; <c>np.hanning</c> calls the default engine's
        ///     member.</remarks>
        public abstract NDArray Hanning(double M);

        /// <summary>
        ///     NumPy's <c>np.kaiser(M, beta)</c>: the Kaiser window of <paramref name="M"/> points,
        ///     <c>i0(beta * sqrt(1 - ((n - alpha)/alpha)**2)) / i0(beta)</c> over <c>n = arange(0, M)</c> with
        ///     <c>alpha = (M - 1)/2</c>.
        /// </summary>
        /// <param name="M">
        ///     The number of points, NumPy's <c>_FloatLike_co</c>. Unlike the cosine windows kaiser has no <c>M &lt; 1</c>
        ///     guard: <c>arange(0, M)</c> is empty for <c>M &lt;= 0</c> but one point for <c>0 &lt; M &lt; 1</c>, and only
        ///     <c>M == 1</c> is special-cased (<c>ones(1)</c>, which avoids <c>alpha == 0</c>). A fractional count past a
        ///     whole one leaves a trailing NaN where <c>|(n - alpha)/alpha| &gt; 1</c>, as in NumPy.
        /// </param>
        /// <param name="beta">The shape parameter: 0 is a rectangular window and a larger value a narrower one (5 is close to
        ///     a Hamming window, 6 to a Hann window, 8.6 to a Blackman window). Any double is taken; past about 709.78, where
        ///     the series' <c>exp</c> overflows, <c>i0(beta)</c> is infinite, so the window is NaN where its numerator
        ///     overflows too (<c>inf / inf</c>) and zero elsewhere.</param>
        /// <returns>A new 1-D float64 array.</returns>
        /// <exception cref="ValueError"><paramref name="M"/> is NaN, or an infinity or too large for arange to size.</exception>
        /// <exception cref="OutOfMemoryException">A finite count whose window does not fit in memory.</exception>
        /// <remarks>
        ///     DefaultEngine runs the cephes i0 series NumPy's <c>np.i0</c> runs once per element of a fused pass, and takes
        ///     <c>i0(beta)</c> from its own <see cref="I0"/> (NumPy's <c>i0(float(beta))</c>), so the values are NumPy's bit for
        ///     bit where the host's <c>exp</c> and <c>sqrt</c> are. <c>np.kaiser</c> calls the default engine's member.
        /// </remarks>
        public abstract NDArray Kaiser(double M, double beta);

        /// <summary>
        ///     NumPy's <c>np.diff(a, n=1, axis=-1, prepend=&lt;no value&gt;, append=&lt;no value&gt;)</c>: the n-th
        ///     discrete difference along an axis, <c>a = op(a[1:], a[:-1])</c> repeated n times after concatenating
        ///     prepend / append along the axis, where op is <c>!=</c> for bool and <c>-</c> otherwise.
        /// </summary>
        /// <param name="a">The input, at least 1-D (not null).</param>
        /// <param name="n">How many times to difference; 0 returns <paramref name="a"/> itself.</param>
        /// <param name="axis">The axis to difference along (negative counts from the end).</param>
        /// <param name="prepend">Values to put before <paramref name="a"/> along the axis (a scalar is broadcast to
        ///     length 1), or null for none.</param>
        /// <param name="append">Values to put after <paramref name="a"/> along the axis, or null for none.</param>
        /// <returns><paramref name="a"/> itself for <paramref name="n"/> 0; otherwise a new array, the axis shorter
        ///     by <paramref name="n"/>, in the subtraction's dtype (bool for a bool input).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ArgumentException">A negative <paramref name="n"/> (<c>order must be non-negative but got
        ///     -1</c>).</exception>
        /// <exception cref="ValueError">A 0-D <paramref name="a"/> (<c>diff requires input that is at least one
        ///     dimensional</c>).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range.</exception>
        /// <exception cref="TypeError">Text with <paramref name="n"/> above 0: subtract has no string loop.</exception>
        public abstract NDArray Diff(NDArray a, int n = 1, int axis = -1, object prepend = null, object append = null);

        /// <summary>
        ///     NumPy's <c>np.digitize(x, bins, right=False)</c>: the index of the bin each value of <paramref name="x"/>
        ///     falls in, <c>searchsorted</c> on the bins (reversed for decreasing bins) with the side <paramref name="right"/>
        ///     selects.
        /// </summary>
        /// <param name="x">The values to bin, any shape (not null).</param>
        /// <param name="bins">The 1-D monotonic bin edges (not null).</param>
        /// <param name="right">Whether an interval includes its right edge instead of its left one.</param>
        /// <returns>The int64 bin indices, of <paramref name="x"/>'s shape.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> or <paramref name="bins"/> is null.</exception>
        /// <exception cref="IncorrectTypeException">A complex <paramref name="x"/> (<c>x may not be complex</c>,
        ///     checked first).</exception>
        /// <exception cref="ValueError"><paramref name="bins"/> not 1-D, or not monotonic (<c>bins must be
        ///     monotonically increasing or decreasing</c>).</exception>
        /// <exception cref="InvalidCastException"><paramref name="bins"/> that do not cast 'safely' to float64 (NumPy's
        ///     TypeError text).</exception>
        /// <exception cref="TypeError">A StringDType <paramref name="x"/> against numeric bins: Python cannot order the
        ///     pair.</exception>
        public abstract NDArray Digitize(NDArray x, NDArray bins, bool right = false);

        /// <summary>
        ///     NumPy's <c>np.unwrap(p, discont=None, axis=-1, *, period=6.283185307179586)</c>: along the axis, every jump
        ///     between neighbours larger than <c>max(discont, period / 2)</c> is corrected by multiples of the period
        ///     (the cumulative correction added to <c>p[1:]</c>).
        /// </summary>
        /// <param name="p">The input, at least 1-D, not complex (not null).</param>
        /// <param name="discont">The largest jump left alone, or null for <c>period / 2</c> (a smaller one acts as
        ///     <c>period / 2</c>).</param>
        /// <param name="axis">The axis to unwrap along (negative counts from the end).</param>
        /// <param name="period">NumPy's period, a Python float or int: a boxed <see cref="double"/> (or null for
        ///     <c>2 pi</c>) takes the float path, a boxed <see cref="long"/> or <see cref="int"/> NumPy's integer path for
        ///     an integer or bool input.</param>
        /// <returns>A new array of <paramref name="p"/>'s shape: float64 (or the input's float dtype) on the float path,
        ///     the integer dtype on the integer path.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="p"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="period"/> is neither null, a double, a long nor an
        ///     int.</exception>
        /// <exception cref="ValueError">A 0-D <paramref name="p"/>.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> out of range.</exception>
        /// <exception cref="TypeError">A complex <paramref name="p"/>.</exception>
        /// <exception cref="OverflowException">On the integer path, a bound or the period that does not fit the input
        ///     dtype (NumPy's OverflowError).</exception>
        public abstract NDArray Unwrap(NDArray p, double? discont = null, int axis = -1, object period = null);

        /// <summary>
        ///     NumPy's <c>np.interp(x, xp, fp, left=None, right=None, period=None)</c>: the piecewise-linear interpolant
        ///     through the points (xp, fp), evaluated at x. NumPy picks the loop from fp alone (complex128 for a complex fp,
        ///     float64 otherwise); with a period, x and xp are taken modulo it and the fill values are ignored.
        /// </summary>
        /// <param name="x">The coordinates to evaluate at, any shape.</param>
        /// <param name="xp">The sample points, 1-D, increasing unless a period is given (not checked: NumPy's loop answers
        ///     other points deterministically).</param>
        /// <param name="fp">The sample values, 1-D, as many as xp.</param>
        /// <param name="left">The value below <c>xp[0]</c>, NumPy's Python object: null (None, meaning <c>fp[0]</c>), a
        ///     double (a float) or a Complex (a complex, which the real loop refuses).</param>
        /// <param name="right">The value above <c>xp[-1]</c>, as <paramref name="left"/> (null means <c>fp[-1]</c>).</param>
        /// <param name="period">The period of the x-coordinates, or null; it must not be zero.</param>
        /// <returns>A new array of x's shape (0-d for a 0-d x): complex128 for a complex fp, float64 otherwise.</returns>
        /// <exception cref="ArgumentException"><paramref name="left"/> or <paramref name="right"/> is neither null, a
        ///     double nor a Complex.</exception>
        /// <exception cref="NullReferenceException">An operand is null.</exception>
        /// <exception cref="ValueError">fp or xp not 1-D (<c>object too deep for desired array</c> / <c>object of too
        ///     small depth for desired array</c>), empty sample points, lengths that differ, a zero period; with a period,
        ///     points that are not 1-D (<c>Data points must be 1-D sequences</c>) or text that does not parse as a
        ///     float.</exception>
        /// <exception cref="InvalidCastException">Without a period, an operand that does not cast 'safely' to the loop's
        ///     dtype (NumPy's TypeError text).</exception>
        /// <exception cref="TypeError">A real fp, no period and a complex fill value (<c>must be real number, not
        ///     complex</c>), raised after the operand and length checks.</exception>
        public abstract NDArray Interp(NDArray x, NDArray xp, NDArray fp, object left = null, object right = null,
            double? period = null);

        /// <summary>
        ///     NumPy's <c>np.sort_complex(a)</c>: a sorted copy along the last axis (<c>b = array(a, copy=True);
        ///     b.sort()</c>), in the input's own dtype, then converted to complex (<c>astype('D')</c>) unless it is.
        /// </summary>
        /// <param name="a">The input (not null).</param>
        /// <returns>A new complex128 array of <paramref name="a"/>'s shape, sorted along its last axis (complex values
        ///     lexicographically: real part, then imaginary part).</returns>
        /// <exception cref="AxisError">A 0-d array (<c>axis -1 is out of bounds for array of dimension 0</c>, which
        ///     NumPy's <c>b.sort()</c> raises).</exception>
        /// <exception cref="ValueError">Text that is not a complex number (<c>complex() arg is a malformed
        ///     string</c>).</exception>
        /// <remarks>The sort runs in the original dtype, so an int64 past 2^53 orders by its exact value, not by its
        ///     double image; only the sorted result is widened.</remarks>
        public abstract NDArray SortComplex(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.trim_zeros(filt, trim='fb', axis=None)</c>: <paramref name="filt"/> cut to the bounding box of
        ///     its nonzero elements along the trimmed axes - the leading (<c>'f'</c>) and / or trailing (<c>'b'</c>)
        ///     hyperplanes that hold only zeros removed. The values decide which view comes back, so it is a computation
        ///     (the plan's borderline decision), though nothing is computed into a new buffer.
        /// </summary>
        /// <param name="filt">
        ///     The input, a plain array: <c>np.trim_zeros</c> decides the <c>numpy.matrix</c> and
        ///     <c>numpy.char.chararray</c> classes before it calls this member. A text array is trimmed by its truth values
        ///     (an empty string is a zero), as NumPy's <c>argwhere</c> reads them through the dtype's <c>nonzero</c>.
        /// </param>
        /// <param name="trim">
        ///     The sides to trim: <c>"f"</c> (front), <c>"b"</c> (back), <c>"fb"</c> or <c>"bf"</c> (both), in any case;
        ///     null reads as the empty string, which is refused.
        /// </param>
        /// <param name="axis">The axes to trim (a negative axis counts from the end), or null for every axis; an empty
        ///     array trims nothing.</param>
        /// <returns>
        ///     A view of <paramref name="filt"/> with the zero hyperplanes cut from the sides <paramref name="trim"/> names
        ///     (an axis without a nonzero element is cut to length 0, whatever <paramref name="trim"/> says), or
        ///     <paramref name="filt"/> itself when no axis is trimmed (an empty <paramref name="axis"/>, or a 0-d
        ///     <paramref name="filt"/> with <paramref name="axis"/> null). The dtype and the number of dimensions are
        ///     <paramref name="filt"/>'s.
        /// </returns>
        /// <exception cref="ValueError"><paramref name="trim"/> holds another spelling (<c>unexpected character(s) in
        ///     `trim`: 'fx'</c>, checked first), or an axis is repeated (<c>repeated axis in `axis` argument</c>).</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="filt"/> (<c>axis: axis 3 is out of
        ///     bounds for array of dimension 2</c>), checked for every axis before the repeat.</exception>
        /// <exception cref="NullReferenceException"><paramref name="filt"/> is null (read after the trim check).</exception>
        /// <remarks>
        ///     NumPy's <c>_arg_trim_zeros</c> takes the edges from <c>argwhere(filt)</c>'s minimum and maximum. The
        ///     bounding box is separable, so DefaultEngine takes each trimmed axis' first and last index from
        ///     <c>nonzero(any(filt, other axes))</c> instead: the same view, without every nonzero coordinate.
        /// </remarks>
        public abstract NDArray TrimZeros(NDArray filt, string trim = "fb", int[] axis = null);
    }
}
