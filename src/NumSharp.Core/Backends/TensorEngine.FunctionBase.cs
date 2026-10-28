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
    }
}
