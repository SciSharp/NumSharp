// The functions of numpy/lib/_scimath_impl.py (numpy.emath, a.k.a. numpy.lib.scimath): NumPy's automatic-domain siblings
// of sqrt, log, log10, log2, power, arccos, arcsin and arctanh. NumPy writes them in Python; here each is an engine
// member, prefixed Emath because the ufunc members already carry the plain names (TensorEngine.Umath.cs), and
// DefaultEngine implements it with NumPy's statements.

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.emath.sqrt(x)</c>: <c>_fix_real_lt_zero(x)</c>, then <c>sqrt</c>. A real input with any element
        ///     below zero is converted to complex as a whole before the square root, so a negative element gets its
        ///     principal complex root where <see cref="Sqrt"/> returns NaN.
        /// </summary>
        /// <param name="x">The input, any dtype and layout (a C# scalar or array converts at the facade).</param>
        /// <returns>
        ///     The ufunc's result: complex128 when <paramref name="x"/> is complex or a real element is negative (NumPy
        ///     picks complex64 for a float32, int8, int16, uint8 or uint16 input; NumSharp has one complex type), else the
        ///     ordinary real result (integers and bool promote as <see cref="Sqrt"/> does). <c>-0.0</c> and NaN never
        ///     trigger the conversion. A 0-d input gives a 0-d array.
        /// </returns>
        /// <exception cref="TypeError">A text operand (S, U or StringDType): the domain scan's <c>x &lt; 0</c> has no loop
        ///     for text and a Python int, NumPy's <c>ufunc 'less' did not contain a loop ...</c>, even for an empty
        ///     array.</exception>
        /// <remarks>The scan reads every element of a real operand once (an early exit at the first negative one); a
        ///     complex operand skips it, because NumPy's conversion of a complex array is value-identical.</remarks>
        public abstract NDArray EmathSqrt(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.emath.log(x)</c>: <c>_fix_real_lt_zero(x)</c>, then <c>log</c>. A negative real element makes
        ///     the whole result complex (<c>log(-e) == 1 + pi i</c>) where <see cref="Log"/> returns NaN.
        /// </summary>
        /// <param name="x">The input, any dtype and layout.</param>
        /// <returns>The natural logarithm: complex128 when <paramref name="x"/> is complex or a real element is negative
        ///     (complex64 in NumPy for the narrow dtypes listed on <see cref="EmathSqrt"/>), else real; <c>log(0)</c> is
        ///     <c>-inf</c> and <c>log(inf)</c> <c>inf</c>, both real.</returns>
        /// <exception cref="TypeError">A text operand, with <see cref="EmathSqrt"/>'s text.</exception>
        public abstract NDArray EmathLog(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.emath.log10(x)</c>: <c>_fix_real_lt_zero(x)</c>, then <c>log10</c>; a negative real element
        ///     makes the whole result complex.
        /// </summary>
        /// <param name="x">The input, any dtype and layout.</param>
        /// <returns>The base-10 logarithm, complex under <see cref="EmathLog"/>'s conditions, else real.</returns>
        /// <exception cref="TypeError">A text operand, with <see cref="EmathSqrt"/>'s text.</exception>
        public abstract NDArray EmathLog10(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.emath.log2(x)</c>: <c>_fix_real_lt_zero(x)</c>, then <c>log2</c>; a negative real element
        ///     makes the whole result complex.
        /// </summary>
        /// <param name="x">The input, any dtype and layout.</param>
        /// <returns>The base-2 logarithm, complex under <see cref="EmathLog"/>'s conditions, else real.</returns>
        /// <exception cref="TypeError">A text operand, with <see cref="EmathSqrt"/>'s text.</exception>
        public abstract NDArray EmathLog2(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.emath.logn(n, x)</c>: <c>_fix_real_lt_zero(x)</c>, <c>_fix_real_lt_zero(n)</c>, then
        ///     <c>log(x) / log(n)</c>. Each operand is converted on its own, so a negative base alone (<c>logn(-2, 8)</c>)
        ///     makes <c>log(n)</c> and with it the quotient complex.
        /// </summary>
        /// <param name="n">The base(s), any dtype and layout. NumPy's argument order: the base comes first.</param>
        /// <param name="x">The value(s), broadcast against <paramref name="n"/> by the division.</param>
        /// <returns>The quotient of the two logarithms, in the division's promoted dtype. A 0-d log result is read as a
        ///     weak scalar by the division (the ufuncs' rule on this branch), where NumPy reads it as strong: a 0-d
        ///     <paramref name="x"/> beside a float16 <paramref name="n"/> gives float16 here and float64 in
        ///     NumPy.</returns>
        /// <exception cref="TypeError">A text operand, with <see cref="EmathSqrt"/>'s text; <paramref name="x"/> is
        ///     scanned first, as in NumPy.</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together (NumPy's ValueError
        ///     <c>operands could not be broadcast together with shapes ...</c>), raised by the division after both
        ///     logarithms.</exception>
        public abstract NDArray EmathLogn(NDArray n, NDArray x);

        /// <summary>
        ///     NumPy's <c>np.emath.power(x, p)</c>: <c>_fix_real_lt_zero(x)</c>, <c>_fix_int_lt_zero(p)</c>, then
        ///     <c>power(x, p)</c>. A negative real base makes the computation complex; a negative real exponent turns
        ///     <paramref name="p"/> into a float (<c>p * 1.0</c>), so an integer base with a negative integer exponent
        ///     does not hit <see cref="Power"/>'s "Integers to negative integer powers" error.
        /// </summary>
        /// <param name="x">The base(s), any dtype and layout.</param>
        /// <param name="p">The exponent(s), broadcast against <paramref name="x"/> by the power ufunc.</param>
        /// <returns>The power ufunc's result in its promoted dtype (complex128 after a negative real base). A 0-d operand
        ///     is read as a weak scalar by the power ufunc (the ufuncs' rule on this branch), where NumPy reads a 0-d array
        ///     as strong: a uint16 base with a 0-d int64 exponent gives uint16 here and int64 in NumPy.</returns>
        /// <exception cref="TypeError">A text operand, with <see cref="EmathSqrt"/>'s text; <paramref name="x"/> is
        ///     scanned first.</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together, raised by the power
        ///     ufunc.</exception>
        /// <remarks>A negative base with a non-integer exponent takes the complex power, which is <c>allclose</c> to
        ///     NumPy's but not bit-exact (the F5 complex-power divergence); integer exponents are exact.</remarks>
        public abstract NDArray EmathPower(NDArray x, NDArray p);

        /// <summary>
        ///     NumPy's <c>np.emath.arccos(x)</c>: <c>_fix_real_abs_gt_1(x)</c>, then <c>arccos</c>. A real element with
        ///     <c>|x| &gt; 1</c> (an infinity included) makes the whole result complex where <see cref="ACos"/> returns
        ///     NaN.
        /// </summary>
        /// <param name="x">The input, any dtype and layout.</param>
        /// <returns>The inverse cosine: real in <c>[0, pi]</c> while every real element has <c>|x| &lt;= 1</c> (NaN does
        ///     not trigger the conversion), complex128 otherwise or for a complex input.</returns>
        /// <exception cref="TypeError">A text operand: the scan's <c>abs(x)</c> has no text loop, NumPy's <c>ufunc
        ///     'absolute' did not contain a loop with signature matching types ... -&gt; None</c>, even for an empty
        ///     array.</exception>
        public abstract NDArray EmathArccos(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.emath.arcsin(x)</c>: <c>_fix_real_abs_gt_1(x)</c>, then <c>arcsin</c>; a real element with
        ///     <c>|x| &gt; 1</c> makes the whole result complex.
        /// </summary>
        /// <param name="x">The input, any dtype and layout.</param>
        /// <returns>The inverse sine: real in <c>[-pi/2, pi/2]</c> while every real element has <c>|x| &lt;= 1</c>,
        ///     complex128 otherwise or for a complex input.</returns>
        /// <exception cref="TypeError">A text operand, with <see cref="EmathArccos"/>'s text.</exception>
        public abstract NDArray EmathArcsin(NDArray x);

        /// <summary>
        ///     NumPy's <c>np.emath.arctanh(x)</c>: <c>_fix_real_abs_gt_1(x)</c>, then <c>arctanh</c>; a real element with
        ///     <c>|x| &gt; 1</c> makes the whole result complex. <c>|x| == 1</c> stays real and gives <c>±inf</c>.
        /// </summary>
        /// <param name="x">The input, any dtype and layout.</param>
        /// <returns>The inverse hyperbolic tangent: real while every real element has <c>|x| &lt;= 1</c>, complex128
        ///     otherwise or for a complex input.</returns>
        /// <exception cref="TypeError">A text operand, with <see cref="EmathArccos"/>'s text.</exception>
        public abstract NDArray EmathArctanh(NDArray x);
    }
}
