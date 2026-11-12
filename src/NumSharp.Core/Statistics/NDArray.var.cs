using System;
using System.Diagnostics.CodeAnalysis;
using NumSharp.Backends;

namespace NumSharp
{
    [SuppressMessage("ReSharper", "ParameterHidesMember")]
    public partial class NDArray
    {
        /// <summary>
        ///     NumPy's <c>ndarray.var(axis=None, dtype=None, out=None, ddof=0, *, keepdims=&lt;no value&gt;,
        ///     where=&lt;no value&gt;, mean=&lt;no value&gt;)</c> (<c>_methods._var</c>): the variance over
        ///     <paramref name="axis"/>. This is the only overload: NumPy's whole signature (np-layer dtype plan P10, D10); the
        ///     parameters behave as <see cref="np.var"/>'s. There is no <c>correction=</c>: NumPy 2.4.2's method refuses it
        ///     (<c>_var() got an unexpected keyword argument 'correction'</c>); <see cref="np.var"/> takes it.
        /// </summary>
        /// <param name="axis">The axis to compute along (negative counts from the end); <c>null</c> uses every element.
        ///     A tuple (<c>(0, 2)</c>) reduces every axis it names together, in one pass (NumPy's tuple axis; the empty
        ///     tuple reduces none, and a tuple item gets no 0-d allowance).</param>
        /// <param name="dtype">The loop of both sums, or <c>null</c> for NumPy's default (float64 for bool / integer input).</param>
        /// <param name="out">The array the variance is written into and returned (NumPy hands it to the last sum only).
        ///     Footgun: the THIRD positional parameter, as in NumPy; write <c>ddof:</c> and <c>keepdims:</c> by name.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>max(N - ddof, 0)</c>.</param>
        /// <param name="keepdims">Keep the reduced axes as length-1 dimensions.</param>
        /// <param name="where">A bool mask of the elements each output takes, or <c>null</c>.</param>
        /// <param name="mean">The mean to subtract as given (see <see cref="np.var"/>), or <c>null</c> to compute it.</param>
        /// <returns><paramref name="out"/> itself when given; else a fresh array — 0-d (NumPy's read-only scalar) when the
        ///     result has no dimensions.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds.</exception>
        /// <exception cref="ValueError">A read-only or misshapen <paramref name="out"/>. Also a tuple that names an
        ///     axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/>, <paramref name="where"/> or
        ///     <paramref name="mean"/> does not broadcast.</exception>
        /// <exception cref="TypeError">A boolean subtraction (a bool input in a bool loop, or a bool mean of a bool input); a
        ///     C# scalar passed as <paramref name="out"/>; NumPy's text refusals.</exception>
        /// <remarks>The member keeps the plain-array meaning for a numpy.matrix object reached through an
        ///     <see cref="NDArray"/> reference.</remarks>
        [PlainReceiver]
        public NDArray var(NDAxis? axis = null, DType dtype = null, NDArray @out = null, int ddof = 0, bool keepdims = false,
            NDArray @where = null, NDArray mean = null)
        {
            return TensorEngine.ReduceVar(this, axis, keepdims, ddof, dtype, @out, @where, mean);
        }

        /// <summary>
        ///     <see cref="var(NDAxis?, DType, NDArray, int, bool, NDArray, NDArray)"/> with the axis as one <c>int</c>:
        ///     the overload a C# integer binds (its identity conversion beats NDAxis' implicit one), forwarding to the
        ///     NDAxis form, so every answer, error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="var(NDAxis?, DType, NDArray, int, bool, NDArray, NDArray)"/>
        public NDArray var(int axis, DType dtype = null, NDArray @out = null, int ddof = 0, bool keepdims = false, NDArray @where = null, NDArray mean = null)
            => var((NDAxis?)axis, dtype, @out, ddof, keepdims, @where, mean);

        /// <summary>
        ///     <see cref="var(NDAxis?, DType, NDArray, int, bool, NDArray, NDArray)"/> with the axes as an
        ///     <c>int[]</c>: the overload an <c>int[]</c> binds, and a bare <c>null</c> too (null converts to
        ///     <c>int[]</c> better than to NDAxis), forwarding to the NDAxis form: a null array is NumPy's None, any
        ///     other array NumPy's tuple (<c>new int[0]</c> is <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="var(NDAxis?, DType, NDArray, int, bool, NDArray, NDArray)"/>
        public NDArray var(int[] axis, DType dtype = null, NDArray @out = null, int ddof = 0, bool keepdims = false, NDArray @where = null, NDArray mean = null)
            => var((NDAxis?)axis, dtype, @out, ddof, keepdims, @where, mean);
    }
}
