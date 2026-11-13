using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return numbers spaced evenly on a log scale (a geometric progression) - NumPy's
        ///     <c>np.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)</c>. Each output sample is a constant
        ///     multiple of the previous - like <see cref="logspace(double,double,long,bool,double,DType,int)"/> but with the
        ///     endpoints specified directly rather than as exponents. Array ends give one progression per end pair (they
        ///     broadcast), the samples along <paramref name="axis"/>; complex ends give a logarithmic spiral.
        /// </summary>
        /// <param name="start">The starting value(s): a number (a C# number converts, read as NumPy reads a Python number),
        ///     a <see cref="System.Numerics.Complex"/>, or an array that broadcasts with <paramref name="stop"/>. No element
        ///     may be zero (a geometric sequence cannot include zero).</param>
        /// <param name="stop">
        ///     The final value(s) of the sequence, unless <paramref name="endpoint"/> is False. In that case <c>num + 1</c>
        ///     values are spaced over the interval in log-space, of which all but the last are returned. No element may be
        ///     zero.
        /// </param>
        /// <param name="num">Number of samples to generate. Default 50. Must be non-negative.</param>
        /// <param name="endpoint">If True (default) <paramref name="stop"/> is the last sample; otherwise it is excluded.</param>
        /// <param name="dtype">
        ///     The dtype of the output — one descriptor parameter, like NumPy's <c>dtype=</c> (Type/NPTypeCode/string/DType
        ///     convert implicitly). If null (default) the result has the computation dtype: float64 for real number ends,
        ///     complex128 for complex ones. The samples are computed in NumPy's <c>dt = result_type(start, stop, float(num),
        ///     zeros((), dtype))</c> and only then cast (an integer dtype TRUNCATES toward zero); requesting a complex dtype
        ///     makes the computation complex (the interior then differs from the real path's), and float32 array ends with a
        ///     float32 dtype compute in float32, as NumPy's do.
        /// </param>
        /// <param name="axis">The axis in the result that holds the samples (NumPy's trailing <c>moveaxis(result, 0,
        ///     axis)</c>): any axis in range for the result's rank, which is one more than the broadcast ends'. For number
        ///     ends only 0 and -1 exist.</param>
        /// <returns>The samples, shape <c>(num, *broadcast(start, stop).shape)</c> with the sample axis moved to
        ///     <paramref name="axis"/> (for number ends a 1-D array of <paramref name="num"/> samples).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="start"/> or <paramref name="stop"/> is null.</exception>
        /// <exception cref="ValueError">An end holds a zero (<c>Geometric sequence cannot include zero</c>, NumPy's first
        ///     check), or <paramref name="num"/> is negative.</exception>
        /// <exception cref="DTypePromotionError">A <c>StringDType</c> <paramref name="dtype"/> (no common DType with the
        ///     ends).</exception>
        /// <exception cref="TypeError">An <c>S</c> or <c>U</c> <paramref name="dtype"/> (<c>sign</c> has no string
        ///     loop).</exception>
        /// <exception cref="IncorrectShapeException">The ends do not broadcast.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for the result (NumPy's verbatim
        ///     <c>destination:</c> message).</exception>
        /// <remarks>
        ///     <para>https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html</para>
        ///     <para>API-parity batch 10: one signature over NumPy's array_like ends. The <c>double</c> and
        ///     <see cref="System.Numerics.Complex"/> overloads went: a number converts to a 0-d array, and number ends keep
        ///     their answers bit for bit (real) or within the complex-unary envelope (complex, as before). Computed by the
        ///     start operand's engine (<see cref="TensorEngine.Geomspace(NDArray, NDArray, long, bool, DType, int)"/>). A
        ///     <c>numpy.matrix</c> or masked operand computes on its plain data and gives a plain array (NumPy's own answer
        ///     for a matrix is a matrix-product error from <c>result *= out_sign</c>).</para>
        /// </remarks>
        public static NDArray geomspace(NDArray start, NDArray stop, long num = 50, bool endpoint = true, DType dtype = null, int axis = 0)
        {
            // C#'s convention for a missing operand, before an engine is chosen through it.
            if (start is null)
                throw new ArgumentNullException(nameof(start));
            if (stop is null)
                throw new ArgumentNullException(nameof(stop));
            return start.TensorEngine.Geomspace(start, stop, num, endpoint, dtype, axis);
        }
    }
}
