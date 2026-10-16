using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The text branch shared by every <c>np.quantile</c> / <c>np.percentile</c> overload: NumPy's checks after q's
        ///     (the axes, the method, the interpolating methods' refusals — <see cref="Backends.Strings.NDStringUfuncs.QuantileAnswer"/>),
        ///     then its take path over the text (<see cref="QuantileText.Compute"/>).
        /// </summary>
        /// <param name="op">The NumPy function name (<c>quantile</c> or <c>percentile</c>), for the refusals' texts.</param>
        /// <param name="a">The text data (<c>S</c> / <c>U</c>), or a chararray operand of any dtype (see
        ///     <paramref name="chararray"/>).</param>
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
        ///     (see <see cref="QuantileText.Compute"/>).</remarks>
        private static NDArray QuantileOfText(string op, NDArray a, double[] q, long[] qShape, bool qIsInteger, int[] axes,
            NDArray @out, string method, bool keepdims, bool chararray)
        {
            // _ureduce normalizes the axes before anything else reads them (AxisError, or a repeated axis's ValueError);
            // QuantileText.Compute then runs the rest in NumPy's order (keepdims' out index, q's rank, the method) — and for
            // a chararray operand stops at NumPy's arr.partition(..., axis=0), the STRING partition's TypeError.
            Backends.Strings.NDStringUfuncs.ValidateReduceAxes(op, a.ndim, axes);
            return QuantileText.Compute(op, a, q, qShape, qIsInteger, axes, @out, method, keepdims, chararray);
        }

        /// <summary>Whether a q array holds integers or bools: NumPy's <c>(n - 1) * q</c> is then an integer index.</summary>
        /// <param name="q">The q array.</param>
        /// <returns>True for a bool or integer dtype.</returns>
        private static bool IsIntegerQuantile(NDArray q)
            => q.typecode is NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16
                or NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64;
    }
}
