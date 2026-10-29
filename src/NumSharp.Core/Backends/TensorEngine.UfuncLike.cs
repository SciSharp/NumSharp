// The functions of numpy/lib/_ufunclike_impl.py that NumSharp's engine implements: isposinf and isneginf.

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        // np.isposinf / np.isneginf (numpy/lib/_ufunclike_impl.py). Not ufuncs — no
        // where/dtype; the single settable side is out (positional in NumPy).

        /// <summary>
        ///     NumPy's <c>np.isposinf(x, out=None)</c>, <c>isinf(x) &amp; ~signbit(x)</c>: True where an element is +inf.
        ///     Integers and bool are never infinite.
        /// </summary>
        /// <param name="a">The input.</param>
        /// <param name="dtype">Unused by the NumPy function (it has no dtype=); NumSharp's facade passes bool.</param>
        /// <param name="out">An array to write the result into and return, or null.</param>
        /// <param name="where">Unused by the NumPy function (null).</param>
        /// <returns>A bool array of <paramref name="a"/>'s shape, or <paramref name="out"/>.</returns>
        /// <exception cref="TypeError">A complex input (<c>This operation is not supported for complex128 values because it
        ///     would be ambiguous.</c>, NumPy's signbit refusal, raised first), or text (isinf's refusal).</exception>
        public abstract NDArray IsPosInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);

        /// <summary>
        ///     NumPy's <c>np.isneginf(x, out=None)</c>, <c>isinf(x) &amp; signbit(x)</c>: True where an element is -inf.
        ///     Integers and bool are never infinite.
        /// </summary>
        /// <param name="a">The input.</param>
        /// <param name="dtype">Unused by the NumPy function (it has no dtype=); NumSharp's facade passes bool.</param>
        /// <param name="out">An array to write the result into and return, or null.</param>
        /// <param name="where">Unused by the NumPy function (null).</param>
        /// <returns>A bool array of <paramref name="a"/>'s shape, or <paramref name="out"/>.</returns>
        /// <exception cref="TypeError">A complex input (NumPy's signbit refusal, raised first), or text (isinf's
        ///     refusal).</exception>
        public abstract NDArray IsNegInf(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null);
    }
}
