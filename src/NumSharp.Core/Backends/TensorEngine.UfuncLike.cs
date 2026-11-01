// The functions of numpy/lib/_ufunclike_impl.py that NumSharp's engine implements: fix, isposinf and isneginf.

using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Generic;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.fix(x, out=None)</c>: rounds toward zero, element-wise. In NumPy 2.4.2 its body is the one
        ///     statement <c>return nx.trunc(x, out=out)</c>, so the dtype is kept (integers and bool unchanged, floats
        ///     truncated), NaN and the infinities pass through, the sign of -0.0 survives, and every error is trunc's.
        /// </summary>
        /// <param name="x">The input (not null).</param>
        /// <param name="out">An array to write the result into and return, or null: it joins the ufunc's broadcast and takes
        ///     a same_kind cast from the loop dtype.</param>
        /// <returns>A new array of <paramref name="x"/>'s shape and dtype, or <paramref name="out"/>.</returns>
        /// <exception cref="TypeError">An input trunc has no loop for: text (NumPy's no-loop texts), or complex (<c>ufunc
        ///     'trunc' not supported for the input types</c>).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/> (<c>output array is read-only</c>).</exception>
        /// <exception cref="ArgumentException">An <paramref name="out"/> the result does not reach under same_kind
        ///     (<c>Cannot cast ufunc 'trunc' output from ...</c>), or one it does not broadcast into.</exception>
        /// <exception cref="NullReferenceException"><paramref name="x"/> is null.</exception>
        /// <remarks>
        ///     DefaultEngine's member is that statement over <see cref="Truncate"/>, so trunc's error texts (they name
        ///     <c>trunc</c>, as NumPy's fix leaks them) are fix's. <c>np.fix</c> calls it after its matrix rule; an override
        ///     replaces it for every array whose engine it is. Unlike a ufunc it has no <c>where=</c> / <c>dtype=</c>.
        /// </remarks>
        public abstract NDArray Fix(NDArray x, NDArray @out = null);

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
