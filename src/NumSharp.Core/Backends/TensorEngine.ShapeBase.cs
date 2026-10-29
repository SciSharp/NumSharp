// The functions of numpy/lib/_shape_base_impl.py that compute: kron. The module's other functions (np.take_along_axis,
// np.put_along_axis, np.apply_along_axis, the splits and stacks, np.tile, np.expand_dims, ...) are the array object's
// (Appendix D of docs/plans/np-layer-dtype-logic.md), not engine members.

using System;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.kron(a, b)</c>, the whole call: the Kronecker product, a composite array of blocks of
        ///     <paramref name="b"/> scaled by the elements of <paramref name="a"/>. For <c>a.shape = (r0, ..., rN)</c> and
        ///     <c>b.shape = (s0, ..., sN)</c> the result has shape <c>(r0*s0, ..., rN*sN)</c> and
        ///     <c>kron(a, b)[k0, ..., kN] = a[i0, ..., iN] * b[j0, ..., jN]</c> with <c>kt = it*st + jt</c>.
        /// </summary>
        /// <param name="a">The first operand.</param>
        /// <param name="b">The second operand. The operand of smaller rank is read as if prepended with length-1 axes
        ///     (NumPy's <c>ndmin</c>).</param>
        /// <returns>A fresh, writeable, C-contiguous array in np.multiply's result dtype (NEP 50); with a 0-d operand the
        ///     element-wise product, the degenerate Kronecker product.</returns>
        /// <exception cref="NullReferenceException"><paramref name="a"/> or <paramref name="b"/> is null.</exception>
        /// <exception cref="TypeError">Operands np.multiply refuses (text has no multiply loop: NumPy's ufunc
        ///     refusal).</exception>
        /// <remarks>New in 0.71.0 (the np-layer dtype plan): the public <c>np.kron</c> applies the <c>numpy.matrix</c> and
        ///     <c>numpy.char.chararray</c> rules and calls this.</remarks>
        public abstract NDArray Kron(NDArray a, NDArray b);
    }
}
