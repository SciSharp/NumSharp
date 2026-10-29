// The functions of numpy/lib/_type_check_impl.py that compute: nan_to_num (and, as the np-layer dtype plan's phase P5
// moves them, iscomplex, isreal and real_if_close). NumPy writes them in Python; here each is an engine member that
// DefaultEngine implements with NumPy's statements. The dtype routines of the same NumPy file (iscomplexobj, isrealobj,
// common_type, mintypecode, typename) belong to the DType layer, and real / imag to the array object (Appendix D).

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None)</c>: NaN replaced with
        ///     <paramref name="nan"/>, +inf with <paramref name="posinf"/> and -inf with <paramref name="neginf"/>, in
        ///     the real and imaginary parts of a complex input alike. An input that is not inexact (bool, integers, char,
        ///     decimal, text) holds no NaN or inf and comes back as it is.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="copy">Whether to work on a copy (true) or in place on <paramref name="x"/> (false: the result is
        ///     <paramref name="x"/> itself, its memory written).</param>
        /// <param name="nan">The fill for NaN: null for NumPy's 0.0, a scalar (a C# number, bool or float16), or an
        ///     array_like (an NDArray or a C# array) broadcast against <paramref name="x"/>; cast to x's real dtype.</param>
        /// <param name="posinf">The fill for +inf, as <paramref name="nan"/>; null for the largest finite value of x's
        ///     real dtype (float16, float32, or float64 for float64 and complex128).</param>
        /// <param name="neginf">The fill for -inf, as <paramref name="nan"/>; null for the smallest finite value.</param>
        /// <returns>The array with its non-finite values replaced: a copy, or <paramref name="x"/> when
        ///     <paramref name="copy"/> is false.</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="x"/> is null.</exception>
        /// <exception cref="ValueError">An array fill that does not broadcast into <paramref name="x"/> (np.copyto's
        ///     text).</exception>
        /// <exception cref="System.FormatException">A text fill on an inexact input (.NET's number parse; NumPy raises
        ///     ValueError).</exception>
        /// <exception cref="System.InvalidCastException">A fill of a type that does not convert to a number.</exception>
        /// <remarks>DefaultEngine fills scalar fills into a contiguous target in one fused pass (the
        ///     <c>DirectILKernelGenerator.NanToNum</c> kernel); array fills and other layouts run NumPy's three
        ///     <c>copyto(where=)</c> statements.</remarks>
        public abstract NDArray NanToNum(NDArray x, bool copy = true, object nan = null, object posinf = null,
            object neginf = null);
    }
}
