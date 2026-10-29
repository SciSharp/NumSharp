// The functions of numpy/lib/_type_check_impl.py that compute: nan_to_num, isreal, iscomplex and real_if_close (the
// np-layer dtype plan's phase P5). NumPy writes them in Python; here each is an engine member that DefaultEngine
// implements with NumPy's statements (named as Isin is: NumPy's name, its first letter a capital). The dtype routines of the same NumPy file (iscomplexobj, isrealobj,
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

        /// <summary>
        ///     NumPy's <c>np.isreal(x)</c>, <c>imag(x) == 0</c>: True where an element's imaginary part is zero (a -0.0
        ///     too); every element of a dtype that is not complex is real, and no element of text is (NumPy's
        ///     <c>'' == 0</c> has no loop and answers False).
        /// </summary>
        /// <param name="a">The input.</param>
        /// <returns>A new bool array of <paramref name="a"/>'s shape (C order whatever its layout).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="a"/> is null.</exception>
        public abstract NDArray Isreal(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.iscomplex(x)</c>: for a complex input <c>imag(x) != 0</c> (True where an element's imaginary
        ///     part is not zero: NaN and inf count, -0.0 does not), for any other dtype all False.
        /// </summary>
        /// <param name="a">The input.</param>
        /// <returns>A bool array of <paramref name="a"/>'s shape: for a dtype that is not complex, NumPy's
        ///     <c>zeros(shape, bool)[()]</c>, a view of a fresh buffer (OWNDATA false; the 0-d array itself for a 0-d
        ///     input).</returns>
        /// <exception cref="System.ArgumentNullException"><paramref name="a"/> is null.</exception>
        public abstract NDArray Iscomplex(NDArray a);

        /// <summary>
        ///     NumPy's <c>np.real_if_close(a, tol=100)</c>: a complex input whose every imaginary part is strictly within
        ///     <paramref name="tol"/> of zero becomes its real part; anything else is returned as it is.
        /// </summary>
        /// <param name="a">The input.</param>
        /// <param name="tol">Above 1, a multiple of the complex dtype's machine epsilon (float64's, 2^-52); at most 1,
        ///     an absolute bound. At 0 or below nothing collapses: a magnitude is never strictly below it.</param>
        /// <returns>A float64 view of the real parts (sharing <paramref name="a"/>'s memory, as NumPy's
        ///     <c>a.real</c>) when every imaginary part is within the bound (an empty complex array always is);
        ///     otherwise <paramref name="a"/> itself.</returns>
        /// <exception cref="System.NullReferenceException"><paramref name="a"/> is null.</exception>
        /// <remarks>DefaultEngine decides in one early-exit pass over the imaginary parts (NaN and inf never
        ///     collapse).</remarks>
        public abstract NDArray RealIfClose(NDArray a, double tol = 100);
    }
}
