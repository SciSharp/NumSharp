using NumSharp.Backends;

namespace NumSharp
{
    public partial class NDArray
    {
        /// <summary>
        ///     The real part of the array (NumPy's <c>ndarray.real</c>) — a read/write accessor.
        ///
        ///     <para>
        ///     GET: for a COMPLEX array, a float64 VIEW onto the real lane that SHARES memory and is
        ///     writeable (so <c>z.real[i] = x</c> writes through to <c>z[i]</c>'s real part); for a real /
        ///     integer / boolean array, the array itself (the real part of a real number is the number),
        ///     dtype preserved. Delegates to <see cref="np.real(NDArray)"/>.
        ///     </para>
        ///     <para>
        ///     SET: copies <c>value</c> into the real part (broadcasting to its shape) with NumPy's
        ///     <c>PyArray_CopyInto</c> semantics — UNSAFE casting, so a float value assigned to an integer
        ///     array TRUNCATES (<c>a.real = 3.9</c> stores 3), an out-of-range integer WRAPS
        ///     (<c>int8.real = 300</c> stores 44), and a complex value keeps only its real part. For a real
        ///     array this overwrites the whole array (<c>a.real = 5</c>); for a complex array it overwrites
        ///     only the real lane, leaving the imaginary parts untouched. Writing to a read-only array
        ///     raises the standard read-only error.
        ///     </para>
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.real.html</remarks>
        public NDArray real
        {
            get
            {
                using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
                return np.real(this);
            }
            // PyArray_CopyInto uses NPY_UNSAFE_CASTING; np.real(this) is the writeable target (the real
            // lane view for complex, `this` for a real array) whose writeability guard reproduces NumPy's
            // "assignment destination is read-only". The scope reclaims the lane view: it holds a counted
            // reference on this array's buffer (a byte-reinterpreting alias forwards its ARC to the owner),
            // so leaving it to the finalizer would keep the buffer out of the pool after this array is
            // disposed. `this` (a real array's target) was not created in the scope and is never touched.
            [NDScoped]
            set => np.copyto(np.real(this), value, casting: "unsafe");
        }

        /// <summary>
        ///     The imaginary part of the array (NumPy's <c>ndarray.imag</c>) — a read/write accessor.
        ///
        ///     <para>
        ///     GET: for a COMPLEX array, a float64 VIEW onto the imaginary lane that SHARES memory and is
        ///     writeable; for a real / integer / boolean array, a fresh READ-ONLY all-zeros array of the
        ///     same shape and dtype (the imaginary part of a real number is zero). Delegates to
        ///     <see cref="np.imag(NDArray)"/>.
        ///     </para>
        ///     <para>
        ///     SET: for a COMPLEX array, copies <c>value</c> into the imaginary lane (same UNSAFE-cast,
        ///     broadcasting <c>PyArray_CopyInto</c> semantics as <see cref="real"/>). For a real array
        ///     there is no imaginary lane to write, so it raises <see cref="TypeError"/>
        ///     ("array does not have imaginary part to set"), matching NumPy.
        ///     </para>
        /// </summary>
        /// <exception cref="TypeError">Assigned to a non-complex array (NumPy raises the same message).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.imag.html</remarks>
        public NDArray imag
        {
            get
            {
                using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
                return np.imag(this);
            }
            // NumPy's array_imag_set (ImagSet, NDArray.TypeCheck.cs): the dtype check and the copy into the
            // imaginary lane; its scope reclaims the lane view the copy targets (a counted reference on this
            // array's buffer, see the real setter), not a finalizer's job.
            set
            {
                using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
                ImagSet(this, value);
            }
        }

        /// <summary>
        ///     Return the complex conjugate, element-wise (NumPy's <c>ndarray.conjugate</c> method — the port
        ///     of <c>PyArray_Conjugate</c>, which is NOT the <c>np.conjugate</c> ufunc). For a COMPLEX array
        ///     the imaginary sign is flipped. For a real / integer / <b>boolean</b> array the values are
        ///     already their own conjugate, so — unlike the <see cref="np.conjugate(NDArray, NDArray, NDArray, DType)"/>
        ///     FUNCTION, which has no bool loop and promotes bool→int8 — the method PRESERVES the dtype:
        ///     with no <paramref name="out"/> it returns THIS array itself (NumPy returns <c>self</c>), and
        ///     with an <paramref name="out"/> it copies the values there under NumPy's default (<c>same_kind</c>)
        ///     assignment casting.
        /// </summary>
        /// <param name="out">Optional destination. When given it receives the result and is returned.</param>
        /// <returns>The conjugate: a new array, <paramref name="out"/> itself, or — for a non-complex array without
        ///     <paramref name="out"/> — this array itself (not a copy: a caller must not dispose it as its own).</returns>
        /// <exception cref="TypeError">A string array: "cannot conjugate non-numeric dtype" (NumPy's text).</exception>
        /// <exception cref="System.InvalidCastException"><paramref name="out"/>'s dtype is not reachable under same_kind
        ///     casting (np.copyto's, or the ufunc's out cast for a complex array).</exception>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only or its shape does not take the result.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conjugate.html</remarks>
        public NDArray conjugate(NDArray @out = null)
        {
            using var plain = NDMatrixBoundary.PlainFor(this);   // the member keeps the plain-array meaning for a matrix object
            // NumPy's PyArray_Conjugate: the statements are the array object's (NDArray.Multiarray.cs)
            return conjugate(this, @out);
        }

        /// <summary>
        ///     Alias of <see cref="conjugate(NDArray)"/> — return the complex conjugate, element-wise
        ///     (NumPy: <c>ndarray.conj</c> is <c>ndarray.conjugate</c>).
        /// </summary>
        /// <param name="out">Optional destination. When given it receives the result and is returned.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.conj.html</remarks>
        public NDArray conj(NDArray @out = null) => conjugate(@out);
    }
}
