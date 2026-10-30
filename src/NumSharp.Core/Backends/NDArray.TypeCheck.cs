using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <remarks>
    ///     The array object's home for NumPy's <c>lib/_type_check_impl.py</c> (the np-layer dtype plan's Appendix D):
    ///     np.real and np.imag are <c>val.real</c> / <c>val.imag</c> there, so their statements are the ndarray
    ///     accessors' (NumPy's <c>array_real_get</c>, <c>array_imag_get</c> and <c>array_imag_set</c> in
    ///     <c>getset.c</c>). The facades (<c>Math/np.real.cs</c>, <c>Math/np.imag.cs</c>) keep the numpy.matrix boundary;
    ///     the members <see cref="real"/> and <see cref="imag"/> (<c>NDArray.ComplexAccessors.cs</c>) reach the same
    ///     statements.
    /// </remarks>
    public partial class NDArray
    {
        /// <summary>
        ///     The real part of <paramref name="val"/>, NumPy's <c>val.real</c> (<c>array_real_get</c>): a float64 view onto
        ///     the real lane of a complex array, the array itself for any other dtype.
        /// </summary>
        /// <param name="val">The array.</param>
        /// <returns>
        ///     For complex128, a new writeable float64 view that SHARES <paramref name="val"/>'s memory (a write through it
        ///     changes the real parts); otherwise <paramref name="val"/> itself, the same object (the real part of a real
        ///     number is the number), so a caller must not dispose it as its own.
        /// </returns>
        /// <remarks>
        ///     No numpy.matrix rule here: np.real's facade applies it (a complex matrix's real part is a matrix, a real
        ///     matrix comes back as itself). The lane view holds a counted reference on the complex buffer (a
        ///     byte-reinterpreting alias forwards its ARC to the owner), so a caller that drops it should dispose it or
        ///     create it inside a scope.
        /// </remarks>
        internal static NDArray RealPart(NDArray val)
        {
            // Complex: a strided float64 view onto the real lane (writes through, shares memory) — the
            // same strided complex->real lane read the FFT driver relies on, exposed as a view.
            if (val.typecode == NPTypeCode.Complex)
                return new NDArray(val.Storage.AliasComplexLane(imaginary: false));

            // Real / int / bool: NumPy's `val.real` IS `val` for a real array (same object, shared memory,
            // dtype preserved). Return it unchanged.
            return val;
        }

        /// <summary>
        ///     The imaginary part of <paramref name="val"/>, NumPy's <c>val.imag</c> (<c>array_imag_get</c>): a float64 view
        ///     onto the imaginary lane of a complex array, a fresh read-only array of zeros of <paramref name="val"/>'s shape
        ///     and dtype for any other dtype.
        /// </summary>
        /// <param name="val">The array.</param>
        /// <returns>
        ///     For complex128, a new writeable float64 view that SHARES <paramref name="val"/>'s memory; otherwise a new
        ///     owning, NON-writeable array of zeros (empty strings for a string dtype), F-ordered when
        ///     <paramref name="val"/> is F-contiguous and not C-contiguous (NumPy's <c>PyArray_ISFORTRAN</c>).
        /// </returns>
        /// <remarks>
        ///     No numpy.matrix rule here: np.imag's facade applies it. The zeros are allocated from the DESCRIPTOR, so a
        ///     text array's imaginary part is its own dtype (<c>np.imag(U3)</c> is <c>&lt;U3</c> of <c>''</c>).
        /// </remarks>
        internal static NDArray ImagPart(NDArray val)
        {
            // Complex: a strided float64 view onto the imaginary lane (writes through, shares memory).
            if (val.typecode == NPTypeCode.Complex)
                return new NDArray(val.Storage.AliasComplexLane(imaginary: true));

            // Real / int / bool: NumPy returns a READ-ONLY all-zeros array of the SAME shape and dtype
            // (probed: `np.imag(real).flags.writeable` is False, dtype preserved — and it OWNS its zeros:
            // owndata=True, base=None, so clear WRITEABLE on the owned array itself rather than aliasing
            // a view, which would misreport owndata=False. setflags(write=True) on it then succeeds,
            // exactly as NumPy's _IsWriteable allows for an owner).
            // The descriptor, not the storage lane: a string array's imaginary part is its own dtype filled with
            // empty strings (all-zero bytes), which only the descriptor can allocate (probed: np.imag(U3) is <U3 '').
            // The layout is array_imag_get's PyArray_ISFORTRAN(self): F order for an F-contiguous array that is not also
            // C-contiguous (a transposed C block's imaginary part is F-contiguous), C otherwise.
            bool fortran = val.Shape.IsFContiguous && !val.Shape.IsContiguous;
            var zeros = new NDArray(val.dtype, Shape.Adopt(val.Shape.dimensions, fortran ? 'F' : 'C'), true);
            zeros.Storage.SetShapeUnsafe(zeros.Shape.WithFlags(flagsToClear: ArrayFlags.WRITEABLE));
            return zeros;
        }

        /// <summary>
        ///     NumPy's <c>array_imag_set</c> (<c>getset.c</c>), the statements of <c>self.imag = value</c>: copies
        ///     <paramref name="value"/> into the imaginary lane of a complex array, refuses any other dtype.
        /// </summary>
        /// <param name="self">The array whose imaginary parts are written.</param>
        /// <param name="value">The values, broadcast to <paramref name="self"/>'s shape and cast UNSAFELY into float64
        ///     (NumPy's <c>PyArray_CopyObject</c>: a complex value keeps its real part).</param>
        /// <exception cref="TypeError"><paramref name="self"/> is not complex: <c>array does not have imaginary part to
        ///     set</c> (NumPy's text).</exception>
        /// <exception cref="ValueError">np.copyto's: the array is read-only ("assignment destination is read-only"), or
        ///     <paramref name="value"/> does not broadcast to its shape ("could not broadcast input array from shape (4,)
        ///     into shape (3,)").</exception>
        /// <remarks>
        ///     The scope reclaims the imaginary-lane view the copy targets: it holds a counted reference on the array's
        ///     buffer, so leaving it to the finalizer would keep the buffer out of the pool after the array is disposed.
        ///     The caller (the <see cref="imag"/> setter) opens the numpy.matrix region first.
        /// </remarks>
        [NDScoped]
        internal static void ImagSet(NDArray self, NDArray value)
        {
            if (self.typecode != NPTypeCode.Complex)
                throw new TypeError("array does not have imaginary part to set");
            UnmanagedStorage.Copyto(ImagPart(self), value, NPY_CASTING.NPY_UNSAFE_CASTING, null);
        }
    }
}
