using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns a bool array, where True if input element is real.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Boolean array of same shape, True where element has no imaginary part.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.isreal.html
        ///
        /// For non-complex arrays, all elements are considered real.
        /// For complex arrays, elements with zero imaginary part are real.
        /// </remarks>
        /// <example>
        /// <code>
        /// var a = np.array(new int[] {1, 2, 3});
        /// np.isreal(a)  // [True, True, True]
        /// </code>
        /// </example>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.equal
        public static NDArray isreal(NDArray a)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped(a, static s => isreal(s));
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy (_type_check_impl.isreal): `return imag(x) == 0`.
            // For a real/int/bool dtype the imaginary part is identically zero, so the answer is
            // all-True regardless of value — emit a fresh C-contiguous bool array from the DIMENSIONS
            // only (never `a.Shape`, whose strides/offset for a sliced/broadcast/strided view make
            // np.ones read past the logical window and return garbage bytes).
            // Text: imag(x) is the empty string, and `'' == 0` has no comparison loop, so NumPy's == answers False
            // everywhere (_failed_comparison_workaround) — a string array is never "real" (probed 2.4.2).
            if (a.GetTypeCode == NPTypeCode.String)
                return np.zeros(Shape.Adopt(a.Shape.dimensions), NPTypeCode.Boolean);

            if (a.GetTypeCode != NPTypeCode.Complex)
                return np.ones(Shape.Adopt(a.Shape.dimensions), NPTypeCode.Boolean);

            // Complex: the imaginary lane (a strided float64 view via np.imag) compared to zero.
            // IEEE `==` gives NumPy's semantics exactly: -0.0 == 0 (real), NaN/Inf != 0 (not real).
            return np.equal(np.imag(a), NDArray.Scalar(0d));
        }

        /// <summary>
        /// Returns a bool array, where True if input element is complex.
        /// </summary>
        /// <param name="a">Input array.</param>
        /// <returns>Boolean array of same shape, True where element has non-zero imaginary part.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.iscomplex.html
        ///
        /// For non-complex arrays, all elements are considered not complex.
        /// For complex arrays, elements with non-zero imaginary part are complex.
        /// </remarks>
        /// <example>
        /// <code>
        /// var a = np.array(new int[] {1, 2, 3});
        /// np.iscomplex(a)  // [False, False, False]
        /// </code>
        /// </example>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        [NDScoped] // reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.not_equal
        public static NDArray iscomplex(NDArray a)
        {
            // A numpy.matrix operand: a complex one is `ax.imag != 0`, a ufunc (a matrix comes back); any other
            // dtype is a new zeros array, which is plain.
            if (NDMatrixBoundary.Sees(a))
                return a.GetTypeCode == NPTypeCode.Complex
                    ? NDMatrixBoundary.Wrapped(a, static s => iscomplex(s))
                    : NDMatrixBoundary.Plain(a, static s => iscomplex(s));
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // NumPy (_type_check_impl.iscomplex): complex dtype -> `ax.imag != 0`, otherwise
            // `zeros(shape, bool)`. What is tested is the VALUE of the imaginary part, not the dtype.
            // Non-complex short-circuits to all-False from the DIMENSIONS only (not `a.Shape`, whose
            // view strides/offset would make np.zeros emit garbage bytes on a strided/broadcast input).
            if (a.GetTypeCode != NPTypeCode.Complex)
            {
                // NumPy returns `res[()]` of `res = zeros(shape, bool)`: for an array that is a VIEW of the fresh buffer
                // (flags.owndata false), for a 0-d input the scalar; the view takes its own reference on the buffer.
                var res = np.zeros(Shape.Adopt((long[])a.Shape.dimensions.Clone()), NPTypeCode.Boolean);
                if (res.ndim == 0)
                    return res;
                var view = new NDArray(res.Storage.Alias(res.Shape));
                res.Dispose();
                return view;
            }

            // Complex: the imaginary lane (a strided float64 view via np.imag) compared to zero.
            // IEEE `!=` gives NumPy's semantics exactly: -0.0 != 0 is False, NaN/Inf != 0 is True.
            return np.not_equal(np.imag(a), NDArray.Scalar(0d));
        }

        /// <summary>
        /// Return True if x is a not complex type or an array of complex numbers.
        /// </summary>
        /// <param name="a">Input array or scalar.</param>
        /// <returns>True if the array's dtype is not complex.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.isrealobj.html
        ///
        /// The type of the input is checked, not the value. Even an array of complex
        /// numbers with zero imaginary parts will return False.
        /// </remarks>
        /// <example>
        /// <code>
        /// var a = np.array(new int[] {1, 2, 3});
        /// np.isrealobj(a)  // True (dtype is int, not complex)
        /// </code>
        /// </example>
        public static bool isrealobj(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            return a.GetTypeCode != NPTypeCode.Complex;
        }

        /// <summary>
        /// Return True if x is a complex type or an array of complex numbers.
        /// </summary>
        /// <param name="a">Input array or scalar.</param>
        /// <returns>True if the array's dtype is complex.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.iscomplexobj.html
        ///
        /// The type of the input is checked, not the value.
        /// </remarks>
        /// <example>
        /// <code>
        /// var a = np.array(new int[] {1, 2, 3});
        /// np.iscomplexobj(a)  // False (dtype is int, not complex)
        /// </code>
        /// </example>
        public static bool iscomplexobj(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            return a.GetTypeCode == NPTypeCode.Complex;
        }
    }
}
