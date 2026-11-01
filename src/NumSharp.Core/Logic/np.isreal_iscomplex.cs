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
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        [NDScoped] // reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.equal
        public static NDArray isreal(NDArray a)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Wrapped(a, static s => isreal(s));
            return a.TensorEngine.Isreal(a);
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
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        [NDScoped] // reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.not_equal
        public static NDArray iscomplex(NDArray a)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // A numpy.matrix operand: a complex one is `ax.imag != 0`, a ufunc (a matrix comes back); any other
            // dtype is a new zeros array, which is plain.
            if (NDMatrixBoundary.Sees(a))
                return a.GetTypeCode == NPTypeCode.Complex
                    ? NDMatrixBoundary.Wrapped(a, static s => iscomplex(s))
                    : NDMatrixBoundary.Plain(a, static s => iscomplex(s));
            return a.TensorEngine.Iscomplex(a);
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
