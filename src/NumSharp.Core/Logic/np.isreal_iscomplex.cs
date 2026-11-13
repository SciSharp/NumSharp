using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        /// Returns a bool array, where True if input element is real.
        /// </summary>
        /// <param name="x">Input array (NumPy's keyword <c>x</c>).</param>
        /// <returns>Boolean array of same shape, True where element has no imaginary part.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.isreal.html
        ///
        /// For non-complex arrays, all elements are considered real.
        /// For complex arrays, elements with zero imaginary part are real.
        /// <para>[NDScoped] reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.equal.</para>
        /// </remarks>
        /// <example>
        /// <code>
        /// var a = np.array(new int[] {1, 2, 3});
        /// np.isreal(a)  // [True, True, True]
        /// </code>
        /// </example>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        [NDScoped]
        public static NDArray isreal(NDArray x)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            return x.TensorEngine.Isreal(x);
        }

        /// <summary>
        /// Returns a bool array, where True if input element is complex.
        /// </summary>
        /// <param name="x">Input array (NumPy's keyword <c>x</c>).</param>
        /// <returns>Boolean array of same shape, True where element has non-zero imaginary part.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.iscomplex.html
        ///
        /// For non-complex arrays, all elements are considered not complex.
        /// For complex arrays, elements with non-zero imaginary part are complex.
        /// <para>[NDScoped] reclaims the np.imag view and the Scalar(0) temp the complex branch hands to np.not_equal.</para>
        /// </remarks>
        /// <example>
        /// <code>
        /// var a = np.array(new int[] {1, 2, 3});
        /// np.iscomplex(a)  // [False, False, False]
        /// </code>
        /// </example>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
        [NDScoped]
        public static NDArray iscomplex(NDArray x)
        {
            if (x is null) throw new ArgumentNullException(nameof(x));
            return x.TensorEngine.Iscomplex(x);
        }

        /// <summary>
        /// Return True if x is a not complex type or an array of complex numbers.
        /// </summary>
        /// <param name="x">Input array or scalar (NumPy's keyword <c>x</c>; a C# scalar converts to a 0-d array).</param>
        /// <returns>True if the array's dtype is not complex.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
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
        public static bool isrealobj(NDArray x)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));

            return x.GetTypeCode != NPTypeCode.Complex;
        }

        /// <summary>
        /// Return True if x is a complex type or an array of complex numbers.
        /// </summary>
        /// <param name="x">Input array or scalar (NumPy's keyword <c>x</c>; a C# scalar converts to a 0-d array).</param>
        /// <returns>True if the array's dtype is complex.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
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
        public static bool iscomplexobj(NDArray x)
        {
            if (x is null)
                throw new ArgumentNullException(nameof(x));

            return x.GetTypeCode == NPTypeCode.Complex;
        }
    }
}
