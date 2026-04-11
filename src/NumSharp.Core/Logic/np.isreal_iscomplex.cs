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
        public static NDArray isreal(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // For non-complex types, all elements are real
            if (a.GetTypeCode != NPTypeCode.Complex)
            {
                return np.ones(a.Shape, NPTypeCode.Boolean);
            }

            // For complex arrays, check if imaginary part is zero
            // TODO: Implement when complex support is added
            return np.ones(a.Shape, NPTypeCode.Boolean);
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
        public static NDArray iscomplex(NDArray a)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // For non-complex types, no elements are complex
            if (a.GetTypeCode != NPTypeCode.Complex)
            {
                return np.zeros(a.Shape, NPTypeCode.Boolean);
            }

            // For complex arrays, check if imaginary part is non-zero
            // TODO: Implement when complex support is added
            return np.zeros(a.Shape, NPTypeCode.Boolean);
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
