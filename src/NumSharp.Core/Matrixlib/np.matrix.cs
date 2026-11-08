using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.matrix(data, dtype=None, copy=True)</c>: builds an <see cref="NDMatrix"/> — a
        ///     two-dimensional array whose <c>*</c> is the matrix product — from an array. Use
        ///     <see cref="asmatrix(NDArray, DType)"/> when a view is enough; this call copies by default.
        /// </summary>
        /// <param name="data">The array. Zero axes become <c>(1, 1)</c>, one axis a row <c>(1, N)</c>; more than
        ///     two axes are allowed only when all but two have length 1.</param>
        /// <param name="dtype">The dtype of the matrix, or <c>null</c> to keep the dtype of
        ///     <paramref name="data"/>.</param>
        /// <param name="copy"><c>false</c> returns a view of <paramref name="data"/> (or
        ///     <paramref name="data"/> itself when it is a matrix) if no dtype conversion is needed.</param>
        /// <returns>The matrix. With <paramref name="copy"/> it owns its data in C order.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">More than two axes of <paramref name="data"/> are longer than 1
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="data"/> has more than two axes and one of
        ///     extent 0.</exception>
        /// <remarks>
        ///     <para>NumPy calls the matrix class "not the recommended way to represent matrices" and plans to
        ///     remove it; it exists here so that code written against it ports. See <see cref="NDMatrix"/> for how
        ///     the matrix semantics bind in C#.</para>
        ///     <para>A C# number passed directly (<c>np.matrix(5)</c>) binds this overload as a 0-d array of its C#
        ///     type and gives a <c>(1, 1)</c> matrix (int32 for an <see cref="int"/>, where NumPy's Python int is
        ///     int64).</para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.matrix.html
        /// </remarks>
        public static NDMatrix matrix(NDArray data, DType dtype = null, bool copy = true) => NDMatrix.Create(data, dtype, copy);

        /// <summary>
        ///     NumPy's <c>np.matrix("1 2; 3 4")</c>: builds a matrix from a literal. Rows are separated by
        ///     <c>;</c>, columns by commas and/or white space; square brackets are ignored.
        /// </summary>
        /// <param name="data">The literal. Each token is one Python literal: an integer in any base
        ///     (<c>10</c>, <c>0x1F</c>, <c>1_000</c>), a float (<c>.5</c>, <c>1e3</c>), a complex number
        ///     (<c>1+2j</c>), or <c>True</c> / <c>False</c>.</param>
        /// <param name="dtype">The dtype, or <c>null</c> for NumPy's discovered dtype: bool when every token is a
        ///     bool, int64 for integers, float64 once a float appears, complex128 once a complex appears, and
        ///     float64 for a literal without elements.</param>
        /// <param name="copy">Accepted for signature parity; a literal is always converted into new memory.</param>
        /// <returns>A new <c>(rows, columns)</c> matrix; <c>(1, 0)</c> for an empty literal.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">Rows of different lengths (<c>Rows not the same size.</c>), or a token
        ///     that is Python but not a literal (<c>nan</c>, <c>2*3</c>).</exception>
        /// <exception cref="SyntaxError">A token that is not Python syntax (<c>1L</c>, a lone <c>+</c>).</exception>
        /// <exception cref="NotSupportedException">A string, <c>None</c> or <c>...</c> token, or an integer beyond
        ///     uint64: NumPy's str / object dtypes.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.matrix.html</remarks>
        public static NDMatrix matrix(string data, DType dtype = null, bool copy = true) => NDMatrix.Create(data, dtype);

        /// <summary>
        ///     NumPy's <c>np.matrix(data)</c> for a C# array: a typed array of a NumSharp dtype
        ///     (<c>double[,]</c>, <c>int[]</c>) is converted like <c>np.array</c>; any other array
        ///     (<c>object[]</c>, a jagged <c>double[][]</c>, an <c>NDArray[]</c>) is a Python list and is coerced by
        ///     NumPy's sequence rules.
        /// </summary>
        /// <param name="data">The array, with at most two axes after conversion.</param>
        /// <param name="dtype">The dtype, or <c>null</c> for the element type (a typed array) or NumPy's
        ///     discovered dtype (a list: C# integers are int64, as Python ints are).</param>
        /// <param name="copy">Accepted for signature parity; a C# array is always copied into new memory.</param>
        /// <returns>A new matrix. Its <c>flags.owndata</c> is false, as in NumPy, where the matrix is a view of
        ///     the array <c>np.array</c> built.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">More than two axes (<c>matrix must be 2-dimensional</c>), or a ragged list
        ///     (NumPy's inhomogeneous-shape text).</exception>
        /// <exception cref="NotSupportedException">A list holding strings, <c>null</c> or other objects.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.matrix.html</remarks>
        public static NDMatrix matrix(Array data, DType dtype = null, bool copy = true) => NDMatrix.Create((object)data, dtype, copy);

        /// <summary>
        ///     NumPy's <c>np.matrix(data)</c> for any other value: a tuple or a collection (a Python sequence, nested
        ///     up to two levels), an <see cref="NDArray"/> or string held in an <see cref="object"/> reference, or
        ///     a boxed number.
        /// </summary>
        /// <param name="data">The value.</param>
        /// <param name="dtype">The dtype, or <c>null</c> for the discovered one.</param>
        /// <param name="copy">Read for an <see cref="NDArray"/> only; every other value is converted into new
        ///     memory.</param>
        /// <returns>The matrix.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c> (NumPy builds an object
        ///     matrix holding <c>None</c>).</exception>
        /// <exception cref="ValueError">More than two axes, a ragged sequence, or an array that cannot be a
        ///     matrix.</exception>
        /// <exception cref="SyntaxError">A matrix literal with a token that is not Python syntax.</exception>
        /// <exception cref="NotSupportedException">Strings or other objects inside a sequence.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.matrix.html</remarks>
        public static NDMatrix matrix(object data, DType dtype = null, bool copy = true) => NDMatrix.Create(data, dtype, copy);
    }
}
