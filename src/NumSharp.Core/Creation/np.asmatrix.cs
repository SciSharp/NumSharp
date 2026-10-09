using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Interpret the input as a matrix. Unlike <see cref="matrix(NDArray, DType, bool)"/>,
        ///     <c>asmatrix</c> does not copy an array: it returns an <see cref="NDMatrix"/> VIEW that shares
        ///     memory, and a matrix argument is returned itself.
        /// </summary>
        /// <param name="data">Input array.</param>
        /// <param name="dtype">Data-type of the output. <c>null</c> keeps the input dtype; another dtype
        ///     converts, which copies.</param>
        /// <returns>
        ///     <paramref name="data"/> interpreted as a matrix. A 0-D input becomes shape <c>(1, 1)</c>,
        ///     a 1-D input of length N becomes <c>(1, N)</c>, and a 2-D input keeps its shape.
        /// </returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">More than two axes of <paramref name="data"/> are longer than 1
        ///     (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="data"/> has more than two axes and one of
        ///     extent 0 (NumPy's <c>cannot reshape array of size 0 into shape (1,3)</c>).</exception>
        /// <remarks>
        ///     Port of NumPy 2.x <c>numpy.asmatrix</c> ≙ <c>matrix(data, dtype=dtype, copy=False)</c>. The
        ///     dimensional coercion is NumPy's <c>matrix.__array_finalize__</c> (<see cref="Shape.AsMatrix"/>),
        ///     including the &gt;2-D behaviour: axes of length 1 are dropped and the result must then be 2-D. The
        ///     view is preserved for strided, transposed and reversed inputs (no copy), and it keeps the input's
        ///     writeability. The result is an <see cref="NDMatrix"/>: <c>*</c> is the matrix product on it and it
        ///     has <c>.T</c>, <c>.H</c>, <c>.I</c>, <c>.A</c>, <c>.A1</c>; seen through an <see cref="NDArray"/>
        ///     reference it is a plain 2-D array.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html
        /// </remarks>
        public static NDMatrix asmatrix(NDArray data, DType dtype = null)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));
            return NDMatrix.Create(data, dtype, copy: false);
        }

        /// <summary>
        ///     Interpret a matrix string as a matrix. Rows are separated by <c>';'</c> and columns by
        ///     commas and/or whitespace, e.g. <c>"1 2; 3 4"</c>. Surrounding brackets are ignored. The dtype
        ///     is NumPy's discovered dtype (bool, int64, float64 or complex128, by the tokens) unless
        ///     <paramref name="dtype"/> is given.
        /// </summary>
        /// <param name="data">Matrix string such as <c>"1 2; 3 4"</c>. Each token is one Python literal.</param>
        /// <param name="dtype">Data-type of the output. <c>null</c> infers it from the values.</param>
        /// <returns>A new matrix of shape <c>(rows, columns)</c>; <c>(1, 0)</c> float64 for an empty string.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">If the rows are not all the same length (<c>Rows not the same size.</c>),
        ///     or a token is Python syntax but not a literal (a name such as <c>nan</c>).</exception>
        /// <exception cref="SyntaxError">A token is not Python syntax.</exception>
        /// <exception cref="NotSupportedException">A token is a string, <c>None</c> or <c>...</c>, or an integer
        ///     beyond uint64.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html</remarks>
        public static NDMatrix asmatrix(string data, DType dtype = null)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));
            return NDMatrix.Create(data, dtype);
        }

        /// <summary>
        ///     Interpret a C# array as a matrix: a typed array of a NumSharp dtype is converted like
        ///     <c>np.array</c>, any other array is a Python list coerced by NumPy's sequence rules. A C# array is
        ///     always copied into new memory (as NumPy copies a list).
        /// </summary>
        /// <param name="data">The array, with at most two axes after conversion.</param>
        /// <param name="dtype">Data-type of the output, or <c>null</c> for the discovered one.</param>
        /// <returns>A new matrix.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">More than two axes (<c>matrix must be 2-dimensional</c>), or a ragged list.</exception>
        /// <exception cref="NotSupportedException">A list holding strings, <c>null</c> or other objects.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html</remarks>
        public static NDMatrix asmatrix(Array data, DType dtype = null)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));
            return NDMatrix.Create((object)data, dtype, copy: false);
        }

        /// <summary>
        ///     Interpret any other value as a matrix: a tuple or collection (a Python sequence), an array or string
        ///     held in an <see cref="object"/> reference, or a boxed number.
        /// </summary>
        /// <param name="data">The value.</param>
        /// <param name="dtype">Data-type of the output, or <c>null</c> for the discovered one.</param>
        /// <returns>The matrix: a view for an <see cref="NDArray"/>, new memory for everything else.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is <c>null</c>.</exception>
        /// <exception cref="ValueError">More than two axes, a ragged sequence, or an array that cannot be a
        ///     matrix.</exception>
        /// <exception cref="SyntaxError">A matrix literal with a token that is not Python syntax.</exception>
        /// <exception cref="NotSupportedException">Strings or other objects inside a sequence.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asmatrix.html</remarks>
        public static NDMatrix asmatrix(object data, DType dtype = null)
        {
            if (data is null)
                throw new ArgumentNullException(nameof(data));
            return NDMatrix.Create(data, dtype, copy: false);
        }
    }
}
