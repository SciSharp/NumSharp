// The functions of numpy/lib/_twodim_base_impl.py that compute: vander (the np-layer dtype plan's phase P3, batch 9).
// NumPy writes it in Python over a ufunc (multiply.accumulate makes the powers); here it is an engine member that
// DefaultEngine implements with NumPy's statements. The other functions of the same NumPy file (diag, diagflat, eye,
// fliplr, flipud, tri, tril, triu and the index helpers) create, place or select elements by position: they are the
// array object's (NDArray.TwoDimBase.cs, Appendix D).

using System;

namespace NumSharp
{
    public abstract partial class TensorEngine
    {
        /// <summary>
        ///     NumPy's <c>np.vander(x, N=None, increasing=False)</c>: the Vandermonde matrix of a 1-D <paramref name="x"/>,
        ///     whose columns are powers of <paramref name="x"/> - <c>x^(N-1)</c> down to <c>x^0</c>, or <c>x^0</c> up to
        ///     <c>x^(N-1)</c> when <paramref name="increasing"/> is set. NumPy makes the powers with
        ///     <c>multiply.accumulate</c> in <c>promote_types(x.dtype, int)</c>, so an integer matrix wraps where NumPy's
        ///     does and every float power is the product NumPy rounds, not <c>x ** k</c>.
        /// </summary>
        /// <param name="x">
        ///     The 1-D input, a plain array: <c>np.vander</c> computes a <c>numpy.matrix</c> operand as a plain array
        ///     before it calls this member. A 0-d or N-D array is refused.
        /// </param>
        /// <param name="N">The number of columns, or null for a square matrix (<c>len(x)</c> columns).</param>
        /// <param name="increasing">Whether the powers increase from left to right.</param>
        /// <returns>
        ///     A new C-contiguous matrix of shape <c>(len(x), N)</c> that owns its buffer, in
        ///     <c>promote_types(x.dtype, int64)</c>: int64 for bool, char and the integers up to int64, float64 for uint64
        ///     and every float dtype, decimal and complex128 as they are. A fixed-width text <paramref name="x"/> gives a
        ///     21-character text matrix (NumPy promotes int64 to text): the text <c>'1'</c> in every row for one column, an
        ///     empty matrix for no column.
        /// </returns>
        /// <exception cref="ValueError">
        ///     <paramref name="x"/> is not one-dimensional (<c>x must be a one-dimensional array or sequence.</c>), or
        ///     <paramref name="N"/> is negative (the allocation's <c>negative dimensions are not allowed</c>).
        /// </exception>
        /// <exception cref="TypeError">
        ///     A fixed-width text <paramref name="x"/> (<c>S</c> or <c>U</c>) with two or more columns: the powers come from
        ///     <c>multiply</c>, which has no string loop (NumPy's no-loop text).
        /// </exception>
        /// <exception cref="DTypePromotionError">
        ///     A <c>StringDType</c> <paramref name="x"/>, whatever <paramref name="N"/>: <c>promote_types(StringDType,
        ///     int64)</c> finds no common DType, before the allocation.
        /// </exception>
        /// <exception cref="OverflowException">
        ///     <paramref name="N"/> is null and <paramref name="x"/> has more than <see cref="int.MaxValue"/> elements (the
        ///     column count is an <see cref="int"/>).
        /// </exception>
        /// <remarks>
        ///     NumPy's statements: <c>v = empty((len(x), N), dtype=promote_types(x.dtype, int))</c>; the first column
        ///     (or the last, decreasing) is 1; the others start as <c>x</c> and <c>multiply.accumulate</c> along the
        ///     columns turns column <c>k</c> into <c>x^(k+1)</c>. An override must keep that product order: a float
        ///     power computed another way rounds differently.
        /// </remarks>
        public abstract NDArray Vander(NDArray x, int? N = null, bool increasing = false);
    }
}
