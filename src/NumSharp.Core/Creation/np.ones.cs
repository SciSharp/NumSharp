using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, filled with ones — NumPy's one signature,
        ///     <c>np.ones(shape, dtype=None, order='C', *, device=None)</c>.
        /// </summary>
        /// <param name="shape">The dimensions: an int (<c>2</c>), a tuple (<c>(2, 3)</c>), an <c>int[]</c> / <c>long[]</c> or a
        ///     <see cref="Shape"/> (all convert implicitly). Only the dimensions are read: the layout comes from
        ///     <paramref name="order"/>, as NumPy's shape is a tuple of ints — a <see cref="Shape"/> taken from a transposed
        ///     view gives a C-ordered array unless <paramref name="order"/> says <c>'F'</c>.</param>
        /// <param name="dtype">
        ///     The desired dtype — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> (<see cref="uint8"/>) all
        ///     convert implicitly. Null (NumPy's None) is float64.
        /// </param>
        /// <param name="order"><c>'C'</c> (row-major, the default) or <c>'F'</c> (column-major), in either case. NumPy's
        ///     <c>'A'</c> and <c>'K'</c> mean nothing without a source array and are refused.</param>
        /// <param name="device">Null or <c>"cpu"</c> (the Array-API device; NumPy has no other).</param>
        /// <returns>A new, owned, writeable array of ones: the Python int 1 as the dtype's <c>setitem</c> converts it (True,
        ///     1+0j, ...), and the text <c>'1'</c> in every element of a string dtype (an unsized <c>'U'</c> / <c>'S'</c> request
        ///     is ONE character, NumPy's allocation rule).</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="order"/> is not one of C, F, A, K (<c>order must be
        ///     one of 'C', 'F', 'A', or 'K' (got 'X')</c>); <paramref name="device"/> is not <c>"cpu"</c> (<c>Device not
        ///     understood. Only "cpu" is allowed, but received: gpu</c>); <paramref name="order"/> is A or K (<c>only 'C' or 'F'
        ///     order is permitted</c>); a negative dimension (<c>negative dimensions are not allowed</c>) or a byte count past
        ///     the allocation limit.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> names a class NumSharp has no storage for.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.ones.html — NumPy's <c>a = empty(shape, dtype, order,
        ///     device=device); multiarray.copyto(a, 1, casting='unsafe')</c>: <see cref="CreationLayout"/> is empty's argument
        ///     handling, <see cref="NDArray.ones(Shape, DType)"/> the allocation and the fill.
        ///     <para>One signature since API-parity batch 8 (2026-11-12): the <c>int</c> / <c>int[]</c> / <c>long[]</c> /
        ///     <c>(int[], DType)</c> / <c>Shape</c>-only overloads, the <c>(Shape, DType dtype, string device)</c> form whose
        ///     dtype was required and the <c>(Shape, char order, DType)</c> form with the order before the dtype all went, as
        ///     did the generic <c>ones&lt;T&gt;</c> form (write the dtype: <c>np.ones(shape, np.float64)</c>). A positional char
        ///     after the shape no longer compiles: NumPy reads <c>np.ones(s, 'F')</c>'s <c>'F'</c> as the DTYPE (complex64), so
        ///     write <c>order: 'F'</c>.</para>
        /// </remarks>
        public static NDArray ones(Shape shape, DType dtype = null, char order = 'C', string device = null)
            => NDArray.ones(CreationLayout(shape, order, device), dtype);
    }
}
