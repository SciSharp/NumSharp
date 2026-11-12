using System;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, filled with <paramref name="fill_value"/> — NumPy's one signature,
        ///     <c>np.full(shape, fill_value, dtype=None, order='C', *, device=None)</c>.
        /// </summary>
        /// <param name="shape">The dimensions: an int (<c>2</c>), a tuple (<c>(2, 3)</c>), an <c>int[]</c> / <c>long[]</c> or a
        ///     <see cref="Shape"/> (all convert implicitly). Only the dimensions are read: the layout comes from
        ///     <paramref name="order"/>, as NumPy's shape is a tuple of ints.</param>
        /// <param name="fill_value">Fill value: a scalar, or an array (an <see cref="NDArray"/> of rank ≥ 1, a C# array) that
        ///     broadcasts into the result, as NumPy's <c>copyto(a, fill_value, casting='unsafe')</c> broadcasts it. A C# value's
        ///     own type gives the default dtype (a C# <c>int</c> is int32, where NumPy's Python int is int64).</param>
        /// <param name="dtype">
        ///     The desired dtype — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     Null infers it from <paramref name="fill_value"/> (an array fill value gives its own dtype).
        /// </param>
        /// <param name="order"><c>'C'</c> (row-major, the default) or <c>'F'</c> (column-major), in either case: NumPy's
        ///     <c>empty(shape, dtype, order)</c> layout, so <c>'A'</c> and <c>'K'</c> are refused.</param>
        /// <param name="device">Null or <c>"cpu"</c> (the Array-API device; NumPy has no other).</param>
        /// <returns>A new, owned, writeable array of <paramref name="shape"/>'s dimensions with <paramref name="fill_value"/>
        ///     cast into every element.</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="order"/> is not one of C, F, A, K;
        ///     <paramref name="device"/> is not <c>"cpu"</c>; <paramref name="order"/> is A or K (<c>only 'C' or 'F' order is
        ///     permitted</c>); a negative dimension; <paramref name="fill_value"/> is null and <paramref name="dtype"/> is a
        ///     <c>StringDType(coerce=False)</c>; an array fill value that does not broadcast into the result (<c>could not
        ///     broadcast input array from shape (2,) into shape (3,)</c>), or whose text does not parse into a numeric
        ///     <paramref name="dtype"/>.</exception>
        /// <exception cref="NotSupportedException"><paramref name="fill_value"/> is null (Python's <c>None</c>) and
        ///     <paramref name="dtype"/> is null: NumPy builds an object array there, a dtype NumSharp does not have.</exception>
        /// <exception cref="TypeError"><paramref name="fill_value"/> is null and <paramref name="dtype"/> is an integer (or
        ///     <c>decimal</c>) dtype, for a non-empty shape: Python's <c>int(None)</c>, raised by NumPy's
        ///     <c>copyto(a, None, casting='unsafe')</c> (an empty result converts nothing and returns).</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.full.html — NumPy's <c>a = empty(shape, dtype, order,
        ///     device=device); multiarray.copyto(a, fill_value, casting='unsafe')</c>: <see cref="CreationLayout"/> is empty's
        ///     argument handling, <see cref="NDArray.full(Shape, object, DType)"/> the inference, allocation and fill.
        ///     <para>A null <paramref name="fill_value"/> is Python's <c>None</c>, filled the way NumPy's
        ///     <c>copyto(a, None, casting='unsafe')</c> fills it: NaN (the positive <c>NPY_NAN</c>) for a float dtype,
        ///     <c>nan+nanj</c> for complex, <c>False</c> for bool, <c>'None'</c> (truncated) for <c>S</c> / <c>U</c>, a
        ///     <c>StringDType</c>'s NA object or <c>'None'</c>.</para>
        ///     <para>One signature since API-parity batch 8 (2026-11-12): the <c>int[]</c> / <c>long[]</c> shape overloads went
        ///     (both convert to <see cref="Shape"/>), the order arrived, and the generic <c>full&lt;T&gt;</c> forms went (write
        ///     the dtype: <c>np.full(shape, value, np.int32)</c>). The device moved from the fourth position to the fifth.</para>
        /// </remarks>
        public static NDArray full(Shape shape, object fill_value, DType dtype = null, char order = 'C', string device = null)
            => NDArray.full(CreationLayout(shape, order, device), fill_value, dtype);
    }
}
