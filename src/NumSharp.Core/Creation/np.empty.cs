using System;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Return a new array of given shape and type, without initializing entries — NumPy's one signature,
        ///     <c>np.empty(shape, dtype=None, order='C', *, device=None)</c>.
        /// </summary>
        /// <param name="shape">The dimensions: an int (<c>2</c>), a tuple (<c>(2, 3)</c>), an <c>int[]</c> / <c>long[]</c> or a
        ///     <see cref="Shape"/> (all convert implicitly). Only the dimensions are read: the layout comes from
        ///     <paramref name="order"/>, as NumPy's shape is a tuple of ints — a <see cref="Shape"/> taken from a transposed
        ///     view gives a C-ordered array unless <paramref name="order"/> says <c>'F'</c>.</param>
        /// <param name="dtype">
        ///     The desired dtype — one descriptor parameter, like NumPy's <c>dtype</c>: a C# <see cref="Type"/>, an
        ///     <see cref="NPTypeCode"/>, a NumPy dtype string (<c>"f4"</c>) or a <see cref="DType"/> all convert implicitly.
        ///     Null (NumPy's None) is float64.
        /// </param>
        /// <param name="order"><c>'C'</c> (row-major, the default) or <c>'F'</c> (column-major), in either case. NumPy's
        ///     <c>'A'</c> and <c>'K'</c> mean nothing without a source array and are refused.</param>
        /// <param name="device">Null or <c>"cpu"</c> (the Array-API device; NumPy has no other).</param>
        /// <returns>A new, owned, writeable array whose elements are uninitialized (arbitrary): write every element before
        ///     reading it.</returns>
        /// <exception cref="ValueError">In NumPy's order: <paramref name="order"/> is not one of C, F, A, K (<c>order must be
        ///     one of 'C', 'F', 'A', or 'K' (got 'X')</c>); <paramref name="device"/> is not <c>"cpu"</c> (<c>Device not
        ///     understood. Only "cpu" is allowed, but received: gpu</c>); <paramref name="order"/> is A or K (<c>only 'C' or 'F'
        ///     order is permitted</c>); a negative dimension (<c>negative dimensions are not allowed</c>) or a byte count past
        ///     the allocation limit.</exception>
        /// <exception cref="NotSupportedException"><paramref name="dtype"/> names a class NumSharp has no storage for.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.empty.html — NumPy's <c>array_empty</c>
        ///     (<c>multiarraymodule.c</c>).
        ///     <para>One signature since API-parity batch 8 (2026-11-12): the <c>int</c> / <c>int[]</c> / <c>long[]</c> /
        ///     <c>Shape</c>-only overloads, the <c>(Shape, DType dtype, string device)</c> form whose dtype was required and the
        ///     <c>(Shape, char order, DType)</c> form with the order before the dtype all went, as did the generic
        ///     <c>empty&lt;T&gt;</c> forms (write the dtype: <c>np.empty(shape, np.float64)</c>). A positional char after the
        ///     shape no longer compiles: NumPy reads <c>np.empty(s, 'F')</c>'s <c>'F'</c> as the DTYPE (complex64), so write
        ///     <c>order: 'F'</c>.</para>
        /// </remarks>
        public static NDArray empty(Shape shape, DType dtype = null, char order = 'C', string device = null)
            => new NDArray(dtype ?? DType.Double, CreationLayout(shape, order, device), false);

        /// <summary>
        ///     The argument handling NumPy's <c>array_empty</c> / <c>array_zeros</c> do before they allocate, shared by
        ///     <see cref="empty(Shape, DType, char, string)"/>, <see cref="zeros(Shape, DType, char, string)"/>,
        ///     <see cref="ones(Shape, DType, char, string)"/> and <see cref="full(Shape, object, DType, char, string)"/> (the
        ///     last two are NumPy's <c>empty</c> plus a <c>copyto</c>): the layout their fresh array gets.
        /// </summary>
        /// <param name="shape">The requested dimensions (their layout, if the <see cref="Shape"/> carries one, is ignored).</param>
        /// <param name="order">The order argument as passed.</param>
        /// <param name="device">The device argument as passed.</param>
        /// <returns>A C- or F-ordered layout of <paramref name="shape"/>'s dimensions, sharing its dimension array (nothing
        ///     writes into it).</returns>
        /// <exception cref="ValueError">The order letter, the device, then an A / K order, each with NumPy's text; a negative
        ///     dimension is the allocation's (NumPy's <c>PyArray_NewFromDescr</c>), raised by the caller's allocation.</exception>
        internal static Shape CreationLayout(Shape shape, char order, string device)
        {
            // NumPy's argument parser converts the order letter (PyArray_OrderConverter) before the device
            // (PyArray_DeviceConverterOptional); the body then switches on C / F, refusing A and K. Each error therefore
            // wins over the ones after it, exactly as in np.empty(2, order='A', device='gpu') (the device's ValueError).
            NPY_ORDER converted = OrderResolver.Convert(order);
            DeviceArguments.ValidateDevice(device);
            bool cOrder = OrderResolver.RequireCOrFOrder(converted);
            // Only the dimensions are NumPy's: a Shape taken from a view (transposed, permuted, stepped) would otherwise hand
            // its own layout to the allocator, which keeps a dense F / KEEPORDER one (gap VS01: np.zeros(t.Shape) of a
            // transposed t came back F-ordered where NumPy's np.zeros(t.shape) is C).
            return new Shape(shape.dimensions ?? Array.Empty<long>(), cOrder ? 'C' : 'F', copy: false);
        }
    }
}
