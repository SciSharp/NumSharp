using System;

namespace NumSharp
{
    public partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.size(a, axis=None)</c>: the number of elements along the given axes - every element by default,
        ///     the length of one axis, or the product of the lengths of a tuple of axes.
        /// </summary>
        /// <param name="a">Input data.</param>
        /// <param name="axis">NumPy's <c>axis</c>: <c>null</c> (None, the default) counts every element; one axis
        ///     (<c>1</c>, <c>-1</c>) gives that axis' length; a tuple (<c>(0, 2)</c>, <c>new[] { 0, 2 }</c>) the product of
        ///     its axes' lengths, and the empty tuple 1. Negative axes count from the end.</param>
        /// <returns>The number of elements along the given axes.</returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.size.html
        ///     <para>
        ///     NumPy 2.4's statements: <c>axis = normalize_axis_tuple(axis, len(shape))</c>, then the product of those
        ///     lengths — so a 0-d input has no axis at all (only the empty tuple, whose product is 1).
        ///     </para>
        ///     <para>One signature since API-parity batch 4 (2026-11-12): the axis was an <c>int?</c>, so NumPy's tuple form
        ///     had no spelling; an <c>int?</c> argument converts to <see cref="NDAxis"/>.</para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError">An axis is out of range ("axis … is out of bounds for array of dimension …", the
        ///     first one in order, as given; every axis is range-checked before the repeat check).</exception>
        /// <exception cref="ValueError">A tuple names an axis twice (<c>repeated axis</c>).</exception>
        public static long size(NDArray a, NDAxis? axis = null)
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            if (axis is null)
                return a.size;

            // normalize_axis_tuple's AxisError and repeat check, on the axes as given (the old wrap-then-index read the
            // dimension array out of its bounds for an axis past either end, and a 0-d input), then math.prod of the
            // lengths: 1 for the empty tuple.
            long product = 1;
            var dims = a.Shape.dimensions;
            foreach (int ax in axis.Value.Normalize(a.ndim))
                product *= dims[ax];
            return product;
        }

        /// <summary>
        ///     <see cref="size(NDArray, NDAxis?)"/> with the axis as one <c>int</c>: the overload a C# integer binds
        ///     (its identity conversion beats NDAxis' implicit one), forwarding to the NDAxis form, so every answer,
        ///     error, layout and subclass result is that overload's.
        /// </summary>
        /// <param name="axis">The axis (NumPy's <c>axis=k</c>; a negative value counts from the end).</param>
        /// <inheritdoc cref="size(NDArray, NDAxis?)"/>
        public static long size(NDArray a, int axis)
            => size(a, (NDAxis?)axis);

        /// <summary>
        ///     <see cref="size(NDArray, NDAxis?)"/> with the axes as an <c>int[]</c>: the overload an <c>int[]</c>
        ///     binds, and a bare <c>null</c> too (null converts to <c>int[]</c> better than to NDAxis), forwarding to
        ///     the NDAxis form: a null array is NumPy's None, any other array NumPy's tuple (<c>new int[0]</c> is
        ///     <c>()</c>).
        /// </summary>
        /// <param name="axis">The axes as NumPy's tuple, or <c>null</c> for None.</param>
        /// <inheritdoc cref="size(NDArray, NDAxis?)"/>
        public static long size(NDArray a, int[] axis)
            => size(a, (NDAxis?)axis);

        //based on:
        //a=np.random.randint(0,10,(4,2,3))
        //print(np.size(a))
        //print(np.size(a, 0))
        //print(np.size(a, 1))
        //print(np.size(a, 2))
        //print(np.size(a, -1))
    }
}
