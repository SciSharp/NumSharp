using System;
using NumSharp.Backends.Iteration;

namespace NumSharp
{
    /// <summary>
    ///     Resolves NumPy memory order specifiers ('C', 'F', 'A', 'K') to physical storage orders.
    ///     NumPy defines four order modes but only two physical layouts (C and F);
    ///     'A' and 'K' are logical decisions that resolve to either 'C' or 'F' based on an input array.
    /// </summary>
    /// <remarks>
    ///     NumPy reference: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#memory-layout
    ///     <list type="bullet">
    ///         <item><c>'C'</c> - Row-major (last axis varies fastest). Always resolves to 'C'.</item>
    ///         <item><c>'F'</c> - Column-major (first axis varies fastest). Always resolves to 'F'.</item>
    ///         <item><c>'A'</c> - "Any": resolves to 'F' if source is F-contiguous and not C-contiguous, else 'C'.</item>
    ///         <item><c>'K'</c> - "Keep": preserves source layout. F-contig source -&gt; F, else C.</item>
    ///     </list>
    ///     For 'A' and 'K' with no source, the resolver defaults to 'C' (NumPy behavior for creation functions).
    /// </remarks>
    internal static class OrderResolver
    {
        /// <summary>
        ///     Resolves any NumPy order char to a physical storage order ('C' or 'F').
        /// </summary>
        /// <param name="order">User-facing order char ('C'/'F'/'A'/'K', case-insensitive).</param>
        /// <param name="source">Source shape for A/K resolution. Null = no reference: a creation function, where NumPy
        ///     refuses A and K.</param>
        /// <returns>Physical order: 'C' or 'F'.</returns>
        /// <exception cref="ValueError">
        ///     <paramref name="order"/> is not one of C/F/A/K in either case — NumPy's <c>PyArray_OrderConverter</c> text
        ///     verbatim, <c>order must be one of 'C', 'F', 'A', or 'K' (got 'X')</c> —, or it is A/K with no
        ///     <paramref name="source"/> (NumPy's creation functions: <c>only 'C' or 'F' order is permitted</c>, probed
        ///     2.4.2 for np.zeros / ones / empty / eye). <see cref="ValueError"/> is an <see cref="ArgumentException"/>, the
        ///     type this method threw before it carried NumPy's texts.
        /// </exception>
        public static char Resolve(char order, Shape? source = null)
        {
            switch (order)
            {
                case 'C':
                case 'c':
                    return 'C';

                case 'F':
                case 'f':
                    return 'F';

                case 'A':
                case 'a':
                    // "Any" requires a source array. Matches NumPy: creation functions that do not
                    // accept 'A' raise "only 'C' or 'F' order is permitted" (verbatim, a ValueError).
                    if (!source.HasValue)
                        throw new ValueError("only 'C' or 'F' order is permitted");
                    // Prefer F only when source is strictly F-contiguous (not also C-contiguous).
                    if (source.Value.IsFContiguous && !source.Value.IsContiguous)
                        return 'F';
                    return 'C';

                case 'K':
                case 'k':
                    // "Keep" requires a source array. Matches NumPy: creation functions that do not
                    // accept 'K' raise "only 'C' or 'F' order is permitted" (verbatim, a ValueError).
                    if (!source.HasValue)
                        throw new ValueError("only 'C' or 'F' order is permitted");
                    if (source.Value.IsContiguous)
                        return 'C';
                    if (source.Value.IsFContiguous)
                        return 'F';
                    return 'C';  // Non-contig source: conservative fallback

                default:
                    // NumPy's PyArray_OrderConverter text, verbatim (it used to lack the ", or" and carry .NET's
                    // " (Parameter 'order')" suffix).
                    throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            }
        }

        /// <summary>
        ///     Resolves the <c>order</c> argument of a constructor that lays a CALLER'S buffer out from dimensions alone —
        ///     NumPy's <c>ndarray(shape, buffer=values, order=order)</c>: <c>'F'</c> is the column-major layout, and every
        ///     other valid order (<c>'C'</c>, <c>'A'</c>, <c>'K'</c>, in either case) is the row-major one.
        /// </summary>
        /// <param name="order">The order char the caller passed.</param>
        /// <returns><c>'F'</c> for <c>'F'</c> or <c>'f'</c>; <c>'C'</c> for <c>'C'</c>, <c>'A'</c> or <c>'K'</c> in either
        ///     case.</returns>
        /// <exception cref="ValueError">
        ///     <paramref name="order"/> is none of the eight accepted chars. The text is NumPy 2.4.2's verbatim:
        ///     <c>order must be one of 'C', 'F', 'A', or 'K' (got 'X')</c>.
        /// </exception>
        /// <remarks>
        ///     Unlike <see cref="Resolve"/>, <c>'A'</c> and <c>'K'</c> need no source array here and never raise. NumPy's
        ///     <c>array_new</c> (<c>numpy/_core/src/multiarray/arrayobject.c</c>) converts the argument with
        ///     <c>PyArray_OrderConverter</c> and then only asks whether it is <c>NPY_FORTRANORDER</c> (<c>is_f_order</c>), so
        ///     every other order gets the C strides of <c>_array_fill_strides</c> (probed: <c>order='A'</c>, <c>'K'</c>,
        ///     <c>'a'</c> and <c>'k'</c> all give the C layout). Use it only where NumPy's own entry point behaves that way;
        ///     a function that resolves <c>'A'</c> or <c>'K'</c> against an input array uses <see cref="Resolve"/>.
        /// </remarks>
        public static char ResolveBufferOrder(char order)
        {
            switch (order)
            {
                case 'F':
                case 'f':
                    return 'F';

                case 'C':
                case 'c':
                case 'A':
                case 'a':
                case 'K':
                case 'k':
                    return 'C';

                default:
                    throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            }
        }

        /// <summary>
        ///     NumPy's <c>PyArray_OrderConverter</c> (<c>numpy/_core/src/multiarray/conversion_utils.c</c>,
        ///     <c>order_parser</c>): the four order letters in EITHER case, with no source array consulted and no
        ///     decision about which orders the caller accepts.
        /// </summary>
        /// <param name="order">The order char the caller passed.</param>
        /// <returns>The order NumPy's converter produces: <c>'C'</c>/<c>'c'</c> → C, <c>'F'</c>/<c>'f'</c> → F,
        ///     <c>'A'</c>/<c>'a'</c> → ANY, <c>'K'</c>/<c>'k'</c> → KEEP.</returns>
        /// <exception cref="ValueError"><paramref name="order"/> is none of the eight accepted chars. The text is NumPy
        ///     2.4.2's verbatim: <c>order must be one of 'C', 'F', 'A', or 'K' (got 'X')</c>.</exception>
        /// <remarks>
        ///     Use it where NumPy converts the argument at PARSE time and decides later what each order means: the
        ///     conversion error is then raised before any other check of the call (an argument parse precedes the body),
        ///     while the function's own refusal of <c>'A'</c> / <c>'K'</c> (<c>np.unravel_index</c>'s and
        ///     <c>np.ravel_multi_index</c>'s <c>only 'C' or 'F' order is permitted</c>) comes at its own point, after the
        ///     checks NumPy runs first. Unlike <see cref="Resolve"/>, a lowercase letter is NOT refused: NumPy's parser
        ///     accepts both cases for every letter (probed 2.4.2: <c>np.unravel_index([1, 2], (2, 3), order='f')</c> is
        ///     <c>order='F'</c>).
        /// </remarks>
        public static NPY_ORDER Convert(char order)
        {
            switch (order)
            {
                case 'C':
                case 'c':
                    return NPY_ORDER.NPY_CORDER;

                case 'F':
                case 'f':
                    return NPY_ORDER.NPY_FORTRANORDER;

                case 'A':
                case 'a':
                    return NPY_ORDER.NPY_ANYORDER;

                case 'K':
                case 'k':
                    return NPY_ORDER.NPY_KEEPORDER;

                default:
                    throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            }
        }
    }
}
