using System;

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
        /// <param name="source">Source shape for A/K resolution. Null = no reference (A/K fall back to C).</param>
        /// <returns>Physical order: 'C' or 'F'.</returns>
        /// <exception cref="ArgumentException">Thrown when order is not one of C/F/A/K.</exception>
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
                    // accept 'A' raise "only 'C' or 'F' order is permitted".
                    if (!source.HasValue)
                        throw new ArgumentException(
                            "only 'C' or 'F' order is permitted (order='A' requires a source array)",
                            nameof(order));
                    // Prefer F only when source is strictly F-contiguous (not also C-contiguous).
                    if (source.Value.IsFContiguous && !source.Value.IsContiguous)
                        return 'F';
                    return 'C';

                case 'K':
                case 'k':
                    // "Keep" requires a source array. Matches NumPy: creation functions that do not
                    // accept 'K' raise "only 'C' or 'F' order is permitted".
                    if (!source.HasValue)
                        throw new ArgumentException(
                            "only 'C' or 'F' order is permitted (order='K' requires a source array)",
                            nameof(order));
                    if (source.Value.IsContiguous)
                        return 'C';
                    if (source.Value.IsFContiguous)
                        return 'F';
                    return 'C';  // Non-contig source: conservative fallback

                default:
                    throw new ArgumentException(
                        $"order must be one of 'C', 'F', 'A', 'K' (got '{order}')",
                        nameof(order));
            }
        }
    }
}
