using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evaluates the lowest-cost contraction order for an <see cref="einsum(string, NDArray[])"/>
        ///     expression, considering the creation of intermediate arrays.
        /// </summary>
        /// <param name="subscripts">The einsum subscripts, e.g. <c>"ij,jk,kl-&gt;il"</c>.</param>
        /// <param name="operands">The arrays the subscripts label — only their SHAPES are read.</param>
        /// <returns>
        ///     The <see cref="EinsumPath"/> (NumPy's <c>['einsum_path', …]</c> list) and a printable
        ///     representation of the path.
        /// </returns>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html
        ///     <para>
        ///     A route-for-route port of NumPy 2.4.2's <c>numpy.einsum_path</c> (its greedy/optimal
        ///     planner in <c>einsumfunc.py</c>). The <see cref="EinsumPath"/> and every numeric metric
        ///     in the string are byte-identical to NumPy; the only difference is the placeholder letters
        ///     an <c>...</c> expands to in the printed string, which NumPy itself does not pin (they are
        ///     drawn from a hash-randomized set and vary per process). NumSharp draws them
        ///     deterministically, so the path and the numbers always match.
        ///     </para>
        ///     <para>
        ///     Spell the operands either as an <c>NDArray[]</c> (this overload) or variadically through
        ///     <see cref="einsum_path(object[])"/> — <c>np.einsum_path("ij,jk", a, b)</c>. The default
        ///     is <c>optimize: "greedy"</c>, NumPy's default for <c>einsum_path</c> (note
        ///     <see cref="einsum(string, NDArray[])"/>'s own default is <c>false</c>).
        ///     </para>
        /// </remarks>
        public static (EinsumPath path, string repr) einsum_path(string subscripts, NDArray[] operands)
            => EinsumPathCore(subscripts, operands, "greedy");

        /// <summary>
        ///     Evaluates the contraction order, choosing the path type via <paramref name="optimize"/>.
        /// </summary>
        /// <param name="subscripts">The einsum subscripts, e.g. <c>"ij,jk,kl-&gt;il"</c>.</param>
        /// <param name="operands">The arrays the subscripts label — only their SHAPES are read.</param>
        /// <param name="optimize">
        ///     <c>false</c>/<c>null</c> (no optimization), <c>true</c> (≙ <c>"greedy"</c>),
        ///     <c>"greedy"</c>, <c>"optimal"</c>, a precomputed <see cref="EinsumPath"/> (an explicit
        ///     path), or a <c>("greedy"|"optimal", maxIntermediateSize)</c> tuple that caps the largest
        ///     intermediate. Anything else is rejected exactly as NumPy rejects it.
        /// </param>
        /// <inheritdoc cref="einsum_path(string, NDArray[])"/>
        public static (EinsumPath path, string repr) einsum_path(string subscripts, NDArray[] operands, object optimize)
            => EinsumPathCore(subscripts, operands, optimize);

        /// <summary>
        ///     Evaluates the contraction order variadically — <c>np.einsum_path("ij,jk", a, b)</c> — and
        ///     in NumPy's SUBLIST spelling — <c>np.einsum_path(a, [0,1], b, [1,2], [0,2])</c>.
        ///     Optimization is <c>"greedy"</c>; use the <c>NDArray[]</c> overload to choose a different
        ///     <c>optimize</c>.
        /// </summary>
        /// <inheritdoc cref="einsum_path(string, NDArray[])"/>
        public static (EinsumPath path, string repr) einsum_path(params object[] operands)
        {
            if (operands is null || operands.Length == 0)
                throw new ValueError(
                    "must specify the einstein sum subscripts string and at least one operand, " +
                    "or at least one operand and its corresponding subscripts list");

            // A leading string is the ordinary spelling reached through this overload.
            if (operands[0] is string subscripts)
            {
                var arrays = new NDArray[operands.Length - 1];
                for (int i = 1; i < operands.Length; i++)
                {
                    arrays[i - 1] = operands[i] as NDArray
                                    ?? throw new TypeError(
                                        $"einsum_path operand {i - 1} must be an NDArray, got {operands[i]?.GetType().Name ?? "null"}");
                }

                return EinsumPathCore(subscripts, arrays, "greedy");
            }

            string rendered = EinsumSubscripts.FromSublists(operands, out NDArray[] parsed);
            return EinsumPathCore(rendered, parsed, "greedy");
        }

        private static (EinsumPath path, string repr) EinsumPathCore(string subscripts, NDArray[] operands, object optimize)
        {
            if (subscripts is null)
                throw new ArgumentNullException(nameof(subscripts));

            operands ??= Array.Empty<NDArray>();
            long[][] shapes = ExtractShapes(operands);
            EinsumPathPlanner.Directive directive = EinsumPathPlanner.ResolveOptimize(optimize);
            var (path, repr) = EinsumPathPlanner.Compute(subscripts, shapes, directive);
            return (new EinsumPath(path), repr);
        }

        private static long[][] ExtractShapes(NDArray[] operands)
        {
            var shapes = new long[operands.Length][];
            for (int i = 0; i < operands.Length; i++)
            {
                if (operands[i] is null)
                    throw new ValueError($"einsum_path operand {i} is null");
                shapes[i] = operands[i].Shape.dimensions;
            }

            return shapes;
        }
    }
}
