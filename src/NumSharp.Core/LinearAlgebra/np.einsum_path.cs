using System;
using NumSharp.Backends;

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
        ///     <para>
        ///     The statements are <see cref="TensorEngine.EinsumPath"/>, on the first operand's engine (the default one
        ///     with none).
        ///     </para>
        /// </remarks>
        /// <exception cref="ArgumentNullException"><paramref name="subscripts"/> is null.</exception>
        /// <exception cref="ValueError">A null operand, or subscripts NumPy's parser refuses (NumPy's texts); the variadic
        ///     overload with nothing to parse.</exception>
        /// <exception cref="KeyError">An unknown path name.</exception>
        /// <exception cref="RuntimeError">An explicit path that does not contract every operand.</exception>
        /// <exception cref="TypeError">An <c>optimize</c> NumPy does not understand; the variadic overload's operand that is
        ///     not an NDArray.</exception>
        /// <exception cref="IndexOutOfRangeException">An ellipsis with no operand to read its rank from.</exception>
        public static (EinsumPath path, string repr) einsum_path(string subscripts, NDArray[] operands)
            => EinsumPathOnEngine(subscripts, operands, "greedy");

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
            => EinsumPathOnEngine(subscripts, operands, optimize);

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

                return EinsumPathOnEngine(subscripts, arrays, "greedy");
            }

            string rendered = EinsumSubscripts.FromSublists(operands, out NDArray[] parsed);
            return EinsumPathOnEngine(rendered, parsed, "greedy");
        }

        /// <summary>
        ///     Runs the plan on the engine np.einsum would contract on: the first operand's (as np.dot dispatches on its
        ///     first), or the default engine when there is none or the first is null - its member then raises NumPy's
        ///     error for that.
        /// </summary>
        /// <param name="subscripts">The einsum subscripts (null is refused by the member).</param>
        /// <param name="operands">The operands, or null for none.</param>
        /// <param name="optimize">NumPy's <c>optimize=</c>.</param>
        /// <returns>The member's path and report.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="subscripts"/> is null.</exception>
        /// <exception cref="ValueError">A null operand, or subscripts NumPy's parser refuses.</exception>
        /// <exception cref="KeyError">An unknown path name.</exception>
        /// <exception cref="RuntimeError">An explicit path that does not contract every operand.</exception>
        /// <exception cref="TypeError">An <paramref name="optimize"/> NumPy does not understand.</exception>
        /// <exception cref="IndexOutOfRangeException">An ellipsis with no operand to read its rank from.</exception>
        private static (EinsumPath path, string repr) EinsumPathOnEngine(string subscripts, NDArray[] operands, object optimize)
        {
            NDArray first = operands is { Length: > 0 } ? operands[0] : null;
            return (first?.TensorEngine ?? BackendFactory.GetEngine()).EinsumPath(subscripts, operands, optimize);
        }
    }
}
