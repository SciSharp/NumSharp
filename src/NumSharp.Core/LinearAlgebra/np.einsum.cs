using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Evaluates the Einstein summation convention on the operands.
        /// </summary>
        /// <param name="subscripts">
        ///     Comma-separated subscript labels, optionally followed by <c>-&gt;</c> and the output
        ///     labels — e.g. <c>"ij,jk-&gt;ik"</c>. Spaces are ignored and <c>...</c> broadcasts.
        ///     Without <c>-&gt;</c> the output is INFERRED: every label used exactly once, in ASCII
        ///     order (so an upper-case label sorts before a lower-case one), preceded by the
        ///     broadcast dimensions.
        /// </param>
        /// <param name="operands">The arrays the subscripts label, in order.</param>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
        ///     <para>
        ///     <b>The subscripts are fully parsed and validated; the CONTRACTION is not implemented</b>
        ///     and raises <see cref="NotSupportedException"/> from the engine. Every rejection below
        ///     therefore behaves exactly as NumPy's today — a malformed expression, a wrong operand
        ///     count, a bad ellipsis, an impossible diagonal or a shape conflict is reported the same
        ///     way and with the same text.
        ///     </para>
        ///     <para>
        ///     Unlike the LAPACK entry points this is not waiting on a backend. What is missing is a
        ///     contraction kernel and a path planner, both NumSharp's own work. Express the
        ///     contraction with <see cref="tensordot(NDArray, NDArray, int)"/>, <see cref="dot"/>,
        ///     <see cref="matmul"/> or <see cref="vecdot"/> in the meantime.
        ///     </para>
        ///     <para>
        ///     <b>One deliberate divergence.</b> NumPy carries TWO independent einsum parsers — the C
        ///     one behind the default <c>optimize=False</c>, and a Python one behind the
        ///     <c>optimize</c> path — and they word their rejections differently for the same input.
        ///     NumSharp reproduces the C one, since that is what a default call hits, and uses it
        ///     whatever <c>optimize</c> says. The single exception is a label whose extents disagree
        ///     between operands: NumPy's C path leaks its ITERATOR's "remapped shapes" text there,
        ///     which describes axis bookkeeping rather than the contraction, so NumSharp raises
        ///     NumPy's other wording for the identical error —
        ///     <c>Size of label 'j' for operand 1 (3) does not match previous terms (4).</c>
        ///     </para>
        /// </remarks>
        /// <exception cref="NotSupportedException">Always, once the subscripts validate.</exception>
        public static NDArray einsum(string subscripts, params NDArray[] operands)
            => einsum(subscripts, operands, null);

        /// <summary>
        ///     Evaluates the Einstein summation convention, with NumPy's full keyword surface.
        /// </summary>
        /// <param name="out">Where the calculation would be deposited. Its RANK is validated now.</param>
        /// <param name="dtype">Forces the accumulation dtype.</param>
        /// <param name="order">Memory layout of the result — <c>'C'</c>, <c>'F'</c>, <c>'A'</c> or <c>'K'</c>.</param>
        /// <param name="casting">
        ///     Casting rule — <c>"no"</c>, <c>"equiv"</c>, <c>"safe"</c>, <c>"same_kind"</c> or
        ///     <c>"unsafe"</c>.
        /// </param>
        /// <param name="optimize">
        ///     <c>false</c> (the default), <c>true</c>, <c>"greedy"</c> or <c>"optimal"</c>. NumPy
        ///     also takes a precomputed contraction path; that is not modelled, because nothing
        ///     plans one yet.
        /// </param>
        /// <remarks>
        ///     <b>Pass the keywords BY NAME</b> — <c>np.einsum("ij->i", ops, @out: dst)</c>. They are
        ///     keyword-only in NumPy, and naming them is also what keeps them unambiguous here:
        ///     NumSharp converts scalars to <see cref="NDArray"/> implicitly, so a fully positional
        ///     <c>einsum(subscripts, ops, null, null, 'K', "safe", true)</c> matches this overload
        ///     AND the <c>params</c> one, and the compiler rejects the call as ambiguous.
        ///     <inheritdoc cref="einsum(string, NDArray[])"/>
        /// </remarks>
        public static NDArray einsum(string subscripts, NDArray[] operands, NDArray @out = null,
            NPTypeCode? dtype = null, char order = 'K', string casting = "safe", object optimize = null)
        {
            if (subscripts is null)
                throw new ArgumentNullException(nameof(subscripts));

            if (operands is null || operands.Length == 0)
                throw new ValueError(
                    "must specify the einstein sum subscripts string and at least one operand");

            RequireOrder(order);
            RequireCasting(casting);
            RequireOptimize(optimize);

            // Parses AND validates: rank, ellipsis grammar, operand count, output labels, out='s
            // rank, every diagonal and every label extent. Nothing below this line can be reached
            // by an expression NumPy would reject.
            var plan = EinsumSubscripts.Parse(subscripts, operands, @out);

            return operands[0].TensorEngine.Einsum(subscripts, operands, @out, dtype, order, casting, optimize, plan.OutputShape);
        }

        /// <summary>
        ///     Evaluates the Einstein summation convention in NumPy's SUBLIST spelling —
        ///     <c>np.einsum(a, [0,1], b, [1,2], [0,2])</c> — where each operand is followed by its
        ///     axis labels as integers, and a trailing list gives the output.
        /// </summary>
        /// <param name="operands">
        ///     Alternating <see cref="NDArray"/> and subscript list, optionally closed by a lone
        ///     output list. A subscript list is an <c>int[]</c>, or an <c>object[]</c> mixing
        ///     integers with <see cref="Slice.Ellipsis"/> where NumPy writes <c>Ellipsis</c>.
        /// </param>
        /// <remarks>
        ///     Labels are indices into NumPy's <c>einsum_symbols</c>, UPPER case first: 0-25 are
        ///     <c>A-Z</c> and 26-51 are <c>a-z</c>, so <c>[0,1]</c> means <c>"AB"</c> (see the
        ///     encoding note on <see cref="EinsumSubscripts.FromSublists"/>). Anything outside that range raises
        ///     <c>ValueError("subscript is not within the valid range [0, 52)")</c>, and a non-integer
        ///     entry raises <c>TypeError("each subscript must be either an integer or an ellipsis")</c>
        ///     — both NumPy's, verbatim.
        ///     <inheritdoc cref="einsum(string, NDArray[])"/>
        /// </remarks>
        public static NDArray einsum(params object[] operands)
        {
            if (operands is null || operands.Length == 0)
                throw new ValueError(
                    "must specify the einstein sum subscripts string and at least one operand, " +
                    "or at least one operand and its corresponding subscripts list");

            // A leading string is the ordinary spelling reached through this overload — e.g. from a
            // params array built at runtime. Route it rather than treating the string as an operand.
            if (operands[0] is string subscripts)
            {
                var arrays = new NDArray[operands.Length - 1];
                for (int i = 1; i < operands.Length; i++)
                {
                    arrays[i - 1] = operands[i] as NDArray
                                    ?? throw new TypeError(
                                        $"einsum operand {i - 1} must be an NDArray, got {operands[i]?.GetType().Name ?? "null"}");
                }

                return einsum(subscripts, arrays, null);
            }

            string rendered = EinsumSubscripts.FromSublists(operands, out var parsed);
            return einsum(rendered, parsed, null);
        }

        private static void RequireOrder(char order)
        {
            if (order is not ('C' or 'F' or 'A' or 'K'))
                throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
        }

        private static void RequireCasting(string casting)
        {
            if (casting is not ("no" or "equiv" or "safe" or "same_kind" or "unsafe"))
                throw new ValueError(
                    $"casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '{casting}')");
        }

        private static void RequireOptimize(object optimize)
        {
            // NumPy accepts False/True/'greedy'/'optimal' — and silently tolerates any other STRING,
            // falling through to greedy. Anything else is a contraction path, which it rejects
            // unless the list opens with the marker 'einsum_path'.
            if (optimize is null or bool or string)
                return;

            throw new TypeError($"Did not understand the path: {optimize}");
        }
    }
}
