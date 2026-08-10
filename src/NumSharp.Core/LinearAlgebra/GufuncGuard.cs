using NumSharp.Backends;

namespace NumSharp
{
    /// <summary>
    ///     The two rejections every generalized ufunc shares, reproduced verbatim from NumPy's
    ///     <c>_parse_axes</c>/<c>_get_coredim_sizes</c> in <c>numpy/_core/src/umath/ufunc_object.c</c>.
    /// </summary>
    /// <remarks>
    ///     A gufunc's core dimensions are matched EXACTLY — they do not broadcast, so a length-1 core
    ///     dim against a length-3 one is an error, not a stretch. Only the LEADING (loop) axes
    ///     broadcast, and those are left to the ordinary elementwise machinery downstream.
    /// </remarks>
    internal static class GufuncGuard
    {
        /// <summary>
        ///     "<c>{name}: Input operand {i} does not have enough dimensions (has {n}, gufunc core
        ///     with signature {sig} requires {k})</c>"
        /// </summary>
        internal static void RequireRank(string name, string signature, int operandIndex, NDArray operand, int required)
        {
            if (operand.ndim >= required)
                return;

            throw new ValueError(
                $"{name}: Input operand {operandIndex} does not have enough dimensions " +
                $"(has {operand.ndim}, gufunc core with signature {signature} requires {required})");
        }

        /// <summary>
        ///     "<c>{name}: Input operand {i} has a mismatch in its core dimension {d}, with gufunc
        ///     signature {sig} (size {actual} is different from {expected})</c>"
        /// </summary>
        /// <param name="actual">The size this operand carries — reported FIRST, as NumPy does.</param>
        /// <param name="expected">The size an earlier operand already fixed for this core dimension.</param>
        internal static void RequireCoreSize(string name, string signature, int operandIndex, int coreDimension,
            long actual, long expected)
        {
            if (actual == expected)
                return;

            throw new ValueError(
                $"{name}: Input operand {operandIndex} has a mismatch in its core dimension {coreDimension}, " +
                $"with gufunc signature {signature} (size {actual} is different from {expected})");
        }

        /// <summary>
        ///     Applies a gufunc's <c>dtype=</c> (the loop selector) and <c>out=</c> in NumPy's order:
        ///     the loop runs at <paramref name="dtype"/>, and only then is the answer delivered into
        ///     <paramref name="out"/>.
        /// </summary>
        internal static NDArray Deliver(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result;

            np.copyto(@out, result);
            return @out;
        }

        /// <summary>Casts an operand to the requested loop dtype, or returns it unchanged.</summary>
        internal static NDArray ToLoop(NDArray operand, NPTypeCode? dtype)
            => dtype is null ? operand : operand.TensorEngine.Cast(operand, dtype.Value, copy: false);

        /// <summary>
        ///     Validates a gufunc <c>axes=</c> list and normalizes each INPUT entry against its
        ///     operand's rank.
        /// </summary>
        /// <param name="coreRanks">
        ///     How many core dimensions each of the three slots has — <c>{x1, x2, out}</c>. For
        ///     <c>vecdot</c>'s <c>(n),(n)-&gt;()</c> that is <c>{1, 1, 0}</c>.
        /// </param>
        /// <returns>
        ///     Three entries. The two input entries hold non-negative axis indices; the OUTPUT entry
        ///     is returned as given, because the result's rank is not known until the loop has run.
        /// </returns>
        /// <remarks>
        ///     <c>axes</c> names, per operand, which axes carry the core dimensions — so it is the
        ///     general form of <c>axis</c>, and NumPy rejects the two together. The output entry may
        ///     be omitted ONLY when the output has no core axes, which among this family is
        ///     <c>vecdot</c> alone.
        /// </remarks>
        /// <param name="outputCoreRankOverride">
        ///     The output's EFFECTIVE core rank when it differs from the signature's — namely under
        ///     <c>keepdims</c>, where the reduced core dimension is kept as size 1 and a PROVIDED
        ///     output entry must name it. It changes only the length a provided output entry is
        ///     checked against; whether the entry may be OMITTED still keys off the signature rank
        ///     (<c>coreRanks[2]</c>), because keepdims does not add a signature core axis. Only
        ///     <c>vecdot</c> passes it — <c>matvec</c>/<c>vecmat</c> reject keepdims before reaching
        ///     here — so their behaviour is unchanged (<c>null</c> ⇒ the signature rank).
        /// </param>
        internal static int[][] NormalizeAxes(string name, int[][] axes, int[] coreRanks, NDArray x1, NDArray x2,
            int? outputCoreRankOverride = null)
        {
            if (axes.Length != 3 && !(axes.Length == 2 && coreRanks[2] == 0))
                throw new ValueError(
                    "axes should be a list with an entry for all 3 inputs and outputs; entries for " +
                    "outputs can only be omitted if none of them has core axes.");

            var resolved = new int[3][];
            resolved[2] = System.Array.Empty<int>();

            for (int i = 0; i < axes.Length; i++)
            {
                var entry = axes[i] ?? System.Array.Empty<int>();
                int expected = i == 2 ? (outputCoreRankOverride ?? coreRanks[2]) : coreRanks[i];
                if (entry.Length != expected)
                    throw new AxisError(
                        $"{name}: operand {i} has {expected} core dimensions, " +
                        $"but {entry.Length} dimensions are specified by axes tuple.");

                if (i == 2)
                {
                    resolved[2] = (int[])entry.Clone();
                    continue;
                }

                var operand = i == 0 ? x1 : x2;
                var normalized = new int[entry.Length];
                for (int k = 0; k < entry.Length; k++)
                {
                    int axis = entry[k] < 0 ? entry[k] + operand.ndim : entry[k];
                    if (axis < 0 || axis >= operand.ndim)
                        throw new AxisError(entry[k], operand.ndim);
                    normalized[k] = axis;
                }

                resolved[i] = normalized;
            }

            return resolved;
        }

        /// <summary>NumPy rejects <c>axis</c> and <c>axes</c> together, whichever gufunc it is.</summary>
        internal static void RejectAxisWithAxes(int[][] axes, int? axis)
        {
            if (axes is not null && axis is not null)
                throw new TypeError("cannot specify both 'axis' and 'axes'");
        }

        /// <summary>
        ///     The two rejections a gufunc whose core dimensions are DISTINCT raises for
        ///     <c>axis</c> and <c>keepdims</c> — <c>matvec</c> and <c>vecmat</c>, never
        ///     <c>vecdot</c>.
        /// </summary>
        /// <param name="secondOperandCoreRank">
        ///     How many core dimensions input 1 has — 1 for <c>matvec</c>, 2 for <c>vecmat</c>.
        ///     NumPy names that operand in the keepdims message.
        /// </param>
        internal static void RejectAxisAndKeepdims(string name, string signature, int? axis, bool keepdims,
            int secondOperandCoreRank)
        {
            if (axis is not null)
                throw new TypeError(
                    $"{name}: axis can only be used with a single shared core dimension, not with " +
                    $"the 2 distinct ones implied by signature {signature}.");

            if (keepdims)
                throw new TypeError(
                    $"{name} does not support keepdims: its signature {signature} requires input 1 " +
                    $"to have {secondOperandCoreRank} core dimensions, but keepdims can only be used " +
                    "when all inputs have the same number of core dimensions and all outputs have no " +
                    "core dimensions.");
        }
    }
}
