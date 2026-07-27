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
    }
}
