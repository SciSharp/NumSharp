using NumSharp.Backends.Kernels;

namespace NumSharp {
    public static partial class np
    {
        /// <summary>
        ///     True if the inputs are shape consistent and all elements are equal.
        /// </summary>
        /// <param name="a1">First input array.</param>
        /// <param name="a2">Second input array.</param>
        /// <returns>True if equivalent, False otherwise.</returns>
        /// <remarks>
        ///     "Shape consistent" means the two shapes are either equal or one can be BROADCAST to the
        ///     other — the key difference from <see cref="array_equal(NDArray,NDArray,bool)"/>, which
        ///     requires an exact shape match. NaN never compares equal (there is no <c>equal_nan</c>
        ///     option, matching NumPy), so <c>array_equiv(a, a)</c> is <c>false</c> whenever <paramref name="a1"/>
        ///     holds a NaN. Empty inputs of a broadcast-consistent shape are equivalent (all() over the
        ///     empty comparison is vacuously true). Mirrors <c>numpy.array_equiv(a1, a2)</c> (NumPy 2.4.2):
        ///     it converts both to arrays, checks they broadcast together, then returns
        ///     <c>bool((a1 == a2).all())</c>.
        ///     https://numpy.org/doc/stable/reference/generated/numpy.array_equiv.html
        /// </remarks>
        /// <exception cref="StringComputationNotSupportedException">Two strings of one class with a non-empty broadcast: NumPy
        ///     compares the text (the next phase of NumSharp's string support). Other string pairs are answered from shape and
        ///     dtype.</exception>
        public static bool array_equiv(NDArray a1, NDArray a2)
        {
            using var plain = NDMatrixBoundary.PlainFor(a1, a2);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // A null "array" cannot broadcast against or equal a real one; two nulls are trivially
            // equivalent (mirrors the NDArray == null convention).
            if (a1 is null || a2 is null)
                return a1 is null && a2 is null;

            // NumPy attempts multiarray.broadcast(a1, a2) and treats a failure as "not equivalent".
            // Non-broadcastable shapes short-circuit to False before any element comparison.
            if (!np.are_broadcastable(a1, a2))
                return false;

            // Text: (a1 == a2).all() — a pair equal has no loop for gets the operator's all-False array, so only an empty
            // broadcast is equivalent; two strings of one class compare their text (the next phase), unless the
            // broadcast is empty.
            if (Backends.Strings.NDStringUfuncs.IsString(a1) || Backends.Strings.NDStringUfuncs.IsString(a2))
            {
                bool empty = Shape.ResolveReturnShape(a1.Shape, a2.Shape).size == 0;
                if (empty || !Backends.Strings.NDStringUfuncs.ComparesText(a1.dtype, a2.dtype))
                    return empty;
                throw Backends.Strings.NDStringUfuncs.NextPhase("array_equiv", Backends.Strings.NDStringUfuncs.IsString(a1) ? a1 : a2);
            }

            // Equal-shape fast path: when the two operands are already the same shape (no broadcast
            // needed), same dtype and dense-contiguous, one early-exiting SIMD pass (EqualityScan)
            // replaces the bool temp + all(). Bit-identical to the composition (Vector.EqualsAll gives
            // NaN != NaN / -0.0 == +0.0). A genuine broadcast (different shapes) returns false here and
            // takes the composition below, where the == operator stretches the size-1 axes.
            if (EqualityScan.TryAllEqual(a1, a2, out bool fused))
                return fused;

            // The == operator broadcasts the operands to their common shape; all() then reduces to a
            // single bool. `using` reclaims the comparison result (all() only reads it).
            using var eq = a1 == a2;
            return np.all(eq);
        }
    }
}
