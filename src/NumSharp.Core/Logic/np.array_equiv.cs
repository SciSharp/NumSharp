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
        public static bool array_equiv(NDArray a1, NDArray a2)
        {
            // A null "array" cannot broadcast against or equal a real one; two nulls are trivially
            // equivalent (mirrors the NDArray == null convention).
            if (a1 is null || a2 is null)
                return a1 is null && a2 is null;

            // NumPy attempts multiarray.broadcast(a1, a2) and treats a failure as "not equivalent".
            // Non-broadcastable shapes short-circuit to False before any element comparison.
            if (!np.are_broadcastable(a1, a2))
                return false;

            // The == operator broadcasts the operands to their common shape; all() then reduces to a
            // single bool. `using` reclaims the comparison result (all() only reads it).
            using var eq = a1 == a2;
            return np.all(eq);
        }
    }
}
