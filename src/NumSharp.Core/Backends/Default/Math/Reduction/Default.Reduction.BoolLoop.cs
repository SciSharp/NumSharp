namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // The bool LOOP of the add / multiply reductions and accumulations (a dtype=bool request)
        // =============================================================================
        //
        // NumPy registers bool loops for add and multiply that are not arithmetic: BOOL_add is logical_or and
        // BOOL_multiply is logical_and (loops_logical / loops.c.src). So np.cumsum(x, dtype=bool) is "any element so far is
        // true", np.cumprod(x, dtype=bool) "every element so far is true", np.prod(x, dtype=bool) "every element is true"
        // (np.sum(x, dtype=bool), "any", already runs as the add reduction's cast loop). The input reaches the loop through
        // its cast to bool, NumPy's truth value of each element: a NaN is true, a complex value is true when either part is
        // nonzero, a text is parsed by the str -> bool cast. The engine's scan and product kernels have no bool loop, and
        // computing in the input's own dtype first gets those truth values wrong, so these routes cast the input to bool
        // first, then run the exact int64 count (add) or product (multiply) of the zeros and ones and read it back as bool:
        // a running count of ones never passes the element count, a product of zeros and ones stays zero or one.
        // =============================================================================

        /// <summary>
        ///     <c>np.cumsum</c> / <c>np.cumprod</c> under a bool loop (<c>dtype=bool</c>): the running logical or (add) or logical
        ///     and (multiply) of the input's truth values, along <paramref name="axis_"/> or the flattened input.
        /// </summary>
        /// <param name="arr">The input (numbers or bool; a text input arrives here already parsed to bool).</param>
        /// <param name="axis_">The accumulated axis, or null for the flattened input.</param>
        /// <param name="logicalAnd">True for multiply's logical and, false for add's logical or.</param>
        /// <returns>A fresh bool array: the input's shape with an axis, 1-D for the flattened form.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range (the int64 scan's own check).</exception>
        /// <remarks>
        ///     The layout is the int64 scan's (the add scan keeps the input's memory order; the multiply scan allocates C order,
        ///     NumSharp's open gap MX02 for every cumprod), read back as bool in that layout.
        /// </remarks>
        private NDArray LogicalAccumulate(NDArray arr, int? axis_, bool logicalAnd)
        {
            NDArray truth = arr.typecode == NPTypeCode.Boolean ? arr : arr.astype(np.@bool, true, 'K', "unsafe");
            try
            {
                // The bool input takes the scans' own bool -> int64 route (True as 1, False as 0).
                using NDArray counts = logicalAnd ? ReduceCumMul(truth, axis_, np.int64) : ReduceCumAdd(truth, axis_, np.int64);
                return counts.astype(np.@bool, true, 'K', "unsafe");
            }
            finally
            {
                // The cast is this call's temporary; a bool input is the caller's own array.
                if (!ReferenceEquals(truth, arr))
                    truth.Dispose();
            }
        }

        /// <summary>
        ///     <c>np.prod</c> under a bool loop (<c>dtype=bool</c>): whether every one of the input's truth values is true, over
        ///     <paramref name="axis_"/> or every axis (an empty reduction is True, multiply's identity).
        /// </summary>
        /// <param name="arr">The input (numbers or bool; a text input arrives here already parsed to bool).</param>
        /// <param name="axis_">The reduced axis, or null for every axis.</param>
        /// <param name="keepdims">Whether the reduced axes stay with length one.</param>
        /// <returns>The bool result: a read-only 0-d scalar for a full reduction (as every product), else an array.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range (the int64 product's own check).</exception>
        private NDArray LogicalProduct(NDArray arr, int? axis_, bool keepdims)
        {
            NDArray truth = arr.typecode == NPTypeCode.Boolean ? arr : arr.astype(np.@bool, true, 'K', "unsafe");
            try
            {
                using NDArray product = ReduceProduct(truth, axis_, keepdims, np.int64);
                NDArray result = product.astype(np.@bool, true, 'K', "unsafe");
                // A full reduction answers with NumPy's scalar, NumSharp's read-only 0-d array.
                return axis_ is null ? result.MarkReductionScalar() : result;
            }
            finally
            {
                // The cast is this call's temporary; a bool input is the caller's own array.
                if (!ReferenceEquals(truth, arr))
                    truth.Dispose();
            }
        }
    }
}
