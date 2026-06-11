using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for finiteness (not infinity and not NaN).
        /// Returns True for all integer types (always finite), and checks float/double values.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <returns>Boolean array where True indicates the element is finite</returns>
        /// <remarks>
        /// NumPy behavior:
        /// - Float/Double: True if not infinity and not NaN
        /// - Integer types: Always True (integers cannot be Inf or NaN)
        /// - Empty arrays: Returns empty bool array
        /// </remarks>
        public override NDArray<bool> IsFinite(NDArray a)
        {
            // Use IL kernel with UnaryOp.IsFinite
            // The kernel handles:
            // - Float/Double: calls float.IsFinite/double.IsFinite
            // - All other types: returns true (integers are always finite)
            var result = ExecuteUnaryOp(a, UnaryOp.IsFinite, NPTypeCode.Boolean);
            return result.MakeGeneric<bool>();
        }

        /// <summary>
        /// ufunc out=/where= overload: rides the shared unary Into-path with a
        /// Boolean loop dtype (the predicate body emits bool at the INPUT
        /// dtype); a non-bool out engages the windowed bool→X flush. Returns
        /// the provided out (no MakeGeneric — out may be any numeric dtype).
        /// </summary>
        public override NDArray IsFinite(NDArray a, NDArray @out, NDArray where = null)
            => ExecuteUnaryOp(a, UnaryOp.IsFinite, NPTypeCode.Boolean, @out, where);
    }
}
