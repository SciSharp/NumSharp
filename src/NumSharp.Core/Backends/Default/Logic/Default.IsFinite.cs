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
    }
}
