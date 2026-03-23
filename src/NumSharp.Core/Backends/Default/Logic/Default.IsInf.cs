using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for positive or negative infinity.
        /// Returns False for all integer types (cannot be Inf), and checks float/double values.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <returns>Boolean array where True indicates the element is positive or negative infinity</returns>
        /// <remarks>
        /// NumPy behavior:
        /// - Float/Double: True if value is +Inf or -Inf
        /// - Integer types: Always False (integers cannot be Inf)
        /// - NaN: Returns False (NaN is not infinity)
        /// - Empty arrays: Returns empty bool array
        /// </remarks>
        public override NDArray<bool> IsInf(NDArray a)
        {
            // Use IL kernel with UnaryOp.IsInf
            // The kernel handles:
            // - Float/Double: calls float.IsInfinity/double.IsInfinity
            // - All other types: returns false (integers cannot be Inf)
            var result = ExecuteUnaryOp(a, UnaryOp.IsInf, NPTypeCode.Boolean);
            return result.MakeGeneric<bool>();
        }
    }
}
