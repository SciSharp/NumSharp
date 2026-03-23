using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for NaN (Not a Number).
        /// Returns False for all integer types (cannot be NaN), and checks float/double values.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <returns>Boolean array where True indicates the element is NaN</returns>
        /// <remarks>
        /// NumPy behavior:
        /// - Float/Double: True if value is NaN
        /// - Integer types: Always False (integers cannot be NaN)
        /// - Infinity: Returns False (Inf is not NaN)
        /// - Empty arrays: Returns empty bool array
        /// </remarks>
        public override NDArray<bool> IsNan(NDArray a)
        {
            // Use IL kernel with UnaryOp.IsNan
            // The kernel handles:
            // - Float/Double: calls float.IsNaN/double.IsNaN
            // - All other types: returns false (integers cannot be NaN)
            var result = ExecuteUnaryOp(a, UnaryOp.IsNan, NPTypeCode.Boolean);
            return result.MakeGeneric<bool>();
        }
    }
}
