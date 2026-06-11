using NumSharp.Backends.Kernels;
using NumSharp.Generic;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Test element-wise for positive or negative infinity.
        /// </summary>
        /// <param name="a">Input array</param>
        /// <returns>Boolean array where True indicates the element is +/-Inf</returns>
        /// <remarks>
        /// NumPy behavior:
        /// - Float/Double/Half: True if value is +Inf or -Inf
        /// - Complex: True if either real or imaginary part is Inf
        /// - Integer types: Always False (integers cannot be Inf)
        /// - NaN: Returns False (NaN is not infinity)
        /// - Empty arrays: Returns empty bool array
        /// </remarks>
        public override NDArray<bool> IsInf(NDArray a)
        {
            var result = ExecuteUnaryOp(a, UnaryOp.IsInf, NPTypeCode.Boolean);
            return result.MakeGeneric<bool>();
        }

        /// <summary>
        /// ufunc out=/where= overload: rides the shared unary Into-path with a
        /// Boolean loop dtype (the predicate body emits bool at the INPUT
        /// dtype); a non-bool out engages the windowed bool→X flush. Returns
        /// the provided out (no MakeGeneric — out may be any numeric dtype).
        /// </summary>
        public override NDArray IsInf(NDArray a, NDArray @out, NDArray where = null)
            => ExecuteUnaryOp(a, UnaryOp.IsInf, NPTypeCode.Boolean, @out, where);
    }
}
