using System;
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
        public override NDArray IsFinite(NDArray a, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            // typeCode is validate-only: isfinite has bool-output loops only
            // (NumPy: dtype=bool is a no-op, anything else raises no-loop).
            ValidateBoolLoopDtype(typeCode, "isfinite");

            // Plain call: keep the typed NDArray<bool> instance (TensorEngine
            // contract). The IL kernel handles:
            // - Float/Double: calls float.IsFinite/double.IsFinite
            // - All other types: returns true (integers are always finite)
            //
            // AsGeneric (NOT MakeGeneric): the engine result is a freshly-produced
            // array we own, so we wrap its storage in place rather than aliasing it
            // — the same cheap bool-return the sibling comparison ops use. See
            // Default.IsNan.cs for the measured per-call saving.
            if (@out is null && where is null)
            {
                using var result = ExecuteUnaryOp(a, UnaryOp.IsFinite, NPTypeCode.Boolean);
                return result.AsGeneric<bool>();
            }

            // ufunc out=/where=: rides the shared unary Into-path with a
            // Boolean loop dtype (the predicate body emits bool at the INPUT
            // dtype); a non-bool out engages the windowed bool→X flush. Returns
            // the provided out (no MakeGeneric — out may be any numeric dtype).
            return ExecuteUnaryOp(a, UnaryOp.IsFinite, NPTypeCode.Boolean, @out, where);
        }
    }
}
