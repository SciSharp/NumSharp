using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    /// <summary>
    /// Bitwise operation dispatch using IL-generated kernels.
    ///
    /// NumPy loop resolution (probed 2.4.2): the bitwise ufuncs have loops for
    /// bool and the integer dtypes ONLY — float/complex/decimal inputs raise
    /// the no-loop TypeError. Validation order pinned by probes
    /// (OUT_WHERE_NPYITER_FAMILIES_PLAN.md §2.7):
    ///   ① where must be bool → ② loop resolution (no-loop TypeError) →
    ///   ③ out same_kind cast → ④ broadcast/shape.
    /// ①② run here; ③④ run inside the shared ufunc Into-path (which
    /// re-validates ① — idempotent).
    /// </summary>
    public partial class DefaultEngine
    {
        /// <summary>
        /// Execute bitwise AND operation.
        /// </summary>
        public override NDArray BitwiseAnd(NDArray lhs, NDArray rhs, NDArray @out = null, NDArray where = null)
        {
            ValidateWhereMask(where);
            ValidateBitwiseLoop(np._FindCommonType(lhs, rhs), "bitwise_and");
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.BitwiseAnd, @out, where);
        }

        /// <summary>
        /// Execute bitwise OR operation.
        /// </summary>
        public override NDArray BitwiseOr(NDArray lhs, NDArray rhs, NDArray @out = null, NDArray where = null)
        {
            ValidateWhereMask(where);
            ValidateBitwiseLoop(np._FindCommonType(lhs, rhs), "bitwise_or");
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.BitwiseOr, @out, where);
        }

        /// <summary>
        /// Execute bitwise XOR operation.
        /// </summary>
        public override NDArray BitwiseXor(NDArray lhs, NDArray rhs, NDArray @out = null, NDArray where = null)
        {
            ValidateWhereMask(where);
            ValidateBitwiseLoop(np._FindCommonType(lhs, rhs), "bitwise_xor");
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.BitwiseXor, @out, where);
        }

        /// <summary>
        /// NumPy loop resolution for the bitwise family: loops exist for bool
        /// and integer dtypes only (Char rides along as a NumSharp integer
        /// extension). Anything else raises NumPy's no-loop TypeError —
        /// verbatim text, doubled quotes around safe included — BEFORE the
        /// out-cast/shape validation (probed order, plan §2.7: a float-input
        /// bitwise call with a bad out still reports the no-loop error).
        /// </summary>
        private static void ValidateBitwiseLoop(NPTypeCode loopType, string ufuncName)
        {
            if (loopType == NPTypeCode.Boolean || NpyExprTypeRules.IsIntegerKind(loopType))
                return;

            throw new TypeError(
                $"ufunc '{ufuncName}' not supported for the input types, and the inputs " +
                "could not be safely coerced to any supported types according to the casting rule ''safe''");
        }
    }
}
