using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise C-style remainder (np.fmod) using IL-generated kernels.
        ///
        /// Unlike <see cref="Mod"/> (floored — result has the sign of the DIVISOR, Python's %),
        /// fmod uses truncated division so the result has the sign of the DIVIDEND (C fmod):
        /// fmod(-7, 3) == -1 where mod(-7, 3) == 2. Routes through the same <see cref="ExecuteBinaryOp"/>
        /// machinery as Mod/FloorDivide — same NEP50 promotion (integer stays integer, bool -> int8),
        /// the scalar-only per-element kernel (EmitFmodOperation), and full out=/where=/dtype= support.
        /// </summary>
        public override NDArray Fmod(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();

            // fmod has no complex loop, so ANY complex operand with no explicit dtype reaches no
            // loop: NumPy raises the generic ufunc TypeError (NOT the kernel's
            // NotSupportedException) and validates the LOOP, not the data — so a zero-size complex
            // operand is rejected too (probed 2.4.2). Same guard shape as Mod/FloorDivide/np.fabs;
            // fmod landed after the K4/K5 sweep and missed the guard until the errors_full
            // regeneration surfaced its 29 complex cells (2026-09-18).
            if (typeCode is null && (lhs.GetTypeCode == NPTypeCode.Complex || rhs.GetTypeCode == NPTypeCode.Complex))
                throw new TypeError(
                    "ufunc 'fmod' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to " +
                    "the casting rule ''safe''");

            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Fmod, @out, where, typeCode);
        }
    }
}
