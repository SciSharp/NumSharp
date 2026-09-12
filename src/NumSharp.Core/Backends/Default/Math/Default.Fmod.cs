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
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Fmod, @out, where, typeCode);
        }
    }
}
