using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray Rint(NDArray nd, Type dtype) => Rint(nd, dtype?.GetTypeCode());

        /// <summary>
        /// Element-wise round-half-to-even (NumPy <c>np.rint</c>) using IL-generated kernels.
        ///
        /// rint is a FLOAT-tier unary ufunc: it has no integer/bool loops (only e/f/d/g and the
        /// complex loops in NumPy), so integer/bool inputs promote to the smallest matching float
        /// — bool/int8/uint8 -> float16, int16/uint16/char -> float32, int32/uint32/int64/uint64 ->
        /// float64, floats/complex/decimal preserved (<see cref="ResolveUnaryFloatReturnType"/>).
        /// This is the dtype behavior of sqrt/sin, NOT of np.round/around (which keep integer dtype).
        ///
        /// The VALUE computation is round-half-to-even == <see cref="UnaryOp.Round"/>'s existing
        /// kernel (Math.Round/MathF.Round default to MidpointRounding.ToEven; complex rounds real
        /// and imag separately; decimal uses decimal.Round). Reusing that op means no new kernel:
        /// UfuncName(UnaryOp.Round) is already "rint", so out=/dtype= cast errors name 'rint'.
        ///
        /// out=/where=/dtype= compose exactly like the other unary ufuncs (probed 2.4.2): dtype=
        /// selects the loop (must be a float/complex loop — an integer dtype raises the verbatim
        /// "No loop matching the specified signature ..."), inputs reach the loop via a same_kind
        /// cast, and a provided out receives a same_kind-cast write-back.
        /// </summary>
        public override NDArray Rint(NDArray nd, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            return ExecuteUnaryOp(nd, UnaryOp.Round, ResolveUnaryFloatReturnType(nd, typeCode, "rint"), @out, where);
        }
    }
}
