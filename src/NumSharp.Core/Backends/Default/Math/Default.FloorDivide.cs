using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray FloorDivide(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();

            // floor_divide has no complex loop, so ANY complex operand with no explicit dtype reaches
            // no loop: NumPy raises the generic ufunc TypeError (NOT the kernel's NotSupportedException)
            // and validates the LOOP, not the data — so a zero-size complex operand is rejected too
            // (probed 2.4.2). Same guard shape as np.fabs; resolves the oracle K4/K5 excuses.
            if (typeCode is null && (lhs.GetTypeCode == NPTypeCode.Complex || rhs.GetTypeCode == NPTypeCode.Complex))
                throw new TypeError(
                    "ufunc 'floor_divide' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            // ufunc dtype=/out=/where= compose exactly like NumPy 2.4.2 (probed):
            // dtype= selects the LOOP — floor_divide(i32,i32,dtype=f64) computes
            // the float loop (-7//2 → -4.0); floor_divide(f64,f64,dtype=i32)
            // raises the same_kind input-cast UFuncTypeError; and with out= the
            // loop value is same_kind-cast into out (floor_divide(i32,i32,
            // out=f32,dtype=f64) lands -4.0f). ExecuteBinaryOp implements the
            // override + validation; no post-cast — NumPy never computes in the
            // promoted dtype and casts afterwards.
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.FloorDivide, @out, where, typeCode);
        }
    }
}

