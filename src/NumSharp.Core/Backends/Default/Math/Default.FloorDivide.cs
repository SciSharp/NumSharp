using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        public override NDArray FloorDivide(NDArray lhs, NDArray rhs, System.Type dtype)
            => FloorDivide(lhs, rhs, dtype?.GetTypeCode());

        public override NDArray FloorDivide(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
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

