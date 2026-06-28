using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise division using IL-generated kernels.
        /// Supports all 144 type combinations with automatic type promotion.
        /// </summary>
        public override NDArray Divide(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            // NumPy true_divide has float/complex loops ONLY: an integer/bool
            // dtype= request has no loop to select (probed 2.4.2:
            // divide(i32,i32,dtype=i32) and divide(f64,f64,dtype=bool) raise
            // the no-loop TypeError; dtype=f32 runs the float32 loop).
            if (typeCode.HasValue && typeCode.Value < NPTypeCode.Single)
                throw new IncorrectTypeException(
                    "No loop matching the specified signature and casting was found for ufunc divide");

            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Divide, @out, where, typeCode);
        }
    }
}
