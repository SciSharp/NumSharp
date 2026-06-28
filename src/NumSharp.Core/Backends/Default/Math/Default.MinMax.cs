using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // Element-wise min/max ufuncs, routed through the same IL-generated binary kernel
        // pipeline as Add/Subtract/etc. (broadcasting, NEP50 promotion, out=, where=, all
        // execution paths). maximum/minimum PROPAGATE NaN (NumPy: a NaN operand wins);
        // fmax/fmin IGNORE NaN (the non-NaN operand wins). The op-specific NaN semantics live
        // in DirectILKernelGenerator's EmitScalarOperation / EmitVectorOperation.

        /// <summary>Element-wise maximum, NaN-propagating (np.maximum).</summary>
        public override NDArray Maximum(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Maximum, @out, where, typeCode);
        }

        /// <summary>Element-wise minimum, NaN-propagating (np.minimum).</summary>
        public override NDArray Minimum(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Minimum, @out, where, typeCode);
        }

        /// <summary>Element-wise maximum, NaN-ignoring (np.fmax).</summary>
        public override NDArray FMax(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.FMax, @out, where, typeCode);
        }

        /// <summary>Element-wise minimum, NaN-ignoring (np.fmin).</summary>
        public override NDArray FMin(NDArray lhs, NDArray rhs, NPTypeCode? typeCode = null, NDArray @out = null, NDArray where = null)
        {
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.FMin, @out, where, typeCode);
        }
    }
}
