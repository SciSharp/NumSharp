using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise subtraction using IL-generated kernels.
        /// Supports all 144 type combinations with automatic type promotion.
        /// </summary>
        /// <param name="lhs">The minuend.</param>
        /// <param name="rhs">The subtrahend.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: the loop to compute in, or null to resolve it from the operands.</param>
        /// <param name="out">NumPy's <c>out=</c>, or null for a new array.</param>
        /// <param name="where">NumPy's <c>where=</c> mask, or null for every element.</param>
        /// <returns>The difference (<paramref name="out"/> itself when given).</returns>
        /// <exception cref="TypeError">The loop resolves to bool: two bool operands without a <paramref name="dtype"/> (an
        ///     <paramref name="out"/> does not choose the loop), or <c>dtype=bool</c> (NumPy's boolean-subtract text).</exception>
        /// <remarks>Two bool operands with a non-bool <paramref name="dtype"/> subtract in that loop, as NumPy's do
        ///     (<c>np.subtract(a, b, dtype=float)</c> is <c>[1., -1., 0.]</c>): np.linspace subtracts bool ends that way.</remarks>
        public override NDArray Subtract(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "subtract", lhs, rhs, @out, where);
            // NumPy's PyUFunc_SubtractionTypeResolver raises after the loop is resolved, when its first operand type is
            // bool: there is no subtract loop for bool. Two bool operands resolve to it unless dtype= names another loop
            // (np.subtract(bools, bools, dtype=float) computes in float64); mixed bool + numeric promotes to the numeric
            // type and subtracts fine. An out= array never chooses the loop, so it does not lift the refusal.
            if (lhs.GetTypeCode == NPTypeCode.Boolean && rhs.GetTypeCode == NPTypeCode.Boolean
                && (typeCode is null || typeCode == NPTypeCode.Boolean))
                throw new TypeError(
                    "numpy boolean subtract, the `-` operator, is not supported, " +
                    "use the bitwise_xor, the `^` operator, or the logical_xor function instead.");

            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Subtract, @out, where, typeCode);
        }
    }
}
