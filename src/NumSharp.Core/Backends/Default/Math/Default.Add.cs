using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// Element-wise addition using IL-generated kernels.
        /// Supports all 144 type combinations with automatic type promotion.
        /// </summary>
        /// <param name="lhs">The first operand.</param>
        /// <param name="rhs">The second operand.</param>
        /// <param name="dtype">The <c>dtype=</c> loop request, or null. A string request over numbers (or a number and a
        ///     string of another class) formats the numbers as text and concatenates
        ///     (<see cref="Strings.NDStringUfuncs.AddUnderStringRequest"/>).</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>The sums (or concatenations), or <paramref name="out"/> when one is given.</returns>
        /// <exception cref="TypeError">A sized string request, two string classes, a concatenation wider than 2^31 - 1
        ///     bytes, or no loop for the operands (NumPy's texts).</exception>
        /// <exception cref="ArgumentException">A cast the loop refuses, a non-bool mask, or operands that do not broadcast
        ///     (NumPy's texts).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        public override NDArray Add(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // A string dtype= request over numbers (or a number and a string): NumPy's string add promoter casts the inputs
            // to the requested class and concatenates; the numeric engine never sees the call.
            if (Strings.NDStringUfuncs.AddUnderStringRequest(dtype, lhs, rhs, @out, where) is { } text)
                return text;
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "add", lhs, rhs, @out, where);
            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Add, @out, where, typeCode);
        }
    }
}
