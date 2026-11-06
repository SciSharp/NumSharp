using System;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     np.mod / np.remainder (one NumPy ufunc, named <c>remainder</c>) — the floored remainder, element-wise: the
        ///     result takes the DIVISOR's sign (Python's <c>%</c>). IL-generated kernels on NumPy's integer and float loops
        ///     only: a bool pair promotes to the int8 loop, and complex has no loop at all.
        /// </summary>
        /// <param name="lhs">The dividends (<c>x1</c>).</param>
        /// <param name="rhs">The divisors (<c>x2</c>); broadcasts against <paramref name="lhs"/>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: the loop the remainder is computed in (each input must reach it by a
        ///     same_kind cast), or null for the NEP 50 promotion of the inputs. A bool or complex request names no loop.</param>
        /// <param name="out">NumPy's <c>out=</c>: joins the broadcast without being stretched and must take the loop dtype by
        ///     a same_kind cast; returned itself.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask; a False position keeps what <paramref name="out"/> held
        ///     (and is uninitialized in a fresh result, as in NumPy).</param>
        /// <returns>The remainders: <paramref name="out"/> itself when given, else a fresh array of the loop dtype.</returns>
        /// <exception cref="IncorrectTypeException">A bool or complex <paramref name="dtype"/>: NumPy's <c>No loop matching
        ///     the specified signature and casting was found for ufunc remainder</c> (the ufunc's name, for np.mod too),
        ///     raised after the out and mask checks and before any input cast or shape check (parity QA-53).</exception>
        /// <exception cref="TypeError">A <paramref name="dtype"/> naming more than a DType class; a complex operand with no
        ///     <paramref name="dtype"/> (<c>ufunc 'remainder' not supported for the input types, …</c>, a zero-size operand
        ///     too); an input <paramref name="dtype"/> cannot take, or a loop dtype <paramref name="out"/> cannot take, by a
        ///     same_kind cast (<c>Cannot cast ufunc 'remainder' input 0 / output …</c>).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/> (<c>output array is read-only</c>, checked
        ///     first).</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>; an out or mask that does not broadcast,
        ///     or an out that would be stretched.</exception>
        /// <exception cref="IncorrectShapeException">The inputs do not broadcast together.</exception>
        public override NDArray Mod(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = UfuncLoopCode(dtype, "remainder", lhs, rhs, @out, where);

            // remainder registers integer and float loops only (no '??->?', no complex): a bool or complex dtype= request
            // selects no loop, and NumPy reports that after the out and mask checks above but before any input cast, out
            // cast or shape check (probed 2.4.2, np.mod quoting the ufunc's own name 'remainder'; parity QA-53).
            if (typeCode is NPTypeCode.Boolean or NPTypeCode.Complex)
                throw new IncorrectTypeException(
                    "No loop matching the specified signature and casting was found for ufunc remainder");

            // mod (NumPy ufunc name 'remainder') has no complex loop, so ANY complex operand with no
            // explicit dtype reaches no loop: NumPy raises the generic ufunc TypeError (NOT the kernel's
            // NotSupportedException) and validates the LOOP, not the data — so a zero-size complex operand
            // is rejected too (probed 2.4.2). Same guard shape as np.fabs; resolves oracle K4/K5.
            if (typeCode is null && (lhs.GetTypeCode == NPTypeCode.Complex || rhs.GetTypeCode == NPTypeCode.Complex))
                throw new TypeError(
                    "ufunc 'remainder' not supported for the input types, and the inputs " +
                    "could not be safely coerced to any supported types according to the casting rule ''safe''");

            return ExecuteBinaryOp(lhs, rhs, BinaryOp.Mod, @out, where, typeCode);
        }
    }
}
