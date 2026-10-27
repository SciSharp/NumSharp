namespace NumSharp
{
    public static partial class np
    {
        // NumPy's logical_and/or/xor/not are C ufuncs, so their calls are TensorEngine members (LogicalAnd / LogicalOr /
        // LogicalXor / LogicalNot, Backends/Default/Logic/Default.Logical.cs: the text guard, the bool-only dtype= gate,
        // the truth reduction and the bitwise combine). These facades keep what sits above the engine: the numpy.matrix
        // boundary and the numpy.char.chararray boundary, which must see the operands' classes before any loop runs.
        // Return type is NDArray (not NDArray<bool>) to match the comparison-op shape: a plain call still yields a bool
        // array, but out=<numeric> returns that numeric out (True->1).

        /// <summary>
        /// Compute the truth value of x1 AND x2 element-wise.
        /// Mirrors NumPy's ufunc signature: <c>logical_and(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Input array (any dtype; reduced to its truth value).</param>
        /// <param name="x2">Input array (any dtype; reduced to its truth value).</param>
        /// <param name="out">A location into which the result is stored; any numeric dtype (bool casts
        /// same_kind to all of them, True→1); returned as-is. The result shape is set by broadcasting.</param>
        /// <param name="where">Boolean mask: only mask-true elements are written; masked-off out slots keep
        /// prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): logical ufuncs have bool-output loops only, so
        /// any non-bool request raises the no-loop TypeError.</param>
        /// <returns>The element-wise AND of the operands' truth values (bool, or <paramref name="out"/>).
        ///     A <c>numpy.char.chararray</c> input gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>), a plain array of a result of another dtype; a chararray
        ///     <paramref name="out"/> comes back as a new chararray view of it.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A short <c>numpy.char.chararray</c> input (0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements) that NumPy's trivial loop copies into a loop dtype
        ///     that is not string data — bool for the logical ufuncs, StringDType beside a
        ///     StringDType operand — without a <paramref name="where"/> mask: the finalize error
        ///     (<c>Can only create a chararray from string data.</c>).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (NumPy's no-loop text).</exception>
        /// <exception cref="TypeError">A detailed or non-legacy <paramref name="dtype"/> descriptor, or a text call NumPy's
        ///     string loop refuses.</exception>
        /// <exception cref="SystemError">A <c>U</c> operand holding a unit above U+10FFFF (NumPy's bool cast of the
        ///     text).</exception>
        /// <exception cref="System.ArgumentException">A <paramref name="where"/> that is not bool, an
        ///     <paramref name="out"/> the bool result cannot be cast into, or a mask or <paramref name="out"/> that does
        ///     not broadcast with the operands.</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together.</exception>
        public static NDArray logical_and(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x1, x2))
                return NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => logical_and(s.x1, s.x2, s.@out, s.where, s.dtype), @out);
            // a numpy.char.chararray operand or out=: chararray.__array_wrap__'s view of a text result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(x1, x2, @out))
                return NDCharArrayBoundary.WrappedUfunc((x1, x2, @out, where, dtype), static s => logical_and(s.x1, s.x2, s.@out, s.where, s.dtype), x1, x2, @out);
            return x1.TensorEngine.LogicalAnd(x1, x2, dtype, @out, where);
        }

        /// <summary>
        /// Compute the truth value of x1 OR x2 element-wise.
        /// Mirrors NumPy's ufunc signature: <c>logical_or(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Input array (any dtype; reduced to its truth value).</param>
        /// <param name="x2">Input array (any dtype; reduced to its truth value).</param>
        /// <param name="out">A location into which the result is stored; any numeric dtype; returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are written; masked-off out slots keep
        /// prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): non-bool raises the no-loop TypeError.</param>
        /// <returns>The element-wise OR of the operands' truth values (bool, or <paramref name="out"/>).
        ///     A <c>numpy.char.chararray</c> input gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>), a plain array of a result of another dtype; a chararray
        ///     <paramref name="out"/> comes back as a new chararray view of it.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A short <c>numpy.char.chararray</c> input (0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements) that NumPy's trivial loop copies into a loop dtype
        ///     that is not string data — bool for the logical ufuncs, StringDType beside a
        ///     StringDType operand — without a <paramref name="where"/> mask: the finalize error
        ///     (<c>Can only create a chararray from string data.</c>).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (NumPy's no-loop text).</exception>
        /// <exception cref="TypeError">A detailed or non-legacy <paramref name="dtype"/> descriptor, or a text call NumPy's
        ///     string loop refuses.</exception>
        /// <exception cref="SystemError">A <c>U</c> operand holding a unit above U+10FFFF.</exception>
        /// <exception cref="System.ArgumentException">A <paramref name="where"/> that is not bool, an
        ///     <paramref name="out"/> the bool result cannot be cast into, or a mask or <paramref name="out"/> that does
        ///     not broadcast with the operands.</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together.</exception>
        public static NDArray logical_or(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x1, x2))
                return NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => logical_or(s.x1, s.x2, s.@out, s.where, s.dtype), @out);
            // a numpy.char.chararray operand or out=: chararray.__array_wrap__'s view of a text result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(x1, x2, @out))
                return NDCharArrayBoundary.WrappedUfunc((x1, x2, @out, where, dtype), static s => logical_or(s.x1, s.x2, s.@out, s.where, s.dtype), x1, x2, @out);
            return x1.TensorEngine.LogicalOr(x1, x2, dtype, @out, where);
        }

        /// <summary>
        /// Compute the truth value of x1 XOR x2 element-wise.
        /// Mirrors NumPy's ufunc signature: <c>logical_xor(x1, x2, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x1">Input array (any dtype; reduced to its truth value).</param>
        /// <param name="x2">Input array (any dtype; reduced to its truth value).</param>
        /// <param name="out">A location into which the result is stored; any numeric dtype; returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are written; masked-off out slots keep
        /// prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): non-bool raises the no-loop TypeError.</param>
        /// <returns>The element-wise XOR of the operands' truth values (bool, or <paramref name="out"/>).
        ///     A <c>numpy.char.chararray</c> input gives a chararray VIEW of a text result (NumPy's
        ///     <c>__array_wrap__</c>), a plain array of a result of another dtype; a chararray
        ///     <paramref name="out"/> comes back as a new chararray view of it.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A short <c>numpy.char.chararray</c> input (0-d, or 1-d with at most
        ///     <c>np.getbufsize()</c> elements) that NumPy's trivial loop copies into a loop dtype
        ///     that is not string data — bool for the logical ufuncs, StringDType beside a
        ///     StringDType operand — without a <paramref name="where"/> mask: the finalize error
        ///     (<c>Can only create a chararray from string data.</c>).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (NumPy's no-loop text).</exception>
        /// <exception cref="TypeError">A detailed or non-legacy <paramref name="dtype"/> descriptor, or a text call NumPy's
        ///     string loop refuses.</exception>
        /// <exception cref="SystemError">A <c>U</c> operand holding a unit above U+10FFFF.</exception>
        /// <exception cref="System.ArgumentException">A <paramref name="where"/> that is not bool, an
        ///     <paramref name="out"/> the bool result cannot be cast into, or a mask or <paramref name="out"/> that does
        ///     not broadcast with the operands.</exception>
        /// <exception cref="IncorrectShapeException">Operands that do not broadcast together.</exception>
        public static NDArray logical_xor(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x1, x2))
                return NDMatrixBoundary.Wrapped((x1, x2, @out, where, dtype), static s => logical_xor(s.x1, s.x2, s.@out, s.where, s.dtype), @out);
            // a numpy.char.chararray operand or out=: chararray.__array_wrap__'s view of a text result, as from NumPy (NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(x1, x2, @out))
                return NDCharArrayBoundary.WrappedUfunc((x1, x2, @out, where, dtype), static s => logical_xor(s.x1, s.x2, s.@out, s.where, s.dtype), x1, x2, @out);
            return x1.TensorEngine.LogicalXor(x1, x2, dtype, @out, where);
        }

        /// <summary>
        /// Compute the truth value of NOT x element-wise.
        /// Mirrors NumPy's ufunc signature: <c>logical_not(x, /, out=None, *, where=True, dtype=None)</c>.
        /// </summary>
        /// <param name="x">Input array (any dtype). Logical NOT is applied to its truth value: nonzero→False,
        /// zero→True. A NaN is truthy (so logical_not(NaN)→False, matching <c>x == 0</c>).</param>
        /// <param name="out">A location into which the result is stored; any numeric dtype; returned as-is.</param>
        /// <param name="where">Boolean mask: only mask-true elements are written; masked-off out slots keep
        /// prior contents.</param>
        /// <param name="dtype">Validate-only (NumPy parity): non-bool raises the no-loop TypeError.</param>
        /// <returns>The element-wise NOT of <paramref name="x"/>'s truth values (bool, or <paramref name="out"/>).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html</remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        /// <exception cref="ValueError">A read-only <paramref name="out"/>.</exception>
        /// <exception cref="IncorrectTypeException">A non-bool <paramref name="dtype"/> (NumPy's no-loop text).</exception>
        /// <exception cref="TypeError">A text operand (no string loop in NumPy), or a detailed or non-legacy
        ///     <paramref name="dtype"/> descriptor.</exception>
        /// <exception cref="System.ArgumentException">A <paramref name="where"/> that is not bool, an
        ///     <paramref name="out"/> the bool result cannot be cast into, a fixed-width text operand under
        ///     <c>dtype=bool</c> (NumPy's input-cast refusal), or a mask or <paramref name="out"/> that does not broadcast
        ///     with the operand.</exception>
        public static NDArray logical_not(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(x))
                return NDMatrixBoundary.Wrapped((x, @out, where, dtype), static s => logical_not(s.x, s.@out, s.where, s.dtype), @out);
            return x.TensorEngine.LogicalNot(x, dtype, @out, where);
        }
    }
}
