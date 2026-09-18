namespace NumSharp
{
    public static partial class np
    {
        // NumPy's logical_and/or/xor/not are bool-output ufuncs whose loops are ??->? .. GG->? (and
        // OO->O for object). Two facts drive the C# shape (both probed against NumPy 2.4.2):
        //   • The output dtype is ALWAYS bool, so — like the comparison ufuncs — dtype= is validate-only:
        //     any non-bool request raises "No loop matching the specified signature and casting was found
        //     for ufunc logical_X". bool is safe-castable to every dtype, so a provided out never fails
        //     the same_kind out-cast — which is why composing over the bitwise engine (whose out-cast
        //     error would name "bitwise_and") is fully NumPy-faithful: that error can never fire.
        //   • The inputs are reduced to their truth value (nonzero) first, then combined. A fused
        //     "nonzero(a) OP nonzero(b)" IL kernel would need EmitConvertTo(float->bool), which mixes an
        //     int-0 compare against a float on the IL stack — a landmine — so the truth reduction stays
        //     the proven `x != 0` comparison (NaN is truthy: `nan != 0` is True, matching NumPy) and the
        //     bitwise engine with out=/where= does the combine + masked write in one validated path.
        // Return type is NDArray (not NDArray<bool>) to match the comparison-op shape: a plain call still
        // yields a bool array, but out=<numeric> returns that numeric out (True->1). No caller relied on
        // the NDArray<bool> return (verified), so this is a benign breaking change.

        /// <summary>
        /// bool-only dtype= gate shared by the four logical ufuncs. Their loops emit a bool output, so a
        /// non-bool dtype= has no loop to bind — NumPy raises the verbatim no-loop TypeError (dtype=bool
        /// is a legal no-op). Mirrors <c>DefaultEngine.ValidateBoolLoopDtype</c> at the np layer so the
        /// error names the LOGICAL ufunc rather than the bitwise op the call composes over.
        /// </summary>
        /// <param name="dtype">The requested loop dtype, or null.</param>
        /// <param name="ufuncName">The logical ufunc name for the error text (e.g. "logical_and").</param>
        /// <exception cref="IncorrectTypeException">When <paramref name="dtype"/> is a non-bool dtype.</exception>
        private static void ValidateLogicalDtype(DType dtype, string ufuncName)
        {
            if (dtype is not null && dtype.GetTypeCode() != NPTypeCode.Boolean)
                throw new IncorrectTypeException(
                    $"No loop matching the specified signature and casting was found for ufunc {ufuncName}");
        }

        /// <summary>
        /// Reduce an operand to its element-wise truth value (nonzero) as a bool array — the identity for
        /// a bool input, else the <c>x != 0</c> comparison (NaN is truthy, matching NumPy's logical loops;
        /// a complex value is truthy when either component is nonzero).
        /// </summary>
        /// <param name="x">The operand to reduce.</param>
        /// <returns>A bool array of <paramref name="x"/>'s truth values (or <paramref name="x"/> itself if
        /// already bool).</returns>
        private static NDArray AsTruthOperand(NDArray x)
            => x.typecode == NPTypeCode.Boolean ? x : (x != 0);

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
        /// <returns>The element-wise AND of the operands' truth values (bool, or <paramref name="out"/>).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html</remarks>
        [NDScoped]
        public static NDArray logical_and(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            ValidateLogicalDtype(dtype, "logical_and");
            // Reduce both operands to bool truth values, then AND them through the bitwise engine's
            // out=/where= path (bool & bool == logical and; the masked write + same_kind out cast are
            // handled there). The b1/b2 temps are reclaimed by the [NDScoped] weaver.
            var b1 = AsTruthOperand(x1);
            var b2 = AsTruthOperand(x2);
            return x1.TensorEngine.BitwiseAnd(b1, b2, null, @out, where);
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
        /// <returns>The element-wise OR of the operands' truth values (bool, or <paramref name="out"/>).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html</remarks>
        [NDScoped]
        public static NDArray logical_or(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            ValidateLogicalDtype(dtype, "logical_or");
            var b1 = AsTruthOperand(x1);
            var b2 = AsTruthOperand(x2);
            return x1.TensorEngine.BitwiseOr(b1, b2, null, @out, where);
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
        /// <returns>The element-wise XOR of the operands' truth values (bool, or <paramref name="out"/>).</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html</remarks>
        [NDScoped]
        public static NDArray logical_xor(NDArray x1, NDArray x2, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            ValidateLogicalDtype(dtype, "logical_xor");
            var b1 = AsTruthOperand(x1);
            var b2 = AsTruthOperand(x2);
            return x1.TensorEngine.BitwiseXor(b1, b2, null, @out, where);
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
        [NDScoped]
        public static NDArray logical_not(NDArray x, NDArray @out = null, NDArray where = null, DType dtype = null)
        {
            ValidateLogicalDtype(dtype, "logical_not");
            // A bool input's logical NOT is exactly its bitwise invert (~True == False); route through the
            // invert ufunc so the bool→bool loop is used (np.negative(bool) is a TypeError, so Invert, not
            // Negate). For any other dtype, logical NOT is the truth test `x == 0` (zero→True, nonzero→False;
            // NaN != 0 so logical_not(NaN)→False), which the equality comparison computes with out=/where=.
            if (x.typecode == NPTypeCode.Boolean)
                return x.TensorEngine.Invert(x, null, @out, where);
            return x.TensorEngine.Compare(x, NDArray.Scalar(0), null, @out, where);
        }
    }
}
