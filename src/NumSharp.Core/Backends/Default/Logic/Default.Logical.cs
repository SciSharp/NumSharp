namespace NumSharp.Backends
{
    public partial class DefaultEngine
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
        // yields a bool array, but out=<numeric> returns that numeric out (True->1).
        //
        // np.logical_and / or / xor / not (Logic/np.logical.cs) keep the numpy.matrix and numpy.char.chararray
        // boundaries and call these members; the bodies moved here from those facades unchanged (np-layer dtype
        // plan, P2), as NumPy's loops are C.

        /// <summary>
        ///     The bool-only <c>dtype=</c> gate of the four logical ufuncs. Their loops emit a bool output, so a non-bool
        ///     <c>dtype=</c> has no loop to bind and NumPy raises its no-loop TypeError (<c>dtype=bool</c> is a legal
        ///     no-op). Unlike <see cref="ValidateBoolLoopDtype"/> it has no StringDType branch: every caller runs it after
        ///     the text guard, so no text operand reaches it.
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
        ///     Reduces an operand to its element-wise truth value (nonzero) as a bool array: the operand itself for a bool
        ///     input, else the <c>x != 0</c> comparison (NaN is truthy, matching NumPy's logical loops; a complex value is
        ///     truthy when either component is nonzero).
        /// </summary>
        /// <param name="x">The operand to reduce (numeric: the text guard handles text before this runs).</param>
        /// <returns>A bool array of <paramref name="x"/>'s truth values, or <paramref name="x"/> itself if already bool
        ///     (a caller must not dispose what it did not allocate: the caller's scope owns only a new array).</returns>
        private static NDArray AsTruthOperand(NDArray x)
            => x.typecode == NPTypeCode.Boolean ? x : (x != 0);

        /// <inheritdoc/>
        [NDScoped]
        public override NDArray LogicalAnd(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // logical_* read a string's truth value (non-empty is True) through NumPy's bool cast: the guard computes the
            // call over the operands' truth arrays and returns it (an empty broadcast reads nothing: NumPy's empty bool
            // result, or out=). A text out= stays deferred (it receives formatted 'True' / 'False').
            if (Strings.NDStringUfuncs.BinaryGuard("logical_and", lhs, rhs, @out, UfuncLoopCode(dtype, "logical_and", lhs, rhs, @out, where), where) is { } textResult)
                return textResult;
            ValidateLogicalDtype(dtype, "logical_and");
            // Reduce both operands to bool truth values, then AND them through the bitwise engine's
            // out=/where= path (bool & bool == logical and; the masked write + same_kind out cast are
            // handled there). The b1/b2 temps are reclaimed by the [NDScoped] weaver.
            var b1 = AsTruthOperand(lhs);
            var b2 = AsTruthOperand(rhs);
            return BitwiseAnd(b1, b2, null, @out, where);
        }

        /// <inheritdoc/>
        [NDScoped]
        public override NDArray LogicalOr(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // The text guard as in LogicalAnd: a string operand computes over the truth arrays.
            if (Strings.NDStringUfuncs.BinaryGuard("logical_or", lhs, rhs, @out, UfuncLoopCode(dtype, "logical_or", lhs, rhs, @out, where), where) is { } textResult)
                return textResult;
            ValidateLogicalDtype(dtype, "logical_or");
            var b1 = AsTruthOperand(lhs);
            var b2 = AsTruthOperand(rhs);
            return BitwiseOr(b1, b2, null, @out, where);
        }

        /// <inheritdoc/>
        [NDScoped]
        public override NDArray LogicalXor(NDArray lhs, NDArray rhs, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // The text guard as in LogicalAnd: a string operand computes over the truth arrays.
            if (Strings.NDStringUfuncs.BinaryGuard("logical_xor", lhs, rhs, @out, UfuncLoopCode(dtype, "logical_xor", lhs, rhs, @out, where), where) is { } textResult)
                return textResult;
            ValidateLogicalDtype(dtype, "logical_xor");
            var b1 = AsTruthOperand(lhs);
            var b2 = AsTruthOperand(rhs);
            return BitwiseXor(b1, b2, null, @out, where);
        }

        /// <inheritdoc/>
        [NDScoped]
        public override NDArray LogicalNot(NDArray nd, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            // The dtype= class check, then for text the out= / where= parse and a string request's no-loop text; logical_not
            // has no string loop in NumPy (unlike logical_and/or/xor). The text guard gets the loop request and out= /
            // where= so it answers as NumPy does: the parse errors first, then the loop (a StringDType operand's no-loop text
            // names the requested output class; a fixed-width one meets logical_not's bool-only loops, see
            // NDStringUfuncs.UnaryError).
            var loop = UfuncLoopCode(dtype, "logical_not", nd, null, @out, where);
            Strings.NDStringUfuncs.ThrowIfUnary("logical_not", nd, loop, @out, where);
            ValidateLogicalDtype(dtype, "logical_not");
            // A bool input's logical NOT is exactly its bitwise invert (~True == False); route through the
            // invert ufunc so the bool→bool loop is used (np.negative(bool) is a TypeError, so Invert, not
            // Negate). For any other dtype, logical NOT is the truth test `x == 0` (zero→True, nonzero→False;
            // NaN != 0 so logical_not(NaN)→False), which the equality comparison computes with out=/where=.
            if (nd.typecode == NPTypeCode.Boolean)
                return Invert(nd, null, @out, where);
            return Compare(nd, NDArray.Scalar(0), null, @out, where);
        }
    }
}
