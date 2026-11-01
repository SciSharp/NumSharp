using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;

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
        //
        // The composition makes two bool temporaries and three passes, where NumPy's loop makes one: at 1,000 float64
        // elements it took 2.4 us against NumPy's 1.8 (measured, two pinned P-cores), almost all of it the two
        // comparisons' and the combine's fixed costs. TryLogicalRun serves the common call in one pass: contiguous
        // operands of one shape (or an operand of one value) and a fresh result, or a bool out= of that layout. It
        // gives the composition's bytes by construction (the same truth values, combined by the same byte
        // operation); every other call keeps the composition.

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

        /// <summary>The elements of one block of truth values: two blocks of this size live on the stack (8 KB).</summary>
        private const int LogicalRunBlock = 4096;

        /// <summary>
        ///     Test hook: true makes <see cref="TryLogicalRun"/> decline every call, so a test can hold the route to the
        ///     composition it replaces in one process. Thread-static: it reaches only the test's own thread, so parallel tests
        ///     never see it.
        /// </summary>
        [ThreadStatic] internal static bool DisableLogicalRun;

        /// <summary>
        ///     Test hook: the calls <see cref="TryLogicalRun"/> served on this thread, so a test can prove the route engaged
        ///     (a declined call leaves it unchanged). Thread-static, like <see cref="DisableLogicalRun"/>.
        /// </summary>
        [ThreadStatic] internal static long LogicalRuns;

        /// <summary>
        ///     logical_and / or / xor in one pass, for the common call: each block of elements is reduced to its truth values
        ///     on the stack (the comparison engine's <c>x != 0</c> run) and the two blocks are combined into the result by the
        ///     bitwise ufunc's own kernel (bool &amp; bool, | or ^ over contiguous bytes). The composition the callers fall back
        ///     to makes two bool temporaries and three passes over the elements.
        /// </summary>
        /// <param name="lhs">The first operand (numeric or bool: the text guard answered text before this runs).</param>
        /// <param name="rhs">The second operand.</param>
        /// <param name="op">How two truth values combine: <see cref="BinaryOp.BitwiseAnd"/>, <see cref="BinaryOp.BitwiseOr"/> or
        ///     <see cref="BinaryOp.BitwiseXor"/>.</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>
        ///     The result (a fresh bool array, or <paramref name="out"/>), or null when the call is not one this route serves
        ///     and the caller must run the composition, which serves every call. Served: no mask; operands that are not both
        ///     bool (two bool operands are already one bitwise pass); either two operands of one shape, both
        ///     C-contiguous or both strictly F-contiguous, or one operand of one value (any layout) whose rank does not
        ///     exceed the other's, with the other C- or F-contiguous; a result of at least one axis and one element; no
        ///     <c>out=</c>, or a bool <c>out=</c> of that shape and memory order that shares no memory with an operand; and
        ///     a dtype the comparison engine has a run for.
        /// </returns>
        /// <remarks>
        ///     <para><b>The same bytes as the composition, by construction.</b> The composition reduces a numeric operand to
        ///     <c>x != 0</c> and passes a bool operand through as it is, then combines the two with the bitwise ufunc, which
        ///     combines a bool operand's TRUTH (a nonzero byte is True: a bool array viewed from uint8 bytes 2 and 1 ANDs to 1,
        ///     255 and 1 XOR to 0, probed). This route computes the same truth values, a bool operand's included, with the
        ///     comparison against a zero of the operand's own dtype (zero converts exactly into every dtype, so the comparison
        ///     dtype cannot change a truth value; NaN is True, -0.0 False, a complex value True when either part is nonzero),
        ///     and combines them with the bitwise ufunc's same-dtype bool kernel, the composition's own last step. The result's
        ///     layout is the composition's: C for C operands, F for strictly F ones (the comparison and the bitwise ufunc
        ///     keep an all-F operand set F).</para>
        ///     <para><b>Why the shape rules.</b> Each block is a linear run of every operand's memory, which is the element
        ///     order of the iteration only when the operands share one dense layout (or one of them is a single value), and
        ///     a single-valued operand of higher rank would add leading axes to the result (NumPy's broadcast: (1, 1)
        ///     against (3,) is (1, 3)). An <c>out=</c> that overlaps an operand could be written before the operand is read,
        ///     which NumPy's COPY_IF_OVERLAP orders; the composition keeps those, and the empty and 0-d results, whose
        ///     layouts it decides.</para>
        ///     <para><b>[SkipLocalsInit].</b> The two 4 KB stack blocks are not zeroed (zeroing them cost about a tenth of a
        ///     call at 1,000 float64 elements, 0.08 of 0.78 us, measured): every byte the combine reads is written first, by
        ///     the comparison run or the fill. The 16-byte zero is cleared explicitly.</para>
        /// </remarks>
        [System.Runtime.CompilerServices.SkipLocalsInit]
        private unsafe NDArray TryLogicalRun(NDArray lhs, NDArray rhs, BinaryOp op, NDArray @out, NDArray where)
        {
            if (where is not null || DisableLogicalRun)
                return null;
            NPTypeCode lType = lhs.typecode, rType = rhs.typecode;
            // Two bool operands: the composition is already one bitwise pass over their bytes.
            if (lType == NPTypeCode.Boolean && rType == NPTypeCode.Boolean)
                return null;

            // The array operand(s): the layout every block is read in, and the operand of one value if there is one.
            Shape ls = lhs.Shape, rs = rhs.Shape;
            bool lOne = ls.size == 1, rOne = rs.size == 1;
            Shape arrShape;
            if (lOne ^ rOne)
            {
                var one = lOne ? ls : rs;
                arrShape = lOne ? rs : ls;
                if (one.NDim > arrShape.NDim || arrShape.IsBroadcasted || !(arrShape.IsContiguous || arrShape.IsFContiguous))
                    return null;
            }
            else if (!lOne)
            {
                if (!ls.Equals(rs) || ls.IsBroadcasted || rs.IsBroadcasted)
                    return null;
                bool bothC = ls.IsContiguous && rs.IsContiguous;
                if (!bothC && !(ls.IsFContiguous && rs.IsFContiguous))
                    return null;
                arrShape = ls;
            }
            else
            {
                return null;   // two single values: the composition decides the (all-1) shape and its 0-d case
            }
            long n = arrShape.size;
            if (n == 0 || arrShape.NDim == 0)
                return null;
            // A strictly F operand set is read in F order, so the result (or out=) must be F too.
            bool runIsF = !arrShape.IsContiguous;

            // Every operand's truth values come from the comparison engine, a bool operand's too (its nonzero bytes are True,
            // as the bitwise ufunc reads them). The run is null where runtime IL generation is unavailable: the composition
            // then serves the call.
            ComparisonRun lRun = ComparisonRun.Get(ComparisonOp.NotEqual, lType, lType);
            ComparisonRun rRun = ComparisonRun.Get(ComparisonOp.NotEqual, rType, rType);
            if (lRun is null || rRun is null)
                return null;
            // The combine is the bitwise ufunc's own kernel, the composition's last step: bool & bool (| or ^) over the two
            // blocks of truth bytes, vectorized on byte lanes. Fetched before the result is allocated, so nothing is left
            // behind if IL generation is unavailable (the runs above are null then, and the composition serves the call).
            MixedTypeKernel combine = DirectILKernelGenerator.GetMixedTypeKernel(
                new MixedTypeKernelKey(NPTypeCode.Boolean, NPTypeCode.Boolean, NPTypeCode.Boolean, op, ExecutionPath.SimdFull));

            NDArray result;
            if (@out is null)
            {
                result = NewArray(NPTypeCode.Boolean, CanonicalResultShape(arrShape, runIsF), false);
            }
            else
            {
                // A bool out= of the iteration's shape and memory order, sharing no memory with an operand (the read-only
                // check ran in UfuncLoopCode). Anything else is the composition's: a cast, a broadcast into out=, an
                // overlap NumPy's COPY_IF_OVERLAP orders.
                var os = @out.Shape;
                if (@out.typecode != NPTypeCode.Boolean || !os.Equals(arrShape) || os.IsBroadcasted
                    || !(runIsF ? os.IsFContiguous : os.IsContiguous)
                    || NDMemOverlap.SolveMayShareMemory(@out, lhs, maxWork: 0) != MemOverlap.No
                    || NDMemOverlap.SolveMayShareMemory(@out, rhs, maxWork: 0) != MemOverlap.No)
                    return null;
                result = @out;
            }

            int li = lhs.dtypesize, ri = rhs.dtypesize;
            byte* lp = (byte*)lhs.Address + ls.offset * li;
            byte* rp = (byte*)rhs.Address + rs.offset * ri;
            byte* o = (byte*)result.Address + result.Shape.offset;
            // A zero of every dtype is all-zero bytes (+0.0, 0+0j, 0m, '\0'); the comparison reads it as a value of stride 0.
            byte* zero = stackalloc byte[16];
            new Span<byte>(zero, 16).Clear();
            byte* lBlock = stackalloc byte[LogicalRunBlock];
            byte* rBlock = stackalloc byte[LogicalRunBlock];

            // An operand of one value has one truth value: its block is filled once and reused by every block of the
            // other operand.
            if (lOne)
                FillTruth(lp, lRun, zero, lBlock);
            if (rOne)
                FillTruth(rp, rRun, zero, rBlock);

            for (long i = 0; i < n; i += LogicalRunBlock)
            {
                int len = (int)Math.Min(LogicalRunBlock, n - i);
                byte* lTruth = lOne ? lBlock : TruthBlock(lp + i * li, li, lRun, zero, lBlock, len);
                byte* rTruth = rOne ? rBlock : TruthBlock(rp + i * ri, ri, rRun, zero, rBlock, len);
                // The SimdFull path reads only the three pointers and the count: no strides or shape.
                combine(lTruth, rTruth, o + i, null, null, null, 1, len);
            }
            LogicalRuns++;
            return result;
        }

        /// <summary>
        ///     logical_not of a numeric operand in one comparison pass: the comparison engine's <c>x == 0</c> run written
        ///     straight into the result, or into a bool <c>out=</c> of the operand's layout. The composition the caller falls
        ///     back to compares against a 0-d zero it allocates (about 0.25 us a call: <c>a != 0</c> with a literal took 0.78 us
        ///     against 0.52 with a held zero, 1,000 float64 elements) and, with an <c>out=</c>, builds an iterator.
        /// </summary>
        /// <param name="nd">The operand (numeric: a bool operand takes the invert ufunc before this; text was answered by the
        ///     guard).</param>
        /// <param name="out">The <c>out=</c> operand, or null.</param>
        /// <param name="where">The <c>where=</c> mask, or null.</param>
        /// <returns>
        ///     The result (a fresh <c>NDArray&lt;bool&gt;</c>, the comparison's own result type, or <paramref name="out"/>), or
        ///     null when the call is not one this route serves and the composition must serve it: a mask, an operand that is
        ///     broadcast or neither C- nor F-contiguous, an empty or 0-d result, an <c>out=</c> that is not a bool of the
        ///     operand's shape and memory order or that shares memory with it, or a dtype the comparison engine has no run
        ///     for.
        /// </returns>
        /// <remarks>
        ///     The same bytes and layout as the composition: it is the same comparison (<c>x == 0</c> against a zero of the
        ///     operand's own dtype, into which the composition's 0-d zero converts exactly), laid out as the comparison lays
        ///     it out (C, or F for a strictly F operand). The run reads its whole input and writes its whole output in one
        ///     call, so an operand and an <c>out=</c> that share no memory need no ordering.
        /// </remarks>
        private unsafe NDArray TryLogicalNotRun(NDArray nd, NDArray @out, NDArray where)
        {
            if (where is not null || DisableLogicalRun)
                return null;
            var s = nd.Shape;
            if (s.size == 0 || s.NDim == 0 || s.IsBroadcasted || !(s.IsContiguous || s.IsFContiguous))
                return null;
            var run = ComparisonRun.Get(ComparisonOp.Equal, nd.typecode, nd.typecode);
            if (run is null)
                return null;
            // A strictly F operand is read in F order, so the result (or out=) is F too.
            bool runIsF = !s.IsContiguous;
            NDArray result;
            if (@out is null)
            {
                result = NewArray<bool>(CanonicalResultShape(s, runIsF), false);
            }
            else
            {
                // A bool out= of the operand's shape and memory order, sharing no memory with it (the read-only check ran in
                // UfuncLoopCode); a cast, a broadcast into out= and an overlap are the composition's.
                var os = @out.Shape;
                if (@out.typecode != NPTypeCode.Boolean || !os.Equals(s) || os.IsBroadcasted
                    || !(runIsF ? os.IsFContiguous : os.IsContiguous)
                    || NDMemOverlap.SolveMayShareMemory(@out, nd, maxWork: 0) != MemOverlap.No)
                    return null;
                result = @out;
            }
            // A zero of every dtype is all-zero bytes; the comparison reads it as a value of stride 0.
            byte* zero = stackalloc byte[16];
            new Span<byte>(zero, 16).Clear();
            int item = nd.dtypesize;
            run.Run((byte*)nd.Address + s.offset * item, item, zero, 0, (byte*)result.Address + result.Shape.offset, 1, s.size);
            LogicalRuns++;
            return result;
        }

        /// <summary>
        ///     The truth values of one block of an operand: the comparison <c>x != 0</c> written into <paramref name="block"/>
        ///     as 0 / 1 bytes.
        /// </summary>
        /// <param name="p">The block's first element.</param>
        /// <param name="itemSize">The operand's element size in bytes.</param>
        /// <param name="run">The comparison run (<c>!=</c>) for the operand's dtype.</param>
        /// <param name="zero">A zero of the operand's dtype (all-zero bytes).</param>
        /// <param name="block">The stack block the comparison writes into.</param>
        /// <param name="len">The block's element count.</param>
        /// <returns><paramref name="block"/>, holding the block's truth values.</returns>
        private static unsafe byte* TruthBlock(byte* p, int itemSize, ComparisonRun run, byte* zero, byte* block, int len)
        {
            run.Run(p, itemSize, zero, 0, block, 1, len);
            return block;
        }

        /// <summary>
        ///     Fills a whole block with the truth value of an operand of one value: the comparison <c>x != 0</c> of its one
        ///     element, repeated so every block of the other operand combines with it.
        /// </summary>
        /// <param name="p">The operand's element.</param>
        /// <param name="run">The comparison run (<c>!=</c>) for the operand's dtype.</param>
        /// <param name="zero">A zero of the operand's dtype (all-zero bytes).</param>
        /// <param name="block">The block to fill (<see cref="LogicalRunBlock"/> bytes).</param>
        private static unsafe void FillTruth(byte* p, ComparisonRun run, byte* zero, byte* block)
        {
            byte truth;
            run.Run(p, 0, zero, 0, &truth, 1, 1);
            new Span<byte>(block, LogicalRunBlock).Fill(truth);
        }

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
            if (TryLogicalRun(lhs, rhs, BinaryOp.BitwiseAnd, @out, where) is { } run)
                return run;
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
            // The one-pass route first (see LogicalAnd), then the composition.
            if (TryLogicalRun(lhs, rhs, BinaryOp.BitwiseOr, @out, where) is { } run)
                return run;
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
            // The one-pass route first (see LogicalAnd), then the composition.
            if (TryLogicalRun(lhs, rhs, BinaryOp.BitwiseXor, @out, where) is { } run)
                return run;
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
            // The one comparison pass for the common call (see TryLogicalNotRun), then the composition.
            if (TryLogicalNotRun(nd, @out, where) is { } run)
                return run;
            // logical_not is unary, but the comparison computing it has a second, hidden input (the 0-d zero), which its
            // broadcast error would list ("(3,4) () (3,4,2)"). Joining the operand with out= / where= first gives NumPy's
            // text, which lists the operand, then out= and where=. NumPy's checks before its broadcast run first, in its
            // order: a read-only out= and a non-bool mask (its argument parsing), then the output cast, which a bool result
            // never fails (it casts same_kind into every out=, text included). An out= of the operand's own dimensions
            // without a mask cannot fail the join and skips it, as the ufunc routes do.
            if (where is not null || (@out is not null && !SameDims(nd.Shape, @out.Shape)))
            {
                Strings.NDStringUfuncs.ThrowIfOutWhereInvalid(@out, where);
                ResolveUfuncIterationShape(nd.Shape.Clean(), new[] { nd }, @out, where);
            }
            return Compare(nd, NewScalar(0), null, @out, where);
        }
    }
}
