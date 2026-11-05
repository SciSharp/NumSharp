using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;
using SysUnsafe = System.Runtime.CompilerServices.Unsafe;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        /// np.divmod — the fused two-output ufunc <c>(floor_divide(lhs, rhs), remainder(lhs, rhs))</c>,
        /// element-wise. Item1 is the floored quotient, Item2 the floored remainder (divisor's sign).
        ///
        /// Two routes, both bit-identical to NumPy 2.4.2:
        /// <list type="bullet">
        /// <item>When <paramref name="out"/> (a 2-tuple, either slot may be null) or <paramref name="where"/> is supplied,
        /// or an explicit <paramref name="dtype"/> loop is requested, NumPy's two-output ufunc call
        /// (<see cref="DivModInto"/>): ONE iterator over <c>(x1, x2, out1, out2[, where])</c> whose inner loop reads each
        /// element pair once and writes both results, so an out that is also an input
        /// (<c>np.divmod(a, 2.0, out=(a, r))</c>) still leaves the remainders of the ORIGINAL dividends in <c>r</c>.</item>
        /// <item>Otherwise run the FUSED single-pass IL kernel (<see cref="DirectILKernelGenerator.GetDivModKernel"/>),
        /// which computes both outputs from one <see cref="Utilities.NDDivision"/> Divmod* call per
        /// element over contiguous, promoted operands — measurably faster than composing (no second
        /// pass, one idiv for integers).</item>
        /// </list>
        /// Complex raises (NumPy has no complex divmod loop; matches <see cref="Mod"/>'s refusal).
        /// </summary>
        /// <param name="lhs">The dividends (<c>x1</c>).</param>
        /// <param name="rhs">The divisors (<c>x2</c>); broadcasts against <paramref name="lhs"/>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: the loop both results are computed in (each input must reach it by a
        /// same_kind cast), or null for the NEP 50 promotion of the inputs (bool promotes to the int8 loop).</param>
        /// <param name="out">NumPy's <c>out=(out1, out2)</c>: the arrays the quotient and the remainder are written into
        /// (either may be null for a fresh one). Each must take the loop dtype by a same_kind cast and have the broadcast
        /// shape (it is never stretched); it may be an input itself.</param>
        /// <param name="where">NumPy's <c>where=</c>: a bool mask broadcast over the iteration; a False position keeps
        /// what a given out held (and is uninitialized in a fresh result, as in NumPy).</param>
        /// <returns>The (quotient, remainder) pair: each given out itself, or a fresh array of the loop dtype laid out the
        /// way NumPy's iterator allocates it.</returns>
        /// <exception cref="TypeError">A <paramref name="dtype"/> naming more than a DType class (<c>'&gt;f8'</c>), a text
        /// operand or request divmod has no loop for, or (out= route) a bool / complex <paramref name="dtype"/>: NumPy's
        /// <c>No loop matching the specified signature and casting was found for ufunc divmod</c>.</exception>
        /// <exception cref="NotSupportedException">A complex operand, or inputs that promote to complex.</exception>
        /// <exception cref="ValueError">A read-only array in <paramref name="out"/> (<c>output array is read-only</c>,
        /// checked for both slots before anything else), or a text out under <paramref name="where"/> whose current text
        /// does not parse into the loop dtype.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>; an input <paramref name="dtype"/> cannot
        /// take, or an out that cannot take the loop dtype, by a same_kind cast (NumPy's <c>Cannot cast ufunc 'divmod'
        /// input 0 / output 2 …</c> texts); an out or mask that does not broadcast, or an out that would be
        /// stretched.</exception>
        /// <exception cref="IncorrectShapeException">The inputs do not broadcast together.</exception>
        /// <exception cref="OverflowException">A C# integer literal operand (NumPy's Python int) outside the integer loop's
        /// range (<c>Python integer 1000 out of bounds for int8</c>).</exception>
        public override (NDArray Quotient, NDArray Remainder) DivMod(
            NDArray lhs, NDArray rhs, DType dtype = null,
            (NDArray Quotient, NDArray Remainder) @out = default, NDArray where = null)
        {
            // NumPy parses BOTH outputs, then where=, before it resolves the loop: with text involved (a string operand or
            // a string dtype= request) a read-only out of either slot or a non-bool mask is reported first. The class check
            // runs first of all, and its lane is the loop the out= route computes in.
            NPTypeCode? loopCode = UfuncLoopCode(dtype);
            if (Strings.NDStringUfuncs.IsText(lhs) || Strings.NDStringUfuncs.IsText(rhs) || Strings.NDStringUfuncs.IsStringRequest(dtype))
            {
                Strings.NDStringUfuncs.ThrowIfOutWhereInvalid(@out.Quotient, null);
                Strings.NDStringUfuncs.ThrowIfOutWhereInvalid(@out.Remainder, where);
            }
            // divmod has no string loop (the legacy resolver's text); a string request is the no-loop text.
            Strings.NDStringUfuncs.ThrowIfBinary("divmod", lhs, rhs, @out.Quotient, UfuncLoopCode(dtype, "divmod", lhs, rhs), where);
            // Promotion mirrors ExecuteBinaryOp's Mod/FloorDivide path: result_type, then the
            // no-bool-loop bump (True divmod True -> int8). Complex is unsupported (NumPy TypeError).
            var resultType = np._FindCommonType(lhs, rhs);
            if (resultType == NPTypeCode.Complex ||
                lhs.GetTypeCode == NPTypeCode.Complex || rhs.GetTypeCode == NPTypeCode.Complex)
                throw new NotSupportedException("Operation DivMod not supported for Complex");

            bool hasOut = @out.Quotient is not null || @out.Remainder is not null;

            // NumPy's two-output ufunc call whenever out=/where=/dtype= is present: one iterator, one read of each element
            // pair, both results written from it. Composing floor_divide and remainder here (as this route used to) let
            // the quotient overwrite an input that the remainder pass then read (parity QA-44).
            if (hasOut || where is not null || dtype is not null)
                return DivModInto(lhs, rhs, resultType, loopCode, @out.Quotient, @out.Remainder, where);

            if (resultType == NPTypeCode.Boolean)
                resultType = NPTypeCode.SByte;

            // Broadcast to the common result shape.
            var (leftShape, rightShape) = Broadcast(lhs.Shape, rhs.Shape);
            var cleanShape = leftShape.Clean();

            // Scalars (0-d) and empties: the fused contiguous kernel wants ndim >= 1 with elements.
            // Route the rare 0-d / empty cases through the compose path, which handles them exactly.
            var kernel = cleanShape.NDim >= 1 && cleanShape.size > 0
                ? DirectILKernelGenerator.GetDivModKernel(resultType)
                : null;
            if (kernel is null)
            {
                var qOut = FloorDivide(lhs, rhs);
                var rOut = Mod(lhs, rhs);
                return (qOut, rOut);
            }

            // NumPy's result layout: divmod has two outputs, so NumPy never takes its trivial loop and the iterator
            // allocates both outputs in the inputs' memory order (K order: F for F inputs, a transposed input's own axis
            // order, C where the votes conflict). The fused kernel writes C order, so a non-C layout runs it over the
            // operands viewed in that order and relabels both C results (DefaultEngine.UfuncResultLayout.cs).
            NDArray lv = lhs, rv = rhs;
            Shape runShape = cleanShape;
            bool relabel = TryNonCResultLayout(new[] { lhs.Shape, rhs.Shape }, cleanShape.dimensions,
                                               trivialLoopEligible: false, out var layout);
            try
            {
                if (relabel)
                {
                    int[] perm = layout.DenseAxisOrder();
                    lv = OperandInLayoutOrder(lhs, cleanShape.NDim, perm);
                    rv = OperandInLayoutOrder(rhs, cleanShape.NDim, perm);
                    var runDims = new long[perm.Length];
                    for (int i = 0; i < perm.Length; i++)
                        runDims[i] = cleanShape.dimensions[perm[i]];
                    runShape = Shape.Adopt(runDims);
                }

                // Materialize both operands as C-contiguous arrays of resultType (reads through any
                // broadcast/strided/mixed-dtype layout). ascontiguousarray is a no-op view when the
                // operand is already C-contiguous at resultType — so the common same-dtype contiguous
                // case (and a dense operand viewed in its own memory order) pays no copy. (np.modf uses
                // the same materialize-then-fused-kernel pattern.)
                var aC = MaterializeContiguous(lv, runShape, resultType);
                var bC = MaterializeContiguous(rv, runShape, resultType);

                var q = NewArray(resultType, runShape, fillZeros: false);
                var r = NewArray(resultType, runShape, fillZeros: false);

                unsafe
                {
                    kernel((void*)aC.Address, (void*)bC.Address, (void*)q.Address, (void*)r.Address, runShape.size);
                }

                // Reclaim the materialized operands. MaterializeContiguous returns the operand itself
                // (passthrough, no copy) when it is already a C-contiguous resultType array; otherwise a
                // fresh copy. Dispose only the ones we minted — never an operand — so the pooled buffer
                // returns promptly instead of falling to the finalizer (the zero-leak UndisposedIntermediate gate).
                if (!ReferenceEquals(aC, lv)) aC.Dispose();
                if (!ReferenceEquals(bC, rv)) bC.Dispose();

                if (relabel)
                    return (NDArray.AdoptLayout(q, layout), NDArray.AdoptLayout(r, layout));
                return (q, r);
            }
            finally
            {
                DisposeLayoutView(lv, lhs);
                DisposeLayoutView(rv, rhs);
            }
        }

        // =============================================================================
        // np.divmod — the out= / where= / dtype= route (NumPy's two-output ufunc call)
        // =============================================================================
        //
        // NumPy runs divmod as ONE ufunc call with nin=2, nout=2: a single NpyIter over (x1, x2, out1, out2[, where])
        // whose inner loop (BINARY_LOOP_TWO_OUT) reads in1 and in2 into locals and only then stores both results. Every
        // aliasing pattern follows from that: an out that IS an input (same memory, same strides) is read before it is
        // written at each element, and any other overlap is resolved by the iterator's COPY_IF_OVERLAP copy. A two-output
        // ufunc never takes NumPy's trivial loop (try_trivial_single_output_loop is nout == 1 only), so the iterator always
        // runs. This route is that structure: the same operands, flags and buffered casts np.clip's iterator route uses,
        // driving a two-output inner loop over NDDivision's Divmod* cores (the ones the fused no-out kernel calls, so the
        // values are that kernel's to the bit).
        // =============================================================================

        /// <summary>
        /// NumPy's text when a ufunc has no loop for the requested <c>dtype=</c> (divmod has no bool and no complex loop).
        /// </summary>
        private const string DivModNoLoopText = "No loop matching the specified signature and casting was found for ufunc divmod";

        /// <summary>
        /// np.divmod's <c>out=</c> / <c>where=</c> / <c>dtype=</c> call as NumPy runs it: the arguments checked in NumPy's
        /// order, the loop resolved, then ONE iterator over <c>(x1, x2, out1, out2[, where])</c> whose inner loop reads each
        /// element pair once and writes both the floored quotient and the remainder.
        /// </summary>
        /// <param name="lhs">The dividends; any layout, broadcast to the iteration.</param>
        /// <param name="rhs">The divisors; any layout, broadcast to the iteration.</param>
        /// <param name="resultType">The NEP 50 promotion of the two inputs (<c>np._FindCommonType</c>), the loop when no
        /// <paramref name="dtype"/> is given.</param>
        /// <param name="dtype">The <c>dtype=</c> lane after the class check, or null.</param>
        /// <param name="outQ">The quotient's out (NumPy's operand 2), or null for a fresh array. A text array receives the
        /// quotients formatted as text.</param>
        /// <param name="outR">The remainder's out (operand 3), or null for a fresh array; may be the same array as
        /// <paramref name="outQ"/> (each element then keeps the value NumPy's loop stores last).</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <returns>(<paramref name="outQ"/> or a fresh quotient, <paramref name="outR"/> or a fresh remainder). A fresh
        /// array has the loop dtype and the layout NumPy's iterator allocates (<see cref="DivModResultLayout"/>); with
        /// <paramref name="where"/> its masked-off positions are uninitialized.</returns>
        /// <exception cref="NumSharpException">A read-only out: the <see cref="ValueError"/> <c>output array is read-only</c>
        /// that <see cref="NumSharpException.ThrowIfNotWriteable"/> raises, both slots checked before anything else.</exception>
        /// <exception cref="ValueError">A text out under <paramref name="where"/> whose current text does not parse into the
        /// loop dtype.</exception>
        /// <exception cref="ArgumentException">A non-bool <paramref name="where"/>; an input <paramref name="dtype"/> cannot
        /// take, or an out that cannot take the loop dtype, under same_kind; an out or mask that does not broadcast, or an
        /// out that would be stretched (NumPy's texts).</exception>
        /// <exception cref="IncorrectTypeException">A bool or complex <paramref name="dtype"/> (<see cref="DivModNoLoopText"/>).</exception>
        /// <exception cref="OverflowException">A weak C# integer literal outside the integer loop's range.</exception>
        /// <exception cref="IncorrectShapeException">The inputs do not broadcast together (every operand listed).</exception>
        /// <remarks>
        /// <para><b>Order of the checks</b> (NumPy's <c>ufunc_generic_fastcall</c>): both outputs' writeability (argument
        /// parsing), the mask's dtype, the loop (no bool / complex loop for <c>dtype=</c>), the weak Python-int pack, the
        /// <c>dtype=</c> input casts, the output casts numbered by operand position (<c>output 2</c> / <c>output 3</c>),
        /// then the iterator's broadcast. Nothing is written before every check passed: the composition this replaces wrote
        /// the quotient before the remainder's out was even checked.</para>
        /// <para><b>Casting.</b> A 0-d input of another dtype is converted up front (NumPy's iterator copies a read-only 0-d
        /// operand that needs a cast eagerly), so the common "array op Python scalar" call stays unbuffered; any other
        /// operand of another dtype is cast through the iterator's buffers, UNSAFE because every cast was validated
        /// above.</para>
        /// <para><b>Text outs.</b> The iterator does not buffer text, so a text out's slot is computed into a fresh array
        /// of the loop dtype and copied into the text out afterwards (masked by <paramref name="where"/>), as the
        /// single-output ufuncs and np.modf do.</para>
        /// </remarks>
        private (NDArray Quotient, NDArray Remainder) DivModInto(NDArray lhs, NDArray rhs, NPTypeCode resultType,
            NPTypeCode? dtype, NDArray outQ, NDArray outR, NDArray where)
        {
            // Argument parsing: NumPy converts both outputs (refusing a read-only one) and then the mask, before any loop
            // question; so a read-only remainder out beats a bad quotient-out cast, as in NumPy.
            if (outQ is not null)
                NumSharpException.ThrowIfNotWriteable(outQ.Shape, "output array");
            if (outR is not null)
                NumSharpException.ThrowIfNotWriteable(outR.Shape, "output array");
            ValidateWhereMask(where);

            // Loop resolution: divmod registers integer, float16/32/64 loops only ('bb' starts at int8), so a bool or a
            // complex request finds none. A bool PROMOTION is not a request: it runs the int8 loop (True divmod True).
            if (dtype is NPTypeCode.Boolean or NPTypeCode.Complex)
                throw new IncorrectTypeException(DivModNoLoopText);
            var lhsType = lhs.GetTypeCode;
            var rhsType = rhs.GetTypeCode;
            // NEP 50: a C# integer literal is NumPy's Python int, packed into the loop with a range check (int8 loop,
            // 1000 → OverflowError) while the loop resolves, before the dtype= input casts.
            CheckWeakPyIntOperand(lhs, rhs, dtype ?? resultType, loopFromPromotion: !dtype.HasValue);
            NPTypeCode loop = resultType;
            if (dtype.HasValue)
            {
                if (dtype.Value != resultType)
                    ValidateBinaryInputCasts(lhsType, rhsType, dtype.Value, "divmod");
                loop = dtype.Value;
            }
            if (loop == NPTypeCode.Boolean)
                loop = NPTypeCode.SByte;

            // A weak int that must round twice into a float loop (float32 array, 2**60 + 2**36 + 1) is replaced by its
            // NumPy pack; the temp is this call's and no result aliases it.
            using var weakInt = AdoptWeakIntOperand(ref lhs, ref rhs, loop);

            // Output casts, numbered by OPERAND position as NumPy's multi-output text numbers them (x1, x2 are 0 and 1).
            if (outQ is not null)
                ValidateNumberedOutCast("divmod", 2, loop, outQ);
            if (outR is not null)
                ValidateNumberedOutCast("divmod", 3, loop, outR);

            // The iterator's broadcast (all operands joined, no out stretched), then a masked text out's READWRITE read.
            long[] dims = DivModIterationDims(lhs, rhs, outQ, outR, where);
            ReadMaskedTextOut(outQ, loop, where);
            ReadMaskedTextOut(outR, loop, where);

            bool qText = outQ is not null && outQ.typecode == NPTypeCode.String;
            bool rText = outR is not null && outR.typecode == NPTypeCode.String;
            NDArray a = lhs, b = rhs, q = null, r = null;
            bool completed = false;
            try
            {
                // The iterator's targets: a numeric out is written (and cast) in place; an omitted or text slot gets a
                // fresh array laid out as NumPy's iterator allocates it, every given operand voting.
                Shape fresh = outQ is null || outR is null || qText || rText
                    ? DivModResultLayout(dims, lhs, rhs, outQ, outR, where)
                    : default;
                q = outQ is not null && !qText ? outQ : NewArray(loop, fresh, false);
                r = outR is not null && !rText ? outR : NewArray(loop, fresh, false);

                if (q.size != 0)
                {
                    // NumPy's iterator copies a read-only 0-d operand that needs a cast eagerly; doing the same keeps the
                    // iterator unbuffered when the outs are the loop dtype (np.divmod(f64_array, 2, out=...)).
                    if (a.ndim == 0 && a.GetTypeCode != loop)
                        a = CastScalarBound(a, loop);
                    if (b.ndim == 0 && b.GetTypeCode != loop)
                        b = CastScalarBound(b, loop);
                    // Reference tests, never NDArray's elementwise == operator.
                    RunDivModIterator(a, b, q, r, where, loop,
                        qPassed: ReferenceEquals(q, outQ), rPassed: ReferenceEquals(r, outR));
                }

                // A text slot was computed into its fresh array: format it into the caller's text out (masked).
                if (qText)
                    UnmanagedStorage.Copyto(outQ, q, NPY_CASTING.NPY_SAME_KIND_CASTING, where);
                if (rText)
                    UnmanagedStorage.Copyto(outR, r, NPY_CASTING.NPY_SAME_KIND_CASTING, where);
                completed = true;
                return (outQ ?? q, outR ?? r);
            }
            finally
            {
                // Reclaim what this call minted and does not return: the converted 0-d inputs, the arrays a text slot was
                // computed into, and, when a check or the loop threw, the fresh results too (never a caller's array).
                if (!ReferenceEquals(a, lhs))
                    a.Dispose();
                if (!ReferenceEquals(b, rhs))
                    b.Dispose();
                if (q is not null && !ReferenceEquals(q, outQ) && (qText || !completed))
                    q.Dispose();
                if (r is not null && !ReferenceEquals(r, outR) && (rText || !completed))
                    r.Dispose();
            }
        }

        /// <summary>
        /// The iteration dimensions of a divmod <c>out=</c> / <c>where=</c> call, in the two phases of NumPy's iterator
        /// (<c>npyiter_fill_axisdata</c>) and with its texts: every operand broadcasts together (the inputs, out1, out2,
        /// the mask), then no output may be stretched to the joint shape.
        /// </summary>
        /// <param name="lhs">The dividends.</param>
        /// <param name="rhs">The divisors.</param>
        /// <param name="outQ">The quotient's out, or null (an omitted out is not an operand NumPy lists).</param>
        /// <param name="outR">The remainder's out, or null.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>A fresh dimension array (the caller may hand it to a shape that adopts it).</returns>
        /// <exception cref="IncorrectShapeException">The inputs do not broadcast: "operands could not be broadcast together
        /// with shapes (3,) (4,) (3,) (3,) " (every operand, in operand order, trailing space included).</exception>
        /// <exception cref="ArgumentException">An out or the mask does not broadcast with the inputs (the same text), or an
        /// out would be stretched: "non-broadcastable output operand with shape (3,) doesn't match the broadcast shape
        /// (2,3)", checked for out1 before out2.</exception>
        private static long[] DivModIterationDims(NDArray lhs, NDArray rhs, NDArray outQ, NDArray outR, NDArray where)
        {
            Shape full;
            try
            {
                full = Shape.Broadcast(lhs.Shape, rhs.Shape).LeftShape.Clean();
            }
            catch (IncorrectShapeException e)
            {
                // The house type of an input broadcast failure (BroadcastUfuncInputs), with NumPy's all-operand list.
                throw new IncorrectShapeException(
                    $"operands could not be broadcast together with shapes {DivModOperandShapes(lhs, rhs, outQ, outR, where)}", e);
            }

            // Phase 1: the joint broadcast, folding each further operand in pairwise (the N-way broadcast NumPy computes).
            try
            {
                if (outQ is not null)
                    full = Shape.ResolveReturnShape(full, outQ.Shape);
                if (outR is not null)
                    full = Shape.ResolveReturnShape(full, outR.Shape);
                if (where is not null)
                    full = Shape.ResolveReturnShape(full, where.Shape);
            }
            catch (Exception e)
            {
                throw new ArgumentException(
                    $"operands could not be broadcast together with shapes {DivModOperandShapes(lhs, rhs, outQ, outR, where)}", e);
            }

            // Phase 2: an output is NO_BROADCAST, so the joint shape (the mask's extra axes included) must be its own.
            if (outQ is not null && !full.Equals(outQ.Shape))
                throw new ArgumentException(
                    $"non-broadcastable output operand with shape {NumPyShapeRepr(outQ.Shape)} " +
                    $"doesn't match the broadcast shape {NumPyShapeRepr(full)}");
            if (outR is not null && !full.Equals(outR.Shape))
                throw new ArgumentException(
                    $"non-broadcastable output operand with shape {NumPyShapeRepr(outR.Shape)} " +
                    $"doesn't match the broadcast shape {NumPyShapeRepr(full)}");

            return (long[])full.dimensions.Clone();
        }

        /// <summary>
        /// NumPy's operand list in a divmod broadcast error: each present operand's shape and a space, in iterator order
        /// (x1, x2, out1, out2, where).
        /// </summary>
        /// <param name="lhs">The dividends.</param>
        /// <param name="rhs">The divisors.</param>
        /// <param name="outQ">The quotient's out, or null (not listed).</param>
        /// <param name="outR">The remainder's out, or null (not listed).</param>
        /// <param name="where">The mask, or null (not listed).</param>
        /// <returns>The shapes, each followed by a space: <c>"(3,) (4,) (3,) (3,) "</c>.</returns>
        private static string DivModOperandShapes(NDArray lhs, NDArray rhs, NDArray outQ, NDArray outR, NDArray where)
        {
            string s = NumPyShapeRepr(lhs.Shape) + " " + NumPyShapeRepr(rhs.Shape) + " ";
            if (outQ is not null) s += NumPyShapeRepr(outQ.Shape) + " ";
            if (outR is not null) s += NumPyShapeRepr(outR.Shape) + " ";
            if (where is not null) s += NumPyShapeRepr(where.Shape) + " ";
            return s;
        }

        /// <summary>
        /// The layout NumPy's iterator allocates an omitted divmod output in (<see cref="Shape.NpyIterOutputShape"/>): every
        /// operand the call passed votes — the inputs, a given out, the mask — so two F inputs give F results, while a C
        /// out next to them makes the other result C (operands that disagree leave C order, probed 2.4.2).
        /// </summary>
        /// <param name="dims">The iteration dimensions (cloned here; the shape adopts its own copy).</param>
        /// <param name="lhs">The dividends.</param>
        /// <param name="rhs">The divisors.</param>
        /// <param name="outQ">The quotient's out, or null.</param>
        /// <param name="outR">The remainder's out, or null.</param>
        /// <param name="where">The mask, or null.</param>
        /// <returns>A fresh, dense shape of <paramref name="dims"/> (0-d for no dimensions).</returns>
        private static Shape DivModResultLayout(long[] dims, NDArray lhs, NDArray rhs, NDArray outQ, NDArray outR, NDArray where)
        {
            int n = 2 + (outQ is null ? 0 : 1) + (outR is null ? 0 : 1) + (where is null ? 0 : 1);
            var votes = new Shape[n];
            int k = 0;
            votes[k++] = lhs.Shape;
            votes[k++] = rhs.Shape;
            if (outQ is not null) votes[k++] = outQ.Shape;
            if (outR is not null) votes[k++] = outR.Shape;
            if (where is not null) votes[k] = where.Shape;
            return Shape.NpyIterOutputShape((long[])dims.Clone(), votes);
        }

        /// <summary>
        /// Builds and drives the divmod iterator over <c>(x1, x2, out1, out2[, where])</c>.
        /// </summary>
        /// <param name="a">The dividends (a 0-d one already of the loop dtype).</param>
        /// <param name="b">The divisors (a 0-d one already of the loop dtype).</param>
        /// <param name="q">The quotient target: the caller's numeric out, or a fresh array of the loop dtype.</param>
        /// <param name="r">The remainder target, likewise.</param>
        /// <param name="where">The bool mask, or null.</param>
        /// <param name="loop">The loop dtype the inner loop computes in.</param>
        /// <param name="qPassed">True when <paramref name="q"/> is the caller's out (READWRITE under a mask, see
        /// <see cref="DivModIterOpFlags"/>), false for a fresh array.</param>
        /// <param name="rPassed">True when <paramref name="r"/> is the caller's out.</param>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> has no divmod core (never reached: the loop
        /// resolution above refuses bool and complex).</exception>
        private static unsafe void RunDivModIterator(NDArray a, NDArray b, NDArray q, NDArray r, NDArray where, NPTypeCode loop,
            bool qPassed, bool rPassed)
        {
            int nop = where is null ? 4 : 5;
            var ops = where is null ? new[] { a, b, q, r } : new[] { a, b, q, r, where };

            // Buffered casting only when a data operand is not of the loop dtype (an input array of another dtype, an out
            // of another dtype); UNSAFE because every cast was validated against NumPy's rules before this point.
            bool needsCast = a.GetTypeCode != loop || b.GetTypeCode != loop || q.GetTypeCode != loop || r.GetTypeCode != loop;
            var globalFlags = NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            NPTypeCode[] opDtypes = null;
            if (needsCast)
            {
                globalFlags |= NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER | NDIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
                opDtypes = where is null
                    ? new[] { loop, loop, loop, loop }
                    : new[] { loop, loop, loop, loop, NPTypeCode.Empty };   // the mask keeps its own (bool) dtype
            }

            var kernel = DivModInnerLoopFor(loop);
            // K order (memory order where the operands agree), as NumPy's ufunc iterates; the order is not observable:
            // each element is computed from its own inputs and COPY_IF_OVERLAP resolves every non-elementwise overlap.
            using var iter = NDIterRef.MultiNew(nop, ops, globalFlags, NPY_ORDER.NPY_KEEPORDER, casting,
                DivModIterOpFlags(where is not null, qPassed, rPassed), opDtypes);
            iter.ForEach(kernel);
        }

        /// <summary>
        /// The per-operand iterator flags of one divmod call, shared between calls (one array per combination, built at
        /// type initialization): the two inputs read-only, each output NO_BROADCAST, and the mask READONLY | ARRAYMASK.
        /// </summary>
        /// <param name="masked">True when a where= mask rides as the trailing operand (each output WRITEMASKED).</param>
        /// <param name="qPassed">True when out1 is the caller's array.</param>
        /// <param name="rPassed">True when out2 is the caller's array.</param>
        /// <returns>The flag array (never written: NDIterRef.MultiNew only reads it).</returns>
        /// <remarks>
        /// A caller's out under a mask is READWRITE, NumPy's masked-ufunc rule ("In the masked version, we consider the
        /// output read-write"): when it overlaps an input, COPY_IF_OVERLAP computes into a temporary that is written back
        /// whole, and only a READ operand's temporary starts as the array's contents, so the masked-off positions come
        /// back intact. A fresh output stays WRITEONLY. Every data operand is OVERLAP_ASSUME_ELEMENTWISE, so an input that
        /// IS an output (same data, dims and strides) is not copied: the inner loop reads it before writing.
        /// </remarks>
        private static NDIterPerOpFlags[] DivModIterOpFlags(bool masked, bool qPassed, bool rPassed)
            => s_divModIterOpFlags[(masked ? 4 : 0) + (qPassed ? 2 : 0) + (rPassed ? 1 : 0)];

        /// <summary>The eight flag arrays behind <see cref="DivModIterOpFlags"/>, indexed by its three arguments.</summary>
        private static readonly NDIterPerOpFlags[][] s_divModIterOpFlags = BuildDivModIterOpFlags();

        /// <summary>Builds the eight <see cref="DivModIterOpFlags"/> arrays (see there for the flags and why).</summary>
        /// <returns>The table, indexed by <c>masked × 4 + qPassed × 2 + rPassed</c>.</returns>
        private static NDIterPerOpFlags[][] BuildDivModIterOpFlags()
        {
            const NDIterPerOpFlags Elw = NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
            var table = new NDIterPerOpFlags[8][];
            for (int i = 0; i < table.Length; i++)
            {
                bool masked = (i & 4) != 0, qPassed = (i & 2) != 0, rPassed = (i & 1) != 0;
                var flags = new NDIterPerOpFlags[masked ? 5 : 4];
                flags[0] = NDIterPerOpFlags.READONLY | Elw;
                flags[1] = NDIterPerOpFlags.READONLY | Elw;
                flags[2] = (masked && qPassed ? NDIterPerOpFlags.READWRITE : NDIterPerOpFlags.WRITEONLY)
                           | NDIterPerOpFlags.NO_BROADCAST | Elw | (masked ? NDIterPerOpFlags.WRITEMASKED : 0);
                flags[3] = (masked && rPassed ? NDIterPerOpFlags.READWRITE : NDIterPerOpFlags.WRITEONLY)
                           | NDIterPerOpFlags.NO_BROADCAST | Elw | (masked ? NDIterPerOpFlags.WRITEMASKED : 0);
                if (masked)
                    flags[4] = NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK;
                table[i] = flags;
            }
            return table;
        }

        /// <summary>
        /// The two-output inner loop for <paramref name="loop"/>: a cached delegate over
        /// <see cref="DivModInnerLoop{T}"/> (C# caches a static method-group conversion, so no call allocates).
        /// </summary>
        /// <param name="loop">The loop dtype (an integer, float16/32/64 or decimal lane).</param>
        /// <returns>The inner-loop delegate the iterator drives.</returns>
        /// <exception cref="NotSupportedException"><paramref name="loop"/> is bool, complex or another lane without a
        /// divmod core.</exception>
        private static unsafe NDInnerLoopFunc DivModInnerLoopFor(NPTypeCode loop) => loop switch
        {
            NPTypeCode.SByte => DivModInnerLoop<sbyte>,
            NPTypeCode.Byte => DivModInnerLoop<byte>,
            NPTypeCode.Int16 => DivModInnerLoop<short>,
            NPTypeCode.UInt16 => DivModInnerLoop<ushort>,
            NPTypeCode.Char => DivModInnerLoop<char>,
            NPTypeCode.Int32 => DivModInnerLoop<int>,
            NPTypeCode.UInt32 => DivModInnerLoop<uint>,
            NPTypeCode.Int64 => DivModInnerLoop<long>,
            NPTypeCode.UInt64 => DivModInnerLoop<ulong>,
            NPTypeCode.Half => DivModInnerLoop<Half>,
            NPTypeCode.Single => DivModInnerLoop<float>,
            NPTypeCode.Double => DivModInnerLoop<double>,
            NPTypeCode.Decimal => DivModInnerLoop<decimal>,
            _ => throw new NotSupportedException($"divmod has no {loop} loop")
        };

        /// <summary>
        /// NumPy's two-output divmod inner loop (<c>BINARY_LOOP_TWO_OUT</c>) for one loop dtype: for each element, read
        /// the dividend and the divisor, compute both results with one <see cref="NDDivision"/> Divmod* call, then store
        /// them in the order NumPy's loop for that dtype stores them.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        /// <param name="data">The operand pointers: x1, x2, out1, out2 (a trailing mask pointer is not read: the iterator's
        /// masked driver calls this once per mask-true run).</param>
        /// <param name="strides">The operands' inner byte strides (0 for a broadcast operand).</param>
        /// <param name="count">The elements of this inner loop.</param>
        /// <param name="auxdata">Unused.</param>
        /// <remarks>
        /// <para>Both inputs are read before either output is written, so an output that IS an input (the elementwise
        /// alias the iterator does not copy) still yields the ORIGINAL element's results.</para>
        /// <para>The store order is NumPy's and shows only when out1 and out2 are one array: the integer loops store the
        /// quotient then the remainder (the remainder is kept); the float16/32/64 loops write the remainder inside
        /// <c>npy_divmod</c> and the returned quotient after it (the quotient is kept). Decimal, which NumPy does not have,
        /// follows the float loops.</para>
        /// </remarks>
        private static unsafe void DivModInnerLoop<T>(void** data, long* strides, long count, void* auxdata) where T : unmanaged
        {
            byte* a = (byte*)data[0], b = (byte*)data[1], q = (byte*)data[2], r = (byte*)data[3];
            long sa = strides[0], sb = strides[1], sq = strides[2], sr = strides[3];
            for (long i = 0; i < count; i++)
            {
                T x = *(T*)a, y = *(T*)b;
                T quotient = DivModElement(x, y, out T remainder);
                // A JIT-time constant per T: the branch folds away in each instantiation.
                if (DivModStoresRemainderFirst<T>())
                {
                    *(T*)r = remainder;
                    *(T*)q = quotient;
                }
                else
                {
                    *(T*)q = quotient;
                    *(T*)r = remainder;
                }
                a += sa;
                b += sb;
                q += sq;
                r += sr;
            }
        }

        /// <summary>
        /// Whether NumPy's divmod loop for <typeparamref name="T"/> stores the remainder before the quotient: the float
        /// loops (<c>*op1 = npy_divmod(in1, in2, op2)</c>) do; the integer loops store <c>op1</c> first.
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type.</typeparam>
        /// <returns>True for float16, float32, float64 (and decimal, which follows them).</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool DivModStoresRemainderFirst<T>() where T : unmanaged
            => typeof(T) == typeof(Half) || typeof(T) == typeof(float) || typeof(T) == typeof(double) || typeof(T) == typeof(decimal);

        /// <summary>
        /// One element of NumPy's divmod for <typeparamref name="T"/>: the floored quotient, and the remainder (the
        /// divisor's sign) through <paramref name="remainder"/> — the <see cref="NDDivision"/> Divmod* core the fused no-out
        /// kernel calls, so both routes agree to the bit (integer ÷0 → (0, 0), MIN / -1 → (MIN, 0), float ÷0 → (±inf or
        /// nan, nan)).
        /// </summary>
        /// <typeparam name="T">The loop dtype's element type; each instantiation keeps only its own branch.</typeparam>
        /// <param name="a">The dividend.</param>
        /// <param name="b">The divisor.</param>
        /// <param name="remainder">Receives the floored remainder.</param>
        /// <returns>The floored quotient.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no divmod core (bool, complex).</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static T DivModElement<T>(T a, T b, out T remainder) where T : unmanaged
        {
            if (typeof(T) == typeof(sbyte))
            {
                sbyte quo = NDDivision.DivmodSByte(SysUnsafe.As<T, sbyte>(ref a), SysUnsafe.As<T, sbyte>(ref b), out sbyte rem);
                remainder = SysUnsafe.As<sbyte, T>(ref rem);
                return SysUnsafe.As<sbyte, T>(ref quo);
            }
            if (typeof(T) == typeof(byte))
            {
                byte quo = NDDivision.DivmodByte(SysUnsafe.As<T, byte>(ref a), SysUnsafe.As<T, byte>(ref b), out byte rem);
                remainder = SysUnsafe.As<byte, T>(ref rem);
                return SysUnsafe.As<byte, T>(ref quo);
            }
            if (typeof(T) == typeof(short))
            {
                short quo = NDDivision.DivmodInt16(SysUnsafe.As<T, short>(ref a), SysUnsafe.As<T, short>(ref b), out short rem);
                remainder = SysUnsafe.As<short, T>(ref rem);
                return SysUnsafe.As<short, T>(ref quo);
            }
            if (typeof(T) == typeof(ushort))
            {
                ushort quo = NDDivision.DivmodUInt16(SysUnsafe.As<T, ushort>(ref a), SysUnsafe.As<T, ushort>(ref b), out ushort rem);
                remainder = SysUnsafe.As<ushort, T>(ref rem);
                return SysUnsafe.As<ushort, T>(ref quo);
            }
            if (typeof(T) == typeof(char))
            {
                char quo = NDDivision.DivmodChar(SysUnsafe.As<T, char>(ref a), SysUnsafe.As<T, char>(ref b), out char rem);
                remainder = SysUnsafe.As<char, T>(ref rem);
                return SysUnsafe.As<char, T>(ref quo);
            }
            if (typeof(T) == typeof(int))
            {
                int quo = NDDivision.DivmodInt32(SysUnsafe.As<T, int>(ref a), SysUnsafe.As<T, int>(ref b), out int rem);
                remainder = SysUnsafe.As<int, T>(ref rem);
                return SysUnsafe.As<int, T>(ref quo);
            }
            if (typeof(T) == typeof(uint))
            {
                uint quo = NDDivision.DivmodUInt32(SysUnsafe.As<T, uint>(ref a), SysUnsafe.As<T, uint>(ref b), out uint rem);
                remainder = SysUnsafe.As<uint, T>(ref rem);
                return SysUnsafe.As<uint, T>(ref quo);
            }
            if (typeof(T) == typeof(long))
            {
                long quo = NDDivision.DivmodInt64(SysUnsafe.As<T, long>(ref a), SysUnsafe.As<T, long>(ref b), out long rem);
                remainder = SysUnsafe.As<long, T>(ref rem);
                return SysUnsafe.As<long, T>(ref quo);
            }
            if (typeof(T) == typeof(ulong))
            {
                ulong quo = NDDivision.DivmodUInt64(SysUnsafe.As<T, ulong>(ref a), SysUnsafe.As<T, ulong>(ref b), out ulong rem);
                remainder = SysUnsafe.As<ulong, T>(ref rem);
                return SysUnsafe.As<ulong, T>(ref quo);
            }
            if (typeof(T) == typeof(Half))
            {
                Half quo = NDDivision.DivmodHalf(SysUnsafe.As<T, Half>(ref a), SysUnsafe.As<T, Half>(ref b), out Half rem);
                remainder = SysUnsafe.As<Half, T>(ref rem);
                return SysUnsafe.As<Half, T>(ref quo);
            }
            if (typeof(T) == typeof(float))
            {
                float quo = NDDivision.DivmodSingle(SysUnsafe.As<T, float>(ref a), SysUnsafe.As<T, float>(ref b), out float rem);
                remainder = SysUnsafe.As<float, T>(ref rem);
                return SysUnsafe.As<float, T>(ref quo);
            }
            if (typeof(T) == typeof(double))
            {
                double quo = NDDivision.DivmodDouble(SysUnsafe.As<T, double>(ref a), SysUnsafe.As<T, double>(ref b), out double rem);
                remainder = SysUnsafe.As<double, T>(ref rem);
                return SysUnsafe.As<double, T>(ref quo);
            }
            if (typeof(T) == typeof(decimal))
            {
                decimal quo = NDDivision.DivmodDecimal(SysUnsafe.As<T, decimal>(ref a), SysUnsafe.As<T, decimal>(ref b), out decimal rem);
                remainder = SysUnsafe.As<decimal, T>(ref rem);
                return SysUnsafe.As<decimal, T>(ref quo);
            }
            throw new NotSupportedException($"divmod has no {typeof(T).Name} loop");
        }

        /// <summary>
        /// Return <paramref name="x"/> broadcast to <paramref name="shape"/> as a C-contiguous array of
        /// <paramref name="rt"/>. A no-op passthrough (no copy) when <paramref name="x"/> already IS a
        /// C-contiguous, offset-0, non-broadcast array of <paramref name="rt"/> with that shape.
        /// </summary>
        /// <param name="x">The operand (any layout; 0-d or lower-rank operands broadcast).</param>
        /// <param name="shape">The iteration shape (at least one dimension).</param>
        /// <param name="rt">The loop dtype.</param>
        /// <returns><paramref name="x"/> itself, or a fresh C-contiguous copy the caller disposes. Either way the result's
        ///     first element is at its storage's base (offset 0), which the kernels read through <c>.Address</c>.</returns>
        private NDArray MaterializeContiguous(NDArray x, Shape shape, NPTypeCode rt)
        {
            var v = x.Shape.Equals(shape) && !x.Shape.IsBroadcasted ? x : NDArray.broadcast_to(x, shape);
            if (v.GetTypeCode == rt && v.Shape.IsContiguous && !v.Shape.IsBroadcasted)
            {
                if (v.Shape.offset == 0)
                    return v;
                // A C-contiguous view at a non-zero offset (an np.split child, a row of a matrix): the kernels read from
                // .Address, the storage's BASE, and np.ascontiguousarray returns such a view as it is — the divmod / frexp
                // / ldexp kernels read the owner's start (relayout gap RL03: silently wrong values). Its elements are
                // copied to a fresh buffer instead.
                var fresh = v.copy('C');
                if (!ReferenceEquals(v, x))
                    v.Dispose();
                return fresh;
            }
            // asarray(order='C') materializes a C-contiguous resultType copy through any layout,
            // preserving the (ndim>=1) shape — unlike ascontiguousarray it does not reshape 0-d,
            // but this path only runs for ndim>=1 (0-d routes to the compose fallback above).
            var copy = NDArray.ascontiguousarray(v, rt);
            // A broadcast view made here is dead once copied: back to the owner's count rather than the finalizer.
            if (!ReferenceEquals(v, x) && !ReferenceEquals(v, copy))
                v.Dispose();
            return copy;
        }
    }
}
