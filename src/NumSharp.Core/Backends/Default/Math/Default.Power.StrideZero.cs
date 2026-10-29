using System;
using System.Reflection.Emit;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // np.power's float32 / float64 loops — NumPy's zero-exponent-step fast path
        // =============================================================================
        //
        // NumPy 2.4.2's DOUBLE_power / FLOAT_power (loops_umath_fp.dispatch.c.src) test the exponent's
        // STEP in each inner-loop call. When it is 0 they read the exponent once and, for -1, 0, 0.5, 1
        // or 2, write 1.0/x, 1.0, sqrt(x), x or x*x; any other exponent, or a nonzero step, calls pow.
        // The answers differ (757.7617475757002 ** 2: x*x 0x412185f5bb7003c1, pow ...03c2; -0.0 ** 0.5:
        // sqrt -0.0, pow +0.0; -inf ** 0.5: sqrt NaN, pow +inf; sNaN ** 1: x keeps it signalling).
        //
        // Which calls see step 0 is decided by NumPy's ufunc machinery, so it is replayed here through
        // the shared port in DefaultEngine.UfuncInnerSteps.cs: a 0-d exponent has step 0 on every route;
        // a one-element exponent of the base's own shape does NOT in the trivial loop ((1,) ** (1,) is
        // pow, its 1-D step is 8 bytes), but does through NpyIter ((5,) ** (1,) is the fast path); a
        // column exponent (R,1) against (R,n) reaches the loop with step 0 only when NpyIter does not
        // buffer it, which its cost model decides from n and the buffer size (n > 4096 for the default
        // 8192-element buffer and contiguous operands). The decision is uniform over the call, so the
        // whole result is either NumPy's fast-path values or pow's.
        //
        // Execution:
        //   * a single-valued exponent (0-d, one element, or a stride-0 broadcast of one) with a fast value:
        //     the matching SIMD unary kernel (square / sqrt / reciprocal / positive) or a fill with 1,
        //     computed in the loop dtype straight into the result, out= and where= included;
        //   * an exponent that varies but reaches NumPy's calls with step 0 (wide columns): one NDIter pass
        //     whose per-element body is NumPy's per-call branch (EmitPowerStrideZeroBody), with a vector
        //     body (EmitPowerStrideZeroVectorBody) when both inputs are already the loop dtype — the
        //     broadcast loops hand it a uniform exponent vector, so one compare picks one vector operation;
        //   * everything else: the general power kernel (pow on every element).
        // float_power never comes here: its loop (PyUFunc_dd_d with pow) has no fast path.
        // =============================================================================

        /// <summary>The operation NumPy's stride-0 float power loop substitutes for <c>pow</c>, by exponent value.</summary>
        private enum PowerStrideZeroOp
        {
            /// <summary>Not one of the five exponents: <c>pow</c> even with step 0.</summary>
            None,
            /// <summary>Exponent -1: <c>1.0 / x</c> (np.reciprocal's float loop).</summary>
            Reciprocal,
            /// <summary>Exponent 0 (either sign): <c>1.0</c>, whatever <c>x</c> is (a NaN included).</summary>
            One,
            /// <summary>Exponent 0.5: <c>sqrt(x)</c> (sqrt(-0.0) is -0.0, sqrt of a negative is NaN).</summary>
            Sqrt,
            /// <summary>Exponent 1: <c>x</c> itself, bits untouched (np.positive's float loop).</summary>
            Identity,
            /// <summary>Exponent 2: <c>x * x</c> (np.square's float loop).</summary>
            Square,
        }

        /// <summary>
        /// np.power over a float32 / float64 loop, as NumPy computes it: NumPy's zero-exponent-step fast path where
        /// NumPy's inner-loop calls see the exponent with step 0, <c>pow</c> everywhere else.
        /// </summary>
        /// <param name="lhs">The bases.</param>
        /// <param name="rhs">The exponents.</param>
        /// <param name="loop">The loop dtype (<see cref="NPTypeCode.Single"/> or <see cref="NPTypeCode.Double"/>).</param>
        /// <param name="typeCode">The <c>dtype=</c> request, or null (its input casts are validated first, as the
        /// general route does).</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The bool where= mask, or null.</param>
        /// <returns>The powers: a fresh array of <paramref name="loop"/>, or <paramref name="out"/>.</returns>
        /// <exception cref="ArgumentException">An input cannot reach a requested <c>dtype=</c> loop, the mask is not bool,
        /// the out is read-only, or the loop cannot cast into the out (the general route's texts, the ufunc named
        /// 'power').</exception>
        /// <exception cref="IncorrectShapeException">The operands do not broadcast, or the out / mask does not fit
        /// the broadcast shape.</exception>
        /// <remarks>
        /// The checks run in the general route's order (<see cref="ExecuteBinaryOp"/> then
        /// <c>ExecuteBinaryUfuncInto</c>), so a call raises the same error whichever way it would have computed.
        /// </remarks>
        private NDArray PowerFloatLoop(NDArray lhs, NDArray rhs, NPTypeCode loop, NPTypeCode? typeCode,
                                       NDArray @out, NDArray where)
        {
            // One exponent value reaches every NumPy call (0-d, one element, or a stride-0 broadcast of one): its
            // fast operation is uniform. A value that is not one of the five takes pow in NumPy whatever the step.
            bool singleValued = rhs.size > 0 && (rhs.size == 1 || IsAllStrideZero(rhs.Shape));
            PowerStrideZeroOp fastOp = PowerStrideZeroOp.None;
            if (singleValued)
            {
                fastOp = ClassifyStrideZeroExponent(ReadExponentInLoop(rhs, loop));
                if (fastOp == PowerStrideZeroOp.None)
                    return ExecuteBinaryOp(lhs, rhs, BinaryOp.Power, @out, where, typeCode);

                // The common x ** scalar: a 0-d exponent adds no dimension, cannot fail to broadcast and has step 0 on
                // every route, so the plain call needs neither the shape prologue nor the step decision — only the
                // dtype= input casts the general route would check first.
                if (rhs.ndim == 0 && @out is null && where is null)
                {
                    if (typeCode.HasValue && typeCode.Value != np._FindCommonType(lhs, rhs))
                        ValidateBinaryInputCasts(lhs.GetTypeCode, rhs.GetTypeCode, typeCode.Value, UfuncName(BinaryOp.Power));
                    return fastOp == PowerStrideZeroOp.One
                        ? np.ones(PowerPlainResultShape(lhs, rhs, lhs.Shape.Clean()), loop)
                        : ExecuteUnaryOp(lhs, StrideZeroUnaryOp(fastOp), loop);
                }
            }

            // NumPy's prologue first, so a bad call raises what the general route raises, in its order.
            Shape iterShape = ValidatePowerUfuncCall(lhs, rhs, loop, typeCode, @out, where);

            // No element, or NumPy's calls step through the exponent: pow on every element (the general kernel).
            if (iterShape.size == 0 || !NumPyPowerTakesStrideZeroLoop(lhs, rhs, @out, where, loop, iterShape.dimensions))
                return ExecuteBinaryOp(lhs, rhs, BinaryOp.Power, @out, where, typeCode);

            return singleValued
                ? PowerStrideZeroUniform(lhs, rhs, fastOp, loop, iterShape, @out, where)
                : PowerStrideZeroElementwise(lhs, rhs, loop, iterShape, @out, where);
        }

        /// <summary>
        /// Whether NumPy 2.4.2's float power loop sees the exponent with step 0 in its inner-loop calls for this call
        /// (so an exponent of -1, 0, 0.5, 1 or 2 takes the fast path), through its trivial loop or its iterator.
        /// </summary>
        /// <param name="lhs">The bases as passed. A 0-d base counts as a Python scalar (NumSharp cannot tell a C#
        /// literal from a 0-d array), as NumPy converts a Python scalar into a fresh 0-d array.</param>
        /// <param name="rhs">The exponents as passed, with the same 0-d rule.</param>
        /// <param name="out">The out= array, or null when NumPy allocates the result.</param>
        /// <param name="where">The where= mask, or null. A mask disables the trivial loop.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="dims">The broadcast iteration shape (inputs, out and mask together).</param>
        /// <returns>True when every inner-loop call has a zero exponent step.</returns>
        /// <remarks>
        /// <para>Probed against NumPy 2.4.2 with values whose fast and pow results differ: a 0-d exponent takes the
        /// fast path on the plain, out= and where= routes; (5,) ** (1,) and a (2,1) ** (1,) one-element exponent take it
        /// (NpyIter, step 0); (1,) ** (1,) and 0-d ** (1,) do not (the trivial loop's 1-D step is the exponent's own
        /// stride); a (2,1) column exponent against a (2,3) base does not (NpyIter buffers it into a contiguous
        /// buffer), against (2,5000) it does (the cost model leaves it unbuffered, step 0 per row).</para>
        /// <para>A 0-d exponent short-circuits: its step is 0 in the trivial loop, unbuffered, and buffered alike.</para>
        /// </remarks>
        internal static bool NumPyPowerTakesStrideZeroLoop(NDArray lhs, NDArray rhs, NDArray @out, NDArray where,
                                                           NPTypeCode loop, long[] dims)
        {
            if (rhs.ndim == 0)
                return true;

            var ops = new UfuncLoopOperand[2]
            {
                UfuncLoopOperand.OfInput(lhs, loop),
                UfuncLoopOperand.OfInput(rhs, loop),
            };
            Span<long> steps = stackalloc long[2];
            NumPyUfuncInputSteps(ops, @out, where, loop, dims, steps);
            return steps[1] == 0;
        }

        /// <summary>
        /// The general route's validation for a power call, in its order, without computing anything: the
        /// <c>dtype=</c> input casts (<see cref="ExecuteBinaryOp"/>), then for out= / where= the read-only out, the mask
        /// dtype, the out cast, the input broadcast and the out / mask join (<c>ExecuteBinaryUfuncInto</c>); a plain
        /// call only broadcasts the inputs.
        /// </summary>
        /// <param name="lhs">The bases.</param>
        /// <param name="rhs">The exponents.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="typeCode">The <c>dtype=</c> request, or null.</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <returns>The clean (C-order, offset 0) iteration shape: the out's dims when one is given.</returns>
        /// <exception cref="ArgumentException">An input cast, the mask dtype, a read-only out or the out cast is
        /// refused.</exception>
        /// <exception cref="IncorrectShapeException">The shapes do not broadcast, or the out / mask does not fit.</exception>
        private static Shape ValidatePowerUfuncCall(NDArray lhs, NDArray rhs, NPTypeCode loop, NPTypeCode? typeCode,
                                                    NDArray @out, NDArray where)
        {
            // ExecuteBinaryOp: a dtype= loop other than the promoted one checks both input casts before anything else.
            if (typeCode.HasValue && typeCode.Value != np._FindCommonType(lhs, rhs))
                ValidateBinaryInputCasts(lhs.GetTypeCode, rhs.GetTypeCode, typeCode.Value, UfuncName(BinaryOp.Power));

            if (@out is null && where is null)
                return Broadcast(lhs.Shape, rhs.Shape).LeftShape.Clean();

            // ExecuteBinaryUfuncInto's prologue: read-only out, mask dtype, out cast, then the shapes.
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape, "output array");
            ValidateWhereMask(where);
            if (@out is not null)
                ValidateOutCast(loop, @out.typecode, UfuncName(BinaryOp.Power));
            var (leftShape, _) = Broadcast(lhs.Shape, rhs.Shape);
            return ResolveUfuncIterationShape(leftShape.Clean(), new[] { lhs, rhs }, @out, where).Clean();
        }

        /// <summary>
        /// The power result for an exponent with ONE value that NumPy's calls see with step 0: the fast operation for
        /// that value over the bases, cast to the loop dtype first, written into the result or out.
        /// </summary>
        /// <param name="lhs">The bases (any dtype that reaches the loop).</param>
        /// <param name="rhs">The exponent (only its shape is read here: it may add leading dimensions).</param>
        /// <param name="fastOp">The fast operation for the exponent's value (never <see cref="PowerStrideZeroOp.None"/>).</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="iterShape">The validated iteration shape.</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <returns>A fresh result (the binary route's layout: F only when the operands vote F), or <paramref name="out"/>.</returns>
        /// <remarks>
        /// The unary kernels are bit-identical to NumPy's fast bodies: <c>x * x</c>, <c>sqrt</c>, <c>1.0 / x</c> (the
        /// float32 quotient equals NumPy's double quotient rounded to float) and a raw copy, after the same input cast
        /// NumPy's iterator makes. Their own validation passes (it repeats checks the power prologue already made),
        /// and an out that overlaps the bases is copied first by their iterator, as NumPy's COPY_IF_OVERLAP does.
        /// </remarks>
        private NDArray PowerStrideZeroUniform(NDArray lhs, NDArray rhs, PowerStrideZeroOp fastOp, NPTypeCode loop,
                                               Shape iterShape, NDArray @out, NDArray where)
        {
            if (@out is null && where is null)
            {
                // A plain call: NumPy writes 1.0 for exponent 0 without reading the bases.
                if (fastOp == PowerStrideZeroOp.One)
                    return np.ones(PowerPlainResultShape(lhs, rhs, iterShape), loop);
                // The result has the bases' shape: the unary op's own allocation (C, or F for a strictly F base — the
                // binary route's choice for a one-element exponent) is the result.
                if (SameDims(lhs.Shape, iterShape))
                    return ExecuteUnaryOp(lhs, StrideZeroUnaryOp(fastOp), loop);
                // The exponent adds leading axes ((3,) ** (1,1) is (1,3)): compute into a result of the broadcast shape.
                var result = new NDArray(loop, PowerPlainResultShape(lhs, rhs, iterShape), false);
                ExecuteUnaryOp(lhs, StrideZeroUnaryOp(fastOp), loop, result, null);
                return result;
            }

            // where= without out: NumPy leaves the masked-off positions uninitialized (fillZeros: false).
            NDArray target = @out ?? new NDArray(loop, iterShape, false);

            if (where is null)
            {
                // out= alone: the out's cast of 1.0 is exact in every float / complex dtype same_kind reaches.
                if (fastOp == PowerStrideZeroOp.One)
                    target.fill(1.0);
                else
                    ExecuteUnaryOp(lhs, StrideZeroUnaryOp(fastOp), loop, target, null);
                return target;
            }

            // where=: NumPy's masked loop computes the fast operation for the mask-true runs only. Computing it for
            // every base element into a temporary and then making np.copyto's masked bit copy into the target gives the
            // same values — the operation is elementwise, the temporary is read only after it is complete (as
            // COPY_IF_OVERLAP would make an overlapping out), and the copy casts to the out's dtype as the loop's
            // output does — 2.5-9x faster than one unary kernel call per mask-true run (measured on runs of two).
            if (fastOp == PowerStrideZeroOp.One)
            {
                using var one = NDArray.Scalar(1.0, loop);
                np.copyto(target, one, casting: "unsafe", where: where);
            }
            else
            {
                using var computed = ExecuteUnaryOp(lhs, StrideZeroUnaryOp(fastOp), loop);
                np.copyto(target, computed, casting: "unsafe", where: where);
            }
            return target;
        }

        /// <summary>
        /// The power result for an exponent that varies but reaches NumPy's calls with step 0 (a wide column, a
        /// stride-0 axis NpyIter leaves unbuffered): one NDIter pass whose per-element body is NumPy's per-call branch
        /// (<see cref="DirectILKernelGenerator.EmitPowerStrideZeroBody"/>) — every element of such a NumPy call carries
        /// the call's one exponent, so testing each element's exponent is NumPy's test.
        /// </summary>
        /// <param name="lhs">The bases.</param>
        /// <param name="rhs">The exponents.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="iterShape">The validated iteration shape.</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <returns>A fresh result, or <paramref name="out"/>.</returns>
        /// <remarks>
        /// <para>The iterator is configured as the power ufunc's out= / where= route configures it (COPY_IF_OVERLAP for
        /// an overlapping out, buffers for an out of another dtype, the mask as an ARRAYMASK operand); only the kernel
        /// body differs. The scalar body converts each operand to the loop dtype itself, so mixed input dtypes need no
        /// buffer.</para>
        /// <para>When both inputs are the loop dtype the kernel also gets the vector body
        /// (<see cref="DirectILKernelGenerator.EmitPowerStrideZeroVectorBody"/>): the shell's broadcast loops and the
        /// 2-D block kernel load a stride-0 exponent once per row and Vector.Create it, so a row costs one compare per
        /// vector plus the vector operation, where the scalar body paid a call per element ((2, 5000) float64 with a
        /// [2, 0.5] column: 0.85× NumPy before).</para>
        /// </remarks>
        private unsafe NDArray PowerStrideZeroElementwise(NDArray lhs, NDArray rhs, NPTypeCode loop, Shape iterShape,
                                                          NDArray @out, NDArray where)
        {
            NDArray target = @out ?? new NDArray(loop,
                where is null ? PowerPlainResultShape(lhs, rhs, iterShape) : iterShape, false);

            NPTypeCode lhsType = lhs.GetTypeCode, rhsType = rhs.GetTypeCode;
            Action<ILGenerator> scalarBody = il => DirectILKernelGenerator.EmitPowerStrideZeroBody(il, lhsType, rhsType, loop);
            // The vector body needs every kernel operand in the loop dtype (the shell's SIMD rule). The out operand
            // always is: a cast out is a buffered loop-dtype operand. So the body depends only on the key's dtypes.
            Action<ILGenerator> vectorBody = lhsType == loop && rhsType == loop
                ? il => DirectILKernelGenerator.EmitPowerStrideZeroVectorBody(il, loop)
                : null;
            var cacheKey = InnerLoopKernelKey.PowerStrideZero(lhsType, rhsType, loop);

            // A dtype-mismatched out is a CAST operand: the kernel writes the loop dtype into its buffer and the flush
            // casts (UNSAFE: the cast was validated as same_kind by the prologue), as the binary out= route does.
            bool outNeedsCast = target.typecode != loop;
            var globalFlags = NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            if (outNeedsCast)
            {
                globalFlags |= NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER | NDIterGlobalFlags.DELAY_BUFALLOC;
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
            }

            if (where is null)
            {
                NPTypeCode[] opDtypes = outNeedsCast ? new[] { lhsType, rhsType, loop } : null;
                using var iter = NDIterRef.MultiNew(
                    3, new[] { lhs, rhs, target },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncBinaryOutFlags,
                    opDtypes);
                iter.ExecuteElementWiseBinary(lhsType, rhsType, loop, scalarBody, vectorBody, cacheKey);
            }
            else
            {
                NPTypeCode[] opDtypes = outNeedsCast ? new[] { lhsType, rhsType, loop, NPTypeCode.Empty } : null;
                using var iter = NDIterRef.MultiNew(
                    4, new[] { lhs, rhs, target, where },
                    globalFlags, NPY_ORDER.NPY_CORDER, casting,
                    s_ufuncBinaryOutMaskedFlags,
                    opDtypes);
                iter.ExecuteElementWise(new[] { lhsType, rhsType, loop }, scalarBody, vectorBody, cacheKey);
            }
            return target;
        }

        /// <summary>
        /// The layout of a fresh plain power result: the binary route's choice (<see cref="ExecuteBinaryOp"/>) — F when
        /// the operands vote F (<see cref="ShouldProduceFContigOutput(NDArray, NDArray, Shape)"/>), C otherwise — so the
        /// fast and general paths of one call lay their results out alike.
        /// </summary>
        /// <param name="lhs">The bases.</param>
        /// <param name="rhs">The exponents.</param>
        /// <param name="iterShape">The clean iteration shape.</param>
        /// <returns>An offset-0 shape of <paramref name="iterShape"/>'s dims, C or F ordered.</returns>
        private static Shape PowerPlainResultShape(NDArray lhs, NDArray rhs, Shape iterShape)
            => ShouldProduceFContigOutput(lhs, rhs, iterShape)
                ? Shape.Adopt((long[])iterShape.dimensions.Clone(), 'F')
                : iterShape;

        /// <summary>
        /// True when every axis of <paramref name="s"/> longer than 1 has stride 0, i.e. the array repeats one element
        /// (a broadcast of a scalar). Such an exponent carries one value, like a one-element one.
        /// </summary>
        /// <param name="s">The shape to test.</param>
        /// <returns>True for a scalar broadcast (and for any shape with no axis longer than 1).</returns>
        private static bool IsAllStrideZero(in Shape s)
        {
            for (int d = 0; d < s.NDim; d++)
                if (s.dimensions[d] > 1 && s.strides[d] != 0)
                    return false;
            return true;
        }

        /// <summary>
        /// The exponent's value as NumPy's loop compares it: the first element converted to the loop dtype (a float64
        /// exponent in a float32 loop is rounded to float32 first, so 0.50000001 becomes 0.5f and takes sqrt).
        /// </summary>
        /// <param name="rhs">A non-empty exponent (one value).</param>
        /// <param name="loop">The loop dtype (float32 or float64).</param>
        /// <returns>The value in the loop's precision, widened to double for the comparisons.</returns>
        private static double ReadExponentInLoop(NDArray rhs, NPTypeCode loop)
        {
            double v = ReadScalarAsDouble(rhs);
            return loop == NPTypeCode.Single ? (float)v : v;
        }

        /// <summary>
        /// The fast operation NumPy's stride-0 float power loop runs for an exponent value: <c>in2 == -1.0</c>,
        /// <c>0.0</c>, <c>0.5</c>, <c>1.0</c>, <c>2.0</c> in that order (so -0.0 is the 0 case and NaN none).
        /// </summary>
        /// <param name="e">The exponent in the loop's precision.</param>
        /// <returns>The substituted operation, or <see cref="PowerStrideZeroOp.None"/> for pow.</returns>
        private static PowerStrideZeroOp ClassifyStrideZeroExponent(double e)
            => e == -1.0 ? PowerStrideZeroOp.Reciprocal
             : e == 0.0 ? PowerStrideZeroOp.One
             : e == 0.5 ? PowerStrideZeroOp.Sqrt
             : e == 1.0 ? PowerStrideZeroOp.Identity
             : e == 2.0 ? PowerStrideZeroOp.Square
             : PowerStrideZeroOp.None;

        /// <summary>The unary ufunc kernel whose float loop is the fast operation's body.</summary>
        /// <param name="op">A fast operation other than <see cref="PowerStrideZeroOp.One"/> and <see cref="PowerStrideZeroOp.None"/>.</param>
        /// <returns>The unary op.</returns>
        /// <exception cref="ArgumentOutOfRangeException"><paramref name="op"/> has no unary kernel (One or None).</exception>
        private static UnaryOp StrideZeroUnaryOp(PowerStrideZeroOp op) => op switch
        {
            PowerStrideZeroOp.Reciprocal => UnaryOp.Reciprocal,
            PowerStrideZeroOp.Sqrt => UnaryOp.Sqrt,
            PowerStrideZeroOp.Identity => UnaryOp.Positive,
            PowerStrideZeroOp.Square => UnaryOp.Square,
            _ => throw new ArgumentOutOfRangeException(nameof(op), op, "No unary kernel computes this power fast path."),
        };
    }
}
