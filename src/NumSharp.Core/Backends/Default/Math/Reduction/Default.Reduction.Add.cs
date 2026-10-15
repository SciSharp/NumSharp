using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.sum</c> / <c>ndarray.sum</c> / <c>np.add.reduce</c>: the add reduction of <paramref name="arr"/>
        ///     over <paramref name="axis_"/> (every axis when null), computed the way NumPy's
        ///     <c>PyUFunc_ReduceWrapper</c> computes it — the result is first assigned the loop's IDENTITY (+0.0 for an
        ///     inexact loop, 0 / False otherwise) and only then does the add loop run over every element. A sum of only
        ///     negative zeros is therefore +0.0 (each part of a complex), including the one-element and length-1-axis
        ///     reductions that look like a plain copy.
        /// </summary>
        /// <param name="arr">The array to reduce; any dtype and any layout (contiguous, strided, reversed, F, broadcast, 0-d).</param>
        /// <param name="axis_">The axis to reduce (negative counts from the last), or null to reduce every axis. A 0-d
        /// array also accepts 0 and -1, which ufunc.reduce reads as "every axis" (NumPy's backward-compatibility rule).</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: the LOOP dtype — every element is cast to it (unsafely, element by
        /// element) before it is added, and the sum is computed and returned in it. Null selects the NEP 50 accumulator
        /// (int64 for bool and signed integers, uint64 for unsigned integers and Char, the input dtype otherwise).</param>
        /// <param name="out">Optional output that receives the result and is returned; its shape must be the result's.</param>
        /// <returns>The sum: a read-only 0-d scalar for a full reduction without keepdims, otherwise a fresh array (or
        /// <paramref name="out"/>).</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is out of range for <paramref name="arr"/>'s rank.</exception>
        /// <exception cref="IncorrectShapeException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <remarks>
        ///     <para>
        ///     The loop dtype decides the arithmetic, so three dtype relations are told apart before anything is summed:
        ///     the default accumulator (or a <c>dtype=</c> naming it) and the input's own dtype keep the existing
        ///     same-type / widening kernels; any OTHER loop casts the input first (<see cref="ReduceAddInCastLoop"/>),
        ///     exactly NumPy's semantics (its buffered iterator casts each element before the add loop sees it). The
        ///     kernels only ever converted per element for a few widening pairs: float→int accumulated in double and
        ///     truncated the TOTAL, a negative int saturated into an unsigned loop, and a complex / float16 loop read the
        ///     input buffer as its own dtype (out-of-bounds reads, a CLR crash for complex→real).
        ///     </para>
        ///     <para>
        ///     The bool loop is NumPy's logical-OR add (<c>True + True</c> is <c>True</c>), i.e. <c>np.any</c>.
        ///     </para>
        /// </remarks>
        public override NDArray ReduceAdd(NDArray arr, int? axis_, bool keepdims = false, DType dtype = null, NDArray @out = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;
            var accumulatorType = inputType.GetAccumulatingType();
            // NumPy's loop dtype: the dtype= request, else the NEP 50 accumulator.
            var loopType = typeCode ?? accumulatorType;

            if (shape.IsEmpty)
            {
                // The UNINITIALIZED shape sentinel (not a size-0 array — that is the next branch): the identity of the loop.
                var defaultVal = loopType.GetDefaultValue();
                if (@out is not null) { @out.SetAtIndex(defaultVal, 0); return @out; }
                return NDArray.Scalar(defaultVal);
            }

            if (shape.size == 0)
                return HandleEmptyArrayReduction(arr, axis_, keepdims, typeCode, @out, ReductionOp.Sum);

            // The flat sum of a float loop is NumPy's exact add.reduce — the identity start, NpyIter's call schedule and a
            // pairwise sum per call — whether the input already is the loop dtype or is cast into it (NumPyAddReduce). One
            // element keeps SumOfOneElement below; a broadcast operand and the rest of the engine's declines keep the
            // routes that follow.
            if (axis_ == null && shape.size > 1 && NumPyAddReduce.SupportsLoop(loopType))
            {
                var exact = TryExactFlatSum(arr, keepdims, loopType, @out);
                if (exact is not null)
                    return exact;
            }

            // An axis that leaves ONE output — a 1-D operand's only axis, or any axis whose siblings all have extent 1
            // ((n,1) along 0, (1,n,1) along 1) — is the flat reduction in NumPy: its iterator drops the extent-1 axes, so
            // the call schedule is the axis=None one, np.getbufsize() fills of a cast loop included. The axis routes
            // below summed a cast loop's whole cast copy in one pairwise call instead (float64 -> float32 over 20,000
            // elements: 0x4706ada8 where NumPy gives 0x4706ada6; plan docs/plans/api-parameter-parity.md, item X5).
            // A cast-free loop over a forward axis stays on the axis kernel: both routes are then one pairwise call over
            // the run, and the kernel measured 10-20 % faster for it (1-D float64 at 100K: 6.2 vs 7.4 µs). A REVERSED
            // axis takes the flat engine even without a cast: NumPy walks it in logical order (DONT_NEGATE_STRIDES),
            // the flat engine does too, and the axis kernel walks memory order (finding 5 of the plan).
            if (axis_ != null && shape.size > 1 && @out is null && NumPyAddReduce.SupportsLoop(loopType))
            {
                int oneAxis = NormalizeAxis(axis_.Value, shape.NDim);
                if (IsOnlyNonUnitAxis(shape, oneAxis) && (loopType != inputType || shape.strides[oneAxis] < 0))
                {
                    var exact = TryExactOneOutputAxisSum(arr, oneAxis, keepdims, loopType);
                    if (exact is not null)
                        return exact;
                }
            }

            // ONE element per output — a one-element input reduced flat (or over its only axis; a 0-d input reads axis 0
            // and -1 as every axis) — is still identity + x in NumPy, never a copy of x. HandleScalarReduction stays the
            // copy amin/amax need (they have no identity). Taken before the cast-loop and bool-loop branches: the
            // identity add casts the lone element into any loop itself (WriteIdentityPlus), and for a bool loop
            // False OR bool(x) is that cast, so the cast copy those branches would make first is one allocation and one
            // pass too many.
            if (shape.size == 1 && (axis_ == null || shape.NDim <= 1))
            {
                if (axis_ != null)
                {
                    if (shape.NDim == 0)
                    {
                        if (axis_.Value != 0 && axis_.Value != -1)
                            throw new AxisError(axis_.Value, 0);
                    }
                    else
                        NormalizeAxis(axis_.Value, shape.NDim);
                }

                return SumOfOneElement(arr, keepdims, loopType, @out);
            }

            // A length-1 axis gives every output exactly one element: identity + x again (HandleTrivialAxisReduction's
            // copy stays for amin/amax/nanmin/nanmax, which have no identity) — before the cast loop for the same reason.
            if (axis_ != null && shape.NDim > 0)
            {
                var unitAxis = NormalizeAxis(axis_.Value, arr.ndim);
                if (shape[unitAxis] == 1)
                    return SumOverLengthOneAxis(arr, unitAxis, keepdims, loopType, @out);
            }

            // A bool sum along an axis is the count of its nonzero elements: exact in int64, and in float64 (a count never
            // reaches 2^53), whatever the order. np.evaluate's CountNonzero axis stream counts a C- or F-contiguous operand
            // in NumPy's result layout; the axis kernel walked every output's column one element at a time (np.sum(bool
            // (2500,4000), axis=0, dtype=float64) took 27 ms, NumPy 5 ms; plan item X7).
            if (axis_ != null && inputType == NPTypeCode.Boolean && @out is null
                && (loopType == NPTypeCode.Int64 || loopType == NPTypeCode.Double))
            {
                var counted = TryCountBoolAxisSum(arr, NormalizeAxis(axis_.Value, arr.ndim), keepdims, loopType);
                if (counted is not null)
                    return counted;
            }

            // A loop dtype that is neither the input's own nor its default accumulator: NumPy casts every element into it
            // before the add loop runs, so the cast copy is reduced in its own dtype (no kernel converts per element for
            // every such pair — see the remarks). The axis kernels' own widening paths are NOT the cast copy's order in
            // general — float32 -> float64 and float16 -> float32 summed a contiguous axis differently from NumPy — but
            // float32 -> float64 over a non-innermost axis is (IsSequentialWideningAxisSum), and so is an integer ->
            // float64 sum the axis kernel provably computes exactly (IsExactIntegerWideningAxisSum): both run directly.
            if (loopType != inputType && loopType != accumulatorType
                && !(axis_ != null && (IsSequentialWideningAxisSum(arr, axis_.Value, inputType, loopType)
                                       || IsExactIntegerWideningAxisSum(arr, axis_.Value, inputType, loopType))))
                return ReduceAddInCastLoop(arr, axis_, keepdims, loopType, @out);

            // NumPy's bool add loop is logical OR: a bool loop over a bool input is np.any (identity False).
            if (loopType == NPTypeCode.Boolean)
                return ReduceAddBooleanLoop(arr, axis_, keepdims, @out);

            if (axis_ == null)
                return HandleElementWiseSum(arr, keepdims, typeCode, @out);

            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var outputType = loopType;

            // Half AXIS sum: reproduce NumPy's HALF_add reduce (float32 intermediate, narrowed per
            // inner-loop call) in a FLOAT32 shadow accumulator, then narrow to Half ONCE. The NDIter
            // HalfSumKernel rounds the shadow to float16 precision (RoundToF16) per call — PINNED
            // (contiguous reduced axis) folds the stripe in float32 pairwise (ones(4096)->4096); SLAB
            // (reduced axis outer) rounds per step and SATURATES (ones((4096,3),axis=0)->2048). The
            // float32 shadow (vs re-widening the half accumulator from memory every step) + the
            // in-place RoundToF16 is what makes the large-kept SLAB beat NumPy instead of losing to
            // its hardware F16C. The OLD Double-accumulator path returned 4096 for that SLAB case — a
            // real divergence this fixes. dtype=float16 on a float16 input IS this loop (NumPy runs the same
            // HALF_add), so the explicit spelling takes it too; other loops arrived here already cast.
            if (arr.typecode == NPTypeCode.Half && loopType == NPTypeCode.Half)
            {
                // Both temporaries are this branch's own intermediates: `wide` is consumed by the
                // narrowing astype COPY, and `halfResult` is element-copied into @out when one is
                // given — leaving either undisposed stranded a pooled buffer per f16 axis sum
                // (caught by UndisposedIntermediateTests via the instance oracle tier, 2026-09-18).
                var wide = ExecuteAxisReduction(arr, axis, keepdims, NPTypeCode.Single, null, ReductionOp.Sum);
                NDArray halfResult;
                try
                {
                    halfResult = wide.astype(NPTypeCode.Half);
                }
                finally
                {
                    wide.Dispose();
                }
                if (@out is null) return halfResult;
                try
                {
                    for (long i = 0; i < halfResult.size; i++) @out.SetAtIndex(halfResult.GetAtIndex(i), i);
                }
                finally
                {
                    halfResult.Dispose();
                }
                return @out;
            }

            return ExecuteAxisReduction(arr, axis, keepdims, outputType, @out, ReductionOp.Sum);
        }

        /// <summary>
        ///     Whether <c>np.sum(arr, axis, dtype=float64)</c> of a float32 <paramref name="arr"/> may run the axis kernel's
        ///     own float32 → float64 path instead of reducing a cast copy: when the reduced axis is NOT the operand's
        ///     smallest-|stride| axis, every output is a SEQUENTIAL sum (<c>out += next slab</c>) in both NumPy's buffered
        ///     loop and that kernel, and the per-element conversion is exact, so the bits are the cast copy's — 10 of 10
        ///     probed layouts (C, F, strided rows / columns, reversed, 3-D, transposed 3-D), and NumPy itself on C axis 0 /
        ///     F axis 1. Over the innermost axis NumPy sums each output pairwise, which only the cast copy reproduces (the
        ///     kernel differed on all 8 such probes), so that case keeps the copy.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="axis">The reduced axis (negative counts from the last; out of range returns false so the copy
        /// route raises NumPy's AxisError).</param>
        /// <param name="src">The input dtype.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <returns>True for a float32 → float64 reduction over a non-innermost axis of a non-broadcast operand.</returns>
        private static bool IsSequentialWideningAxisSum(NDArray arr, int axis, NPTypeCode src, NPTypeCode loop)
        {
            if (src != NPTypeCode.Single || loop != NPTypeCode.Double)
                return false;
            var shape = arr.Shape;
            int nd = shape.NDim;
            if (axis < -nd || axis >= nd || nd < 2 || shape.IsBroadcasted)
                return false;
            int ax = axis < 0 ? axis + nd : axis;
            if (shape.dimensions[ax] < 2)
                return false;
            long axisStride = Math.Abs(shape.strides[ax]);
            for (int d = 0; d < nd; d++)
                if (d != ax && shape.dimensions[d] > 1 && Math.Abs(shape.strides[d]) < axisStride)
                    return true;   // some kept axis is strictly tighter: the reduced axis is not the innermost
            return false;          // (a tie — two axes of one stride — counts as innermost and keeps the copy)
        }

        /// <summary>
        ///     Whether <c>np.sum(arr, axis, dtype=float64)</c> of an integer <paramref name="arr"/> may run the axis kernel
        ///     (its integer → float64 widening path) instead of reducing a cast copy, because the kernel's result is then
        ///     provably NumPy's bits:
        ///     <list type="bullet">
        ///         <item>an input of 32 bits or less whose largest possible sum, <c>axis length × max |x|</c>, is at most
        ///         2^53: every partial sum is an integer float64 holds exactly, so every summation order (NumPy's pairwise
        ///         innermost order, the kernel's accumulators) gives the same result;</item>
        ///         <item>an int64 / uint64 input over a NON-innermost axis: its sums can round, but every route the
        ///         kernel takes for these two pairs is sequential per output (the widening kernel keeps them off its
        ///         multi-accumulator path), which is NumPy's slab order (<c>out += next slab</c>), and the per-element
        ///         conversion is the same correctly rounded one NumPy's cast makes.</item>
        ///     </list>
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="axis">The reduced axis (negative counts from the last; out of range returns false so the copy
        /// route raises NumPy's AxisError).</param>
        /// <param name="src">The input dtype.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <returns>True when the axis kernel's sum is NumPy's; false keeps the cast copy (an int64 / uint64 innermost
        /// axis, which NumPy sums pairwise, or a bound that does not hold).</returns>
        private static bool IsExactIntegerWideningAxisSum(NDArray arr, int axis, NPTypeCode src, NPTypeCode loop)
        {
            if (loop != NPTypeCode.Double)
                return false;
            // The largest magnitude one element of the input dtype holds; 0 marks a 64-bit integer (no useful bound).
            double maxAbs = IntegerMaxMagnitude(src);
            if (maxAbs < 0)
                return false;
            var shape = arr.Shape;
            int nd = shape.NDim;
            if (axis < -nd || axis >= nd || shape.IsBroadcasted)
                return false;
            int ax = axis < 0 ? axis + nd : axis;
            if (maxAbs > 0)
                return shape.dimensions[ax] * maxAbs <= 9007199254740992.0;   // 2^53, computed in double: no overflow

            // int64 / uint64: only where some kept axis is strictly tighter than the reduced one (a slab reduction).
            if (nd < 2 || shape.dimensions[ax] < 2)
                return false;
            long axisStride = Math.Abs(shape.strides[ax]);
            for (int d = 0; d < nd; d++)
                if (d != ax && shape.dimensions[d] > 1 && Math.Abs(shape.strides[d]) < axisStride)
                    return true;
            return false;
        }

        /// <summary>
        ///     The largest magnitude one element of an integer, bool or char dtype holds — the bound that makes a sum of
        ///     <c>n</c> such elements exact in float64 whenever <c>n × bound ≤ 2^53</c> (every partial sum is then an
        ///     integer float64 holds, so every summation order gives the same, correctly rounded result).
        /// </summary>
        /// <param name="src">The element dtype.</param>
        /// <returns>The bound (1 for bool, 255 for uint8, 2^31 for int32, …); 0 for int64 / uint64, whose elements alone
        /// can exceed 2^53 (no useful bound); -1 for a dtype that is not an integer, bool or char.</returns>
        internal static double IntegerMaxMagnitude(NPTypeCode src) => src switch
        {
            NPTypeCode.Boolean => 1.0,
            NPTypeCode.SByte => 128.0,
            NPTypeCode.Byte => 255.0,
            NPTypeCode.Int16 => 32768.0,
            NPTypeCode.UInt16 or NPTypeCode.Char => 65535.0,
            NPTypeCode.Int32 => 2147483648.0,
            NPTypeCode.UInt32 => 4294967295.0,
            NPTypeCode.Int64 or NPTypeCode.UInt64 => 0.0,
            _ => -1.0,
        };

        /// <summary>
        ///     <c>np.sum(arr, axis)</c> of a bool <paramref name="arr"/> in the int64 or float64 loop, as the count of its
        ///     nonzero elements (NumPy's bool → number cast is <c>x != 0</c>, so a count is the sum exactly, in any order).
        ///     Served by np.evaluate's CountNonzero axis stream, which reads a C- or F-contiguous operand in place and
        ///     returns the counts in that operand's order (C for C, F for F), NumPy's K-order reduction layout.
        /// </summary>
        /// <param name="arr">A bool operand with at least one element.</param>
        /// <param name="axis">The normalized reduced axis.</param>
        /// <param name="keepdims">Keep the reduced axis with length 1.</param>
        /// <param name="loopType">Int64 (np.sum's default bool loop) or Double (a <c>dtype=float64</c> request).</param>
        /// <returns>The sums, a read-only 0-d scalar when every axis is reduced without keepdims; null when the operand is
        /// neither C- nor F-contiguous or dynamic code is unavailable (the caller's routes then run).</returns>
        private static NDArray TryCountBoolAxisSum(NDArray arr, int axis, bool keepdims, NPTypeCode loopType)
        {
            var shape = arr.Shape;
            if (!DirectILKernelGenerator.Enabled || shape.size == 0 || !(shape.IsContiguous || shape.IsFContiguous))
                return null;

            var counts = np.evaluate(NDExpr.CountNonzero(NDExpr.Arr(arr), axis, keepdims));
            if (loopType == NPTypeCode.Double)
            {
                // Exact: a count is an integer below 2^53. astype keeps the counts' layout (order 'K').
                var wide = counts.astype(np.float64, copy: true);
                counts.Dispose();
                counts = wide;
            }

            return counts.ndim == 0 ? counts.MarkReductionScalar() : counts;
        }

        private NDArray HandleElementWiseSum(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var result = sum_elementwise_il(arr, typeCode);
            if (@out is not null) { @out.SetAtIndex(result, 0); return @out; }
            var r = NDArray.Scalar(result);
            if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            return r.MarkReductionScalar();
        }

        private unsafe NDArray ExecuteAxisReduction(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out, ReductionOp op)
        {
            var shape = arr.Shape;
            var inputType = arr.GetTypeCode;

            // NDIter-driven per-chunk path (the migration target). Currently serves the
            // (dtype, op) combinations that the legacy DirectILKernelGenerator path covers
            // only with a slow scalar kernel — Complex sum/prod/min/max. Returns null when
            // no per-chunk kernel exists yet, so we fall through to the Direct path below.
            bool useNDIter = UseNDIterReduce(inputType, outputType, op);
            // f64/f32 Min/Max only WIN on the stride-ordered NDIter reduce path where the
            // Direct whole-array kernel collapses to a cache-hostile coordinate walk — i.e.
            // broadcast (stride 0) or negative strides. For C/F-contiguous and positive-strided
            // inputs the Direct kernel's per-array (not per-stripe) traversal is faster, so keep
            // it there. (Sum/Mean always prefer NDIter — their pairwise/streaming kernels beat
            // Direct on every layout. Complex/Decimal Min/Max are NOT gated: their NDIter path
            // is a strict win over the legacy scalar Direct kernel on all layouts.)
            if (useNDIter && (op == ReductionOp.Min || op == ReductionOp.Max)
                && (inputType == NPTypeCode.Double || inputType == NPTypeCode.Single)
                && !MinMaxLayoutFavorsNDIter(shape))
                useNDIter = false;
            if (useNDIter)
            {
                var npyIterResult = ExecuteAxisReductionNDIter(arr, axis, keepdims, outputType, @out, op);
                if (npyIterResult is not null) return npyIterResult;
            }

            var key = new AxisReductionKernelKey(inputType, outputType, op, shape.IsContiguous && axis == arr.ndim - 1);
            var kernel = DirectILKernelGenerator.TryGetAxisReductionKernel(key);
            if (kernel == null)
                throw new NotSupportedException($"Axis reduction kernel not available for {op}({inputType}) -> {outputType}.");

            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++) if (d != axis) outputDims[od++] = shape.dimensions[d];

            var outputShape = outputDims.Length > 0 ? Shape.Adopt(outputDims) : Shape.Scalar;
            NDArray result;
            if (@out is not null) { if (@out.Shape != outputShape) throw new IncorrectShapeException($"Output shape mismatch"); result = @out; }
            else result = AllocateReductionResult(outputType, outputDims, shape, axis);

            long axisSize = shape.dimensions[axis];
            long outputSize = result.size > 0 ? result.size : 1;
            byte* inputAddr = (byte*)arr.Address + shape.offset * arr.dtypesize;

            fixed (long* inputStrides = shape.strides)
            fixed (long* inputDims = shape.dimensions)
            fixed (long* outputStrides = result.Shape.strides)
            {
                kernel((void*)inputAddr, (void*)result.Address, inputStrides, inputDims, outputStrides, axis, axisSize, arr.ndim, outputSize);
            }

            if (keepdims)
                result.Storage.ExpandDimension(axis);
            // A fresh 0-d result (1-D input reduced over its only axis) is a numpy SCALAR at the
            // boundary — read-only; an out= operand returns writeable (PyArray_Return semantics).
            return @out is not null ? result : result.MarkReductionScalar();
        }

        /// <summary>
        ///     Allocates a single-axis reduction's output in the layout NumPy's reduce iterator gives it
        ///     (<c>npyiter_new_temp_array</c>): the input's axes in NpyIter's order — sorted by |stride|, C winning
        ///     ties — with the reduced axis dropped (<see cref="Shape.ReductionOutputShape"/>). A C-contiguous input
        ///     yields a C result, an F-contiguous one an F result (issue #610), and any other order — a transposed 3-D
        ///     input — keeps the input's memory order for the remaining axes (until 2026-10-05 those were reordered to
        ///     C; plan api-parameter-parity A4). The reduction kernels write each output element through
        ///     <c>outputStrides</c> (the general/slab paths use them directly; the C-contiguous fast paths are gated on a
        ///     C-contiguous INPUT, which gives a C result), so filling such a buffer costs no extra copy — exactly how NumPy
        ///     allocates the output operand and writes into it. A 0-D or 1-D result is both C- and F-contiguous, so only
        ///     rank &gt;= 2 results consult the order. keepdims re-inserts the reduced axis with
        ///     <see cref="UnmanagedStorage.ExpandDimension"/>, which preserves the order (unlike a Reshape to a fresh
        ///     C-shape, which would reset it). ArgMax/ArgMin do NOT use this: NumPy allocates their index output in
        ///     C-order regardless of input (probed 2.4.2).
        /// </summary>
        /// <param name="outputType">The result dtype.</param>
        /// <param name="outputDims">The result's dims: the input's without <paramref name="axis"/>, or the keepdims form (the
        ///     input's with <paramref name="axis"/> set to 1) — told apart by their length. Adopted by the result's shape.</param>
        /// <param name="inputShape">The reduced array's shape (the layout the order is taken from).</param>
        /// <param name="axis">The reduced axis, already normalized.</param>
        /// <returns>A fresh, uninitialized result.</returns>
        internal static NDArray AllocateReductionResult(NPTypeCode outputType, long[] outputDims, Shape inputShape, int axis)
        {
            if (outputDims.Length == 0)
                return new NDArray(outputType, Shape.Scalar, false);
            // A 1-D result has one layout, and a C-contiguous input's order is C's (NpyIter's sort keeps it).
            if (outputDims.Length == 1 || inputShape.IsContiguous)
                return new NDArray(outputType, Shape.Adopt(outputDims), false);
            return new NDArray(outputType, Shape.ReductionOutputShape(outputDims, inputShape, axis), false);
        }

        /// <summary>
        ///     Zero-filled sibling of <see cref="AllocateReductionResult"/>, for the degenerate
        ///     size-1-axis std/var paths whose result is all zeros: the layout still follows the
        ///     input's memory order (F for an F-contig input), matching NumPy (issue #610). Unlike
        ///     the executors above, <paramref name="dims"/> is already the FINAL result shape (the
        ///     caller having applied keepdims), so no ExpandDimension follows.
        /// </summary>
        internal static NDArray AllocateReductionZeros(NPTypeCode outputType, long[] dims, Shape inputShape)
        {
            var order = dims.Length >= 2 ? OrderResolver.Resolve('K', inputShape) : 'C';
            var zeroShape = order == 'F' ? Shape.Adopt(dims, 'F') : Shape.Adopt(dims);
            return np.zeros(zeroShape, outputType);
        }

        /// <summary>
        ///     Gate for the NDIter-driven per-chunk reduction path. Returns true only for
        ///     (dtype, op) combinations that have a kernel in
        ///     <see cref="Kernels.ILKernelGenerator.GetReduceInnerLoop"/>; everything else
        ///     stays on the legacy <see cref="Kernels.DirectILKernelGenerator"/> path.
        ///     Acts as the per-dtype rollback switch (Plan §6).
        /// </summary>
        private static bool UseNDIterReduce(NPTypeCode inputType, NPTypeCode outputType, ReductionOp op)
        {
            // Complex same-type sum/prod/min/max/mean. The legacy complex axis paths were:
            // a scalar axis kernel (sum/prod/min/max — already ~parity under -c Release) and,
            // for the DEFAULT complex mean, the per-output-row-allocating MeanAxisComplex
            // (15–45× slower than NumPy — the genuine bottleneck). The NDIter double-pair
            // path puts all five on the migration-target architecture at parity-or-better and
            // collapses mean to a one-pass sum kernel + scalar divide.
            if (inputType == NPTypeCode.Complex && outputType == NPTypeCode.Complex)
                return op == ReductionOp.Sum || op == ReductionOp.Prod ||
                       op == ReductionOp.Min || op == ReductionOp.Max ||
                       op == ReductionOp.Mean;

            // Half SUM routes to the float32-shadow HalfSumKernel (outputType==Single; ReduceAdd
            // narrows to Half) — NumPy's exact per-orientation reduce via per-call RoundToF16 (PINNED
            // pairwise, SLAB per-step-round saturation). Half MEAN still accumulates in Double then
            // divides and casts back (outputType==Double). Half PROD/MIN/MAX stay on the Direct path.
            if (inputType == NPTypeCode.Half)
                return (op == ReductionOp.Sum && outputType == NPTypeCode.Single)
                    || (op == ReductionOp.Mean && outputType == NPTypeCode.Double);

            // Decimal: the legacy path is both cache-hostile AND lossy (it accumulates through
            // a double bridge). The NDIter kernels are full-precision Decimal on contiguous
            // stripes — 7–12× faster everywhere AND more accurate. No NumPy reference type.
            if (inputType == NPTypeCode.Decimal && outputType == NPTypeCode.Decimal)
                return op == ReductionOp.Sum || op == ReductionOp.Prod ||
                       op == ReductionOp.Min || op == ReductionOp.Max ||
                       op == ReductionOp.Mean;

            // Phase 6 — numeric migration onto the per-chunk target architecture.
            // Double AND float32 SUM, MEAN (mean = Sum kernel + MeanDivideByCount), and
            // MIN/MAX. SUM's PINNED path uses PairwiseFold (ported 1:1 from NumPy's
            // pairwise_sum) so it is BIT-FOR-BIT identical to NumPy for both dtypes — which
            // is what makes float32 safe to route (its earlier exclusion was a flat-
            // accumulator divergence the pairwise leaf removes); SLAB stays the streaming
            // Vector256 add. MIN/MAX route to SimdMinMaxSameType (NaN-propagating, dual-mode)
            // so f64/f32 min/max are SIMD on every layout instead of collapsing on the
            // C-contiguity-gated Direct kernel for negcol/broadcast/strided inputs.
            // Integer Sum (NEP50 widening), integer Min/Max, and Prod stay on the Direct
            // path (GetReduceInnerLoop returns null for those → caller falls back).
            if ((inputType == NPTypeCode.Double || inputType == NPTypeCode.Single) && outputType == inputType)
                return op == ReductionOp.Sum || op == ReductionOp.Mean ||
                       op == ReductionOp.Min || op == ReductionOp.Max;

            return false;
        }

        /// <summary>
        ///     Layout gate for the f64/f32 Min/Max NDIter routing (see ExecuteAxisReduction).
        ///     Returns true only where the Direct whole-array axis kernel collapses to a
        ///     cache-hostile coordinate walk — broadcast (stride 0, repeated reads) or any
        ///     negative stride (backward traversal). Contiguous and positive-strided inputs
        ///     stay on the faster Direct kernel.
        /// </summary>
        private static bool MinMaxLayoutFavorsNDIter(Shape shape)
        {
            if (shape.IsBroadcasted) return true;
            var strides = shape.strides;
            for (int i = 0; i < strides.Length; i++)
                if (strides[i] < 0) return true;
            return false;
        }

        /// <summary>
        ///     Axis reduction via the 2-operand REDUCE iterator + a per-chunk
        ///     <see cref="Kernels.ILKernelGenerator"/> kernel. Mirrors
        ///     <see cref="ExecuteAxisReduction"/>'s output-shape / keepdims / out= handling,
        ///     but seeds the reduction identity and lets the iterator drive the inner loop.
        ///     Returns null when no per-chunk kernel exists for the key (caller falls back).
        /// </summary>
        private unsafe NDArray ExecuteAxisReductionNDIter(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out, ReductionOp op)
        {
            // Mean is computed as a one-pass Sum kernel followed by a scalar divide by the
            // reduced-axis length — there is no separate "mean" inner loop.
            var kernelOp = op == ReductionOp.Mean ? ReductionOp.Sum : op;
            var key = new ILKernelGenerator.ReduceKernelKey(kernelOp, arr.GetTypeCode, outputType);
            var kernel = ILKernelGenerator.GetReduceInnerLoop(key);
            if (kernel is null) return null;

            var shape = arr.Shape;
            var outputDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++) if (d != axis) outputDims[od++] = shape.dimensions[d];
            var outputShape = outputDims.Length > 0 ? Shape.Adopt(outputDims) : Shape.Scalar;

            NDArray result;
            if (@out is not null)
            {
                if (@out.Shape != outputShape) throw new IncorrectShapeException($"Output shape mismatch");
                result = @out;
            }
            else
            {
                result = AllocateReductionResult(outputType, outputDims, shape, axis);
            }

            // The per-chunk kernel folds into the existing output slot(s), so the output
            // must carry the reduction identity (0/1/±inf) before the iterator runs.
            ILKernelGenerator.SeedReduceIdentity(result, kernelOp);

            // COPY_IF_OVERLAP only matters when a user-supplied out= may alias the input; a
            // fresh allocation can never overlap, so the hot path skips the overlap probe.
            var extraFlags = @out is not null ? NDIterGlobalFlags.COPY_IF_OVERLAP : NDIterGlobalFlags.None;
            using (var iter = NDIterRef.NewReduce(arr, result, axis, extraFlags))
                iter.ForEach(kernel);

            // Mean: divide the accumulated sums by the reduced-axis length (NumPy parity).
            if (op == ReductionOp.Mean)
                ILKernelGenerator.MeanDivideByCount(result, shape.dimensions[axis]);

            if (keepdims)
                result.Storage.ExpandDimension(axis);
            // Same PyArray_Return rule as ExecuteAxisReduction's exit; the output was already
            // allocated in KEEPORDER (F for an F-contig input), so there is nothing to reorder.
            return @out is not null ? result : result.MarkReductionScalar();
        }

        /// <summary>
        /// Handle empty array min/max reductions.
        /// NumPy behavior:
        /// - np.min([]) raises ValueError (no identity for min/max)
        /// - np.min(zeros((0,3)), axis=0) raises ValueError (reducing along empty dimension)
        /// - np.min(zeros((0,3)), axis=1) returns [] with shape (0,) (output is also empty)
        /// </summary>
        private NDArray HandleEmptyArrayMinMaxReduction(NDArray arr, int? axis_, bool keepdims, NPTypeCode? typeCode, string opName)
        {
            var shape = arr.Shape;

            // No axis specified - always throw for empty arrays (no identity element for min/max)
            if (axis_ == null)
                throw new ArgumentException($"zero-size array to reduction operation {opName} which has no identity");

            var axis = NormalizeAxis(axis_.Value, arr.ndim);

            // If the axis being reduced has size 0, we're reducing over an empty dimension
            // which results in an error (no values to take min/max of)
            if (shape.dimensions[axis] == 0)
                throw new ArgumentException($"zero-size array to reduction operation {opName} which has no identity");

            // If the axis being reduced has size > 0, but the result would be empty,
            // return an empty array of the correct shape
            var resultShape = Shape.GetAxis(shape, axis);
            var outputType = typeCode ?? arr.GetTypeCode;
            var result = new NDArray(outputType, Shape.Adopt(resultShape), false);

            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++)
                    ks[d] = (d == axis) ? 1 : resultShape[sd++];
                result.Storage.Reshape(Shape.Adopt(ks));
            }
            return result;
        }

        private NDArray HandleEmptyArrayReduction(NDArray arr, int? axis_, bool keepdims, NPTypeCode? typeCode, NDArray @out, ReductionOp op)
        {
            var shape = arr.Shape;
            if (axis_ == null)
            {
                // NumPy parity: empty reduction uses accumulating type (int/bool -> int64/uint64, floats preserved).
                var defaultType = typeCode ?? arr.typecode.GetAccumulatingType();
                var defaultVal = defaultType.GetDefaultValue();
                if (@out is not null) { @out.SetAtIndex(defaultVal, 0); return @out; }
                var r = NDArray.Scalar(defaultVal);
                if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
                return r.MarkReductionScalar();
            }
            var axis = NormalizeAxis(axis_.Value, arr.ndim);
            var resultShape = Shape.GetAxis(shape, axis);
            var outputType = typeCode ?? arr.GetTypeCode.GetAccumulatingType();
            var result = np.zeros(Shape.Adopt(resultShape), outputType);
            if (keepdims)
            {
                var ks = new long[arr.ndim];
                for (int d = 0, sd = 0; d < arr.ndim; d++) ks[d] = (d == axis) ? 1 : resultShape[sd++];
                result.Storage.Reshape(Shape.Adopt(ks));
            }
            if (@out is not null) { np.copyto(@out, result); return @out; }
            return result;
        }

        /// <summary>
        ///     Degenerate flat reduction of a 0-d scalar or single-element 1-D array: the result is the
        ///     lone element, reshaped to a numpy scalar (0-d) or, under <paramref name="keepdims"/>, to
        ///     all-ones. A COPY, which is right only for a reduction whose result IS the lone element —
        ///     prod (which passes its RESOLVED accumulating dtype, so the element is cast/widened; 1 * x is
        ///     x) and amin/amax/nanmin/nanmax (which pass <c>null</c> to PRESERVE the input dtype; they
        ///     have no identity, NumPy copies the first element). The helper does not decide widening, the
        ///     caller does: a null <paramref name="typeCode"/> clones the input verbatim, a non-null one
        ///     casts to it. np.sum / np.mean do NOT come here — their identity add turns a -0.0 into +0.0
        ///     (<see cref="SumOfOneElement"/>).
        /// </summary>
        /// <param name="arr">The size-&#8804;1 operand.</param>
        /// <param name="keepdims">When true, the result keeps the input rank with every axis length 1.</param>
        /// <param name="typeCode">The resolved output dtype (already widened for sum/prod), or null to preserve <paramref name="arr"/>'s dtype (amin/amax).</param>
        /// <param name="out">Optional pre-allocated output; when non-null the single value is written into slot 0 and it is returned.</param>
        /// <returns>The reduced scalar as an <see cref="NDArray"/> — a read-only numpy scalar on the 0-d exit, or the writeable <paramref name="out"/> when supplied.</returns>
        private NDArray HandleScalarReduction(NDArray arr, bool keepdims, NPTypeCode? typeCode, NDArray @out)
        {
            var r = typeCode.HasValue ? Cast(arr, typeCode.Value, true) : arr.Clone();
            if (@out is not null) { @out.SetAtIndex(r.GetAtIndex(0), 0); return @out; }
            if (keepdims) { var ks = new long[arr.ndim]; for (int i = 0; i < arr.ndim; i++) ks[i] = 1; r.Storage.Reshape(Shape.Adopt(ks)); }
            else if (!r.Shape.IsScalar && r.Shape.size == 1 && r.ndim == 1) r.Storage.Reshape(Shape.Scalar);
            // A 0-d exit — including keepdims over a 0-d input, whose (1,)*0 reshape is still
            // 0-d — is a numpy SCALAR (read-only); (1,) keepdims results pass through writeable.
            return r.MarkReductionScalar();
        }

        /// <summary>
        ///     Reduction over an axis of LENGTH 1 by COPYING each lone element (cast to
        ///     <paramref name="outputType"/>): right only for the reductions without an identity —
        ///     amin/amax/nanmin/nanmax, where NumPy copies the first element (PyArray_CopyInitialReduceValues) —
        ///     and for prod (1 * x is x). np.sum and np.mean go to <see cref="SumOverLengthOneAxis"/> instead: their
        ///     identity add makes a -0.0 element +0.0.
        /// </summary>
        /// <param name="arr">The input; <c>arr.shape[axis] == 1</c>.</param>
        /// <param name="axis">The normalized reduced axis.</param>
        /// <param name="keepdims">When true the reduced axis stays in the result with length 1.</param>
        /// <param name="outputType">The result dtype each element is converted to.</param>
        /// <param name="out">Unsupported: a non-null value returns null (every caller passes null).</param>
        /// <returns>The copied slab in KEEPORDER layout, a 0-d scalar for a 1-D input, or null when <paramref name="out"/> is given.</returns>
        private NDArray HandleTrivialAxisReduction(NDArray arr, int axis, bool keepdims, NPTypeCode outputType, NDArray @out)
        {
            if (@out is not null) return null;
            var shape = arr.Shape;
            long[] resultDims;
            if (keepdims) { resultDims = (long[])shape.dimensions.Clone(); resultDims[axis] = 1; }
            else { resultDims = new long[arr.ndim - 1]; for (int d = 0, rd = 0; d < arr.ndim; d++) if (d != axis) resultDims[rd++] = shape.dimensions[d]; }
            if (resultDims.Length == 0)
            {
                var v = arr.GetAtIndex(0);
                if (outputType != arr.GetTypeCode) v = Converts.ChangeType(v, outputType);
                return NDArray.Scalar(v).MarkReductionScalar();
            }
            // KEEPORDER: reducing a size-1 axis of an F-contiguous input keeps F-contig; SetAtIndex
            // writes each element through the result's strides, so an F-order buffer fills correctly
            // with no copy (resultDims already carries the final keepdims shape) (issue #610).
            var result = AllocateReductionResult(outputType, resultDims, shape, axis);
            if (outputType == arr.GetTypeCode) for (long i = 0; i < result.size; i++) result.SetAtIndex(arr.GetAtIndex(i), i);
            else for (long i = 0; i < result.size; i++) result.SetAtIndex(Converts.ChangeType(arr.GetAtIndex(i), outputType), i);
            return result;
        }

        /// <summary>
        ///     Normalizes a possibly-negative axis to a non-negative index and validates bounds.
        ///     Matches NumPy's axis normalization exactly.
        /// </summary>
        /// <param name="axis">The axis value (can be negative).</param>
        /// <param name="ndim">The number of dimensions in the array.</param>
        /// <returns>The normalized non-negative axis.</returns>
        /// <exception cref="AxisError">If the axis is out of bounds after normalization.</exception>
        internal static int NormalizeAxis(int axis, int ndim)
        {
            int originalAxis = axis;
            if (axis < 0)
                axis += ndim;
            if (axis < 0 || axis >= ndim)
                throw new AxisError(originalAxis, ndim);
            return axis;
        }
    }
}
