using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    // =============================================================================
    // The add reduction's IDENTITY — how NumPy STARTS np.sum / np.mean / np.nansum.
    //
    // PyUFunc_ReduceWrapper (numpy/_core/src/umath/reduction.c) asks the add loop for its initial value
    // (get_initial_from_ufunc in legacy_array_method.c packs the ufunc identity 0 into the loop dtype: +0.0 for
    // every inexact loop), assigns it to the result with raw_array_assign_scalar, and only THEN runs the add loop
    // over the elements — even when an output owns ONE element (a one-element input, a length-1 axis). An
    // identity-seeded IEEE sum is never -0.0: +0.0 + (-0.0) is +0.0, so np.sum of only negative zeros is +0.0 (each
    // part of a complex), and a signalling NaN comes back quiet. The helpers below are the single-element routes
    // of that rule; the multi-element kernels seed the identity themselves (SeedReduceIdentity, the pairwise
    // PINNED route's `*dst += fold`).
    // =============================================================================
    public partial class DefaultEngine
    {
        /// <summary>
        ///     True for the loop dtypes whose identity add is NOT the identity on bits — float16, float32, float64 and
        ///     complex128: <c>+0.0 + x</c> turns <c>x = -0.0</c> into <c>+0.0</c> and quiets a signalling NaN. For every
        ///     other loop the identity add returns <c>x</c> exactly (integers / bool / char wrap nothing at 0, and .NET's
        ///     <c>0m + x</c> keeps a decimal's value, scale and sign), so the single-element routes may copy there.
        /// </summary>
        /// <param name="loopType">The reduction's loop dtype.</param>
        /// <returns>Whether the single-element routes must run the identity add rather than a copy.</returns>
        private static bool IsIeeeLoop(NPTypeCode loopType)
            => loopType is NPTypeCode.Half or NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex;

        /// <summary>
        ///     Writes <c>identity + (loop)src</c> into <paramref name="result"/>, element by element: NumPy's add reduction
        ///     of outputs that each own exactly one element. <paramref name="src"/> must have <paramref name="result"/>'s
        ///     logical shape; any layout (strided, reversed, F, a 0-d view at an offset) is read through its strides.
        /// </summary>
        /// <param name="src">The lone element of every output, laid out with the result's logical shape.</param>
        /// <param name="result">A fresh, writeable array of the loop dtype; receives the sums.</param>
        /// <param name="loopType">The loop dtype (<paramref name="result"/>'s dtype).</param>
        /// <remarks>
        ///     An IEEE loop runs the binary add kernel with a 0-d identity operand (one fused pass, SIMD where the layout
        ///     allows; when <paramref name="src"/> is in another dtype — not reached by np.sum, whose cast loops arrive
        ///     already cast — the cast runs first and the add follows in place). Every other loop is a cast copy: its
        ///     identity add changes nothing. Both go through the IL kernels / NDIter, never a managed element loop.
        /// </remarks>
        private void WriteIdentityPlus(NDArray src, NDArray result, NPTypeCode loopType)
        {
            if (!IsIeeeLoop(loopType))
            {
                NDIter.Copy(result, src);
                return;
            }

            // The identity is a STRONG 0-d operand of the loop dtype, so the add runs in the loop dtype (no promotion);
            // it is the LEFT operand, as in NumPy's `io1 = identity; io1 += x` (the order cannot change an IEEE sum when
            // only one operand may be NaN, but it keeps the call honest).
            using var identity = NDArray.Scalar(loopType.GetDefaultValue(), loopType);
            if (src.GetTypeCode == loopType)
            {
                ExecuteBinaryOp(identity, src, BinaryOp.Add, @out: result);
            }
            else
            {
                NDIter.Copy(result, src);
                ExecuteBinaryOp(identity, result, BinaryOp.Add, @out: result);
            }
        }

        /// <summary>
        ///     The add reduction of a ONE-element input over every axis (or, for a 0-d or 1-D input, over its only axis):
        ///     <c>identity + (loop)x</c>, shaped 0-d — or <c>(1,)*ndim</c> under <paramref name="keepdims"/>.
        /// </summary>
        /// <param name="arr">The one-element input, any layout (a strided or offset view of one element included).</param>
        /// <param name="keepdims">When true the result keeps <paramref name="arr"/>'s rank with every axis 1 (a 0-d input stays 0-d).</param>
        /// <param name="loopType">The loop dtype the element is cast to and added in.</param>
        /// <param name="out">Optional output that receives the value (broadcast, cast) and is returned instead.</param>
        /// <param name="divideByCount">np.mean's route: the sum is then divided by its count (1) the way np.mean divides —
        /// see <see cref="DivideOneElementSumsByCount"/>.</param>
        /// <returns>A read-only 0-d scalar (NumPy's scalar return), the writeable keepdims array, or <paramref name="out"/>.</returns>
        /// <remarks>
        ///     The result is first allocated with <paramref name="arr"/>'s own rank and every axis 1, which lines up with
        ///     the lone element exactly, so the identity add is one elementwise op over the two; the non-keepdims result is
        ///     then relabelled 0-d — a Storage reshape of one element, nothing moves.
        /// </remarks>
        private NDArray SumOfOneElement(NDArray arr, bool keepdims, NPTypeCode loopType, NDArray @out, bool divideByCount = false)
        {
            var ones = new long[arr.ndim];
            for (int d = 0; d < ones.Length; d++)
                ones[d] = 1;
            var result = new NDArray(loopType, ones.Length == 0 ? Shape.Scalar : Shape.Adopt(ones), false);
            try
            {
                WriteIdentityPlus(arr, result, loopType);
                if (divideByCount)
                    DivideOneElementSumsByCount(result, loopType);
            }
            catch
            {
                // The fresh result would otherwise strand its pooled buffer on a failing kernel.
                result.Dispose();
                throw;
            }

            if (!keepdims && arr.ndim > 0)
                result.Storage.Reshape(Shape.Scalar);
            return DeliverReduction(result, @out);
        }

        /// <summary>
        ///     The add reduction over an axis of LENGTH 1: every output owns one element, so each is
        ///     <c>identity + (loop)x</c> — NumPy assigns the identity and runs the add loop here too, it never copies.
        /// </summary>
        /// <param name="arr">The input; <c>arr.shape[axis] == 1</c>, rank ≥ 1, any layout.</param>
        /// <param name="axis">The normalized reduced axis.</param>
        /// <param name="keepdims">When true the reduced axis stays in the result with length 1.</param>
        /// <param name="loopType">The loop dtype.</param>
        /// <param name="out">Optional output that receives the result (broadcast, cast) and is returned instead.</param>
        /// <param name="divideByCount">np.mean's route: every sum is then divided by its count (1) the way np.mean
        /// divides — see <see cref="DivideOneElementSumsByCount"/>.</param>
        /// <returns>A fresh array in KEEPORDER layout (F for an F-contiguous input, like every reduction output), a
        /// read-only 0-d scalar when <paramref name="arr"/> is 1-D without keepdims, or <paramref name="out"/>.</returns>
        private NDArray SumOverLengthOneAxis(NDArray arr, int axis, bool keepdims, NPTypeCode loopType, NDArray @out,
            bool divideByCount = false)
        {
            var shape = arr.Shape;
            var outDims = new long[arr.ndim - 1];
            for (int d = 0, od = 0; d < arr.ndim; d++)
                if (d != axis)
                    outDims[od++] = shape.dimensions[d];

            // The lone element of every output: index 0 along the reduced axis — a view, nothing is copied.
            var slices = new Slice[arr.ndim];
            for (int d = 0; d < slices.Length; d++)
                slices[d] = d == axis ? Slice.Index(0) : Slice.All;
            var src = arr[slices];
            try
            {
                var result = AllocateReductionResult(loopType, outDims, shape);
                try
                {
                    WriteIdentityPlus(src, result, loopType);
                    if (divideByCount)
                        DivideOneElementSumsByCount(result, loopType);
                }
                catch
                {
                    result.Dispose();
                    throw;
                }

                if (keepdims)
                    result.Storage.ExpandDimension(axis);
                return DeliverReduction(result, @out);
            }
            finally
            {
                // The view holds a counted reference on arr's buffer; release it now rather than at finalization.
                src.Dispose();
            }
        }

        /// <summary>
        ///     np.mean's division of one-element sums by their count, 1. Every real dtype's <c>x / 1</c> is <c>x</c> itself
        ///     (the sums are already identity-added, so no -0.0 or signalling NaN is left to change), so only a complex loop
        ///     runs it: NumPy divides by <c>1+0j</c> with its complex loop, <c>((re + im*0)*1, (im - re*0)*1)</c>, which turns
        ///     an infinite or NaN part into NaN in the OTHER part (<c>np.mean([5+infj])</c> is <c>nan+infj</c>).
        /// </summary>
        /// <param name="result">The fresh, still-writeable sums (dense block).</param>
        /// <param name="loopType">Their dtype.</param>
        private static void DivideOneElementSumsByCount(NDArray result, NPTypeCode loopType)
        {
            if (loopType == NPTypeCode.Complex)
                ILKernelGenerator.MeanDivideByCount(result, 1);
        }

        /// <summary>
        ///     The add reduction in a loop dtype that is neither the input's own nor its default accumulator — NumPy's
        ///     <c>dtype=</c> cast loop. NumPy's buffered iterator casts every element into the loop dtype before the add
        ///     loop sees it, so the input is cast first (<see cref="Cast(NDArray, NPTypeCode, bool)"/>: float→int truncates,
        ///     int→uint wraps, complex→real keeps the real part, X→bool tests != 0) and the cast copy is reduced in its own
        ///     dtype.
        /// </summary>
        /// <param name="arr">The input array (any dtype, any layout).</param>
        /// <param name="axis">The axis to reduce, or null for every axis.</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="loopType">The requested loop dtype.</param>
        /// <param name="out">Optional output, forwarded to the reduction of the cast copy.</param>
        /// <returns>The sum in <paramref name="loopType"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <remarks>
        ///     The cast copy is this method's own intermediate: every reduction route returns a FRESH result (never a view
        ///     of its input), so it is disposed here instead of being left to the finalizer.
        /// </remarks>
        private NDArray ReduceAddInCastLoop(NDArray arr, int? axis, bool keepdims, NPTypeCode loopType, NDArray @out)
        {
            if (axis != null && TryReduceAddInCastLoopBlocked(arr, axis.Value, keepdims, loopType, @out, out var blocked))
                return blocked;
            if (axis == null && TryReduceAddInCastLoopFlatBlocked(arr, keepdims, loopType, @out, out var flat))
                return flat;

            var cast = Cast(arr, loopType, copy: true);
            try
            {
                return ReduceAdd(cast, axis, keepdims, loopType, @out);
            }
            finally
            {
                cast.Dispose();
            }
        }

        /// <summary>
        ///     The cast copy an axis reduction in a cast loop reduces, in blocks of at most this many elements along a
        ///     non-reduced axis (internal and settable so tests can force many blocks on small arrays).
        /// </summary>
        /// <remarks>
        ///     One full cast copy of a large operand is mostly page faults on a fresh allocation: <c>np.sum(f32, axis=1,
        ///     dtype=float64)</c> over 10M elements took 17.7 ms that way, NumPy 7.0 ms. Blocks of 128K elements (1 MB of
        ///     float64) come back from the pool already mapped and stay in L2 between the cast and the reduction: 4.0 ms at
        ///     10M, 248 µs at 1M (NumPy 448 µs) — 512K-element blocks measured 4.6 ms / 371 µs.
        /// </remarks>
        internal static long CastLoopBlockElements = 1L << 17;

        /// <summary>
        ///     <see cref="ReduceAddInCastLoop"/> for an AXIS reduction of a large operand, in blocks: the operand is cut
        ///     along its outermost NON-reduced axis, each block cast (the same <see cref="Cast"/> the whole-array route
        ///     makes) and reduced (the same <see cref="ReduceAdd"/>), and the block's outputs written into the result.
        /// </summary>
        /// <remarks>
        ///     Every output's elements lie inside one block, in the same order along the reduced axis, and the cast copy
        ///     of a block has the layout class of the whole cast copy (both are fresh copies of the same kind of view), so
        ///     the reduction computes each output exactly as the whole-array route does — pinned by
        ///     <c>ReduceAddCastLoopBlockTests</c> (blocked against whole, bit for bit, across layouts, axes and dtype
        ///     pairs).
        /// </remarks>
        /// <param name="arr">The operand.</param>
        /// <param name="axis">The reduced axis (negative counts from the last).</param>
        /// <param name="keepdims">Keep the reduced axis with length 1.</param>
        /// <param name="loopType">The loop dtype.</param>
        /// <param name="out">Optional output, receiving the assembled result.</param>
        /// <param name="result">Receives the reduction when blocking applies.</param>
        /// <returns>False — nothing computed — when the operand is small, 1-D, broadcast, or its non-reduced axes cannot
        /// be cut (the whole-array route then runs).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        private bool TryReduceAddInCastLoopBlocked(NDArray arr, int axis, bool keepdims, NPTypeCode loopType, NDArray @out, out NDArray result)
        {
            result = null;
            int nd = arr.ndim;
            long blockElems = CastLoopBlockElements;
            if (nd < 2 || arr.size <= blockElems || arr.Shape.IsBroadcasted)
                return false;

            int ax = NormalizeAxis(axis, nd);
            int split = ax == 0 ? 1 : 0;
            long extent = arr.Shape.dimensions[split];
            long perIndex = arr.size / extent;
            // Every block spans at least TWO indices of the cut axis: a block one index wide is contiguous in both orders,
            // which can switch the reduction kernel to another summation order than the whole copy's (probed: a one-row
            // block of an F-ordered float32 -> float64 axis-1 sum). The blocks share the extent evenly, so none is short.
            long step = Math.Max(2, blockElems / Math.Max(1, perIndex));
            long blocks = extent / step;
            if (blocks < 2)
                return false;

            // The result's dimensions, and where the split axis lands in them.
            var outDims = new long[keepdims ? nd : nd - 1];
            for (int d = 0, od = 0; d < nd; d++)
            {
                if (d == ax)
                {
                    if (keepdims) outDims[od++] = 1;
                    continue;
                }
                outDims[od++] = arr.Shape.dimensions[d];
            }
            int splitOut = !keepdims && split > ax ? split - 1 : split;

            var full = AllocateReductionResult(loopType, outDims, arr.Shape);
            try
            {
                var inSlices = new Slice[nd];
                var outSlices = new Slice[outDims.Length];
                for (int d = 0; d < nd; d++) inSlices[d] = Slice.All;
                for (int d = 0; d < outDims.Length; d++) outSlices[d] = Slice.All;
                for (long k = 0; k < blocks; k++)
                {
                    // block k covers [k·extent/blocks, (k+1)·extent/blocks): sizes differ by at most one, all ≥ step ≥ 2.
                    long b0 = k * extent / blocks, b1 = (k + 1) * extent / blocks;
                    inSlices[split] = new Slice(b0, b1);
                    outSlices[splitOut] = new Slice(b0, b1);
                    using var view = arr[inSlices];
                    using var cast = Cast(view, loopType, copy: true);
                    using var part = ReduceAdd(cast, ax, keepdims, loopType, null);
                    using var target = full[outSlices];
                    NDIter.Copy(target, part);
                }
            }
            catch
            {
                full.Dispose();
                throw;
            }

            result = DeliverReduction(full, @out);
            return true;
        }

        /// <summary>
        ///     <see cref="ReduceAddInCastLoop"/> for a FLAT reduction of a large operand in an ORDER-FREE loop, in blocks:
        ///     an integer loop adds modulo 2^bits and the bool loop is a logical OR, so every partition of the elements
        ///     gives NumPy's answer. The operand is cut along its outermost axis in memory into blocks of about
        ///     <see cref="CastLoopBlockElements"/> elements, each block cast (the same <see cref="Cast"/> the whole-array
        ///     route makes) and summed (the same <see cref="ReduceAdd"/>), and the block sums are summed in the loop dtype.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="keepdims">Keep every axis with length 1.</param>
        /// <param name="loopType">The loop dtype.</param>
        /// <param name="out">Optional output, receiving the result.</param>
        /// <param name="result">Receives the reduction when blocking applies.</param>
        /// <returns>False — nothing computed — for an order-dependent loop (float16 / float32 / float64 / complex128 take
        /// NumPy's exact schedule elsewhere, a Decimal sum rounds), a small, 0-d or broadcast operand, or one with no axis
        /// long enough to cut; the whole-array route then runs.</returns>
        /// <remarks>
        ///     Why: the whole-array route writes a full-size cast copy and reads it back — <c>np.sum(f64, dtype=int32)</c>
        ///     over 10M elements read 80 MB, wrote 40 MB and read 40 MB (6.8 ms, NumPy 7.0 ms) — where a block stays in L2
        ///     between its cast and its sum.
        /// </remarks>
        private bool TryReduceAddInCastLoopFlatBlocked(NDArray arr, bool keepdims, NPTypeCode loopType, NDArray @out, out NDArray result)
        {
            result = null;
            if (IsIeeeLoop(loopType) || loopType == NPTypeCode.Decimal)
                return false;

            var shape = arr.Shape;
            long blockElems = CastLoopBlockElements;
            if (shape.NDim == 0 || shape.size <= blockElems || shape.IsBroadcasted)
                return false;

            // The outermost axis in memory (the largest |stride| among the axes long enough to cut), so each block is one
            // compact region of the operand's memory.
            int split = -1;
            long widest = -1;
            for (int d = 0; d < shape.NDim; d++)
            {
                if (shape.dimensions[d] < 2)
                    continue;
                long s = Math.Abs(shape.strides[d]);
                if (s > widest)
                {
                    widest = s;
                    split = d;
                }
            }

            if (split < 0)
                return false;
            long extent = shape.dimensions[split];
            long step = Math.Max(1, blockElems / Math.Max(1, shape.size / extent));
            long blocks = (extent + step - 1) / step;
            if (blocks < 2)
                return false;

            // The block sums, in the loop dtype; summed at the end by the same (order-free) reduction.
            var parts = new NDArray(loopType, new Shape(blocks), false);
            try
            {
                var slices = new Slice[shape.NDim];
                for (int d = 0; d < slices.Length; d++)
                    slices[d] = Slice.All;
                for (long k = 0; k < blocks; k++)
                {
                    slices[split] = new Slice(k * step, Math.Min(extent, (k + 1) * step));
                    using var view = arr[slices];
                    using var cast = Cast(view, loopType, copy: true);
                    using var part = ReduceAdd(cast, null, false, loopType, null);
                    parts.SetAtIndex(part.GetAtIndex(0), k);
                }

                using var total = ReduceAdd(parts, null, false, loopType, null);
                var shaped = NewFlatReductionResult(loopType, arr.ndim, keepdims);
                shaped.SetAtIndex(total.GetAtIndex(0), 0);
                result = DeliverReduction(shaped, @out);
                CastLoopFlatBlockedRuns++;
                return true;
            }
            finally
            {
                // The block sums are this method's own intermediate; the result is a fresh array.
                parts.Dispose();
            }
        }

        /// <summary>
        ///     Test hook: the number of flat cast-loop reductions <see cref="TryReduceAddInCastLoopFlatBlocked"/> served on
        ///     the calling thread (it only ever increases; a test reads it around a call to learn whether blocking ran).
        /// </summary>
        [ThreadStatic] internal static int CastLoopFlatBlockedRuns;

        /// <summary>
        ///     The add reduction in the bool loop: NumPy's bool add is logical OR (<c>True + True</c> is <c>True</c>), so
        ///     <c>add.reduce(a, dtype=bool)</c> is <c>np.any(a)</c> with the identity <c>False</c>.
        /// </summary>
        /// <param name="arr">A bool input (other dtypes reach the bool loop through <see cref="ReduceAddInCastLoop"/>).</param>
        /// <param name="axis">The axis to reduce, or null for every axis; a 0-d input reads 0 / -1 as every axis.</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="out">Optional output that receives the result and is returned.</param>
        /// <returns>The OR reduction as a bool array (a read-only 0-d scalar for a full reduction).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range (for a 0-d input: anything but 0 and -1).</exception>
        private static NDArray ReduceAddBooleanLoop(NDArray arr, int? axis, bool keepdims, NDArray @out)
        {
            if (axis != null && arr.ndim == 0)
            {
                // ufunc.reduce's backward-compatibility rule: a 0-d array takes axis 0 / -1 as "every axis".
                if (axis.Value != 0 && axis.Value != -1)
                    throw new AxisError(axis.Value, 0);
                axis = null;
            }

            return np.any(arr, axis, @out, keepdims);
        }

        /// <summary>
        ///     Hands a freshly computed reduction back: writes it into <paramref name="out"/> (broadcast and cast, NumPy's
        ///     out= copy) and returns <paramref name="out"/>, or marks a 0-d result as NumPy's read-only scalar return.
        /// </summary>
        /// <param name="result">The fresh result (owned by this call).</param>
        /// <param name="out">The caller's output array, or null.</param>
        /// <returns><paramref name="out"/> when given (and <paramref name="result"/> disposed), else <paramref name="result"/>.</returns>
        private static NDArray DeliverReduction(NDArray result, NDArray @out)
        {
            if (@out is null)
                return result.MarkReductionScalar();

            try
            {
                NDIter.Copy(@out, result);
            }
            finally
            {
                result.Dispose();
            }

            return @out;
        }
    }
}
