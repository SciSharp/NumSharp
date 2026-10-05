using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     <c>np.var</c> / <c>np.std</c>: NumPy's <c>_var</c> / <c>_std</c> (<c>numpy/_core/_methods.py</c>) statement
        ///     for statement. There is no special path for a small size or a large <paramref name="ddof"/>: every result
        ///     comes from the same arithmetic, so one element, a length-1 axis, <c>ddof &gt;= n</c> and an empty input give
        ///     NumPy's NaN / +inf / dtype by construction.
        /// </summary>
        /// <param name="arr">The operand (any dtype, any layout).</param>
        /// <param name="axis_">The reduced axis (negative counts from the end), or null for every axis.</param>
        /// <param name="keepdims">When true the reduced axes stay in the result with length 1.</param>
        /// <param name="ddof">Delta degrees of freedom: the divisor is <c>max(n - ddof, 0)</c>.</param>
        /// <param name="dtype">NumPy's <c>dtype=</c>: the loop of both sums (an integer or bool input defaults to
        /// float64, every other input to its own dtype), so it decides the result dtype.</param>
        /// <param name="takeSqrt">True for <c>np.std</c> (the square root of the variance, NumPy's <c>_std</c>).</param>
        /// <returns>The variance or standard deviation: a read-only 0-d scalar when the result has no dimensions,
        /// otherwise a fresh array. Its dtype is the deviation sum's — the input's float dtype, float64 for an integer,
        /// bool or complex128 input, or <paramref name="dtype"/>.</returns>
        /// <exception cref="AxisError"><paramref name="axis_"/> is outside the operand, a 0-d operand included (NumPy
        /// normalizes the axis before any arithmetic, unlike <c>ufunc.reduce</c>).</exception>
        /// <exception cref="NotSupportedException">A bool input with <c>dtype=bool</c>: NumPy's boolean-subtract
        /// TypeError, raised by <c>arr - mean</c>.</exception>
        /// <exception cref="ArgumentException"><c>np.std</c> with an integer or bool <paramref name="dtype"/> and a result
        /// that has dimensions: NumPy's <c>sqrt</c> output-cast UFuncTypeError (a 0-d result casts instead).</exception>
        /// <remarks>
        ///     <para>
        ///     NumPy's statements and their NumSharp counterparts:
        ///     </para>
        ///     <list type="number">
        ///         <item><c>rcount = _count_reduce_items(arr, axis)</c> — the axis is normalized here, first.</item>
        ///         <item><c>dtype = f8</c> for an integer or bool input without <c>dtype=</c> (NumSharp's Char counts as
        ///         an unsigned integer, the way NumPy's uint16 proxy does everywhere else).</item>
        ///         <item><c>arrmean = umr_sum(arr, axis, dtype, keepdims=True)</c> → <see cref="ReduceAdd"/>, NumPy's
        ///         add.reduce schedule; then <c>true_divide(arrmean, rcount, out=arrmean, casting='unsafe')</c> →
        ///         <see cref="DivideByCountLikeNumPy"/>.</item>
        ///         <item><c>x = subtract(arr, arrmean)</c> squared: <c>square(x)</c> for a float or integer input (a
        ///         complex square when <c>dtype=</c> is complex), <c>x.real² + x.imag²</c> for a complex deviation of a
        ///         complex or bool input, <c>(x * conj(x)).real</c> — which is <c>x²</c> — for a bool input. Fused into
        ///         one <c>np.evaluate</c> pass (Decimal: the eager ufuncs), written in the layout NumPy gives <c>x</c>
        ///         (<see cref="Shape.NpyIterOutputShape"/>), because the next sum's order follows that layout.</item>
        ///         <item><c>ret = umr_sum(x, axis, dtype, keepdims=keepdims)</c> → <see cref="ReduceAdd"/>; a flat
        ///         reduction whose <c>x</c> is one memory run with no cast streams instead (no temporary; the same
        ///         pairwise schedule, see <c>DefaultEngine.Evaluate.Stream.cs</c>).</item>
        ///         <item><c>ret / max(rcount - ddof, 0)</c> → <see cref="DivideByCountLikeNumPy"/>.</item>
        ///         <item><c>_std</c>: <c>sqrt(ret, out=ret)</c> for an array, <c>ret.dtype.type(sqrt(ret))</c> for a
        ///         scalar → <see cref="SqrtLikeNumPy"/>.</item>
        ///     </list>
        ///     <para>
        ///     Consequences, all NumPy's (probed 2.4.2): a NaN or ±inf element gives NaN even alone; <c>ddof &gt;= n</c>
        ///     gives +inf for a positive sum of squares and NaN for a zero one, never a negative variance; a float16 /
        ///     float32 input keeps its dtype at every size; an integer <c>dtype=</c> truncates the mean
        ///     (<c>np.var([1, 2, 3, 4], dtype=np.int32)</c> is 1); a complex input with <c>dtype=float64</c> keeps only the
        ///     real part of the mean; <c>np.std(x, axis, dtype=np.int32)</c> raises while the flat form truncates.
        ///     </para>
        /// </remarks>
        internal NDArray VarStdNumPy(NDArray arr, int? axis_, bool keepdims, int ddof, DType dtype, bool takeSqrt)
        {
            var shape = arr.Shape;

            // The uninitialized-shape sentinel (not a size-0 array) has nothing to reduce; it passes through as before.
            if (shape.IsEmpty)
                return arr;

            // _count_reduce_items(arr, axis): normalize_axis_index runs FIRST, so an axis outside the operand raises
            // AxisError before any arithmetic — for a 0-d operand too, where ufunc.reduce would read axis 0 / -1 as
            // "every axis".
            int? axis = null;
            long rcount;
            if (axis_ is null)
                rcount = shape.size;
            else
            {
                axis = NormalizeAxis(axis_.Value, shape.NDim);
                rcount = shape.dimensions[axis.Value];
            }

            // `if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool)): dtype = f8`.
            NPTypeCode inT = arr.GetTypeCode;
            bool integerOrBool = inT.IsInteger() || inT == NPTypeCode.Boolean || inT == NPTypeCode.Char;
            DType loop = dtype ?? (integerOrBool ? np.float64 : null);

            // arrmean = umr_sum(arr, axis, dtype, keepdims=True), then divided by the count IN PLACE (NumPy's out=arrmean
            // keeps the reduction's layout, which votes for x's layout below). A 0-d sum comes back as the read-only
            // reduction scalar, so it is copied first (0-d has no layout to keep).
            var arrmean = ReduceAdd(arr, axis, keepdims: true, dtype: loop, @out: null);
            if (!arrmean.Shape.IsWriteable)
                arrmean = arrmean.copy();
            DivideByCountLikeNumPy(arrmean, rcount);

            // x = subtract(arr, arrmean): both are arrays, so NEP 50 promotes them as strong operands.
            NPTypeCode xT = np.result_type(arr.dtype, arrmean.dtype).GetTypeCode();
            // NumPy squares x with umr_multiply(x, x) for a float or integer INPUT (a complex x too, when dtype= is
            // complex), takes real² + imag² of a complex x otherwise (a complex input, or a bool input with
            // dtype=complex), and computes (x * conj(x)).real for a bool input with a real x — x² again.
            bool floatOrIntegerInput = inT.IsFloatingPoint() || inT.IsInteger() || inT == NPTypeCode.Char;
            var square = !floatOrIntegerInput && xT == NPTypeCode.Complex ? DeviationSquare.Modulus
                : xT == NPTypeCode.Complex ? DeviationSquare.ComplexProduct
                : DeviationSquare.Real;
            NPTypeCode sqT = square == DeviationSquare.Modulus ? NPTypeCode.Double : xT;
            bool castFree = loop is null || loop.GetTypeCode() == sqT;

            // A flat reduction hoists the mean as a 0-d PARAMETER of the fused pass (read once, no stride-0 operand);
            // an axis reduction broadcasts the keepdims mean.
            bool flat = axis is null;
            NDArray meanOperand = flat ? arrmean.reshape(Shape.NewScalar()) : arrmean;

            // The deviations are computed by ONE fused np.evaluate pass, except where the eager ufuncs are the better
            // (or the only) engine: Decimal has no fused kernels, a host without dynamic code has none at all, and
            // float16 runs np.evaluate's scalar half arithmetic (~8 ns per element) where np.subtract / np.multiply
            // take the SIMD widen-compute-narrow kernels — the same HALF loops, so the same bits.
            bool eager = !DirectILKernelGenerator.Enabled || inT == NPTypeCode.Decimal || xT == NPTypeCode.Half;

            NDArray ret;
            if (!eager && flat && castFree && StreamsAsOneRun(arr)
                && sqT is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex)
            {
                // ret = umr_sum(x**2) over ONE memory run without a cast: NumPy's x is a fresh array in arr's memory order,
                // summed as one pairwise call; np.evaluate's flat Sum streams exactly that schedule over the fused child,
                // so x never reaches memory.
                var sq = SquaredDeviationExpr(arr, meanOperand, square);
                ret = np.evaluate(keepdims ? NDExpr.Sum(sq, keepdims: true) : NDExpr.Sum(sq));
            }
            else
            {
                // x laid out as NumPy lays it out (DeviationLayout); the deviation sum's order — pairwise per row or
                // slab by slab — follows this layout.
                var layout = DeviationLayout(shape, axis);
                long x2Bytes = layout.size * sqT.SizeOf();
                ret = null;
                long chunkBytes = VarStdChunkBytesTestOverride > 0 ? VarStdChunkBytesTestOverride : VarStdChunkBytes;
                long chunkMin = VarStdChunkBytesTestOverride > 0 ? VarStdChunkBytesTestOverride : VarStdChunkMinBytes;
                if (x2Bytes >= chunkMin)
                {
                    if (!flat && (castFree || CastLoopChunksExactly(layout, axis.Value)))
                        ret = ChunkedDeviationSum(arr, meanOperand, layout, axis.Value, keepdims, loop, xT, sqT, square, eager, chunkBytes);
                    else if (flat && !castFree && !eager && StreamsAsOneRun(arr)
                             && loop.GetTypeCode() is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Complex)
                        ret = FlatFillDeviationSum(arr, meanOperand, keepdims, loop.GetTypeCode(), xT, sqT, square, chunkBytes);
                }
                if (ret is null)
                {
                    var x2 = new NDArray(sqT, layout, false);
                    FillSquaredDeviations(arr, meanOperand, x2, xT, square, eager);
                    ret = ReduceAdd(x2, axis, keepdims, loop, @out: null);
                }
            }

            // ret / max(rcount - ddof, 0). A 0-d result is NumPy's scalar: computed on a writeable copy, handed out read-only.
            bool scalarResult = ret.ndim == 0;
            if (!ret.Shape.IsWriteable)
                ret = ret.copy();
            DivideByCountLikeNumPy(ret, Math.Max(rcount - ddof, 0));
            if (takeSqrt)
                SqrtLikeNumPy(ret, scalarResult);
            return scalarResult ? ret.MarkReductionScalar() : ret;
        }

        /// <summary>
        ///     Whether a flat reduction of the fused deviation child over <paramref name="arr"/> walks the same element order
        ///     NumPy's sum over its fresh <c>x</c> does: a C- or F-contiguous operand, any dense axis permutation of one
        ///     block, or a 1-D operand at any stride. NumPy's <c>x</c> then lies in one memory run in that order, and
        ///     <c>np.evaluate</c>'s flat Sum reduces the child in that order too. A non-dense N-D view (a slice of a
        ///     transposed block, say) gets an <c>x</c> in its own K order, which only the materialized route reproduces.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <returns>True when the streamed flat sum is NumPy's schedule over <c>x</c>.</returns>
        private static bool StreamsAsOneRun(NDArray arr)
        {
            var s = arr.Shape;
            if (s.IsBroadcasted)
                return false;
            return s.NDim <= 1 || s.IsContiguous || s.IsFContiguous || IsSharedDensePermutation(new[] { arr });
        }

        /// <summary>
        ///     How <see cref="VarStdNumPy"/> squares the deviation <c>x = arr - mean</c> — the three statements of NumPy's
        ///     <c>_var</c>, which differ in their bits for complex values.
        /// </summary>
        private enum DeviationSquare
        {
            /// <summary><c>umr_multiply(x, x)</c> of a real <c>x</c> (a float, integer, char, decimal or bool input with a
            /// real loop): <c>x²</c>, rounded once.</summary>
            Real,

            /// <summary><c>umr_multiply(x, x)</c> of a COMPLEX <c>x</c> of a real input (<c>dtype=</c> complex): NumPy's
            /// complex multiply of x by itself, so the result is complex.</summary>
            ComplexProduct,

            /// <summary><c>x.real² + x.imag²</c> (NumPy's view-multiply-add) of a complex <c>x</c> of a complex or bool
            /// input: two rounded squares, then a rounded add, into a float64 result.</summary>
            Modulus,
        }

        /// <summary>
        ///     The smallest squared-deviation array (in bytes) <see cref="VarStdNumPy"/> computes chunk by chunk
        ///     (<see cref="ChunkedDeviationSum"/>) instead of in one buffer: 8 MiB, where a one-buffer pass starts to pay
        ///     main-memory traffic (and, beyond the pool's 64 MiB, fresh-page faults: 6.5 of the 21.5 ms of a 10M float64
        ///     axis variance).
        /// </summary>
        internal const long VarStdChunkMinBytes = 8L << 20;

        /// <summary>
        ///     The target size of one chunk of squared deviations: 256 KB, which stays in L2 between its write and the
        ///     sum that reads it back.
        /// </summary>
        internal const long VarStdChunkBytes = 256L << 10;

        /// <summary>
        ///     Test hook: when positive (on the current thread), both the chunking threshold and the chunk size of
        ///     <see cref="VarStdNumPy"/>, so a test can run the chunked route on small arrays and compare it with the
        ///     one-buffer route byte for byte. Thread-static so a parallel test never perturbs another's run.
        /// </summary>
        [ThreadStatic] internal static long VarStdChunkBytesTestOverride;

        /// <summary>
        ///     Test hook: incremented (on the current thread) each time <see cref="ChunkedDeviationSum"/> computes a
        ///     result, so a test can prove the chunked route ran rather than silently falling back to the one-buffer route
        ///     (which gives the same bits and would hide a dead path). Thread-static for the same reason as
        ///     <see cref="VarStdChunkBytesTestOverride"/>.
        /// </summary>
        [ThreadStatic] internal static int VarStdChunkedRuns;

        /// <summary>
        ///     The layout of NumPy's <c>x = arr - arrmean</c> (NpyIter's K order of the two operands, see
        ///     <see cref="Shape.NpyIterOutputShape"/>). A C- or F-contiguous operand gives a C or F <c>x</c> (the mean,
        ///     laid out in arr's own order, agrees on every axis), so those skip the vote.
        /// </summary>
        /// <param name="arrShape">The operand's shape.</param>
        /// <param name="axis">The reduced axis (normalized), or null for a flat reduction (the mean is then a 0-d
        /// array, which never votes).</param>
        /// <returns>A fresh, dense shape of <paramref name="arrShape"/>'s dimensions.</returns>
        private static Shape DeviationLayout(Shape arrShape, int? axis)
        {
            var dims = (long[])arrShape.dimensions.Clone();
            if (arrShape.IsContiguous)
                return Shape.Adopt(dims);
            if (arrShape.IsFContiguous)
                return Shape.Adopt(dims, 'F');
            if (axis is null)
                return Shape.NpyIterOutputShape(dims, new ReadOnlySpan<Shape>(in arrShape));
            return Shape.NpyIterOutputShape(dims, new[] { arrShape, KeepdimsReductionLayout(arrShape, axis.Value) });
        }

        /// <summary>
        ///     The layout NumPy allocates for <c>umr_sum(arr, axis, keepdims=True)</c>: the reduction iterator sorts arr's
        ///     axes into its K order (only arr votes — the unallocated result has no strides yet), and
        ///     <c>npyiter_new_temp_array</c> gives the result dense strides in that order, the reduced axis with extent 1.
        ///     Used only as the mean's VOTE in <c>x = arr - arrmean</c>'s layout, so its strides must order the axes as
        ///     NumPy's do; an extent-1 axis abstains there, whatever its stride.
        /// </summary>
        /// <param name="arrShape">The operand's shape.</param>
        /// <param name="axis">The reduced axis, already normalized.</param>
        /// <returns>A dense shape of <paramref name="arrShape"/>'s dimensions with <paramref name="axis"/> set to 1.</returns>
        /// <remarks>
        ///     NumSharp's own keepdims reduction result is C-ordered for an input that is neither C- nor F-contiguous, so
        ///     its strides would out-vote arr on a permuted input ("C order wins conflicts") and lay <c>x</c> out in C
        ///     order — a different summation order for the deviation sum (probed: <c>np.var(a.transpose(2, 0, 1),
        ///     axis=2)</c> differed by 1 ULP).
        /// </remarks>
        private static Shape KeepdimsReductionLayout(Shape arrShape, int axis)
        {
            // arr's K order over its FULL dims (the reduced axis votes too, as it does in the reduction iterator).
            var full = Shape.NpyIterOutputShape((long[])arrShape.dimensions.Clone(), new ReadOnlySpan<Shape>(in arrShape));
            var dims = (long[])arrShape.dimensions.Clone();
            dims[axis] = 1;
            return DenseInOrder(dims, full.DenseAxisOrder());
        }

        /// <summary>
        ///     A dense layout of <paramref name="dims"/> whose axes lie in memory in <paramref name="outerFirst"/>'s order
        ///     — the innermost axis last in the array gets stride 1, each further one the product of the extents inside
        ///     it (an extent-1 axis takes the running stride without growing it), as <c>npyiter_new_temp_array</c>
        ///     assigns them.
        /// </summary>
        /// <param name="dims">The dimensions (adopted by the returned shape).</param>
        /// <param name="outerFirst">A permutation of the axes, outermost first (<see cref="Shape.DenseAxisOrder"/>).</param>
        /// <returns>The dense shape.</returns>
        private static Shape DenseInOrder(long[] dims, int[] outerFirst)
        {
            var strides = new long[dims.Length];
            long stride = 1;
            for (int i = outerFirst.Length - 1; i >= 0; i--)
            {
                int ax = outerFirst[i];
                strides[ax] = stride;
                stride *= dims[ax];
            }
            return Shape.Adopt(dims, strides);
        }

        /// <summary>
        ///     <c>ret = umr_sum(x², axis, dtype, keepdims)</c> without the whole <c>x²</c> in memory: the deviations are
        ///     computed and summed in chunks along the OUTERMOST kept axis of <c>x</c>'s layout, each chunk a set of whole
        ///     outputs, into a buffer that stays in L2. Each output's sum runs exactly as over the full array — NumPy
        ///     sums every output on its own (one pairwise call per row when the reduced axis is innermost, one add per
        ///     element of the reduced axis otherwise), so splitting the outputs between calls changes no bit. When every
        ///     kept axis lies inside the reduced axis in memory, the reduced axis is split instead
        ///     (<see cref="RowBlockDeviationSum"/>).
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="mean">The keepdims mean (same rank as <paramref name="arr"/>).</param>
        /// <param name="layout">The layout of the full <c>x²</c> (<see cref="DeviationLayout"/>).</param>
        /// <param name="axis">The reduced axis, normalized.</param>
        /// <param name="keepdims">Whether the result keeps the reduced axis.</param>
        /// <param name="loop">The sum's <c>dtype=</c>, or null. A cast loop is chunked only where its buffered fills
        /// cannot split an output (<see cref="CastLoopChunksExactly"/>).</param>
        /// <param name="xT">The deviation dtype.</param>
        /// <param name="sqT">The squared-deviation dtype.</param>
        /// <param name="square">How the deviation is squared.</param>
        /// <param name="eager">True to compute the deviations with the eager ufuncs instead of np.evaluate.</param>
        /// <param name="chunkBytes">The target chunk size in bytes.</param>
        /// <returns>The deviation sums, laid out as the one-buffer route's sum would be; null when the layout has no
        /// kept axis to split, or one chunk would be the whole array — the caller then takes the one-buffer route.</returns>
        /// <remarks>
        ///     A chunk is never 1 wide along a kept axis that lies INSIDE the reduced axis in memory: with one such
        ///     column left, NumPy would see a one-output reduction and sum it pairwise, not one add per element. Every
        ///     chunk is at least 2 wide, and a remainder of 1 joins the chunk before it.
        /// </remarks>
        private NDArray ChunkedDeviationSum(NDArray arr, NDArray mean, Shape layout, int axis, bool keepdims, DType loop,
            NPTypeCode xT, NPTypeCode sqT, DeviationSquare square, bool eager, long chunkBytes)
        {
            var dims = layout.dimensions;
            int nd = dims.Length;
            int[] order = layout.DenseAxisOrder();
            int c = -1;
            foreach (int ax in order)
            {
                if (ax != axis && dims[ax] > 1)
                {
                    c = ax;
                    break;
                }
            }
            if (c < 0)
                return null;

            // Every kept axis inside the reduced axis in memory (the reduced axis outermost, as axis 0 of a C array):
            // chunks of whole outputs would be narrow column blocks, each reading thousands of short strided rows, so
            // the reduced axis itself is split instead (RowBlockDeviationSum).
            if (Array.IndexOf(order, c) > Array.IndexOf(order, axis))
                return RowBlockDeviationSum(arr, mean, layout, order, axis, keepdims, loop, xT, sqT, square, eager, chunkBytes);

            long unit = layout.size / dims[c];
            long per = Math.Max(2, chunkBytes / Math.Max(1, unit * sqT.SizeOf()));
            if (per >= dims[c])
                return null;

            var src = new Slice[nd];
            for (int d = 0; d < nd; d++)
                src[d] = Slice.All;
            // The result's axis that indexes like layout axis c: the same with keepdims, one lower past the reduced
            // axis without.
            int rc = keepdims || c < axis ? c : c - 1;
            var dst = new Slice[keepdims ? nd : nd - 1];
            for (int d = 0; d < dst.Length; d++)
                dst[d] = Slice.All;

            NDArray ret = null, reused = null;
            long start = 0;
            while (start < dims[c])
            {
                long e = Math.Min(per, dims[c] - start);
                if (dims[c] - start - e == 1)
                    e++; // never leave a 1-wide last chunk (see remarks)

                src[c] = new Slice(start, start + e);
                var arrC = arr[src];
                var meanC = mean[src];
                var cdims = (long[])dims.Clone();
                cdims[c] = e;
                // A full-size chunk reuses one buffer; the (at most one) other size gets its own.
                NDArray x2c;
                if (e == per && reused is not null)
                    x2c = reused;
                else
                {
                    x2c = new NDArray(sqT, DenseInOrder(cdims, order), false);
                    if (e == per)
                        reused = x2c;
                }

                FillSquaredDeviations(arrC, meanC, x2c, xT, square, eager);
                var part = ReduceAdd(x2c, axis, keepdims, loop, @out: null);
                if (ret is null)
                {
                    // The first chunk names the result dtype; the layout is the one ReduceAdd gives the full x².
                    var reduced = new long[nd - 1];
                    for (int d = 0, k = 0; d < nd; d++)
                        if (d != axis)
                            reduced[k++] = dims[d];
                    ret = AllocateReductionResult(part.GetTypeCode, reduced, layout);
                    if (keepdims)
                        ret.Storage.ExpandDimension(axis);
                }

                dst[rc] = new Slice(start, start + e);
                np.copyto(ret[dst], part);
                part.Dispose();
                if (!ReferenceEquals(x2c, reused))
                    x2c.Dispose();
                start += e;
            }

            reused?.Dispose();
            VarStdChunkedRuns++;
            return ret;
        }

        /// <summary>
        ///     The smallest number of outputs per slab <see cref="RowBlockDeviationSum"/> accepts: each slab costs one
        ///     <c>np.add</c> call, so a narrower slab leaves the one-buffer route faster.
        /// </summary>
        private const long VarStdRowBlockMinOutputs = 512;

        /// <summary>
        ///     The chunked deviation sum for a layout whose reduced axis is OUTERMOST in memory (every kept axis inside
        ///     it): the outputs all advance together, one slab of <c>x²</c> per index of the reduced axis, so the reduced
        ///     axis is split into row blocks instead. NumPy's sum there starts each output at the identity (+0) and adds
        ///     the slabs in order (<c>out = out + x[a]</c>, one add per element per slab), which is exactly a running
        ///     accumulator updated by <c>np.add(acc, slab, out=acc)</c> slab by slab, however the slabs are grouped.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="mean">The keepdims mean.</param>
        /// <param name="layout">The layout of the full <c>x²</c>.</param>
        /// <param name="order">The layout's axes, outermost first (<see cref="Shape.DenseAxisOrder"/>).</param>
        /// <param name="axis">The reduced axis, normalized.</param>
        /// <param name="keepdims">Whether the result keeps the reduced axis.</param>
        /// <param name="loop">The sum's <c>dtype=</c>, or null: the accumulator's dtype (each slab is cast into it by the
        /// add, as NumPy's buffer casts it before the add loop).</param>
        /// <param name="xT">The deviation dtype.</param>
        /// <param name="sqT">The squared-deviation dtype (the accumulator's when <paramref name="loop"/> is null).</param>
        /// <param name="square">How the deviation is squared.</param>
        /// <param name="eager">True to compute the deviations with the eager ufuncs.</param>
        /// <param name="chunkBytes">The target block size in bytes.</param>
        /// <returns>The deviation sums, laid out as the one-buffer route's sum would be; null (the caller takes the
        /// one-buffer route) when a slab holds fewer than <see cref="VarStdRowBlockMinOutputs"/> outputs or one block
        /// would be the whole array.</returns>
        private NDArray RowBlockDeviationSum(NDArray arr, NDArray mean, Shape layout, int[] order, int axis, bool keepdims,
            DType loop, NPTypeCode xT, NPTypeCode sqT, DeviationSquare square, bool eager, long chunkBytes)
        {
            var dims = layout.dimensions;
            int nd = dims.Length;
            long rows = dims[axis];
            long nOut = layout.size / rows;
            if (nOut < VarStdRowBlockMinOutputs)
                return null;
            long rowsPer = Math.Max(1, chunkBytes / (nOut * sqT.SizeOf()));
            if (rowsPer >= rows)
                return null;

            // A block is a C buffer of the axes in MEMORY order (outermost first). The reduced axis is the outermost
            // non-unit one, so a block is rowsPer whole slabs back to back; viewed through the inverse permutation it is
            // the logical, layout-ordered destination of the deviation pass.
            var inverse = new int[nd];
            for (int p = 0; p < nd; p++)
                inverse[order[p]] = p;
            var src = new Slice[nd];
            for (int d = 0; d < nd; d++)
                src[d] = Slice.All;

            // The running sums, one per output, in the kept axes' memory order: the identity +0 (np.zeros), which NumPy
            // assigns before the first slab.
            NPTypeCode accT = loop?.GetTypeCode() ?? sqT;
            var acc = np.zeros(new Shape(nOut), accT);
            NDArray block = null;
            long blockRows = -1;
            for (long a0 = 0; a0 < rows; a0 += rowsPer)
            {
                long r = Math.Min(rowsPer, rows - a0);
                if (r != blockRows)
                {
                    block?.Dispose();
                    var memDims = new long[nd];
                    for (int p = 0; p < nd; p++)
                        memDims[p] = order[p] == axis ? r : dims[order[p]];
                    block = new NDArray(sqT, Shape.Adopt(memDims), false);
                    blockRows = r;
                }

                src[axis] = new Slice(a0, a0 + r);
                var arrBlock = arr[src];
                var logical = np.transpose(block, inverse);
                FillSquaredDeviations(arrBlock, mean, logical, xT, square, eager);
                logical.Dispose();
                arrBlock.Dispose();

                // One add per slab, in slab order: acc = acc + x[a] (NumPy's SLAB step, the slab the second operand).
                var slabs = block.reshape(r, nOut);
                for (long t = 0; t < r; t++)
                {
                    var slab = slabs[t];
                    np.add(acc, slab, @out: acc, dtype: accT);
                    slab.Dispose();
                }
                slabs.Dispose();
            }
            block?.Dispose();

            // acc is the result in the kept axes' memory order: view it logically and copy it into the layout the
            // one-buffer route's ReduceAdd allocates.
            var keptMem = new long[nd - 1];
            var keptOrder = new int[nd - 1];
            for (int p = 0, k = 0; p < nd; p++)
            {
                if (order[p] == axis)
                    continue;
                keptMem[k] = dims[order[p]];
                keptOrder[k++] = order[p];
            }
            var keptAxes = new int[nd - 1];
            for (int d = 0, j = 0; d < nd; d++)
            {
                if (d == axis)
                    continue;
                keptAxes[j++] = Array.IndexOf(keptOrder, d);
            }
            var accLogical = np.transpose(acc.reshape(Shape.Adopt(keptMem)), keptAxes);

            var reduced = new long[nd - 1];
            for (int d = 0, k = 0; d < nd; d++)
                if (d != axis)
                    reduced[k++] = dims[d];
            var ret = AllocateReductionResult(acc.GetTypeCode, reduced, layout);
            np.copyto(ret, accLogical);
            if (keepdims)
                ret.Storage.ExpandDimension(axis);
            VarStdChunkedRuns++;
            return ret;
        }

        /// <summary>
        ///     Whether an axis sum in a CAST loop can be computed chunk by chunk with NumPy's bits: NumPy buffers the cast
        ///     operand in fills of <c>np.getbufsize()</c> elements, and a fill never changes a SLAB's order (each output
        ///     adds its elements one at a time, whatever the fills hold), but it SPLITS a pinned row longer than the buffer
        ///     into several pairwise calls. So a SLAB (some kept axis inside the reduced axis in memory) always chunks, a
        ///     pinned reduction (the reduced axis innermost) only when its rows fit one buffer.
        /// </summary>
        /// <param name="layout">The layout of the full <c>x²</c>.</param>
        /// <param name="axis">The reduced axis, normalized.</param>
        /// <returns>True when the chunked routes reproduce the cast loop's schedule.</returns>
        private static bool CastLoopChunksExactly(Shape layout, int axis)
        {
            var dims = layout.dimensions;
            var strides = layout.strides;
            for (int d = 0; d < dims.Length; d++)
                if (d != axis && dims[d] > 1 && strides[d] < strides[axis])
                    return true; // a kept axis inside the reduced one: a SLAB
            return dims[axis] <= np.getbufsize();
        }

        /// <summary>
        ///     The flat deviation sum in a CAST loop (<c>np.var(float64, dtype=np.float32)</c>), chunk by chunk: NumPy's
        ///     <c>x</c> is one run in memory order, and its buffered sum is <c>out += pairwise_sum(cast(fill))</c> per fill of
        ///     <c>np.getbufsize()</c> elements, from the identity. Each chunk here is a whole number of fills of that run:
        ///     the deviations of the matching block of <paramref name="arr"/> (in memory order) are written into an L2
        ///     buffer, each fill is reduced by NumPy's exact flat engine (<see cref="NumPyAddReduce.TryReduceFlat"/>: one
        ///     fill, so one pairwise call) and added to the running sum in the loop dtype.
        /// </summary>
        /// <param name="arr">The operand: one dense run in memory order, or 1-D (<see cref="StreamsAsOneRun"/>).</param>
        /// <param name="mean">The 0-d mean.</param>
        /// <param name="keepdims">Whether the result keeps every axis with extent 1.</param>
        /// <param name="loopT">The loop dtype: Single, Double or Complex. (float16 is excluded: NumPy adds the fill's
        /// UNROUNDED float32 pairwise sum to the running half, and the engine returns it rounded.)</param>
        /// <param name="xT">The deviation dtype.</param>
        /// <param name="sqT">The squared-deviation dtype.</param>
        /// <param name="square">How the deviation is squared.</param>
        /// <param name="chunkBytes">The target chunk size in bytes (rounded down to whole fills, at least one).</param>
        /// <returns>The 0-d (or keepdims) sum in the loop dtype; null when the flat engine declines a fill (the caller
        /// then takes the one-buffer route, which reaches the same engine).</returns>
        /// <remarks>
        ///     The engine's result is <c>0 + pairwise(fill)</c>, while NumPy adds the raw pairwise sum. The two agree: the
        ///     only value the identity add changes is <c>-0.0</c>, and adding <c>-0.0</c> or <c>+0.0</c> to the running
        ///     sum (itself never <c>-0.0</c>: it starts at <c>+0.0</c>) gives the same bits.
        /// </remarks>
        private unsafe NDArray FlatFillDeviationSum(NDArray arr, NDArray mean, bool keepdims, NPTypeCode loopT,
            NPTypeCode xT, NPTypeCode sqT, DeviationSquare square, long chunkBytes)
        {
            long n = arr.size;
            long fill = np.getbufsize();
            long fillsPer = Math.Max(1, chunkBytes / Math.Max(1, fill * sqT.SizeOf()));
            long per = fillsPer * fill;
            if (per >= n)
                return null;

            // arr's elements in MEMORY order as one 1-D view: a dense layout transposed into its memory order is
            // C-contiguous, so the reshape is a view; a 1-D operand already is that order (any stride).
            NDArray run = arr.ndim <= 1 ? arr.reshape(n) : np.transpose(arr, arr.Shape.DenseAxisOrder()).reshape(n);

            byte* slot = stackalloc byte[16];
            double accD = 0d;
            float accF = 0f;
            double accRe = 0d, accIm = 0d;
            var buffer = new NDArray(sqT, new Shape(per), false);
            for (long s0 = 0; s0 < n; s0 += per)
            {
                long m = Math.Min(per, n - s0);
                var block = run[new Slice(s0, s0 + m)];
                var x2c = m == per ? buffer : buffer[new Slice(0, m)];
                FillSquaredDeviations(block, mean, x2c, xT, square, eager: false);
                block.Dispose();
                for (long f0 = 0; f0 < m; f0 += fill)
                {
                    var one = x2c[new Slice(f0, Math.Min(f0 + fill, m))];
                    bool ok = NumPyAddReduce.TryReduceFlat(one, loopT, fill, slot);
                    one.Dispose();
                    if (!ok)
                    {
                        if (!ReferenceEquals(x2c, buffer))
                            x2c.Dispose();
                        buffer.Dispose();
                        return null;
                    }
                    // out += fill's sum, in the loop dtype (a complex add is two independent double adds).
                    switch (loopT)
                    {
                        case NPTypeCode.Single: accF += *(float*)slot; break;
                        case NPTypeCode.Double: accD += *(double*)slot; break;
                        default:
                            accRe += ((double*)slot)[0];
                            accIm += ((double*)slot)[1];
                            break;
                    }
                }
                if (!ReferenceEquals(x2c, buffer))
                    x2c.Dispose();
            }
            buffer.Dispose();

            NDArray ret = loopT switch
            {
                NPTypeCode.Single => NDArray.Scalar(accF),
                NPTypeCode.Double => NDArray.Scalar(accD),
                _ => NDArray.Scalar(new System.Numerics.Complex(accRe, accIm)),
            };
            if (keepdims)
            {
                var ones = new long[arr.ndim];
                for (int d = 0; d < ones.Length; d++)
                    ones[d] = 1;
                ret = ret.reshape(Shape.Adopt(ones));
            }
            VarStdChunkedRuns++;
            return ret;
        }

        /// <summary>
        ///     Writes NumPy's squared deviations of <paramref name="arr"/> from <paramref name="mean"/> into
        ///     <paramref name="x2"/> (already laid out and typed): one fused np.evaluate pass, or the eager ufuncs.
        /// </summary>
        /// <param name="arr">The operand (or a chunk of it).</param>
        /// <param name="mean">The mean: a 0-d array (flat) or the keepdims mean (or its chunk).</param>
        /// <param name="x2">The destination.</param>
        /// <param name="xT">The deviation dtype (the eager subtract's loop).</param>
        /// <param name="square">How the deviation is squared.</param>
        /// <param name="eager">True for the eager ufuncs.</param>
        /// <remarks>
        ///     A real-float deviation of an operand of ANOTHER dtype (an integer, bool or char input, or a float input with
        ///     a wider <c>dtype=</c>) is written in two steps: the operand cast into <paramref name="x2"/> by the SIMD cast
        ///     kernels, then the fused pass over <paramref name="x2"/> in place. np.evaluate converts a mixed-width leaf
        ///     one element at a time (int32 → float64 cost 0.42 ns per element against 0.16 for a float64 leaf), and
        ///     the two steps give the same bits: NumPy's subtract casts that operand to the loop dtype the same way (exact
        ///     for every integer up to 32 bits and every narrower float, round-to-nearest-even for int64 / uint64) before
        ///     it subtracts.
        /// </remarks>
        private static void FillSquaredDeviations(NDArray arr, NDArray mean, NDArray x2, NPTypeCode xT, DeviationSquare square, bool eager)
        {
            if (eager)
            {
                SquaredDeviationsEager(arr, mean, x2, xT, square);
                return;
            }

            if (square == DeviationSquare.Real && arr.GetTypeCode != xT && xT is NPTypeCode.Single or NPTypeCode.Double)
            {
                np.copyto(x2, arr, casting: "unsafe");
                np.evaluate(NDExpr.Square(NDExpr.Arr(x2) - NDExpr.Arr(mean)), @out: x2);
                return;
            }

            np.evaluate(SquaredDeviationExpr(arr, mean, square), @out: x2);
        }

        /// <summary>
        ///     The fused <c>(arr - mean)²</c> tree of <see cref="VarStdNumPy"/>, typed by NumPy's per-ufunc rules: the
        ///     subtract promotes the two arrays (a 0-d mean is a STRONG operand in <see cref="NDExpr"/>, as NumPy's
        ///     keepdims mean array is), the square keeps that dtype.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="mean">The mean: a 0-d array (flat) or the keepdims mean (axis).</param>
        /// <param name="square">How the deviation is squared: <c>x²</c>; <c>x * x</c> (NumPy's complex multiply,
        /// whose bits are not the complex square's in every case); or <c>real² + imag²</c>, the two squares rounded
        /// separately and then added, as NumPy's view-multiply-add does.</param>
        /// <returns>The elementwise tree; the caller wraps it in a Sum or evaluates it into a buffer.</returns>
        private static NDExpr SquaredDeviationExpr(NDArray arr, NDArray mean, DeviationSquare square)
        {
            NDExpr d = NDExpr.Arr(arr) - NDExpr.Arr(mean);
            return square switch
            {
                DeviationSquare.Modulus => NDExpr.Square(NDExpr.Real(d)) + NDExpr.Square(NDExpr.Imag(d)),
                DeviationSquare.ComplexProduct => d * (NDExpr.Arr(arr) - NDExpr.Arr(mean)),
                _ => NDExpr.Square(d),
            };
        }

        /// <summary>
        ///     The eager form of <see cref="SquaredDeviationExpr"/>: <c>x = subtract(arr, mean)</c> in the deviation dtype
        ///     <paramref name="xT"/> written into <paramref name="x2"/>'s layout, then <c>multiply(x, x, out=x)</c>; or, for
        ///     <see cref="DeviationSquare.Modulus"/>, the deviation in its own complex buffer and then
        ///     <c>real² + imag²</c> into <paramref name="x2"/>. Used for Decimal (no fused kernels; the decimal subtract and
        ///     multiply keep all 28 digits), float16 (the SIMD half kernels) and a host without dynamic code.
        /// </summary>
        /// <param name="arr">The operand.</param>
        /// <param name="mean">The mean: a 0-d array (flat) or the keepdims mean (axis).</param>
        /// <param name="x2">The destination, already laid out and typed as the squared deviations.</param>
        /// <param name="xT">The deviation dtype. Passed as the subtract's loop because NumSharp's eager ufuncs read a 0-d
        /// operand of the same kind as a WEAK scalar, while NumPy's mean is an array (strong): a float32 input with a
        /// float64 mean must subtract in float64.</param>
        /// <param name="square">How the deviation is squared.</param>
        private static void SquaredDeviationsEager(NDArray arr, NDArray mean, NDArray x2, NPTypeCode xT, DeviationSquare square)
        {
            if (square == DeviationSquare.Modulus)
            {
                var x = new NDArray(NPTypeCode.Complex, x2.Shape, false);
                np.subtract(arr, mean, @out: x, dtype: xT);
                var re = np.real(x);
                var im = np.imag(x);
                np.multiply(re, re, @out: re);
                np.multiply(im, im, @out: im);
                np.add(re, im, @out: x2);
                return;
            }

            np.subtract(arr, mean, @out: x2, dtype: xT);
            np.multiply(x2, x2, @out: x2);
        }

        /// <summary>
        ///     NumPy's <c>true_divide(a, rcount, out=a, casting='unsafe')</c> (and <c>a.dtype.type(a / rcount)</c> for a
        ///     scalar, the same arithmetic), in place. <c>rcount</c> is an <c>np.intp</c> — a STRONG int64 — so every real
        ///     dtype divides in the float64 loop and the quotient is rounded ONCE back into <paramref name="a"/>'s dtype (a
        ///     float32 count of 2²⁴ + 1 is not rounded first); complex divides in the complex loop by <c>n + 0j</c>
        ///     (Smith's algorithm); an integer or bool <paramref name="a"/> (an integer <c>dtype=</c>) gets NumPy's unsafe
        ///     cast of the float64 quotient (truncation; NaN and ±inf as NumPy's casts give them).
        /// </summary>
        /// <param name="a">The sum to divide; overwritten with the quotient. Writeable.</param>
        /// <param name="count">The divisor (the element count, or <c>max(n - ddof, 0)</c>); 0 gives NumPy's NaN / ±inf.</param>
        private static void DivideByCountLikeNumPy(NDArray a, long count)
        {
            var divisor = NDArray.Scalar(count);
            switch (a.GetTypeCode)
            {
                case NPTypeCode.Double:
                case NPTypeCode.Complex:
                case NPTypeCode.Decimal:
                    np.true_divide(a, divisor, @out: a);
                    break;
                case NPTypeCode.Half:
                case NPTypeCode.Single:
                    // The float64 loop, then a same_kind cast into the narrower float (one rounding from double).
                    np.true_divide(a, divisor, @out: a, dtype: np.float64);
                    break;
                default:
                {
                    // An integer / bool / char sum (an integer or bool dtype=): float64 quotient, unsafe cast back.
                    var q = np.true_divide(a, divisor);
                    np.copyto(a, q, casting: "unsafe");
                    break;
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>_std</c> step on the variance, in place: <c>sqrt(ret, out=ret)</c> for an array (the ufunc's own
        ///     loop: float16 via float32, float32, float64, complex <c>csqrt</c>) and <c>ret.dtype.type(sqrt(ret))</c> for a
        ///     scalar. They differ only for an integer or bool variance (an integer <c>dtype=</c>): the array form raises
        ///     NumPy's output-cast error (the sqrt loop is float, the out integer, <c>same_kind</c> refuses), the scalar
        ///     form computes the float root and truncates it into the dtype. Decimal (no NumPy dtype) takes
        ///     <see cref="DecimalMath.Sqrt"/> per element, keeping 28 digits where the decimal sqrt ufunc rounds through
        ///     double.
        /// </summary>
        /// <param name="ret">The variance; overwritten with its square root. Writeable.</param>
        /// <param name="scalarResult">True when <paramref name="ret"/> is 0-d (NumPy's scalar path).</param>
        /// <exception cref="ArgumentException">An integer or bool <paramref name="ret"/> with dimensions: "Cannot cast
        /// ufunc 'sqrt' output from dtype('float64') to dtype('int32') with casting rule 'same_kind'" (the float dtype
        /// names the sqrt loop the integer width selects).</exception>
        private static void SqrtLikeNumPy(NDArray ret, bool scalarResult)
        {
            var rt = ret.GetTypeCode;
            if (rt == NPTypeCode.Decimal)
            {
                // One root per output; the typed NDIter walk writes each in place.
                foreach (ref decimal v in np.nditer<decimal>(ret, writeable: true))
                    v = DecimalMath.Sqrt(v);
                return;
            }

            if ((rt.IsInteger() || rt == NPTypeCode.Boolean || rt == NPTypeCode.Char) && scalarResult)
            {
                var root = np.sqrt(ret);
                np.copyto(ret, root, casting: "unsafe");
                return;
            }

            // Float / complex: the ufunc in place. Integer / bool with dimensions: the ufunc's own same_kind refusal.
            np.sqrt(ret, @out: ret);
        }
    }
}
