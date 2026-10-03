using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // Which of clip's two float loops NumPy runs
        // =============================================================================
        //
        // NumPy's clip ufunc loop (numpy/_core/src/umath/clip.cpp, _npy_clip) tests the bound
        // STEPS of each inner-loop call: when both are 0 it runs _npy_clip_const_minmax_, and for
        // float32/float64 that fast path is a compare-select (x < lo ? lo : x, then x > hi ? hi : x;
        // a NaN bound fills the call with that NaN). Otherwise it runs _NPY_MIN(_NPY_MAX(x, lo), hi).
        // The two disagree on ties: np.clip(-0.0, 0.0, 1.0) keeps -0.0 on the const path and returns
        // +0.0 on the general one. They also disagree on the NaN a NaN bound produces.
        //
        // The steps are decided far from the loop: by the ufunc's trivial-loop shortcut, or by
        // NpyIter's axis order, coalescing and buffering cost model. A bound that varies along
        // the result can still reach the loop with stride 0 (a (2,1) bound against a (2,3)
        // array, when NpyIter does not buffer it), and a 0-d bound copied into a buffer still has
        // stride 0. This file ports exactly that decision:
        //   1. check_for_trivial_loop (ufunc_object.c): a 0-d or a short 1-D input needing a cast
        //      is cast into a contiguous copy; any other cast makes the trivial loop impossible;
        //   2. try_trivial_single_output_loop: same-shape, contiguous operands (a 0-d input
        //      broadcasts with step 0) run ONE call with fixed steps;
        //   3. otherwise NpyIter: npyiter_fill_axisdata, npyiter_find_best_axis_ordering,
        //      npyiter_flip_negative_strides, COPY_IF_OVERLAP, npyiter_coalesce_axes and
        //      npyiter_find_buffering_setup, whose per-operand inner strides are the steps.
        // Validated against live NumPy 2.4.2: 70,000 random layout / dtype / out= / where= /
        // overlap cases, every one deciding the same loop (the -0.0-vs-+0.0 tie reveals it).
        // =============================================================================

        /// <summary>
        /// Whether NumPy 2.4.2 runs clip's CONST float loop (both bound steps 0) for this call, rather than the
        /// general <c>MIN(MAX(x, lo), hi)</c> loop. Only meaningful for a two-bound clip in a float32/float64 loop;
        /// the other dtypes compute the same result either way.
        /// </summary>
        /// <param name="a">The array being clipped, as the caller passed it (before any NumSharp cast).</param>
        /// <param name="lo">The lower bound as passed. A 0-d bound counts as a Python scalar (NumSharp cannot tell a
        /// C# literal from a 0-d array): NumPy converts it into a fresh 0-d array of the loop dtype.</param>
        /// <param name="hi">The upper bound as passed, with the same 0-d rule.</param>
        /// <param name="out">The out= array, or null when NumPy allocates the result.</param>
        /// <param name="where">The where= mask, or null. A mask disables the trivial loop.</param>
        /// <param name="loop">The resolved loop dtype.</param>
        /// <param name="dims">The broadcast iteration shape (inputs, out and mask together).</param>
        /// <returns>True when the const loop runs, so ties keep the element and a NaN bound fills with itself.</returns>
        /// <remarks>
        /// <para>The result is uniform over the call: the trivial loop is one inner-loop call, and NpyIter gives every
        /// inner-loop call of an elementwise iteration the same strides (no reduce operand exists here).</para>
        /// <para>Strides are compared in BYTES, as NumPy does; the copies a step creates are modelled as fresh,
        /// contiguous arrays of the loop dtype that cannot overlap anything.</para>
        /// </remarks>
        internal static bool NumPyClipRunsConstLoop(NDArray a, NDArray lo, NDArray hi, NDArray @out, NDArray where,
                                                   NPTypeCode loop, long[] dims)
        {
            // Two scalar bounds: every route gives them stride 0 (the trivial loop's 0-d rule, NpyIter's broadcast
            // and its single-stride buffering). The common case decides here.
            if (lo.ndim == 0 && hi.ndim == 0)
                return true;

            long bufsize = np.getbufsize();
            int loopSize = loop.SizeOf();
            var ops = new ClipLoopOperand[3]
            {
                ClipLoopOperand.Of(a),
                ClipLoopOperand.OfBound(lo, loop),
                ClipLoopOperand.OfBound(hi, loop),
            };

            // ---- check_for_trivial_loop: only without a mask (the masked path never tries the trivial loop).
            bool trivial = where is null;
            if (trivial)
            {
                for (int i = 0; i < 3 && trivial; i++)
                {
                    ref var o = ref ops[i];
                    if (o.Dtype == loop)
                        continue;
                    // A safe (here: already validated same_kind) cast of a 0-d or a 1-D input no longer than the
                    // buffer is done eagerly into a contiguous copy, which keeps the trivial loop possible.
                    if (o.NDim == 0 || (o.NDim == 1 && o.Dims[0] <= bufsize))
                        o = ClipLoopOperand.Copy(o.Dims, loop);
                    else
                        trivial = false;
                }
                // An out of another dtype needs a cast on the way out, which the trivial loop cannot do.
                if (trivial && @out is not null && @out.typecode != loop)
                    trivial = false;
            }

            if (trivial && TryTrivialLoop(ops, @out, loopSize, out bool constTrivial))
                return constTrivial;

            return NpyIterConstLoop(ops, @out, where, loop, loopSize, dims, bufsize);
        }

        /// <summary>
        /// Port of <c>try_trivial_single_output_loop</c>: one inner-loop call over operands of one shape (0-d inputs
        /// broadcast with step 0), every N-D operand contiguous in one shared order, and no input overlapping the
        /// output in a way the single pass cannot honour.
        /// </summary>
        /// <param name="ops">The three inputs after <c>check_for_trivial_loop</c>'s copies.</param>
        /// <param name="out">The out= array, or null (the result is then allocated to fit).</param>
        /// <param name="loopSize">The loop dtype's item size, in bytes.</param>
        /// <param name="constLoop">When this returns true: whether both bound steps are 0.</param>
        /// <returns>True when NumPy takes the trivial loop; false when it falls through to NpyIter.</returns>
        private static bool TryTrivialLoop(ClipLoopOperand[] ops, NDArray @out, int loopSize, out bool constLoop)
        {
            constLoop = false;
            int operationNdim = 0;
            long[] operationShape = null;
            int operationOrder = 0;   // 0 = unset; else C (1) | F (2), as the flags read
            Span<long> fixedStride = stackalloc long[4];

            for (int i = 0; i < 4; i++)
            {
                ClipLoopOperand o;
                if (i < 3)
                    o = ops[i];
                else if (@out is null)
                    continue;
                else
                    o = ClipLoopOperand.Of(@out);

                // 0-d inputs broadcast with a 0 step (a 0-d out has no such special case).
                if (o.NDim == 0 && i < 3)
                {
                    fixedStride[i] = 0;
                    continue;
                }

                if (operationNdim == 0)
                {
                    operationNdim = o.NDim;
                    operationShape = o.Dims;
                }
                else if (o.NDim != operationNdim || !SameLongs(o.Dims, operationShape))
                {
                    return false;   // a broadcast: the trivial loop has a single stride per operand
                }

                if (o.NDim == 1)
                {
                    fixedStride[i] = o.ByteStrides[0];
                }
                else
                {
                    fixedStride[i] = o.ItemSize;
                    int order = (o.CContiguous ? 1 : 0) | (o.FContiguous ? 2 : 0);
                    if (order == 0)
                        return false;   // N-D operands must be contiguous
                    if (operationOrder == 0)
                        operationOrder = order;
                    else if (operationOrder != order)
                        return false;
                }
            }

            if (@out is not null)
            {
                var outOp = ClipLoopOperand.Of(@out);
                for (int i = 0; i < 3; i++)
                    if (!EquivalentlyIterableOverlapOk(ops[i], outOp))
                        return false;   // the full path makes the copy that keeps reads ahead of writes
                // A 1-D out walked backwards or onto itself cannot be written in one pass.
                if (operationNdim == 1 && outOp.ByteStrides[0] < outOp.ItemSize && outOp.ByteStrides[0] != 0)
                    return false;
            }

            constLoop = fixedStride[1] == 0 && fixedStride[2] == 0;
            return true;
        }

        /// <summary>
        /// Port of <c>PyArray_EQUIVALENTLY_ITERABLE_OVERLAP_OK(input, out, READ, NOREAD)</c>: true when reading
        /// <paramref name="input"/> and writing <paramref name="output"/> in one ascending pass is safe — the very same
        /// array, no shared memory, or an input that strides ahead of the output.
        /// </summary>
        /// <param name="input">The input operand (read).</param>
        /// <param name="output">The output operand (written, not read).</param>
        /// <returns>True when the trivial loop may proceed for this pair.</returns>
        private static bool EquivalentlyIterableOverlapOk(in ClipLoopOperand input, in ClipLoopOperand output)
        {
            long stride1 = TrivialPairIterationStride(input);
            // `arr1 == arr2` in NumPy is object identity: the in-place np.clip(x, lo, hi, out=x).
            if (input.Array is not null && ReferenceEquals(input.Array, output.Array) && stride1 != 0)
                return true;
            if (!input.MayShareMemory(output))
                return true;

            long stride2 = TrivialPairIterationStride(output);
            // An input that strides at least as fast as the output and starts at or after it is always read before
            // the output overwrites it. A zero stride (a reused element) is never ahead.
            if (stride1 > 0)
                return stride1 >= stride2 && input.DataStart >= output.DataStart;
            if (stride1 < 0)
                return stride1 <= stride2 && input.DataStart <= output.DataStart;
            return false;
        }

        /// <summary><c>PyArray_TRIVIAL_PAIR_ITERATION_STRIDE</c>: 0 for one element, the stride for 1-D, the item size otherwise.</summary>
        /// <param name="o">The operand.</param>
        /// <returns>The byte step the trivial loop advances the operand by.</returns>
        private static long TrivialPairIterationStride(in ClipLoopOperand o)
            => o.Size == 1 ? 0 : (o.NDim == 1 ? o.ByteStrides[0] : o.ItemSize);

        /// <summary>
        /// The bound steps NumPy's NpyIter would hand the clip loop: broadcast every operand over
        /// <paramref name="dims"/>, sort the axes, flip all-negative axes, apply COPY_IF_OVERLAP to the output,
        /// allocate it, coalesce, then run <c>npyiter_find_buffering_setup</c>'s cost model.
        /// </summary>
        /// <param name="ops">The three inputs (after any trivial-loop copies, which persist into this path).</param>
        /// <param name="out">The out= array, or null for an allocated result.</param>
        /// <param name="where">The mask (an extra read operand), or null.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="loopSize">The loop dtype's item size, in bytes.</param>
        /// <param name="dims">The broadcast iteration shape.</param>
        /// <param name="bufsize">The ufunc buffer size (<see cref="np.getbufsize"/>).</param>
        /// <returns>True when both bounds reach the inner loop with step 0.</returns>
        private static bool NpyIterConstLoop(ClipLoopOperand[] ops, NDArray @out, NDArray where,
                                             NPTypeCode loop, int loopSize, long[] dims, long bufsize)
        {
            int nd = dims.Length;
            int iout = 3;
            int nop = where is null ? 4 : 5;

            var opnd = new ClipLoopOperand[nop];
            ops.CopyTo(opnd, 0);
            bool outAllocated = @out is null;
            opnd[iout] = outAllocated ? default : ClipLoopOperand.Of(@out);
            if (where is not null)
                opnd[4] = ClipLoopOperand.Of(where);

            // NPY_OP_ITFLAG_CAST: an operand whose dtype is not the loop's is buffered through a cast. A read-only
            // 0-d input never is: npyiter_allocate_arrays copies it into the loop dtype eagerly and clears the flag.
            var cast = new bool[nop];
            var itemSize = new int[nop];
            for (int i = 0; i < nop; i++)
            {
                if (i < 3)
                    cast[i] = opnd[i].Dtype != loop && opnd[i].NDim != 0;
                else if (i == iout)
                    cast[i] = !outAllocated && opnd[i].Dtype != loop;
                // Buffer elements have the requested (loop) dtype; the mask keeps its own bool.
                itemSize[i] = i <= iout ? loopSize : opnd[i].ItemSize;
            }

            // ---- npyiter_fill_axisdata: per-operand byte stride along each iterator axis, innermost first
            //      (iterator axis iax = array axis nd-1-iax); an extent-1 or missing axis contributes 0.
            var st = new long[nop * Math.Max(nd, 1)];
            for (int i = 0; i < nop; i++)
            {
                if (i == iout && outAllocated)
                    continue;
                var od = opnd[i].Dims;
                var os = opnd[i].ByteStrides;
                int ond = od.Length;
                for (int iax = 0; iax < nd; iax++)
                {
                    int ax = nd - 1 - iax;
                    int k = ax - (nd - ond);
                    st[i * nd + iax] = dims[ax] == 1 || k < 0 || od[k] == 1 ? 0 : os[k];
                }
            }

            // ---- npyiter_find_best_axis_ordering: stable insertion sort; the first operand seeing both axes decides,
            //      a later disagreeing operand cancels a move (C order wins conflicts), unseen pairs are skipped.
            var perm = new int[Math.Max(nd, 1)];
            for (int i = 0; i < nd; i++)
                perm[i] = i;
            for (int axI0 = 1; axI0 < nd; axI0++)
            {
                int axIpos = axI0;
                int axJ0 = perm[axI0];
                for (int axI1 = axI0 - 1; axI1 >= 0; axI1--)
                {
                    bool ambig = true, shouldSwap = false;
                    int axJ1 = perm[axI1];
                    for (int i = 0; i < nop; i++)
                    {
                        long s0 = st[i * nd + axJ0], s1 = st[i * nd + axJ1];
                        if (s0 != 0 && s1 != 0)
                        {
                            if (Math.Abs(s1) <= Math.Abs(s0))
                                shouldSwap = false;
                            else if (ambig)
                                shouldSwap = true;
                            ambig = false;
                        }
                    }
                    if (!ambig)
                    {
                        if (shouldSwap)
                            axIpos = axI1;
                        else
                            break;
                    }
                }
                if (axIpos != axI0)
                {
                    for (int axI1 = axI0; axI1 > axIpos; axI1--)
                        perm[axI1] = perm[axI1 - 1];
                    perm[axIpos] = axJ0;
                }
            }

            // The axisdata in iteration order: shape and per-operand stride of sorted position k.
            int nax = Math.Max(nd, 1);
            var shp = new long[nax];
            var ad = new long[nax * nop];
            for (int k = 0; k < nax; k++)
            {
                shp[k] = nd == 0 ? 1 : dims[nd - 1 - perm[k]];
                for (int i = 0; i < nop; i++)
                    ad[k * nop + i] = nd == 0 ? 0 : st[i * nd + perm[k]];
            }

            // ---- npyiter_flip_negative_strides: never with an allocated output; otherwise an axis whose strides are
            //      all <= 0, one of them negative, is walked forwards.
            if (!outAllocated)
            {
                for (int k = 0; k < nax; k++)
                {
                    bool anyNegative = false, allNonPositive = true;
                    for (int i = 0; i < nop; i++)
                    {
                        long s = ad[k * nop + i];
                        if (s < 0) anyNegative = true;
                        else if (s != 0) allNonPositive = false;
                    }
                    if (anyNegative && allNonPositive)
                        for (int i = 0; i < nop; i++)
                            ad[k * nop + i] = -ad[k * nop + i];
                }
            }

            // ---- COPY_IF_OVERLAP: an out sharing memory with any read operand (and not its exact elementwise alias)
            //      is replaced by a temporary laid out like an allocated output, written back afterwards.
            bool forceCopy = false;
            if (!outAllocated)
            {
                for (int i = 0; i < nop && !forceCopy; i++)
                {
                    if (i == iout || opnd[i].IsFresh)
                        continue;
                    // Only the inputs carry NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE; the ARRAYMASK operand does not, so a
                    // mask that is the out itself still forces the copy.
                    if (i < 3 && opnd[i].IsExactElementwiseAlias(opnd[iout]))
                        continue;
                    if (opnd[iout].MayShareMemory(opnd[i]))
                        forceCopy = true;
                }
            }
            if (outAllocated || forceCopy)
            {
                long s = loopSize;
                for (int k = 0; k < nax; k++)
                {
                    ad[k * nop + iout] = shp[k] == 1 ? 0 : s;
                    s *= shp[k];
                }
                if (forceCopy)
                    cast[iout] = false;   // the temporary has the loop dtype; the write-back casts
            }

            // ---- npyiter_coalesce_axes: merge neighbours every operand walks as one run.
            int ncoal = 1;
            for (int k = 1; k < nax; k++)
            {
                long s0 = shp[ncoal - 1], s1 = shp[k];
                bool can = true;
                for (int i = 0; i < nop; i++)
                {
                    long a0 = ad[(ncoal - 1) * nop + i], a1 = ad[k * nop + i];
                    if (!((s0 == 1 && a0 == 0) || (s1 == 1 && a1 == 0)) && a0 * s0 != a1)
                    {
                        can = false;
                        break;
                    }
                }
                if (can)
                {
                    shp[ncoal - 1] = s0 * s1;
                    for (int i = 0; i < nop; i++)
                        if (ad[(ncoal - 1) * nop + i] == 0)
                            ad[(ncoal - 1) * nop + i] = ad[k * nop + i];
                }
                else
                {
                    shp[ncoal] = s1;
                    for (int i = 0; i < nop; i++)
                        ad[ncoal * nop + i] = ad[k * nop + i];
                    ncoal++;
                }
            }

            // ---- npyiter_find_buffering_setup: grow the core over the next axis while (1 + n_buffered) / size
            //      improves. An operand whose strides stop collapsing becomes buffered (a cost) unless it already is.
            int cost = 1;
            for (int i = 0; i < nop; i++)
                if (cast[i])
                    cost++;
            var single = new int[nop];
            for (int i = 0; i < nop; i++)
                single[i] = 1;
            long size = shp[0];
            int bestDim = 0, bestCost = cost;
            long bestSize = size;
            for (int idim = 1; idim < ncoal; idim++)
            {
                // GROWINNER is always set for ufuncs, so only a buffered iteration stops at the buffer size.
                if (size >= bufsize && cost > 1)
                    break;
                long prevShape = shp[idim - 1];
                for (int i = 0; i < nop; i++)
                {
                    if (single[i] != idim)
                        continue;
                    if (ad[(idim - 1) * nop + i] * prevShape == ad[idim * nop + i])
                    {
                        single[i]++;
                        continue;
                    }
                    if (!cast[i])
                        cost++;
                }
                size *= shp[idim];
                if (size == 0)
                    break;
                double bufferedSize = size > bufsize && cost > 1 ? bufsize : size;
                if (cost * (double)bestSize <= bestCost * bufferedSize)
                {
                    bestCost = cost;
                    bestSize = size;
                    bestDim = idim;
                }
            }

            // Inner stride of each bound: an unbuffered operand keeps its axis-0 stride; a buffered one gets the item
            // size, unless it is single-stride over the core and broadcast there (the buffer then repeats one element).
            long InnerStride(int i)
            {
                bool singleStride = single[i] > bestDim;
                bool buffered = cast[i] || !singleStride;
                long axis0 = ad[i];
                if (!buffered)
                    return axis0;
                return singleStride && axis0 == 0 ? 0 : itemSize[i];
            }

            return InnerStride(1) == 0 && InnerStride(2) == 0;
        }

        /// <summary>True when two dimension arrays are element-wise equal.</summary>
        private static bool SameLongs(long[] x, long[] y)
        {
            if (x.Length != y.Length)
                return false;
            for (int i = 0; i < x.Length; i++)
                if (x[i] != y[i])
                    return false;
            return true;
        }

        /// <summary>
        /// One operand of the clip ufunc call as NumPy's machinery sees it: its own dims and byte strides, dtype,
        /// contiguity flags, first-element address, and the array itself (for overlap questions). A fresh operand — a
        /// copy NumPy makes, or a Python scalar it converts — has no array and overlaps nothing.
        /// </summary>
        private readonly struct ClipLoopOperand
        {
            /// <summary>The operand's dims (empty for 0-d).</summary>
            public readonly long[] Dims;
            /// <summary>The operand's strides in BYTES.</summary>
            public readonly long[] ByteStrides;
            /// <summary>The operand's dtype as the ufunc receives it.</summary>
            public readonly NPTypeCode Dtype;
            /// <summary>Bytes per element of <see cref="Dtype"/>.</summary>
            public readonly int ItemSize;
            /// <summary>NumPy's C_CONTIGUOUS flag.</summary>
            public readonly bool CContiguous;
            /// <summary>NumPy's F_CONTIGUOUS flag.</summary>
            public readonly bool FContiguous;
            /// <summary>Byte address of element 0 (<c>PyArray_BYTES</c>); 0 for a fresh operand.</summary>
            public readonly long DataStart;
            /// <summary>The array, or null for a fresh copy / converted scalar.</summary>
            public readonly NDArray Array;

            private ClipLoopOperand(long[] dims, long[] byteStrides, NPTypeCode dtype, bool c, bool f, long start, NDArray array)
            {
                Dims = dims;
                ByteStrides = byteStrides;
                Dtype = dtype;
                ItemSize = dtype.SizeOf();
                CContiguous = c;
                FContiguous = f;
                DataStart = start;
                Array = array;
            }

            /// <summary>Number of dimensions.</summary>
            public int NDim => Dims.Length;

            /// <summary>Element count.</summary>
            public long Size
            {
                get
                {
                    long n = 1;
                    foreach (var d in Dims)
                        n *= d;
                    return n;
                }
            }

            /// <summary>True for a copy or converted scalar, which cannot share memory with anything.</summary>
            public bool IsFresh => Array is null;

            /// <summary>An existing array, read as NumPy reads its descriptor.</summary>
            /// <param name="x">The array.</param>
            /// <returns>The operand view of <paramref name="x"/>.</returns>
            public static unsafe ClipLoopOperand Of(NDArray x)
            {
                int item = x.dtypesize;
                var dims = (long[])x.Shape.dimensions.Clone();
                var strides = new long[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    strides[i] = x.Shape.strides[i] * item;
                long start = (long)((byte*)x.Address + x.Shape.offset * item);
                return new ClipLoopOperand(dims, strides, x.typecode, x.Shape.IsContiguous, x.Shape.IsFContiguous, start, x);
            }

            /// <summary>
            /// A bound as NumPy receives it: a 0-d bound is a Python scalar, converted into a fresh 0-d array of the
            /// loop dtype; any other bound is the array itself.
            /// </summary>
            /// <param name="bound">The bound.</param>
            /// <param name="loop">The loop dtype.</param>
            /// <returns>The operand view of the bound.</returns>
            public static ClipLoopOperand OfBound(NDArray bound, NPTypeCode loop)
                => bound.ndim == 0 ? Copy(System.Array.Empty<long>(), loop) : Of(bound);

            /// <summary>A fresh C-contiguous array of <paramref name="dtype"/> with <paramref name="dims"/> — a copy NumPy makes.</summary>
            /// <param name="dims">The copy's dims (0-d or 1-D here).</param>
            /// <param name="dtype">The copy's dtype.</param>
            /// <returns>The operand view of the copy.</returns>
            public static ClipLoopOperand Copy(long[] dims, NPTypeCode dtype)
            {
                int item = dtype.SizeOf();
                var strides = new long[dims.Length];
                long s = item;
                for (int i = dims.Length - 1; i >= 0; i--)
                {
                    strides[i] = s;
                    s *= dims[i];
                }
                return new ClipLoopOperand((long[])dims.Clone(), strides, dtype, true, true, 0, null);
            }

            /// <summary>
            /// <c>solve_may_share_memory(a, b, max_work=1) != 0</c>, NumPy's COPY_IF_OVERLAP test; a fresh operand never
            /// shares memory.
            /// </summary>
            /// <param name="other">The other operand.</param>
            /// <returns>True when the two may share memory.</returns>
            public bool MayShareMemory(in ClipLoopOperand other)
            {
                if (Array is null || other.Array is null)
                    return false;
                return NDMemOverlap.SolveMayShareMemory(Array, other.Array, 1) != MemOverlap.No;
            }

            /// <summary>
            /// The case COPY_IF_OVERLAP lets through under NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE: the same first byte, dims,
            /// strides and dtype, with no internal overlap (a broadcast view repeats elements, so it never qualifies).
            /// </summary>
            /// <param name="other">The written operand.</param>
            /// <returns>True when the two are walked element for element in lockstep.</returns>
            public bool IsExactElementwiseAlias(in ClipLoopOperand other)
            {
                if (Array is null || other.Array is null || DataStart != other.DataStart || Dtype != other.Dtype)
                    return false;
                if (!SameLongs(Dims, other.Dims) || !SameLongs(ByteStrides, other.ByteStrides))
                    return false;
                return NDMemOverlap.SolveMayHaveInternalOverlap(Array, 1) == MemOverlap.No;
            }
        }
    }
}
