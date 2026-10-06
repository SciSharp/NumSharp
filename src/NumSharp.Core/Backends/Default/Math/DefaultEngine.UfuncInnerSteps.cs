using System;
using NumSharp.Backends.Iteration;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // The steps NumPy hands a ufunc's inner loop
        // =============================================================================
        //
        // A few NumPy inner loops branch on their STEPS (the byte stride of each operand within one
        // inner-loop call), not only on their values, and the two branches produce different bits:
        //   - clip (numpy/_core/src/umath/clip.cpp, _npy_clip): both bound steps 0 → the const
        //     compare-select loop (a tie keeps the element, a NaN bound fills the call with that NaN);
        //     otherwise _NPY_MIN(_NPY_MAX(x, lo), hi). np.clip(-0.0, 0.0, 1.0) keeps -0.0 on one and
        //     returns +0.0 on the other.
        //   - power (numpy/_core/src/umath/loops_umath_fp.dispatch.c.src, DOUBLE_power / FLOAT_power):
        //     exponent step 0 and an exponent of -1, 0, 0.5, 1 or 2 → 1/x, 1, sqrt(x), x, x*x; otherwise
        //     the C library pow. 757.7617475757002 ** 2 is one ULP apart between the two, sqrt(-0.0) is
        //     -0.0 where pow(-0.0, 0.5) is +0.0, and x for a signalling NaN keeps it signalling.
        //
        // The steps are decided far from the loop: by the ufunc's trivial-loop shortcut, or by NpyIter's
        // axis order, coalescing and buffering cost model. An operand that varies along the result can
        // still reach the loop with step 0 (a (2,1) operand against a (2,3) array, when NpyIter does not
        // buffer it), and a 0-d operand copied into a buffer still has step 0. This file ports exactly
        // that decision:
        //   1. check_for_trivial_loop (ufunc_object.c): a 0-d or a short 1-D input needing a cast (or
        //      unaligned) is cast into a contiguous copy; any other such input makes the trivial loop
        //      impossible, and so does an out of another dtype;
        //   2. try_trivial_single_output_loop: same-shape, contiguous operands (a 0-d input broadcasts
        //      with step 0) run ONE call with fixed steps;
        //   3. otherwise NpyIter: npyiter_fill_axisdata, npyiter_find_best_axis_ordering,
        //      npyiter_flip_negative_strides, COPY_IF_OVERLAP, npyiter_coalesce_axes and
        //      npyiter_find_buffering_setup, whose per-operand inner strides are the steps.
        // Every inner-loop call of one ufunc call gets the same steps: the trivial loop is one call, and
        // NpyIter fixes each operand's inner stride when it sets up buffering (an elementwise iteration
        // has no reduce operand), so the answer is ONE step per input.
        // Validated against live NumPy 2.4.2 through np.clip (70,000 random layout / dtype / out= /
        // where= / overlap cases, every one deciding the same loop) and np.power (see
        // NumPyPowerTakesStrideZeroLoop).
        // =============================================================================

        /// <summary>
        /// The byte step each INPUT of a ufunc call has in every inner-loop call NumPy 2.4.2 makes for it, through
        /// the trivial-loop shortcut or NpyIter (see the file header). Use it when a NumPy inner loop branches on
        /// its steps (np.clip's const loop, np.power's scalar-exponent loop) and NumSharp must take the same branch.
        /// </summary>
        /// <param name="inputs">The inputs as NumPy receives them (see <see cref="UfuncLoopOperand.Of"/> and
        /// <see cref="UfuncLoopOperand.OfInput"/>). Not modified: the copies NumPy makes are modelled on a clone.</param>
        /// <param name="out">The out= array, or null when NumPy allocates the result.</param>
        /// <param name="where">The where= mask, or null. A mask disables the trivial loop and joins the iterator as
        /// a read operand.</param>
        /// <param name="loop">The resolved loop dtype (every input is cast to it).</param>
        /// <param name="dims">The broadcast iteration shape (inputs, out and mask together); NumPy's iterator
        /// iterates it. The trivial loop does not read it.</param>
        /// <param name="steps">Receives one byte step per input (length at least <c>inputs.Length</c>).</param>
        /// <remarks>
        /// <para>Every cast the call needs must already be validated: NumPy's trivial-loop check accepts any input
        /// cast its casting rule allows, which the caller's validation established.</para>
        /// <para>Strides are compared in BYTES, as NumPy does; the copies a step creates are modelled as fresh,
        /// contiguous, aligned arrays of the loop dtype that cannot overlap anything.</para>
        /// </remarks>
        internal static void NumPyUfuncInputSteps(UfuncLoopOperand[] inputs, NDArray @out, NDArray where,
                                                  NPTypeCode loop, long[] dims, Span<long> steps)
        {
            long bufsize = np.getbufsize();
            int loopSize = loop.SizeOf();
            int nin = inputs.Length;
            // The copies check_for_trivial_loop makes replace the operand for the rest of the call (NumPy
            // reassigns op[i]), so they persist into the iterator path when the trivial loop is refused later.
            var ops = (UfuncLoopOperand[])inputs.Clone();

            // ---- check_for_trivial_loop: only without a mask (the masked path never tries the trivial loop).
            bool trivial = where is null;
            if (trivial)
            {
                for (int i = 0; i < nin && trivial; i++)
                {
                    ref var o = ref ops[i];
                    if (o.Dtype == loop && o.Aligned)
                        continue;
                    // A cast (already validated) or an unaligned operand: a 0-d or a 1-D input no longer than the
                    // buffer is copied eagerly into a contiguous, aligned array of the loop dtype, which keeps the
                    // trivial loop possible; any other refuses it (NumPy returns at the first such input).
                    if (o.NDim == 0 || (o.NDim == 1 && o.Dims[0] <= bufsize))
                        o = UfuncLoopOperand.Copy(o.Dims, loop);
                    else
                        trivial = false;
                }
                // An out of another dtype (or unaligned) needs a cast on the way out, which the trivial loop
                // cannot do; outputs are never copied there.
                if (trivial && @out is not null && (@out.typecode != loop || !UfuncLoopOperand.IsAligned(@out)))
                    trivial = false;
            }

            if (trivial && TryTrivialLoopSteps(ops, @out, loopSize, steps))
                return;

            NpyIterInputSteps(ops, @out, where, loop, loopSize, dims, bufsize, steps);
        }

        /// <summary>
        /// Port of <c>try_trivial_single_output_loop</c>: one inner-loop call over operands of one shape (0-d inputs
        /// broadcast with step 0), every N-D operand contiguous in one shared order, and no input overlapping the
        /// output in a way the single pass cannot honour.
        /// </summary>
        /// <param name="ops">The inputs after <c>check_for_trivial_loop</c>'s copies.</param>
        /// <param name="out">The out= array, or null (the result is then allocated to fit).</param>
        /// <param name="loopSize">The loop dtype's item size, in bytes.</param>
        /// <param name="steps">When this returns true: the step of each input in the one call.</param>
        /// <returns>True when NumPy takes the trivial loop; false when it falls through to NpyIter.</returns>
        private static bool TryTrivialLoopSteps(UfuncLoopOperand[] ops, NDArray @out, int loopSize, Span<long> steps)
        {
            int nin = ops.Length;
            int operationNdim = 0;
            long[] operationShape = null;
            int operationOrder = 0;   // 0 = unset; else C (1) | F (2), as the flags read

            for (int i = 0; i <= nin; i++)
            {
                UfuncLoopOperand o;
                if (i < nin)
                    o = ops[i];
                else if (@out is null)
                    continue;
                else
                    o = UfuncLoopOperand.Of(@out);

                // 0-d inputs broadcast with a 0 step (a 0-d out has no such special case).
                if (o.NDim == 0 && i < nin)
                {
                    steps[i] = 0;
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

                long step;
                if (o.NDim == 1)
                {
                    step = o.ByteStrides[0];
                }
                else
                {
                    step = o.ItemSize;
                    int order = (o.CContiguous ? 1 : 0) | (o.FContiguous ? 2 : 0);
                    if (order == 0)
                        return false;   // N-D operands must be contiguous
                    if (operationOrder == 0)
                        operationOrder = order;
                    else if (operationOrder != order)
                        return false;
                }
                if (i < nin)
                    steps[i] = step;
            }

            if (@out is not null)
            {
                var outOp = UfuncLoopOperand.Of(@out);
                for (int i = 0; i < nin; i++)
                    if (!EquivalentlyIterableOverlapOk(ops[i], outOp))
                        return false;   // the full path makes the copy that keeps reads ahead of writes
                // A 1-D out walked backwards or onto itself cannot be written in one pass.
                if (operationNdim == 1 && outOp.ByteStrides[0] < outOp.ItemSize && outOp.ByteStrides[0] != 0)
                    return false;
            }

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
        private static bool EquivalentlyIterableOverlapOk(in UfuncLoopOperand input, in UfuncLoopOperand output)
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
        private static long TrivialPairIterationStride(in UfuncLoopOperand o)
            => o.Size == 1 ? 0 : (o.NDim == 1 ? o.ByteStrides[0] : o.ItemSize);

        /// <summary>
        /// The input steps NumPy's NpyIter would hand the loop: broadcast every operand over
        /// <paramref name="dims"/>, sort the axes, flip all-negative axes, apply COPY_IF_OVERLAP to the output,
        /// allocate it, coalesce, then run <c>npyiter_find_buffering_setup</c>'s cost model.
        /// </summary>
        /// <param name="ops">The inputs (after any trivial-loop copies, which persist into this path).</param>
        /// <param name="out">The out= array, or null for an allocated result.</param>
        /// <param name="where">The mask (an extra read operand), or null.</param>
        /// <param name="loop">The loop dtype.</param>
        /// <param name="loopSize">The loop dtype's item size, in bytes.</param>
        /// <param name="dims">The broadcast iteration shape.</param>
        /// <param name="bufsize">The ufunc buffer size (<see cref="np.getbufsize"/>).</param>
        /// <param name="steps">Receives the inner byte step of each input.</param>
        private static void NpyIterInputSteps(UfuncLoopOperand[] ops, NDArray @out, NDArray where,
                                              NPTypeCode loop, int loopSize, long[] dims, long bufsize, Span<long> steps)
        {
            int nd = dims.Length;
            int nin = ops.Length;
            int iout = nin;
            int nop = nin + 1 + (where is null ? 0 : 1);

            var opnd = new UfuncLoopOperand[nop];
            ops.CopyTo(opnd, 0);
            bool outAllocated = @out is null;
            opnd[iout] = outAllocated ? default : UfuncLoopOperand.Of(@out);
            if (where is not null)
                opnd[iout + 1] = UfuncLoopOperand.Of(where);

            // NPY_OP_ITFLAG_CAST: an operand whose dtype is not the loop's, or which is unaligned, is buffered through
            // a cast. A read-only 0-d input never is: npyiter_allocate_arrays copies it into the loop dtype eagerly and
            // clears the flag.
            var cast = new bool[nop];
            var itemSize = new int[nop];
            for (int i = 0; i < nop; i++)
            {
                if (i < nin)
                    cast[i] = (opnd[i].Dtype != loop || !opnd[i].Aligned) && opnd[i].NDim != 0;
                else if (i == iout)
                    cast[i] = !outAllocated && (opnd[i].Dtype != loop || !opnd[i].Aligned);
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
                    if (i < nin && opnd[i].IsExactElementwiseAlias(opnd[iout]))
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

            // Inner stride of each input: an unbuffered operand keeps its axis-0 stride; a buffered one gets the item
            // size, unless it is single-stride over the core and broadcast there (the buffer then repeats one element).
            for (int i = 0; i < nin; i++)
            {
                bool singleStride = single[i] > bestDim;
                bool buffered = cast[i] || !singleStride;
                long axis0 = ad[i];
                steps[i] = !buffered ? axis0 : (singleStride && axis0 == 0 ? 0 : itemSize[i]);
            }
        }

        /// <summary>
        /// True when two <c>long</c> arrays hold the same values in the same order. The step decision compares
        /// dims with it (an operand whose dims differ from the operation's is a broadcast, which NumPy's trivial loop
        /// refuses) and the exact-alias test compares both dims and byte strides; np.clip's routes use it too.
        /// </summary>
        /// <param name="x">The first array (dims or byte strides).</param>
        /// <param name="y">The second array, of the same kind as <paramref name="x"/>.</param>
        /// <returns><c>true</c> when both arrays have the same length and equal elements; <c>false</c> otherwise.</returns>
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
        /// One operand of a ufunc call as NumPy's machinery sees it: its own dims and byte strides, dtype,
        /// contiguity and alignment flags, first-element address, and the array itself (for overlap questions). A
        /// fresh operand — a copy NumPy makes, or a Python scalar it converts — has no array and overlaps nothing.
        /// </summary>
        internal readonly struct UfuncLoopOperand
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
            /// <summary>
            /// NumPy's ALIGNED flag. An unaligned operand is buffered like a cast one (ufuncs request
            /// <c>NPY_ITER_ALIGNED</c>) and refuses the trivial loop unless it can be copied.
            /// </summary>
            public readonly bool Aligned;
            /// <summary>Byte address of element 0 (<c>PyArray_BYTES</c>); 0 for a fresh operand.</summary>
            public readonly long DataStart;
            /// <summary>The array, or null for a fresh copy / converted scalar.</summary>
            public readonly NDArray Array;

            /// <summary>
            /// Stores already-computed parts. Only the factories (<see cref="Of"/>, <see cref="OfInput"/>,
            /// <see cref="Copy"/>) call it, and they derive every part from an array or a copy NumPy makes, so the
            /// constructor checks nothing; the arrays are kept by reference, not copied.
            /// </summary>
            /// <param name="dims">The operand's dims (empty for 0-d).</param>
            /// <param name="byteStrides">The operand's strides in bytes, one per dimension.</param>
            /// <param name="dtype">The operand's dtype as the ufunc receives it.</param>
            /// <param name="c">NumPy's C_CONTIGUOUS flag of the operand.</param>
            /// <param name="f">NumPy's F_CONTIGUOUS flag of the operand.</param>
            /// <param name="aligned">NumPy's ALIGNED flag of the operand.</param>
            /// <param name="start">Byte address of element 0 (<c>PyArray_BYTES</c>), or 0 for a fresh operand.</param>
            /// <param name="array">The array, or <c>null</c> for a fresh copy or converted scalar.</param>
            private UfuncLoopOperand(long[] dims, long[] byteStrides, NPTypeCode dtype, bool c, bool f, bool aligned,
                                     long start, NDArray array)
            {
                Dims = dims;
                ByteStrides = byteStrides;
                Dtype = dtype;
                ItemSize = dtype.SizeOf();
                CContiguous = c;
                FContiguous = f;
                Aligned = aligned;
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

            /// <summary>
            /// NumPy's <c>PyArray_ISALIGNED</c> for a NumSharp array: the ALIGNED shape bit (cleared by
            /// <c>setflags(align=False)</c>) and the memory's real alignment (an array laid over a caller's buffer at a
            /// misaligning byte offset is not aligned).
            /// </summary>
            /// <param name="x">The array.</param>
            /// <returns>True when NumPy would treat <paramref name="x"/> as aligned.</returns>
            public static bool IsAligned(NDArray x)
                => (x.Shape.Flags & ArrayFlags.ALIGNED) != 0 && x.Storage.IsDataAligned;

            /// <summary>An existing array, read as NumPy reads its descriptor.</summary>
            /// <param name="x">The array.</param>
            /// <returns>The operand view of <paramref name="x"/>.</returns>
            public static unsafe UfuncLoopOperand Of(NDArray x)
            {
                int item = x.dtypesize;
                var dims = (long[])x.Shape.dimensions.Clone();
                var strides = new long[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    strides[i] = x.Shape.strides[i] * item;
                long start = (long)((byte*)x.Address + x.Shape.offset * item);
                return new UfuncLoopOperand(dims, strides, x.typecode, x.Shape.IsContiguous, x.Shape.IsFContiguous,
                                            IsAligned(x), start, x);
            }

            /// <summary>
            /// An input as NumPy receives it when NumSharp cannot tell a Python scalar from a 0-d array: a 0-d input is
            /// a Python scalar, converted into a fresh 0-d array of the loop dtype; any other input is the array itself.
            /// </summary>
            /// <param name="input">The input.</param>
            /// <param name="loop">The loop dtype.</param>
            /// <returns>The operand view of the input.</returns>
            /// <remarks>
            /// NumPy indexes a scalar out of an array as a fresh NumPy scalar (<c>x[0]</c>), where NumSharp returns a
            /// 0-d VIEW; reading a 0-d input as fresh keeps such a view from looking like an overlap NumPy never sees.
            /// The step of a 0-d input is 0 on every route either way.
            /// </remarks>
            public static UfuncLoopOperand OfInput(NDArray input, NPTypeCode loop)
                => input.ndim == 0 ? Copy(System.Array.Empty<long>(), loop) : Of(input);

            /// <summary>A fresh C-contiguous array of <paramref name="dtype"/> with <paramref name="dims"/> — a copy NumPy makes.</summary>
            /// <param name="dims">The copy's dims (0-d or 1-D here).</param>
            /// <param name="dtype">The copy's dtype.</param>
            /// <returns>The operand view of the copy.</returns>
            public static UfuncLoopOperand Copy(long[] dims, NPTypeCode dtype)
            {
                int item = dtype.SizeOf();
                var strides = new long[dims.Length];
                long s = item;
                for (int i = dims.Length - 1; i >= 0; i--)
                {
                    strides[i] = s;
                    s *= dims[i];
                }
                return new UfuncLoopOperand((long[])dims.Clone(), strides, dtype, true, true, true, 0, null);
            }

            /// <summary>
            /// <c>solve_may_share_memory(a, b, max_work=1) != 0</c>, NumPy's COPY_IF_OVERLAP test; a fresh operand never
            /// shares memory.
            /// </summary>
            /// <param name="other">The other operand.</param>
            /// <returns>True when the two may share memory.</returns>
            public bool MayShareMemory(in UfuncLoopOperand other)
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
            public bool IsExactElementwiseAlias(in UfuncLoopOperand other)
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
