using System.Reflection.Emit;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // np.clip — the iterator route
        // =============================================================================
        //
        // NumPy runs np.clip as a ufunc call (um.clip, or um.minimum / um.maximum / um.positive
        // after _methods._clip routed it). The ufunc machinery builds ONE NpyIter over
        // (a, min, max, out[, where]): it broadcasts every operand together, casts operands of
        // another dtype through its buffers, copies an input that overlaps the output
        // (COPY_IF_OVERLAP), honours the where= mask (WRITEMASKED + ARRAYMASK) and hands each
        // inner loop to the dtype's clip loop. This route is that structure: an NDIter over the
        // same operands with the same flags, driving the Tier-3B clip kernel whose bodies live in
        // DirectILKernelGenerator.Clip.InnerLoop.cs.
        //
        // It serves every call the flat route declines: broadcast operands (a bound larger than
        // the array, an out larger than every input), strided / reversed / mixed-order layouts,
        // an out= of another dtype, where= masks, and an out that overlaps an input other than
        // as its exact elementwise alias.
        // =============================================================================

        /// <summary>
        /// The iterator flags of every clip INPUT: read-only, and declared element-wise so that an input which IS
        /// the output (the in-place <c>np.clip(x, lo, hi, out=x)</c>) is not copied by COPY_IF_OVERLAP.
        /// </summary>
        private const NDIterPerOpFlags ClipIterInputFlags =
            NDIterPerOpFlags.READONLY | NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;

        /// <summary>
        /// Runs np.clip's routed ufunc through NDIter, NumPy's own structure (see the file header).
        /// </summary>
        /// <param name="lhs">The array being clipped (input 0), any layout, possibly smaller than the result.</param>
        /// <param name="min">The kept lower bound, or null. A 0-d bound is a scalar.</param>
        /// <param name="max">The kept upper bound, or null.</param>
        /// <param name="out">The out= array, or null for a fresh result. Its dtype may differ from
        /// <paramref name="loop"/> (already validated as a same_kind cast).</param>
        /// <param name="where">The bool where= mask, or null. Validated by the caller.</param>
        /// <param name="loop">The loop dtype every input is cast to and the kernel computes in.</param>
        /// <param name="mode">The bounds present, or null for no bound (NumPy's <c>positive</c>: a copy).</param>
        /// <param name="dims">The broadcast iteration dimensions (<see cref="ClipIterationDims"/>).</param>
        /// <returns><paramref name="out"/>, or a fresh array of <paramref name="loop"/> laid out the way NumPy's
        /// iterator allocates it (<see cref="ClipResultLayout"/>). Without <paramref name="out"/>, the
        /// positions <paramref name="where"/> masks off are uninitialized, as in NumPy.</returns>
        /// <remarks>
        /// <para><b>Casting.</b> A 0-d bound of another dtype is cast up front, as NumPy's iterator copies a
        /// read-only 0-d operand that needs a cast eagerly (<c>npyiter_allocate_arrays</c>). Any other operand
        /// of another dtype — the array, an array bound, the out — is cast through the iterator's buffers
        /// (BUFFERED + GROWINNER + DELAY_BUFALLOC, UNSAFE casting since the caller validated every cast):
        /// one pass over the data instead of a converted copy followed by the clip.</para>
        /// <para><b>The float loop.</b> For a two-bound float32/float64 call NumPy's clip loop takes its const
        /// fast path when both bound strides of an inner-loop call are 0, and the two paths differ on signed-zero
        /// ties and NaN bounds. Which path NumPy takes depends on its trivial-loop test and its iterator's
        /// buffering decision, so <see cref="NumPyClipRunsConstLoop"/> replays that decision and the kernel is
        /// compiled for the answer.</para>
        /// <para><b>Order.</b> The iterator walks in K order (memory order where the operands agree), as NumPy's
        /// ufunc does. The traversal order is not observable: every element is computed from its own inputs,
        /// COPY_IF_OVERLAP makes overlapping operands behave as if copied, and masked-off positions are not
        /// written.</para>
        /// </remarks>
        private NDArray ClipViaIterator(NDArray lhs, NDArray min, NDArray max, NDArray @out, NDArray where,
                                        NPTypeCode loop, DirectILKernelGenerator.ClipMode? mode, long[] dims)
        {
            // NumPy's iterator copies a read-only 0-d operand that needs a cast eagerly, so a scalar bound
            // reaches the loop as a 0-d array of the loop dtype. Doing the same here keeps the iterator
            // unbuffered when nothing else needs a cast (the common "array of one dtype, Python bounds" call).
            NDArray lo = min is not null && min.ndim == 0 && min.GetTypeCode != loop ? CastScalarBound(min, loop) : min;
            NDArray hi = max is not null && max.ndim == 0 && max.GetTypeCode != loop ? CastScalarBound(max, loop) : max;

            NDArray target = @out ?? new NDArray(loop, ClipResultLayout(dims, lhs, min, max, where), false);

            // NumPy's const-vs-general float loop, decided from the caller's operands (the decision
            // reads strides, dtypes and overlap of what NumPy would see, not of the casted copies).
            bool constSelect = mode == DirectILKernelGenerator.ClipMode.BothBounds
                               && (loop == NPTypeCode.Single || loop == NPTypeCode.Double)
                               && NumPyClipRunsConstLoop(lhs, min, max, @out, where, loop, dims);

            // ---- Operand list in NumPy's ufunc order: the inputs, the output, then the mask.
            int nIn = 1 + (lo is null ? 0 : 1) + (hi is null ? 0 : 1);
            int nop = nIn + 1 + (where is null ? 0 : 1);
            var ops = new NDArray[nop];
            int k = 0;
            ops[k++] = lhs;
            if (lo is not null) ops[k++] = lo;
            if (hi is not null) ops[k++] = hi;
            ops[k++] = target;
            if (where is not null) ops[k] = where;

            // ---- Casting through the buffers: needed when any data operand is not of the loop dtype.
            bool needsCast = target.GetTypeCode != loop;
            for (int i = 0; i < nIn; i++)
                needsCast |= ops[i].GetTypeCode != loop;

            var globalFlags = NDIterGlobalFlags.EXTERNAL_LOOP | NDIterGlobalFlags.COPY_IF_OVERLAP;
            var casting = NPY_CASTING.NPY_SAFE_CASTING;
            NPTypeCode[] opDtypes = null;
            if (needsCast)
            {
                globalFlags |= NDIterGlobalFlags.BUFFERED | NDIterGlobalFlags.GROWINNER | NDIterGlobalFlags.DELAY_BUFALLOC;
                // Every cast was validated against NumPy's rules before this point (same_kind for dtype= and
                // out=, the Python-int packing check), so the iterator itself runs UNSAFE, as NumPy's ufunc
                // hands its resolved loop to the iterator.
                casting = NPY_CASTING.NPY_UNSAFE_CASTING;
                opDtypes = new NPTypeCode[nop];
                for (int i = 0; i <= nIn; i++)
                    opDtypes[i] = loop;
                if (where is not null)
                    opDtypes[nop - 1] = NPTypeCode.Empty;   // the mask keeps its own (bool) dtype
            }

            var opFlags = ClipIterOpFlags(nIn, where is not null, outPassedIn: @out is not null);

            // ---- The kernel: one body per (mode, const, dtype), compiled once per packed key. The bodies and the
            //      operand-type list are built once per kernel too (ClipIterKernel.For): they are only read on a
            //      kernel-cache miss, but the per-call closure and delegates were a fixed cost of every call.
            var kernel = ClipIterKernel.For(loop, mode, constSelect);

            using (var iter = NDIterRef.MultiNew(nop, ops, globalFlags, NPY_ORDER.NPY_KEEPORDER, casting, opFlags, opDtypes))
            {
                iter.ExecuteElementWise(kernel.Types, kernel.ScalarBody, kernel.VectorBody, kernel.Key);
            }

            return target;
        }

        /// <summary>
        /// A 0-d bound converted to the loop dtype — what NumPy's iterator does eagerly for a read-only 0-d operand that
        /// needs a cast — built directly: a fresh 0-d array of the loop dtype written by the contiguous cast kernel
        /// <see cref="NDArray.astype(DType, bool)"/> itself runs for the pair, so the value is astype's to the bit.
        /// </summary>
        /// <param name="bound">The 0-d bound (any offset: a bound taken as <c>arr[3]</c> reads <c>arr[3]</c>).</param>
        /// <param name="loop">The loop dtype.</param>
        /// <returns>A new 0-d array of <paramref name="loop"/> (a transient of the calling ClipNDArray scope).</returns>
        /// <remarks>astype's general dispatch cost ~0.35 µs per bound, a third of a 1K float32 clip with Python-float
        /// bounds. A pair without the contiguous kernel (bool, char, half, complex or decimal involved) uses astype.</remarks>
        private static unsafe NDArray CastScalarBound(NDArray bound, NPTypeCode loop)
        {
            var cast = DirectILKernelGenerator.TryGetCastKernel(bound.GetTypeCode, loop);
            if (cast is null)
                return bound.astype(loop);
            var converted = new NDArray(loop, Shape.NewScalar(), false);
            cast((byte*)bound.Address + bound.Shape.offset * bound.dtypesize, (void*)converted.Address, 1);
            return converted;
        }

        /// <summary>
        /// Everything <see cref="NDIterRef.ExecuteElementWise(NPTypeCode[], System.Action{ILGenerator}, System.Action{ILGenerator}, in InnerLoopKernelKey)"/>
        /// needs for one clip kernel — the operand types, the scalar and vector body emitters and the packed key — built
        /// once per (loop dtype, mode, const) and shared by every later call.
        /// </summary>
        /// <remarks>
        /// The instances are immutable and the iterator only READS the type list, so one instance may serve concurrent
        /// calls; a race on first use builds two equal instances and keeps either. The table is indexed by
        /// <c>((mode slot × 2 + const) × 32) + (dtype &amp; 31)</c>: the NPTypeCode values in use are 3–16 and 128
        /// (Complex, slot 0 — the slot of <see cref="NPTypeCode.Empty"/>, which is never a loop dtype), so the low five
        /// bits identify the dtype.
        /// </remarks>
        private sealed class ClipIterKernel
        {
            /// <summary>The kernel's operand dtypes: the inputs then the output, all the loop dtype.</summary>
            public readonly NPTypeCode[] Types;
            /// <summary>The per-element body (<see cref="DirectILKernelGenerator.EmitClipInnerScalar"/>).</summary>
            public readonly System.Action<ILGenerator> ScalarBody;
            /// <summary>The per-vector body, or null when the dtype has no vector form.</summary>
            public readonly System.Action<ILGenerator> VectorBody;
            /// <summary>The kernel's packed cache key.</summary>
            public readonly InnerLoopKernelKey Key;

            private static readonly ClipIterKernel[] s_table = new ClipIterKernel[4 * 2 * 32];

            private ClipIterKernel(NPTypeCode loop, DirectILKernelGenerator.ClipMode? mode, bool constSelect)
            {
                int nIn = mode is null ? 1 : mode == DirectILKernelGenerator.ClipMode.BothBounds ? 3 : 2;
                Types = new NPTypeCode[nIn + 1];
                for (int i = 0; i < Types.Length; i++)
                    Types[i] = loop;
                ScalarBody = il => DirectILKernelGenerator.EmitClipInnerScalar(il, loop, mode, constSelect);
                bool vectorizable = mode is null
                    ? DirectILKernelGenerator.CanUseSimd(loop)             // positive: a plain vector copy
                    : DirectILKernelGenerator.CanEmitClipInnerVector(loop);
                VectorBody = vectorizable
                    ? il => DirectILKernelGenerator.EmitClipInnerVector(il, loop, mode, constSelect)
                    : null;
                Key = InnerLoopKernelKey.Clip(mode, constSelect, loop);
            }

            /// <summary>The shared kernel inputs for a clip call.</summary>
            /// <param name="loop">The loop dtype.</param>
            /// <param name="mode">The bounds present, or null for no bound (positive).</param>
            /// <param name="constSelect">NumPy's const float path (see <see cref="NumPyClipRunsConstLoop"/>).</param>
            /// <returns>The cached instance (built on first use).</returns>
            public static ClipIterKernel For(NPTypeCode loop, DirectILKernelGenerator.ClipMode? mode, bool constSelect)
            {
                int modeSlot = mode is null ? 3 : (int)mode.Value;
                int index = ((modeSlot * 2 + (constSelect ? 1 : 0)) * 32) + ((int)loop & 31);
                var kernel = System.Threading.Volatile.Read(ref s_table[index]);
                if (kernel is not null)
                    return kernel;
                // Publish with a full fence (CompareExchange) so another thread never sees the reference before the
                // fields; the first published instance wins and a losing racer's equal copy is dropped.
                kernel = new ClipIterKernel(loop, mode, constSelect);
                return System.Threading.Interlocked.CompareExchange(ref s_table[index], kernel, null) ?? kernel;
            }
        }

        /// <summary>
        /// The per-operand iterator flags of one clip call: <paramref name="nIn"/> read-only inputs, the output, and
        /// the where= mask when present.
        /// </summary>
        /// <param name="nIn">The number of inputs (1 for positive, 2 for minimum / maximum, 3 for clip).</param>
        /// <param name="masked">True when a where= mask rides as the trailing operand.</param>
        /// <param name="outPassedIn">True when the output is the caller's out= array, false for a fresh result.</param>
        /// <returns>The flag array, one entry per iterator operand. It is SHARED between calls (one per combination of
        /// the three arguments, built on first use): NDIterRef.MultiNew only reads its opFlags argument, so callers must
        /// never write into it.</returns>
        /// <remarks>
        /// <para>The output is NO_BROADCAST (NumPy refuses to stretch an output), plus WRITEMASKED + the mask's
        /// READONLY | ARRAYMASK when a mask is present, so the mask's False positions are never written.</para>
        /// <para><b>READWRITE for a masked caller out.</b> NumPy's masked ufunc loop flags a passed-in output READWRITE,
        /// not WRITEONLY (ufunc_object.c: "In the masked version, we consider the output read-write"). It matters when
        /// the output overlaps an input: COPY_IF_OVERLAP then computes into a temporary and writes it back WHOLE, and
        /// only a READ operand has its temporary initialized from the array. With WRITEONLY the masked-off positions
        /// came back as whatever the temporary held (<c>np.clip(x[:-1], 1, 4, out=x[1:], where=m)</c> wrote heap bytes
        /// where NumPy keeps <c>x</c>'s values). A fresh result stays WRITEONLY: its masked-off positions are
        /// uninitialized in NumPy too.</para>
        /// <para>Every data operand is OVERLAP_ASSUME_ELEMENTWISE, NumPy's ufunc flag that lets an input which IS the
        /// output (same data, dims and strides) skip the COPY_IF_OVERLAP copy.</para>
        /// </remarks>
        private static NDIterPerOpFlags[] ClipIterOpFlags(int nIn, bool masked, bool outPassedIn)
        {
            // 3 input counts × masked × out passed in = 12 shared arrays (index 0..11).
            int index = ((nIn - 1) * 2 + (masked ? 1 : 0)) * 2 + (outPassedIn ? 1 : 0);
            var cached = System.Threading.Volatile.Read(ref s_clipIterOpFlags[index]);
            if (cached is not null)
                return cached;
            var built = BuildClipIterOpFlags(nIn, masked, outPassedIn);
            return System.Threading.Interlocked.CompareExchange(ref s_clipIterOpFlags[index], built, null) ?? built;
        }

        /// <summary>The shared flag arrays behind <see cref="ClipIterOpFlags"/>, indexed by its three arguments.</summary>
        private static readonly NDIterPerOpFlags[][] s_clipIterOpFlags = new NDIterPerOpFlags[12][];

        /// <summary>Builds one <see cref="ClipIterOpFlags"/> array (see there for the flags and why).</summary>
        /// <param name="nIn">The number of inputs.</param>
        /// <param name="masked">True when a where= mask rides as the trailing operand.</param>
        /// <param name="outPassedIn">True when the output is the caller's out= array.</param>
        /// <returns>A fresh flag array.</returns>
        private static NDIterPerOpFlags[] BuildClipIterOpFlags(int nIn, bool masked, bool outPassedIn)
        {
            const NDIterPerOpFlags Elw = NDIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP;
            var flags = new NDIterPerOpFlags[nIn + 1 + (masked ? 1 : 0)];
            for (int i = 0; i < nIn; i++)
                flags[i] = ClipIterInputFlags;
            var access = masked && outPassedIn ? NDIterPerOpFlags.READWRITE : NDIterPerOpFlags.WRITEONLY;
            flags[nIn] = access | NDIterPerOpFlags.NO_BROADCAST | Elw | (masked ? NDIterPerOpFlags.WRITEMASKED : 0);
            if (masked)
                flags[nIn + 1] = NDIterPerOpFlags.READONLY | NDIterPerOpFlags.ARRAYMASK;
            return flags;
        }
    }
}
