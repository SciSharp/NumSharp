using System;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // ClipNDArray — np.clip dispatcher
        // =============================================================================
        //
        // A port of numpy/_core/_methods.py::_clip and of the ufunc machinery it routes to:
        //   1. _clip's drop rule: on an integer array, a Python-int bound at or past the
        //      dtype's limit cannot clip any element, so it is dropped (None);
        //   2. routing: no bound left → positive, one → minimum / maximum, two → clip. The
        //      routed name is the one NumPy's cast errors quote;
        //   3. the ufunc's argument checks in NumPy's order: read-only out, the where= mask,
        //      the Python-int packing check (OverflowError, never a wrap), the dtype= input
        //      casts, the out= cast rule (same_kind), then the joint broadcast of every operand
        //      (inputs, out, where) with NumPy's two error texts;
        //   4. execution on one of two routes:
        //      * the FLAT route (TryClipFlat) — every operand covers the whole iteration and is
        //        C-contiguous once its axes are put in the result's memory order: the
        //        whole-array IL kernel DirectILKernelGenerator.Clip (SIMD where the dtype has
        //        hardware min/max), writing a C buffer that is then relabelled to NumPy's
        //        result layout (NDArray.AdoptLayout). NumPy's trivial loop runs these calls;
        //      * the ITERATOR route (ClipViaIterator, Default.ClipNDArray.Iter.cs) — NumPy's own
        //        structure: an NDIter over (a, lo, hi, out[, where]) drives a per-chunk kernel.
        //        It serves broadcasting, every layout, out= casts, where= masks and overlap
        //        between out and an input (COPY_IF_OVERLAP).
        //   The const-vs-general float loop NumPy runs for two bounds (signed-zero ties, NaN
        //   bounds) is decided per call: on the flat route it follows from the bound kinds, on
        //   the iterator route NumPyClipRunsConstLoop (Default.ClipNDArray.ConstLoop.cs) ports
        //   NumPy's own decision.
        // =============================================================================

        /// <summary>
        /// np.clip's engine entry: NumPy's <c>_methods._clip</c> plus the ufunc it routes to. A Python-int
        /// bound that cannot clip anything is dropped. A kept bound that does not fit the loop dtype raises
        /// <see cref="OverflowException"/> instead of wrapping. The array and the bounds broadcast together
        /// (with <paramref name="out"/> and <paramref name="where"/>), as the ufunc's operands do.
        /// </summary>
        /// <param name="lhs">The array to clip. Any layout; it may be smaller than the result (it broadcasts).</param>
        /// <param name="min">The lower bound, or null for none. A 0-d bound is a scalar. A 0-d bound of an
        /// integer dtype (any of the eight C# integer primitives, or char) is NumPy's Python int, because
        /// a C# integer literal arrives as one. An array bound (ndim ≥ 1) broadcasts with the other operands.</param>
        /// <param name="max">The upper bound, or null for none, with the same scalar rules as <paramref name="min"/>.</param>
        /// <param name="dtype">The loop dtype (NumPy's <c>dtype=</c>). It replaces the promoted dtype, and every input
        /// must reach it by a same_kind cast. The drop rule still uses <paramref name="lhs"/>'s own range.</param>
        /// <param name="out">The array to write into, or null for a fresh result. The loop dtype must reach it by a
        /// same_kind cast. It takes part in the broadcast but is never stretched. It is returned as-is.</param>
        /// <param name="where">NumPy's <c>where=</c> mask (bool), or null for everywhere. It broadcasts with the other
        /// operands; only the True positions are written. Without <paramref name="out"/> the other positions of the
        /// fresh result are uninitialized, as in NumPy.</param>
        /// <returns>The clipped array: <paramref name="out"/> when given, otherwise a fresh array of the loop dtype in the
        /// memory layout NumPy's ufunc allocates (the operands' memory order; C order breaks ties).</returns>
        /// <exception cref="ValueError"><paramref name="out"/> is read-only (checked first, as in NumPy); or the operands
        /// do not broadcast together ("operands could not be broadcast together with shapes (3,) (4,) () "); or
        /// <paramref name="out"/> would have to be stretched ("non-broadcastable output operand with shape ... doesn't match
        /// the broadcast shape ...").</exception>
        /// <exception cref="ArgumentException"><paramref name="where"/> is not a bool array (NumPy's 'safe' cast text);
        /// or a <paramref name="dtype"/> the inputs cannot reach, or a loop dtype <paramref name="out"/> cannot hold, under
        /// the same_kind rule (NumPy's UFuncTypeError texts, naming the routed ufunc).</exception>
        /// <exception cref="OverflowException">A kept Python-int bound does not fit an integer loop dtype
        /// ("Python integer 1000 out of bounds for int8"), raised before any element is read, so also for an
        /// empty <paramref name="lhs"/>. See <see cref="NDExprTypeRules.CheckPyIntPacksIntoLoop"/>.</exception>
        /// <exception cref="TypeError">No bound is left and the loop would be bool: NumPy's <c>positive</c> has no bool loop.</exception>
        [NDScoped]
        public override unsafe NDArray ClipNDArray(NDArray lhs, NDArray min, NDArray max, DType dtype = null, NDArray @out = null, NDArray where = null)
        {
            NPTypeCode? typeCode = dtype?.GetTypeCode();

            // ---- @out writeability FIRST: NumPy's out converter refuses a read-only out before any
            //      operand is converted, so it wins even over the Python-int OverflowError below (probed
            //      2.4.2: clip(int8, 1000, 2000, out=read_only) reports "output array is read-only").
            if (@out is not null)
                NumSharpException.ThrowIfNotWriteable(@out.Shape);

            // ---- where= is converted right after out, before the loop is resolved (probed 2.4.2: a non-bool mask
            //      wins over the OverflowError, the dtype= cast errors and the broadcast error; a read-only out wins over it).
            ValidateWhereMask(where);

            // ---- numpy/_core/_methods.py::_clip, block 1. On an integer array a Python-int bound at or
            //      past the dtype's limit cannot clip any element, so NumPy drops it (sets it to None)
            //      before the ufunc runs. Kept, it would have to be packed into the array's dtype, where it
            //      does not fit. That gap is the old silent wrap: clip(uint8, 0, 256) packed 256 as 0 and
            //      returned all zeros, where NumPy returns the array unchanged.
            DropUnclippingPyIntBounds(lhs.GetTypeCode, ref min, ref max);

            // ---- _clip, block 2: route. No bound left → um.positive(a, out=out, **kwargs), a typed copy with
            //      positive's own rules (no bool loop, 'positive' in every cast error).
            if (min is null && max is null)
                return ClipPositive(lhs, typeCode, @out, where);

            // One bound left → um.minimum / um.maximum, both → um.clip. NumPy's cast errors name the routed
            // ufunc, so clip(int8, -1000, 50, dtype=uint8) reports "ufunc 'minimum' input 0".
            string ufunc = min is null ? "minimum" : max is null ? "maximum" : "clip";

            // ---- Loop dtype: an explicit `dtype=` wins; otherwise NEP-50 promotion of lhs with the bounds.
            NPTypeCode outType = typeCode ?? PromoteClipBound(PromoteClipBound(lhs.typecode, min), max);

            // ---- NEP 50: each kept Python-int bound is packed into the loop dtype, which raises rather than
            //      wraps. NumPy does it while resolving descriptors, so it precedes every cast check below
            //      (probed: clip(int8, -1, 5, dtype=uint64) reports the -1, not the int8->uint64 input cast).
            //      Operand order: min, then max.
            CheckPyIntBoundPacks(min, outType);
            CheckPyIntBoundPacks(max, outType);

            // ---- dtype=: every input must reach the requested loop by a same_kind cast (validate_casting).
            if (typeCode.HasValue)
                ValidateClipInputCasts(lhs, min, max, outType, ufunc);

            // ---- @out: the cast rule before the shape — NumPy resolves dtypes, then builds the iterator.
            //      A same_kind-reachable out of another dtype is legal: the loop runs at outType and the
            //      result is cast into out (np.clip(uint8, 5, 100, out=int8) is NumPy-legal).
            if (@out is not null)
                ValidateOutCast(outType, @out.GetTypeCode, ufunc);

            // ---- The iterator's joint broadcast of every operand: the bounds may be larger than lhs, and out
            //      may be larger than all inputs (they broadcast up into it), but out itself never stretches.
            long[] dims = ClipIterationDims(lhs, min, max, @out, where);

            DirectILKernelGenerator.ClipMode mode =
                min is null ? DirectILKernelGenerator.ClipMode.MaxOnly :
                max is null ? DirectILKernelGenerator.ClipMode.MinOnly :
                              DirectILKernelGenerator.ClipMode.BothBounds;

            // ---- Empty iteration — nothing to compute. Every check above still ran, as in NumPy, where
            //      clip(empty int8, 1000, 2000) raises the OverflowError.
            if (DimsSize(dims) == 0)
                return @out ?? new NDArray(outType, ClipResultLayout(dims, lhs, min, max, where), false);

            if (where is null && TryClipFlat(lhs, min, max, @out, outType, mode, dims, out NDArray flat))
                return flat;

            return ClipViaIterator(lhs, min, max, @out, where, outType, mode, dims);
        }

        /// <summary>
        /// The no-bound route: NumPy's <c>um.positive(a, out=out, **kwargs)</c> — a typed copy of <paramref name="lhs"/>,
        /// with positive's own rules (no bool loop; 'positive' in every cast error).
        /// </summary>
        /// <param name="lhs">The array being "clipped".</param>
        /// <param name="typeCode">The <c>dtype=</c> loop, or null to keep <paramref name="lhs"/>'s dtype.</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <returns>The copy (<paramref name="out"/> when given).</returns>
        /// <exception cref="TypeError">The loop would be bool: positive has no bool loop (NumPy's verbatim texts).</exception>
        /// <exception cref="ArgumentException">The loop dtype cannot be cast into <paramref name="out"/> under same_kind.</exception>
        /// <exception cref="ValueError">The operands do not broadcast, or <paramref name="out"/> would be stretched.</exception>
        private NDArray ClipPositive(NDArray lhs, NPTypeCode? typeCode, NDArray @out, NDArray where)
        {
            ValidatePositiveLoop(lhs.GetTypeCode, typeCode);
            NPTypeCode copyType = typeCode ?? lhs.GetTypeCode;
            if (@out is not null)
                ValidateOutCast(copyType, @out.GetTypeCode, "positive");

            long[] dims = ClipIterationDims(lhs, null, null, @out, where);

            // A plain copy of a non-broadcast input: Cast keeps the input's memory order (K), which is the layout
            // positive's iterator allocates for a single non-broadcast operand. A broadcast input is NOT handled here:
            // K order sorts its stride-0 axis innermost, where the ufunc's layout vote ignores it (C order).
            if (where is null && @out is null && !lhs.Shape.IsBroadcasted && SameLongs(lhs.Shape.dimensions, dims))
                return Cast(lhs, copyType, copy: true);

            if (DimsSize(dims) == 0)
                return @out ?? new NDArray(copyType, ClipResultLayout(dims, lhs, null, null, where), false);

            return ClipViaIterator(lhs, null, null, @out, where, copyType, mode: null, dims);
        }

        /// <summary>
        /// The joint broadcast NumPy's iterator performs over every operand of the routed ufunc
        /// (<c>npyiter_fill_axisdata</c>): the array, each kept bound, <paramref name="out"/> and <paramref name="where"/>,
        /// right-aligned. <paramref name="out"/> takes part but must already have the broadcast shape.
        /// </summary>
        /// <param name="lhs">The array being clipped.</param>
        /// <param name="min">The kept lower bound, or null.</param>
        /// <param name="max">The kept upper bound, or null.</param>
        /// <param name="out">The out= array, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <returns>The iteration (= result) dimensions.</returns>
        /// <exception cref="ValueError">The operands do not broadcast: NumPy lists every operand's shape, 0-d ones as
        /// <c>()</c>, in operand order with a trailing space ("operands could not be broadcast together with shapes
        /// (3,) (4,) () (3,) "). Or <paramref name="out"/> is smaller than the broadcast shape ("non-broadcastable output
        /// operand with shape (3,) doesn't match the broadcast shape (2,3)").</exception>
        /// <remarks>
        /// The two texts are NumPy's two branches: <c>broadcast_error</c>, raised while the shape is being
        /// accumulated over ALL operands (so a mismatch among the inputs alone still lists out and where), and
        /// <c>operand_different_than_broadcast</c>, raised afterwards for the NO_BROADCAST output.
        /// </remarks>
        private static long[] ClipIterationDims(NDArray lhs, NDArray min, NDArray max, NDArray @out, NDArray where)
        {
            // The five operands are visited by name rather than through an NDArray[]: this runs on every call, and
            // at 1K elements the array (plus the loop over it) was a measurable share of the call.
            int nd = lhs.ndim;
            if (min is not null && min.ndim > nd) nd = min.ndim;
            if (max is not null && max.ndim > nd) nd = max.ndim;
            if (@out is not null && @out.ndim > nd) nd = @out.ndim;
            if (where is not null && where.ndim > nd) nd = where.ndim;

            var b = new long[nd];
            for (int i = 0; i < nd; i++)
                b[i] = 1;

            bool ok = BroadcastInto(b, lhs) && BroadcastInto(b, min) && BroadcastInto(b, max)
                      && BroadcastInto(b, @out) && BroadcastInto(b, where);

            if (!ok)
            {
                var sb = new System.Text.StringBuilder("operands could not be broadcast together with shapes ");
                foreach (var o in new[] { lhs, min, max, @out, where })
                    if (o is not null)
                        sb.Append(ShapeRepr(o.Shape.dimensions)).Append(' ');
                throw new ValueError(sb.ToString());
            }

            if (@out is not null && !SameLongs(@out.Shape.dimensions, b))
                throw new ValueError(
                    $"non-broadcastable output operand with shape {ShapeRepr(@out.Shape.dimensions)} " +
                    $"doesn't match the broadcast shape {ShapeRepr(b)}");

            return b;
        }

        /// <summary>
        /// Folds one operand into the running right-aligned broadcast <paramref name="b"/> (NumPy's per-axis rule:
        /// an extent-1 axis adopts the operand's extent, an equal extent or an operand extent of 1 keeps it).
        /// </summary>
        /// <param name="b">The broadcast dimensions so far, already as long as the widest operand; updated in place.</param>
        /// <param name="o">The operand, or null (absent: nothing to fold).</param>
        /// <returns>False when the operand conflicts with <paramref name="b"/> (the caller raises NumPy's text).</returns>
        private static bool BroadcastInto(long[] b, NDArray o)
        {
            if (o is null)
                return true;
            var od = o.Shape.dimensions;
            int shift = b.Length - od.Length;
            for (int i = 0; i < od.Length; i++)
            {
                ref long bs = ref b[shift + i];
                if (bs == 1)
                    bs = od[i];
                else if (od[i] != bs && od[i] != 1)
                    return false;
            }
            return true;
        }

        /// <summary>NumPy's tuple repr of a shape in ufunc error texts: <c>()</c>, <c>(4,)</c>, <c>(2,4)</c>.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The repr.</returns>
        private static string ShapeRepr(long[] dims)
        {
            if (dims.Length == 0) return "()";
            if (dims.Length == 1) return $"({dims[0]},)";
            return $"({string.Join(",", dims)})";
        }

        /// <summary>The element count of a shape (0 when any dimension is 0).</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>Their product.</returns>
        private static long DimsSize(long[] dims)
        {
            long n = 1;
            foreach (var d in dims)
                n *= d;
            return n;
        }

        /// <summary>
        /// The layout NumPy's ufunc allocates the result in (<see cref="Shape.NpyIterOutputShape"/>): its iterator sorts
        /// the axes by the INPUT operands' strides (the mask votes too; 0-d operands and broadcast axes abstain), so an
        /// F-ordered or permuted array keeps its memory order and C order wins any disagreement.
        /// </summary>
        /// <param name="dims">The broadcast result dimensions.</param>
        /// <param name="lhs">The array being clipped.</param>
        /// <param name="min">The kept lower bound, or null.</param>
        /// <param name="max">The kept upper bound, or null.</param>
        /// <param name="where">The where= mask, or null.</param>
        /// <returns>A fresh dense shape of <paramref name="dims"/>.</returns>
        private static Shape ClipResultLayout(long[] dims, NDArray lhs, NDArray min, NDArray max, NDArray where)
        {
            int n = 1 + (min is null ? 0 : 1) + (max is null ? 0 : 1) + (where is null ? 0 : 1);
            var votes = new Shape[n];
            int k = 0;
            votes[k++] = lhs.Shape;
            if (min is not null) votes[k++] = min.Shape;
            if (max is not null) votes[k++] = max.Shape;
            if (where is not null) votes[k] = where.Shape;
            return Shape.NpyIterOutputShape((long[])dims.Clone(), votes);
        }

        // =============================================================================
        // The flat route
        // =============================================================================

        /// <summary>
        /// Runs the whole-array IL kernel when every operand can be walked as one C-contiguous run: the array and each
        /// array bound cover the whole iteration, and, with their axes put in the result's memory order, the array, the
        /// array bounds and <paramref name="out"/> are C-contiguous at offset 0. These are the calls NumPy runs through its
        /// trivial loop (one inner-loop call), so the bound kinds also give NumPy's float loop: two scalar bounds take the
        /// const path, any full-size array bound the general one.
        /// </summary>
        /// <param name="lhs">The array being clipped (no broadcast allowed here).</param>
        /// <param name="min">The kept lower bound, or null.</param>
        /// <param name="max">The kept upper bound, or null.</param>
        /// <param name="out">The out= array, or null. It must have the loop dtype and not overlap an input other than
        /// as its exact elementwise alias (the in-place <c>np.clip(x, lo, hi, out=x)</c>).</param>
        /// <param name="outType">The loop dtype.</param>
        /// <param name="mode">Which bounds are present.</param>
        /// <param name="dims">The iteration dimensions.</param>
        /// <param name="result">The result when this returns true: <paramref name="out"/>, or a fresh array relabelled
        /// to NumPy's layout.</param>
        /// <returns>False, having touched nothing, when the call needs the iterator route.</returns>
        /// <remarks>
        /// <para><b>Why the permutation.</b> NumPy allocates the result in its operands' memory order. Transposing every
        /// operand by that order (<see cref="Shape.DenseAxisOrder"/>) makes a dense F-ordered or permuted array C-contiguous,
        /// so the flat kernel runs at memory speed and <see cref="NDArray.AdoptLayout"/> relabels the C buffer into NumPy's
        /// layout without moving an element. This replaced the old F-only relay, a second full copy after the clip.</para>
        /// <para><b>Overlap.</b> The kernel reads each element before writing the same position, so the in-place alias is
        /// safe; any other overlap between out and an input (a shifted or transposed view of the same buffer) goes to the
        /// iterator route, whose COPY_IF_OVERLAP gives NumPy's as-if-copied result.</para>
        /// </remarks>
        private unsafe bool TryClipFlat(NDArray lhs, NDArray min, NDArray max, NDArray @out, NPTypeCode outType,
                                        DirectILKernelGenerator.ClipMode mode, long[] dims, out NDArray result)
        {
            result = null;

            // Every operand must cover the whole iteration: a broadcast operand is not one flat run.
            if (lhs.Shape.IsBroadcasted || !SameLongs(lhs.Shape.dimensions, dims))
                return false;
            bool minArray = min is not null && min.ndim != 0;
            bool maxArray = max is not null && max.ndim != 0;
            if (minArray && (min.Shape.IsBroadcasted || !SameLongs(min.Shape.dimensions, dims)))
                return false;
            if (maxArray && (max.Shape.IsBroadcasted || !SameLongs(max.Shape.dimensions, dims)))
                return false;
            if (@out is not null && @out.GetTypeCode != outType)
                return false;   // the cast into out is the iterator route's buffered write

            // A flat run needs DENSE operands: no axis permutation turns a gapped view (a stride-2 slice) into one C
            // run, and finding that out through the result layout below costs more than the whole 1K-element kernel.
            // IsFlatRun rejects every operand this pre-check rejects, so it changes no routing decision, only its cost.
            if (!IsDenseSpan(lhs.Shape) || (minArray && !IsDenseSpan(min.Shape)) || (maxArray && !IsDenseSpan(max.Shape))
                || (@out is not null && !IsDenseSpan(@out.Shape)))
                return false;

            // The result's memory order: NumPy's allocation layout, or out's own.
            Shape layout = @out is not null ? @out.Shape : ClipResultLayout(dims, lhs, min, max, null);
            bool identity = layout.IsContiguous;
            int[] perm = identity ? null : layout.DenseAxisOrder();

            NDArray xv = identity ? lhs : np.transpose(lhs, perm);
            NDArray minv = minArray ? (identity ? min : np.transpose(min, perm)) : min;
            NDArray maxv = maxArray ? (identity ? max : np.transpose(max, perm)) : max;
            NDArray outv = @out is null ? null : (identity ? @out : np.transpose(@out, perm));

            if (!IsFlatRun(xv) || (minArray && !IsFlatRun(minv)) || (maxArray && !IsFlatRun(maxv))
                || (outv is not null && !IsFlatRun(outv)))
                return false;

            // A shifted or transposed overlap between out and an input needs NumPy's copy; the exact alias does not.
            if (@out is not null && (HarmfulOverlap(@out, lhs) || (minArray && HarmfulOverlap(@out, min))
                                     || (maxArray && HarmfulOverlap(@out, max))))
                return false;

            var kind = minArray || maxArray ? DirectILKernelGenerator.ClipBoundsKind.Array : DirectILKernelGenerator.ClipBoundsKind.Scalar;

            // ---- A ONE-element call with an array bound is the one place where the bound kinds do not give NumPy's
            //      float loop. Its iterator gives every extent-1 axis stride 0 (npyiter_fill_axisdata), so when NumPy
            //      leaves the trivial loop (a 2-D operand needing a cast, an in-place out: a one-element input is
            //      never "ahead" of its output) both bound steps are 0 and the CONST loop runs, where the array kind
            //      here would run the general one (np.clip(-0.0 in a (1,1) float32 array, (1,1) zeros, 1.0) keeps
            //      -0.0 in NumPy). The iterator route asks NumPyClipRunsConstLoop instead.
            if (mode == DirectILKernelGenerator.ClipMode.BothBounds && kind == DirectILKernelGenerator.ClipBoundsKind.Array
                && (outType == NPTypeCode.Single || outType == NPTypeCode.Double) && DimsSize(dims) == 1)
                return false;

            long[] runDims = (long[])xv.Shape.dimensions.Clone();

            // ---- Operands at the loop dtype. A cast of a C-contiguous view is a C-contiguous copy (K order).
            NDArray srcArr = xv.GetTypeCode == outType ? xv : Cast(xv, outType, copy: true);

            if (kind == DirectILKernelGenerator.ClipBoundsKind.Scalar)
            {
                // ---- Scalar bounds: ONE value of the loop dtype each, read in place or converted into a stack slot
                //      (ScalarBoundPointer). A 0-d astype per bound allocated an array, its storage and a cast
                //      dispatch on every call: ~0.7 us, most of a 1K float32 clip with Python-float bounds.
                byte* loSlot = stackalloc byte[16];
                byte* hiSlot = stackalloc byte[16];
                void* loPtr = minv is null ? null : ScalarBoundPointer(minv, outType, loSlot);
                void* hiPtr = maxv is null ? null : ScalarBoundPointer(maxv, outType, hiSlot);

                NDArray dstS = outv ?? new NDArray(outType, new Shape(runDims), false);

                // ---- NumPy's const path with a NaN scalar bound fills the whole call with that bound's own NaN
                //      (clip.cpp: npy_isnan(min_val) ? min_val : max_val), payload and sign included. The flat
                //      kernel's const select would let the elements through, so the fill is done here. Only the
                //      two-bound float32/float64 call takes it: a one-sided clip is minimum/maximum, whose NaN
                //      propagation the kernel does per element (an element's own NaN keeps its bits there). The
                //      NaN test reads the CONVERTED value (a float32 or float64 NaN stays NaN through either cast).
                bool loNaN = loPtr != null && ValueIsNaN(loPtr, outType);
                bool hiNaN = hiPtr != null && ValueIsNaN(hiPtr, outType);
                if (mode == DirectILKernelGenerator.ClipMode.BothBounds
                    && (outType == NPTypeCode.Single || outType == NPTypeCode.Double)
                    && (loNaN || hiNaN))
                    FillWithNaNBound(dstS, loNaN ? loPtr : hiPtr, outType);
                else
                    DirectILKernelGenerator.Clip(outType, mode, kind, (void*)srcArr.Address, (void*)dstS.Address, dstS.size, loPtr, hiPtr);

                if (@out is not null)
                    result = @out;
                else
                    result = identity ? dstS : NDArray.AdoptLayout(dstS, layout);
                return true;
            }

            // ---- Array bounds: C-contiguous loop-dtype arrays of the run's dims, read per element.
            NDArray loCast = PrepareFlatBound(minv, kind, outType, runDims);
            NDArray hiCast = PrepareFlatBound(maxv, kind, outType, runDims);

            NDArray dst = outv ?? new NDArray(outType, new Shape(runDims), false);
            RunClipKernel(srcArr, dst, loCast, hiCast, outType, mode, kind);

            if (@out is not null)
                result = @out;
            else
                result = identity ? dst : NDArray.AdoptLayout(dst, layout);
            return true;
        }

        /// <summary>
        /// A necessary condition for an array to be dense (to occupy exactly <c>size</c> consecutive elements in some axis
        /// order): the distance from its lowest to its highest element, <c>Σ (dim − 1)·|stride|</c>, is <c>size − 1</c>.
        /// A gapped view (a stride-2 slice, a column of a wider matrix) spans more; only a dense layout spans exactly that.
        /// </summary>
        /// <param name="s">The shape (no stride-0 broadcast axis of extent &gt; 1 — the caller excluded those).</param>
        /// <returns>False when the array cannot be dense; true when it may be (the exact test is <see cref="IsFlatRun"/>
        /// after the layout permutation).</returns>
        private static bool IsDenseSpan(in Shape s)
        {
            var dims = s.dimensions;
            var strides = s.strides;
            long span = 0;
            for (int i = 0; i < dims.Length; i++)
                if (dims[i] > 1)
                    span += (dims[i] - 1) * Math.Abs(strides[i]);
            return span == s.size - 1;
        }

        /// <summary>True when an array is one C-ordered run starting at its buffer's first element.</summary>
        /// <param name="x">The array (a view after the layout permutation).</param>
        /// <returns>True for a C-contiguous, offset-0 array.</returns>
        private static bool IsFlatRun(NDArray x) => x.Shape.IsContiguous && x.Shape.offset == 0;

        /// <summary>
        /// True when writing <paramref name="out"/> while reading <paramref name="input"/> in one ascending pass could read an
        /// already-written element: they may share memory and are not the exact same elements in the same order.
        /// </summary>
        /// <param name="out">The output array.</param>
        /// <param name="input">An input array.</param>
        /// <returns>True when the flat route must not run.</returns>
        private static unsafe bool HarmfulOverlap(NDArray @out, NDArray input)
        {
            if (NDMemOverlap.SolveMayShareMemory(@out, input, 0) == MemOverlap.No)
                return false;
            long outStart = (long)((byte*)@out.Address + @out.Shape.offset * @out.dtypesize);
            long inStart = (long)((byte*)input.Address + input.Shape.offset * input.dtypesize);
            return outStart != inStart || input.GetTypeCode != @out.GetTypeCode
                   || !SameLongs(input.Shape.dimensions, @out.Shape.dimensions)
                   || !SameLongs(input.Shape.strides, @out.Shape.strides);
        }

        /// <summary>
        /// A bound ready for the flat kernel: for scalar bounds, one value of the loop dtype at buffer offset 0; for array
        /// bounds, a C-contiguous offset-0 array of <paramref name="runDims"/> in the loop dtype. A 0-d bound next to an array
        /// bound is materialized to the full size, since the kernel's array kind reads both bounds per element.
        /// </summary>
        /// <param name="bound">The (permuted) bound, or null.</param>
        /// <param name="kind">The kernel's bound kind.</param>
        /// <param name="outType">The loop dtype.</param>
        /// <param name="runDims">The permuted iteration dimensions.</param>
        /// <returns>The prepared bound, or null.</returns>
        private static NDArray PrepareFlatBound(NDArray bound, DirectILKernelGenerator.ClipBoundsKind kind, NPTypeCode outType, long[] runDims)
        {
            if (bound is null)
                return null;
            if (kind == DirectILKernelGenerator.ClipBoundsKind.Scalar)
                // ScalarBoundReady avoids the astype CLONE when the bound is already a contiguous offset-0 scalar of
                // outType (the dominant np.clip(arr, lo, hi) case where lo/hi arrive as same-dtype scalars) —
                // RunClipKernel reads the value through .Address, which is only correct at offset 0.
                return ScalarBoundReady(bound, outType) ? bound : bound.astype(outType);
            if (bound.ndim == 0)
                return np.full(new Shape(runDims), bound.astype(outType).GetAtIndex(0), outType);
            return PrepareBound(bound, new Shape(runDims), outType);
        }

        /// <summary>
        /// Fills a C-contiguous result with a NaN bound's own bits, as NumPy's const float loop does for a call whose bound
        /// is NaN.
        /// </summary>
        /// <param name="dst">The C-contiguous, offset-0 destination.</param>
        /// <param name="nanValue">The NaN bound's value, already in <paramref name="outType"/> (a slot or the bound's element).</param>
        /// <param name="outType">float32 or float64.</param>
        private static unsafe void FillWithNaNBound(NDArray dst, void* nanValue, NPTypeCode outType)
        {
            long n = dst.size;
            if (outType == NPTypeCode.Double)
            {
                double v = *(double*)nanValue;
                var p = (double*)dst.Address;
                for (long i = 0; i < n; i += int.MaxValue)
                    new Span<double>(p + i, (int)Math.Min(int.MaxValue, n - i)).Fill(v);
            }
            else
            {
                float v = *(float*)nanValue;
                var p = (float*)dst.Address;
                for (long i = 0; i < n; i += int.MaxValue)
                    new Span<float>(p + i, (int)Math.Min(int.MaxValue, n - i)).Fill(v);
            }
        }

        /// <summary>
        /// The address of a scalar bound's one value in the loop dtype, without allocating in the common cases: the bound's
        /// own element (read through its offset) when it already has the loop dtype, else the value converted into
        /// <paramref name="slot"/> by the contiguous cast kernel <see cref="NDArray.astype(DType, bool)"/> itself runs for
        /// the pair. A pair without that kernel (bool, char, half, complex or decimal involved) falls back to a 0-d
        /// <c>astype</c>, which the enclosing <c>[NDScoped]</c> ClipNDArray call keeps alive until it returns.
        /// </summary>
        /// <param name="bound">The 0-d bound (after the layout permutation, which leaves a 0-d array unchanged).</param>
        /// <param name="outType">The loop dtype.</param>
        /// <param name="slot">A 16-byte (the widest element) stack slot owned by the caller's frame.</param>
        /// <returns>A pointer to the value; valid for the rest of the caller's frame.</returns>
        private static unsafe void* ScalarBoundPointer(NDArray bound, NPTypeCode outType, byte* slot)
        {
            byte* element = (byte*)bound.Address + bound.Shape.offset * bound.dtypesize;
            if (bound.GetTypeCode == outType)
                return element;
            var cast = DirectILKernelGenerator.TryGetCastKernel(bound.GetTypeCode, outType);
            if (cast is not null)
            {
                cast(element, slot, 1);
                return slot;
            }
            var converted = bound.astype(outType);
            return (byte*)converted.Address + converted.Shape.offset * converted.dtypesize;
        }

        /// <summary>Whether a value of a floating loop dtype is NaN (false for every other dtype).</summary>
        /// <param name="value">The value's address.</param>
        /// <param name="dtype">Its dtype.</param>
        /// <returns>True for a float32 / float64 / float16 NaN.</returns>
        private static unsafe bool ValueIsNaN(void* value, NPTypeCode dtype) => dtype switch
        {
            NPTypeCode.Double => double.IsNaN(*(double*)value),
            NPTypeCode.Single => float.IsNaN(*(float*)value),
            NPTypeCode.Half => Half.IsNaN(*(Half*)value),
            _ => false,
        };

        // Extract pointers and invoke the flat IL clip kernel. src/dst (and array bounds) must
        // be C-contiguous offset-0; scalar bounds are single values.
        private static unsafe void RunClipKernel(NDArray src, NDArray dst, NDArray loCast, NDArray hiCast,
            NPTypeCode outType, DirectILKernelGenerator.ClipMode mode, DirectILKernelGenerator.ClipBoundsKind kind)
        {
            void* srcPtr = (void*)src.Address;
            void* dstPtr = (void*)dst.Address;
            void* loPtr  = loCast is null ? null : (void*)loCast.Address;
            void* hiPtr  = hiCast is null ? null : (void*)hiCast.Address;
            DirectILKernelGenerator.Clip(outType, mode, kind, srcPtr, dstPtr, dst.size, loPtr, hiPtr);
        }

        // ---- Policy helpers ---------------------------------------------------

        /// <summary>
        /// NEP-50 promotion of the running loop dtype with one bound. A 0-d bound is a scalar and is
        /// promoted through the array-scalar table every binary ufunc uses (<see cref="np._FindCommonArrayScalarType(NPTypeCode, NPTypeCode)"/>),
        /// so np.clip agrees with np.minimum / np.maximum, the ufuncs NumPy's <c>_clip</c> routes one-sided
        /// clips to. A C# integer literal therefore adopts the array's dtype the way a Python int does
        /// (clip(float32, 1, 3) stays float32), and a higher-kind scalar promotes (a float bound on an int array → float64).
        /// An array bound promotes strongly via <see cref="np.result_type(NPTypeCode, NPTypeCode)"/>.
        /// </summary>
        /// <param name="outType">The dtype promoted so far (the array's, then after the first bound).</param>
        /// <param name="bound">The bound to fold in, or null for none.</param>
        /// <returns>The promoted dtype; <paramref name="outType"/> itself when <paramref name="bound"/> is null.</returns>
        /// <remarks>
        /// The scalar branch used the kind-index rule of <c>np._FindCommonType(NPTypeCode[], NPTypeCode[])</c>.
        /// That rule disagreed with the table on one family: a Char array (kind 'S' in the legacy kind list, ranked
        /// above every number) kept the char dtype for a float or complex scalar, and astype then TRUNCATED the
        /// bound. clip(char, 0.5, 100.5) clipped to [0, 100] as char, where NumPy's uint16 proxy computes in float64.
        /// The two rules agree on every other dtype pair.
        /// </remarks>
        private static NPTypeCode PromoteClipBound(NPTypeCode outType, NDArray bound)
        {
            if (bound is null) return outType;
            if (bound.ndim == 0)
                return np._FindCommonArrayScalarType(outType, bound.typecode);
            return np.result_type(outType, bound.typecode);
        }

        /// <summary>
        /// The first block of NumPy's <c>numpy/_core/_methods.py::_clip</c>:
        /// <code>
        /// if a.dtype.kind in "iu":
        ///     if type(min) is int and min &lt;= np.iinfo(a.dtype).min: min = None
        ///     if type(max) is int and max &gt;= np.iinfo(a.dtype).max: max = None
        /// </code>
        /// A Python-int bound at or past the input dtype's limit cannot change any element, so it is dropped
        /// before the ufunc would have to pack it into the dtype. A bound exactly AT the limit is dropped too,
        /// which is harmless, since clipping to the limit is the identity.
        /// </summary>
        /// <param name="inputType">The dtype of the array being clipped. Its range decides, not the <c>dtype=</c> loop's.</param>
        /// <param name="min">The lower bound; set to null when it is a Python int ≤ the dtype's minimum.</param>
        /// <param name="max">The upper bound; set to null when it is a Python int ≥ the dtype's maximum.</param>
        /// <remarks>
        /// Only integer arrays take this block (NumPy's 'iu' kinds; Char rides uint16 as its NumPy proxy). On a bool,
        /// float, complex or decimal array the bounds go to the ufunc unchanged. Only Python ints are dropped
        /// (<see cref="TryGetPyInt"/>): a bool bound (<c>type(True) is int</c> is false in Python) or a float bound
        /// is kept, and an array bound always is.
        /// </remarks>
        private static void DropUnclippingPyIntBounds(NPTypeCode inputType, ref NDArray min, ref NDArray max)
        {
            if (!NDExprTypeRules.IsIntegerKind(inputType))
                return;

            (long lo, ulong hi) = NDExprTypeRules.IntBounds(inputType);
            if (TryGetPyInt(min, out Int128 vmin) && vmin <= lo)
                min = null;
            if (TryGetPyInt(max, out Int128 vmax) && vmax >= hi)
                max = null;
        }

        /// <summary>
        /// Reads a bound as NumPy's Python int, when it is one. np.clip takes NDArray bounds, so a C# integer literal
        /// (<c>np.clip(a, -1, 100)</c>) arrives as a 0-d array of the literal's dtype. Every 0-d bound of an integer
        /// dtype (the eight C# integer primitives, and char) stands for a Python int. That matches the bound's
        /// promotion: the array-scalar table makes such a bound adopt the array's dtype, the way a weak Python int
        /// does, so its value must be checked the way a Python int's is.
        /// </summary>
        /// <param name="bound">The bound to read, or null.</param>
        /// <param name="value">The bound's value when this returns true. 128 bits hold every int64 and uint64 value.</param>
        /// <returns>True for a 0-d integer-dtype bound; false for null, an array bound, or a bool/float/complex/decimal scalar.</returns>
        /// <exception cref="InvalidOperationException">Never in practice: raised only if an integer kind the guard admits had
        /// no read below (a new integer dtype added to <see cref="NDExprTypeRules.IsIntegerKind"/> without one here).</exception>
        /// <remarks>
        /// A strong 0-d NDArray (NumPy's np.int64(5) or np.array(5)) cannot be told apart from a literal at this boundary,
        /// so it is treated the same. NumPy would promote with it instead: clip(int8, np.int64(-1000), np.int64(1000))
        /// is int64 there and an unchanged int8 here. The values agree, only the dtype differs, and the old behaviour
        /// returned wrapped garbage for the same call.
        /// </remarks>
        private static bool TryGetPyInt(NDArray bound, out Int128 value)
        {
            value = default;
            if (bound is null || bound.ndim != 0 || !NDExprTypeRules.IsIntegerKind(bound.GetTypeCode))
                return false;

            // The element through the view's offset (a bound taken as arr[3] reads arr[3]), read UNBOXED: the boxing
            // GetAtIndex this replaced ran up to four times per call (drop rule + packing check, both bounds) and was
            // most of a 1K integer clip's managed garbage. Widening to Int128 is exact for every integer dtype.
            unsafe
            {
                byte* p = (byte*)bound.Address + bound.Shape.offset * bound.dtypesize;
                value = bound.GetTypeCode switch
                {
                    NPTypeCode.SByte => *(sbyte*)p,
                    NPTypeCode.Byte => *p,
                    NPTypeCode.Int16 => *(short*)p,
                    NPTypeCode.UInt16 => *(ushort*)p,
                    NPTypeCode.Char => *(char*)p,
                    NPTypeCode.Int32 => *(int*)p,
                    NPTypeCode.UInt32 => *(uint*)p,
                    NPTypeCode.Int64 => *(long*)p,
                    NPTypeCode.UInt64 => *(ulong*)p,   // past long.MaxValue: only uint64 holds it
                    _ => throw new InvalidOperationException($"TryGetPyInt: {bound.GetTypeCode} is not an integer kind."),
                };
            }
            return true;
        }

        /// <summary>
        /// NEP 50's packing of a kept Python-int bound into the loop dtype: NumPy raises OverflowError where the old
        /// <c>astype</c> wrapped (1000 into int8 became -24). Scalars of any other kind, and array bounds, are cast
        /// as before.
        /// </summary>
        /// <param name="bound">A bound that survived <see cref="DropUnclippingPyIntBounds"/>, or null.</param>
        /// <param name="loop">The resolved loop dtype (<c>dtype=</c> or the promoted dtype).</param>
        /// <exception cref="OverflowException">The Python int does not fit an integer <paramref name="loop"/> — see
        /// <see cref="NDExprTypeRules.CheckPyIntPacksIntoLoop"/> for the texts.</exception>
        private static void CheckPyIntBoundPacks(NDArray bound, NPTypeCode loop)
        {
            if (TryGetPyInt(bound, out Int128 value))
                NDExprTypeRules.CheckPyIntPacksIntoLoop(value, loop);
        }

        /// <summary>
        /// NumPy's <c>validate_casting</c> for an explicit <c>dtype=</c> loop: each input must reach the loop by a
        /// same_kind cast, checked in operand order, and the first failure is reported. In the routed ufunc's numbering
        /// <paramref name="lhs"/> is input 0 and each remaining bound is the next input (minimum/maximum have only
        /// input 1; clip has 1 and 2).
        /// </summary>
        /// <param name="lhs">The array being clipped (input 0), always a strong operand.</param>
        /// <param name="min">The kept lower bound, or null.</param>
        /// <param name="max">The kept upper bound, or null.</param>
        /// <param name="loop">The requested loop dtype.</param>
        /// <param name="ufunc">The routed ufunc's name, as NumPy's message quotes it.</param>
        /// <exception cref="ArgumentException">An input cannot reach <paramref name="loop"/>:
        /// "Cannot cast ufunc 'clip' input 1 from dtype('float64') to dtype('int8') with casting rule 'same_kind'".</exception>
        /// <remarks>
        /// A 0-d bound is a weak Python scalar and is checked as NEP 50's abstract DType, named by its default
        /// descriptor. A Python int reaches every integer and inexact loop but not bool (named int64). A Python
        /// float reaches only inexact loops (named float64). A Python complex reaches only complex (named complex128).
        /// So clip(uint8, 5, 10, dtype=uint8) is legal even though int64→uint8 is not same_kind, while
        /// clip(bool, 0, 2, dtype=bool) reports "input 1 from dtype('int64')". A 0-d bool, float16, decimal, and every
        /// array bound are strong and use the plain same_kind table.
        /// </remarks>
        private static void ValidateClipInputCasts(NDArray lhs, NDArray min, NDArray max, NPTypeCode loop, string ufunc)
        {
            ValidateClipInputCast(lhs.GetTypeCode, strongOnly: true, isScalar: false, 0, loop, ufunc);
            int next = 1;
            if (min is not null)
                ValidateClipInputCast(min.GetTypeCode, strongOnly: false, min.ndim == 0, next++, loop, ufunc);
            if (max is not null)
                ValidateClipInputCast(max.GetTypeCode, strongOnly: false, max.ndim == 0, next, loop, ufunc);
        }

        /// <summary>One operand of <see cref="ValidateClipInputCasts"/>; see there for the rules.</summary>
        /// <param name="from">The operand's dtype.</param>
        /// <param name="strongOnly">True for the clipped array itself, which is never a weak scalar.</param>
        /// <param name="isScalar">True when the operand is a 0-d bound (a weak Python scalar for int/float/complex dtypes).</param>
        /// <param name="index">The operand's input index in the routed ufunc.</param>
        /// <param name="loop">The requested loop dtype.</param>
        /// <param name="ufunc">The routed ufunc's name.</param>
        /// <exception cref="ArgumentException">The operand cannot reach <paramref name="loop"/> (NumPy's UFuncTypeError text).</exception>
        private static void ValidateClipInputCast(NPTypeCode from, bool strongOnly, bool isScalar, int index, NPTypeCode loop, string ufunc)
        {
            bool ok;
            string fromName;
            bool loopInexact = NDExprTypeRules.IsFloatKind(loop) || loop == NPTypeCode.Complex || loop == NPTypeCode.Decimal;
            if (!strongOnly && isScalar && NDExprTypeRules.IsIntegerKind(from))
            {
                ok = loop != NPTypeCode.Boolean;                 // a Python int never adapts to bool
                fromName = "int64";
            }
            else if (!strongOnly && isScalar && (from == NPTypeCode.Single || from == NPTypeCode.Double))
            {
                ok = loopInexact;                                // a Python float adapts to inexact loops only
                fromName = "float64";
            }
            else if (!strongOnly && isScalar && from == NPTypeCode.Complex)
            {
                ok = loop == NPTypeCode.Complex;                 // a Python complex adapts to complex only
                fromName = "complex128";
            }
            else
            {
                ok = from == loop || NDIterCasting.CanCast(from, loop, NPY_CASTING.NPY_SAME_KIND_CASTING);
                fromName = from.AsNumpyDtypeName();
            }

            if (!ok)
                throw new ArgumentException(
                    $"Cannot cast ufunc '{ufunc}' input {index} from dtype('{fromName}') to " +
                    $"dtype('{loop.AsNumpyDtypeName()}') with casting rule 'same_kind'");
        }

        private static bool IsSameKind(NPTypeCode a, NPTypeCode b)
        {
            int ga = a.GetGroup();
            int gb = b.GetGroup();
            // 0=Byte/Char, 1=signed int, 2=unsigned int  →  all "integer kind".
            bool aInt = ga >= 0 && ga <= 2;
            bool bInt = gb >= 0 && gb <= 2;
            if (aInt && bInt) return true;
            return ga == gb;
        }

        // Returns `bound` ready for the array-bounds kernel: matches the target shape,
        // dtype outType, contiguous, offset zero. Returns the original instance
        // when those already hold (skips the otherwise-mandatory clone done by
        // `astype` on a same-dtype input).
        private static NDArray PrepareBound(NDArray bound, Shape targetShape, NPTypeCode outType)
        {
            if (bound is null) return null;
            if (bound.GetTypeCode == outType
                && bound.Shape.Equals(targetShape)
                && bound.Shape.IsContiguous
                && bound.Shape.Offset == 0)
            {
                return bound;
            }
            // Broadcast to the target shape and convert dtype. On the flat route the bound already has the
            // target dims and is C-contiguous, so this is a plain dtype conversion (astype keeps C order).
            return np.broadcast_to(bound, targetShape).astype(outType);
        }

        // True when a scalar (0-d) bound is already a value of outType at buffer offset 0,
        // so RunClipKernel can read it straight through .Address with no astype clone. A 0-d
        // VIEW with a non-zero offset (e.g. arr[i] used as a bound) still needs astype to
        // materialize the value at offset 0 — .Address ignores the offset.
        private static bool ScalarBoundReady(NDArray bound, NPTypeCode outType)
            => bound.GetTypeCode == outType && bound.Shape.offset == 0;

        // Returns true when `bound` is a 0-d scalar NDArray whose value is NaN.
        // Used to short-circuit float NaN-in-bounds → all-NaN result (NumPy).
        private static bool ScalarIsNaN(NDArray bound)
        {
            if (bound is null || bound.ndim != 0) return false;
            switch (bound.GetTypeCode)
            {
                case NPTypeCode.Double: return double.IsNaN(bound.GetDouble(0));
                case NPTypeCode.Single: return float.IsNaN(bound.GetSingle(0));
                case NPTypeCode.Half:   return Half.IsNaN(bound.GetHalf(0));
                default: return false;
            }
        }
    }
}
