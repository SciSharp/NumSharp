using System;
using NumSharp.Backends.Strings;

namespace NumSharp.Backends.Iteration
{
    /// <summary>
    ///     The DESCRIPTOR half of the iterator's construction — what <see cref="NDIterRef.AdvancedNew"/> runs when it is given
    ///     NumPy's resolved per-operand <c>op_dtype</c> descriptors (<c>np.nditer</c> always passes them; the internal callers
    ///     never do). It ports the three steps of <c>NpyIter_AdvancedNew</c> that work on descriptors rather than type codes:
    ///     <c>npyiter_check_casting</c> (the read and write-back cast checks and their <see cref="TypeError"/> texts),
    ///     <c>npyiter_allocate_arrays</c>' copy routes (a read-only 0-d operand copied into its iteration dtype, a
    ///     <c>copy</c> / <c>updateifcopy</c> / <c>copy_if_overlap</c> operand copied into a temporary laid out in the
    ///     iterator's order, and the "required copying or buffering" <see cref="TypeError"/>), and
    ///     <c>npyiter_replace_axisdata</c> (the operand's axis data re-pointed at the temporary).
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why the routes are split in two.</b> NumPy checks every cast after the axis data is filled (so a broadcast or
    ///     reduction error wins over a cast error) and makes the copies only after the iteration order is final (each
    ///     temporary is dense in that order, so the coalescing that follows merges it like a fresh array).
    ///     <see cref="PlanDescriptorRoutes"/> therefore decides everything up front without touching memory, the casting
    ///     error it found is raised after the axis data is filled, and <see cref="MaterializeDescriptorRoutes"/> runs between
    ///     the axis reorder and the coalescing — the same seam NumPy's <c>npyiter_allocate_arrays</c> occupies.
    ///     </para>
    ///     <para>
    ///     <b>Write-backs are registered only on success.</b> NumPy discards a <c>WRITEBACKIFCOPY</c> temporary when the
    ///     construction fails (<c>NpyIter_Deallocate</c> under a pending error); here a temporary becomes a write-back only
    ///     after every route of every operand succeeded, and a failure releases the temporaries made so far.
    ///     </para>
    /// </remarks>
    public unsafe ref partial struct NDIterRef
    {
        /// <summary>How an operand reaches its iteration descriptor (<c>npyiter_allocate_arrays</c>' branches).</summary>
        private enum DescriptorRoute : byte
        {
            /// <summary>The operand already IS its iteration descriptor (or the iterator allocated it): no conversion.</summary>
            None = 0,

            /// <summary>A cast through the buffers — legal only under <see cref="NDIterGlobalFlags.BUFFERED"/>.</summary>
            Cast = 1,

            /// <summary>A read-only 0-d operand that needs a cast: copied into a 0-d temporary whatever the flags say.</summary>
            ScalarCopy = 2,

            /// <summary>
            ///     A temporary in the iteration descriptor, laid out in the iterator's order: a cast operand flagged
            ///     <c>copy</c> / <c>updateifcopy</c>, or a <c>copy_if_overlap</c> write operand (cast or not).
            /// </summary>
            TempCopy = 3,
        }

        /// <summary>
        ///     The descriptor-level decisions of one construction, made before the axis data is filled and consumed after it.
        /// </summary>
        private sealed class DescriptorPlan
        {
            /// <summary>
            ///     Per operand, the descriptor the iterator walks: the resolved <c>op_dtype</c>, except for an operand the
            ///     iterator allocated, which walks its allocated array's own (sized) descriptor — an unsized <c>U</c> request
            ///     allocates <c>U1</c>, and the element size must be the array's.
            /// </summary>
            public DType[] IterDescrs;

            /// <summary>Per operand, how it reaches <see cref="IterDescrs"/>.</summary>
            public DescriptorRoute[] Routes;

            /// <summary>The first casting-rule violation (NumPy's operand order, read check before write check), or null.</summary>
            public TypeError CastError;

            /// <summary>
            ///     Per operand, whether the iterator allocated it: such an operand is re-laid out in the settled iteration
            ///     order by <c>MaterializeDescriptorRoutes</c> (NumPy's <c>npyiter_new_temp_array</c> for an allocated output).
            /// </summary>
            public bool[] Allocated;
        }

        /// <summary>
        ///     NumPy's <c>npyiter_check_casting</c> plus the route decisions of <c>npyiter_allocate_arrays</c>, computed from
        ///     descriptors and flags alone (no memory is touched).
        /// </summary>
        /// <param name="nop">The operand count.</param>
        /// <param name="op">The operands, the allocated ones already in place.</param>
        /// <param name="opFlags">The per-operand flags.</param>
        /// <param name="opDescrs">The resolved iteration descriptors (an entry may be null only for an allocated operand).</param>
        /// <param name="allocatedOps">Which operands the iterator allocated (they need no cast and no copy).</param>
        /// <param name="forceCopy">The <c>copy_if_overlap</c> verdicts (<see cref="ComputeCopyIfOverlap"/>), or null.</param>
        /// <param name="casting">The casting rule every conversion must satisfy.</param>
        /// <returns>The plan; its <see cref="DescriptorPlan.CastError"/> is set (and the routes are not) when a cast is refused.</returns>
        /// <remarks>
        ///     NumPy's "no cast needed" test is <c>PyArray_SafeCast(op_descr, op_dtype, &amp;view_offset, NPY_NO_CASTING)
        ///     &amp;&amp; view_offset == 0</c>: equivalent descriptors (<c>&lt;U3</c> and another <c>&lt;U3</c> object) iterate in place,
        ///     while a wider string (<c>U3 → U5</c>), another class (<c>S3 → U3</c>) or another byte order
        ///     (<c>&lt;i8 → &gt;i8</c>) is a cast even when the storage lanes coincide.
        /// </remarks>
        private static DescriptorPlan PlanDescriptorRoutes(int nop, NDArray[] op, NDIterPerOpFlags[] opFlags, DType[] opDescrs,
                                                           bool[] allocatedOps, bool[]? forceCopy, NPY_CASTING casting)
        {
            var plan = new DescriptorPlan
            {
                IterDescrs = new DType[nop], Routes = new DescriptorRoute[nop], Allocated = allocatedOps,
            };

            // Every operand's iteration descriptor first: the axis data is filled for ALL operands before the casting
            // verdict is raised (a broadcast or reduction error of a later operand wins over an earlier cast error, as in
            // NumPy), so the descriptors must exist even when the check below stops early.
            for (int i = 0; i < nop; i++)
                plan.IterDescrs[i] = allocatedOps[i] ? op[i].dtype : opDescrs[i] ?? op[i].dtype;

            // npyiter_check_casting: every operand in order, the read (op -> iteration) check before the write-back
            // (iteration -> op) check; the first violation ends the construction.
            for (int i = 0; i < nop; i++)
            {
                if (allocatedOps[i])
                    continue;

                var src = op[i].dtype;
                var iter = plan.IterDescrs[i];
                if (IsNoCast(src, iter))
                    continue;

                bool read = (opFlags[i] & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE)) != 0;
                bool write = (opFlags[i] & (NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.READWRITE)) != 0;
                if (read && !DTypeCasting.CanCastTypeTo(src, iter, casting))
                {
                    plan.CastError = new TypeError(
                        $"Iterator operand {i} dtype could not be cast from {src.ToString(true)} to {iter.ToString(true)} " +
                        $"according to the rule '{DTypeCasting.CastingToString(casting)}'");
                    return plan;
                }
                if (write && !DTypeCasting.CanCastTypeTo(iter, src, casting))
                {
                    plan.CastError = new TypeError(
                        $"Iterator requested dtype could not be cast from {iter.ToString(true)} to {src.ToString(true)}, " +
                        $"the operand {i} dtype, according to the rule '{DTypeCasting.CastingToString(casting)}'");
                    return plan;
                }
                plan.Routes[i] = DescriptorRoute.Cast;
            }

            // npyiter_allocate_arrays' branches, in its order: a read-only 0-d cast operand is copied whatever the flags
            // say; a cast operand flagged copy / updateifcopy, or a copy_if_overlap write operand, gets a temporary; any
            // other cast is left to the buffers (its "required copying or buffering" error is raised at materialization,
            // interleaved with the copies of the operands before it, as NumPy's loop does).
            for (int i = 0; i < nop; i++)
            {
                if (allocatedOps[i])
                    continue;
                bool cast = plan.Routes[i] == DescriptorRoute.Cast;
                bool read = (opFlags[i] & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE)) != 0;
                bool write = (opFlags[i] & (NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.READWRITE)) != 0;
                if (cast && read && !write && op[i].ndim == 0)
                    plan.Routes[i] = DescriptorRoute.ScalarCopy;
                else if ((cast && (opFlags[i] & (NDIterPerOpFlags.COPY | NDIterPerOpFlags.UPDATEIFCOPY)) != 0) ||
                         (forceCopy != null && forceCopy[i]))
                    plan.Routes[i] = DescriptorRoute.TempCopy;
            }

            return plan;
        }

        /// <summary>
        ///     NumPy's "the types are equivalent" test of <c>npyiter_check_casting</c>: a <c>no</c>-cast whose result is a view
        ///     at offset 0.
        /// </summary>
        /// <param name="src">The operand's descriptor.</param>
        /// <param name="iter">The iteration descriptor.</param>
        /// <returns>True when the operand can be iterated in place.</returns>
        private static bool IsNoCast(DType src, DType iter)
        {
            if (ReferenceEquals(src, iter))
                return true;
            var safety = DTypeCasting.GetCastInfo(src, iter, null, out long viewOffset);
            return safety == NPY_CASTING.NPY_NO_CASTING && viewOffset == 0;
        }

        /// <summary>
        ///     Executes the plan's copy routes once the iteration order is final — the operand loop of NumPy's
        ///     <c>npyiter_allocate_arrays</c>: each temporary is allocated dense in the iterator's order, filled from its
        ///     operand (unsafe casting, NumPy's <c>PyArray_CopyInto</c>) when the operand is read, and swapped into the axis
        ///     data (<see cref="ReplaceAxisData"/>); a cast operand left to the buffers raises when there are none.
        /// </summary>
        /// <param name="plan">The construction's plan (its cast check already passed).</param>
        /// <param name="opFlags">The per-operand flags.</param>
        /// <param name="flags">The global flags (<see cref="NDIterGlobalFlags.BUFFERED"/> decides the cast error).</param>
        /// <param name="opAxes">The per-operand axis maps, or null.</param>
        /// <param name="innermostFirst">
        ///     True when the axes are in the pre-coalesce ASCENDING representation (axis 0 innermost — the contiguous path
        ///     before <see cref="NDIterCoalescing.CoalesceAxes"/>), false for the iteration representation (axis
        ///     <c>NDim - 1</c> innermost).
        /// </param>
        /// <exception cref="TypeError">A cast operand is neither copied nor buffered (NumPy's text).</exception>
        /// <exception cref="UnicodeEncodeError">A copy from <c>U</c> to <c>S</c> meets a code point at or above 128.</exception>
        /// <exception cref="UnicodeDecodeError">A copy from <c>S</c> to <c>U</c> meets a byte at or above <c>0x80</c>.</exception>
        /// <exception cref="StringComputationNotSupportedException">A copy between text and numbers (the computational phase).</exception>
        /// <remarks>
        ///     Runs between the axis reorder and the coalescing, so the temporaries' dense strides take part in the coalescing
        ///     exactly as NumPy's do. On any failure the temporaries made so far are released and no write-back is registered
        ///     — NumPy's <c>PyArray_DiscardWritebackIfCopy</c> on a failed construction.
        /// </remarks>
        private void MaterializeDescriptorRoutes(DescriptorPlan plan, NDIterPerOpFlags[] opFlags, NDIterGlobalFlags flags,
                                                 int[][]? opAxes, bool innermostFirst)
        {
            int nop = _state->NOp;
            bool buffered = (flags & NDIterGlobalFlags.BUFFERED) != 0;
            NDArray[]? temps = null;
            try
            {
                for (int i = 0; i < nop; i++)
                {
                    var route = plan.Routes[i];
                    if (route == DescriptorRoute.Cast)
                    {
                        if (!buffered)
                            throw new TypeError(
                                "Iterator operand required copying or buffering, but neither copying nor buffering was enabled");
                        continue;
                    }
                    if (route != DescriptorRoute.ScalarCopy && route != DescriptorRoute.TempCopy)
                        continue;

                    var original = _operands[i];
                    var descr = plan.IterDescrs[i];
                    int[]? opAxisMap = opAxes != null && i < opAxes.Length ? opAxes[i] : null;
                    bool read = (opFlags[i] & (NDIterPerOpFlags.READONLY | NDIterPerOpFlags.READWRITE)) != 0;

                    // A temporary that is only written starts zeroed (NumPy's is uninitialized; zeros are one valid
                    // reading of that and keep a string temporary decodable); one that is read is overwritten whole.
                    var temp = route == DescriptorRoute.ScalarCopy
                        ? new NDArray(descr, NumSharp.Shape.NewScalar(), fillZeros: !read)
                        : NewTempLikeIterator(original, descr, opAxisMap, innermostFirst, fillZeros: !read);
                    (temps ??= new NDArray[nop])[i] = temp;

                    if (read)
                        np.copyto(temp, original, casting: "unsafe");

                    ReplaceAxisData(i, temp, opAxisMap);
                }
            }
            catch
            {
                if (temps != null)
                    foreach (var t in temps)
                        t?.Dispose();
                throw;
            }

            // NumPy allocates an output after the order is settled, laid out by it (npyiter_new_temp_array): the operand the
            // engine allocated up front (in C order, without a vote in the ordering — see _abstainOps) is replaced by a fresh
            // one dense in the iteration order, so the walk reads it forward and NumPy's it.operands layout is reproduced
            // (an F-ordered walk allocates an F-contiguous output). It was never written; the replacement starts zeroed.
            for (int i = 0; i < nop; i++)
            {
                if (plan.Allocated == null || !plan.Allocated[i])
                    continue;

                var preallocated = _operands[i];
                int[]? opAxisMap = opAxes != null && i < opAxes.Length ? opAxes[i] : null;

                // The common case — a C-order walk of a C-order output — already has the layout: only the strides that
                // abstained from the ordering are put back. A second allocation there cost as much as the first (an
                // np.nditer([a, b, None]) construction over 1,000 doubles took 1.75 µs, the extra allocation ~0.7 µs).
                var strides = TempStridesLikeIterator(preallocated.Shape.dimensions, opAxisMap, innermostFirst);
                if (preallocated.Shape.offset == 0 && StridesEqual(preallocated.Shape.strides, strides))
                {
                    ReplaceAxisData(i, preallocated, opAxisMap);
                    continue;
                }

                var laidOut = NewTempLikeIterator(preallocated, plan.IterDescrs[i], opAxisMap, innermostFirst, fillZeros: true);
                ReplaceAxisData(i, laidOut, opAxisMap);
                _operands[i] = laidOut;
                preallocated.Dispose();
            }

            // The ordering is settled: every operand counts again.
            _abstainOps = null;

            if (temps == null)
                return;

            // Every route succeeded: the temporaries become the operands (NumPy's it.operands shows them). A written one is
            // copied back into its original when the iterator closes (WRITEBACKIFCOPY, unsafe casting); a read-only one is
            // only released then. Either way the slot reverts to the original at teardown.
            for (int i = 0; i < nop; i++)
            {
                var temp = temps[i];
                if (temp is null)
                    continue;

                var original = _operands[i];
                _operands[i] = temp;
                if ((opFlags[i] & (NDIterPerOpFlags.WRITEONLY | NDIterPerOpFlags.READWRITE)) != 0)
                {
                    (_writebackOriginals ??= new NDArray?[nop])[i] = original;
                    _state->SetOpFlags(i, _state->GetOpFlags(i) | NDIterOpFlags.FORCECOPY | NDIterOpFlags.HAS_WRITEBACK);
                }
                else
                {
                    (_releaseOnlyOriginals ??= new NDArray?[nop])[i] = original;
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>npyiter_new_temp_array</c> for a copy of an existing operand: a fresh array of
        ///     <paramref name="descr"/> with the operand's dimensions, dense in the iterator's CURRENT order — the operand axis
        ///     under the innermost iterator axis gets the smallest stride — so iterating it walks memory forward.
        /// </summary>
        /// <param name="original">The operand being copied (its dimensions are kept).</param>
        /// <param name="descr">The temporary's descriptor (the iteration descriptor).</param>
        /// <param name="opAxisMap">The operand's axis map, or null for right-aligned broadcasting.</param>
        /// <param name="innermostFirst">The axis representation (see <see cref="MaterializeDescriptorRoutes"/>).</param>
        /// <param name="fillZeros">True to zero the temporary (it will not be filled from the operand).</param>
        /// <returns>The temporary (owning, writeable).</returns>
        /// <remarks>
        ///     The strides are positive whatever the operand's were: a flipped iterator axis is walked backward by
        ///     <see cref="ReplaceAxisData"/>, as NumPy does (<c>npyiter_undo_iter_axis_perm</c> ignores the flip when it lays
        ///     the temporary out). An operand axis no iterator axis maps to (an <c>op_axes</c> map that skips it) is laid out
        ///     tightest, in C order, and the mapped strides are scaled past it — NumPy's "missing strides" rule.
        /// </remarks>
        private NDArray NewTempLikeIterator(NDArray original, DType descr, int[]? opAxisMap, bool innermostFirst, bool fillZeros)
        {
            var dims = (long[])original.Shape.dimensions.Clone();
            int opNDim = dims.Length;
            var strides = TempStridesLikeIterator(dims, opAxisMap, innermostFirst);

            long size = 1;
            for (int a = 0; a < opNDim; a++)
                size *= dims[a];
            var layout = opNDim == 0 ? NumSharp.Shape.NewScalar() : new NumSharp.Shape(dims, strides, 0, size);
            return new NDArray(descr, layout, fillZeros);
        }

        /// <summary>
        ///     The element strides <see cref="NewTempLikeIterator"/> lays a temporary out with: dense in the iterator's
        ///     CURRENT axis order, the operand axis under the innermost iterator axis tightest — computed without allocating,
        ///     so a caller can tell whether an existing array already has that layout.
        /// </summary>
        /// <param name="dims">The operand's dimensions (not modified).</param>
        /// <param name="opAxisMap">The operand's axis map, or null for right-aligned broadcasting.</param>
        /// <param name="innermostFirst">The axis representation (see <see cref="MaterializeDescriptorRoutes"/>).</param>
        /// <returns>One stride per operand axis, in elements.</returns>
        /// <remarks>An operand axis no iterator axis maps to is laid out tightest, in C order, and the mapped strides are
        ///     scaled past it (NumPy's "missing strides" rule in <c>npyiter_new_temp_array</c>).</remarks>
        private long[] TempStridesLikeIterator(long[] dims, int[]? opAxisMap, bool innermostFirst)
        {
            int opNDim = dims.Length;
            var strides = new long[opNDim];
            var assigned = new bool[opNDim];
            int ndim = _state->NDim;
            long stride = 1;
            int used = 0;

            for (int k = 0; k < ndim; k++)
            {
                int d = innermostFirst ? k : ndim - 1 - k;
                int opAxis = OperandAxisOf(d, opNDim, opAxisMap);
                if (opAxis < 0 || assigned[opAxis])
                    continue;
                assigned[opAxis] = true;
                strides[opAxis] = stride;
                stride *= dims[opAxis];
                used++;
            }

            if (used < opNDim)
            {
                // Axes the map skipped: tighter together than the mapped ones (C order), the mapped strides scaled past them.
                long factor = 1;
                var missing = new long[opNDim];
                for (int a = opNDim - 1; a >= 0; a--)
                {
                    if (!assigned[a])
                    {
                        missing[a] = factor;
                        factor *= dims[a];
                    }
                }
                for (int a = 0; a < opNDim; a++)
                    strides[a] = assigned[a] ? strides[a] * factor : missing[a];
            }

            return strides;
        }

        /// <summary>Whether two stride arrays are the same length and agree on every axis whose extent matters.</summary>
        /// <param name="a">The first strides.</param>
        /// <param name="b">The second strides.</param>
        /// <returns>True when they are element-for-element equal.</returns>
        /// <remarks>Compared exactly, extent-1 axes included: a mismatch there only costs the re-layout allocation the
        ///     comparison exists to avoid, never a wrong layout.</remarks>
        private static bool StridesEqual(long[] a, long[] b)
        {
            if (a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }

        /// <summary>
        ///     NumPy's <c>npyiter_replace_axisdata</c>: re-points operand <paramref name="iop"/>'s axis data at
        ///     <paramref name="temp"/> — its strides along every iterator axis the operand spans (negated on a flipped axis,
        ///     whose start moves to the axis' far end), its base pointer, and its source descriptor (the temporary needs no
        ///     cast).
        /// </summary>
        /// <param name="iop">The operand index.</param>
        /// <param name="temp">The temporary replacing the operand (same dimensions).</param>
        /// <param name="opAxisMap">The operand's axis map, or null for right-aligned broadcasting.</param>
        /// <remarks>
        ///     Only axes where the temporary's extent is not 1 are replaced, exactly as NumPy replaces "just the strides which
        ///     were non-zero": a stretched (broadcast) axis keeps stride 0, and the size-1 ⇒ stride-0 invariant the coalescing
        ///     relies on survives.
        /// </remarks>
        private void ReplaceAxisData(int iop, NDArray temp, int[]? opAxisMap)
        {
            var tDims = temp.Shape.dimensions;
            var tStrides = temp.Shape.strides;
            int tNDim = tDims.Length;
            int ndim = _state->NDim;
            long* stridePtr = _state->GetStridesPointer(iop);
            int itemsize = temp.dtype.itemsize;
            long baseOffset = 0;

            for (int d = 0; d < ndim; d++)
            {
                int opAxis = OperandAxisOf(d, tNDim, opAxisMap);
                if (opAxis < 0 || tDims[opAxis] == 1)
                    continue;
                long s = tStrides[opAxis];
                if (_state->Perm[d] < 0)
                {
                    stridePtr[d] = -s;
                    baseOffset += s * (tDims[opAxis] - 1) * itemsize;
                }
                else
                {
                    stridePtr[d] = s;
                }
            }

            byte* data = (byte*)temp.Address + temp.Shape.offset * (long)itemsize + baseOffset;
            _state->ResetDataPtrs[iop] = (long)data;
            _state->DataPtrs[iop] = (long)data;
            _state->BaseOffsets[iop] = baseOffset;
            _state->SetOpSrcDType(iop, temp.dtype);
            _state->SetOpFlags(iop, _state->GetOpFlags(iop) & ~NDIterOpFlags.CAST);
        }

        /// <summary>
        ///     The operand axis under iterator axis <paramref name="d"/> (whatever its position after the reorder), or -1 when
        ///     the operand does not span it — through its <c>op_axes</c> map, else right-aligned broadcasting.
        /// </summary>
        /// <param name="d">The iterator axis, in the current (reordered, not yet coalesced) representation.</param>
        /// <param name="opNDim">The operand's rank.</param>
        /// <param name="opAxisMap">The operand's axis map, or null.</param>
        /// <returns>The operand axis, or -1.</returns>
        private int OperandAxisOf(int d, int opNDim, int[]? opAxisMap)
        {
            int p = _state->Perm[d];
            int orig = p < 0 ? -1 - p : p;
            if (opAxisMap != null)
                return orig < opAxisMap.Length ? NDIterUtils.GetOpAxis(opAxisMap[orig], out _) : -1;
            int opAxis = orig - (_state->NDim - opNDim);
            return opAxis;
        }

        /// <summary>
        ///     The bytes one element of <paramref name="arr"/> occupies — the storage lane's size, or a fixed-width string's
        ///     item size (the <see cref="NPTypeCode.String"/> lane names no width). Never <c>dtypesize</c>, which is
        ///     <c>Marshal.SizeOf</c> (4 for <see cref="bool"/>).
        /// </summary>
        /// <param name="arr">An operand.</param>
        /// <returns>The element size in bytes.</returns>
        internal static int ElementBytes(NDArray arr)
            => arr.typecode == NPTypeCode.String ? arr.dtype.itemsize : arr.GetTypeCode.SizeOf();
    }
}
