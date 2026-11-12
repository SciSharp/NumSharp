using System;
using System.Collections.Generic;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;
using NumSharp.Generic;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        private static long CountNonZeroDispatch<T>(NDArray nd) where T : unmanaged
            => count_nonzero<T>(nd.MakeGeneric<T>());

        /// <summary>
        ///     <c>np.nonzero</c> — returns a tuple of <c>ndim</c> int64 arrays of length
        ///     <c>N</c>, containing the per-dim coordinates of non-zero elements in C-order.
        ///
        ///     <para>
        ///     <b>Implementation:</b> three IL-emitted kernels keyed off the element type
        ///     (<see cref="DirectILKernelGenerator.GetArgwhereCountKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetArgwhereFlatKernel"/>,
        ///     <see cref="DirectILKernelGenerator.GetArgwhereExpandKernel"/>). The runtime call
        ///     site has zero <c>typeof(T)</c> branches: it looks the kernels up in the
        ///     per-dtype <see cref="System.Collections.Concurrent.ConcurrentDictionary{Type,Object}"/>
        ///     cache and invokes them. Every loop (SIMD popcount, SIMD bit-scan, coord
        ///     expand + carry chain) lives inside the emitted IL.
        ///     </para>
        ///
        ///     <para>
        ///     <b>Two-pass pre-size-then-fill:</b> the SIMD popcount sizes the result
        ///     exactly, the SIMD bit-scan writes flat indices either straight into the
        ///     ndim==1 result buffer or into a temp <c>long[]</c> which the IL row-expand
        ///     kernel walks once to fill the shared <c>(count, ndim)</c> multi-index matrix
        ///     whose columns are the result. Mirrors the design of
        ///     <see cref="DefaultEngine.Argwhere"/>, which fills NumPy's transposed
        ///     <c>(ndim, count)</c> stack instead.
        ///     </para>
        ///
        ///     <para>
        ///     Routing:
        ///     <list type="bullet">
        ///       <item>0-d → NumPy 2.x's ValueError, every dtype, bool with its longer text (parity QA-02: the numeric
        ///             dtypes used to answer like <c>atleast_1d(a).nonzero()</c>, which NumPy's message points callers
        ///             to).</item>
        ///       <item>size == 0 → tuple of <c>ndim</c> empty int64 arrays.</item>
        ///       <item>Contiguous → IL count + IL flat-scan + (ndim==1 ? direct write : IL per-dim expand).</item>
        ///       <item>Non-contiguous → materialize via <c>ascontiguousarray</c> then
        ///             same path as contig.</item>
        ///     </list>
        ///     </para>
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <returns>One int64 index array per dimension.</returns>
        /// <exception cref="ValueError">A 0-d array of any dtype, raised before the element is read (single-argument
        ///     <c>np.where</c> inherits it): NumPy's <c>Calling nonzero on 0d arrays is not allowed. Use
        ///     np.atleast_1d(scalar).nonzero() instead.</c>, which a bool array follows with <c>If the context of this
        ///     error is of the form `arr[nonzero(cond)]`, just use `arr[cond]`.</c> (<see cref="NonZero0dMessage"/>).</exception>
        /// <remarks>A string array of at least one dimension is searched by its truth values (a non-empty string is True;
        ///     the nonzero route never raises).</remarks>
        [NDScoped]
        public override unsafe NDArray<long>[] NonZero(NDArray nd)
        {
            // Text: NumPy's nonzero refuses a 0-d array before it reads the element (the string stand-in raises the
            // non-bool text itself), then reads the strings' truth values: the stand-in is the truth array, searched below.
            if (Strings.NDStringUfuncs.ReduceStandIn("nonzero", nd) is { } standIn)
            {
                using (standIn)
                    return NonZero(standIn);
            }
            // Boundary scope: the yielded tuple is the column VIEWS; the shared multi-index base
            // matrix is reclaimed at exit — ARC keeps the buffer alive through the yielded views until the caller
            // releases them.

            // 0-d: NumPy 2.4.2 refuses for every dtype, before reading the element. The numeric dtypes used to answer like
            // np.atleast_1d(a).nonzero() to spare callers a break; the owner chose NumPy's error (parity QA-02, principle
            // 4), and the message names the one-line migration. np.argwhere keeps its own atleast_1d branch (NumPy's), so
            // it never reaches here 0-d.
            if (nd.ndim == 0)
                throw new ValueError(NonZero0dMessage(nd.typecode));

            int ndim = nd.ndim;
            long size = nd.size;

            // Empty input → tuple of `ndim` empty int64 arrays.
            // Covers shape (0,), (0, 3), (2, 0, 4), ….
            if (size == 0)
                return MakeEmptyNonZeroResult(ndim);

            // Materialize non-contig to C-contig. For contig inputs we read from `nd`
            // directly (no copy); the local `materialized` is only set on the non-contig
            // branch and disposed in `finally` — see the ARC note at the bottom of the
            // method for why the explicit Dispose matters here.
            NDArray materialized = null;
            NDArray source = nd;
            if (!nd.Shape.IsContiguous)
            {
                materialized = NDArray.ascontiguousarray(nd, null);
                source = materialized;
            }
            var sourceShape = source.Shape;

            byte* basePtr = (byte*)source.Storage.Address + sourceShape.offset * nd.dtypesize;

            var countKernel = DirectILKernelGenerator.GetArgwhereCountKernel(nd.dtype.type);
            var flatKernel = DirectILKernelGenerator.GetArgwhereFlatKernel(nd.dtype.type);
            if (countKernel == null || flatKernel == null)
                throw new NotSupportedException($"np.nonzero: no IL kernel available for {nd.dtype.type.Name}");

            try
            {
                long count = countKernel(basePtr, size);
                if (count == 0)
                    return MakeEmptyNonZeroResult(ndim);

                // NumPy's PyArray_Nonzero fills ONE (count, ndim) C-order multi-index buffer and
                // the tuple entries are its COLUMNS — strided views (shape (count,), byte stride
                // ndim*8, owndata=False) all sharing that base (probed 2.4.2: tuple[j].base is one
                // (count, ndim) array; ndim==1 collapses to a unit-stride contiguous view of a
                // (count, 1) base). The base matrix is byte-identical to np.argwhere's result, so
                // the ndim>1 fill drives the SAME row-expand IL kernel (one sequential write
                // stream — better locality than ndim separate column streams).
                var multiIndex = NewArray(NPTypeCode.Int64, new Shape(count, ndim), false);
                long* resPtr = (long*)multiIndex.Storage.Address;

                // ndim == 1: the (count, 1) matrix's single column IS the flat index list —
                // scan straight into the base buffer.
                if (ndim == 1)
                {
                    flatKernel(basePtr, size, resPtr);
                    return MakeNonZeroColumnViews(multiIndex, count, ndim);
                }

                // ndim > 1: scan into a temp flat buffer then row-expand into the shared matrix.
                var flatBuffer = new long[count];
                fixed (long* flatPtr = flatBuffer)
                fixed (long* dimsPtr = sourceShape.dimensions)
                {
                    flatKernel(basePtr, size, flatPtr);

                    // Pre-compute dim strides for the expand kernel:
                    //   dimStrides[ndim-1] = 1
                    //   dimStrides[d]      = dimStrides[d+1] * dims[d+1]
                    // Cheap O(ndim) prologue. Mirrors the argwhere expand-kernel prologue.
                    Span<long> dimStrides = stackalloc long[ndim];
                    dimStrides[ndim - 1] = 1;
                    for (int d = ndim - 2; d >= 0; d--)
                        dimStrides[d] = dimStrides[d + 1] * sourceShape.dimensions[d + 1];

                    fixed (long* dimStridesPtr = dimStrides)
                    {
                        var expandKernel = DirectILKernelGenerator.GetArgwhereExpandKernel();
                        if (expandKernel == null)
                            throw new NotSupportedException("np.nonzero: row-expand IL kernel unavailable");

                        expandKernel(flatPtr, count, dimsPtr, dimStridesPtr, ndim, resPtr);
                    }
                }

                return MakeNonZeroColumnViews(multiIndex, count, ndim);
            }
            finally
            {
                // ARC: source.Storage.Address is an unmanaged pointer that does NOT
                // keep `materialized` GC-alive. Under repeated nonzero() calls on a
                // non-contig input, the JIT can decide `materialized` is dead after
                // basePtr is computed — then the result-NDArray allocations below
                // trigger GC, the freshly materialized array's UnmanagedStorage gets
                // its refcount finalized to zero, and the buffer behind basePtr is
                // freed mid-IL-scan (the bench harness reproduces this as a Release-
                // mode AccessViolationException in IL_ArgwhereFlat_*).
                //
                // Explicit Dispose() is the established ARC-release pattern in this
                // codebase (see commits 392529f2, 294d4329) — it forces the JIT to
                // keep `materialized` rooted until the call site here. For the contig
                // fast path materialized is null and Dispose is skipped.
                materialized?.Dispose();
                // The contiguous route reads the CALLER's operand through basePtr; a temporary operand
                // (`np.argwhere(a > 0)`) is referenced only by this parameter, which the JIT may treat as dead
                // once the pointer is taken — keep it reachable until the kernels are done.
                GC.KeepAlive(nd);
            }
        }

        /// <summary>
        ///     NumPy 2.4.2's refusal text for <c>nonzero</c> of a 0-d array (<c>PyArray_Nonzero</c>,
        ///     <c>item_selection.c</c>): every dtype gets <c>Calling nonzero on 0d arrays is not allowed. Use
        ///     np.atleast_1d(scalar).nonzero() instead.</c>, and a BOOL array gets the hint <c>If the context of this
        ///     error is of the form `arr[nonzero(cond)]`, just use `arr[cond]`.</c> after it, because a 0-d boolean
        ///     condition is usually a mask written as <c>arr[np.nonzero(cond)]</c>.
        /// </summary>
        /// <param name="dtype">The 0-d array's dtype; only <see cref="NPTypeCode.Boolean"/> changes the text.</param>
        /// <returns>The text to raise as a <see cref="ValueError"/>.</returns>
        /// <remarks>Shared by every 0-d nonzero refusal that can see a bool operand (np.nonzero / ndarray.nonzero /
        ///     single-argument np.where, numpy.ma's nonzero through them); the string stand-in and the object array
        ///     (<c>NDListArray</c>) are never bool and keep the short text.</remarks>
        internal static string NonZero0dMessage(NPTypeCode dtype)
            => dtype == NPTypeCode.Boolean
                ? "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead. "
                  + "If the context of this error is of the form `arr[nonzero(cond)]`, just use `arr[cond]`."
                : "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.";

        private NDArray<long>[] MakeEmptyNonZeroResult(int ndim)
        {
            // NumPy: even the empty result is columns of a shared (0, ndim) multi-index buffer —
            // each tuple entry keeps byte stride ndim*8 and owndata=False (probed 2.4.2).
            var emptyBase = NewArray(NPTypeCode.Int64, new Shape(0, ndim), false);
            var views = MakeNonZeroColumnViews(emptyBase, 0, ndim);
            emptyBase.Dispose(); // each column view holds its own counted ref on the shared buffer
            return views;
        }

        /// <summary>
        ///     Wrap the shared (count, ndim) multi-index matrix as NumPy's nonzero tuple: entry
        ///     <c>d</c> is the d-th COLUMN — a strided view (shape (count,), element stride ndim,
        ///     element offset d) whose storage aliases the matrix, so <c>owndata=False</c> and all
        ///     entries report one shared base, exactly like <c>PyArray_Nonzero</c>'s
        ///     <c>ret[..., j]</c> views. ndim==1 collapses to a unit-stride view (C=F-contiguous),
        ///     matching NumPy's (count, 1) base there; size-0 views keep the stride/offset pattern
        ///     (never dereferenced).
        /// </summary>
        private static NDArray<long>[] MakeNonZeroColumnViews(NDArray multiIndex, long count, int ndim)
        {
            var result = new NDArray<long>[ndim];
            for (int d = 0; d < ndim; d++)
            {
                var view = new Shape(new long[] { count }, new long[] { ndim }, d, count * (long)ndim);
                result[d] = new NDArray<long>(multiIndex.Storage.Alias(view)) { TensorEngine = multiIndex.TensorEngine };
            }

            return result;
        }

        /// <summary>
        /// Count the number of non-zero elements in the array.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <returns>The number of elements whose truth value is True.</returns>
        /// <remarks>A string array is counted by its truth values (a non-empty string counts; never raises).</remarks>
        /// <remarks>
        /// NumPy-aligned: np.count_nonzero([0, 1, 0, 2]) = 2
        /// </remarks>
        [NDScoped]
        public override long CountNonZero(NDArray nd)
        {
            // An array with no element counts nothing, before anything reads it (np.count_nonzero's first statement, moved
            // here by the np-layer dtype plan's P1b: an empty text array never builds its truth array).
            if (nd.size == 0)
                return 0;
            // Text: NumPy's count_nonzero reads the strings' truth values (the nonzero route): the stand-in is the truth array.
            if (Strings.NDStringUfuncs.ReduceStandIn("count_nonzero", nd) is { } standIn)
            {
                using (standIn)
                    return CountNonZero(standIn);
            }

            // Scalar return — scope only: reclaims the per-dispatch MakeGeneric alias
            // (a fresh wrapper over nd's storage that otherwise strands on the finalizer).
            return NpFunc.Invoke(nd.typecode, CountNonZeroDispatch<int>, nd);
        }

        /// <summary>
        ///     NumPy's <c>np.count_nonzero(a, axis=None, *, keepdims=False)</c>: the flat count as NumPy's scalar without an
        ///     axis or keepdims, otherwise the int64 sum of the truth values over the given axes.
        /// </summary>
        /// <param name="nd">The array (any dtype, any layout).</param>
        /// <param name="axis">null (None), one axis, or a tuple of axes.</param>
        /// <param name="keepdims">Whether the counted axes stay with length one.</param>
        /// <returns>The int64 counts (see <see cref="TensorEngine.CountNonZero(NDArray, NDAxis?, bool)"/>).</returns>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ValueError">A tuple names an axis twice (<c>duplicate value in 'axis'</c>).</exception>
        /// <remarks>An integer axis keeps its own route (the sum's 0-d special case, the empty-array zeros); a tuple, even of
        ///     one axis, has no 0-d special case, exactly as NumPy's sum of the truth values answers each.</remarks>
        public override NDArray CountNonZero(NDArray nd, NDAxis? axis, bool keepdims = false)
        {
            if (axis is null)
                return CountNonZeroEveryAxis(nd, keepdims);
            if (!axis.Value.IsTuple)
                return CountNonZeroAlong(nd, axis.Value[0], keepdims);
            return CountNonZeroAlong(nd, axis.Value.ToArray(), keepdims);
        }

        /// <summary>
        ///     <c>np.count_nonzero(a, axis=None, keepdims=)</c>: <c>multiarray.count_nonzero(a)</c> (NumPy's intp scalar)
        ///     without keepdims, <c>a_bool.sum(axis=None, dtype=intp, keepdims=True)</c> with it - the one count in an int64
        ///     array of <paramref name="nd"/>'s rank, every axis of length one.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="keepdims">Whether every axis stays with length one.</param>
        /// <returns>A read-only 0-d int64 array (NumPy's scalar) without keepdims or for a 0-d input (NumPy's sum of a 0-d
        ///     array is its scalar, keepdims or not); otherwise a fresh writeable int64 array of shape (1, …, 1).</returns>
        private NDArray CountNonZeroEveryAxis(NDArray nd, bool keepdims)
        {
            // The flat count answers every form: text by its truth values, an array with no element with 0.
            NDArray counted = NewScalar(CountNonZero(nd));
            if (!keepdims || nd.ndim == 0)
                return counted.PyArray_Return();

            // keepdims: the one count at the reduced shape (1, ..., 1), as add.reduce allocates it - a fresh owner. The 0-d
            // owner's storage takes the kept rank in place (one element either way; its strides are all zero-length axes).
            var dims = new long[nd.ndim];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = 1;
            counted.Storage.Reshape(new Shape(dims, copy: false));
            return counted;
        }

        /// <summary>
        ///     <c>np.count_nonzero(a, axis=tuple, keepdims=)</c>: <c>a.astype(bool, copy=False).sum(axis=tuple, dtype=intp,
        ///     keepdims=keepdims)</c> - NumPy's ONE add.reduce over the axis set.
        /// </summary>
        /// <param name="nd">The array (any dtype; text is counted by its truth values).</param>
        /// <param name="axes">The axes as given (not normalized; possibly empty).</param>
        /// <param name="keepdims">Whether the counted axes stay with length one.</param>
        /// <returns>The int64 counts, laid out as NumPy's reduction allocates them (the input's memory order); a read-only
        ///     0-d array (NumPy's scalar) when every axis is counted without keepdims.</returns>
        /// <exception cref="AxisError">An axis is out of range (the first one, as given; a 0-d input has no axis for a tuple
        ///     to name).</exception>
        /// <exception cref="ValueError">A tuple names an axis twice (<c>duplicate value in 'axis'</c>), after every axis'
        ///     range.</exception>
        [NDScoped]
        private NDArray CountNonZeroAlong(NDArray nd, int[] axes, bool keepdims)
        {
            // Text: the tuple is checked item by item, then for repeats, before NumPy reads the strings' truth values
            // (a != ''): the stand-in is the truth array, counted below.
            if (Strings.NDStringUfuncs.ReduceStandIn("count_nonzero", nd, axes) is { } standIn)
            {
                using (standIn)
                    return CountNonZeroAlong(standIn, axes, keepdims);
            }

            // The sum's axis conversion: every axis' range first (an AxisError naming the axis as given), then a repeat. A
            // tuple has no 0-d special case: (0,) on a 0-d array is the AxisError NumPy raises.
            bool[] flags = ReductionArguments.AxisFlags(nd.ndim, axes);
            int count = 0;
            foreach (bool f in flags)
                if (f)
                    count++;
            var reduced = new int[count];
            for (int d = 0, k = 0; d < flags.Length; d++)
                if (flags[d])
                    reduced[k++] = d;

            // a.astype(bool, copy=False), then add.reduce over the whole set in the intp loop: np.sum's own routes for one
            // axis and for every axis, NumPy's reduction iterator for any other set (two or more axes, or none). The counts
            // are integers, so no order of the adds changes them.
            NDArray truth = nd.typecode == NPTypeCode.Boolean ? nd : nd.astype(np.@bool, copy: false);
            try
            {
                return ReduceAddOverAxes(truth, reduced, keepdims, NPTypeCode.Int64);
            }
            finally
            {
                // The cast copy is this call's temporary; a bool input is the caller's array and stays alive.
                if (!ReferenceEquals(truth, nd))
                    truth.Dispose();
            }
        }

        /// <summary>
        /// Count non-zero elements along a specific axis.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="axis">The reduced axis.</param>
        /// <param name="keepdims">Whether the reduced axis stays with length one.</param>
        /// <returns>The int64 counts: NumPy's read-only scalar stand-in when the result is 0-d (a 1-D or 0-d input without
        ///     keepdims, empty or not), a fresh writeable array otherwise.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of bounds — the sum's rule NumPy's count_nonzero
        ///     inherits (<c>a.astype(bool).sum(axis)</c>), so a 0-d input accepts 0 and -1 and nothing else.</exception>
        /// <remarks>A string array is counted by its truth values: NumPy's <c>(a != '').sum(axis)</c> (never raises).</remarks>
        [NDScoped]
        private NDArray CountNonZeroAlong(NDArray nd, int axis, bool keepdims)
        {
            // Text: NumPy's count_nonzero is (a != '').sum(axis) — the strings' truth values, after the sum's axis check (a
            // 0-d input accepts 0 and -1): the stand-in is the truth array, summed below.
            if (Strings.NDStringUfuncs.ReduceStandIn("count_nonzero", nd, (int?)axis) is { } standIn)
            {
                using (standIn)
                    return CountNonZeroAlong(standIn, axis, keepdims);
            }
            if (nd.size == 0)
            {
                // An array with no element: int64 zeros of the reduced shape (np.count_nonzero's branch, moved here by the
                // np-layer dtype plan's P1b). The axis is validated as NumPy's sum validates it: an AxisError naming the
                // axis as given (a wrapping loop used to turn -3 on a 1-D array into axis 0).
                if (axis < -nd.ndim || axis >= nd.ndim)
                    throw new AxisError(axis, nd.ndim);
                if (axis < 0)
                    axis += nd.ndim;

                var resultShape = Shape.GetAxis(nd.Shape, axis);
                // np.zeros(shape, int64)'s own allocation (a zero-filled owner), without the call up into np.*.
                var empty = NewArray(NPTypeCode.Int64, new Shape(resultShape, copy: false), true);
                // The kept axis at NumPy's keepdims place (all-zero strides for an empty result).
                if (keepdims)
                    ExpandReducedAxis(empty, nd.Shape, axis);
                // NumPy's count is a.astype(bool).sum(axis), so a 1-D input's 0-d count is the sum's scalar, as the
                // non-empty route below returns it (np.linalg.matrix_rank of an empty matrix with a tolerance came back
                // writeable through here). A no-op for an array result.
                return empty.PyArray_Return();
            }
            // Boundary scope: reclaims the cast copy of a failed call and the scalar path's temporaries; the result is yielded.

            // NumPy's count_nonzero(a, axis) is a.astype(bool).sum(axis=axis, dtype=intp), so the sum's axis rule holds: a
            // 0-d input accepts 0 and -1 (a reduction over no axis — the 0-d count of its one element), and every other
            // axis is normalize_axis_index's AxisError. The wrap-then-test loop this replaces never ended for a negative
            // axis of a 0-d input, and accepted -ndim - 1 (counting along the last axis).
            if (nd.ndim == 0)
            {
                if (axis != 0 && axis != -1)
                    throw new AxisError(axis, 0);
                // The sum of a 0-d input is NumPy's scalar, keepdims or not (np.count_nonzero(np.array(5), axis=0) is
                // np.int64(1)); NDArray.Scalar alone is a writeable 0-d array.
                return NewScalar(CountNonZero(nd)).PyArray_Return();
            }

            axis = Shape.check_and_adjust_axis(nd.ndim, axis);

            // NumPy's count_nonzero along an axis IS `a.astype(bool, copy=False).sum(axis=axis, dtype=np.intp,
            // keepdims=keepdims)` (numeric.py): the bool cast (NaN and inf true, -0.0 false, a complex value true when
            // either part is nonzero), then add.reduce in the int64 loop. Running exactly that gives NumPy's result
            // layout — the reduce iterator's allocation in the input's memory order, F for an F input (plan
            // api-parameter-parity X10) — and np.sum's bool axis-count routes; the per-output scalar loop it replaces
            // wrote C order and ran 15-100x slower than these routes at 100K-1M elements.
            NDArray truth = nd.typecode == NPTypeCode.Boolean ? nd : nd.astype(np.@bool, copy: false);
            try
            {
                return Sum(truth, axis, np.int64, keepdims);
            }
            finally
            {
                // The cast copy is this call's temporary; a bool input is the caller's array and stays alive.
                if (!ReferenceEquals(truth, nd))
                    truth.Dispose();
            }
        }

        /// <summary>
        /// Generic implementation of count_nonzero (element-wise).
        ///
        /// Contig fast path reuses <see cref="DirectILKernelGenerator.GetArgwhereCountKernel"/>
        /// which is the SAME SIMD popcount kernel used by <c>np.nonzero</c>'s pre-size
        /// pass: load a Vector&lt;T&gt;, compare-ne-zero, ExtractMostSignificantBits,
        /// PopCount the inverted mask. The earlier scalar
        /// <c>EqualityComparer&lt;T&gt;.Default.Equals</c> per-element loop was a
        /// 109× regression vs NumPy (2.1 ms vs 19 µs on 1 M bool).
        /// </summary>
        private static unsafe long count_nonzero<T>(NDArray<T> x) where T : unmanaged
        {
            var shape = x.Shape;
            var size = x.size;

            if (shape.IsContiguous)
            {
                var ilKernel = DirectILKernelGenerator.GetArgwhereCountKernel(typeof(T));
                if (ilKernel != null)
                {
                    // Sliced views: Address ignores shape.offset; advance manually so
                    // the kernel sees the live element window.
                    byte* basePtr = (byte*)x.Address + shape.offset * sizeof(T);
                    return ilKernel(basePtr, size);
                }

                // Fallback (only if IL kernel generation is disabled).
                T* ptr = (T*)x.Address + shape.offset;
                T zero = default;
                long count = 0;
                for (long i = 0; i < size; i++)
                {
                    if (!EqualityComparer<T>.Default.Equals(ptr[i], zero))
                        count++;
                }
                return count;
            }

            // Strided path: use NDIter for layout-aware traversal.
            using var iter = NDIterRef.New(x, NDIterGlobalFlags.EXTERNAL_LOOP);
            return iter.ExecuteReducing<CountNonZeroKernel<T>, long>(default, 0L);
        }

    }
}
