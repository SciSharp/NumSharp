using System;
using System.Runtime.CompilerServices;
using NumSharp.Backends;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        /// <param name="ary">Array to be divided into sub-arrays.</param>
        /// <param name="indices_or_sections">
        /// If an integer, N, the array will be divided into N equal arrays along axis.
        /// If such a split is not possible, an error is raised.
        /// </param>
        /// <param name="axis">The axis along which to split, default is 0.</param>
        /// <returns>A list of sub-arrays as views into ary.</returns>
        /// <exception cref="ArgumentException">If indices_or_sections is an integer and does not result in equal division.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.split.html
        /// </remarks>
        public static NDArray[] split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));

            // array_split's argument validation handles section<=0; we validate first
            // so the equal-division check below doesn't divide by zero (NumPy: raw
            // int /% 0 throws ZeroDivisionError, but our ArgumentException is clearer
            // and consistent with array_split's own check).
            if (indices_or_sections <= 0)
                throw new ArgumentException("number sections must be larger than 0.");

            int ax = NormalizeSplitAxis(axis, ary.ndim);

            long N = ary.Shape.dimensions[ax];
            if (N % indices_or_sections != 0)
                throw new ArgumentException("array split does not result in an equal division");

            return SplitContext.FromParent(ary, ax).SplitBySections(indices_or_sections);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        public static NDArray[] split(NDArray ary, long[] indices, int axis = 0)
            => array_split(ary, indices, axis);

        /// <summary>
        /// Split an array into multiple sub-arrays as views into ary.
        /// </summary>
        public static NDArray[] split(NDArray ary, int[] indices, int axis = 0)
            => array_split(ary, indices, axis);

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        /// <remarks>
        /// The only difference between split and array_split is that array_split allows
        /// indices_or_sections to be an integer that does not equally divide the axis.
        /// https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
        /// </remarks>
        public static NDArray[] array_split(NDArray ary, int indices_or_sections, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));
            if (indices_or_sections <= 0)
                throw new ArgumentException("number sections must be larger than 0.");

            int ax = NormalizeSplitAxis(axis, ary.ndim);
            return SplitContext.FromParent(ary, ax).SplitBySections(indices_or_sections);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        public static NDArray[] array_split(NDArray ary, long[] indices, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int ax = NormalizeSplitAxis(axis, ary.ndim);
            return SplitContext.FromParent(ary, ax).SplitIndices(indices);
        }

        /// <summary>
        /// Split an array into multiple sub-arrays.
        /// </summary>
        public static NDArray[] array_split(NDArray ary, int[] indices, int axis = 0)
        {
            if (ary is null) throw new ArgumentNullException(nameof(ary));
            if (indices is null) throw new ArgumentNullException(nameof(indices));

            int ax = NormalizeSplitAxis(axis, ary.ndim);
            return SplitContext.FromParent(ary, ax).SplitIndices(indices);
        }

        /// <summary>
        ///     Normalises a possibly-negative axis to the [0, ndim) range. Throws when
        ///     the array is 0-d (no axes to split on) or the axis is out of range.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int NormalizeSplitAxis(int axis, int ndim)
        {
            if (ndim == 0)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {ndim}");

            int adjusted = axis < 0 ? axis + ndim : axis;
            if (adjusted < 0 || adjusted >= ndim)
                throw new ArgumentOutOfRangeException(nameof(axis),
                    $"axis {axis} is out of bounds for array of dimension {ndim}");
            return adjusted;
        }

        /// <summary>
        ///     Shared state for one split call. Snapshots the parent's shape /
        ///     storage / engine / per-axis derivation once and exposes
        ///     <see cref="SplitBySections"/> for the integer-section paths
        ///     (<c>split(a, N)</c> / <c>array_split(a, N)</c>) and
        ///     <see cref="SplitIndices(long[])"/> / <see cref="SplitIndices(int[])"/>
        ///     for the explicit-indices paths
        ///     (<c>split(a, [3,5,6])</c> / <c>array_split(a, [3,5,6])</c>).
        ///     Naming note: "Sections" refers to NumPy's
        ///     <c>indices_or_sections</c> integer mode — NOT a dtype. Split is
        ///     dtype-agnostic; only views are produced, no element loop runs.
        ///     Each per-sub-array call:
        ///     <list type="bullet">
        ///         <item>Reuses a shared dims[] when the previous sub had the same length on the split axis (typical case for int sections — at most 2 distinct lengths)</item>
        ///         <item>Derives sub flags from parent flags in O(1) via <see cref="DeriveSubFlags"/></item>
        ///         <item>Derives sub size in O(1) from <c>parent.size * subLen / parent.dim[axis]</c></item>
        ///         <item>Constructs Shape through the no-walk ctor</item>
        ///     </list>
        ///     This pulls the per-sub-array cost from ~640ns (Shape + Alias + NDArray)
        ///     to ~540ns on a 1-D arange(1000) → 4 split benchmark.
        /// </summary>
        private readonly struct SplitContext
        {
            private readonly NDArray _ary;
            private readonly long[] _srcDims;
            private readonly long[] _srcStrides;
            private readonly long _axisStride;
            private readonly long _baseOffset;
            private readonly long _bufSize;
            private readonly long _axisDim;             // parent.dimensions[axis]
            private readonly long _otherDimsProduct;    // parent.size / axisDim (size of one slab along axis)
            private readonly TensorEngine _engine;
            private readonly int _ndim;
            private readonly int _axis;
            private readonly int _parentFlags;
            private readonly bool _parentBroadcasted;

            private SplitContext(NDArray ary, int axis)
            {
                _ary = ary;
                _engine = ary.TensorEngine;
                var shp = ary.Shape;
                _srcDims = shp.dimensions;
                _srcStrides = shp.strides;
                _axisStride = _srcStrides[axis];
                _baseOffset = shp.offset;
                _bufSize = shp.bufferSize > 0 ? shp.bufferSize : shp.size;
                _ndim = _srcDims.Length;
                _axis = axis;
                _axisDim = _srcDims[axis];
                _parentFlags = shp._flags;
                _parentBroadcasted = (_parentFlags & (int)ArrayFlags.BROADCASTED) != 0;
                // Per-slab size: parent.size / axisDim. For axisDim==0 the parent is
                // empty; we keep otherDimsProduct=0 so sub-size collapses to 0.
                _otherDimsProduct = _axisDim == 0 ? 0 : shp.size / _axisDim;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal static SplitContext FromParent(NDArray ary, int axis) => new SplitContext(ary, axis);

            /// <summary>
            ///     Equal-section split (NumPy <c>split(a, N)</c> /
            ///     <c>array_split(a, N)</c> when <c>indices_or_sections</c> is an
            ///     integer). <c>extras = N % Nsections</c> sub-arrays get size
            ///     <c>N/Nsections + 1</c>, the rest get <c>N/Nsections</c>.
            ///     Only TWO distinct sub-lengths exist — we materialise dims[]
            ///     for each and share across all subs with the matching length.
            /// </summary>
            /// <param name="Nsections">Number of sections to split into. Must be &gt; 0.</param>
            internal NDArray[] SplitBySections(int Nsections)
            {
                long Neach = _axisDim / Nsections;
                long extras = _axisDim % Nsections;

                long[] dimsLarge = null;   // for subs of size Neach+1
                long[] dimsSmall = null;   // for subs of size Neach
                // Pre-compute the (flags, size, hash) tuple for both sub-lengths once.
                int flagsLarge = 0, flagsSmall = 0;
                long sizeLarge = 0, sizeSmall = 0;
                int hashLarge = 0, hashSmall = 0;
                if (extras > 0)
                {
                    dimsLarge = BuildSubDims(Neach + 1);
                    flagsLarge = DeriveSubFlags(Neach + 1);
                    sizeLarge = _otherDimsProduct * (Neach + 1);
                    hashLarge = ComputeHashFromDims(dimsLarge, sizeLarge);
                }
                if (Nsections > extras)
                {
                    dimsSmall = BuildSubDims(Neach);
                    flagsSmall = DeriveSubFlags(Neach);
                    sizeSmall = _otherDimsProduct * Neach;
                    hashSmall = ComputeHashFromDims(dimsSmall, sizeSmall);
                }

                var sub_arys = new NDArray[Nsections];
                long cursor = 0;
                for (int i = 0; i < Nsections; i++)
                {
                    bool isLarge = i < extras;
                    long size = isLarge ? Neach + 1 : Neach;
                    sub_arys[i] = BuildView(
                        isLarge ? dimsLarge : dimsSmall,
                        isLarge ? flagsLarge : flagsSmall,
                        isLarge ? sizeLarge : sizeSmall,
                        isLarge ? hashLarge : hashSmall,
                        cursor);
                    cursor += size;
                }

                return sub_arys;
            }

            /// <inheritdoc cref="SplitIndices(int[])"/>
            internal NDArray[] SplitIndices(long[] indices)
            {
                int Nsections = indices.Length + 1;
                var sub_arys = new NDArray[Nsections];
                long prev = 0;
                long lastSubLen = -1;
                long[] cachedDims = null;
                int cachedFlags = 0;
                long cachedSize = 0;
                int cachedHash = 0;

                for (int i = 0; i < Nsections; i++)
                {
                    long raw = (i == indices.Length) ? _axisDim : indices[i];
                    long cur = ClampSlicePoint(raw, _axisDim);
                    long st = prev;
                    long end = cur < st ? st : cur;
                    long subLen = end - st;

                    // Cache the dims/flags/size/hash quadruple keyed on subLen so
                    // adjacent same-length sub-arrays share allocations (common for
                    // even indices like [3,6,9] or repeated indices).
                    if (subLen != lastSubLen)
                    {
                        cachedDims = BuildSubDims(subLen);
                        cachedFlags = DeriveSubFlags(subLen);
                        cachedSize = _otherDimsProduct * subLen;
                        cachedHash = ComputeHashFromDims(cachedDims, cachedSize);
                        lastSubLen = subLen;
                    }
                    sub_arys[i] = BuildView(cachedDims, cachedFlags, cachedSize, cachedHash, st);
                    prev = cur;
                }
                return sub_arys;
            }

            /// <summary>
            ///     Indices-mode split that walks the indices array directly without
            ///     allocating a div_points scratch buffer. Walks the boundary list
            ///     <c>0, indices[0..^1], Ntotal</c> with two cursors (prev, cur).
            ///     Caches the most recently used (dims, flags, size, hash) tuple so
            ///     repeated same-length sub-arrays don't realloc.
            /// </summary>
            internal NDArray[] SplitIndices(int[] indices)
            {
                int Nsections = indices.Length + 1;
                var sub_arys = new NDArray[Nsections];
                long prev = 0;
                long lastSubLen = -1;
                long[] cachedDims = null;
                int cachedFlags = 0;
                long cachedSize = 0;
                int cachedHash = 0;

                for (int i = 0; i < Nsections; i++)
                {
                    long raw = (i == indices.Length) ? _axisDim : indices[i];
                    long cur = ClampSlicePoint(raw, _axisDim);
                    long st = prev;
                    long end = cur < st ? st : cur;
                    long subLen = end - st;

                    if (subLen != lastSubLen)
                    {
                        cachedDims = BuildSubDims(subLen);
                        cachedFlags = DeriveSubFlags(subLen);
                        cachedSize = _otherDimsProduct * subLen;
                        cachedHash = ComputeHashFromDims(cachedDims, cachedSize);
                        lastSubLen = subLen;
                    }
                    sub_arys[i] = BuildView(cachedDims, cachedFlags, cachedSize, cachedHash, st);
                    prev = cur;
                }
                return sub_arys;
            }

            /// <summary>
            ///     Build the sub-array's dims[] by cloning parent's dims and patching
            ///     <c>dims[axis] = subLen</c>. Shape stores dims by reference so
            ///     callers can share this array across same-length sub-arrays.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private long[] BuildSubDims(long subLen)
            {
                var dims = new long[_ndim];
                Array.Copy(_srcDims, dims, _ndim);
                dims[_axis] = subLen;
                return dims;
            }

            /// <summary>
            ///     Build a sub-array NDArray view from a pre-computed (dims, flags,
            ///     size, hash) tuple and a start offset (in elements) along the
            ///     split axis. The Shape uses the no-walk ctor.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private NDArray BuildView(long[] dims, int flags, long size, int hash, long startElems)
            {
                long newOffset = _baseOffset + startElems * _axisStride;
                var newShape = new Shape(dims, _srcStrides, newOffset, _bufSize, flags, size, hash);
                // Hot-path ctor: pre-resolved engine, no `?? BackendFactory.GetEngine()`
                // guard, no `Storage.Engine = tensorEngine` writeback.
                return new NDArray(_ary.Storage.Alias(newShape), _engine, skipEngineResolve: true);
            }

            /// <summary>
            ///     Derive sub-array <see cref="ArrayFlags"/> from parent flags in O(1).
            ///     <list type="bullet">
            ///       <item>Empty sub (any dim==0): NumPy convention → both C and F contig, WRITEABLE, ALIGNED, no BROADCASTED.</item>
            ///       <item>Parent broadcasted: sub inherits BROADCASTED + !WRITEABLE (its stride-0 axes are untouched).</item>
            ///       <item>Parent C-contig: sub stays C-contig iff <c>axis==0</c> or <c>subLen == parent.dim[axis]</c>; otherwise the <c>strides[i] == dims[i+1]*strides[i+1]</c> invariant breaks at <c>i = axis-1</c>.</item>
            ///       <item>Parent F-contig: symmetric — stays F-contig iff <c>axis==ndim-1</c> or <c>subLen == parent.dim[axis]</c>.</item>
            ///       <item>WRITEABLE inherits from parent: read-only views (e.g. <c>np.diagonal</c> output) propagate their non-writeable status to sub-arrays.</item>
            ///     </list>
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private int DeriveSubFlags(long subLen)
            {
                // NumPy convention: any 0-dim → both C and F contig (vacuously),
                // ALIGNED. WRITEABLE inherits from parent (so a read-only view's
                // empty sub stays read-only).
                int parentWriteable = _parentFlags & (int)ArrayFlags.WRITEABLE;
                if (subLen == 0 || _otherDimsProduct == 0)
                    return (int)(ArrayFlags.C_CONTIGUOUS | ArrayFlags.F_CONTIGUOUS
                                | ArrayFlags.ALIGNED) | parentWriteable;

                // Broadcast preserved (sub's broadcast axes are inherited unchanged).
                // BROADCASTED implies non-writeable per NumPy.
                if (_parentBroadcasted)
                    return (int)(ArrayFlags.BROADCASTED | ArrayFlags.ALIGNED);

                int flags = (int)ArrayFlags.ALIGNED | parentWriteable;
                bool sameLen = subLen == _axisDim;
                bool parentC = (_parentFlags & (int)ArrayFlags.C_CONTIGUOUS) != 0;
                bool parentF = (_parentFlags & (int)ArrayFlags.F_CONTIGUOUS) != 0;

                // C-contig invariant breaks at i=axis-1 when subLen < axisDim and axis>0.
                if (parentC && (_axis == 0 || sameLen))
                    flags |= (int)ArrayFlags.C_CONTIGUOUS;

                // F-contig invariant breaks at i=axis+1 when subLen < axisDim and axis<ndim-1.
                if (parentF && (_axis == _ndim - 1 || sameLen))
                    flags |= (int)ArrayFlags.F_CONTIGUOUS;

                return flags;
            }

            /// <summary>
            ///     Reconstruct Shape's standard hash for a sub-dims[] array. Mirrors
            ///     <c>ComputeSizeAndHash</c> exactly — same seed (<c>'C' * 397</c>),
            ///     same XOR formula — so Shape.GetHashCode stays consistent and
            ///     IsEmpty (which checks <c>_hashCode == 0</c>) never misfires.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static int ComputeHashFromDims(long[] dims, long finalSize)
            {
                if (dims == null || dims.Length == 0)
                    return int.MinValue;
                // Seed must match Shape.ComputeSizeAndHash: `layout * 397` where
                // layout == 'C' (67). Without the non-zero seed, XOR-fold can land
                // on 0 (e.g. dims=[4,2]) and Shape.IsEmpty reads as true.
                int hash = unchecked('C' * 397);
                long size = 1;
                unchecked
                {
                    foreach (var v in dims)
                    {
                        size *= v;
                        hash ^= ((int)(size & 0x7FFFFFFF) * 397) * ((int)(v & 0x7FFFFFFF) * 397);
                    }
                }
                return hash;
            }
        }

        /// <summary>
        ///     NumPy slice clamping: <c>n &lt; 0 ? max(0, n + N) : min(n, N)</c>.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static long ClampSlicePoint(long n, long N)
        {
            if (n < 0)
            {
                long wrapped = n + N;
                return wrapped < 0 ? 0 : wrapped;
            }
            return n > N ? N : n;
        }
    }
}
