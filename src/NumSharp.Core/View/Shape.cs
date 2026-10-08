using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Utilities;

namespace NumSharp
{
    /// <summary>
    ///     NumPy-aligned array flags. Cached at shape creation for O(1) access.
    ///     Matches numpy/core/include/numpy/ndarraytypes.h flag definitions.
    /// </summary>
    [Flags]
    public enum ArrayFlags
    {
        /// <summary>No flags set.</summary>
        None = 0,

        /// <summary>Data is C-contiguous (row-major, last dimension varies fastest).</summary>
        C_CONTIGUOUS = 0x0001,

        /// <summary>Data is F-contiguous (column-major).</summary>
        F_CONTIGUOUS = 0x0002,

        /// <summary>
        ///     Array owns its data buffer (NumPy's <c>NPY_ARRAY_OWNDATA</c>). Not derivable from
        ///     dims/strides — maintained by <see cref="NumSharp.Backends.UnmanagedStorage.OnReshaped"/>
        ///     at the storage funnels: set when the storage allocated its buffer
        ///     (NumPy <c>ctors.c</c>: <c>fa-&gt;flags |= NPY_ARRAY_OWNDATA</c> when <c>data == NULL</c>),
        ///     cleared for views and foreign-memory wraps (<c>base != NULL ⟹ !OWNDATA</c>).
        /// </summary>
        OWNDATA = 0x0004,

        /// <summary>Data is aligned for the CPU (always true for managed allocations).</summary>
        ALIGNED = 0x0100,

        /// <summary>Data is writeable (false for broadcast views).</summary>
        WRITEABLE = 0x0400,

        /// <summary>Shape has a broadcast dimension (stride=0 with dim > 1).</summary>
        BROADCASTED = 0x1000, // NumSharp extension for cached IsBroadcasted
    }

    /// <summary>
    ///     Represents a shape of an N-D array. Immutable after construction (NumPy-aligned).
    /// </summary>
    /// <remarks>Handles slicing, indexing based on coordinates or linear offset and broadcastted indexing.</remarks>
    public readonly partial struct Shape : ICloneable, IEquatable<Shape>
    {
        /// <summary>
        ///     Cached array flags computed at shape creation.
        ///     Use ArrayFlags enum for bit meanings.
        /// </summary>
        internal readonly int _flags;

        /// <summary>
        ///     Hash seed constant used in <see cref="ComputeSizeAndHash"/> for stable Shape hash values.
        ///     NOT the physical memory order — use <see cref="Order"/>, <see cref="IsContiguous"/>,
        ///     or <see cref="IsFContiguous"/> for actual memory layout information.
        /// </summary>
        internal const char layout = 'C';

        internal readonly int _hashCode;
        internal readonly long size;
        internal readonly long[] dimensions;
        internal readonly long[] strides;

        /// <summary>
        ///     Size of the underlying buffer (NumPy-aligned architecture).
        ///     For non-view shapes, equals size. For sliced/broadcast shapes,
        ///     this is the actual buffer size (not the view size), used for
        ///     bounds checking and InternalArray slicing.
        /// </summary>
        internal readonly long bufferSize;

        /// <summary>
        ///     Base offset into storage (NumPy-aligned architecture).
        ///     Computed at slice/broadcast time, enabling simple element access:
        ///     element[indices] = data[offset + sum(indices * strides)]
        /// </summary>
        internal readonly long offset;

        /// <summary>
        ///     True if this shape represents a view (sliced) into underlying data.
        ///     A shape is "sliced" if it doesn't represent the full original buffer.
        ///     This includes: non-zero offset, different size than buffer, or non-contiguous strides.
        /// </summary>
        public readonly bool IsSliced
        {
            [MethodImpl(Inline)]
            get => offset != 0 || (bufferSize > 0 && bufferSize != size) || !IsContiguous;
        }

        /// <summary>
        ///     Does this Shape represent contiguous unmanaged memory in C-order (row-major)?
        ///     Cached flag computed at shape creation, matching NumPy's flags['C_CONTIGUOUS'] algorithm.
        /// </summary>
        /// <remarks>
        ///     NumPy algorithm (from numpy/_core/src/multiarray/flagsobject.c:116-160):
        ///     Scan right-to-left. stride[-1] must equal 1 (itemsize in NumPy, but NumSharp uses element strides).
        ///     stride[i] must equal shape[i+1] * stride[i+1]. Size-1 dimensions are skipped (stride doesn't matter).
        ///     Empty arrays (any dimension is 0) are considered contiguous by definition.
        /// </remarks>
        public readonly bool IsContiguous
        {
            [MethodImpl(Inline)]
            get => (_flags & (int)ArrayFlags.C_CONTIGUOUS) != 0;
        }

        /// <summary>
        ///     Does this Shape represent contiguous unmanaged memory in F-order (column-major)?
        ///     Cached flag computed at shape creation, matching NumPy's flags['F_CONTIGUOUS'] algorithm.
        /// </summary>
        /// <remarks>
        ///     NumPy algorithm: scan left-to-right. stride[0] must equal 1.
        ///     stride[i] must equal shape[i-1] * stride[i-1]. Size-1 dimensions are skipped.
        ///     Empty arrays are considered contiguous by definition.
        ///     A 1-D array that is C-contiguous is also F-contiguous (same memory layout).
        /// </remarks>
        public readonly bool IsFContiguous
        {
            [MethodImpl(Inline)]
            get => (_flags & (int)ArrayFlags.F_CONTIGUOUS) != 0;
        }

#region Static Flag/Hash Computation (for readonly struct)

        /// <summary>
        ///     Computes array flags from dimensions and strides (static for readonly struct).
        /// </summary>
        [MethodImpl(Inline)]
        private static int ComputeFlagsStatic(long[] dims, long[] strides)
        {
            // Empty arrays (any dim == 0) short-circuit per NumPy _UpdateContiguousFlags:
            // unconditionally both C- and F-contiguous, writeable, and NOT broadcast.
            // With no elements, broadcast semantics have no meaning.
            if (dims != null)
            {
                for (int i = 0; i < dims.Length; i++)
                {
                    if (dims[i] == 0)
                    {
                        return (int)(ArrayFlags.C_CONTIGUOUS
                                   | ArrayFlags.F_CONTIGUOUS
                                   | ArrayFlags.ALIGNED
                                   | ArrayFlags.WRITEABLE);
                    }
                }
            }

            int flags = 0;

            // Check BROADCASTED first (only meaningful for non-empty arrays).
            bool isBroadcasted = ComputeIsBroadcastedStatic(dims, strides);
            if (isBroadcasted)
                flags |= (int)ArrayFlags.BROADCASTED;

            // Compute C- and F-contiguity together in a single pass (NumPy-aligned).
            // Broadcast shapes are never flagged as contiguous even if the inner
            // stride pattern would otherwise qualify.
            if (!isBroadcasted)
            {
                var (isC, isF) = ComputeContiguousFlagsStatic(dims, strides);
                if (isC) flags |= (int)ArrayFlags.C_CONTIGUOUS;
                if (isF) flags |= (int)ArrayFlags.F_CONTIGUOUS;
            }

            // ALIGNED is always true because NumSharp uses unaligned SIMD loads (Vector.Load, not LoadAligned)
            flags |= (int)ArrayFlags.ALIGNED;

            // WRITEABLE is true unless this is a broadcast shape (NumPy behavior).
            // Broadcast arrays have stride=0 dimensions where multiple logical positions
            // map to the same physical memory - writing would corrupt shared data.
            if (!isBroadcasted)
                flags |= (int)ArrayFlags.WRITEABLE;

            return flags;
        }

        /// <summary>
        ///     Computes whether any dimension is broadcast (stride=0 with dim > 1).
        ///     Internal so view-producers that pre-derive flags for the no-walk ctor
        ///     (<see cref="np.split"/>'s <c>DeriveSubFlags</c>) judge broadcastness with the
        ///     SAME rule the walking ctor uses.
        /// </summary>
        [MethodImpl(Inline)]
        internal static bool ComputeIsBroadcastedStatic(long[] dims, long[] strides)
        {
            if (strides == null || strides.Length == 0)
                return false;
            for (int i = 0; i < strides.Length; i++)
                if (strides[i] == 0 && dims[i] > 1)
                    return true;
            return false;
        }

        /// <summary>
        ///     Computes both C- and F-contiguity in a single call, matching NumPy's
        ///     <c>_UpdateContiguousFlags</c> in <c>numpy/_core/src/multiarray/flagsobject.c</c>.
        /// </summary>
        /// <remarks>
        ///     From NumPy's source comments:
        ///     <list type="bullet">
        ///         <item>C-contiguous: <c>strides[-1] == itemsize</c> and <c>strides[i] == shape[i+1] * strides[i+1]</c></item>
        ///         <item>F-contiguous: <c>strides[0] == itemsize</c> and <c>strides[i] == shape[i-1] * strides[i-1]</c></item>
        ///         <item>A 0- or 1-dimensional array is either both C- and F-contiguous, or neither.</item>
        ///         <item>Multi-dim arrays can be C, F, or neither, but not both (unless only one element).</item>
        ///         <item>Size-1 dimensions don't count (their strides are unused).</item>
        ///         <item>Any dimension of size 0 makes the array trivially both C- and F-contiguous.</item>
        ///     </list>
        ///     NumSharp uses element-indexed strides (sd starts at 1) rather than byte strides.
        ///     Internal so view-producers that pre-derive flags for the no-walk ctor
        ///     (<see cref="np.split"/>'s <c>DeriveSubFlags</c>) compute contiguity with the SAME
        ///     size-1-relaxed walk the walking ctor uses — parent-flag algebra cannot express the
        ///     relaxation (a <c>(1,4)</c> child of a C-only parent is BOTH C- and F-contiguous).
        /// </remarks>
        [MethodImpl(Inline)]
        internal static (bool isC, bool isF) ComputeContiguousFlagsStatic(long[] dims, long[] strides)
        {
            if (dims == null || dims.Length == 0)
                return (true, true); // scalar is both

            // Empty arrays (any dim == 0) are trivially both C- and F-contiguous (NumPy convention).
            for (int i = 0; i < dims.Length; i++)
            {
                if (dims[i] == 0)
                    return (true, true);
            }

            // C-contiguity: scan right-to-left, stride[-1] must be 1, stride[i] = shape[i+1] * stride[i+1]
            bool isC = true;
            {
                long sd = 1;
                for (int i = dims.Length - 1; i >= 0; i--)
                {
                    long dim = dims[i];
                    if (dim != 1)
                    {
                        if (strides[i] != sd) { isC = false; break; }
                        sd *= dim;
                    }
                }
            }

            // F-contiguity: scan left-to-right, stride[0] must be 1, stride[i] = shape[i-1] * stride[i-1]
            bool isF = true;
            {
                long sd = 1;
                for (int i = 0; i < dims.Length; i++)
                {
                    long dim = dims[i];
                    if (dim != 1)
                    {
                        if (strides[i] != sd) { isF = false; break; }
                        sd *= dim;
                    }
                }
            }

            return (isC, isF);
        }

        /// <summary>
        ///     Computes size and hash from dimensions.
        /// </summary>
        [MethodImpl(Inline)]
        private static (long size, int hash) ComputeSizeAndHash(long[] dims)
        {
            if (dims == null || dims.Length == 0)
                return (1, int.MinValue); // scalar

            long size = 1;
            int hash = layout * 397;
            unchecked
            {
                foreach (var v in dims)
                {
                    size *= v;
                    hash ^= ((int)(size & 0x7FFFFFFF) * 397) * ((int)(v & 0x7FFFFFFF) * 397);
                }
            }

            return (size, hash);
        }

        /// <summary>
        ///     Computes C-contiguous strides for given dimensions.
        /// </summary>
        [MethodImpl(Inline)]
        private static long[] ComputeContiguousStrides(long[] dims)
        {
            if (dims == null || dims.Length == 0)
                return Array.Empty<long>();

            var strides = new long[dims.Length];
            strides[dims.Length - 1] = 1;
            for (int i = dims.Length - 2; i >= 0; i--)
                strides[i] = strides[i + 1] * dims[i + 1];
            return strides;
        }

        /// <summary>
        ///     Computes F-contiguous (column-major) strides for given dimensions.
        ///     strides[0] = 1, strides[i] = dims[i-1] * strides[i-1].
        /// </summary>
        [MethodImpl(Inline)]
        private static long[] ComputeFContiguousStrides(long[] dims)
        {
            if (dims == null || dims.Length == 0)
                return Array.Empty<long>();

            var strides = new long[dims.Length];
            strides[0] = 1;
            for (int i = 1; i < dims.Length; i++)
                strides[i] = strides[i - 1] * dims[i - 1];
            return strides;
        }

        /// <summary>
        ///     Converts int[] dimensions to long[] for backwards compatibility.
        /// </summary>
        [MethodImpl(Inline)]
        public static long[] ComputeLongShape(int[] dimensions)
        {
            if (dimensions == null) return null;
            var result = new long[dimensions.Length];
            for (int i = 0; i < dimensions.Length; i++)
                result[i] = dimensions[i];
            return result;
        }

#endregion

        /// <summary>
        ///     Is this a simple sliced shape that uses the fast GetOffsetSimple path?
        ///     True when: IsSliced && !IsBroadcasted
        ///     For simple slices, element access is: offset + sum(indices * strides)
        /// </summary>
        public readonly bool IsSimpleSlice
        {
            [MethodImpl(Inline)]
            get => IsSliced && !IsBroadcasted;
        }

        /// <summary>
        ///     Is this shape a broadcast (has any stride=0 with dimension > 1)?
        ///     Cached flag computed at shape creation for O(1) access.
        /// </summary>
        public readonly bool IsBroadcasted
        {
            [MethodImpl(Inline)]
            get => (_flags & (int)ArrayFlags.BROADCASTED) != 0;
        }

        /// <summary>
        ///     Is this array writeable? False for broadcast views (NumPy behavior).
        ///     Cached flag computed at shape creation for O(1) access.
        /// </summary>
        public readonly bool IsWriteable
        {
            [MethodImpl(Inline)]
            get => (_flags & (int)ArrayFlags.WRITEABLE) != 0;
        }

        /// <summary>
        ///     Does this array own its data buffer (NumPy's <c>flags.owndata</c>)?
        ///     True for freshly allocated arrays (<c>np.arange</c>, <c>np.zeros</c>, <c>.copy()</c>);
        ///     false for views (slices, transposes, broadcasts) and foreign-memory wraps (memmaps).
        ///     Maintained at the <see cref="NumSharp.Backends.UnmanagedStorage"/> funnels
        ///     (<see cref="NumSharp.Backends.UnmanagedStorage.OnReshaped"/>), mirroring
        ///     <c>UnmanagedStorage.OwnsData</c> — meaningful only on a shape read off a storage
        ///     (<c>nd.Shape</c>); a hand-built <see cref="Shape"/> carries no ownership claim.
        /// </summary>
        public readonly bool OwnsData
        {
            [MethodImpl(Inline)]
            get => (_flags & (int)ArrayFlags.OWNDATA) != 0;
        }

        /// <summary>
        ///     Get all array flags as a single integer.
        ///     Use ArrayFlags enum for bit meanings.
        /// </summary>
        public readonly ArrayFlags Flags
        {
            [MethodImpl(Inline)]
            get => (ArrayFlags)_flags;
        }

        /// <summary>
        ///     Is this shape a scalar that was broadcast to a larger shape?
        ///     True when all strides are 0, meaning all dimensions are broadcast from a scalar.
        ///     Used for optimization: when iterating, we can use a single value instead of indexing.
        /// </summary>
        public readonly bool IsScalarBroadcast
        {
            [MethodImpl(Inline)]
            get
            {
                if (strides == null || strides.Length == 0)
                    return false;
                for (int i = 0; i < strides.Length; i++)
                {
                    if (strides[i] != 0)
                        return false;
                }

                return true;
            }
        }

        /// <summary>
        ///     Computes the size of the original (non-broadcast) data.
        ///     This is the product of all dimensions where stride != 0.
        ///     For a non-broadcast shape, this equals size.
        ///     For a broadcast shape, this is the actual data size before broadcast.
        /// </summary>
        public readonly long OriginalSize
        {
            [MethodImpl(Inline)]
            get
            {
                if (strides == null || strides.Length == 0)
                    return IsScalar ? 1 : size;

                long originalSize = 1;
                for (int i = 0; i < strides.Length; i++)
                {
                    if (strides[i] != 0)
                        originalSize *= dimensions[i];
                }

                return originalSize == 0 ? 1 : originalSize; // At least 1 for scalar broadcasts
            }
        }

        /// <summary>
        ///     Is this shape a scalar? (<see cref="NDim"/>==0 && <see cref="size"/> == 1)
        /// </summary>
        public readonly bool IsScalar;

        /// <summary>
        /// True if the shape is not initialized.
        /// Note: A scalar shape is not empty.
        /// </summary>
        public readonly bool IsEmpty => _hashCode == 0;

        /// <summary>
        ///     Physical memory layout: 'F' if strictly F-contiguous, otherwise 'C'.
        ///     1-D and scalar shapes (both C- and F-contig) report 'C' by convention.
        ///     Non-contiguous shapes also report 'C' as the default reference order.
        /// </summary>
        public readonly char Order => (IsFContiguous && !IsContiguous) ? 'F' : 'C';

        /// <summary>
        ///     Singleton instance of a <see cref="Shape"/> that represents a scalar.
        /// </summary>
        public static readonly Shape Scalar = Shape.Adopt(Array.Empty<long>());

        /// <summary>
        ///     Create a new scalar shape
        /// </summary>
        [MethodImpl(Inline)]
        internal static Shape NewScalar()
        {
            return new Shape();
        }

        /// <summary>
        ///     Create a shape that represents a vector.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Vector(long length)
        {
            return new Shape(new long[] { length }, new long[] { 1 }, 0, length);
        }

        /// <summary>
        ///     Create a shape that represents a matrix.
        /// </summary>
        /// <remarks>Faster than calling Shape's constructor</remarks>
        public static Shape Matrix(long rows, long cols)
        {
            long sz = rows * cols;
            return new Shape(new[] { rows, cols }, new long[] { cols, 1 }, 0, sz);
        }

        public readonly int NDim
        {
            [MethodImpl(Inline)]
            get => dimensions.Length;
        }

        /// <summary>
        ///     The extent of every axis (NumPy's <c>ndarray.shape</c>), as a NEW array on every read: NumPy hands out an
        ///     immutable tuple, so writing into the returned array never changes this shape or any array built on it.
        /// </summary>
        /// <value>A copy of the dimensions, the caller's to keep or change. A 0-d shape returns its empty array (there is
        ///     no element to write into) and an uninitialized <c>default(Shape)</c> returns <c>null</c>.</value>
        /// <remarks>
        ///     <para>
        ///     <b>Why a copy.</b> Shapes and arrays SHARE dimension arrays (an allocation keeps its caller's shape, an
        ///     elementwise result keeps its input's), so returning the shared array let <c>x.Shape.Dimensions[0] = 100</c>
        ///     change <c>x</c>, and every array sharing it, to <c>(100, 4)</c> over 12 elements: the cached size, flags and
        ///     buffer stayed, so the new rows read past the buffer. Plan <c>docs/plans/view-shape-open-bugs.md</c>, bug 2c.
        ///     </para>
        ///     <para>
        ///     The copy costs one small allocation per read. Code inside NumSharp reads the internal <c>dimensions</c> field
        ///     instead; a caller that needs one extent should use <see cref="this[int]"/> or <see cref="NDim"/>, which do not
        ///     allocate, and a caller that reads the whole array in a hot loop should hoist it.
        ///     </para>
        /// </remarks>
        public readonly long[] Dimensions
        {
            [MethodImpl(Inline)]
            get
            {
                // An empty array (0-d) has no element to write into and a default Shape has no array at all, so both are
                // handed out as they are; anything else is cloned so the shared array never leaves the struct.
                var dims = dimensions;
                return dims is null || dims.Length == 0 ? dims : (long[])dims.Clone();
            }
        }

        /// <summary>
        ///     The stride of every axis in ELEMENTS, as a NEW array on every read: writing into the returned array never
        ///     changes this shape or any array built on it. NumPy's <c>ndarray.strides</c> counts BYTES; that is
        ///     <see cref="NDArray.strides"/>, which scales these by the item size.
        /// </summary>
        /// <value>A copy of the element strides (0 on a broadcast axis, negative on a reversed one), the caller's to keep or
        ///     change. A 0-d shape returns its empty array and an uninitialized <c>default(Shape)</c> returns
        ///     <c>null</c>.</value>
        /// <remarks>
        ///     <para>
        ///     <b>Why a copy.</b> Shapes and arrays SHARE stride arrays, and the cached flags (C/F-contiguity, broadcast,
        ///     writeable) and every kernel's addressing come from them. Returning the shared array let
        ///     <c>x.Shape.Strides[0] = 1000</c> retarget <c>x</c>'s reads past its buffer with no check and with stale flags.
        ///     NumPy has no unchecked route: its <c>strides</c> setter validates the request (<c>PyArray_CheckStrides</c>).
        ///     Plan <c>docs/plans/view-shape-open-bugs.md</c>, bug 2c.
        ///     </para>
        ///     <para>
        ///     The copy costs one small allocation per read. Code inside NumSharp reads the internal <c>strides</c> field
        ///     instead; a caller that reads the whole array in a hot loop should hoist it.
        ///     </para>
        /// </remarks>
        public readonly long[] Strides
        {
            [MethodImpl(Inline)]
            get
            {
                // Same rule as Dimensions: the empty and the missing array are handed out as they are, anything else is
                // cloned so the shared array never leaves the struct.
                var st = strides;
                return st is null || st.Length == 0 ? st : (long[])st.Clone();
            }
        }

        /// <summary>
        ///     The linear size of this shape.
        /// </summary>
        public readonly long Size
        {
            [MethodImpl(Inline)]
            get => size;
        }

        /// <summary>
        ///     Base offset into storage (like NumPy's adjusted data pointer).
        ///     For non-view shapes this is 0. For sliced/broadcast shapes,
        ///     this will be computed at slice/broadcast time in future phases.
        /// </summary>
        public readonly long Offset
        {
            [MethodImpl(Inline)]
            get => offset;
        }

        /// <summary>
        ///     Size of the underlying buffer (NumPy-aligned architecture).
        ///     For non-view shapes, equals Size. For sliced/broadcast shapes,
        ///     this is the actual buffer size (not the view size).
        /// </summary>
        public readonly long BufferSize
        {
            [MethodImpl(Inline)]
            get => bufferSize > 0 ? bufferSize : size;
        }

#region Constructors

        /// <summary>
        ///     Creates a scalar shape (ndim=0, size=1).
        ///     Equivalent to NumPy's empty tuple shape ().
        /// </summary>
        /// <remarks>
        ///     In NumPy: np.array(42).shape == ()
        ///     In NumSharp: new Shape() creates a scalar shape.
        /// </remarks>
        public Shape()
        {
            this.dimensions = Array.Empty<long>();
            this.strides = Array.Empty<long>();
            this.offset = 0;
            this.bufferSize = 1;
            this.size = 1;
            this._hashCode = int.MinValue; // Scalar hash
            this.IsScalar = true;
            // Scalars are trivially both C- and F-contiguous
            this._flags = (int)(ArrayFlags.C_CONTIGUOUS | ArrayFlags.F_CONTIGUOUS | ArrayFlags.ALIGNED | ArrayFlags.WRITEABLE);
        }

        /// <summary>
        ///     Complete constructor for views/broadcasts (NumPy-aligned).
        ///     All parameters are set explicitly, flags computed from dims/strides.
        /// </summary>
        /// <param name="dims">Dimension sizes (not cloned - caller must provide fresh array)</param>
        /// <param name="strides">Stride values (not cloned - caller must provide fresh array)</param>
        /// <param name="offset">Offset into underlying buffer</param>
        /// <param name="bufferSize">Size of underlying buffer</param>
        internal Shape(long[] dims, long[] strides, long offset, long bufferSize)
        {
            this.dimensions = dims ?? Array.Empty<long>();
            this.strides = strides ?? Array.Empty<long>();
            this.offset = offset;
            this.bufferSize = bufferSize;

            (this.size, this._hashCode) = ComputeSizeAndHash(dims);
            this.IsScalar = size == 1 && (dims == null || dims.Length == 0);
            this._flags = ComputeFlagsStatic(dims, strides);
        }

        /// <summary>
        ///     Creates a shape with modified flags (for clearing WRITEABLE on broadcasts, or
        ///     setting/clearing OWNDATA at the <see cref="NumSharp.Backends.UnmanagedStorage"/> funnels).
        /// </summary>
        /// <remarks>
        ///     Dimensions and strides are unchanged, so <see cref="size"/> and <see cref="_hashCode"/>
        ///     are carried over verbatim through the no-walk constructor instead of being recomputed.
        /// </remarks>
        public Shape WithFlags(ArrayFlags flagsToSet = ArrayFlags.None, ArrayFlags flagsToClear = ArrayFlags.None)
        {
            int newFlags = (_flags | (int)flagsToSet) & ~(int)flagsToClear;
            return new Shape(dimensions, strides, offset, bufferSize, newFlags, size, _hashCode);
        }

        /// <summary>
        ///     Internal constructor with explicit flags (for WithFlags).
        /// </summary>
        private Shape(long[] dims, long[] strides, long offset, long bufferSize, int flags)
        {
            this.dimensions = dims;
            this.strides = strides;
            this.offset = offset;
            this.bufferSize = bufferSize;
            this._flags = flags;

            (this.size, this._hashCode) = ComputeSizeAndHash(dims);
            this.IsScalar = size == 1 && (dims == null || dims.Length == 0);
        }

        /// <summary>
        ///     Hot-path constructor for view kernels that already know the final
        ///     <paramref name="flags"/>, <paramref name="size"/>, and
        ///     <paramref name="hashCode"/>. Skips both <c>ComputeFlagsStatic</c>
        ///     (3 walks of dims/strides) and <c>ComputeSizeAndHash</c> (1 walk).
        ///     Caller must guarantee values are consistent with
        ///     <paramref name="dims"/> / <paramref name="strides"/>; this ctor
        ///     does no validation.
        /// </summary>
        /// <remarks>
        ///     Used by <c>np.split</c> / <c>np.array_split</c> where every
        ///     sub-array shares the parent strides and only <c>dims[axis]</c>
        ///     changes, so the parent's flags and size scale by an O(1) ratio.
        /// </remarks>
        internal Shape(long[] dims, long[] strides, long offset, long bufferSize, int flags, long size, int hashCode)
        {
            this.dimensions = dims;
            this.strides = strides;
            this.offset = offset;
            this.bufferSize = bufferSize;
            this._flags = flags;
            this.size = size;
            this._hashCode = hashCode;
            this.IsScalar = size == 1 && (dims == null || dims.Length == 0);
        }

        public Shape(Shape other)
        {
            if (other.IsEmpty)
            {
                this = default;
                return;
            }

            //this.layout = other.layout;
            this._hashCode = other._hashCode;
            this.size = other.size;
            this.bufferSize = other.bufferSize;
            this.dimensions = (long[])other.dimensions.Clone();
            this.strides = (long[])other.strides.Clone();
            this.offset = other.offset;
            this.IsScalar = other.IsScalar;
            this._flags = other._flags;
        }

        /// <summary>
        ///     A shape with explicit element <paramref name="strides"/> over COPIES of the caller's arrays, at offset 0 over
        ///     a buffer of exactly <c>size</c> elements. Changing either array afterwards never changes this shape or an array
        ///     built on it (NumPy copies every shape or strides sequence it is given; plan
        ///     <c>docs/plans/view-shape-open-bugs.md</c>, bug 2d).
        /// </summary>
        /// <param name="dims">The extent of every axis. Copied.</param>
        /// <param name="strides">The stride of every axis, in ELEMENTS (0 for a broadcast axis, negative for a reversed
        ///     one). Copied. Not validated against a buffer: an array that installs this shape must own enough memory.</param>
        /// <exception cref="ArgumentNullException"><paramref name="dims"/> or <paramref name="strides"/> is
        ///     <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The two arrays differ in length.</exception>
        /// <remarks>
        ///     The copy costs two small allocations. Code inside NumSharp that built both arrays itself hands them over with
        ///     the internal <see cref="Adopt(long[], long[])"/> instead.
        /// </remarks>
        public Shape(long[] dims, long[] strides)
        {
            // The checks run before anything is copied, so a bad argument costs no allocation.
            ValidateDimsAndStrides(dims, strides);
            this = new Shape(CopyOf(dims), CopyOf(strides), AdoptTag.Owned);
        }

        /// <summary>
        ///     A shape with explicit element <paramref name="strides"/> over COPIES of the caller's arrays, at offset 0, whose
        ///     buffer is <paramref name="originalShape"/>'s (its buffer size, or its size when the buffer size is unset): the
        ///     form a broadcast of <paramref name="originalShape"/> takes. Changing either array afterwards never changes this
        ///     shape (bug 2d).
        /// </summary>
        /// <param name="dims">The extent of every axis. Copied.</param>
        /// <param name="strides">The stride of every axis, in elements. Copied.</param>
        /// <param name="originalShape">The shape whose buffer this one addresses; only its buffer size is read.</param>
        /// <exception cref="ArgumentNullException"><paramref name="dims"/> or <paramref name="strides"/> is
        ///     <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The two arrays differ in length.</exception>
        public Shape(long[] dims, long[] strides, Shape originalShape)
        {
            ValidateDimsAndStrides(dims, strides);
            // For broadcast shapes, bufferSize is the original (pre-broadcast) size. The adopting view constructor
            // computes the size, hash and flags from the two copies exactly as this constructor did.
            this = new Shape(CopyOf(dims), CopyOf(strides), 0,
                originalShape.bufferSize > 0 ? originalShape.bufferSize : originalShape.size);
        }

        /// <summary>
        ///     A C-contiguous shape over a COPY of the caller's dimensions (NumPy's <c>np.empty(shape)</c> layout). Changing
        ///     <paramref name="dims"/> afterwards never changes this shape or an array built on it: NumPy copies the sequence
        ///     (<c>PyArray_IntpConverter</c>) and an array owns its dimensions (<c>PyArray_NewFromDescr_int</c>).
        /// </summary>
        /// <param name="dims">The extent of every axis; <c>null</c> or empty is the 0-d shape. Copied.</param>
        /// <remarks>
        ///     <para>
        ///     <b>Why a copy.</b> Shapes and arrays SHARE dimension arrays, so keeping the caller's array let
        ///     <c>long[] s = {3, 4}; var a = np.zeros(s); s[0] = 100;</c> turn <c>a</c> into <c>(100, 4)</c> over 12
        ///     elements: the cached size and the buffer stayed, so the new rows read past the buffer (plan
        ///     <c>docs/plans/view-shape-open-bugs.md</c>, bug 2d).
        ///     </para>
        ///     <para>
        ///     Inline values (<c>new Shape(3, 4)</c>) bind the <see cref="Shape(ReadOnlySpan{long})"/> overload, which
        ///     allocates the one array this shape keeps. An array argument costs one copy; code inside NumSharp that built
        ///     the array itself hands it over with the internal <see cref="Adopt(long[])"/> instead.
        ///     </para>
        /// </remarks>
        [MethodImpl(Optimize)]
        public Shape(params long[] dims) : this(CopyOf(dims), AdoptTag.Owned)
        {
        }

        /// <summary>
        ///     A C-contiguous shape over the given dimensions, copied into the one array the shape keeps. This is the
        ///     overload inline values bind (<c>new Shape(3, 4)</c>, <c>new Shape(n)</c>): the values arrive on the caller's
        ///     stack, so building the shape allocates its dimension array once, as the array form did before it copied.
        /// </summary>
        /// <param name="dims">The extent of every axis; empty is the 0-d shape.</param>
        [MethodImpl(Optimize)]
        public Shape(params ReadOnlySpan<long> dims) : this(dims.IsEmpty ? Array.Empty<long>() : dims.ToArray(), AdoptTag.Owned)
        {
        }

        /// <summary>
        ///     Marks the private constructors that take OWNERSHIP of the arrays they are given (no copy), so their
        ///     signatures cannot collide with the public constructors, which copy.
        /// </summary>
        private enum AdoptTag : byte
        {
            /// <summary>The arrays are the new shape's own: nothing else holds them, or nothing writes into them.</summary>
            Owned
        }

        /// <summary>
        ///     The adopting core of the C-contiguous constructors: <paramref name="ownedDims"/> becomes this shape's
        ///     dimension array as it is.
        /// </summary>
        /// <param name="ownedDims">The dimensions, owned from now on by this shape (never <c>null</c>).</param>
        /// <param name="tag">Disambiguates the overload; always <see cref="AdoptTag.Owned"/>.</param>
        [MethodImpl(Optimize)]
        private Shape(long[] ownedDims, AdoptTag tag)
        {
            this.dimensions = ownedDims;
            this.strides = ComputeContiguousStrides(ownedDims);
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(ownedDims);
            this.bufferSize = size;
            this.IsScalar = _hashCode == int.MinValue;
            this._flags = ComputeFlagsStatic(ownedDims, strides);
        }

        /// <summary>
        ///     The adopting core of the explicit-strides constructor: both arrays become this shape's as they are, at offset
        ///     0 over a buffer of exactly <c>size</c> elements.
        /// </summary>
        /// <param name="ownedDims">The dimensions, owned from now on by this shape (already validated).</param>
        /// <param name="ownedStrides">The element strides, owned from now on by this shape (same length).</param>
        /// <param name="tag">Disambiguates the overload; always <see cref="AdoptTag.Owned"/>.</param>
        private Shape(long[] ownedDims, long[] ownedStrides, AdoptTag tag)
        {
            this.dimensions = ownedDims;
            this.strides = ownedStrides;
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(ownedDims);
            this.bufferSize = size;
            this.IsScalar = size == 1 && ownedDims.Length == 0;
            this._flags = ComputeFlagsStatic(ownedDims, ownedStrides);
        }

        /// <summary>
        ///     The adopting core of the ordered constructor: <paramref name="ownedDims"/> becomes this shape's dimension
        ///     array as it is, with C or F strides.
        /// </summary>
        /// <param name="order">'C' or 'F', already validated.</param>
        /// <param name="ownedDims">The dimensions, owned from now on by this shape (never <c>null</c>).</param>
        /// <param name="tag">Disambiguates the overload; always <see cref="AdoptTag.Owned"/>.</param>
        [MethodImpl(Optimize)]
        private Shape(char order, long[] ownedDims, AdoptTag tag)
        {
            this.dimensions = ownedDims;
            this.strides = order == 'F'
                ? ComputeFContiguousStrides(ownedDims)
                : ComputeContiguousStrides(ownedDims);
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(ownedDims);
            this.bufferSize = size;
            this.IsScalar = _hashCode == int.MinValue;
            this._flags = ComputeFlagsStatic(ownedDims, strides);
        }

        /// <summary>
        ///     A C-contiguous shape that ADOPTS <paramref name="dims"/>: the array becomes the shape's own, with no copy.
        ///     NumSharp's internal counterpart of <see cref="Shape(long[])"/> for an array it just built.
        /// </summary>
        /// <param name="dims">An array nothing else will write into (a fresh one, or another shape's, which is
        ///     immutable); <c>null</c> is the 0-d shape.</param>
        /// <returns>The shape, sharing <paramref name="dims"/>.</returns>
        /// <remarks>
        ///     Never pass a caller's array (a public parameter): the caller could change it afterwards and the shape, its
        ///     cached size and every array built on it would disagree (bug 2d). Public entry points copy through the public
        ///     constructors.
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static Shape Adopt(long[] dims) => new Shape(dims ?? Array.Empty<long>(), AdoptTag.Owned);

        /// <summary>
        ///     A C- or F-contiguous shape that ADOPTS <paramref name="dims"/> (no copy). NumSharp's internal counterpart of
        ///     <see cref="Shape(long[], char)"/> for an array it just built.
        /// </summary>
        /// <param name="dims">An array nothing else will write into; <c>null</c> is the 0-d shape.</param>
        /// <param name="order">Physical memory order: 'C' or 'F'.</param>
        /// <returns>The shape, sharing <paramref name="dims"/>.</returns>
        /// <exception cref="ArgumentException"><paramref name="order"/> is not 'C' or 'F'.</exception>
        /// <remarks>Never pass a caller's array; see <see cref="Adopt(long[])"/>.</remarks>
        [MethodImpl(OptimizeAndInline)]
        internal static Shape Adopt(long[] dims, char order) => new Shape(ValidPhysicalOrder(order), dims ?? Array.Empty<long>(), AdoptTag.Owned);

        /// <summary>
        ///     A shape with explicit element strides that ADOPTS both arrays (no copy), at offset 0 over a buffer of exactly
        ///     <c>size</c> elements. NumSharp's internal counterpart of <see cref="Shape(long[], long[])"/> for arrays it
        ///     just built.
        /// </summary>
        /// <param name="dims">The dimensions; nothing else may write into the array.</param>
        /// <param name="strides">The element strides; nothing else may write into the array.</param>
        /// <returns>The shape, sharing both arrays.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dims"/> or <paramref name="strides"/> is
        ///     <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The two arrays differ in length.</exception>
        /// <remarks>Never pass a caller's array; see <see cref="Adopt(long[])"/>.</remarks>
        internal static Shape Adopt(long[] dims, long[] strides)
        {
            ValidateDimsAndStrides(dims, strides);
            return new Shape(dims, strides, AdoptTag.Owned);
        }

        /// <summary>
        ///     The caller's dimensions as an array the new shape may own: a clone, or the shared empty array for
        ///     <c>null</c> and empty input (a 0-d shape has nothing to write into).
        /// </summary>
        /// <param name="source">The caller's array, or <c>null</c>.</param>
        /// <returns>An array nothing but the new shape holds.</returns>
        [MethodImpl(Inline)]
        private static long[] CopyOf(long[] source) =>
            source is null || source.Length == 0 ? Array.Empty<long>() : (long[])source.Clone();

        /// <summary>
        ///     The argument checks of the explicit-strides constructors, run before anything is copied.
        /// </summary>
        /// <param name="dims">The dimensions to check.</param>
        /// <param name="strides">The strides to check.</param>
        /// <exception cref="ArgumentNullException">Either array is <c>null</c>.</exception>
        /// <exception cref="ArgumentException">The two arrays differ in length.</exception>
        private static void ValidateDimsAndStrides(long[] dims, long[] strides)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (strides == null)
                throw new ArgumentNullException(nameof(strides));

            if (dims.Length != strides.Length)
                throw new ArgumentException($"While trying to construct a shape, given dimensions and strides does not match size ({dims.Length} != {strides.Length})");
        }

        /// <summary>
        ///     The physical-order check of the ordered constructors: only 'C' and 'F' name a layout here.
        /// </summary>
        /// <param name="order">The requested order.</param>
        /// <returns><paramref name="order"/>, unchanged.</returns>
        /// <exception cref="ArgumentException"><paramref name="order"/> is not 'C' or 'F' (a logical order such as 'A' or
        ///     'K' must be resolved first, through <c>OrderResolver</c>).</exception>
        private static char ValidPhysicalOrder(char order)
        {
            if (order != 'C' && order != 'F')
                throw new ArgumentException(
                    $"Physical order must be 'C' or 'F' (got '{order}'). Use OrderResolver to resolve 'A' or 'K'.",
                    nameof(order));
            return order;
        }

        /// <summary>
        ///     Primary constructor with long dimensions.
        /// </summary>
        [MethodImpl(Optimize)]
        public Shape(IEnumerable<long> dims)
        {
            var dimsArray = dims?.ToArray() ?? Array.Empty<long>();

            this.dimensions = dimsArray;
            this.strides = ComputeContiguousStrides(dimsArray);
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(dimsArray);
            this.bufferSize = size;
            this.IsScalar = _hashCode == int.MinValue;
            this._flags = ComputeFlagsStatic(dimsArray, strides);
        }

        /// <summary>
        ///     Backward-compatible constructor with int dimensions.
        /// </summary>
        [MethodImpl(Optimize)]
        public Shape(int[] dims)
        {
            if (dims == null)
            {
                this.dimensions = Array.Empty<long>();
            }
            else
            {
                this.dimensions = new long[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    this.dimensions[i] = dims[i];
            }

            this.strides = ComputeContiguousStrides(this.dimensions);
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(this.dimensions);
            this.bufferSize = size;
            this.IsScalar = _hashCode == int.MinValue;
            this._flags = ComputeFlagsStatic(this.dimensions, this.strides);
        }

        /// <summary>
        ///     Constructs a Shape with a specified physical memory order over a COPY of the caller's dimensions.
        ///     Only 'C' (row-major) and 'F' (column-major) are valid — logical orders
        ///     ('A', 'K') must be resolved to a physical order first via OrderResolver. Changing
        ///     <paramref name="dims"/> afterwards never changes this shape (bug 2d).
        /// </summary>
        /// <param name="dims">Dimension sizes; <c>null</c> is the 0-d shape. Copied.</param>
        /// <param name="order">Physical memory order: 'C' or 'F'.</param>
        /// <exception cref="ArgumentException">Thrown if order is not 'C' or 'F' (checked before anything is
        ///     copied).</exception>
        /// <remarks>
        ///     Code inside NumSharp that built the array itself hands it over with the internal
        ///     <see cref="Adopt(long[], char)"/> instead of paying for the copy.
        /// </remarks>
        [MethodImpl(Optimize)]
        public Shape(long[] dims, char order) : this(ValidPhysicalOrder(order), CopyOf(dims), AdoptTag.Owned)
        {
        }

#endregion

        /// <summary>
        ///     An empty shape without any fields set (all dimensions are 0).
        /// </summary>
        /// <remarks>Used internally for building shapes that will be filled in.</remarks>
        [MethodImpl(OptimizeAndInline)]
        public static Shape Empty(int ndim)
        {
            // Create shape with zero dimensions and zero strides
            return new Shape(new long[ndim], new long[ndim], 0, 0);
        }

        /// <summary>
        ///     The extent of dimension <paramref name="dim"/> — NumPy's <c>a.shape[dim]</c>. Read-only: a
        ///     <see cref="Shape"/> is immutable, like the tuple NumPy hands out.
        /// </summary>
        /// <param name="dim">The axis; a negative axis counts from the end (<c>-1</c> is the last).</param>
        /// <returns>The number of elements along that axis.</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="dim"/> is outside <c>[-NDim, NDim)</c>.</exception>
        /// <remarks>
        ///     There is no setter, on purpose. Shapes and arrays SHARE dimension arrays (an allocation keeps the caller's
        ///     shape, an elementwise result keeps its input's), so a setter that wrote into the array changed every array
        ///     sharing it — and only its dimensions: the cached size and the buffer stayed, so the new rows read past the
        ///     buffer (<c>var s = new Shape(3, 4); var a = np.zeros(s); s[0] = 100;</c> made <c>a</c> <c>(100, 4)</c> over
        ///     12 elements). NumPy's arrays own their dimensions and hand out immutable tuples; build a new shape instead
        ///     (<c>new Shape(...)</c>, <c>reshape</c>).
        /// </remarks>
        public readonly long this[int dim]
        {
            [MethodImpl(Inline)]
            get => dimensions[dim < 0 ? dimensions.Length + dim : dim];
        }

        /// <summary>
        ///     Retrieve the transformed offset if the shape is non-contiguous,
        ///     otherwise returns <paramref name="offset"/>.
        /// </summary>
        /// <param name="offset">The offset within the bounds of <see cref="size"/>.</param>
        /// <returns>The transformed offset.</returns>
        /// <remarks>For contiguous shapes, returns offset directly. For non-contiguous, translates through coordinates.</remarks>
        [MethodImpl(Inline)]
        public readonly long TransformOffset(long offset)
        {
            // For contiguous shapes, direct return
            if (IsContiguous)
                return this.offset + offset;

            // Non-contiguous: translate through coordinates
            return GetOffset(GetCoordinates(offset));
        }

        /// <summary>
        ///     Get offset index out of coordinate indices.
        ///     NumPy-aligned: offset + sum(indices * strides)
        /// </summary>
        /// <param name="indices">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        [MethodImpl(OptimizeAndInline)]
        public readonly long GetOffset(params long[] indices)
        {
            // Scalar with single index: direct offset access
            if (dimensions.Length == 0)
                return offset + (indices.Length > 0 ? indices[0] : 0);

            // NumPy formula: data_ptr + sum(indices * strides)
            return GetOffsetSimple(indices);
        }

        /// <summary>
        ///     Backward-compatible GetOffset with int indices.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public readonly long GetOffset(int[] indices)
        {
            // Scalar with single index: direct offset access
            if (dimensions.Length == 0)
                return offset + (indices.Length > 0 ? indices[0] : 0);

            // NumPy formula: data_ptr + sum(indices * strides)
            long off = offset;
            unchecked
            {
                for (int i = 0; i < indices.Length; i++)
                    off += indices[i] * strides[i];
            }

            return off;
        }

        /// <summary>
        ///     Get offset index out of a single coordinate index (1D fast path).
        ///     NumPy-aligned: offset + stride[0] * index
        /// </summary>
        /// <param name="index">The 1D coordinate to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        [MethodImpl(OptimizeAndInline)]
        internal readonly long GetOffset_1D(long index)
        {
            // Scalar case: direct offset access
            if (dimensions.Length == 0)
                return offset + index;

            return offset + index * strides[0];
        }


        /// <summary>
        ///     NumPy-aligned offset calculation: offset + sum(indices * strides).
        ///     This is the core formula - offset is computed at slice/broadcast time,
        ///     strides include step factors, and stride=0 handles broadcasting.
        /// </summary>
        /// <param name="indices">The coordinates to turn into linear offset</param>
        /// <returns>The index in the memory block that refers to a specific value.</returns>
        [MethodImpl(Inline)]
        internal readonly long GetOffsetSimple(params long[] indices)
        {
            long off = offset;
            unchecked
            {
                for (int i = 0; i < indices.Length; i++)
                    off += indices[i] * strides[i];
            }

            return off;
        }

        /// <summary>
        ///     Simplified offset calculation for 1D access.
        /// </summary>
        [MethodImpl(Inline)]
        internal readonly long GetOffsetSimple(long index)
        {
            // Scalar case: direct offset access
            if (strides.Length == 0)
                return offset + index;
            return offset + index * strides[0];
        }

        /// <summary>
        ///     Simplified offset calculation for 2D access.
        /// </summary>
        [MethodImpl(Inline)]
        internal readonly long GetOffsetSimple(long i, long j)
        {
            return offset + i * strides[0] + j * strides[1];
        }

        /// <summary>
        ///     Simplified offset calculation for 3D access.
        /// </summary>
        [MethodImpl(Inline)]
        internal readonly long GetOffsetSimple(long i, long j, long k)
        {
            return offset + i * strides[0] + j * strides[1] + k * strides[2];
        }


        /// <summary>
        ///     Gets the shape based on given <see cref="indicies"/> and the index offset (C-Contiguous) inside the current storage.
        /// </summary>
        /// <param name="indicies">The selection of indexes 0 based.</param>
        /// <returns></returns>
        /// <remarks>Used for slicing, returned shape is the new shape of the slice and offset is the offset from current address.</remarks>
        [MethodImpl(OptimizeAndInline)]
        public readonly (Shape Shape, long Offset) GetSubshape(params long[] indicies)
        {
            if (indicies.Length == 0)
                return (this, 0);

            long offset;
            var dim = indicies.Length;
            var newNDim = dimensions.Length - dim;
            if (IsBroadcasted)
            {
                indicies = (long[])indicies.Clone(); //we must copy because we make changes to it.

                // NumPy-aligned: compute unreduced shape on the fly
                // Unreduced shape has 1 for broadcast dimensions (stride=0)
                var unreducedDims = new long[NDim];
                for (int i = 0; i < NDim; i++)
                    unreducedDims[i] = strides[i] == 0 ? 1 : dimensions[i];

                // Unbroadcast indices (wrap around for broadcast dimensions)
                for (int i = 0; i < dim; i++)
                    indicies[i] = indicies[i] % unreducedDims[i];

                // Compute offset using strides (stride=0 means index doesn't affect offset)
                offset = this.offset;
                for (int i = 0; i < dim; i++)
                    offset += strides[i] * indicies[i];

                var retShape = new long[newNDim];
                var retStrides = new long[newNDim];
                for (int i = 0; i < newNDim; i++)
                {
                    retShape[i] = this.dimensions[dim + i];
                    retStrides[i] = this.strides[dim + i];
                }

                // Create result with bufferSize preserved (immutable constructor)
                long bufSize = this.bufferSize > 0 ? this.bufferSize : this.size;
                var result = new Shape(retShape, retStrides, offset, bufSize);
                return (result, offset);
            }

            //compute offset
            offset = GetOffset(indicies);

            // Use bufferSize for bounds checking (NumPy-aligned: no ViewInfo dependency)
            long boundSize = bufferSize > 0 ? bufferSize : size;
            if (offset >= boundSize && AddressesAnElement(dim))
                throw new IndexOutOfRangeException($"The offset {offset} is out of range in Shape {boundSize}");

            if (indicies.Length == dimensions.Length)
                return (Scalar, offset);

            //compute subshape
            var innerShape = new long[newNDim];
            for (int i = 0; i < innerShape.Length; i++)
                innerShape[i] = this.dimensions[dim + i];

            //TODO! This is not full support of sliced,
            //TODO! when sliced it usually diverts from this function but it would be better if we add support for sliced arrays too.
            return (Shape.Adopt(innerShape), offset);
        }

        /// <summary>
        ///     Does the sub-view left after consuming <paramref name="consumed"/> leading axes
        ///     actually address an element?
        /// </summary>
        /// <remarks>
        ///     The offset bounds-check above is a BUFFER backstop — the per-axis index validation
        ///     that matches NumPy (<c>index N is out of bounds for axis I with size D</c>) already
        ///     ran in <see cref="InferNegativeCoordinates(long[], long[])"/> before this method was
        ///     called. A sub-view with a zero-sized extent will never be dereferenced, so there is
        ///     no buffer to overrun and the backstop must not fire: <c>a[0]</c> on a <c>(2,3,0)</c>
        ///     is <c>(3,0)</c> in NumPy, and used to throw here because a zero-sized array's
        ///     bufferSize is 0 and so EVERY offset — the valid one included — compared "out of
        ///     range". The same holds for an empty window carved out of a live buffer, where the
        ///     offset can legitimately land one past the last element (<c>x[:, 2:2][4]</c>).
        ///     <para>
        ///     Only called once the check is already about to throw, so the ordinary path pays
        ///     nothing for it.
        ///     </para>
        /// </remarks>
        [MethodImpl(OptimizeAndInline)]
        private readonly bool AddressesAnElement(int consumed)
        {
            for (int i = consumed; i < dimensions.Length; i++)
                if (dimensions[i] == 0)
                    return false;
            return true;
        }

        /// <summary>
        ///     Backward-compatible GetSubshape with int indices.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public readonly (Shape Shape, long Offset) GetSubshape(int[] indicies)
        {
            if (indicies.Length == 0)
                return (this, 0);

            var longIndicies = new long[indicies.Length];
            for (int i = 0; i < indicies.Length; i++)
                longIndicies[i] = indicies[i];
            return GetSubshape(longIndicies);
        }

        /// <summary>
        ///  Gets coordinates in this shape from index in this shape (slicing is ignored).
        ///  Example: Shape (2,3)
        /// 0 => [0, 0]
        /// 1 => [0, 1]
        /// ...
        /// 6 => [1, 2]
        /// </summary>
        /// <param name="offset">the index if you would iterate from 0 to shape.size in row major order</param>
        /// <returns></returns>
        [MethodImpl(OptimizeAndInline)]
        public readonly long[] GetCoordinates(long offset)
        {
            // For non-contiguous shapes (transposed, stepped slices, broadcast), strides
            // don't match the standard C-contiguous pattern. Stride-based decomposition
            // doesn't work because the linear index can't be decomposed using memory strides.
            //
            // Use dimension-based decomposition instead, matching NumPy's PyArray_ITER_GOTO1D
            // which uses factors (product of trailing dimensions) rather than strides.
            // This correctly maps linear index 0..size-1 to logical coordinates regardless
            // of the actual memory layout.
            if (!IsContiguous)
            {
                var coords = new long[dimensions.Length];
                long remaining = offset;
                for (int i = 0; i < dimensions.Length; i++)
                {
                    long factor = 1;
                    for (int j = i + 1; j < dimensions.Length; j++)
                        factor *= dimensions[j];
                    coords[i] = remaining / factor;
                    remaining %= factor;
                }

                return coords;
            }

            long[] coords2 = null;

            if (strides.Length == 1)
                coords2 = new long[] { offset };

            long counter = offset;
            coords2 = new long[strides.Length];
            long stride;
            for (int i = 0; i < strides.Length; i++)
            {
                unchecked
                {
                    stride = strides[i];
                    if (stride == 0)
                    {
                        coords2[i] = 0;
                    }
                    else
                    {
                        coords2[i] = counter / stride;
                        counter -= coords2[i] * stride;
                    }
                }
            }

            return coords2;
        }

        [MethodImpl(OptimizeAndInline)]
        public static long GetSize(long[] dims)
        {
            long size = 1;
            unchecked
            {
                for (int i = 0; i < dims.Length; i++)
                    size *= dims[i];
            }

            return size;
        }

        /// <summary>
        ///     Backward-compatible GetSize with int dimensions.
        /// </summary>
        [MethodImpl(OptimizeAndInline)]
        public static long GetSize(int[] dims)
        {
            long size = 1;
            unchecked
            {
                for (int i = 0; i < dims.Length; i++)
                    size *= dims[i];
            }

            return size;
        }

        public static long[] GetAxis(ref Shape shape, int axis)
        {
            return GetAxis(shape.dimensions, axis);
        }

        public static long[] GetAxis(Shape shape, int axis)
        {
            return GetAxis(shape.dimensions, axis);
        }

        /// <summary>
        ///     The dimensions with one axis removed — the shape a reduction over <paramref name="axis"/> leaves.
        /// </summary>
        /// <param name="dims">The dimensions.</param>
        /// <param name="axis">The removed axis; negative values count from the end.</param>
        /// <returns>A new array without <paramref name="axis"/> (empty for 0-d dimensions, whatever the axis).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="dims"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for a 1-D or higher shape.</exception>
        /// <remarks>
        ///     A negative axis is NumPy's normalization (<c>-2</c> is the second-to-last axis). This helper used to map
        ///     EVERY negative axis to the last one and raise the legacy parenthesized AxisOutOfRangeException; its callers
        ///     pass normalized axes, so only a direct caller ever saw the difference.
        /// </remarks>
        public static long[] GetAxis(long[] dims, int axis)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (dims.Length == 0)
                return Array.Empty<long>();

            axis = Backends.DefaultEngine.check_and_adjust_axis(dims.Length, axis);
            return dims.RemoveAt(axis);
        }

        /// <summary>
        ///     Extracts the shape of given <paramref name="array"/>.
        /// </summary>
        /// <remarks>Supports both jagged and multi-dim.</remarks>
        [MethodImpl(Optimize)]
        public static int[] ExtractShape(Array array)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            bool isJagged = false;

            {
                var type = array.GetType();
                isJagged = array.Rank == 1 && type.IsArray && type.GetElementType().IsArray;
            }

            var l = new List<int>(16);
            if (isJagged)
            {
                // ReSharper disable once PossibleNullReferenceException
                Array arr = array;
                do
                {
                    l.Add(arr.Length);
                    arr = arr.GetValue(0) as Array;
                } while (arr != null && arr.GetType().IsArray);
            }
            else
            {
                //jagged or regular
                for (int dim = 0; dim < array.Rank; dim++)
                {
                    l.Add(array.GetLength(dim));
                }
            }

            return l.ToArray();
        }

#region Slicing support

        [MethodImpl(OptimizeAndInline)]
        public readonly Shape Slice(string slicing_notation) =>
            this.Slice(NumSharp.Slice.ParseSlices(slicing_notation));

        [MethodImpl(OptimizeAndInline)]
        public readonly Shape Slice(params Slice[] input_slices)
        {
            if (IsEmpty)
                throw new InvalidOperationException("Unable to slice an empty shape.");

            // NumPy-pure architecture: Each slice is independent - use PARENT, not ROOT.
            // No merging of slices. The offset and strides encode the full path.
            //
            // NumPy formula for slice element access:
            //   element[i] = data[offset + i * stride]
            // where offset = parent.offset + sum(parent.strides[d] * start[d])
            // and stride = parent.stride * step

            var sliced_axes = new List<long>();
            var sliced_strides_list = new List<long>();
            long sliceOffset = this.offset;

            for (int i = 0; i < NDim; i++)
            {
                var dim = dimensions[i];
                var slice = input_slices.Length > i ? input_slices[i] : NumSharp.Slice.All;
                var slice_def = slice.ToSliceDef(dim);

                // Add start offset: offset += parent.strides[i] * slice.Start
                sliceOffset += strides[i] * slice_def.Start;

                if (slice_def.IsIndex)
                {
                    // Index reduces dimension - skip this axis in output
                    continue;
                }

                // Non-index slice: add to output dimensions and strides
                long count = Math.Abs(slice_def.Count);
                sliced_axes.Add(count);

                // new_stride = parent.stride * step
                // Negative step produces negative stride (for reversed iteration)
                sliced_strides_list.Add(strides[i] * slice_def.Step);
            }

            // Preserve bufferSize from parent (or compute from parent.size if not set)
            long parentBufferSize = bufferSize > 0 ? bufferSize : size;

            if (sliced_axes.Count == 0) // Result is a scalar
            {
                // Create scalar via constructor with offset/bufferSize
                var scalar = new Shape(Array.Empty<long>(), Array.Empty<long>(), sliceOffset, parentBufferSize);
                // Inherit WRITEABLE from parent
                if (!IsWriteable)
                    return scalar.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
                return scalar;
            }

            var sliced_dims = sliced_axes.ToArray();
            var sliced_strides = sliced_strides_list.ToArray();

            // Create slice result via constructor
            var result = new Shape(sliced_dims, sliced_strides, sliceOffset, parentBufferSize);
            // Inherit WRITEABLE from parent
            if (!IsWriteable)
                return result.WithFlags(flagsToClear: ArrayFlags.WRITEABLE);
            return result;
        }

#endregion

#region Implicit Operators

        public static explicit operator long[](Shape shape) =>
            (long[])shape.dimensions.Clone(); //we clone to avoid any changes

        public static explicit operator int[](Shape shape)
        {
            var result = new int[shape.dimensions.Length];
            for (int i = 0; i < shape.dimensions.Length; i++)
            {
                if (shape.dimensions[i] > int.MaxValue)
                    throw new OverflowException($"Dimension {i} value {shape.dimensions[i]} exceeds int.MaxValue");
                result[i] = (int)shape.dimensions[i];
            }

            return result;
        }

        public static implicit operator Shape?(long[]? shape)
            => shape is not null ? new Shape(shape) : (Shape?)null;

        public static implicit operator Shape?(int[]? shape)
            => shape is not null ? new Shape(shape) : (Shape?)null;

        public static implicit operator Shape(long[] dims) =>
            new Shape(dims);

        public static implicit operator Shape(int[] dims) =>
            new Shape(dims);

        public static explicit operator long[](Shape? shape) =>
            shape.HasValue ? (long[])shape.Value : null;

        public static explicit operator int[](Shape? shape)
            => shape.HasValue ? (int[])shape.Value : null;

        public static explicit operator long(Shape shape) =>
            shape.Size;

        public static explicit operator int(Shape shape)
        {
            if (shape.Size > int.MaxValue)
                throw new OverflowException($"Shape size {shape.Size} exceeds int.MaxValue");
            return (int)shape.Size;
        }

        public static implicit operator Shape(long dim) =>
            Shape.Vector(dim);

        public static implicit operator Shape(int dim) =>
            Shape.Vector(dim);

        public static explicit operator (long, long)(Shape shape) =>
            shape.dimensions.Length == 2 ? (shape.dimensions[0], shape.dimensions[1]) : (0, 0);

        public static implicit operator Shape((long, long) dims) =>
            Shape.Matrix(dims.Item1, dims.Item2);

        public static implicit operator Shape((int, int) dims) =>
            Shape.Matrix(dims.Item1, dims.Item2);

        public static explicit operator (long, long, long)(Shape shape) =>
            shape.dimensions.Length == 3 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2]) : (0, 0, 0);

        public static implicit operator Shape((long, long, long) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static implicit operator Shape((int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3);

        public static explicit operator (long, long, long, long)(Shape shape) =>
            shape.dimensions.Length == 4 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3]) : (0, 0, 0, 0);

        public static implicit operator Shape((long, long, long, long) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public static implicit operator Shape((int, int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4);

        public static explicit operator (long, long, long, long, long)(Shape shape) =>
            shape.dimensions.Length == 5 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4]) : (0, 0, 0, 0, 0);

        public static implicit operator Shape((long, long, long, long, long) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5);

        public static implicit operator Shape((int, int, int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5);

        public static explicit operator (long, long, long, long, long, long)(Shape shape) =>
            shape.dimensions.Length == 6 ? (shape.dimensions[0], shape.dimensions[1], shape.dimensions[2], shape.dimensions[3], shape.dimensions[4], shape.dimensions[5]) : (0, 0, 0, 0, 0, 0);

        public static implicit operator Shape((long, long, long, long, long, long) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6);

        public static implicit operator Shape((int, int, int, int, int, int) dims) =>
            new Shape(dims.Item1, dims.Item2, dims.Item3, dims.Item4, dims.Item5, dims.Item6);

#endregion

#region Deconstructor

        public readonly void Deconstruct(out long dim1, out long dim2)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
        }

        public readonly void Deconstruct(out long dim1, out long dim2, out long dim3)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
        }

        public readonly void Deconstruct(out long dim1, out long dim2, out long dim3, out long dim4)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
        }

        public readonly void Deconstruct(out long dim1, out long dim2, out long dim3, out long dim4, out long dim5)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
            dim5 = dims[4];
        }

        public readonly void Deconstruct(out long dim1, out long dim2, out long dim3, out long dim4, out long dim5, out long dim6)
        {
            var dims = this.dimensions;
            dim1 = dims[0];
            dim2 = dims[1];
            dim3 = dims[2];
            dim4 = dims[3];
            dim5 = dims[4];
            dim6 = dims[5];
        }

#endregion

#region Equality

        public static bool operator ==(Shape a, Shape b)
        {
            if (a.IsEmpty && b.IsEmpty)
                return true;

            if (a.IsEmpty || b.IsEmpty)
                return false;

            if (a.size != b.size || a.NDim != b.NDim)
                return false;

            var dim = a.NDim;
            for (int i = 0; i < dim; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !(a == b);
        }

        public override readonly bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj))
            {
                return false;
            }

            if (obj.GetType() != this.GetType())
            {
                return false;
            }

            return Equals((Shape)obj);
        }

        /// <summary>Indicates whether the current object is equal to another object of the same type.</summary>
        /// <param name="other">An object to compare with this object.</param>
        /// <returns>true if the current object is equal to the <paramref name="other">other</paramref> parameter; otherwise, false.</returns>
        public readonly bool Equals(Shape other)
        {
            if ((_hashCode == 0 && _hashCode == other._hashCode) || dimensions == null && other.dimensions == null) //they are empty.
                return true;

            if ((dimensions == null && other.dimensions != null) || (dimensions != null && other.dimensions == null)) //they are empty.
                return false;

            if (size != other.size /*|| layout != other.layout*/ || dimensions.Length != other.dimensions.Length)
                return false;

            // ReSharper disable once LoopCanBeConvertedToQuery
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] != other.dimensions[i])
                {
                    return false;
                }
            }

            return true;
        }

        /// <summary>Serves as the default hash function.</summary>
        /// <returns>A hash code for the current object.</returns>
        public override int GetHashCode()
        {
            // ReSharper disable once NonReadonlyMemberInGetHashCode
            return _hashCode;
        }

#endregion

        /// <summary>
        ///     Translates coordinates with negative indices, e.g:<br></br>
        ///     np.arange(9)[-1] == np.arange(9)[8]<br></br>
        ///     np.arange(9)[-2] == np.arange(9)[7]<br></br>
        /// </summary>
        /// <param name="dimensions">The dimensions these coordinates are targeting</param>
        /// <param name="coords">The coordinates.</param>
        /// <returns>Coordinates without negative indices.</returns>
        [SuppressMessage("ReSharper", "ParameterHidesMember"), MethodImpl(Optimize)]
        public static long[] InferNegativeCoordinates(long[] dimensions, long[] coords)
        {
            // NumPy validates EACH integer coordinate against its own axis: a valid index
            // lies in [-dim, dim-1] (ranges clamp, but an integer index raises). The previous
            // code only wrapped negatives, leaving per-axis OOB to slip through whenever the
            // resulting flat offset still landed inside the buffer (e.g. a[0,4] returned a[1,0]).
            if (coords.Length > dimensions.Length)
                throw new IndexError($"too many indices for array: array is {dimensions.Length}-dimensional, but {coords.Length} were indexed");
            for (int i = 0; i < coords.Length; i++)
            {
                var curr = coords[i];
                var dim = dimensions[i];
                if (curr < -dim || curr >= dim)
                    throw new IndexError($"index {curr} is out of bounds for axis {i} with size {dim}");
                if (curr < 0)
                    coords[i] = dim + curr;
            }

            return coords;
        }

        /// <summary>
        ///     Backward-compatible InferNegativeCoordinates with int arrays.
        /// </summary>
        [SuppressMessage("ReSharper", "ParameterHidesMember"), MethodImpl(Optimize)]
        public static int[] InferNegativeCoordinates(long[] dimensions, int[] coords)
        {
            if (coords.Length > dimensions.Length)
                throw new IndexError($"too many indices for array: array is {dimensions.Length}-dimensional, but {coords.Length} were indexed");
            for (int i = 0; i < coords.Length; i++)
            {
                var curr = coords[i];
                var dim = dimensions[i];
                if (curr < -dim || curr >= dim)
                    throw new IndexError($"index {curr} is out of bounds for axis {i} with size {dim}");
                if (curr < 0)
                    coords[i] = (int)(dim + curr);
            }

            return coords;
        }

        /// <summary>
        ///     InferNegativeCoordinates via int pointer (backward-compatible, for internal use).
        ///     See Shape.Unmanaged.cs for the long* version.
        /// </summary>
        [SuppressMessage("ReSharper", "ParameterHidesMember"), MethodImpl(Optimize)]
        public static unsafe void InferNegativeCoordinates(long[] dimensions, int* coords, int ndims)
        {
            if (ndims > dimensions.Length)
                throw new IndexError($"too many indices for array: array is {dimensions.Length}-dimensional, but {ndims} were indexed");
            for (int i = 0; i < ndims; i++)
            {
                var curr = coords[i];
                var dim = dimensions[i];
                if (curr < -dim || curr >= dim)
                    throw new IndexError($"index {curr} is out of bounds for axis {i} with size {dim}");
                if (curr < 0)
                    coords[i] = (int)(dim + curr);
            }
        }

        public override string ToString() =>
            "(" + string.Join(", ", dimensions) + ")";

        /// <summary>
        ///     This shape written the way a Python tuple prints — no space after the comma, and a
        ///     TRAILING comma on a one-element shape (<c>(3,)</c>, not <c>(3)</c>).
        /// </summary>
        /// <remarks>
        ///     <see cref="ToString"/> renders for humans and is used all over the library, so it is
        ///     left alone; this is the spelling error messages need in order to read verbatim like
        ///     NumPy's — "shapes (2,3) and (3,2) not aligned", never "shapes (2, 3) and …".
        /// </remarks>
        internal readonly string ToPythonTuple()
        {
            if (dimensions == null || dimensions.Length == 0)
                return "()";
            if (dimensions.Length == 1)
                return "(" + dimensions[0] + ",)";
            return "(" + string.Join(",", dimensions) + ")";
        }

        /// <summary>Creates a new object that is a copy of the current instance.</summary>
        /// <returns>A new object that is a copy of this instance.</returns>
        readonly object ICloneable.Clone() =>
            Clone(true, false, false);

        /// <summary>
        ///     Creates a complete copy of this Shape.
        /// </summary>
        /// <param name="deep">Should make a complete deep clone or a shallow if false.</param>
        public readonly Shape Clone(bool deep = true, bool unview = false, bool unbroadcast = false)
        {
            if (IsEmpty)
                return default;

            if (IsScalar)
            {
                if (unview || unbroadcast)
                    return Scalar;

                return deep ? new Shape(this) : this;
            }

            if (!deep && !unview && !unbroadcast)
                return this; // readonly struct copy

            if (unview || unbroadcast)
                return Shape.Adopt((long[])this.dimensions.Clone());

            // Deep clone via copy constructor
            if (deep)
                return new Shape(this);

            return this;
        }

        /// <summary>
        ///     Returns a clean shape based on this (offset=0, standard strides).
        /// </summary>
        public readonly Shape Clean()
        {
            if (IsScalar)
                return NewScalar();

            return Shape.Adopt((long[])this.dimensions.Clone());
        }
    }
}
