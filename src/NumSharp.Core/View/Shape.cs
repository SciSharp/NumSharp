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

        /// <summary>Data is F-contiguous (column-major). Reserved, always false for NumSharp.</summary>
        F_CONTIGUOUS = 0x0002,

        /// <summary>Array owns its data buffer.</summary>
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
        ///     Dense data are stored contiguously in memory, addressed by a single index (the memory address). <br></br>
        ///     Array memory ordering schemes translate that single index into multiple indices corresponding to the array coordinates.<br></br>
        ///     0: Row major<br></br>
        ///     1: Column major
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
            int flags = 0;

            // Check BROADCASTED first
            bool isBroadcasted = ComputeIsBroadcastedStatic(dims, strides);
            if (isBroadcasted)
                flags |= (int)ArrayFlags.BROADCASTED;

            // Check C_CONTIGUOUS (depends on not being broadcasted)
            if (!isBroadcasted && ComputeIsContiguousStatic(dims, strides))
                flags |= (int)ArrayFlags.C_CONTIGUOUS;

            // Check F_CONTIGUOUS (depends on not being broadcasted)
            // Note: 1-D contiguous arrays are both C and F contiguous
            if (!isBroadcasted && ComputeIsFContiguousStatic(dims, strides))
                flags |= (int)ArrayFlags.F_CONTIGUOUS;

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
        /// </summary>
        [MethodImpl(Inline)]
        private static bool ComputeIsBroadcastedStatic(long[] dims, long[] strides)
        {
            if (strides == null || strides.Length == 0)
                return false;
            for (int i = 0; i < strides.Length; i++)
                if (strides[i] == 0 && dims[i] > 1)
                    return true;
            return false;
        }

        /// <summary>
        ///     Computes C-contiguity from stride values (NumPy algorithm).
        /// </summary>
        [MethodImpl(Inline)]
        private static bool ComputeIsContiguousStatic(long[] dims, long[] strides)
        {
            if (dims == null || dims.Length == 0)
                return true;

            long sd = 1;
            for (int i = dims.Length - 1; i >= 0; i--)
            {
                long dim = dims[i];
                if (dim == 0)
                    return true;
                if (dim != 1)
                {
                    if (strides[i] != sd)
                        return false;
                    sd *= dim;
                }
            }

            return true;
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
        ///     Computes F-contiguity from stride values (NumPy algorithm).
        ///     Scans left-to-right; stride[0] must equal 1, stride[i] = shape[i-1] * stride[i-1].
        /// </summary>
        [MethodImpl(Inline)]
        private static bool ComputeIsFContiguousStatic(long[] dims, long[] strides)
        {
            if (dims == null || dims.Length == 0)
                return true;

            long sd = 1;
            for (int i = 0; i < dims.Length; i++)
            {
                long dim = dims[i];
                if (dim == 0)
                    return true;
                if (dim != 1)
                {
                    if (strides[i] != sd)
                        return false;
                    sd *= dim;
                }
            }

            return true;
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
        ///     Does this array own its data buffer?
        ///     False for views (slices, transposes, broadcasts).
        ///     Cached flag set at storage level.
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

        public readonly char Order => layout;

        /// <summary>
        ///     Singleton instance of a <see cref="Shape"/> that represents a scalar.
        /// </summary>
        public static readonly Shape Scalar = new Shape(Array.Empty<long>());

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

        public readonly long[] Dimensions
        {
            [MethodImpl(Inline)]
            get => dimensions;
        }

        public readonly long[] Strides
        {
            [MethodImpl(Inline)]
            get => strides;
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
        ///     Creates a shape with modified flags (for clearing WRITEABLE on broadcasts).
        /// </summary>
        public Shape WithFlags(ArrayFlags flagsToSet = ArrayFlags.None, ArrayFlags flagsToClear = ArrayFlags.None)
        {
            int newFlags = (_flags | (int)flagsToSet) & ~(int)flagsToClear;
            return new Shape(dimensions, strides, offset, bufferSize, newFlags);
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

        public Shape(long[] dims, long[] strides)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (strides == null)
                throw new ArgumentNullException(nameof(strides));

            if (dims.Length != strides.Length)
                throw new ArgumentException($"While trying to construct a shape, given dimensions and strides does not match size ({dims.Length} != {strides.Length})");

            this.dimensions = dims;
            this.strides = strides;
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(dims);
            this.bufferSize = size;
            this.IsScalar = size == 1 && dims.Length == 0;
            this._flags = ComputeFlagsStatic(dims, strides);
        }

        public Shape(long[] dims, long[] strides, Shape originalShape)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (strides == null)
                throw new ArgumentNullException(nameof(strides));

            if (dims.Length != strides.Length)
                throw new ArgumentException($"While trying to construct a shape, given dimensions and strides does not match size ({dims.Length} != {strides.Length})");

            this.dimensions = dims;
            this.strides = strides;
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(dims);
            // For broadcast shapes, bufferSize is the original (pre-broadcast) size
            this.bufferSize = originalShape.bufferSize > 0 ? originalShape.bufferSize : originalShape.size;
            this.IsScalar = size == 1 && dims.Length == 0;
            this._flags = ComputeFlagsStatic(dims, strides);
        }

        /// <summary>
        ///     Primary constructor with long dimensions.
        /// </summary>
        [MethodImpl(Optimize)]
        public Shape(params long[] dims)
        {
            if (dims == null)
                dims = Array.Empty<long>();

            this.dimensions = dims;
            this.strides = ComputeContiguousStrides(dims);
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(dims);
            this.bufferSize = size;
            this.IsScalar = _hashCode == int.MinValue;
            this._flags = ComputeFlagsStatic(dims, strides);
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
        ///     Constructs a Shape with a specified physical memory order.
        ///     Only 'C' (row-major) and 'F' (column-major) are valid — logical orders
        ///     ('A', 'K') must be resolved to a physical order first via OrderResolver.
        /// </summary>
        /// <param name="dims">Dimension sizes.</param>
        /// <param name="order">Physical memory order: 'C' or 'F'.</param>
        /// <exception cref="ArgumentException">Thrown if order is not 'C' or 'F'.</exception>
        [MethodImpl(Optimize)]
        public Shape(long[] dims, char order)
        {
            if (order != 'C' && order != 'F')
                throw new ArgumentException(
                    $"Physical order must be 'C' or 'F' (got '{order}'). Use OrderResolver to resolve 'A' or 'K'.",
                    nameof(order));

            this.dimensions = dims ?? Array.Empty<long>();
            this.strides = order == 'F'
                ? ComputeFContiguousStrides(this.dimensions)
                : ComputeContiguousStrides(this.dimensions);
            this.offset = 0;

            (this.size, this._hashCode) = ComputeSizeAndHash(this.dimensions);
            this.bufferSize = size;
            this.IsScalar = _hashCode == int.MinValue;
            this._flags = ComputeFlagsStatic(this.dimensions, this.strides);
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

        public readonly long this[int dim]
        {
            [MethodImpl(Inline)]
            get => dimensions[dim < 0 ? dimensions.Length + dim : dim];
            [MethodImpl(Inline)]
            set => dimensions[dim < 0 ? dimensions.Length + dim : dim] = value;
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
            if (offset >= boundSize)
                throw new IndexOutOfRangeException($"The offset {offset} is out of range in Shape {boundSize}");

            if (indicies.Length == dimensions.Length)
                return (Scalar, offset);

            //compute subshape
            var innerShape = new long[newNDim];
            for (int i = 0; i < innerShape.Length; i++)
                innerShape[i] = this.dimensions[dim + i];

            //TODO! This is not full support of sliced,
            //TODO! when sliced it usually diverts from this function but it would be better if we add support for sliced arrays too.
            return (new Shape(innerShape), offset);
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

        public static long[] GetAxis(long[] dims, int axis)
        {
            if (dims == null)
                throw new ArgumentNullException(nameof(dims));

            if (dims.Length == 0)
                return Array.Empty<long>();

            if (axis <= -1) axis = dims.Length - 1;
            if (axis >= dims.Length)
                throw new AxisOutOfRangeException(dims.Length, axis);

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
            for (int i = 0; i < coords.Length; i++)
            {
                var curr = coords[i];
                if (curr < 0)
                    coords[i] = dimensions[i] + curr;
            }

            return coords;
        }

        /// <summary>
        ///     Backward-compatible InferNegativeCoordinates with int arrays.
        /// </summary>
        [SuppressMessage("ReSharper", "ParameterHidesMember"), MethodImpl(Optimize)]
        public static int[] InferNegativeCoordinates(long[] dimensions, int[] coords)
        {
            for (int i = 0; i < coords.Length; i++)
            {
                var curr = coords[i];
                if (curr < 0)
                    coords[i] = (int)(dimensions[i] + curr);
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
            for (int i = 0; i < ndims; i++)
            {
                var curr = coords[i];
                if (curr < 0)
                    coords[i] = (int)(dimensions[i] + curr);
            }
        }

        public override string ToString() =>
            "(" + string.Join(", ", dimensions) + ")";

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
                if (unbroadcast || !IsBroadcasted)
                    return Scalar;
                // Scalar broadcast: return scalar with same offset via constructor
                return new Shape(Array.Empty<long>(), Array.Empty<long>(), offset, bufferSize);
            }

            if (deep && unview && unbroadcast)
                return new Shape((long[])this.dimensions.Clone());

            if (!deep && !unview && !unbroadcast)
                return this; // readonly struct copy

            // Deep clone via copy constructor
            if (deep && !unbroadcast)
                return new Shape(this);

            // Unbroadcast: create new shape with standard C-contiguous strides
            if (unbroadcast)
            {
                var newStrides = ComputeContiguousStrides(dimensions);
                return new Shape((long[])dimensions.Clone(), newStrides, 0, size);
            }

            return this;
        }

        /// <summary>
        ///     Returns a clean shape based on this (offset=0, standard strides).
        /// </summary>
        public readonly Shape Clean()
        {
            if (IsScalar)
                return NewScalar();

            return new Shape((long[])this.dimensions.Clone());
        }
    }
}
