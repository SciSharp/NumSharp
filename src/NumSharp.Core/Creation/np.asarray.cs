using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     NumPy's <c>np.asarray('abc')</c>: a 0-d fixed-width Unicode array holding <paramref name="data"/> (see
        ///     <see cref="array(string)"/>).
        /// </summary>
        /// <param name="data">The text.</param>
        /// <returns>A 0-d <c>U</c> array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is null.</exception>
        public static NDArray asarray(string data) => array(data);

        /// <summary>
        ///     NumPy's <c>np.asarray(2)</c>: a new 0-d array holding one C# value. Like <c>np.array(2)</c>, the
        ///     result is an explicit 0-d ARRAY (<see cref="NDArray.IsExplicitZeroDimArray"/>): a matrix multiplied by
        ///     it takes NumPy's 1 x 1 matrix route instead of scaling.
        /// </summary>
        /// <typeparam name="T">The value's type; it must map to a NumSharp dtype.</typeparam>
        /// <param name="data">The value.</param>
        /// <returns>A new writeable 0-d array that owns its one element.</returns>
        /// <exception cref="NotSupportedException"><typeparamref name="T"/> has no NumSharp dtype.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray<T>(T data) where T : struct
        {
            var nd = new NDArray(typeof(T), Shape.Adopt(Array.Empty<long>()));
            nd.ReplaceData(new T[] {data});
            return nd.MarkExplicitZeroDimArray();
        }

        /// <summary>
        ///     NumPy's <c>np.asarray(['a', 'bb'])</c>: a 1-D fixed-width Unicode array (see <see cref="array(string[])"/>).
        /// </summary>
        /// <param name="data">The values.</param>
        /// <param name="ndim">Kept for source compatibility; a string vector is always 1-D.</param>
        /// <returns>The <c>U</c> array.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="data"/> is null.</exception>
        /// <exception cref="NotSupportedException">An entry is null (an <c>object</c> array in NumPy).</exception>
        public static NDArray asarray(string[] data, int ndim = 1) => array(data);

        public static NDArray asarray<T>(T[] data, int ndim = 1) where T : struct
        {
            var nd = new NDArray(typeof(T), new Shape(data.Length));
            nd.ReplaceData(data);
            return nd;
        }

        /// <summary>
        ///     Convert the input to an ndarray, matching NumPy 2.x semantics.
        ///     If <paramref name="a"/> already satisfies the requested dtype/layout, it is returned as-is — no copy.
        /// </summary>
        /// <param name="a">Input ndarray. A C# literal (<c>np.asarray(1000, np.int8)</c>, whose 0-d array is marked
        ///     <see cref="NDArray.WasPythonLiteral"/>) is NumPy's Python scalar: with a <paramref name="dtype"/> it is packed
        ///     with NumPy's range checks into a new array, not cast.</param>
        /// <param name="dtype">Requested dtype. <c>null</c> keeps the input dtype.</param>
        /// <param name="order">'C' (row-major), 'F' (column-major), 'A' (any contiguous), 'K' (keep — default). 'A'/'K' never force a copy on layout grounds.</param>
        /// <param name="copy">Tri-state: <c>null</c> = copy only if needed (default), <c>true</c> = always copy, <c>false</c> = never copy (raises if a copy would be required).</param>
        /// <param name="like">Reference array for array-function dispatch — accepted for NumPy parity but has no observable effect in NumSharp.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted.</param>
        /// <returns>NDArray with the requested dtype and memory layout. Returns <paramref name="a"/> when no copy is needed.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ArgumentException">An unsupported <paramref name="device"/>, or <paramref name="copy"/> is
        ///     <c>false</c> and a copy (or a C# literal's conversion) is required.</exception>
        /// <exception cref="OverflowException">A C# literal out of <paramref name="dtype"/>'s integer range (NumPy's
        ///     OverflowError texts).</exception>
        /// <exception cref="ValueError">A float NaN literal into an integer <paramref name="dtype"/>.</exception>
        /// <exception cref="TypeError">A complex literal into a real <paramref name="dtype"/>.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            NDArray a,
            DType dtype = null,
            char order = 'K',
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Plain((a, dtype, order, copy, like, device), static s => asarray(s.a, s.dtype, s.order, s.copy, s.like, s.device));
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            // Only "cpu" is supported (matches NumPy 2.x). null is treated as "default".
            if (device != null && !string.Equals(device, "cpu", StringComparison.Ordinal))
                throw new ArgumentException(
                    $"Device not understood. Only \"cpu\" is allowed, but received: {device}",
                    nameof(device));

            // `like` exists for NumPy's __array_function__ dispatch protocol. NumSharp has
            // no array-subclass dispatch, so the value is accepted but never read.
            _ = like;

            // A C# literal (NDArray.WasPythonLiteral) is NumPy's Python scalar, which np.asarray(value, dtype=...) converts
            // with PyArray_Pack into a NEW array: np.asarray(1000, dtype=np.int8) raises "Python integer 1000 out of bounds
            // for int8" and np.asarray(1.5, dtype=np.int8) is 1 — where a 0-d ndarray would be cast. A fresh array can never
            // satisfy copy=false (NumPy's "Unable to avoid copy" for any Python scalar).
            if (a.WasPythonLiteral && dtype is not null && !Backends.Strings.NDStringStorage.IsString(dtype))
            {
                var packed = NDArray.PackForStore(a, dtype.GetTypeCode(), pyArrayPackRoute: false);
                if (!ReferenceEquals(packed, a))
                {
                    if (copy == false)
                    {
                        packed.Dispose();
                        throw new ArgumentException(
                            "Unable to avoid copy while creating an array as requested.\n" +
                            "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` " +
                            "to allow a copy when needed (no behavior change in NumPy 1.x).\n" +
                            "For more details, see " +
                            "https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.",
                            nameof(copy));
                    }

                    return packed;
                }
            }

            // A string request is satisfied only by a string array of that class AND width (an unsized 'U' adapts to the
            // array first); the storage lane alone would let a <U4 array pass for 'S' or 'U2'.
            bool typeMatches = dtype == null || Backends.Strings.NDStringArrays.SatisfiesRequest(a, dtype);
            bool layoutMatches = LayoutAlreadyOK(a.Shape, order);
            bool noCopyNeeded = typeMatches && layoutMatches;

            // A copy or a cast of an explicit 0-d ARRAY is an explicit 0-d array again (NumPy returns a 0-d ndarray
            // from np.asarray(np.array(2), dtype=float)); the mark is per object, so it is carried over by hand.
            bool explicitZeroDim = a.IsExplicitZeroDimArray;

            // The copies keep NumPy's order code as given: astype / copy resolve 'K' to the source's own axis order
            // (a transposed or broadcast source keeps its layout), which resolving to C-or-F first would lose.
            if (copy == true)
            {
                NDArray copied = !typeMatches ? a.astype(dtype, copy: true, order: order) : a.copy(order);
                return explicitZeroDim ? copied.MarkExplicitZeroDimArray() : copied;
            }

            if (copy == false && !noCopyNeeded)
                throw new ArgumentException(
                    "Unable to avoid copy while creating an array as requested.\n" +
                    "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` " +
                    "to allow a copy when needed (no behavior change in NumPy 1.x).\n" +
                    "For more details, see " +
                    "https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.",
                    nameof(copy));

            // NumPy (_array_fromobject_generic) returns the array itself when no dtype was given or the requested descriptor
            // IS the array's — a builtin singleton, or an unsized string class, which adapts to the array's own descriptor;
            // an EQUAL descriptor that is another object (every sized string descriptor is a fresh object) gets a view of
            // the array with that descriptor: np.asarray(u, dtype='<U2') shares u's memory but is not u (owndata false).
            if (noCopyNeeded)
            {
                if (dtype is null || ReferenceEquals(dtype, a.dtype) || dtype.itemsize == 0)
                    return a;
                // The view is a new object, and the explicit-0-d mark is per object: NumPy's view of an explicit 0-d array
                // is an explicit 0-d array too, so the mark is carried over like on the copy paths.
                var view = new NDArray(a.Storage.Alias(a.Shape)) { TensorEngine = a.TensorEngine };
                return explicitZeroDim ? view.MarkExplicitZeroDimArray() : view;
            }

            // copy == null && !noCopyNeeded: copy / cast as required.
            NDArray converted = !typeMatches ? a.astype(dtype, copy: true, order: order) : a.copy(order);
            return explicitZeroDim ? converted.MarkExplicitZeroDimArray() : converted;
        }

        /// <summary>
        ///     Convert a <see cref="MemoryView"/> (obtained from <see cref="NDArray.data"/>) to an ndarray —
        ///     the consumer round-trip of <c>ndarray.data</c>. Matches NumPy's <c>np.asarray(a.data)</c>,
        ///     which reads the buffer's shape/strides/dtype and returns a zero-copy VIEW sharing its memory
        ///     (copying only if a dtype/layout change forces it), preserving the source's N-D shape and
        ///     layout. Equivalent to <c>np.asarray(buffer.obj, …)</c>.
        /// </summary>
        /// <param name="buffer">A <see cref="MemoryView"/> over an array.</param>
        /// <param name="dtype">Requested dtype. <c>null</c> keeps the source dtype.</param>
        /// <param name="order">'C', 'F', 'A', or 'K' (default).</param>
        /// <param name="copy">Tri-state copy: <c>null</c> = if-needed, <c>true</c> = always, <c>false</c> = never (raises).</param>
        /// <param name="like">Reference for array-function dispatch — accepted for parity, no effect.</param>
        /// <param name="device">Only <c>"cpu"</c> or <c>null</c>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            MemoryView buffer,
            DType dtype = null,
            char order = 'K',
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(like))
                return NDMatrixBoundary.Plain((buffer, dtype, order, copy, like, device), static s => asarray(s.buffer, s.dtype, s.order, s.copy, s.like, s.device));
            if (buffer is null)
                throw new ArgumentNullException(nameof(buffer));
            return asarray(buffer.obj, dtype, order, copy, like, device);
        }

        /// <summary>
        ///     Convert a <see cref="MemoryView"/> to an ndarray, taking a NumPy-style dtype string
        ///     (e.g. <c>"float32"</c>, <c>"&lt;i4"</c>). See
        ///     <see cref="asarray(MemoryView, Type, char, bool?, NDArray, string)"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            MemoryView buffer,
            string dtype,
            char order = 'K',
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            // a numpy.matrix operand: NumPy converts with asarray here, so the result is a plain array
            if (NDMatrixBoundary.Sees(like))
                return NDMatrixBoundary.Plain((buffer, dtype, order, copy, like, device), static s => asarray(s.buffer, s.dtype, s.order, s.copy, s.like, s.device));
            if (buffer is null)
                throw new ArgumentNullException(nameof(buffer));
            return asarray(buffer.obj, dtype, order, copy, like, device);
        }

        /// <summary>
        ///     Returns true when <paramref name="shape"/>'s memory layout already satisfies the requested
        ///     order ('C', 'F', 'A', 'K'). Mirrors NumPy's <c>STRIDING_OK</c> macro:
        ///     'A' and 'K' impose no layout constraint, so a copy is never forced for layout reasons.
        /// </summary>
        private static bool LayoutAlreadyOK(Shape shape, char order)
        {
            switch (order)
            {
                case 'C':
                case 'c':
                    return shape.IsContiguous;
                case 'F':
                case 'f':
                    return shape.IsFContiguous;
                case 'A':
                case 'a':
                case 'K':
                case 'k':
                    return true;
                default:
                    // NumPy's PyArray_OrderConverter text, verbatim.
                    throw new ValueError($"order must be one of 'C', 'F', 'A', or 'K' (got '{order}')");
            }
        }
    }
}
