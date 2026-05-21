using System;

namespace NumSharp
{
    public static partial class np
    {
        public static NDArray asarray(string data)
        {
            var nd = new NDArray(typeof(string), Array.Empty<long>());
            nd.ReplaceData(new string[] {data});
            return nd;
        }

        public static NDArray asarray<T>(T data) where T : struct
        {
            var nd = new NDArray(typeof(T), Array.Empty<long>());
            nd.ReplaceData(new T[] {data});
            return nd;
        }

        public static NDArray asarray(string[] data, int ndim = 1)
        {
            var nd = new NDArray(typeof(string), new Shape(data.Length));
            nd.ReplaceData(data);
            return nd;
        }

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
        /// <param name="a">Input ndarray.</param>
        /// <param name="dtype">Requested dtype. <c>null</c> keeps the input dtype.</param>
        /// <param name="order">'C' (row-major), 'F' (column-major), 'A' (any contiguous), 'K' (keep — default). 'A'/'K' never force a copy on layout grounds.</param>
        /// <param name="copy">Tri-state: <c>null</c> = copy only if needed (default), <c>true</c> = always copy, <c>false</c> = never copy (raises if a copy would be required).</param>
        /// <param name="like">Reference array for array-function dispatch — accepted for NumPy parity but has no observable effect in NumSharp.</param>
        /// <param name="device">Target device. Only <c>"cpu"</c> and <c>null</c> are accepted.</param>
        /// <returns>NDArray with the requested dtype and memory layout. Returns <paramref name="a"/> when no copy is needed.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            NDArray a,
            Type dtype = null,
            char order = 'K',
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
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

            bool typeMatches = dtype == null || dtype == a.dtype;
            bool layoutMatches = LayoutAlreadyOK(a.Shape, order);
            bool noCopyNeeded = typeMatches && layoutMatches;

            if (copy == true)
            {
                char physical = OrderResolver.Resolve(order, a.Shape);
                if (!typeMatches)
                    return a.astype(dtype, copy: true, order: physical);
                return a.copy(physical);
            }

            if (copy == false && !noCopyNeeded)
                throw new ArgumentException(
                    "Unable to avoid copy while creating an array as requested.\n" +
                    "If using `np.array(obj, copy=False)` replace it with `np.asarray(obj)` " +
                    "to allow a copy when needed (no behavior change in NumPy 1.x).\n" +
                    "For more details, see " +
                    "https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword.",
                    nameof(copy));

            if (noCopyNeeded)
                return a;

            // copy == null && !noCopyNeeded: copy / cast as required.
            char physical2 = OrderResolver.Resolve(order, a.Shape);
            if (!typeMatches)
                return a.astype(dtype, copy: true, order: physical2);
            return a.copy(physical2);
        }

        /// <summary>
        ///     Convert the input to an ndarray. Convenience overload taking a NumPy-style dtype string
        ///     (e.g. <c>"float32"</c>, <c>"&lt;i4"</c>, <c>"complex128"</c>).
        /// </summary>
        /// <param name="a">Input ndarray.</param>
        /// <param name="dtype">NumPy-style dtype string (parsed via <see cref="np.dtype(string)"/>).</param>
        /// <param name="order">'C', 'F', 'A', or 'K' (default).</param>
        /// <param name="copy">Tri-state copy: <c>null</c> = if-needed, <c>true</c> = always, <c>false</c> = never (raises).</param>
        /// <param name="like">Reference for array-function dispatch — accepted for parity, no effect.</param>
        /// <param name="device">Only <c>"cpu"</c> or <c>null</c>.</param>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            NDArray a,
            string dtype,
            char order = 'K',
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            Type resolved = dtype == null ? null : np.dtype(dtype).type;
            return asarray(a, resolved, order, copy, like, device);
        }

        /// <summary>
        ///     Convert the input to an ndarray. Convenience overload taking a <see cref="DType"/> instance.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            NDArray a,
            DType dtype,
            char order = 'K',
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            Type resolved = dtype?.type;
            return asarray(a, resolved, order, copy, like, device);
        }

        /// <summary>
        ///     Convert the input to an ndarray. Convenience overload taking <see cref="NPTypeCode"/>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(
            NDArray a,
            NPTypeCode dtype,
            char order = 'K',
            bool? copy = null,
            NDArray like = null,
            string device = null)
        {
            Type resolved = dtype == NPTypeCode.Empty ? null : dtype.AsType();
            return asarray(a, resolved, order, copy, like, device);
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
                    throw new ArgumentException(
                        $"order must be one of 'C', 'F', 'A', 'K' (got '{order}')",
                        nameof(order));
            }
        }
    }
}
