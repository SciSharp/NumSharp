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
        ///     Convert the input to an array with a specified memory layout.
        ///     If the input is already an NDArray in the requested layout, it is returned as-is (no copy).
        /// </summary>
        /// <param name="a">Input NDArray.</param>
        /// <param name="dtype">By default, the data-type is inferred from the input.</param>
        /// <param name="order">'C' (row-major), 'F' (column-major), 'A' or 'K' (logical — resolved against a).</param>
        /// <returns>NDArray with the requested dtype and memory layout.</returns>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(NDArray a, Type dtype = null, char order = 'K')
        {
            if (a is null)
                throw new ArgumentNullException(nameof(a));

            char physical = OrderResolver.Resolve(order, a.Shape);
            bool typeMatches = dtype == null || dtype == a.dtype;
            bool layoutMatches = physical == 'C'
                ? a.Shape.IsContiguous
                : a.Shape.IsFContiguous;

            if (typeMatches && layoutMatches)
                return a;

            if (!typeMatches)
                return a.astype(dtype, copy: true, order: physical);
            return a.copy(physical);
        }
    }
}
