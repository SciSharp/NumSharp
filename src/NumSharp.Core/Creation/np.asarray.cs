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
        ///     Convert the input to an array. If the input is already an <see cref="NDArray"/>,
        ///     it is returned as-is when no <paramref name="dtype"/> is requested, or converted
        ///     to the target dtype otherwise. Mirrors <c>numpy.asarray(a, dtype=...)</c>.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.asarray.html</remarks>
        public static NDArray asarray(NDArray a, Type dtype = null)
        {
            if (ReferenceEquals(a, null))
                throw new ArgumentNullException(nameof(a));
            if (dtype == null || a.dtype == dtype)
                return a;
            return a.astype(dtype, true);
        }
    }
}
