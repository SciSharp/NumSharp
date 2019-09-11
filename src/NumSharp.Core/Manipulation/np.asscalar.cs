using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <param name="nd">Input NDArray of size 1.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.asscalar.html</remarks>
        public static T asscalar<T>(NDArray nd) where T : unmanaged
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");
            var value = nd.Storage.GetAtIndex(0);
            if (nd.dtype != typeof(T))
                return (T)Converts.ChangeType(value, InfoOf<T>.NPTypeCode);
            return (T)value;
        }

        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <param name="arr">Input array of size 1.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.asscalar.html</remarks>
        public static T asscalar<T>(Array arr)
        {
            if (arr.Length != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");
            var value = arr.GetValue(0);
            if (value.GetType() != typeof(T))
                return (T)Converts.ChangeType(value, InfoOf<T>.NPTypeCode);
            return (T)value;
        }

        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <param name="nd">Input NDArray of size 1.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.asscalar.html</remarks>
        public static ValueType asscalar(NDArray nd)
        {
            if (nd.size != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");
            return nd.Storage.GetAtIndex(0);
        }

        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <param name="arr">Input array of size 1.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.asscalar.html</remarks>
        public static ValueType asscalar(Array arr)
        {
            if (arr.Length != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");
            return (ValueType) arr.GetValue(0);
        }

        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <param name="arr">Input array of size 1.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.asscalar.html</remarks>
        public static T asscalar<T>(ArraySlice<T> arr) where T : unmanaged
        {
            if (arr.Count != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");
            return arr[0];
        }

        /// <summary>
        ///     Convert an array of size 1 to its scalar equivalent.
        /// </summary>
        /// <param name="arr">Input array of size 1.</param>
        /// <returns></returns>
        /// <remarks>https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.asscalar.html</remarks>
        public static T asscalar<T>(IArraySlice arr) where T : unmanaged
        {
            if (arr.Count != 1)
                throw new IncorrectSizeException("Unable to convert NDArray to scalar because size is not 1.");

            return Converts.ChangeType<T>(arr[0]);
        }
    }
}
