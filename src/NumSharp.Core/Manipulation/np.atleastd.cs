using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     View inputs as arrays with at least three dimensions.
        /// </summary>
        /// <param name="arys">One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have three or more dimensions are preserved.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 3. Copies are avoided where possible, and views with three or more dimensions are returned. For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1), and a 2-D array of shape (M, N) becomes a view of shape (M, N, 1).</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_3d.html</remarks>
        public static NDArray atleast_3d(object arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            var arr = asanyarray(arys, null);
            switch (arr.ndim)
            {
                case 0:
                    return arr.reshape(1, 1, 1);
                case 1:
                    return np.expand_dims(np.expand_dims(arr, 1), 0);
                case 2:
                    return np.expand_dims(arr, 2);
                default:
                    return arr;
            }
        }        
        
        /// <summary>
        ///     View inputs as arrays with at least three dimensions.
        /// </summary>
        /// <returns>An array, or list of arrays, each with a.ndim >= 3. Copies are avoided where possible, and views with three or more dimensions are returned. For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1), and a 2-D array of shape (M, N) becomes a view of shape (M, N, 1).</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_3d.html</remarks>
        public static NDArray atleast_3d(NDArray arr)
        {
            if (arr == null)
                throw new ArgumentNullException(nameof(arr));

            switch (arr.ndim)
            {
                case 0:
                    return arr.reshape(1, 1, 1);
                case 1:
                    return np.expand_dims(np.expand_dims(arr, 1), 0);
                case 2:
                    return np.expand_dims(arr, 2);
                default:
                    return arr;
            }
        }

        /// <summary>
        ///     View inputs as arrays with at least three dimensions.
        /// </summary>
        /// <param name="arys">One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have three or more dimensions are preserved.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 3. Copies are avoided where possible, and views with three or more dimensions are returned. For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1), and a 2-D array of shape (M, N) becomes a view of shape (M, N, 1).</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_3d.html</remarks>
        public static NDArray[] atleast_3d(params object[] arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            if (arys.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(arys));

            var ret = new NDArray[arys.Length];
            for (var i = 0; i < arys.Length; i++)
            {
                var arr = asanyarray(arys[i], null);
                switch (arr.ndim)
                {
                    case 0:
                        ret[i] = arr.reshape(1, 1, 1);
                        continue;
                    case 1:
                        ret[i] = np.expand_dims(np.expand_dims(arr, 1), 0);
                        continue;
                    case 2:
                        ret[i] = np.expand_dims(arr, 2);
                        continue;
                    default:
                        ret[i] = arr;
                        continue;
                }
            }

            return ret;
        }

        /// <summary>
        ///     View inputs as arrays with at least three dimensions.
        /// </summary>
        /// <param name="arys">One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have three or more dimensions are preserved.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 3. Copies are avoided where possible, and views with three or more dimensions are returned. For example, a 1-D array of shape (N,) becomes a view of shape (1, N, 1), and a 2-D array of shape (M, N) becomes a view of shape (M, N, 1).</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_3d.html</remarks>
        public static NDArray[] atleast_3d(params NDArray[] arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            if (arys.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(arys));

            var ret = new NDArray[arys.Length];
            for (var i = 0; i < arys.Length; i++)
            {
                var arr = arys[i];
                switch (arr.ndim)
                {
                    case 0:
                        ret[i] = arr.reshape(1, 1, 1);
                        continue;
                    case 1:
                        ret[i] = np.expand_dims(np.expand_dims(arr, 1), 0);
                        continue;
                    case 2:
                        ret[i] = np.expand_dims(arr, 2);
                        continue;
                    default:
                        ret[i] = arr;
                        continue;
                }
            }

            return ret;
        }

        /// <summary>
        ///     View inputs as arrays with at least two dimensions.
        /// </summary>
        /// <param name="arys">One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have two or more dimensions are preserved.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 2. Copies are avoided where possible, and views with two or more dimensions are returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html</remarks>
        public static NDArray atleast_2d(object arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            var arr = asanyarray(arys, null);
            switch (arr.ndim)
            {
                case 0:
                    return arr.reshape(1, 1);
                case 1:
                    return np.expand_dims(arr, 0);
                default:
                    return arr;
            }
        }

        /// <summary>
        ///     View inputs as arrays with at least two dimensions.
        /// </summary>
        /// <param name="arr">One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have two or more dimensions are preserved.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 2. Copies are avoided where possible, and views with two or more dimensions are returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html</remarks>
        public static NDArray atleast_2d(NDArray arr)
        {
            if (arr == null)
                throw new ArgumentNullException(nameof(arr));

            switch (arr.ndim)
            {
                case 0:
                    return arr.reshape(1, 1);
                case 1:
                    return np.expand_dims(arr, 0);
                default:
                    return arr;
            }
        }

        /// <summary>
        ///     View inputs as arrays with at least two dimensions.
        /// </summary>
        /// <param name="arys">One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have two or more dimensions are preserved.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 2. Copies are avoided where possible, and views with two or more dimensions are returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html</remarks>
        public static NDArray[] atleast_2d(params object[] arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            if (arys.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(arys));

            var ret = new NDArray[arys.Length];
            for (var i = 0; i < arys.Length; i++)
            {
                var arr = asanyarray(arys[i], null);
                switch (arr.ndim)
                {
                    case 0:
                        ret[i] = arr.reshape(1, 1);
                        continue;
                    case 1:
                        ret[i] = np.expand_dims(arr, 0);
                        continue;
                    default:
                        ret[i] = arr;
                        continue;
                }
            }

            return ret;
        }
        /// <summary>
        ///     View inputs as arrays with at least two dimensions.
        /// </summary>
        /// <param name="arys">One or more array-like sequences. Non-array inputs are converted to arrays. Arrays that already have two or more dimensions are preserved.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 2. Copies are avoided where possible, and views with two or more dimensions are returned.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_2d.html</remarks>
        public static NDArray[] atleast_2d(params NDArray[] arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            if (arys.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(arys));

            var ret = new NDArray[arys.Length];
            for (var i = 0; i < arys.Length; i++)
            {
                var arr = arys[i];
                switch (arr.ndim)
                {
                    case 0:
                        ret[i] = arr.reshape(1, 1);
                        continue;
                    case 1:
                        ret[i] = np.expand_dims(arr, 0);
                        continue;
                    default:
                        ret[i] = arr;
                        continue;
                }
            }

            return ret;
        }

        /// <summary>
        ///     Convert inputs to arrays with at least one dimension.
        ///     Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
        /// </summary>
        /// <param name="arys">One or more input arrays.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 1. Copies are made only if necessary.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html</remarks>
        public static NDArray atleast_1d(object arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            var arr = asanyarray(arys, null);
            switch (arr.ndim)
            {
                case 0:
                    return arr.reshape(1);
                default:
                    return arr;
            }
        }        
        
        /// <summary>
        ///     Convert inputs to arrays with at least one dimension.
        ///     Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
        /// </summary>
        /// <returns>An array, or list of arrays, each with a.ndim >= 1. Copies are made only if necessary.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html</remarks>
        public static NDArray atleast_1d(NDArray arr)
        {
            if (arr == null)
                throw new ArgumentNullException(nameof(arr));

            switch (arr.ndim)
            {
                case 0:
                    return arr.reshape(1);
                default:
                    return arr;
            }
        }

        /// <summary>
        ///     Convert inputs to arrays with at least one dimension.
        ///     Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
        /// </summary>
        /// <param name="arys">One or more input arrays.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 1. Copies are made only if necessary.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html</remarks>
        public static NDArray[] atleast_1d(params object[] arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            if (arys.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(arys));

            var ret = new NDArray[arys.Length];
            for (var i = 0; i < arys.Length; i++)
            {
                var arr = asanyarray(arys[i], null);
                switch (arr.ndim)
                {
                    case 0:
                        ret[i] = arr.reshape(1);
                        continue;
                    default:
                        ret[i] = arr;
                        continue;
                }
            }

            return ret;
        }        
        
        /// <summary>
        ///     Convert inputs to arrays with at least one dimension.
        ///     Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
        /// </summary>
        /// <param name="arys">One or more input arrays.</param>
        /// <returns>An array, or list of arrays, each with a.ndim >= 1. Copies are made only if necessary.</returns>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.atleast_1d.html</remarks>
        public static NDArray[] atleast_1d(params NDArray[] arys)
        {
            if (arys == null)
                throw new ArgumentNullException(nameof(arys));

            if (arys.Length == 0)
                throw new ArgumentException("Value cannot be an empty collection.", nameof(arys));

            var ret = new NDArray[arys.Length];
            for (var i = 0; i < arys.Length; i++)
            {
                var arr = arys[i];
                switch (arr.ndim)
                {
                    case 0:
                        ret[i] = arr.reshape(1);
                        continue;
                    default:
                        ret[i] = arr;
                        continue;
                }
            }

            return ret;
        }
    }
}
