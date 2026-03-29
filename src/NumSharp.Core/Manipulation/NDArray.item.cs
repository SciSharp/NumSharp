using System;

namespace NumSharp
{
    public partial class NDArray
    {
        #region item() - size-1 arrays

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <returns>A copy of the specified element of the array as a suitable Python scalar.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        ///
        /// When called without arguments, works only for arrays with one element (size 1),
        /// which can have any shape (0-d, 1-element 1-d, 1x1 2-d, etc.).
        ///
        /// This is the NumPy 2.x replacement for the deprecated np.asscalar().
        /// </remarks>
        /// <exception cref="IncorrectSizeException">If array size is not 1.</exception>
        public object item()
        {
            if (size != 1)
                throw new IncorrectSizeException($"Can only convert an array of size 1 to a scalar, but array has size {size}.");
            return Storage.GetAtIndex(0);
        }

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        /// <exception cref="IncorrectSizeException">If array size is not 1.</exception>
        public T item<T>() where T : unmanaged
        {
            if (size != 1)
                throw new IncorrectSizeException($"Can only convert an array of size 1 to a scalar, but array has size {size}.");
            return GetItemAs<T>(Storage.GetAtIndex(0));
        }

        #endregion

        #region item(index) - flat indexing

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <param name="index">Flat index of element to extract (supports negative indexing).</param>
        /// <returns>A copy of the specified element of the array as a suitable Python scalar.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        /// </remarks>
        public object item(long index)
        {
            var arraySize = size;
            if (index < 0)
                index = arraySize + index;
            if (index < 0 || index >= arraySize)
                throw new IndexOutOfRangeException($"Index {index} is out of bounds for array with size {arraySize}.");
            return Storage.GetAtIndex(index);
        }

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <param name="index">Flat index of element to extract (supports negative indexing).</param>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        public T item<T>(long index) where T : unmanaged
        {
            return GetItemAs<T>(item(index));
        }

        #endregion

        #region item(i, j) - 2D indexing

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <param name="i">Index along first dimension.</param>
        /// <param name="j">Index along second dimension.</param>
        /// <returns>A copy of the specified element of the array as a suitable Python scalar.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        /// </remarks>
        public object item(long i, long j) => Storage.GetValue(new[] { i, j });

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <param name="i">Index along first dimension.</param>
        /// <param name="j">Index along second dimension.</param>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        public T item<T>(long i, long j) where T : unmanaged => GetItemAs<T>(item(i, j));

        #endregion

        #region item(i, j, k) - 3D indexing

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <param name="i">Index along first dimension.</param>
        /// <param name="j">Index along second dimension.</param>
        /// <param name="k">Index along third dimension.</param>
        /// <returns>A copy of the specified element of the array as a suitable Python scalar.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        /// </remarks>
        public object item(long i, long j, long k) => Storage.GetValue(new[] { i, j, k });

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <param name="i">Index along first dimension.</param>
        /// <param name="j">Index along second dimension.</param>
        /// <param name="k">Index along third dimension.</param>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        public T item<T>(long i, long j, long k) where T : unmanaged => GetItemAs<T>(item(i, j, k));

        #endregion

        #region item(indices) - N-D indexing

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <param name="indices">Indices of element to extract (one per dimension).</param>
        /// <returns>A copy of the specified element of the array as a suitable Python scalar.</returns>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        /// </remarks>
        public object item(params long[] indices)
        {
            if (indices.Length != ndim)
                throw new ArgumentException($"Incorrect number of indices for array with {ndim} dimensions.");
            return Storage.GetValue(indices);
        }

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <param name="indices">Indices of element to extract (one per dimension).</param>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        public T item<T>(params long[] indices) where T : unmanaged => GetItemAs<T>(item(indices));

        #endregion

        #region Helper

        private T GetItemAs<T>(object value) where T : unmanaged
        {
            if (value is T typed)
                return typed;
            return Utilities.Converts.ChangeType<T>(value);
        }

        #endregion
    }
}
