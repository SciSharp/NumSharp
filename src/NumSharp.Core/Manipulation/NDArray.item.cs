using System;
using System.Globalization;

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
        /// <exception cref="ValueError">The array does not hold exactly one element (NumPy's <c>can only convert an array
        ///     of size 1 to a Python scalar</c>; an <see cref="IncorrectSizeException"/> until API-parity batch 13, parity
        ///     QA-04).</exception>
        public object item()
        {
            // array_item's no-argument form: PyArray_SIZE(self) == 1, else NumPy's ValueError.
            if (size != 1)
                throw new ValueError("can only convert an array of size 1 to a Python scalar");
            return Storage.GetAtIndex(0);
        }

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        /// <exception cref="ValueError">The array does not hold exactly one element (NumPy's <c>can only convert an array
        ///     of size 1 to a Python scalar</c>).</exception>
        public T item<T>() where T : unmanaged
        {
            // The same size test as item(): NumPy has one item, this is its typed read.
            if (size != 1)
                throw new ValueError("can only convert an array of size 1 to a Python scalar");
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
            long original = index;
            if (index < 0)
                index = arraySize + index;
            // NumPy's texts (probed 2.4.2): a 1-D array reports axis 0, any other rank only the flat size; both keep the
            // index as given (a negative one included).
            if (index < 0 || index >= arraySize)
                throw new IndexOutOfRangeException(ndim == 1
                    ? string.Create(CultureInfo.InvariantCulture, $"index {original} is out of bounds for axis 0 with size {arraySize}")
                    : string.Create(CultureInfo.InvariantCulture, $"index {original} is out of bounds for size {arraySize}"));
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
        /// <exception cref="ValueError">The array is not 2-D (NumPy's <c>incorrect number of indices for array</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index outside its axis.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        /// </remarks>
        public object item(long i, long j)
        {
            // NumPy's item(*args) requires exactly ndim indices (or a single flat index). Two indices
            // are valid only for a 2-D array; anything else raises, matching numpy's
            // "incorrect number of indices for array" rather than silently dropping/padding an axis.
            if (ndim != 2)
                throw new ValueError(IncorrectIndexCount);
            return Storage.GetValue(new[] { i, j });
        }

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <param name="i">Index along first dimension.</param>
        /// <param name="j">Index along second dimension.</param>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        /// <exception cref="ValueError">The array is not 2-D (<c>incorrect number of indices for array</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index outside its axis.</exception>
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
        /// <exception cref="ValueError">The array is not 3-D (NumPy's <c>incorrect number of indices for array</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index outside its axis.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        /// </remarks>
        public object item(long i, long j, long k)
        {
            // Three indices are valid only for a 3-D array (see item(i, j) — NumPy requires exactly
            // ndim indices), else "incorrect number of indices for array".
            if (ndim != 3)
                throw new ValueError(IncorrectIndexCount);
            return Storage.GetValue(new[] { i, j, k });
        }

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <param name="i">Index along first dimension.</param>
        /// <param name="j">Index along second dimension.</param>
        /// <param name="k">Index along third dimension.</param>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        /// <exception cref="ValueError">The array is not 3-D (<c>incorrect number of indices for array</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index outside its axis.</exception>
        public T item<T>(long i, long j, long k) where T : unmanaged => GetItemAs<T>(item(i, j, k));

        #endregion

        #region item(indices) - N-D indexing

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <param name="indices">Indices of element to extract (one per dimension).</param>
        /// <returns>A copy of the specified element of the array as a suitable Python scalar.</returns>
        /// <exception cref="ValueError">Not one index per dimension (NumPy's <c>incorrect number of indices for
        ///     array</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index outside its axis.</exception>
        /// <remarks>
        /// https://numpy.org/doc/stable/reference/generated/numpy.ndarray.item.html
        /// </remarks>
        public object item(params long[] indices)
        {
            // array_item's tuple form: exactly ndim indices (the one-index flat form is item(long)).
            if (indices.Length != ndim)
                throw new ValueError(IncorrectIndexCount);
            return Storage.GetValue(indices);
        }

        /// <summary>
        /// Copy an element of an array to a standard Python scalar and return it.
        /// </summary>
        /// <typeparam name="T">The type to convert the value to.</typeparam>
        /// <param name="indices">Indices of element to extract (one per dimension).</param>
        /// <returns>A copy of the specified element of the array as a typed scalar.</returns>
        /// <exception cref="ValueError">Not one index per dimension (<c>incorrect number of indices for array</c>).</exception>
        /// <exception cref="IndexOutOfRangeException">An index outside its axis.</exception>
        public T item<T>(params long[] indices) where T : unmanaged => GetItemAs<T>(item(indices));

        #endregion

        #region Helper

        /// <summary>NumPy's ValueError text for an index count that is neither one flat index nor one per dimension
        ///     (NumSharp said <c>Incorrect number of indices for array with 2 dimensions.</c> until API-parity batch 13).</summary>
        private const string IncorrectIndexCount = "incorrect number of indices for array";

        private T GetItemAs<T>(object value) where T : unmanaged
        {
            if (value is T typed)
                return typed;
            return Utilities.Converts.ChangeType<T>(value);
        }

        #endregion
    }
}
